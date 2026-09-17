"""Tests for clodius.tiles_v2.bbi.

One module serving bigWig and bigBed through three classes: shared plumbing,
a signal tileset emitting dense vectors, and an annotation tileset emitting
records. Both concrete classes derive their info the same way -- build a
quadtree over the chromsizes, then pad ``max_pos`` out to the quadtree extent
so the client's axis matches the grid the tiles are cut from.

That padding is the part under test here. It is applied with
``model_copy(update=...)``, which does not validate: a misspelled key is
carried into the served info verbatim, and a value the constructor would
reject is accepted. So the served info is the only place a broken derivation
becomes visible, and nothing looked at it before.

The fixtures are synthesized with pybigtools in milliseconds. Note that the
chromsizes go to ``write``, not to ``open`` -- the wrapper's ``open`` takes
only a path and a mode.
"""

import base64

import numpy as np
import pybigtools
import pytest

from clodius.core.policies import TilePolicy
from clodius.tiles_v2.bbi import BBIAnnotationTileset, BBISignalTileset

#: Totals 3000 bp, which a four-bin tile size covers with a quadtree extent of
#: 4096 -- so the padded `max_pos` is distinguishable from the genome length.
CHROMSIZES = {"c1": 1000, "c2": 1500, "c3": 500}

TILE_SIZE = 4
QUADTREE_EXTENT = 4096


@pytest.fixture(scope="module")
def bigwig(tmp_path_factory):
    """A bigWig covering every chromosome with flat intervals."""
    path = str(tmp_path_factory.mktemp("bbi") / "signal.bw")
    pybigtools.open(path, "w").write(
        CHROMSIZES,
        iter(
            [
                ("c1", 0, 500, 1.5),
                ("c1", 500, 1000, 3.0),
                ("c2", 0, 1500, 2.0),
                ("c3", 0, 500, 4.0),
            ]
        ),
    )
    return path


@pytest.fixture(scope="module")
def bigbed(tmp_path_factory):
    """A bigBed holding three features."""
    path = str(tmp_path_factory.mktemp("bbi") / "annot.bb")
    pybigtools.open(path, "w").write(
        CHROMSIZES,
        iter(
            [
                ("c1", 10, 20, "a\t100"),
                ("c1", 300, 400, "b\t200"),
                ("c2", 50, 60, "c\t300"),
            ]
        ),
    )
    return path


def bin_count(payload):
    """Bins in a dense payload, discounting values-per-bin."""
    values = np.frombuffer(
        base64.b64decode(payload["dense"]), dtype=payload["dtype"]
    )
    return len(values) // payload.get("size", 1)


@pytest.mark.parametrize(
    "cls,fixture",
    [(BBISignalTileset, "bigwig"), (BBIAnnotationTileset, "bigbed")],
    ids=["signal", "annotation"],
)
def test_info_should_pad_the_range_to_the_quadtree_extent(
    cls, fixture, request
):
    """Test the padded axis both concrete tilesets advertise.

    Given:
        A BBI file over a 3000 bp genome, whose quadtree extent is 4096.
    When:
        Its info is requested.
    Then:
        ``max_pos`` should be the extent, not the genome length. The client
        positions tiles on this axis, so reporting the genome length there
        puts every tile at the wrong scale -- and because the padding is
        applied by a copy that validates nothing, a misspelled update key
        produces exactly that while raising no error at all.
    """
    # Arrange
    tileset = cls(request.getfixturevalue(fixture), tile_size=TILE_SIZE)

    # Act
    info = tileset.info()

    # Assert
    assert info.max_pos == [QUADTREE_EXTENT]
    assert info.max_width == QUADTREE_EXTENT


def test_info_should_keep_the_type_specific_fields(bigwig):
    """Test that deriving the padded info does not drop the extras.

    Given:
        A signal tileset, which advertises the aggregations and range modes
        its modifier slot accepts.
    When:
        Its info is requested.
    Then:
        Those fields should survive alongside the padded range. Rebuilding the
        model field by field is the obvious way to derive a variant of a frozen
        model, and it silently drops everything the subclass contributed --
        leaving a client with no way to know which modifiers it may ask for.
    """
    # Arrange
    tileset = BBISignalTileset(bigwig, tile_size=TILE_SIZE)

    # Act
    served = tileset.info().to_dict()

    # Assert
    assert served["aggregation_modes"]
    assert served["range_modes"]


def test_tiles_should_hold_one_bin_per_tile_slot(bigwig):
    """Test that the grid the tiles are cut from matches the info.

    Given:
        A signal tileset with a four-bin tile size.
    When:
        The whole-genome tile is served.
    Then:
        It should carry exactly four bins. This is the property the
        cross-type conformance suite asserts for the legacy modules, reaching
        the served payload through the same info the previous tests check --
        an update that corrupts the ladder rather than the range shows up
        here rather than there.
    """
    # Arrange
    tileset = BBISignalTileset(bigwig, tile_size=TILE_SIZE)

    # Act
    (_, payload), = tileset.tiles([tileset.parse_tile_id("u.0.0")])

    # Assert
    assert bin_count(payload) == TILE_SIZE


def test_tiles_should_return_nothing_when_the_cap_is_zero(bigbed):
    """Test the record cap end to end, through a real tileset.

    Given:
        An annotation tileset over a bigBed holding three features, under a
        policy capping records at zero.
    When:
        The whole-genome tile is served.
    Then:
        It should return no records. A server configured to serve none must
        not emit an unbounded tile, and the slice that implements the cap
        silently inverts at zero to mean "everything".
    """
    # Arrange
    tileset = BBIAnnotationTileset(
        bigbed, tile_size=TILE_SIZE, policy=TilePolicy(max_records=0)
    )

    # Act
    (_, records), = tileset.tiles([tileset.parse_tile_id("u.0.0")])

    # Assert
    assert records == []


def test_tiles_should_return_an_error_payload_for_a_tile_off_the_canvas(
    bigwig,
):
    """Test that one bad position does not take the batch with it.

    Given:
        A batch of two tile ids, one inside the canvas and one past its last
        tile.
    When:
        The batch is served.
    Then:
        Both entries should come back, the second an error payload naming the
        refusal. A client batches sixteen tiles per request, and a single bad
        position raising through ``tiles()`` discards the fifteen that were
        servable.
    """
    # Arrange
    tileset = BBISignalTileset(bigwig, tile_size=TILE_SIZE)
    n_tiles = tileset.info().canvas(1).n_tiles
    ids = [
        tileset.parse_tile_id("u.1.0"),
        tileset.parse_tile_id(f"u.1.{n_tiles}"),
    ]

    # Act
    results = tileset.tiles(ids)

    # Assert
    assert bin_count(results[0][1]) == TILE_SIZE
    assert results[1][1]["error_type"] == "TileOutOfBounds"
