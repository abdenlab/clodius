"""Tests for clodius.tiles_v2.bbi.

One module serving bigWig and bigBed through four concrete tilesets that share
a single info builder: a signal tileset emitting dense vectors, an annotation
tileset emitting records, and the two bigInteract tilesets -- links, which is
one-dimensional, and rectangles, which is not. All four derive their info the
same way: build a quadtree over the chromsizes, then pad ``max_pos`` out to the
quadtree extent so the client's axis matches the grid the tiles are cut from.

That padding is the part under test here, in both of its dimensions. It is
applied by deriving a variant of the info, and the served info is the only
place a broken derivation becomes visible -- nothing looked at it before.

How many entries it pads matters as much as the value. ``min_pos`` and
``max_pos`` are per-axis on the wire, so a two-dimensional tileset that
publishes one of each leaves a 2D track with no second axis to lay out. The
arity is the only thing the shared builder takes from its subclass, and
``canvas()`` derives its extent from the scalar ``max_width``, so a wrong
arity is invisible everywhere except the served info.

The fixtures are synthesized with pybigtools in milliseconds. Note that the
chromsizes go to ``write``, not to ``open`` -- the wrapper's ``open`` takes
only a path and a mode.
"""

import base64
import hashlib
import pathlib

import numpy as np
import pybigtools
import pytest

from clodius.core.coords import Chromsizes
from clodius.core.policies import TilePolicy
from clodius.tiles_v2.bbi import (
    BBIAnnotationTileset,
    BBIInteraction2DTileset,
    BBIInteractionLinksTileset,
    BBISignalTileset,
    to_bedlike,
)

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


def interact_record(name, source, target):
    """One bed5+13 interact row, as the tab-joined rest of a bigBed record.

    The hull is the record's own start and end; the anchors live in the custom
    fields, which is what makes this schema worth a fixture of its own.
    """
    source_chrom, source_start, source_end = source
    target_chrom, target_start, target_end = target
    return "\t".join(
        [
            name,
            "0",
            "+",
            "hg38",
            "0",
            source_chrom,
            str(source_start),
            str(source_end),
            "0,0,0",
            target_chrom,
            str(target_start),
            str(target_end),
            "0,0,0",
        ]
    )


@pytest.fixture(scope="module")
def bigInteract(tmp_path_factory):
    """A bed5+13 bigBed holding one interaction per contig."""
    path = str(tmp_path_factory.mktemp("bbi") / "interact.bb")
    pybigtools.open(path, "w").write(
        CHROMSIZES,
        iter(
            [
                (
                    "c1",
                    10,
                    900,
                    interact_record("a", ("c1", 10, 60), ("c1", 850, 900)),
                ),
                (
                    "c2",
                    20,
                    700,
                    interact_record("b", ("c2", 20, 70), ("c2", 650, 700)),
                ),
                (
                    "c3",
                    5,
                    400,
                    interact_record("c", ("c3", 5, 40), ("c3", 360, 400)),
                ),
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


def test_tiles_should_refuse_a_zero_cap_without_reading_the_file(
    bigbed, tmp_path
):
    """Test that a cap of zero costs nothing, not merely that it serves nothing.

    Given:
        An annotation tileset over a copy of the file, capped at zero, whose
        file is removed after construction.
    When:
        The whole-genome tile is served.
    Then:
        It should return no records rather than raising. A missing file is the
        only way to observe the absence of I/O from outside: refusing after the
        read costs every record fetched and digested -- seconds on a zoom-0
        tile -- to produce an empty list.
    """
    # Arrange
    path = tmp_path / "annot.bb"
    path.write_bytes(pathlib.Path(bigbed).read_bytes())
    tileset = BBIAnnotationTileset(
        str(path), policy=TilePolicy(max_records=0), tile_size=TILE_SIZE
    )
    path.unlink()

    # Act
    (_, records), = tileset.tiles([tileset.parse_tile_id("u.0.0")])

    # Assert
    assert records == []


def test_tiles_should_read_the_file_when_the_cap_is_not_zero(bigbed, tmp_path):
    """Test the control for the refusal above, so it pins the short-circuit.

    Given:
        The same tileset over the same removed file, capped at one instead.
    When:
        The whole-genome tile is served.
    Then:
        It should raise, because it reaches the file. Without this, the test
        above would pass against a build that never reads anything.
    """
    # Arrange
    path = tmp_path / "annot.bb"
    path.write_bytes(pathlib.Path(bigbed).read_bytes())
    tileset = BBIAnnotationTileset(
        str(path), policy=TilePolicy(max_records=1), tile_size=TILE_SIZE
    )
    path.unlink()

    # Act & assert
    with pytest.raises(OSError):
        tileset.tiles([tileset.parse_tile_id("u.0.0")])


def test_tiles_should_place_records_by_the_ordering_the_tile_selects(bigbed):
    """Test the ``,cos:`` path against a tileset built on that ordering.

    Given:
        An annotation tileset carrying an alternate chromosome ordering, and a
        second tileset built directly on that ordering.
    When:
        The whole-genome tile is served, selecting the alternate on the first.
    Then:
        Both should place the records identically. The alternate's info is now
        built once at construction rather than per tile, and a memo that
        handed back the wrong ordering's info would misplace every record
        while still looking like a well-formed tile.
    """
    # Arrange
    # Only the two contigs the fixture carries records for: pybigtools drops
    # a contig with no records from the file's chrom list, and asking for one
    # that is not there is a KeyError rather than an empty read.
    default = Chromsizes(("c1", "c2"), (1000, 1500))
    reordered = Chromsizes(("c2", "c1"), (1500, 1000))
    tileset = BBIAnnotationTileset(
        bigbed,
        chromsizes=default,
        chromsizes_alts={"alt": reordered},
        tile_size=TILE_SIZE,
    )
    reference = BBIAnnotationTileset(
        bigbed, chromsizes=reordered, tile_size=TILE_SIZE
    )

    # Act
    (_, selected), = tileset.tiles(
        [tileset.parse_tile_id("u.0.0,cos:alt")]
    )
    (_, expected), = reference.tiles([reference.parse_tile_id("u.0.0")])

    # Assert
    assert [r["xStart"] for r in selected] == [r["xStart"] for r in expected]
    assert selected != []


def test_to_bedlike_should_return_none_when_the_contig_is_unknown():
    """Test the converter's own contig check. A regression guard, not a pin.

    Given:
        A raw record on a contig the offsets do not name.
    When:
        It is converted.
    Then:
        It should return ``None``. No call site in the tree can reach this --
        records are named from the same chromsizes the offsets come from -- so
        this guards the converter's contract rather than reproducing a defect.
        The call site used to index the map directly, and a ``KeyError`` there
        is not a ``TileError``, so it would have failed the whole batch.
    """
    # Act
    record = to_bedlike(("cUNKNOWN", 10, 20), {"c1": 0})

    # Assert
    assert record is None


def test_to_bedlike_should_place_a_record_on_a_known_contig():
    """Test the converter's placing path, so the check above is not vacuous.

    Given:
        A raw record on a contig the offsets name.
    When:
        It is converted.
    Then:
        It should return a record positioned at that contig's offset.
    """
    # Act
    record = to_bedlike(("c2", 10, 20), {"c1": 0, "c2": 1000})

    # Assert
    assert record is not None
    assert (record["xStart"], record["xEnd"]) == (1010, 1020)


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


def test_tiles_should_return_every_record_when_the_cap_is_none(bigbed):
    """Test the uncapped path, which now runs through the same thinning call.

    Given:
        An annotation tileset under a policy naming no record cap.
    When:
        The whole-genome tile is served.
    Then:
        It should return every feature. The caller used to short-circuit on
        ``None`` before reaching ``take_most_important``, which left two
        surfaces encoding what ``None`` means and the callee's own branch dead
        -- the one that would be missed when the rule changes.
    """
    # Arrange
    tileset = BBIAnnotationTileset(
        bigbed, tile_size=TILE_SIZE, policy=TilePolicy(max_records=None)
    )

    # Act
    (_, records), = tileset.tiles([tileset.parse_tile_id("u.0.0")])

    # Assert
    assert len(records) == 3


def test_tiles_should_digest_a_record_without_requiring_md5_for_security(
    bigbed, monkeypatch
):
    """Test that the per-record digest survives a FIPS-enforcing build.

    Given:
        An annotation tileset, and an ``md5`` that refuses any call not marked
        as non-security -- which is how a FIPS build behaves.
    When:
        A tile is served.
    Then:
        It should serve the records anyway. The flag was added to
        ``stable_importance``, but the uid is digested first and passed in, so
        this call raises before the marked one is ever reached and the
        hardening never takes effect.
    """
    # Arrange
    real_md5 = hashlib.md5

    def fips_md5(*args, **kwargs):
        if kwargs.get("usedforsecurity", True) is not False:
            raise ValueError("md5 is not available in FIPS mode")
        return real_md5(*args, **kwargs)

    monkeypatch.setattr(hashlib, "md5", fips_md5)
    tileset = BBIAnnotationTileset(bigbed, tile_size=TILE_SIZE)

    # Act
    (_, records), = tileset.tiles([tileset.parse_tile_id("u.0.0")])

    # Assert
    assert len(records) == 3


def test_info_should_return_one_position_per_axis_for_a_2d_tileset(
    bigInteract,
):
    """Test the arity of the info a two-dimensional tileset serves.

    Given:
        A bigInteract served as 2D rectangles, which declares ``ndim = 2``.
    When:
        Its info is requested.
    Then:
        ``min_pos`` and ``max_pos`` should each carry two entries. The client
        reads the y extent from ``max_pos[1]``, so a one-entry list leaves a
        2D track with no second axis to lay out -- and nothing in the serving
        path reads these fields, so the served info is the only place the
        mistake is visible.
    """
    # Arrange
    tileset = BBIInteraction2DTileset(bigInteract, tile_size=TILE_SIZE)

    # Act
    served = tileset.info().to_dict()

    # Assert
    assert served["min_pos"] == [0, 0]
    assert served["max_pos"] == [QUADTREE_EXTENT, QUADTREE_EXTENT]


@pytest.mark.parametrize(
    "cls,fixture",
    [
        (BBISignalTileset, "bigwig"),
        (BBIAnnotationTileset, "bigbed"),
        (BBIInteractionLinksTileset, "bigInteract"),
    ],
    ids=["signal", "annotation", "links"],
)
def test_info_should_return_a_single_position_for_a_1d_tileset(
    cls, fixture, request
):
    """Test that the per-axis padding did not widen the 1D types.

    Given:
        Each of the three tilesets that declare ``ndim = 1``.
    When:
        Info is requested.
    Then:
        ``min_pos`` and ``max_pos`` should each carry exactly one entry.
        Without this the 2D test above passes just as well against a build
        that pads every tileset to two axes, which would misdescribe three
        types to fix one.
    """
    # Arrange
    tileset = cls(request.getfixturevalue(fixture), tile_size=TILE_SIZE)

    # Act
    served = tileset.info().to_dict()

    # Assert
    assert served["min_pos"] == [0]
    assert served["max_pos"] == [QUADTREE_EXTENT]


@pytest.mark.parametrize(
    "cls,fixture",
    [
        (BBISignalTileset, "bigwig"),
        (BBIAnnotationTileset, "bigbed"),
        (BBIInteractionLinksTileset, "bigInteract"),
        (BBIInteraction2DTileset, "bigInteract"),
    ],
    ids=["signal", "annotation", "links", "interaction2d"],
)
def test_info_should_describe_as_many_axes_as_the_tileset_declares(
    cls, fixture, request
):
    """Test the invariant the four tilesets share, rather than four constants.

    Given:
        Each concrete tileset in the module, whatever arity it declares.
    When:
        Info is requested.
    Then:
        ``min_pos`` and ``max_pos`` should both be as long as ``ndim``. The
        two tests above pin the arities the module has today; this one pins
        the rule, so a fifth tileset added later cannot quietly inherit the
        wrong shape.
    """
    # Arrange
    tileset = cls(request.getfixturevalue(fixture), tile_size=TILE_SIZE)

    # Act
    served = tileset.info().to_dict()

    # Assert
    assert len(served["min_pos"]) == cls.ndim
    assert len(served["max_pos"]) == cls.ndim


def test_info_should_pad_every_axis_to_the_quadtree_extent(bigInteract):
    """Test what each axis is padded *to*, not just how many there are.

    Given:
        A 2D tileset over a genome of 3000 bp, whose quadtree extent is 4096.
    When:
        Its info is requested.
    Then:
        Both ``max_pos`` entries should equal ``max_width`` rather than the
        genome length. The client's axis has to match the grid the tiles are
        cut from, and a fix that padded per axis with the genome total would
        satisfy the arity tests above while moving both axes to the wrong
        place.
    """
    # Arrange
    tileset = BBIInteraction2DTileset(bigInteract, tile_size=TILE_SIZE)

    # Act
    served = tileset.info().to_dict()

    # Assert
    assert served["max_pos"] == [served["max_width"]] * 2
    assert served["max_width"] == QUADTREE_EXTENT > sum(CHROMSIZES.values())
