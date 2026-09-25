"""Tests for clodius.tiles_v2.cooler.

A multi-resolution cooler is the one tileset with an explicit resolution
ladder, and the only one that has to tell the client how the matrix it is
serving is stored. A cooler written symmetric-upper holds half the matrix and
the client mirrors it; one written square holds the whole thing and mirroring
it would double every off-diagonal count. That is what ``mirror_tiles`` says,
and it is carried as an extra field on a model that is otherwise frozen --
so it has to be passed at construction, and a test that only checks the
square case passes just as well when the condition is dropped and every
cooler claims to be square.

The fixtures are built here rather than read from ``test/data/``: the
committed multires cooler is a legacy implicit-ladder file with no
``resolutions`` group, which this tileset refuses outright.
"""

import pytest

from clodius.core.coords import Chromsizes
from clodius.core.errors import UnsupportedOption
from clodius.core.tileid import TileId
from clodius.tiles_v2.cooler import CoolerTileset

from ..core.mcool_fixture import build_mcool


@pytest.fixture(scope="module")
def symmetric_mcool(tmp_path_factory):
    """A multires cooler holding the upper triangle, the ordinary case."""
    path = str(tmp_path_factory.mktemp("cooler") / "symmetric.mcool")
    return build_mcool(path)


@pytest.fixture(scope="module")
def square_mcool(tmp_path_factory):
    """A multires cooler holding the whole matrix."""
    path = str(tmp_path_factory.mktemp("cooler") / "square.mcool")
    return build_mcool(path, symmetric_upper=False)


def test_info_should_tell_the_client_not_to_mirror_a_square_matrix(
    square_mcool,
):
    """Test the flag a whole-matrix cooler has to carry.

    Given:
        A multires cooler written with the whole matrix stored.
    When:
        Its info is requested.
    Then:
        It should carry ``mirror_tiles`` set to false. The client mirrors by
        default, which for a square matrix adds every off-diagonal count to
        itself -- a plausible-looking heatmap at twice the contact frequency.
    """
    # Arrange
    tileset = CoolerTileset(square_mcool)

    # Act
    served = tileset.info().to_dict()

    # Assert
    assert served["mirror_tiles"] == "false"


def test_info_should_omit_the_flag_for_a_half_matrix(symmetric_mcool):
    """Test that the flag is conditional rather than always emitted.

    Given:
        A multires cooler written symmetric-upper, the ordinary case.
    When:
        Its info is requested.
    Then:
        ``mirror_tiles`` should be absent, leaving the client to mirror the
        half-matrix it was given. Without this the previous test passes
        unchanged when the storage-mode condition is deleted and every cooler
        is declared square.
    """
    # Arrange
    tileset = CoolerTileset(symmetric_mcool)

    # Act
    served = tileset.info().to_dict()

    # Assert
    assert "mirror_tiles" not in served


def test_parse_tile_id_should_reject_an_option(symmetric_mcool):
    """Test the empty option declaration on a shipped tileset.

    Given:
        A cooler tileset, which declares that it recognizes no ``,key:value``
        options at all, and a tile id carrying one.
    When:
        The id is parsed.
    Then:
        It should raise ``UnsupportedOption``. An empty declaration collapsing
        into "accept anything" is the kind of defect that only shows up on a
        real tileset, because every tileset inherits the empty default and
        none of them would notice.
    """
    # Arrange
    tileset = CoolerTileset(symmetric_mcool)

    # Act & assert
    with pytest.raises(UnsupportedOption, match="bogus"):
        tileset.parse_tile_id("u.0.0.0,bogus:1")


def test_tiles_should_raise_when_the_file_cannot_be_opened():
    """Test that a whole-request failure is not demoted to a per-tile one.

    Given:
        A tileset pointed at a path that does not exist.
    When:
        a tile is served.
    Then:
        It should raise rather than returning an error payload. The per-tile
        catch exists for refusals that are genuinely about one tile; widening
        it to every exception would answer a batch of sixteen with sixteen
        cheerful error payloads and a 200, when the honest answer is that the
        dataset cannot be served at all.
    """
    # Arrange. Chromsizes are passed so that construction stays lazy: the
    # constructor reads them off the file when they are not given, and the
    # failure under test belongs to tiles(), not to __init__.
    tileset = CoolerTileset(
        "/nonexistent/none.mcool", chromsizes=Chromsizes(("chr1",), (1000,))
    )
    tid = TileId.parse("u.0.0.0", ndim=2)

    # Act & assert
    with pytest.raises(Exception) as excinfo:
        tileset.tiles([tid])
    assert not isinstance(excinfo.value, dict)
    assert isinstance(excinfo.value, (OSError, ValueError))


def test_tiles_should_return_an_error_payload_for_a_zoom_past_the_ladder(
    symmetric_mcool,
):
    """Test that a bad zoom refuses one tile rather than the request.

    Given:
        A batch holding one well-formed tile id and one naming a zoom above
        the resolution ladder, as a client with a stale tileset info sends.
    When:
        The batch is served.
    Then:
        It should return a dense payload for the good id and an error payload
        for the bad one. Tiles are batched by ``(zoom, transform)``, so a raise
        here loses the answers already computed for the *other* zoom groups in
        the same request, not only the neighbours of the offending id.
    """
    # Arrange
    tileset = CoolerTileset(symmetric_mcool)
    past_the_end = tileset.info().num_zoom_levels + 5
    ids = [
        tileset.parse_tile_id("u.0.0.0"),
        tileset.parse_tile_id(f"u.{past_the_end}.0.0"),
    ]

    # Act
    served = tileset.tiles(ids)

    # Assert
    assert "error" not in served[0][1]
    assert served[1][1]["error"]


def test_tiles_should_return_an_error_payload_for_an_unknown_transform(
    symmetric_mcool,
):
    """Test the second per-batch key, which fails the same way.

    Given:
        A batch holding one well-formed tile id and one naming a weight column
        the file does not carry. The modifier spec sets ``allow_unknown``, so
        any transform string reaches the resolver.
    When:
        The batch is served.
    Then:
        It should return a dense payload for the good id and an error payload
        for the bad one. Screening positions alone leaves this open: the
        transform is resolved before any position is looked at.
    """
    # Arrange
    tileset = CoolerTileset(symmetric_mcool)
    ids = [
        tileset.parse_tile_id("u.0.0.0"),
        tileset.parse_tile_id("u.0.0.0.nosuchweight"),
    ]

    # Act
    served = tileset.tiles(ids)

    # Assert
    assert "error" not in served[0][1]
    assert served[1][1]["error"]


def test_tiles_should_not_open_a_reader_when_nothing_in_the_batch_is_servable(
    symmetric_mcool,
):
    """Test that an all-refused batch touches the pixel store not at all.

    Given:
        A batch in which every position lies off the lattice, so the position
        screen empties it.
    When:
        The batch is served.
    Then:
        It should answer with error payloads without constructing a reader.
        The constructor is not cheap setup -- it opens the store and
        materializes the whole bin1 offset index, tens of megabytes on a
        finely binned genome -- and a raise inside it would discard the error
        payloads already recorded for the batch.
    """
    # Arrange. `reader` is the instance attribute the batch loop constructs
    # through, so wrapping it counts openings without reaching into internals.
    tileset = CoolerTileset(symmetric_mcool)
    opened = []
    build = tileset.reader

    def counting_reader(*args, **kwargs):
        opened.append(args)
        return build(*args, **kwargs)

    tileset.reader = counting_reader
    ids = [tileset.parse_tile_id(f"u.0.{x}.0") for x in (9999, 8888)]

    # Act
    served = tileset.tiles(ids)

    # Assert
    assert [payload["error"] for _, payload in served]
    assert opened == []
