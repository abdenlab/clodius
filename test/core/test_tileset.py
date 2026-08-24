"""Tests for clodius.core.tileset."""

import inspect

import pytest

from clodius.core.errors import UnsupportedOption
from clodius.core.tileset import (
    BaseTileset,
    ProvidesChromsizes,
    ProvidesRegions,
    Tileset,
    quadtree_depth,
)
from clodius.tiles.bigwig import TILE_SIZE as BIGWIG_TILE_SIZE
from clodius.tiles.bigwig import get_quadtree_depth as quadtree_depth_bigwig
from clodius.tiles.cooler import get_quadtree_depth as quadtree_depth_cooler
from clodius.tiles.utils import get_quadtree_depth as quadtree_depth_utils
from clodius.tiles_v2.bed import BedTileset
from clodius.tiles_v2.bigbed import BigBedTileset
from clodius.tiles_v2.bigwig import BigWigTileset
from clodius.tiles_v2.cooler import CoolerTileset
from clodius.tiles_v2.multivec import MultivecTileset

from ..harness.genome import CANONICAL_CHROMSIZES, MINIMAL_CHROMSIZES

# Every tileset in the new layer, with the shape it declares. Structural
# conformance is checked against these rather than against instances, because
# four of the five need a real file to construct.
TILESETS = [
    (BedTileset, "bedlike", 1),
    (BigBedTileset, "bedlike", 1),
    (BigWigTileset, "vector", 1),
    (CoolerTileset, "matrix", 2),
    (MultivecTileset, "multivec", 1),
]

TILESET_IDS = [cls.__name__ for cls, _, _ in TILESETS]


@pytest.fixture
def bed_tileset(tmp_path):
    """A BedTileset over a three-record BED, needing no LFS fixture.

    Built on the canonical 3000 bp genome rather than the 350 bp one: at 350 bp
    the ladder collapses to ``max_zoom == 0``, a single tile covering
    everything, so any test that went on to request a tile would silently be
    testing a degenerate geometry.
    """
    from clodius.core.coords import Chromsizes

    path = tmp_path / "tiny.bed"
    path.write_text("c1\t10\t20\ta\nc2\t30\t40\tb\nc3\t0\t50\tc\n")
    return BedTileset(path, Chromsizes.from_pairs(CANONICAL_CHROMSIZES))


@pytest.mark.parametrize(
    "total,tile_size,expected",
    [
        (1, 1024, 0),  # smaller than one tile
        (1024, 1024, 0),  # exactly one tile
        (1025, 1024, 1),  # spills into a second
        (2048, 1024, 1),
        (2049, 1024, 2),
        (4096, 1024, 2),
    ],
)
def test_quadtree_depth_should_return_levels_needed_to_cover_the_space(
    total, tile_size, expected
):
    """Test the depth of the power-of-two ladder.

    Given:
        A coordinate space and a tile width.
    When:
        The quadtree depth is computed.
    Then:
        It should be the smallest depth whose top tile covers the space.
    """
    # Act
    result = quadtree_depth(total, tile_size)

    # Assert
    assert result == expected


@pytest.mark.parametrize("bad", [0, -1])
def test_quadtree_depth_should_raise_when_total_length_is_nonpositive(bad):
    """Test rejection of an empty coordinate space.

    Given:
        A total length of zero or less.
    When:
        The quadtree depth is computed.
    Then:
        It should raise ValueError.
    """
    # Act & assert
    with pytest.raises(ValueError, match="total_length must be positive"):
        quadtree_depth(bad, 1024)


@pytest.mark.parametrize("bad", [0, -1])
def test_quadtree_depth_should_raise_when_tile_size_is_nonpositive(bad):
    """Test rejection of an empty tile.

    Given:
        A tile width of zero or less.
    When:
        The quadtree depth is computed.
    Then:
        It should raise ValueError.
    """
    # Act & assert
    with pytest.raises(ValueError, match="tile_size_bp must be positive"):
        quadtree_depth(1024, bad)


@pytest.mark.parametrize("tile_size", [256, 1024, 4096])
def test_quadtree_depth_should_match_the_legacy_utils_call_site(tile_size):
    """Test the differential invariant against ``tiles.utils``.

    Given:
        A real assembly's total length and a tile width.
    When:
        The depth is computed both ways.
    Then:
        It should agree, since utils takes the tile width directly.
    """
    # Arrange
    lengths = [p[1] for p in MINIMAL_CHROMSIZES]
    total = sum(lengths)

    # Act
    result = quadtree_depth(total, tile_size)

    # Assert
    assert result == quadtree_depth_utils(lengths, tile_size)


def test_quadtree_depth_should_match_the_legacy_bigwig_call_site():
    """Test the differential invariant against ``tiles.bigwig``.

    Given:
        A genome's chromosome lengths.
    When:
        The depth is computed both ways.
    Then:
        It should agree, with bigwig's hardcoded TILE_SIZE supplied.
    """
    # Arrange
    lengths = [p[1] for p in MINIMAL_CHROMSIZES]

    # Act
    result = quadtree_depth(sum(lengths), BIGWIG_TILE_SIZE)

    # Assert
    assert result == quadtree_depth_bigwig(lengths)


@pytest.mark.parametrize("binsize", [1000, 5000, 100000])
def test_quadtree_depth_should_match_the_legacy_cooler_call_site(binsize):
    """Test the differential invariant against ``tiles.cooler``.

    Given:
        A genome's chromosome lengths and a bin size.
    When:
        The depth is computed both ways.
    Then:
        It should agree, with cooler's ``256 * binsize`` tile width supplied.
    """
    # Arrange
    lengths = [p[1] for p in MINIMAL_CHROMSIZES]

    # Act
    result = quadtree_depth(sum(lengths), 256 * binsize)

    # Assert
    assert result == quadtree_depth_cooler(lengths, binsize)


class TestBaseTileset:
    """Behavior the convenience base supplies to every tileset."""

    def test_parse_tile_id_should_reject_options_when_none_are_declared(self):
        """Test that an empty option set rejects rather than accepts.

        Given:
            A tileset declaring an empty frozenset of recognized options.
        When:
            A tile id carrying an option is parsed.
        Then:
            It should raise UnsupportedOption, since an empty set means
            options are rejected -- not that any option is allowed.
        """

        # Arrange
        class NoOptions(BaseTileset):
            ndim = 1
            options = frozenset()

        # Act & assert
        with pytest.raises(UnsupportedOption, match="bogus"):
            NoOptions().parse_tile_id("uid.3.4,bogus:1")

    def test_parse_tile_id_should_accept_a_declared_option(self):
        """Test that a recognized option parses through.

        Given:
            A tileset declaring one recognized option key.
        When:
            A tile id carrying that option is parsed.
        Then:
            It should expose the option's value.
        """

        # Arrange
        class OneOption(BaseTileset):
            ndim = 1
            options = frozenset({"cos"})

        # Act
        tid = OneOption().parse_tile_id("uid.3.4,cos:abc")

        # Assert
        assert tid.option("cos") == "abc"


@pytest.mark.parametrize(
    "cls,datatype,ndim", TILESETS, ids=TILESET_IDS
)
def test_tiles_should_be_declared_by_every_tileset(cls, datatype, ndim):
    """Test that each tileset declares the capability surface it promises.

    Given:
        A tileset class in the new layer.
    When:
        Its declared class attributes are read.
    Then:
        It should carry the datatype, arity, tile kind, modifier spec and
        option set the Tileset protocol requires.
    """
    # Act & assert
    assert cls.datatype == datatype
    assert cls.ndim == ndim
    assert cls.tile_kind is not None
    assert isinstance(cls.options, frozenset)
    assert hasattr(cls, "modifiers")


@pytest.mark.parametrize(
    "cls,datatype,ndim", TILESETS, ids=TILESET_IDS
)
def test_info_should_take_no_arguments_on_every_tileset(cls, datatype, ndim):
    """Test the ``info()`` signature the protocol declares.

    Given:
        A tileset class in the new layer.
    When:
        Its ``info`` signature is inspected.
    Then:
        It should take only ``self``, since a server calls it bare.
    """
    # Act
    params = list(inspect.signature(cls.info).parameters)

    # Assert
    assert params == ["self"]


@pytest.mark.parametrize(
    "cls,datatype,ndim", TILESETS, ids=TILESET_IDS
)
def test_tiles_should_take_a_batch_of_ids_on_every_tileset(
    cls, datatype, ndim
):
    """Test the ``tiles()`` signature the protocol declares.

    Given:
        A tileset class in the new layer.
    When:
        Its ``tiles`` signature is inspected.
    Then:
        It should take exactly one argument beyond ``self``, so scan-oriented
        backends can coalesce a whole batch.
    """
    # Act
    params = list(inspect.signature(cls.tiles).parameters)

    # Assert
    assert len(params) == 2 and params[0] == "self"


def test_tiles_should_satisfy_the_tileset_protocol(bed_tileset):
    """Test structural conformance against the declared protocols.

    Given:
        A constructed tileset.
    When:
        It is checked against Tileset and the capability protocols it claims.
    Then:
        It should satisfy all three. ``isinstance`` is used rather than
        ``issubclass``, which raises for protocols with non-method members.
    """
    # Act & assert
    assert isinstance(bed_tileset, Tileset)
    assert isinstance(bed_tileset, ProvidesChromsizes)
    assert isinstance(bed_tileset, ProvidesRegions)


def test___exit___should_close_the_tileset(bed_tileset):
    """Test that a tileset is usable as a context manager.

    Given:
        A constructed tileset.
    When:
        It is used in a ``with`` block.
    Then:
        It should yield itself and close on exit without raising.
    """
    # Act & assert
    with bed_tileset as ts:
        assert ts is bed_tileset
