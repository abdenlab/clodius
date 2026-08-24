"""Tests for clodius.core.tileset."""

import inspect

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from clodius.core.coords import Chromsizes
from clodius.core.errors import (
    MalformedTileId,
    TileOutOfBounds,
    UnsupportedModifier,
    UnsupportedOption,
)
from clodius.core.payloads import TileKind
from clodius.core.policies import DEFAULT_POLICY
from clodius.core.tileid import ModifierSpec
from clodius.core.tileset import (
    BaseTileset,
    Dataset,
    DatasetInfo,
    Ladder,
    ProvidesChromsizes,
    ProvidesRegions,
    Tileset,
    TilesetInfo,
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
def test_datatype_should_be_declared_by_every_tileset(cls, datatype, ndim):
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


def test___init___should_produce_an_object_satisfying_the_protocols(
    bed_tileset,
):
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


def test___enter___should_bind_the_tileset_itself(bed_tileset):
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


PROPERTY = settings(max_examples=200)

# Chromsizes for the two ladder regimes. The explicit genome is a megabase so
# the per-zoom extents it derives come out at visibly different values.
IMPLICIT_CHROMS = [("c1", 600), ("c2", 400)]
EXPLICIT_CHROMS = [("c1", 600_000), ("c2", 400_000)]


@pytest.fixture
def implicit_info():
    """A genuine quadtree: max_width is tile_size times two to the max zoom."""
    return TilesetInfo(
        min_pos=[0],
        max_pos=[1024],
        max_width=1024,
        tile_size=256,
        max_zoom=2,
        chromsizes=IMPLICIT_CHROMS,
    )


@pytest.fixture
def explicit_info():
    """An enumerated ladder, stored ascending as the wire format fixes."""
    return TilesetInfo(
        min_pos=[0],
        max_pos=[1_000_000],
        resolutions=[1000, 5000, 25000],
        tile_size=256,
        max_width=25000 * 256,
        chromsizes=EXPLICIT_CHROMS,
    )


@pytest.fixture
def bare_info():
    """A tileset declaring no ladder at all."""
    return TilesetInfo(min_pos=[0], max_pos=[1024])


@given(
    total=st.integers(min_value=1, max_value=2**60),
    tile_size=st.integers(min_value=1, max_value=2**20),
)
@PROPERTY
def test_quadtree_depth_should_be_the_smallest_covering_depth(
    total, tile_size
):
    """Test the defining property of the ladder depth.

    Given:
        Any positive coordinate span and tile width, drawn well past the
        range a genome occupies.
    When:
        The quadtree depth is computed.
    Then:
        The top tile should cover the space and one level shallower should
        not. Computed with ceil(log2(...)) over a float this fails above
        about 2**48, where the top tile silently stops covering.
    """
    # Act
    depth = quadtree_depth(total, tile_size)

    # Assert
    assert tile_size * 2**depth >= total
    assert depth == 0 or tile_size * 2 ** (depth - 1) < total


@given(
    k=st.integers(min_value=0, max_value=40),
    tile_size=st.sampled_from([1, 2, 256, 1024]),
)
@PROPERTY
def test_quadtree_depth_should_step_at_the_power_of_two_boundary(
    k, tile_size
):
    """Test the ladder identity the implicit regime rests on.

    Given:
        A span exactly filling a power-of-two ladder, and that span plus one.
    When:
        The depth is computed for each.
    Then:
        The exact span should need k levels and one base more should need
        k + 1, which is what makes max_width equal tile_size times two to the
        max zoom.
    """
    # Arrange
    exact = tile_size * 2**k

    # Act & assert
    assert quadtree_depth(exact, tile_size) == k
    assert quadtree_depth(exact + 1, tile_size) == k + 1


class TestLadder:
    """How a tileset serializes its resolution ladder."""

    def test_ladder_should_carry_its_wire_string(self):
        """Test the two declared serializations.

        Given:
            The ladder enum.
        When:
            Its members are compared with their wire strings.
        Then:
            They should compare equal, since the value reaches the client.
        """
        # Act & assert
        assert Ladder.IMPLICIT == "implicit"
        assert Ladder.EXPLICIT == "explicit"

    def test_ladder_should_reject_an_undeclared_serialization(self):
        """Test the closed vocabulary.

        Given:
            The ladder enum.
        When:
            It is looked up by a value it does not declare.
        Then:
            It should raise.
        """
        # Act & assert
        with pytest.raises(ValueError):
            Ladder("quadtree")


class TestDatasetInfo:
    """The fields common to tiled and non-tiled datasets."""

    def test___init___should_default_the_optional_fields(self):
        """Test the minimal construction.

        Given:
            Only the two position fields.
        When:
            A dataset info is constructed.
        Then:
            It should default the extent and coordinate system to None.
        """
        # Act
        info = DatasetInfo(min_pos=[0], max_pos=[10])

        # Assert
        assert info.max_width is None
        assert info.chromsizes is None

    @pytest.mark.parametrize("missing", ["min_pos", "max_pos"])
    def test___init___should_require_both_position_fields(self, missing):
        """Test that the coordinate extent is mandatory.

        Given:
            A construction omitting one of the position fields.
        When:
            A dataset info is constructed.
        Then:
            It should raise, both being required.
        """
        # Arrange
        fields = {"min_pos": [0], "max_pos": [10]}
        del fields[missing]

        # Act & assert
        with pytest.raises(ValidationError):
            DatasetInfo(**fields)

    @pytest.mark.parametrize("bad", [0, -5])
    def test___init___should_reject_a_non_positive_extent(self, bad):
        """Test the extent validator.

        Given:
            An extent of zero or less.
        When:
            A dataset info is constructed.
        Then:
            It should raise naming the field, since a tileset spanning nothing
            has no tiles to serve.
        """
        # Act & assert
        with pytest.raises(ValidationError, match="max_width must be > 0"):
            DatasetInfo(min_pos=[0], max_pos=[10], max_width=bad)

    def test___init___should_retain_an_undeclared_field(self):
        """Test that per-type extensions survive.

        Given:
            A per-type keyword the model does not declare.
        When:
            A dataset info is constructed with it.
        Then:
            It should retain the value and echo it when serialized, which is
            how a type carries its own fields through without the base model
            enumerating every one.
        """
        # Act
        info = DatasetInfo(
            min_pos=[0], max_pos=[10], aggregation_modes={"mean": {}}
        )

        # Assert
        assert info.aggregation_modes == {"mean": {}}
        assert "aggregation_modes" in info.model_dump()

    def test___init___should_normalize_chromsizes_to_pairs(self):
        """Test the coordinate-system field's wire shape.

        Given:
            Chromsizes in the list-of-lists form the wire uses.
        When:
            A dataset info is constructed.
        Then:
            It should hold them as name-length pairs.
        """
        # Act
        info = DatasetInfo(
            min_pos=[0], max_pos=[10], chromsizes=[["c1", 100], ["c2", 200]]
        )

        # Assert
        assert list(info.chromsizes) == [("c1", 100), ("c2", 200)]


class TestTilesetInfo:
    """The served info, and the ladder arithmetic every tile goes through."""

    def test___init___should_default_the_ladder_fields(self):
        """Test the minimal construction.

        Given:
            Only the two position fields.
        When:
            A tileset info is constructed.
        Then:
            It should declare no ladder at all.
        """
        # Act
        info = TilesetInfo(min_pos=[0], max_pos=[10])

        # Assert
        assert (info.max_zoom, info.tile_size, info.resolutions) == (
            None,
            None,
            None,
        )

    def test___init___should_keep_resolutions_in_the_given_order(self):
        """Test the wire order of an enumerated ladder.

        Given:
            Resolutions in the ascending order the wire format fixes.
        When:
            A tileset info is constructed and re-serialized.
        Then:
            It should preserve that order, so two tilesets do not emit the
            same field two ways.
        """
        # Act
        info = TilesetInfo(
            min_pos=[0], max_pos=[10], resolutions=[1000, 5000, 25000]
        )

        # Assert
        assert info.model_dump()["resolutions"] == [1000, 5000, 25000]

    def test___init___should_reject_a_negative_max_zoom(self):
        """Test the zoom validator.

        Given:
            A negative max zoom.
        When:
            A tileset info is constructed.
        Then:
            It should raise naming the field.
        """
        # Act & assert
        with pytest.raises(ValidationError, match="max_zoom must be >= 0"):
            TilesetInfo(min_pos=[0], max_pos=[10], max_zoom=-1)

    def test___init___should_accept_a_single_level_ladder(self):
        """Test the shallowest legal ladder.

        Given:
            A max zoom of zero, a tileset covered by one tile.
        When:
            A tileset info is constructed.
        Then:
            It should accept it, zero being a legal depth rather than a
            missing value.
        """
        # Act & assert
        assert TilesetInfo(min_pos=[0], max_pos=[10], max_zoom=0).max_zoom == 0

    @pytest.mark.parametrize("bad", [0, -3])
    def test___init___should_reject_a_non_positive_tile_size(self, bad):
        """Test the tile-size validator.

        Given:
            A tile size of zero or less.
        When:
            A tileset info is constructed.
        Then:
            It should raise naming the field.
        """
        # Act & assert
        with pytest.raises(ValidationError, match="tile_size must be > 0"):
            TilesetInfo(min_pos=[0], max_pos=[10], tile_size=bad)

    def test___init___should_inherit_the_extent_rule_from_the_base(self):
        """Test that the base model's validator applies to the subclass.

        Given:
            An extent of zero on a tileset info rather than a dataset info.
        When:
            It is constructed.
        Then:
            It should raise, the rule being inherited rather than restated.
        """
        # Act & assert
        with pytest.raises(ValidationError, match="max_width must be > 0"):
            TilesetInfo(min_pos=[0], max_pos=[10], max_width=0)

    def test___setattr___should_reject_assignment(self, implicit_info):
        """Test immutability.

        Given:
            A constructed tileset info.
        When:
            One of its fields is assigned.
        Then:
            It should raise, so a stale ladder can never be installed after
            construction and the derived geometry stays sound.
        """
        # Act & assert
        with pytest.raises(ValidationError):
            implicit_info.max_zoom = 5

    def test___setattr___should_reject_assignment_to_an_extra_field(self):
        """Test that the extension mechanism opens no mutation hole.

        Given:
            A tileset info carrying an undeclared per-type field.
        When:
            That field is assigned.
        Then:
            It should raise as a declared field would, so allowing extra
            fields does not make the model partly mutable.
        """
        # Arrange
        info = TilesetInfo(min_pos=[0], max_pos=[10], row_infos=["a"])

        # Act & assert
        with pytest.raises(ValidationError):
            info.row_infos = ["b"]

    def test___init___should_retain_a_per_type_field(self):
        """Test that a type's own fields survive construction.

        Given:
            A per-type keyword such as multivec's row metadata.
        When:
            A tileset info is constructed with it.
        Then:
            It should retain the value despite the model being frozen, since
            the field is supplied at construction rather than assigned after.
        """
        # Act
        info = TilesetInfo(min_pos=[0], max_pos=[10], row_infos=["a", "b"])

        # Assert
        assert info.row_infos == ["a", "b"]

    # --- ladder ------------------------------------------------------------

    def test_ladder_should_be_implicit_when_only_max_zoom_is_declared(
        self, implicit_info
    ):
        """Test the power-of-two regime.

        Given:
            A tileset declaring max zoom and tile size but no resolutions.
        When:
            Its ladder is read.
        Then:
            It should be implicit.
        """
        # Act & assert
        assert implicit_info.ladder is Ladder.IMPLICIT

    def test_ladder_should_be_explicit_when_resolutions_are_declared(
        self, explicit_info
    ):
        """Test the enumerated regime.

        Given:
            A tileset declaring resolutions.
        When:
            Its ladder is read.
        Then:
            It should be explicit.
        """
        # Act & assert
        assert explicit_info.ladder is Ladder.EXPLICIT

    def test_ladder_should_prefer_resolutions_over_max_zoom(self):
        """Test the precedence when both are declared.

        Given:
            A tileset declaring resolutions and a contradictory max zoom.
        When:
            Its ladder is read.
        Then:
            It should be explicit, the enumerated ladder winning.
        """
        # Arrange
        info = TilesetInfo(
            min_pos=[0], max_pos=[10], resolutions=[10, 20], max_zoom=5
        )

        # Act & assert
        assert info.ladder is Ladder.EXPLICIT

    def test_ladder_should_be_implicit_when_resolutions_are_empty(self):
        """Test an empty resolutions list.

        Given:
            A tileset declaring an empty resolutions list alongside a max
            zoom.
        When:
            Its ladder is read.
        Then:
            It should fall back to implicit. An empty list is falsy, so it
            reads as "no enumerated ladder" rather than "a ladder of no
            levels" -- pinned because the two are not obviously the same.
        """
        # Arrange
        info = TilesetInfo(
            min_pos=[0],
            max_pos=[8],
            resolutions=[],
            max_zoom=1,
            tile_size=2,
            max_width=8,
        )

        # Act & assert
        assert info.ladder is Ladder.IMPLICIT

    # --- num_zoom_levels ----------------------------------------------------

    def test_num_zoom_levels_should_be_one_more_than_max_zoom(
        self, implicit_info
    ):
        """Test the implicit level count.

        Given:
            A tileset whose max zoom is two.
        When:
            Its zoom-level count is read.
        Then:
            It should be three, max zoom being inclusive.
        """
        # Act & assert
        assert implicit_info.num_zoom_levels == 3

    def test_num_zoom_levels_should_count_the_enumerated_resolutions(self):
        """Test the explicit level count.

        Given:
            A tileset declaring two resolutions and a contradictory max zoom
            of five.
        When:
            Its zoom-level count is read.
        Then:
            It should be two, the enumerated ladder's length rather than the
            max zoom.
        """
        # Arrange
        info = TilesetInfo(
            min_pos=[0], max_pos=[10], resolutions=[10, 20], max_zoom=5
        )

        # Act & assert
        assert info.num_zoom_levels == 2

    def test_num_zoom_levels_should_raise_when_no_ladder_is_declared(
        self, bare_info
    ):
        """Test the tileset that declares neither ladder.

        Given:
            A tileset declaring neither resolutions nor a max zoom.
        When:
            Its zoom-level count is read.
        Then:
            It should raise saying so.
        """
        # Act & assert
        with pytest.raises(ValueError, match="neither resolutions nor"):
            bare_info.num_zoom_levels

    # --- resolution_for -----------------------------------------------------

    def test_resolution_for_should_halve_at_each_implicit_level(
        self, implicit_info
    ):
        """Test the power-of-two ladder.

        Given:
            A quadtree of extent 1024, tile size 256 and max zoom two.
        When:
            The resolution is read at each level.
        Then:
            It should halve at every step, ending at one base per bin.
        """
        # Act
        result = [implicit_info.resolution_for(z) for z in range(3)]

        # Assert
        assert result == [4.0, 2.0, 1.0]

    def test_resolution_for_should_index_the_ladder_coarsest_first(
        self, explicit_info
    ):
        """Test the direction z indexes, against the wire order.

        Given:
            An enumerated ladder serialized ascending.
        When:
            The resolution is read at each level.
        Then:
            Zoom zero should be the coarsest resolution while the serialized
            list stays ascending. The two orders are deliberately opposite:
            the wire fixes one and z indexes the other.
        """
        # Act
        result = [explicit_info.resolution_for(z) for z in range(3)]

        # Assert
        assert result == [25000.0, 5000.0, 1000.0]
        assert explicit_info.resolutions == [1000, 5000, 25000]

    def test_resolution_for_should_ignore_the_order_resolutions_arrive_in(
        self,
    ):
        """Test that serving is insensitive to the declared order.

        Given:
            The same three resolutions supplied descending rather than
            ascending.
        When:
            The coarsest level is read.
        Then:
            It should still be the largest resolution, since this method
            re-sorts internally. Nothing normalizes the stored field, so the
            wire order rests on producer discipline alone.
        """
        # Arrange
        info = TilesetInfo(
            min_pos=[0], max_pos=[10], resolutions=[25000, 5000, 1000]
        )

        # Act & assert
        assert info.resolution_for(0) == 25000.0

    @pytest.mark.parametrize(
        "fixture", ["implicit_info", "explicit_info"]
    )
    def test_resolution_for_should_raise_on_a_negative_zoom(
        self, fixture, request
    ):
        """Test the lower bound in both regimes.

        Given:
            A tileset on either ladder.
        When:
            A negative zoom is requested.
        Then:
            It should raise a tile error, which the boundary can render per
            tile.
        """
        # Arrange
        info = request.getfixturevalue(fixture)

        # Act & assert
        with pytest.raises(TileOutOfBounds, match="negative zoom"):
            info.resolution_for(-1)

    def test_resolution_for_should_raise_past_the_enumerated_ladder(
        self, explicit_info
    ):
        """Test the upper bound on an enumerated ladder.

        Given:
            A ladder of three resolutions.
        When:
            The zoom one past the last is requested -- the off-by-one a
            client produces by scrolling out.
        Then:
            It should raise naming the ladder length.
        """
        # Act & assert
        with pytest.raises(TileOutOfBounds, match="exceeds ladder of 3"):
            explicit_info.resolution_for(3)

    def test_resolution_for_should_raise_past_the_declared_max_zoom(
        self, implicit_info
    ):
        """Test the upper bound on a power-of-two ladder.

        Given:
            A tileset whose max zoom is two.
        When:
            Zoom three is requested.
        Then:
            It should raise naming the max zoom.
        """
        # Act & assert
        with pytest.raises(TileOutOfBounds, match="exceeds max_zoom 2"):
            implicit_info.resolution_for(3)

    def test_resolution_for_should_raise_when_the_implicit_ladder_is_partial(
        self,
    ):
        """Test an implicit ladder missing its geometry.

        Given:
            A tileset declaring a max zoom but neither extent nor tile size.
        When:
            A resolution is requested.
        Then:
            It should raise saying both are needed, since the implicit
            formula divides one by the other.
        """
        # Arrange
        info = TilesetInfo(min_pos=[0], max_pos=[10], max_zoom=2)

        # Act & assert
        with pytest.raises(ValueError, match="needs both max_width and"):
            info.resolution_for(0)

    def test_resolution_for_should_be_unbounded_without_a_max_zoom(self):
        """Test an implicit ladder that declares no depth.

        Given:
            A tileset declaring an extent and tile size but no max zoom.
        When:
            An absurdly deep zoom is requested.
        Then:
            It should return a resolution rather than raising, the upper
            bound applying only when a max zoom was declared. Pinned as
            observed: a client can walk arbitrarily far past any sensible
            ladder here.
        """
        # Arrange
        info = TilesetInfo(
            min_pos=[0], max_pos=[1024], max_width=1024, tile_size=256
        )

        # Act & assert
        assert info.resolution_for(50) > 0

    # --- tile_span ----------------------------------------------------------

    def test_tile_span_should_halve_at_each_level(self, implicit_info):
        """Test the tile width on a quadtree.

        Given:
            A tileset of extent 1024.
        When:
            The tile span is read at each level.
        Then:
            It should halve at every step.
        """
        # Act & assert
        assert [implicit_info.tile_span(z) for z in range(3)] == [
            1024.0,
            512.0,
            256.0,
        ]

    def test_tile_span_should_disagree_with_the_geometry_on_an_explicit_ladder(
        self, explicit_info
    ):
        """Test the undeclared precondition on this method.

        Given:
            A tileset on an enumerated ladder.
        When:
            The tile span is read and compared with the width the geometry
            actually uses at that zoom.
        Then:
            They should differ. This method divides the extent by two to the
            zoom, which only describes a quadtree -- its docstring lists only
            implicit call sites, so the precondition is real but undeclared,
            and a caller reaching for it on an mcool gets a wrong answer
            silently. Pinned as observed rather than endorsed.
        """
        # Act
        declared = explicit_info.tile_span(1)
        actual = explicit_info.canvas(1).binsize * explicit_info.tile_size

        # Assert
        assert declared != actual

    def test_tile_span_should_raise_when_no_extent_is_declared(
        self, bare_info
    ):
        """Test the missing-extent case.

        Given:
            A tileset declaring no extent.
        When:
            A tile span is requested.
        Then:
            It should raise saying so.
        """
        # Act & assert
        with pytest.raises(ValueError, match="declares no max_width"):
            bare_info.tile_span(0)

    def test_tile_span_should_not_bounds_check_the_zoom(self, implicit_info):
        """Test the absence of a ladder check on this accessor.

        Given:
            A tileset whose ladder stops at zoom two.
        When:
            The tile span is requested at a negative zoom.
        Then:
            It should return a width rather than raising, unlike
            resolution_for. Only the ladder methods bounds-check.
        """
        # Act & assert
        assert implicit_info.tile_span(-1) == 2048.0

    # --- coordinate_system --------------------------------------------------

    def test_coordinate_system_should_build_chromsizes_from_the_field(
        self, implicit_info
    ):
        """Test the derived coordinate system.

        Given:
            A tileset carrying chromsizes.
        When:
            Its coordinate system is read.
        Then:
            It should carry the same names, lengths and total.
        """
        # Act
        cs = implicit_info.coordinate_system

        # Assert
        assert cs.names == ("c1", "c2")
        assert cs.lengths == (600, 400)
        assert cs.total_length == 1000

    def test_coordinate_system_should_be_none_when_not_genomic(
        self, bare_info
    ):
        """Test a tileset with no coordinate system.

        Given:
            A tileset declaring no chromsizes.
        When:
            Its coordinate system is read.
        Then:
            It should be None, as a non-genomic tileset has none.
        """
        # Act & assert
        assert bare_info.coordinate_system is None

    def test_coordinate_system_should_be_built_once(self, implicit_info):
        """Test that the coordinate system is cached.

        Given:
            A frozen tileset info carrying chromsizes.
        When:
            Its coordinate system is read twice.
        Then:
            It should return the identical object, so a per-tile canvas call
            does not rebuild the cumulative offsets -- and the cache survives
            the model being frozen.
        """
        # Act & assert
        assert (
            implicit_info.coordinate_system
            is implicit_info.coordinate_system
        )

    # --- canvas -------------------------------------------------------------

    def test_canvas_should_derive_the_quadtree_lattice(self, implicit_info):
        """Test the lattice on a power-of-two ladder.

        Given:
            A quadtree of extent 1024, tile size 256 and max zoom two.
        When:
            The canvas is taken at each level.
        Then:
            The extent should stay fixed while the bins double and the tile
            count follows two to the zoom -- the invariant the implicit ladder
            rests on.
        """
        # Act
        canvases = [implicit_info.canvas(z) for z in range(3)]

        # Assert
        assert [c.binsize for c in canvases] == [4.0, 2.0, 1.0]
        assert [c.max_width for c in canvases] == [1024, 1024, 1024]
        assert [c.n_bins for c in canvases] == [256, 512, 1024]
        assert [c.n_tiles for c in canvases] == [1, 2, 4]

    def test_canvas_should_derive_the_extent_per_zoom_when_explicit(
        self, explicit_info
    ):
        """Test that an enumerated ladder has no tileset-wide extent.

        Given:
            A ladder of three resolutions over a megabase genome.
        When:
            The canvas is taken at each level.
        Then:
            The extent should differ per zoom, being however many tiles are
            needed to cover the genome at that resolution. Inheriting one
            value from the coarsest level would inflate the tile count at
            every finer zoom.
        """
        # Act
        canvases = [explicit_info.canvas(z) for z in range(3)]

        # Assert
        assert [c.binsize for c in canvases] == [25000.0, 5000.0, 1000.0]
        assert len({c.max_width for c in canvases}) > 1
        assert [c.n_tiles for c in canvases] == [1, 1, 4]

    def test_canvas_should_hand_through_the_coordinate_system(
        self, implicit_info
    ):
        """Test that the lattice can invert tile positions.

        Given:
            A tileset carrying chromsizes.
        When:
            A canvas is taken and a tile position inverted.
        Then:
            It should yield genomic ranges rather than raising, the
            coordinate system having been passed through.
        """
        # Act & assert
        assert list(implicit_info.canvas(1).invert(0))

    def test_canvas_should_raise_when_an_explicit_ladder_has_no_genome(self):
        """Test the extent derivation without a coordinate system.

        Given:
            An enumerated ladder declaring a tile size but no chromsizes.
        When:
            A canvas is requested.
        Then:
            It should raise, since an explicit ladder derives its extent from
            the genome length and has nothing to derive it from.
        """
        # Arrange
        info = TilesetInfo(
            min_pos=[0], max_pos=[10], resolutions=[10, 20], tile_size=256
        )

        # Act & assert
        with pytest.raises(ValueError, match="needs chromsizes"):
            info.canvas(0)

    @pytest.mark.parametrize("fixture", ["implicit_info", "explicit_info"])
    def test_canvas_should_raise_when_no_tile_size_is_declared(
        self, fixture, request
    ):
        """Test the missing tile size in both regimes.

        Given:
            A tileset on either ladder declaring no tile size.
        When:
            A canvas is requested.
        Then:
            It should raise saying so, the tile size describing the geometry
            rather than the ladder and being needed by both.
        """
        # Arrange
        original = request.getfixturevalue(fixture)
        fields = original.model_dump()
        fields["tile_size"] = None
        info = TilesetInfo(**fields)

        # Act & assert
        with pytest.raises(ValueError, match="declares no tile_size"):
            info.canvas(0)

    @pytest.mark.parametrize("fixture", ["implicit_info", "explicit_info"])
    def test_canvas_should_raise_on_a_negative_zoom(self, fixture, request):
        """Test the lower bound at the geometry entry point.

        Given:
            A tileset on either ladder.
        When:
            A canvas is requested at a negative zoom.
        Then:
            It should raise a tile error, so the geometry entry point rejects
            the same zooms the ladder does.
        """
        # Arrange
        info = request.getfixturevalue(fixture)

        # Act & assert
        with pytest.raises(TileOutOfBounds):
            info.canvas(-1)

    @pytest.mark.parametrize("fixture", ["implicit_info", "explicit_info"])
    def test_canvas_should_raise_one_past_the_ladder(self, fixture, request):
        """Test the upper bound at the geometry entry point.

        Given:
            A tileset on either ladder, both three levels deep.
        When:
            A canvas is requested one level past the last.
        Then:
            It should raise rather than returning a fabricated lattice.
        """
        # Arrange
        info = request.getfixturevalue(fixture)

        # Act & assert
        with pytest.raises(TileOutOfBounds):
            info.canvas(3)

    @given(
        max_zoom=st.integers(min_value=0, max_value=10),
        tile_size=st.sampled_from([64, 128, 256, 1024]),
    )
    @PROPERTY
    def test_canvas_should_agree_with_the_ladder_at_every_zoom(
        self, max_zoom, tile_size
    ):
        """Test that the two accessors describe one quadtree.

        Given:
            Any power-of-two ladder, built so the extent is the tile size
            times two to the max zoom.
        When:
            Every zoom in the ladder is walked.
        Then:
            The tile count should be two to the zoom, the bin count the tile
            size times that, and the resolution should reconstruct the extent
            exactly -- so the lattice and the ladder cannot drift apart.
        """
        # Arrange
        max_width = tile_size * 2**max_zoom
        info = TilesetInfo(
            min_pos=[0],
            max_pos=[max_width],
            max_width=max_width,
            tile_size=tile_size,
            max_zoom=max_zoom,
        )

        # Act & assert
        assert info.num_zoom_levels == max_zoom + 1
        for z in range(info.num_zoom_levels):
            canvas = info.canvas(z)
            assert canvas.n_tiles == 2**z
            assert canvas.n_bins == tile_size * 2**z
            assert info.resolution_for(z) * tile_size * 2**z == max_width

    @given(
        resolutions=st.lists(
            st.integers(min_value=1, max_value=10**7),
            min_size=1,
            max_size=8,
            unique=True,
        ).map(sorted)
    )
    @PROPERTY
    def test_resolution_for_should_walk_the_ladder_coarsest_first(
        self, resolutions
    ):
        """Test the ordering across a whole enumerated ladder.

        Given:
            Any ascending ladder of distinct resolutions.
        When:
            Every zoom level is walked.
        Then:
            The resolutions should decrease strictly, start at the coarsest,
            end at the finest, cover the declared set exactly, and raise one
            level past the end.
        """
        # Arrange
        info = TilesetInfo(
            min_pos=[0], max_pos=[10**8], resolutions=resolutions
        )

        # Act
        walked = [
            info.resolution_for(z) for z in range(info.num_zoom_levels)
        ]

        # Assert
        assert walked == sorted(walked, reverse=True)
        assert walked[0] == float(max(resolutions))
        assert walked[-1] == float(min(resolutions))
        assert {int(r) for r in walked} == set(resolutions)
        with pytest.raises(TileOutOfBounds):
            info.resolution_for(len(resolutions))


class ConformingTileset(BaseTileset):
    """A minimal tileset declaring everything the protocol requires."""

    datatype = "vector"
    ndim = 1
    tile_kind = TileKind.DENSE

    def __init__(self):
        self.policy = DEFAULT_POLICY
        self.close_calls = 0

    def info(self):
        return TilesetInfo(
            min_pos=[0], max_pos=[1024], max_width=1024, tile_size=256,
            max_zoom=2,
        )

    def tiles(self, ids):
        return [(tid, {}) for tid in ids]

    def close(self):
        self.close_calls += 1


class PlainDataset:
    """A non-tiled dataset, as chromsizes and time_interval are."""

    datatype = "chromsizes"

    def info(self):
        return DatasetInfo(min_pos=[0], max_pos=[10])

    def close(self):
        pass


class NoPolicyTileset(ConformingTileset):
    """Declares every member but never assigns the policy attribute."""

    def __init__(self):
        self.close_calls = 0


class ChromsizesProvider:
    """Implements only the coordinate-system capability."""

    def chromsizes(self):
        return Chromsizes.from_pairs(IMPLICIT_CHROMS)


class RegionLister:
    """Implements only the region-listing capability."""

    def regions(self, offset, limit):
        return [], False


class TestDataset:
    """The contract a non-tiled dataset satisfies."""

    def test___instancecheck___should_accept_a_non_tiled_dataset(self):
        """Test that a dataset need not serve tiles.

        Given:
            A stub declaring a datatype, an info method and a close method,
            but no tiles method.
        When:
            It is checked against the dataset protocol.
        Then:
            It should satisfy it, since a chromsizes or interval dataset is
            servable without being tiled.
        """
        # Act & assert
        assert isinstance(PlainDataset(), Dataset)

    def test___instancecheck___should_reject_a_bare_base_tileset(self):
        """Test that the base class alone is not a dataset.

        Given:
            A bare base tileset, which supplies close but neither a datatype
            nor an info method.
        When:
            It is checked against the dataset protocol.
        Then:
            It should not satisfy it.
        """
        # Act & assert
        assert not isinstance(BaseTileset(), Dataset)

    def test___subclasscheck___should_raise_for_a_data_bearing_protocol(self):
        """Test that conformance must be checked on instances.

        Given:
            The dataset protocol, which declares a non-method member.
        When:
            A class is checked against it rather than an instance.
        Then:
            It should raise, so conformance checks must use instances. This
            splits the four protocols in two and is worth pinning so nobody
            "fixes" a passing class check into a crashing one.
        """
        # Act & assert
        with pytest.raises(TypeError):
            issubclass(PlainDataset, Dataset)


class TestTileset:
    """The contract a tile-serving tileset satisfies."""

    def test___instancecheck___should_accept_a_conforming_tileset(self):
        """Test the positive case.

        Given:
            A stub declaring every member the protocol requires.
        When:
            It is checked against the tileset protocol.
        Then:
            It should satisfy it.
        """
        # Act & assert
        assert isinstance(ConformingTileset(), Tileset)

    def test___instancecheck___should_reject_a_dataset_that_serves_no_tiles(
        self,
    ):
        """Test that the two protocols are genuinely distinct.

        Given:
            A non-tiled dataset.
        When:
            It is checked against the tileset protocol.
        Then:
            It should not satisfy it.
        """
        # Act & assert
        assert not isinstance(PlainDataset(), Tileset)

    def test___instancecheck___should_reject_an_unassigned_policy(self):
        """Test that an annotation is not an attribute.

        Given:
            A tileset declaring every class-level member but never assigning
            the policy the base class annotates.
        When:
            It is checked against the tileset protocol.
        Then:
            It should not satisfy it. A subclass that forgets the assignment
            fails conformance silently, with nothing pointing at the cause --
            worth knowing before writing a sixth tileset.
        """
        # Act & assert
        assert not isinstance(NoPolicyTileset(), Tileset)

    def test___instancecheck___should_ignore_method_signatures(self):
        """Test the limit of the runtime check.

        Given:
            A tileset whose tiles method takes no batch argument at all.
        When:
            It is checked against the tileset protocol.
        Then:
            It should still satisfy it, the check testing member presence
            rather than signatures. This is why the signature tests below
            earn their place rather than being redundant.
        """

        # Arrange
        class LooseSignature(ConformingTileset):
            def tiles(self):
                return []

        # Act & assert
        assert isinstance(LooseSignature(), Tileset)

    def test___subclasscheck___should_raise_for_a_data_bearing_protocol(self):
        """Test that tileset conformance is instance-only.

        Given:
            The tileset protocol, which declares several non-method members.
        When:
            a class is checked against it.
        Then:
            It should raise, as for the dataset protocol.
        """
        # Act & assert
        with pytest.raises(TypeError):
            issubclass(ConformingTileset, Tileset)


class TestProvidesChromsizes:
    """The opt-in coordinate-system capability."""

    def test___instancecheck___should_accept_an_implementor(self):
        """Test the positive case.

        Given:
            A stub defining only the coordinate-system method.
        When:
            It is checked against the capability.
        Then:
            It should satisfy it.
        """
        # Act & assert
        assert isinstance(ChromsizesProvider(), ProvidesChromsizes)

    def test___instancecheck___should_reject_a_tileset_without_it(self):
        """Test that the capability is opt-in.

        Given:
            A conforming tileset that defines no coordinate-system method.
        When:
            It is checked against the capability.
        Then:
            It should not satisfy it, so serving tiles does not imply serving
            a coordinate system.
        """
        # Act & assert
        assert not isinstance(ConformingTileset(), ProvidesChromsizes)

    def test___subclasscheck___should_work_for_a_method_only_protocol(self):
        """Test that a method-only capability supports class checks.

        Given:
            An implementor class and a non-implementor class.
        When:
            Each is checked against the capability as a class.
        Then:
            It should answer without raising, unlike the two protocols
            carrying data members.
        """
        # Act & assert
        assert issubclass(ChromsizesProvider, ProvidesChromsizes)
        assert not issubclass(RegionLister, ProvidesChromsizes)


class TestProvidesRegions:
    """The opt-in region-listing capability."""

    def test___instancecheck___should_accept_an_implementor(self):
        """Test the positive case.

        Given:
            A stub defining only the region-listing method.
        When:
            It is checked against the capability.
        Then:
            It should satisfy it.
        """
        # Act & assert
        assert isinstance(RegionLister(), ProvidesRegions)

    def test___instancecheck___should_reject_a_tileset_without_it(self):
        """Test that the capability is opt-in.

        Given:
            A conforming tileset that lists no regions.
        When:
            It is checked against the capability.
        Then:
            It should not satisfy it.
        """
        # Act & assert
        assert not isinstance(ConformingTileset(), ProvidesRegions)

    def test___subclasscheck___should_work_for_a_method_only_protocol(self):
        """Test that a method-only capability supports class checks.

        Given:
            An implementor class and a non-implementor class.
        When:
            Each is checked against the capability as a class.
        Then:
            It should answer without raising.
        """
        # Act & assert
        assert issubclass(RegionLister, ProvidesRegions)
        assert not issubclass(ChromsizesProvider, ProvidesRegions)


AGGREGATION = ModifierSpec(
    values=frozenset({"mean", "max"}), default="mean", kind="aggregation"
)


class OneDim(BaseTileset):
    ndim = 1


class TwoDim(BaseTileset):
    ndim = 2


class Aggregating(BaseTileset):
    ndim = 1
    modifiers = AGGREGATION


class TestBaseTilesetDeclarations:
    """The defaults a subclass inherits when it declares nothing."""

    def test_options_should_default_to_rejecting_every_option(self):
        """Test the inherited option set.

        Given:
            The base class itself.
        When:
            Its recognized option set is read.
        Then:
            It should be an empty frozenset, which rejects every option --
            not None, which would accept any.
        """
        # Act & assert
        assert BaseTileset.options == frozenset()

    def test_modifiers_should_default_to_declaring_none(self):
        """Test the inherited modifier spec.

        Given:
            The base class itself.
        When:
            Its modifier spec is read.
        Then:
            It should be None, meaning the modifier slot accepts nothing.
        """
        # Act & assert
        assert BaseTileset.modifiers is None


class TestBaseTilesetParsing:
    """The declared shape the base class forwards to the parser."""

    def test_parse_tile_id_should_read_one_coordinate_when_one_dimensional(
        self,
    ):
        """Test the arity pass-through for a vector tileset.

        Given:
            A tileset declaring one coordinate slot.
        When:
            A well-formed tile id is parsed.
        Then:
            It should expose the zoom, the single position, and the request
            string verbatim -- the boundary echoes that back as the response
            key.
        """
        # Act
        tid = OneDim().parse_tile_id("uid.3.4")

        # Assert
        assert (tid.uid, tid.z, tid.pos, tid.raw) == ("uid", 3, (4,), "uid.3.4")

    def test_parse_tile_id_should_read_both_coordinates_when_two_dimensional(
        self,
    ):
        """Test the arity pass-through for a matrix tileset.

        Given:
            A tileset declaring two coordinate slots.
        When:
            A tile id carrying two coordinates is parsed.
        Then:
            It should consume both.
        """
        # Act
        tid = TwoDim().parse_tile_id("uid.3.4.5")

        # Assert
        assert (tid.z, tid.pos) == (3, (4, 5))

    def test_parse_tile_id_should_raise_when_the_arity_is_short(self):
        """Test that the declared arity actually reaches the parser.

        Given:
            A tileset declaring two coordinate slots.
        When:
            A one-coordinate tile id is parsed.
        Then:
            It should raise, proving the declaration is forwarded rather than
            assumed.
        """
        # Act & assert
        with pytest.raises(MalformedTileId, match="dotted parts"):
            TwoDim().parse_tile_id("uid.3.4")

    def test_parse_tile_id_should_raise_when_no_arity_is_declared(self):
        """Test a tileset that omits its arity.

        Given:
            A subclass declaring no coordinate slot count.
        When:
            A tile id is parsed.
        Then:
            It should raise an attribute error, the arity being mandatory and
            undeclared on the base. Pinned as observed: this is not a tile
            error, so it escapes the boundary untranslated.
        """

        # Arrange
        class NoArity(BaseTileset):
            pass

        # Act & assert
        with pytest.raises(AttributeError, match="ndim"):
            NoArity().parse_tile_id("uid.3.4")

    def test_parse_tile_id_should_reject_an_undeclared_option_key(self):
        """Test that a non-empty option set is forwarded, not just its truth.

        Given:
            A tileset declaring exactly one recognized option key.
        When:
            A tile id carrying a different key is parsed.
        Then:
            It should raise. The empty-set case is covered separately; this
            one pins that the set's contents are checked rather than only
            whether it is empty.
        """

        # Arrange
        class OneOption(BaseTileset):
            ndim = 1
            options = frozenset({"cos"})

        # Act & assert
        with pytest.raises(UnsupportedOption, match="max"):
            OneOption().parse_tile_id("uid.3.4,max:5")

    def test_parse_tile_id_should_reject_a_modifier_when_none_are_declared(
        self,
    ):
        """Test the inherited modifier default in practice.

        Given:
            A tileset leaving the modifier spec at the inherited None.
        When:
            A tile id carrying a modifier is parsed.
        Then:
            It should raise.
        """
        # Act & assert
        with pytest.raises(UnsupportedModifier, match="declares none"):
            OneDim().parse_tile_id("uid.3.4.mean")

    def test_parse_tile_id_should_fill_in_the_declared_default_modifier(self):
        """Test that the declared spec is forwarded.

        Given:
            A tileset whose modifier spec declares a default.
        When:
            A tile id with no modifier slot is parsed.
        Then:
            It should carry the default, showing the spec reached the parser.
        """
        # Act & assert
        assert Aggregating().parse_tile_id("uid.3.4").modifier == "mean"

    def test_parse_tile_id_should_accept_a_declared_modifier(self):
        """Test the ordinary modifier path.

        Given:
            A tileset declaring two aggregation modes.
        When:
            A tile id naming one is parsed.
        Then:
            It should carry that value.
        """
        # Act & assert
        assert Aggregating().parse_tile_id("uid.3.4.max").modifier == "max"

    def test_parse_tile_id_should_reject_an_undeclared_modifier(self):
        """Test rejection against the declared set.

        Given:
            A tileset declaring two aggregation modes.
        When:
            A tile id naming a third is parsed.
        Then:
            It should raise.
        """
        # Act & assert
        with pytest.raises(UnsupportedModifier, match="median"):
            Aggregating().parse_tile_id("uid.3.4.median")

    @given(text=st.text(max_size=30))
    @PROPERTY
    def test_parse_tile_id_should_raise_only_tile_errors(self, text):
        """Test that the parser is total over arbitrary input.

        Given:
            Any text at all, as a client can put anything in a tile id.
        When:
            It is parsed by a one-dimensional tileset.
        Then:
            It should either return a tile id whose request string is the
            input verbatim, or raise a tile error -- never anything else. An
            untranslatable exception here takes down a whole batch rather
            than one tile, which is the precondition for rendering a bad id
            as a per-tile error at all.
        """
        # Arrange
        tileset = Aggregating()

        # Act
        try:
            tid = tileset.parse_tile_id(text)
        except (MalformedTileId, UnsupportedModifier, UnsupportedOption):
            return

        # Assert
        assert tid.raw == text
        assert tid.z >= 0
        assert len(tid.pos) == 1


class TestBaseTilesetLifetime:
    """The resource contract every tileset inherits."""

    def test_close_should_be_idempotent(self):
        """Test the inherited no-op close.

        Given:
            A bare base tileset, which owns no resource.
        When:
            It is closed twice.
        Then:
            It should return without raising either time, so a stateless
            tileset inherits a valid lifetime rather than needing to declare
            one.
        """
        # Arrange
        tileset = BaseTileset()

        # Act & assert
        assert tileset.close() is None
        assert tileset.close() is None

    def test___exit___should_close_the_tileset(self):
        """Test that leaving the block releases the resource.

        Given:
            A tileset recording how often it is closed.
        When:
            A with block over it completes normally.
        Then:
            It should have been closed exactly once.
        """
        # Arrange
        tileset = ConformingTileset()

        # Act
        with tileset:
            pass

        # Assert
        assert tileset.close_calls == 1

    def test___exit___should_close_and_propagate_when_the_body_raises(self):
        """Test that a failed request does not leak the resource.

        Given:
            A tileset recording how often it is closed.
        When:
            A with block over it raises in its body.
        Then:
            It should still close exactly once and let the exception through,
            since the exit hook releases without suppressing.
        """
        # Arrange
        tileset = ConformingTileset()

        # Act & assert
        with pytest.raises(RuntimeError):
            with tileset:
                raise RuntimeError("boom")
        assert tileset.close_calls == 1
