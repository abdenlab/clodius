"""Tests for clodius.tiles_v2.cooler.

The only 2D tileset, and the only one whose out-of-range behavior is settled:
``clodius/tiles/cooler.py`` skips a tile past the ladder rather than raising,
and ``test/tiles/test_conformance.py`` asserts the resulting empty list, so this
rewrite skips too. Every other ``tiles_v2`` tileset still propagates.

The fixture is a three-resolution mcool at 1/2/4 bp over the canonical 3000 bp
genome, giving 12/6/3 tiles per zoom. Those numbers are what make the interior,
boundary-straddling and overhanging tiles distinguishable; at a realistic
binsize the whole genome collapses into one bin of one tile.
"""

import math

import cooler
import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from clodius.core.policies import DEFAULT_POLICY
from clodius.core.coords import GenomicRange
from clodius.core.errors import TileError
from clodius.core.tileset import Ladder
from clodius.tiles_v2 import cooler as mod
from clodius.tiles_v2.cooler import (
    TILE_SIZE,
    CoolerTileset,
    fetch_block,
    resolve_balance,
)

from ..harness import genome
from ..harness.wire import square


#: For the pure-arithmetic properties in this module, which build nothing and
#: take no fixture. They keep the wide budget; only the file-backed properties
#: below pay for a rebuild per example.
PROPERTY = settings(max_examples=200)

# Property tests here rebuild a real bigWig, bigBed, mcool or HDF5 file per
# example, so they cannot run on the ``pure`` profile's 200-example arithmetic
# budget. ``function_scoped_fixture`` is suppressed for the same reason it is
# in ``test/core/test_coords.py``: the fixtures these draw against are files
# built once per test, not state that leaks between examples.
PROPERTY_IO = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@pytest.fixture
def tileset(shared_mcool):
    """A CoolerTileset over the session-scoped mcool.

    Function-scoped over a session-scoped file: the tileset caches a handle and
    an info, so sharing the *tileset* would let one test's ``close()`` break the
    next. The file itself is immutable and safe to share.
    """
    with CoolerTileset(str(shared_mcool)) as ts:
        yield ts


def bins_past_the_genome(info, z, x):
    """How many trailing bins of tile ``x`` at zoom ``z`` lie past the genome."""
    canvas = info.canvas(z)
    lo, hi = canvas.tile_span(x)
    total = info.coordinate_system.total_length
    if hi <= total:
        return 0
    return int(math.ceil((hi - max(lo, total)) / canvas.binsize))



@pytest.fixture
def clr(shared_mcool):
    """The 4 bp resolution of the shared cooler, opened from the path."""

    return cooler.Cooler(f"{shared_mcool}::/resolutions/4")


class TestBinCount:
    """Bin accounting, which decides whether blocks in a strip agree on shape.

    Reached through ``fetch_block``, whose block is shaped by the bin count on
    each axis, rather than through the private helper that computes it: the
    shape is what the reconciler consumes and what a wrong count corrupts.
    """

    @pytest.mark.parametrize(
        "start,end,expected",
        [
            (0, 1000, 250),
            (0, 150, 38),
            (50, 150, 26),
            (1, 2, 1),
            (0, 0, 0),
            (50, 54, 2),
        ],
        ids=["whole-contig", "aligned-start", "unaligned-start",
             "sub-bin", "empty", "one-bin-wide-across-two"],
    )
    def test_fetch_block_should_round_the_shape_outward_from_both_ends(
        self, clr, start, end, expected
    ):
        """Test the overlap-based bin count, through the block it shapes.

        Given:
            An interval on the first contig and the 4 bp resolution.
        When:
            A block is fetched for it.
        Then:
            Its row count should span ``floor(start / 4)`` to ``ceil(end / 4)``,
            because cooler returns every bin the region *overlaps*. The last
            case is the one that matters: ``50-54`` is exactly one bin wide but
            straddles a boundary, so it touches two.
        """
        # Arrange
        row = GenomicRange(0, "c1", start, end)
        col = GenomicRange(0, "c1", 0, 4)

        # Act
        block = fetch_block(clr, row, col, 4, False)

        # Assert
        assert block.shape[0] == expected

    @PROPERTY
    @given(
        start=st.integers(min_value=0, max_value=10_000),
        span=st.integers(min_value=0, max_value=10_000),
        binsize=st.integers(min_value=1, max_value=500),
    )
    def test_fetch_block_should_never_undercount_the_span(
        self, start, span, binsize
    ):
        """Test the relationship to the naive count.

        Given:
            Any out-of-bounds interval and bin size.
        When:
            The padded block's shape is compared with ``ceil(span / binsize)``.
        Then:
            It should be at least as large, and at most one larger. An
            undercount makes a fetched block narrower than the reconciler
            expects, which is a shape error rather than a wrong value.
        """
        # Arrange
        # A range with no chromosome name is out of bounds, and ``fetch_block``
        # documents that such a block is padding rather than missing data --
        # shaped from the bin count and returned without reading anything. The
        # cooler is therefore ``None``: if the implementation ever did touch it
        # on this path, this test would raise rather than quietly pass.
        interval = GenomicRange(0, None, start, start + span)
        naive = -(-span // binsize)

        # Act
        block = fetch_block(None, interval, interval, binsize, False)

        # Assert
        assert naive <= block.shape[0] <= naive + 1


class TestResolveBalance:
    """Translating the transform modifier into a cooler balance argument."""

    @pytest.fixture
    def unbalanced(self, shared_mcool):
        """A cooler with no weight column, opened from the path."""

        return cooler.Cooler(f"{shared_mcool}::/resolutions/1")

    @pytest.fixture
    def balanced(self, make_mcool):
        """A cooler carrying a constant ``weight`` column."""

        path = make_mcool(name="weighted.mcool", weights={"weight": 0.5})
        return cooler.Cooler(f"{path}::/resolutions/4")

    @pytest.fixture
    def multiply_balanced(self, make_mcool):
        """A cooler carrying two weight vectors, ``weight`` and ``KR``.

        ``weight`` is also the column the default path falls back to, so on a
        cooler that carries only it, honoring a named request and ignoring one
        return the same answer.
        """

        path = make_mcool(
            name="two-weights.mcool", weights={"weight": 0.5, "KR": 2.0}
        )
        return cooler.Cooler(f"{path}::/resolutions/4")

    @pytest.mark.parametrize("transform", [None, "default"])
    def test_resolve_balance_should_use_weight_when_it_exists(
        self, balanced, transform
    ):
        """Test the default path on a balanced cooler.

        Given:
            A cooler with a weight column.
        When:
            No transform, or the literal ``default``, is requested.
        Then:
            It should balance by ``weight``, which is what makes an ICE-balanced
            matrix the default view.
        """
        # Act & assert
        assert resolve_balance(balanced, transform) == "weight"

    @pytest.mark.parametrize("transform", [None, "default"])
    def test_resolve_balance_should_not_balance_when_weight_is_absent(
        self, unbalanced, transform
    ):
        """Test the default path on an unbalanced cooler.

        Given:
            A cooler with no weight column.
        When:
            No transform, or ``default``, is requested.
        Then:
            It should return False rather than naming a column cooler would
            then fail to find.
        """
        # Act & assert
        assert resolve_balance(unbalanced, transform) is False

    def test_resolve_balance_should_refuse_balancing_for_the_none_literal(
        self, balanced
    ):
        """Test the explicit opt-out.

        Given:
            A balanced cooler and the string literal ``"None"``, which is what
            the client sends to request raw counts.
        When:
            The transform is resolved.
        Then:
            It should return False even though a weight column exists.
        """
        # Act & assert
        assert resolve_balance(balanced, "None") is False

    def test_resolve_balance_should_pass_through_a_named_column(
        self, multiply_balanced
    ):
        """Test an explicitly named weight column.

        Given:
            A cooler carrying two weight vectors, and a transform naming the
            one that is not the default.
        When:
            It is resolved.
        Then:
            It should be returned unchanged, so a cooler with several weight
            vectors can serve any of them. Naming ``weight`` here would prove
            nothing: it is also what the default path returns.
        """
        # Act & assert
        assert resolve_balance(multiply_balanced, "KR") == "KR"

    def test_resolve_balance_should_raise_and_list_what_is_available(
        self, balanced
    ):
        """Test an unavailable column.

        Given:
            A transform naming a column the cooler does not have.
        When:
            It is resolved.
        Then:
            It should raise TileError naming the available columns, and the
            coordinate columns should not be offered as balancing candidates.
        """
        # Act
        with pytest.raises(TileError) as excinfo:
            resolve_balance(balanced, "KR")

        # Assert
        message = str(excinfo.value)
        assert "'KR'" in message
        assert "['weight']" in message


class TestFetchBlock:
    """Block fetching, which owns the padding contract."""

    def test_fetch_block_should_pad_with_nan_when_a_range_is_out_of_bounds(
        self,
    ):
        """Test the padding case past the last chromosome.

        Given:
            A row range with no chromosome name.
        When:
            The block is fetched with a cooler that would raise if touched.
        Then:
            It should return a correctly shaped all-NaN block without reading.
            The reconciler needs the shape; the values are padding, not data.
        """
        # Arrange
        row = GenomicRange(0, None, 0, 300)
        col = GenomicRange(0, "c1", 0, 200)

        # Act
        block = mod.fetch_block(None, row, col, 100.0, False)

        # Assert
        assert block.shape == (3, 2)
        assert np.all(np.isnan(block))

    def test_fetch_block_should_return_float32(self, clr):
        """Test the block dtype.

        Given:
            An in-bounds pair of ranges.
        When:
            The block is fetched.
        Then:
            It should be float32, so a NaN-padded strip and a fully-real one
            concatenate without a dtype promotion.
        """
        # Arrange
        row = col = GenomicRange(0, "c1", 0, 400)

        # Act
        block = mod.fetch_block(clr, row, col, 4.0, False)

        # Assert
        assert block.dtype == np.float32


class TestCoolerTilesetDeclarations:
    """What the class states about itself."""

    def test_datatype_should_be_matrix(self):
        """Test the declared datatype.

        Given:
            The tileset class.
        When:
            Its ``datatype`` is read.
        Then:
            It should be "matrix", the string the wire uses for this format.
        """
        # Act & assert
        assert CoolerTileset.datatype == "matrix"

    def test_ndim_should_be_two(self):
        """Test the declared coordinate arity.

        Given:
            The tileset class.
        When:
            Its ``ndim`` is read.
        Then:
            It should be 2, the number of positional slots its tile ids
            carry.
        """
        # Act & assert
        assert CoolerTileset.ndim == 2

    def test_tile_kind_should_be_dense(self):
        """Test the declared payload kind.

        Given:
            The tileset class.
        When:
            Its ``tile_kind`` is read.
        Then:
            It should be "dense", which selects the payload shape a client
            decodes.
        """
        # Act & assert
        assert CoolerTileset.tile_kind == "dense"

    def test_grid_policy_should_be_sequential(self):
        """Test how this tileset bounds a single tile.

        Given:
            The tileset class.
        When:
            Its ``grid_policy`` is read.
        Then:
            It should be "sequential", describing a 2D dense matrix on a sequential grid, which is what makes a tile
            id carry two positions rather than one.
        """
        # Act & assert
        assert CoolerTileset.grid_policy == "sequential"
    def test_modifiers_should_allow_an_unknown_transform_through(self):
        """Test that the modifier spec does not enumerate weight columns.

        Given:
            The declared transform modifier.
        When:
            Its unknown-value policy is read.
        Then:
            It should allow unknowns. A cooler may carry any named weight
            column, so the parser cannot whitelist them; ``resolve_balance``
            validates against the actual bin table instead.
        """
        # Act & assert
        assert CoolerTileset.modifiers.allow_unknown is True
        assert CoolerTileset.modifiers.default == "default"

    def test_options_should_be_empty(self):
        """Test that no tile-id options are accepted.

        Given:
            The declared options.
        When:
            They are read.
        Then:
            They should be empty, so an unrecognized ``,key:value`` is rejected
            at parse time rather than silently ignored.
        """
        # Act & assert
        assert CoolerTileset.options == frozenset()


class TestCoolerTilesetInfo:
    """The tileset info, built from the stored resolutions."""

    def test_resolutions_should_be_ascending(self, make_mcool):
        """Test the ladder order.

        Given:
            An mcool whose resolution groups sort differently as strings than
            as integers -- ``2, 10, 100`` are stored under HDF5 names that
            enumerate as ``10, 100, 2``.
        When:
            The resolutions are read.
        Then:
            They should come back ascending numerically, matching what legacy
            cooler serializes. ``z`` indexes coarsest-first; ``resolution_for``
            re-sorts internally, so the wire order is fixed only so two
            tilesets do not emit the same field two ways.

            The default ``1, 2, 4`` ladder cannot test this: its group names
            enumerate in the same order they sort numerically, so dropping the
            sort entirely leaves the answer unchanged.
        """
        # Arrange
        path = make_mcool(name="wide-ladder.mcool", resolutions=(2, 10, 100))

        # Act
        with CoolerTileset(str(path)) as tileset:
            resolutions = tileset.resolutions

        # Assert
        assert resolutions == (2, 10, 100)

    def test_resolutions_should_be_computed_once(self, tileset):
        """Test the resolution cache.

        Given:
            A tileset whose info build walks the resolutions repeatedly.
        When:
            The property is read twice.
        Then:
            It should return the identical tuple rather than re-reading the
            h5py group each time.
        """
        # Act & assert
        assert tileset.resolutions is tileset.resolutions

    def test_info_should_declare_an_explicit_ladder(self, tileset):
        """Test the ladder form.

        Given:
            An mcool, which enumerates its resolutions.
        When:
            The info is read.
        Then:
            It should be explicit, and carry no ``max_width``: for an explicit
            ladder the extent belongs to the zoom level, since only a
            power-of-two ladder makes it invariant.
        """
        # Act
        info = tileset.info()

        # Assert
        assert info.ladder is Ladder.EXPLICIT
        assert info.resolutions == [1, 2, 4]
        assert info.max_width is None
        assert info.tile_size == TILE_SIZE

    def test_info_should_span_the_genome_on_both_axes(self, tileset):
        """Test the declared extent.

        Given:
            The 3000 bp canonical genome.
        When:
            The info is read.
        Then:
            Both axes should run to the genome length, and ``min_pos`` should
            be one-based on both -- which is what legacy cooler emits.
        """
        # Act
        info = tileset.info()

        # Assert
        assert info.min_pos == [1, 1]
        assert info.max_pos == [genome.CANONICAL_TOTAL] * 2

    def test_canvas_should_size_the_extent_to_the_zoom(self, tileset):
        """Test that an explicit ladder derives its lattice per level.

        Given:
            A three-level ladder at 4, 2 and 1 bp per bin.
        When:
            Each level's canvas is built.
        Then:
            The tile counts should be 3, 6 and 12 -- however many 256-bin tiles
            it takes to cover the genome at that resolution. Inheriting one
            extent from the coarsest level would over-report every finer zoom.
        """
        # Act
        counts = [tileset.info().canvas(z).n_tiles for z in range(3)]

        # Assert
        assert counts == [3, 6, 12]

    def test_info_should_advertise_no_transforms_without_a_weight_column(
        self, tileset
    ):
        """Test the transform dropdown on an unbalanced cooler.

        Given:
            A cooler whose bin table holds only coordinate columns.
        When:
            The info is read.
        Then:
            It should advertise no transforms, since the three coordinate
            columns are not balancing candidates.
        """
        # Act & assert
        assert tileset.info().model_dump()["transforms"] == []

    def test_info_should_label_the_weight_column_as_ice(self, make_mcool):
        """Test the transform dropdown on a balanced cooler.

        Given:
            A cooler carrying a weight column at every resolution.
        When:
            The info is read.
        Then:
            It should offer it under the literal label ``ICE``, which is the
            name the client shows for the standard balancing vector. Reading
            the label back out of ``TRANSFORM_LABELS`` -- the table the unit
            itself indexes -- asserts the source equals itself and never sees
            a wrong label at all.
        """
        # Act
        with CoolerTileset(str(make_mcool(weights={"weight": 0.5}))) as tileset:
            transforms = tileset.info().model_dump()["transforms"]

        # Assert
        assert transforms == [{"name": "ICE", "value": "weight"}]

    def test_info_should_omit_mirror_tiles_for_symmetric_storage(
        self, tileset
    ):
        """Test the default storage mode.

        Given:
            A cooler stored as the upper triangle, which is the norm.
        When:
            The info is read.
        Then:
            It should not set ``mirror_tiles``, leaving the client to mirror
            across the diagonal itself.
        """
        # Act & assert
        assert "mirror_tiles" not in tileset.info().model_dump()

    def test_info_should_set_mirror_tiles_for_square_storage(self, make_mcool):
        """Test the non-symmetric storage mode.

        Given:
            A cooler written as a full square rather than an upper triangle.
        When:
            The info is read.
        Then:
            It should set ``mirror_tiles`` to the string "false", telling the
            client not to reflect the tile across the diagonal -- the stored
            matrix already holds both halves, and mirroring would double them.
        """
        # Arrange
        path = make_mcool(name="square.mcool", symmetric_upper=False)

        # Act
        with CoolerTileset(str(path)) as tileset:
            info = tileset.info().model_dump()

        # Assert
        assert info["mirror_tiles"] == "false"

    def test_policy_should_be_the_one_supplied_at_construction(
        self, shared_mcool
    ):
        """Test that the tileset carries its caller's limits.

        Given:
            A tileset constructed with a non-default policy.
        When:
            Its policy is read.
        Then:
            It should be that policy.
        """
        # Arrange

        policy = DEFAULT_POLICY.with_(max_tile_width=1_000_000)

        # Act
        with CoolerTileset(str(shared_mcool), policy=policy) as tileset:
            # Assert
            assert tileset.policy is policy

    def test_info_should_return_the_same_object_on_every_call(self, tileset):
        """Test that the info is built once.

        Given:
            A tileset whose info build opens a cooler per resolution.
        When:
            ``info`` is called twice.
        Then:
            It should return the identical object, preserving the cached
            coordinate system every tile depends on.
        """
        # Act & assert
        assert tileset.info() is tileset.info()

    def test_chromsizes_should_come_from_the_stored_bin_table(self, tileset):
        """Test the coordinate system.

        Given:
            A cooler built over the canonical genome.
        When:
            Its chromsizes are read.
        Then:
            They should match, in the bin table's own contig order.
        """
        # Act & assert
        assert tileset.chromsizes().to_pairs() == [
            list(p) for p in genome.CANONICAL_CHROMSIZES
        ]


class TestCoolerTilesetLifetime:
    """Handle ownership and the single-resolution refusal."""

    def test_file_should_reject_a_cooler_with_no_resolutions_group(
        self, tmp_path
    ):
        """Test a flat single-resolution .cool.

        Given:
            A plain cooler with no ``resolutions`` group.
        When:
            Its file is opened.
        Then:
            It should raise, naming the legacy module that serves that layout.
            Falling through would fail later with a KeyError from h5py.
        """
        # Arrange

        path = tmp_path / "flat.cool"
        bins = pd.DataFrame(
            {
                "chrom": pd.Categorical(["c1"] * 10, categories=["c1"]),
                "start": np.arange(0, 1000, 100),
                "end": np.arange(100, 1100, 100),
            }
        )
        pixels = pd.DataFrame({"bin1_id": [0], "bin2_id": [1], "count": [5]})
        cooler.create_cooler(str(path), bins, pixels)

        # Act & assert
        with pytest.raises(ValueError, match="no 'resolutions' group"):
            CoolerTileset(str(path)).info()

    def test_file_should_return_the_same_handle_on_every_read(self, shared_mcool):
        """Test the handle cache.

        Given:
            A freshly constructed tileset, which opens nothing.
        When:
            The file property is read twice.
        Then:
            It should hand back the same handle. Note this does not test
            laziness despite the property being lazy -- only reuse; a handle
            opened eagerly in ``__init__`` is equally reused.
        """
        # Arrange
        tileset = CoolerTileset(str(shared_mcool))

        # Act & assert
        assert tileset.file is tileset.file
        tileset.close()

    def test_close_should_be_safe_to_call_twice(self, shared_mcool):
        """Test idempotent release.

        Given:
            A tileset that has opened its file.
        When:
            It is closed twice.
        Then:
            The second call should be a no-op.
        """
        # Arrange
        tileset = CoolerTileset(str(shared_mcool))
        handle = tileset.file

        # Act
        tileset.close()
        tileset.close()

        # Assert
        assert tileset.file is not handle

    def test___exit___should_release_the_handle(self, shared_mcool):
        """Test context-manager use.

        Given:
            A tileset used as a context manager.
        When:
            The block exits.
        Then:
            The handle should be released.
        """
        # Act
        with CoolerTileset(str(shared_mcool)) as tileset:
            handle = tileset.file

        # Assert
        assert tileset.file is not handle


class TestCoolerTilesetTiles:
    """Tile generation."""

    def test_tiles_should_return_one_entry_per_requested_id(self, tileset):
        """Test batch shape.

        Given:
            Three in-range tile ids.
        When:
            They are generated in one call.
        Then:
            The result should pair each id with its payload, in request order.
        """
        # Arrange
        ids = [
            tileset.parse_tile_id(t) for t in ("x.0.0.0", "x.1.1.0", "x.2.3.3")
        ]

        # Act
        result = tileset.tiles(ids)

        # Assert
        assert [tid for tid, _ in result] == ids

    @pytest.mark.parametrize("z,x,y", [(0, 0, 0), (1, 1, 0), (2, 3, 3), (2, 11, 11)])
    def test_tiles_should_always_be_a_square_of_tile_size(
        self, tileset, z, x, y
    ):
        """Test the invariant the 2D reconciler exists to hold.

        Given:
            Any in-range tile position, interior or overhanging.
        When:
            The tile is generated.
        Then:
            It should carry exactly ``tile_size`` squared values. Cooler ships
            no ``shape`` field, so a tile of any other size is silently
            misread rather than rejected.
        """
        # Act
        ((_, payload),) = tileset.tiles(
            [tileset.parse_tile_id(f"x.{z}.{x}.{y}")]
        )

        # Assert
        assert square(payload).shape == (TILE_SIZE, TILE_SIZE)

    def test_tiles_should_pad_only_past_the_genome_end(self, tileset):
        """Test the overhanging tile at the corner of the matrix.

        Given:
            The deepest-zoom diagonal tile that starts inside the genome and
            ends past it -- at 1 bp per bin, tile 11 begins at 2816 and the
            genome ends at 3000.
        When:
            It is generated.
        Then:
            It should hold 184 by 184 real values with the rest NaN, so the
            padding boundary lands on the genome end rather than a tile edge.
        """
        # Arrange
        z, x = genome.overhanging_tile(tileset.info())
        real = genome.CANONICAL_TOTAL - x * TILE_SIZE

        # Act
        ((_, payload),) = tileset.tiles(
            [tileset.parse_tile_id(f"x.{z}.{x}.{x}")]
        )

        # Assert
        matrix = square(payload)
        assert not np.isnan(matrix[:real, :real]).any()
        assert np.isnan(matrix[real:, :]).all()
        assert np.isnan(matrix[:, real:]).all()

    def test_tiles_should_pad_each_axis_independently(self, tileset):
        """Test a strip where one axis overhangs and the other does not.

        Given:
            A tile whose row axis runs past the genome end while its column
            axis sits entirely inside it.
        When:
            It is generated.
        Then:
            Only the rows past the end should be NaN, and every column should
            carry a real value somewhere above them. Padding one axis is what
            the 2D reconciler's block assembly has to keep separable; leaking
            it into the other axis would blank a whole strip of real contacts.

            The column claim is asserted over the in-genome rows rather than
            the whole matrix. A fully-NaN column cannot occur here anyway --
            the rows past the end are legitimately NaN in every column -- so
            ruling one out asserts nothing.
        """
        # Arrange
        z, x = genome.overhanging_tile(tileset.info())
        real = genome.CANONICAL_TOTAL - x * TILE_SIZE

        # Act
        ((_, payload),) = tileset.tiles(
            [tileset.parse_tile_id(f"x.{z}.0.{x}")]
        )

        # Assert
        matrix = square(payload)
        assert np.isnan(matrix).all(axis=1).sum() == TILE_SIZE - real
        assert not np.isnan(matrix[:real, :]).all(axis=0).any()

    def test_tiles_should_transpose_across_the_diagonal(self, tileset):
        """Test that the two positions index the axes consistently.

        Given:
            An off-diagonal tile and its mirror.
        When:
            Both are generated.
        Then:
            One should be the transpose of the other. Swapping the row and
            column axes anywhere in the block assembly produces a matrix that
            is still square, still the right size, and reflected.
        """
        # Act
        ((_, upper),) = tileset.tiles([tileset.parse_tile_id("x.2.4.7")])
        ((_, lower),) = tileset.tiles([tileset.parse_tile_id("x.2.7.4")])

        # Assert
        assert np.array_equal(
            square(upper), square(lower).T, equal_nan=True
        )

    def test_tiles_should_scale_counts_by_the_balancing_weights(
        self, make_mcool
    ):
        """Test that the transform modifier reaches cooler.

        Given:
            A cooler whose every bin carries a weight of 0.5, and the same tile
            requested raw and balanced.
        When:
            Both are generated.
        Then:
            The balanced tile should be a quarter of the raw one: cooler
            multiplies a contact by both bins' weights, so a constant ``w``
            scales by ``w**2``.
        """
        # Arrange
        with CoolerTileset(str(make_mcool(weights={"weight": 0.5}))) as tileset:
            # Act
            ((_, raw),) = tileset.tiles(
                [tileset.parse_tile_id("x.2.0.0.None")]
            )
            ((_, balanced),) = tileset.tiles(
                [tileset.parse_tile_id("x.2.0.0.weight")]
            )

        # Assert
        np.testing.assert_allclose(
            np.nan_to_num(square(balanced)),
            np.nan_to_num(square(raw)) * 0.25,
            rtol=1e-6,
        )

    def test_tiles_should_default_to_the_weight_column_when_present(
        self, make_mcool
    ):
        """Test the default transform on a balanced cooler.

        Given:
            A balanced cooler and a tile id carrying no modifier.
        When:
            It is generated alongside the same id spelled ``.weight``.
        Then:
            They should be byte-identical, so the default is not a third code
            path.
        """
        # Arrange
        with CoolerTileset(str(make_mcool(weights={"weight": 0.5}))) as tileset:
            # Act
            ((_, bare),) = tileset.tiles([tileset.parse_tile_id("x.2.0.0")])
            ((_, explicit),) = tileset.tiles(
                [tileset.parse_tile_id("x.2.0.0.weight")]
            )

        # Assert
        assert bare == explicit

    def test_tiles_should_propagate_an_unavailable_transform(self, tileset):
        """Test that a bad transform is not swallowed with the bounds errors.

        Given:
            A transform naming a column the cooler does not carry.
        When:
            The tile is requested.
        Then:
            It should raise. Only TileOutOfBounds is skipped: a client asking
            for a balancing column that does not exist has made a different
            kind of mistake than one asking for a tile off the end of the
            genome, and silently serving the raw matrix instead would misreport
            the data.
        """
        # Act & assert
        with pytest.raises(TileError, match="no balancing column"):
            tileset.tiles([tileset.parse_tile_id("x.0.0.0.KR")])

    # --- properties ---

    @PROPERTY_IO
    @given(data=st.data())
    def test_tiles_should_hold_the_matrix_shape_at_every_position(
        self, shared_mcool, data
    ):
        """Test the shape across the full ladder.

        Given:
            Any zoom in the ladder and any pair of positions at that zoom.
            The file itself is fixed rather than drawn: a cooler costs ~350 ms
            to write, so a rebuild per example would cost more than the rest of
            the suite put together.
        When:
            The tile is generated.
        Then:
            It should be a ``tile_size`` square whose real values are finite
            and non-negative -- these are contact counts -- and whose cells
            past the genome end are NaN. A shape error anywhere in the block
            assembly is invisible to the client, which infers the side from
            the length.
        """
        # Arrange
        with CoolerTileset(str(shared_mcool)) as tileset:
            info = tileset.info()
            z = data.draw(
                st.integers(min_value=0, max_value=info.num_zoom_levels - 1)
            )
            n_tiles = info.canvas(z).n_tiles
            x = data.draw(st.integers(min_value=0, max_value=n_tiles - 1))
            y = data.draw(st.integers(min_value=0, max_value=n_tiles - 1))

            # Act
            ((_, payload),) = tileset.tiles(
                [tileset.parse_tile_id(f"x.{z}.{x}.{y}")]
            )

        # Assert
        matrix = square(payload)
        assert matrix.shape == (TILE_SIZE, TILE_SIZE)
        real = matrix[~np.isnan(matrix)]
        assert np.all(np.isfinite(real))
        assert np.all(real >= 0)
        # The tile id spells the column first: ``uid.z.<column>.<row>``, which
        # ``test_tiles_should_pad_each_axis_independently`` fixes by arranging
        # the overhang on the row axis and requesting it as ``x.{z}.0.{x}``.
        cols = bins_past_the_genome(info, z, x)
        rows = bins_past_the_genome(info, z, y)
        if rows:
            assert np.all(np.isnan(matrix[-rows:, :]))
        if cols:
            assert np.all(np.isnan(matrix[:, -cols:]))

    @PROPERTY_IO
    @given(data=st.data())
    def test_tiles_should_be_symmetric_at_every_position(
        self, shared_mcool, data
    ):
        """Test transpose symmetry across the whole matrix.

        Given:
            Any off-diagonal tile position -- ``assume`` discards the diagonal,
            where a tile is trivially its own transpose and the property holds
            for an implementation that never mirrors anything.
        When:
            The tile and its mirror are generated.
        Then:
            Each should be the other's transpose. The stored matrix is an upper
            triangle, so the lower half is assembled from the same blocks read
            the other way round -- an axis swap survives every shape check and
            shows up only here.
        """
        # Arrange
        with CoolerTileset(str(shared_mcool)) as tileset:
            info = tileset.info()
            z = data.draw(
                st.integers(min_value=0, max_value=info.num_zoom_levels - 1)
            )
            n_tiles = info.canvas(z).n_tiles
            x = data.draw(st.integers(min_value=0, max_value=n_tiles - 1))
            y = data.draw(st.integers(min_value=0, max_value=n_tiles - 1))
            assume(x != y)

            # Act
            ((_, upper),) = tileset.tiles(
                [tileset.parse_tile_id(f"x.{z}.{x}.{y}")]
            )
            ((_, lower),) = tileset.tiles(
                [tileset.parse_tile_id(f"x.{z}.{y}.{x}")]
            )

        # Assert
        assert np.array_equal(
            square(upper), square(lower).T, equal_nan=True
        )


class TestCoolerTilesetBounds:
    """The skip contract, which is this format's alone.

    ``clodius/tiles/cooler.py`` ``continue``s past a zoom above the ladder in
    both ladder branches and filters out-of-bounds positions before generating
    anything, so the caller gets a short list rather than an error. The ``>`` to
    ``>=`` fix earlier in this branch exists so the boundary case reaches that
    skip instead of falling through to an IndexError, and
    ``test/tiles/test_conformance.py`` asserts the resulting empty list.
    """

    def test_tiles_should_skip_a_zoom_past_the_ladder(self, tileset):
        """Test the case the legacy guard covers.

        Given:
            A zoom one past the last resolution.
        When:
            The tile is requested.
        Then:
            It should be omitted, leaving an empty result rather than raising.
        """
        # Act
        result = tileset.tiles([tileset.parse_tile_id("x.3.0.0")])

        # Assert
        assert result == []

    def test_tiles_should_skip_a_position_past_the_canvas(self, tileset):
        """Test the position filter.

        Given:
            A position beyond the tile count at its zoom.
        When:
            The tile is requested.
        Then:
            It should be omitted. Legacy filters positions against ``max_pos``
            before generating anything, for the same reason.
        """
        # Act
        result = tileset.tiles([tileset.parse_tile_id("x.0.9.0")])

        # Assert
        assert result == []

    def test_tiles_should_keep_the_siblings_of_a_skipped_tile(self, tileset):
        """Test batch integrity, which is the point of skipping over raising.

        Given:
            A batch mixing two in-range ids with an out-of-range zoom and an
            out-of-range position.
        When:
            It is generated.
        Then:
            The two valid tiles should still come back. Letting the error
            propagate discards every sibling in the request, which is the
            failure this guard exists to prevent.
        """
        # Arrange
        ids = [
            tileset.parse_tile_id(t)
            for t in ("x.0.0.0", "x.3.0.0", "x.0.9.0", "x.1.1.1")
        ]

        # Act
        result = tileset.tiles(ids)

        # Assert
        assert [str(tid) for tid, _ in result] == ["x.0.0.0", "x.1.1.1"]
