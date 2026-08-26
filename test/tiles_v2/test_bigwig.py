"""Tests for clodius.tiles_v2.bigwig.

The reference dense 1D tileset: an implicit (quadtree) ladder served under the
SEQUENTIAL grid policy. Every fixture is synthesized into a temp directory, so
the module runs on a checkout with no git-LFS payload.

Chromosome signal is a distinct constant per contig -- ``c1`` is 1.0, ``c2`` is
2.0, ``c3`` is 3.0 -- which is what makes a one-bin shift at a chromosome
boundary readable straight off the decoded array. A run-length reading of the
tile is a stronger assertion than any aggregate, because a reconciler that drops
the wrong bin changes the run lengths while leaving the total count intact.
"""

import numpy as np
import pybigtools
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from clodius.core.policies import DEFAULT_POLICY
from clodius.core.coords import Chromsizes
from clodius.core.errors import TileOutOfBounds
from clodius.core.tileset import Ladder
from clodius.tiles_v2 import bigwig as mod
from clodius.tiles_v2.bigwig import AGGREGATION_MODES, RANGE_MODES, BigWigTileset

from ..harness import genome, strategies
from ..harness.wire import decode, n_bins


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
def tileset(make_bigwig):
    """A BigWigTileset over the canonical 3000 bp genome."""
    return BigWigTileset(str(make_bigwig()))


def runs(values):
    """``[(value, length), ...]`` for a flat array, NaN forming its own runs."""
    out = []
    for v in values:
        key = "nan" if np.isnan(v) else float(v)
        if out and out[-1][0] == key:
            out[-1][1] += 1
        else:
            out.append([key, 1])
    return [(k, n) for k, n in out]


class TestModeTables:
    """The two dropdowns the tileset info advertises."""

    def test_aggregation_modes_should_map_each_key_to_a_display_label(self):
        """Test the aggregation mode table.

        Given:
            The declared aggregation modes.
        When:
            The table is inspected.
        Then:
            Every key should be a pybigtools summary statistic and every value
            a non-empty label.
        """
        # Act & assert
        assert set(AGGREGATION_MODES) == {"mean", "min", "max", "std", "sum"}
        assert all(label for label in AGGREGATION_MODES.values())

    def test_range_modes_should_name_the_statistics_they_expand_to(self):
        """Test the range mode table.

        Given:
            The declared range modes, which return several values per bin.
        When:
            Each entry is inspected.
        Then:
            Its statistic tuple should be drawn from the aggregation modes,
            since those are the names passed through to the reader.
        """
        # Act & assert
        assert set(RANGE_MODES) == {"minMax", "whisker"}
        for _, stats in RANGE_MODES.values():
            assert set(stats) <= set(AGGREGATION_MODES)

    def test_modifiers_should_admit_every_mode_from_both_tables(self):
        """Test that the modifier spec and the mode tables agree.

        Given:
            The declared modifier values.
        When:
            They are compared against the two mode tables.
        Then:
            They should be exactly the union. A mode advertised in the info but
            rejected by the parser would render a dropdown entry that errors
            when selected.
        """
        # Act & assert
        assert set(BigWigTileset.modifiers.values) == set(
            AGGREGATION_MODES
        ) | set(RANGE_MODES)


class TestFetch:
    """The reader wrapper, which owns bin counting and the NaN contract."""

    def test_fetch_should_return_one_value_per_bin_for_a_single_statistic(
        self, make_bigwig
    ):
        """Test the ordinary single-statistic shape.

        Given:
            An in-bounds interval and one statistic.
        When:
            It is fetched at a binsize dividing the span.
        Then:
            It should return a flat array of ``span / binsize`` values.
        """
        # Arrange

        f = pybigtools.open(str(make_bigwig()))
        interval = mod.GenomicRange(0, "c1", 0, 1000)

        # Act
        out = mod.fetch(f, interval, 100.0, ("mean",))

        # Assert
        assert out.shape == (10,)
        assert np.all(out == 1.0)

    def test_fetch_should_return_one_column_per_statistic_for_a_range_mode(
        self, make_bigwig
    ):
        """Test the multi-statistic shape.

        Given:
            An interval and the four statistics a whisker tile carries.
        When:
            It is fetched.
        Then:
            It should return ``(n_bins, 4)`` rather than a flat run, so the
            caller can declare ``size=4`` without reshaping.
        """
        # Arrange

        f = pybigtools.open(str(make_bigwig()))
        interval = mod.GenomicRange(0, "c1", 0, 1000)

        # Act
        out = mod.fetch(f, interval, 100.0, RANGE_MODES["whisker"][1])

        # Assert
        assert out.shape == (10, 4)

    def test_fetch_should_round_the_bin_count_up(self, make_bigwig):
        """Test a binsize that does not divide the span.

        Given:
            A 1000 bp interval and a binsize of 300.
        When:
            It is fetched.
        Then:
            It should return four bins, not three. A short array would make the
            reconciler under-count and the tile come out narrow.
        """
        # Arrange

        f = pybigtools.open(str(make_bigwig()))

        # Act
        out = mod.fetch(f, mod.GenomicRange(0, "c1", 0, 1000), 300.0, ("mean",))

        # Assert
        assert out.shape == (4,)

    def test_fetch_should_return_nan_without_reading_when_out_of_bounds(self):
        """Test the padding contract past the last chromosome.

        Given:
            An out-of-bounds range, which carries no chromosome name.
        When:
            It is fetched with a reader that would raise if touched.
        Then:
            It should return an all-NaN block of the right shape. NaN and zero
            are not interchangeable: the client draws a gap for one and a
            zero-height bar for the other.
        """
        # Arrange
        interval = mod.GenomicRange(0, None, 0, 500)

        # Act
        out = mod.fetch(None, interval, 100.0, ("mean",))

        # Assert
        assert out.shape == (5,)
        assert np.all(np.isnan(out))

    def test_fetch_should_return_nan_when_the_chromosome_is_absent(
        self, make_bigwig
    ):
        """Test a chromosome the coordinate system declares but the file lacks.

        Given:
            Chromsizes naming a contig the bigWig does not carry, which is
            routine for chrM and the unplaced scaffolds.
        When:
            That contig's band is fetched.
        Then:
            It should come back NaN rather than propagating the reader's error.
        """
        # Arrange

        path = make_bigwig(chromsizes=[["c1", 1000], ["c2", 1500]])
        f = pybigtools.open(str(path))

        # Act
        out = mod.fetch(f, mod.GenomicRange(2, "c3", 0, 500), 100.0, ("mean",))

        # Assert
        assert np.all(np.isnan(out))

    def test_fetch_should_reraise_an_error_that_is_not_a_missing_chromosome(
        self, mocker
    ):
        """Test that the except clause is narrow.

        Given:
            A reader whose ``values`` raises for an unrelated reason.
        When:
            An in-bounds interval is fetched.
        Then:
            The error should propagate. Swallowing it would turn a corrupt file
            into a silently blank track.
        """
        # Arrange
        reader = mocker.Mock()
        reader.values.side_effect = RuntimeError("disk fell over")

        # Act & assert
        with pytest.raises(RuntimeError, match="disk fell over"):
            mod.fetch(reader, mod.GenomicRange(0, "c1", 0, 100), 10.0, ("mean",))


class TestBigWigTilesetDeclarations:
    """What the class states about itself before any file is opened."""

    def test_datatype_should_be_vector(self):
        """Test the declared datatype.

        Given:
            The tileset class.
        When:
            Its ``datatype`` is read.
        Then:
            It should be "vector", the string the wire uses for this format.
        """
        # Act & assert
        assert BigWigTileset.datatype == "vector"

    def test_ndim_should_be_one(self):
        """Test the declared coordinate arity.

        Given:
            The tileset class.
        When:
            Its ``ndim`` is read.
        Then:
            It should be 1, the number of positional slots its tile ids
            carry.
        """
        # Act & assert
        assert BigWigTileset.ndim == 1

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
        assert BigWigTileset.tile_kind == "dense"

    def test_grid_policy_should_be_sequential(self):
        """Test how this tileset bounds a single tile.

        Given:
            The tileset class.
        When:
            Its ``grid_policy`` is read.
        Then:
            It should be "sequential", describing a 1D dense vector on a sequential grid.
        """
        # Act & assert
        assert BigWigTileset.grid_policy == "sequential"
    def test_grid_threshold_should_be_one_so_drift_is_floored(self):
        """Test the bin-drift threshold.

        Given:
            The declared grid threshold.
        When:
            It is read.
        Then:
            It should be 1.0. A midpoint threshold changes *which* bins are
            dropped without changing how many, so ``expected_bins`` cannot
            catch the difference and byte comparability against the legacy
            module would quietly break.
        """
        # Act & assert
        assert BigWigTileset.grid_threshold == 1.0

    def test_options_should_declare_the_alternate_chromsizes_selector(self):
        """Test the accepted tile-id options.

        Given:
            The declared options.
        When:
            They are read.
        Then:
            They should be exactly ``cos``, since an option the class does not
            declare is rejected by the parser.
        """
        # Act & assert
        assert BigWigTileset.options == frozenset({"cos"})


class TestBigWigTilesetInfo:
    """The tileset info and the coordinate system behind it."""

    def test_chromsizes_should_be_read_from_the_file_in_natural_order(
        self, make_bigwig
    ):
        """Test the derived coordinate system.

        Given:
            A tileset over a file whose contigs are written in an order that
            is neither natural nor lexicographic.
        When:
            Its coordinate system is read.
        Then:
            It should carry the file's contigs in natural-sort order --
            ``c1, c2, c10``, which is not the file order ``c2, c1, c10`` and
            not the lexicographic order ``c1, c10, c2``.
        """
        # Arrange
        tileset = BigWigTileset(str(make_bigwig(genome.SHUFFLED_CHROMSIZES)))

        # Act
        pairs = tileset.chromsizes().to_pairs()

        # Assert
        assert pairs == [["c1", 1000], ["c2", 1500], ["c10", 500]]

    def test___init___should_prefer_explicitly_supplied_chromsizes(
        self, make_bigwig
    ):
        """Test that a caller can override the file's own contig order.

        Given:
            Chromsizes listing the same contigs in a different order.
        When:
            The tileset is constructed with them.
        Then:
            It should adopt them rather than re-deriving from the file, since
            contig order defines the tiling axis.
        """
        # Arrange
        reordered = Chromsizes.from_pairs(
            [["c2", 1500], ["c1", 1000], ["c3", 500]]
        )

        # Act
        tileset = BigWigTileset(str(make_bigwig()), chromsizes=reordered)

        # Assert
        assert tileset.chromsizes().names == ("c2", "c1", "c3")

    def test_info_should_describe_a_quadtree_over_the_genome(self, tileset):
        """Test the implicit ladder geometry.

        Given:
            A 3000 bp genome and a 1024 bp tile.
        When:
            The tileset info is read.
        Then:
            It should declare max_zoom 2 and a 4096 bp canvas -- the smallest
            power-of-two multiple of the tile size that covers the genome.
        """
        # Act
        info = tileset.info()

        # Assert
        assert info.ladder is Ladder.IMPLICIT
        assert info.tile_size == genome.TILE_SIZE
        assert info.max_zoom == 2
        assert info.max_width == 4096
        assert info.max_pos == [4096]

    def test_info_should_advertise_both_mode_tables(self, tileset):
        """Test the dropdown fields.

        Given:
            The tileset info.
        When:
            Its mode fields are read.
        Then:
            Both should be the labelled lists spelled out below, matching the
            JSON type bigbed emits for the same field. The labels are literals
            rather than a comprehension over the source's own tables: rebuilt
            from ``AGGREGATION_MODES`` the assertion reduces to the table
            equalling itself, and a wrong label ships green.
        """
        # Act
        info = tileset.info().model_dump()

        # Assert
        assert info["aggregation_modes"] == [
            {"name": "Mean", "value": "mean"},
            {"name": "Min", "value": "min"},
            {"name": "Max", "value": "max"},
            {"name": "Standard Deviation", "value": "std"},
            {"name": "Sum", "value": "sum"},
        ]
        assert info["range_modes"] == [
            {"name": "Min-Max", "value": "minMax"},
            {"name": "Whisker", "value": "whisker"},
        ]

    def test_policy_should_be_the_one_supplied_at_construction(
        self, make_bigwig
    ):
        """Test that the tileset carries its caller's limits.

        Given:
            A tileset constructed with a non-default policy.
        When:
            Its policy is read.
        Then:
            It should be that policy. The protocol exposes it so a boundary
            layer can enforce limits the tileset itself does not read.
        """
        # Arrange

        policy = DEFAULT_POLICY.with_(max_tile_width=1_000_000)

        # Act
        tileset = BigWigTileset(str(make_bigwig()), policy=policy)

        # Assert
        assert tileset.policy is policy

    def test_info_should_return_the_same_object_on_every_call(self, tileset):
        """Test that the info is built once.

        Given:
            A constructed tileset.
        When:
            ``info`` is called twice.
        Then:
            It should return the identical object. ``TilesetInfo`` caches its
            coordinate system, so rebuilding per call would discard that cache
            on every tile.
        """
        # Act & assert
        assert tileset.info() is tileset.info()


class TestBigWigTilesetLifetime:
    """Handle ownership."""

    def test_file_should_return_the_same_handle_on_every_read(self, tileset):
        """Test the handle cache.

        Given:
            A tileset whose info was built at construction.
        When:
            The file property is read twice.
        Then:
            It should hand back the same handle rather than reopening, and
            still the same one after a tile has been generated. Note this does
            not test laziness despite the property being lazy: ``__init__``
            builds the info, which opens the file, so nothing is deferred by
            the time a test can look.
        """
        # Act & assert
        assert tileset.file is tileset.file

    def test_close_should_be_safe_to_call_twice(self, tileset):
        """Test idempotent release.

        Given:
            A tileset that has opened its file.
        When:
            It is closed twice.
        Then:
            The second call should be a no-op rather than closing a stale
            handle.
        """
        # Arrange
        handle = tileset.file

        # Act
        tileset.close()
        tileset.close()

        # Assert
        assert tileset.file is not handle

    def test___exit___should_release_the_handle(self, tileset):
        """Test context-manager use.

        Given:
            A tileset used as a context manager.
        When:
            The block exits.
        Then:
            The handle should be released, so a caller need not know the
            tileset holds one.
        """
        # Act
        with tileset as ts:
            handle = ts.file

        # Assert
        assert ts.file is not handle


class TestBigWigTilesetTiles:
    """Tile generation."""

    def test_tiles_should_return_one_entry_per_requested_id(self, tileset):
        """Test batch shape.

        Given:
            Three tile ids.
        When:
            They are generated in one call.
        Then:
            The result should pair each id with its payload, in request order.
        """
        # Arrange
        ids = [tileset.parse_tile_id(t) for t in ("x.1.0", "x.1.1", "x.0.0")]

        # Act
        result = tileset.tiles(ids)

        # Assert
        assert [tid for tid, _ in result] == ids

    @pytest.mark.parametrize("z,x", [(0, 0), (1, 0), (1, 1), (2, 0), (2, 2)])
    def test_tiles_should_always_return_exactly_tile_size_bins(
        self, tileset, z, x
    ):
        """Test the invariant the whole reconciler exists to hold.

        Given:
            Any in-range tile position.
        When:
            The tile is generated.
        Then:
            It should carry exactly ``tile_size`` bins regardless of how many
            chromosome boundaries it straddles.
        """
        # Act
        ((_, payload),) = tileset.tiles([tileset.parse_tile_id(f"x.{z}.{x}")])

        # Assert
        assert n_bins(payload) == genome.TILE_SIZE

    def test_tiles_should_place_each_contig_signal_at_its_own_offset(
        self, tileset
    ):
        """Test that the concatenated bands land where the genome says.

        Given:
            The coarsest tile, whose 1024 bins span the whole 4096 bp canvas at
            4 bp per bin, over a genome whose contigs carry distinct constants.
        When:
            The tile is generated.
        Then:
            The decoded runs should be 250 bins of 1.0, 375 of 2.0, 125 of 3.0
            and 274 of NaN -- each contig's length divided by the bin size,
            then the padding past the genome end. A reconciler dropping from
            the wrong end would preserve the total and change these lengths.
        """
        # Act
        ((_, payload),) = tileset.tiles([tileset.parse_tile_id("x.0.0")])

        # Assert
        assert runs(decode(payload)) == [
            (1.0, 250),
            (2.0, 375),
            (3.0, 125),
            ("nan", 274),
        ]

    def test_tiles_should_be_entirely_nan_past_the_genome(self, tileset):
        """Test the tile position every zoom level has and HiGlass requests.

        Given:
            The deepest-zoom position lying wholly past the genome, which
            exists because ``max_width`` always exceeds the genome length.
        When:
            The tile is generated.
        Then:
            It should be a full-width band of NaN, and its reported extrema
            should serialize as the string "NaN" rather than a JSON literal
            that no parser accepts.
        """
        # Arrange
        z, x = genome.past_genome_tile(tileset.info())

        # Act
        ((_, payload),) = tileset.tiles([tileset.parse_tile_id(f"x.{z}.{x}")])

        # Assert
        assert np.all(np.isnan(decode(payload)))
        assert payload["min_value"] == "NaN"
        assert payload["max_value"] == "NaN"

    @pytest.mark.parametrize("mode,size", [("minMax", 2), ("whisker", 4)])
    def test_tiles_should_widen_a_bin_to_the_range_modes_statistics(
        self, tileset, mode, size
    ):
        """Test the range modes.

        Given:
            A range-mode modifier on a tile id.
        When:
            The tile is generated.
        Then:
            It should declare ``size`` values per bin and ship that many times
            ``tile_size`` values, so the client can stride the buffer.
        """
        # Act
        ((_, payload),) = tileset.tiles(
            [tileset.parse_tile_id(f"x.1.0.{mode}")]
        )

        # Assert
        assert payload["size"] == size
        assert len(decode(payload)) == size * genome.TILE_SIZE
        assert n_bins(payload) == genome.TILE_SIZE

    def test_tiles_should_apply_the_requested_aggregation(self, tileset):
        """Test that the modifier reaches the reader.

        Given:
            The same tile requested as a mean and as a sum, over a contig whose
            signal is a constant 1.0 at 4 bp per bin.
        When:
            Both are generated.
        Then:
            The sum should be four times the mean, since summing four
            positions of the same constant is what a sum means here.
        """
        # Arrange
        mean_id = tileset.parse_tile_id("x.0.0.mean")
        sum_id = tileset.parse_tile_id("x.0.0.sum")

        # Act
        ((_, mean_tile),) = tileset.tiles([mean_id])
        ((_, sum_tile),) = tileset.tiles([sum_id])

        # Assert
        np.testing.assert_allclose(
            decode(sum_tile)[:250], decode(mean_tile)[:250] * 4
        )

    def test_tiles_should_default_to_the_mean(self, tileset):
        """Test the modifier default.

        Given:
            A tile id with no modifier and the same id spelled ``.mean``.
        When:
            Both are generated.
        Then:
            They should be byte-identical, so the default is not a third code
            path -- and a tile asking for a *different* statistic should differ
            from both. Without that second half the test passes on a tileset
            that ignores the modifier altogether, since every spelling then
            returns the same bytes.

            The contrasting mode is ``sum`` rather than ``max``: the fixture's
            signal is constant within each contig, so mean, min and max all
            collapse to the same value and only a sum separates them.
        """
        # Act
        ((_, bare),) = tileset.tiles([tileset.parse_tile_id("x.0.0")])
        ((_, explicit),) = tileset.tiles([tileset.parse_tile_id("x.0.0.mean")])
        ((_, summed),) = tileset.tiles([tileset.parse_tile_id("x.0.0.sum")])

        # Assert
        assert bare == explicit
        assert bare != summed

    # --- properties ---

    @PROPERTY_IO
    @given(chromsizes=strategies.chromsizes_pairs(min_size=1, max_size=5),
           data=st.data())
    def test_tiles_should_hold_the_bin_count_at_every_position(
        self, make_bigwig, chromsizes, data
    ):
        """Test the bin count across the full quadtree.

        Given:
            Any zoom in the ladder and any position at that zoom.
        When:
            The tile is generated in any of the seven display modes.
        Then:
            It should carry exactly ``tile_size`` bins. This is the property
            the example tests sample: the reconciler's whole job is that a
            tile straddling any number of chromosome boundaries still comes
            out one tile wide.
        """
        # Arrange
        tileset = BigWigTileset(str(make_bigwig(chromsizes=chromsizes)))
        info = tileset.info()
        z = data.draw(st.integers(min_value=0, max_value=info.max_zoom))
        x = data.draw(
            st.integers(min_value=0, max_value=info.canvas(z).n_tiles - 1)
        )
        mode = data.draw(st.sampled_from(sorted(BigWigTileset.modifiers.values)))

        # Act
        ((_, payload),) = tileset.tiles(
            [tileset.parse_tile_id(f"x.{z}.{x}.{mode}")]
        )

        # Assert
        assert n_bins(payload) == genome.TILE_SIZE


class TestBigWigAlternateChromsizes:
    """The ``,cos:<uid>`` option, which re-orders the genome per request."""

    @pytest.fixture
    def tileset(self, make_bigwig):
        """A tileset carrying one alternate ordering, registered as ``rev``."""
        return BigWigTileset(
            str(make_bigwig()),
            chromsizes=genome.canonical(),
            chromsizes_alts={
                "rev": Chromsizes.from_pairs(
                    [["c2", 1500], ["c1", 1000], ["c3", 500]]
                )
            },
        )

    def test_tiles_should_reorder_the_axis_for_a_registered_uid(self, tileset):
        """Test that an alternate ordering moves the contig bands.

        Given:
            An alternate chromsizes putting c2 first.
        When:
            The coarsest tile is requested against it.
        Then:
            The runs should lead with c2's 375 bins of 2.0 rather than c1's 250
            of 1.0, while the multiset of runs is unchanged.
        """
        # Act
        ((_, payload),) = tileset.tiles(
            [tileset.parse_tile_id("x.0.0,cos:rev")]
        )

        # Assert
        assert runs(decode(payload)) == [
            (2.0, 375),
            (1.0, 250),
            (3.0, 125),
            ("nan", 274),
        ]

    def test_tiles_should_fall_back_to_the_default_for_an_unknown_uid(
        self, tileset
    ):
        """Test an unregistered ordering.

        Given:
            A ``cos`` uid the tileset never registered.
        When:
            The tile is generated.
        Then:
            It should serve the default ordering rather than raising. The
            option is declared, so the parser accepts it; the uid names a
            client-side selection the server may not know about.
        """
        # Act
        ((_, unknown),) = tileset.tiles(
            [tileset.parse_tile_id("x.0.0,cos:nope")]
        )
        ((_, default),) = tileset.tiles([tileset.parse_tile_id("x.0.0")])

        # Assert
        assert unknown == default


class TestBigWigTilesetBounds:
    """What happens off the end of the ladder.

    Pinned, not asserted as desirable. Unlike cooler -- whose predecessor
    ``continue``s past an out-of-range zoom, a contract this branch preserves
    -- ``clodius/tiles/bigwig.py`` has no bound check at all and lets an
    ``IndexError`` escape. That is an accident rather than a contract, so these
    record what this rewrite does today and are expected to change when the
    boundary layer decides how out-of-range ids are surfaced.
    """

    def test_tiles_should_raise_when_the_zoom_is_past_the_ladder(self, tileset):
        """Test a zoom level above max_zoom.

        Given:
            A zoom one past the deepest the ladder declares.
        When:
            The tile is requested.
        Then:
            It should raise TileOutOfBounds, taking its siblings in the batch
            with it.
        """
        # Act & assert
        with pytest.raises(TileOutOfBounds, match="exceeds max_zoom"):
            tileset.tiles([tileset.parse_tile_id("x.3.0")])

    def test_tiles_should_raise_when_the_position_is_past_the_canvas(
        self, tileset
    ):
        """Test a position beyond the tile count at its zoom.

        Given:
            Position 4 at zoom 1, where the canvas holds two tiles.
        When:
            The tile is requested.
        Then:
            It should raise TileOutOfBounds rather than returning padding.
        """
        # Act & assert
        with pytest.raises(TileOutOfBounds, match="outside the 2 tiles"):
            tileset.tiles([tileset.parse_tile_id("x.1.4")])
