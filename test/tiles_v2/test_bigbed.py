"""Tests for clodius.tiles_v2.bigbed.

The first *record* tileset: a tile is a variable-length list of features rather
than a grid of values, so none of ``Canvas``'s bin machinery applies -- only
``invert``, to turn a tile position into ranges to query. Density is bounded by
subsampling at request time, because bigBed has no aggregation pass to
precompute importance from.

The synthetic fixture tiles each contig with fixed-width entries every 200 bp:
five on c1, eight on c2, three on c3, sixteen in all. Counting them is what
makes the cap and the boundary-straddle assertions readable.
"""

import numpy as np
import pybigtools
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from clodius.core.coords import Chromsizes, GenomicRange
from clodius.core.errors import TileOutOfBounds
from clodius.core.policies import DEFAULT_POLICY
from clodius.core.tileset import Ladder
from clodius.tiles_v2.bigbed import (
    BigBedTileset,
    downsample_ranges,
    fetch_records,
    max_records,
    to_bedlike,
)

from ..harness import genome, strategies

#: Entries the default fixture holds, by contig. ``build_bigbed`` steps by 200
#: and stops before ``length - width``, so c1 gets five, c2 eight, c3 three.
ENTRIES_PER_CONTIG = {"c1": 5, "c2": 8, "c3": 3}
TOTAL_ENTRIES = sum(ENTRIES_PER_CONTIG.values())


#: For the pure-arithmetic properties in this module, which build nothing\n#: and take only the module's own tileset fixture.\nPROPERTY = settings(\n    max_examples=200,\n    suppress_health_check=[HealthCheck.function_scoped_fixture],\n)\n\n# Property tests here rebuild a real bigWig, bigBed, mcool or HDF5 file per
# example, so they cannot run on the ``pure`` profile's 200-example arithmetic
# budget. ``function_scoped_fixture`` is suppressed for the same reason it is
# in ``test/core/test_coords.py``: the fixtures these draw against are files
# built once per test, not state that leaks between examples.
#: For properties that only parse a tile id and read a policy. They take
#: the module's tileset fixture but build nothing per example, so they keep
#: the wide budget.
PROPERTY = settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

PROPERTY_IO = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@pytest.fixture
def tileset(make_bigbed):
    """A BigBedTileset over the canonical 3000 bp genome."""
    return BigBedTileset(str(make_bigbed()))


class TestMaxRecords:
    """The per-tile cap, which is a policy ceiling a request may only lower."""

    def test_max_records_should_fall_back_to_the_policy_ceiling(self, tileset):
        """Test a tile id carrying no cap.

        Given:
            A tile id with no ``,max:N`` option.
        When:
            The cap is resolved.
        Then:
            It should be the policy's own ceiling.
        """
        # Act & assert
        assert (
            max_records(tileset.parse_tile_id("x.0.0"), DEFAULT_POLICY)
            == DEFAULT_POLICY.max_entries_per_tile
        )

    def test_max_records_should_honor_a_request_below_the_ceiling(
        self, tileset
    ):
        """Test a request lowering the cap.

        Given:
            A ``,max:5`` on a tile id, below the policy ceiling.
        When:
            The cap is resolved.
        Then:
            It should be 5.
        """
        # Act & assert
        assert max_records(tileset.parse_tile_id("x.0.0,max:5"), DEFAULT_POLICY) == 5

    def test_max_records_should_clamp_a_request_above_the_ceiling(
        self, tileset
    ):
        """Test that a client cannot raise its own limit.

        Given:
            A ``,max:`` far above the policy ceiling.
        When:
            The cap is resolved.
        Then:
            It should clamp to the ceiling. Otherwise the limit is advisory and
            a client can ask for an unbounded tile.
        """
        # Arrange
        tid = tileset.parse_tile_id("x.0.0,max:999999")

        # Act
        cap = max_records(tid, DEFAULT_POLICY)

        # Assert
        assert cap == DEFAULT_POLICY.max_entries_per_tile

    def test_max_records_should_fall_back_to_the_ceiling_on_a_bad_value(
        self, tileset
    ):
        """Test a non-numeric cap.

        Given:
            A ``,max:`` whose value does not parse as an integer.
        When:
            The cap is resolved.
        Then:
            It should fall back to the ceiling rather than raising, since the
            option is a hint and a malformed one should not cost the tile.
        """
        # Act & assert
        assert (
            max_records(tileset.parse_tile_id("x.0.0,max:abc"), DEFAULT_POLICY)
            == DEFAULT_POLICY.max_entries_per_tile
        )

    @pytest.mark.parametrize("raw", ["0", "-4"])
    def test_max_records_should_floor_a_non_positive_request_at_one(
        self, tileset, raw
    ):
        """Test a cap of zero or below.

        Given:
            A ``,max:`` at or below zero.
        When:
            The cap is resolved.
        Then:
            It should be 1. Passing zero through would reach
            ``take_most_important``, which reads a non-positive cap as "return
            nothing" -- a plausible reading, but not what a client asking for a
            tile means.
        """
        # Act & assert
        assert max_records(tileset.parse_tile_id(f"x.0.0,max:{raw}"), DEFAULT_POLICY) == 1

    def test_max_records_should_return_none_when_the_policy_sets_no_ceiling(
        self, tileset
    ):
        """Test an unlimited policy.

        Given:
            A policy with no per-tile entry cap.
        When:
            A cap is resolved for an id carrying no option.
        Then:
            It should be None, which the thinning helper reads as "no limit".
        """
        # Arrange
        policy = DEFAULT_POLICY.with_(max_entries_per_tile=None)

        # Act & assert
        assert max_records(tileset.parse_tile_id("x.0.0"), policy) is None

    def test_max_records_should_honor_a_request_when_the_policy_is_unlimited(
        self, tileset
    ):
        """Test a request under an unlimited policy.

        Given:
            A policy with no ceiling and an explicit ``,max:7``.
        When:
            The cap is resolved.
        Then:
            It should be 7. ``min`` against a None ceiling would raise, so the
            unlimited case is a separate branch.
        """
        # Arrange
        policy = DEFAULT_POLICY.with_(max_entries_per_tile=None)

        # Act & assert
        assert max_records(tileset.parse_tile_id("x.0.0,max:7"), policy) == 7


    # --- properties ---

    @PROPERTY
    @given(
        requested=st.integers(min_value=-50, max_value=10_000),
        ceiling=st.one_of(st.none(), st.integers(min_value=1, max_value=5_000)),
    )
    def test_max_records_should_never_exceed_the_ceiling(
        self, tileset, requested, ceiling
    ):
        """Test the clamp across the whole request/ceiling domain.

        Given:
            Any requested cap, including negative and absurdly large ones, and
            any policy ceiling including none at all.
        When:
            The effective cap is resolved.
        Then:
            It should never exceed the ceiling and never fall below one. Both
            halves are load-bearing: a request that raises its own ceiling
            makes the limit advisory, and a cap of zero or less makes
            ``ranked[-0:]`` the whole list -- the unbounded tile the cap exists
            to prevent. Seven examples cannot cover a two-dimensional domain
            whose interesting points are all at the boundaries.
        """
        # Arrange
        tid = tileset.parse_tile_id(f"x.0.0,max:{requested}")
        policy = DEFAULT_POLICY.with_(max_entries_per_tile=ceiling)

        # Act
        cap = max_records(tid, policy)

        # Assert
        if ceiling is None:
            assert cap is None or cap >= 1
        else:
            assert 1 <= cap <= ceiling


class TestDownsampleRanges:
    """Contig sampling, for genomes with more contigs than a tile should query."""

    @pytest.fixture
    def ranges(self):
        """Ten ranges of strictly increasing span, on distinct contigs."""
        return [
            GenomicRange(i, f"c{i}", 0, (i + 1) * 100) for i in range(10)
        ]

    def test_downsample_ranges_should_return_every_range_when_under_the_cap(
        self, ranges, tileset, mocker
    ):
        """Test the common case of a genome with few contigs.

        Given:
            Fewer ranges than the sample size.
        When:
            They are downsampled.
        Then:
            Every one should survive, and no random draw should be taken --
            the early return is what keeps a small genome off the sampling
            path entirely, and the spy is how that half of the claim becomes
            an assertion rather than a comment.
        """
        # Arrange
        spy = mocker.spy(np.random, "default_rng")

        # Act
        out = downsample_ranges(ranges, tileset.parse_tile_id("x.0.0"), 99)

        # Assert
        assert out == ranges
        assert spy.call_count == 0

    def test_downsample_ranges_should_return_the_requested_count(
        self, ranges, tileset
    ):
        """Test the sample size.

        Given:
            Ten ranges and a sample size of four.
        When:
            They are downsampled.
        Then:
            Exactly four should come back, in genome order, with no repeats --
            the legacy module samples *with* replacement, so a contig could be
            queried twice while another was dropped.
        """
        # Act
        out = downsample_ranges(ranges, tileset.parse_tile_id("x.0.0"), 4)

        # Assert
        assert len(out) == 4
        assert len({gr.name for gr in out}) == 4
        assert out == sorted(out, key=lambda gr: gr.cid)

    def test_downsample_ranges_should_be_deterministic_for_a_tile(
        self, ranges, tileset
    ):
        """Test reproducibility.

        Given:
            The same ranges and the same tile id.
        When:
            They are downsampled twice.
        Then:
            The same contigs should be chosen, so a tile is cacheable and does
            not change under the client between requests.
        """
        # Arrange
        tid = tileset.parse_tile_id("x.1.1")

        # Act
        first = downsample_ranges(ranges, tid, 3)
        second = downsample_ranges(ranges, tid, 3)

        # Assert
        assert first == second

    def test_downsample_ranges_should_drop_zero_length_ranges_first(
        self, tileset
    ):
        """Test the degenerate range a tile boundary produces.

        Given:
            A range list containing a zero-length range, which ``invert``
            yields whenever a tile boundary lands exactly on a chromosome
            boundary.
        When:
            The list is downsampled.
        Then:
            The empty range should be gone. Left in, it makes the weight vector
            sum to zero when every range is empty, and numpy refuses
            ``replace=False`` once fewer than ``n`` weights are nonzero.
        """
        # Arrange
        ranges = [
            GenomicRange(0, "a", 5, 5),
            GenomicRange(1, "b", 0, 10),
            GenomicRange(2, "c", 7, 7),
        ]

        # Act
        out = downsample_ranges(ranges, tileset.parse_tile_id("x.0.0"), 2)

        # Assert
        assert [gr.name for gr in out] == ["b"]


class TestRecordConversion:
    """Turning a raw bigBed record into the client's bedlike shape."""

    def test_fetch_records_should_return_nothing_for_an_out_of_bounds_range(
        self,
    ):
        """Test the padding case.

        Given:
            A range past the last chromosome, which carries no name.
        When:
            Records are fetched with a reader that would raise if touched.
        Then:
            It should return an empty list without reading.
        """
        # Act & assert
        assert fetch_records(None, GenomicRange(0, None, 0, 100)) == []

    def test_fetch_records_should_prepend_the_chromosome_name(
        self, make_bigbed
    ):
        """Test the record shape the converter expects.

        Given:
            An in-bounds range over the fixture.
        When:
            Records are fetched.
        Then:
            Each should lead with its contig name, since a bigBed record
            carries only coordinates relative to a chromosome it does not
            name.
        """
        # Arrange

        f = pybigtools.open(str(make_bigbed()))

        # Act
        records = fetch_records(f, GenomicRange(0, "c1", 0, 1000))

        # Assert
        assert len(records) == ENTRIES_PER_CONTIG["c1"]
        assert all(r[0] == "c1" for r in records)

    def test_to_bedlike_should_place_the_record_on_the_genome_axis(self):
        """Test the coordinate translation.

        Given:
            A record on a contig whose offset is 1000.
        When:
            It is converted.
        Then:
            Its span should be shifted by that offset, and the offset itself
            reported so the client can undo the shift.
        """
        # Act
        row = to_bedlike(("c2", 10, 60, "name"), {"c1": 0, "c2": 1000})

        # Assert
        assert row["chrOffset"] == 1000
        assert (row["xStart"], row["xEnd"]) == (1010, 1060)

    def test_to_bedlike_should_stringify_every_field(self):
        """Test the wire type of ``fields``.

        Given:
            A record carrying integer coordinates.
        When:
            It is converted.
        Then:
            Every field should be a string, matching what BedlikeTile declares
            and what the bed tileset emits. The legacy module ships the raw
            mixed-type tuple.
        """
        # Act
        row = to_bedlike(("c1", 10, 60, 500), {"c1": 0})

        # Assert
        assert row["fields"] == ["c1", "10", "60", "500"]

    def test_to_bedlike_should_return_none_for_an_unplaceable_contig(self):
        """Test a record whose contig the coordinate system does not know.

        Given:
            A record on a contig absent from the offsets.
        When:
            It is converted.
        Then:
            It should return None rather than raising KeyError, which is not a
            TileError and would escape the server boundary as a 500.
        """
        # Act & assert
        assert to_bedlike(("cUNKNOWN", 0, 10, "z"), {"c1": 0}) is None

    def test_to_bedlike_should_derive_importance_from_the_record_digest(self):
        """Test that importance is stable rather than rolled per request.

        Given:
            The same record converted twice.
        When:
            The two importances are compared.
        Then:
            They should be equal and inside [0, 1), and a different record
            should get a different value. ``random.random()`` -- what the
            legacy module emits -- makes a feature flicker in and out of a
            tile between requests and between zoom levels; a *constant*
            satisfies stability just as well and ranks every record equally,
            which is why the third assertion is here.
        """
        # Act
        first = to_bedlike(("c1", 10, 60, "a"), {"c1": 0})
        second = to_bedlike(("c1", 10, 60, "a"), {"c1": 0})
        other = to_bedlike(("c1", 70, 120, "b"), {"c1": 0})

        # Assert
        assert first["importance"] == second["importance"]
        assert 0.0 <= first["importance"] < 1.0
        assert first["importance"] != other["importance"]

    def test_to_bedlike_should_give_distinct_records_distinct_uids(self):
        """Test that the digest discriminates.

        Given:
            Two records differing only in their name field.
        When:
            Both are converted.
        Then:
            Their uids should differ, so the client can key on one.
        """
        # Act
        a = to_bedlike(("c1", 10, 60, "a"), {"c1": 0})
        b = to_bedlike(("c1", 10, 60, "b"), {"c1": 0})

        # Assert
        assert a["uid"] != b["uid"]


class TestBigBedTilesetDeclarations:
    """What the class states about itself."""

    def test_datatype_should_be_bedlike(self):
        """Test the declared datatype.

        Given:
            The tileset class.
        When:
            Its ``datatype`` is read.
        Then:
            It should be "bedlike", the string the wire uses for this format.
        """
        # Act & assert
        assert BigBedTileset.datatype == "bedlike"

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
        assert BigBedTileset.ndim == 1

    def test_tile_kind_should_be_bedlike(self):
        """Test the declared payload kind.

        Given:
            The tileset class.
        When:
            Its ``tile_kind`` is read.
        Then:
            It should be "bedlike", which selects the payload shape a client
            decodes.
        """
        # Act & assert
        assert BigBedTileset.tile_kind == "bedlike"

    def test_density_policy_should_be_subsampled(self):
        """Test how this tileset bounds a single tile.

        Given:
            The tileset class.
        When:
            Its ``density_policy`` is read.
        Then:
            It should be "subsampled", describing 1D bedlike records thinned at request time, since bigBed has no
            aggregation pass to stratify by.
        """
        # Act & assert
        assert BigBedTileset.density_policy == "subsampled"
    def test_options_should_declare_the_chromsizes_and_cap_selectors(self):
        """Test the accepted tile-id options.

        Given:
            The declared options.
        When:
            They are read.
        Then:
            They should be ``cos`` and ``max``, the two the tile path reads.
        """
        # Act & assert
        assert BigBedTileset.options == frozenset({"cos", "max"})

    def test_modifiers_should_admit_only_the_declared_range_modes(self):
        """Test the modifier spec.

        Given:
            The declared modifier values.
        When:
            They are compared with the range-mode table.
        Then:
            It should admit ``significant`` and nothing else, and default to
            it. Comparing the spec against ``RANGE_MODES`` reduces to
            ``frozenset(RANGE_MODES) == set(RANGE_MODES)`` -- true for any
            table whatsoever, including an empty one. The mode is inert -- the
            legacy module accepts it and never reads it -- but it is kept so
            the dropdown still renders and the id still parses.
        """
        # Act & assert
        assert set(BigBedTileset.modifiers.values) == {"significant"}
        assert BigBedTileset.modifiers.default == "significant"


class TestBigBedTilesetInfo:
    """The tileset info, which this module builds rather than borrowing."""

    def test_info_should_reuse_the_bigwig_quadtree_geometry(self, tileset):
        """Test the ladder.

        Given:
            The 3000 bp genome at a 1024 bp tile size.
        When:
            The info is read.
        Then:
            It should be the same quadtree bigwig declares. Changing
            ``max_width`` would move every tile boundary, so the geometry is
            deliberately inherited even though the rest of the info is not.
        """
        # Act
        info = tileset.info()

        # Assert
        assert info.ladder is Ladder.IMPLICIT
        assert (info.max_zoom, info.max_width, info.tile_size) == (2, 4096, 1024)

    def test_info_should_not_advertise_aggregation_modes(self, tileset):
        """Test the field the legacy module inherits by accident.

        Given:
            The tileset info.
        When:
            It is inspected.
        Then:
            It should carry no aggregation modes. The legacy module copies
            bigwig's info wholesale and so advertises mean/min/max/std/sum on a
            tileset that never bins anything and cannot compute a mean.
        """
        # Act & assert
        assert "aggregation_modes" not in tileset.info().model_dump()

    def test_info_should_emit_range_modes_as_a_list(self, tileset):
        """Test the wire type of ``range_modes``.

        Given:
            The tileset info.
        When:
            ``range_modes`` is read.
        Then:
            It should be a list of name/value objects. The legacy module
            assigns the bare dict, so the same field arrives as an object from
            bigbed and an array from bigwig.
        """
        # Act
        modes = tileset.info().model_dump()["range_modes"]

        # Assert
        assert modes == [{"name": "Significant", "value": "significant"}]

    def test_chromsizes_should_be_read_from_the_file_in_natural_order(
        self, make_bigbed
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
        tileset = BigBedTileset(str(make_bigbed(genome.SHUFFLED_CHROMSIZES)))

        # Act
        pairs = tileset.chromsizes().to_pairs()

        # Assert
        assert pairs == [["c1", 1000], ["c2", 1500], ["c10", 500]]

    def test_policy_should_be_the_one_supplied_at_construction(
        self, make_bigbed
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
        policy = DEFAULT_POLICY.with_(max_entries_per_tile=7)

        # Act
        tileset = BigBedTileset(str(make_bigbed()), policy=policy)

        # Assert
        assert tileset.policy is policy


class TestBigBedTilesetLifetime:
    """Handle ownership."""

    def test_file_should_return_the_same_handle_on_every_read(self, tileset):
        """Test the handle cache.

        Given:
            A constructed tileset.
        When:
            The file property is read twice.
        Then:
            It should hand back the same handle. Note this does not test
            laziness despite the property being lazy -- only reuse; a handle
            opened eagerly in ``__init__`` is equally reused.
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
            The second call should be a no-op.
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
            The handle should be released.
        """
        # Act
        with tileset as ts:
            handle = ts.file

        # Assert
        assert ts.file is not handle


class TestBigBedTilesetTiles:
    """Tile generation."""

    def test_tiles_should_return_one_entry_per_requested_id(self, tileset):
        """Test batch shape.

        Given:
            Three tile ids.
        When:
            They are generated in one call.
        Then:
            The result should pair each id with its records, in request order,
            with the whole-genome id carrying records -- asserting only that
            the ids round-trip is satisfied by ``[(tid, []) for tid in ids]``.
        """
        # Arrange
        ids = [tileset.parse_tile_id(t) for t in ("x.2.0", "x.2.1", "x.0.0")]

        # Act
        result = tileset.tiles(ids)

        # Assert
        assert [tid for tid, _ in result] == ids
        assert result[-1][1]

    def test_tiles_should_return_every_record_at_the_coarsest_zoom(
        self, tileset
    ):
        """Test the whole-genome tile.

        Given:
            The single tile at zoom 0, which spans the entire canvas.
        When:
            It is generated.
        Then:
            It should carry every entry in the file, since the cap is far above
            sixteen.
        """
        # Act
        ((_, records),) = tileset.tiles([tileset.parse_tile_id("x.0.0")])

        # Assert
        assert len(records) == TOTAL_ENTRIES

    def test_tiles_should_return_nothing_past_the_genome(self, tileset):
        """Test the tile position every implicit ladder has.

        Given:
            The deepest-zoom position lying wholly past the genome.
        When:
            It is generated.
        Then:
            It should return no records rather than raising, since every range
            it inverts to is out of bounds.
        """
        # Arrange
        z, x = genome.past_genome_tile(tileset.info())

        # Act
        ((_, records),) = tileset.tiles([tileset.parse_tile_id(f"x.{z}.{x}")])

        # Assert
        assert records == []

    def test_tiles_should_return_a_straddling_record_in_both_tiles(
        self, tileset
    ):
        """Test a feature crossing a tile boundary.

        Given:
            The entry at the start of c2, whose genome-axis span 1000-1050
            crosses the 1024 bp boundary between the first two tiles at the
            deepest zoom.
        When:
            Both tiles are generated.
        Then:
            It should appear in both. A record is not partitioned by tile the
            way a bin is; the client clips it.
        """
        # Act
        ((_, left),) = tileset.tiles([tileset.parse_tile_id("x.2.0")])
        ((_, right),) = tileset.tiles([tileset.parse_tile_id("x.2.1")])

        # Assert
        straddling = [r for r in left if r["xStart"] == 1000]
        assert len(straddling) == 1
        assert straddling[0]["uid"] in {r["uid"] for r in right}

    def test_tiles_should_cap_records_at_the_policy_ceiling(self, make_bigbed):
        """Test the density cap.

        Given:
            A policy capping entries below the sixteen the fixture holds.
        When:
            The whole-genome tile is generated.
        Then:
            It should return exactly the cap, in genomic order.
        """
        # Arrange
        tileset = BigBedTileset(
            str(make_bigbed()),
            policy=DEFAULT_POLICY.with_(max_entries_per_tile=4),
        )

        # Act
        ((_, records),) = tileset.tiles([tileset.parse_tile_id("x.0.0")])

        # Assert
        assert len(records) == 4
        assert [r["xStart"] for r in records] == sorted(
            r["xStart"] for r in records
        )

    def test_tiles_should_lower_the_cap_for_a_request_that_asks(self, tileset):
        """Test the per-request cap.

        Given:
            A ``,max:3`` on a tile that would otherwise return sixteen records.
        When:
            It is generated.
        Then:
            It should return three.
        """
        # Act
        ((_, records),) = tileset.tiles(
            [tileset.parse_tile_id("x.0.0,max:3")]
        )

        # Assert
        assert len(records) == 3

    def test_tiles_should_keep_the_same_records_when_the_cap_is_unchanged(
        self, tileset
    ):
        """Test that thinning is stable across requests.

        Given:
            The same capped tile requested twice.
        When:
            Both are generated.
        Then:
            They should hold the same records. Importance comes from the record
            digest rather than a fresh random draw, so a feature does not
            flicker between requests.
        """
        # Act
        ((_, first),) = tileset.tiles([tileset.parse_tile_id("x.0.0,max:5")])
        ((_, second),) = tileset.tiles([tileset.parse_tile_id("x.0.0,max:5")])

        # Assert
        assert [r["uid"] for r in first] == [r["uid"] for r in second]

    def test_tiles_should_keep_a_surviving_record_at_the_next_zoom(
        self, make_bigbed
    ):
        """Test that thinning is coherent across zoom levels.

        Given:
            A cap tight enough to thin both a coarse tile and the finer tile
            nested inside it.
        When:
            Both are generated.
        Then:
            Every record the coarse tile kept from the finer tile's range
            should also survive there. Ranking by a zoom-independent digest is
            what buys this; ``random.random()`` re-rolls per tile, so a feature
            appears at one zoom and vanishes at the next.
        """
        # Arrange
        tileset = BigBedTileset(
            str(make_bigbed()),
            policy=DEFAULT_POLICY.with_(max_entries_per_tile=4),
        )
        span = tileset.info().canvas(2).tile_span(0)

        # Act
        ((_, coarse),) = tileset.tiles([tileset.parse_tile_id("x.0.0")])
        ((_, fine),) = tileset.tiles([tileset.parse_tile_id("x.2.0")])

        # Assert
        inherited = {
            r["uid"] for r in coarse if span[0] <= r["xStart"] < span[1]
        }
        assert inherited <= {r["uid"] for r in fine}

    def test_tiles_should_return_only_records_from_declared_contigs(
        self, make_bigbed
    ):
        """Test a file carrying a contig the coordinate system omits.

        Given:
            One tileset over all three of the file's contigs and one whose
            chromsizes list only two of them.
        When:
            The whole-genome tile is generated from each.
        Then:
            The narrowed one should hold the same records minus every ``c3``
            one, and the file should genuinely contain ``c3`` records -- so an
            implementation returning nothing at all does not pass.

            Note the mechanism is that ``c3`` is never queried: ``tiles``
            inverts over the declared coordinate system and ``fetch_records``
            asks the reader for those contig names only. The ``offsets.get``
            guard in ``to_bedlike`` is not reached from here and cannot be;
            it is covered directly by
            ``test_to_bedlike_should_return_none_for_an_unplaceable_contig``.
        """
        # Arrange
        path = str(make_bigbed())
        full = BigBedTileset(path)
        narrowed = BigBedTileset(
            path,
            chromsizes=Chromsizes.from_pairs([["c1", 1000], ["c2", 1500]]),
        )

        # Act
        ((_, everything),) = full.tiles([full.parse_tile_id("x.0.0")])
        ((_, placeable),) = narrowed.tiles([narrowed.parse_tile_id("x.0.0")])

        # Assert
        assert {r["fields"][0] for r in everything} == {"c1", "c2", "c3"}
        assert {r["fields"][0] for r in placeable} == {"c1", "c2"}
        assert len(placeable) == len(
            [r for r in everything if r["fields"][0] != "c3"]
        )

    def test_tiles_should_query_only_a_sample_of_a_many_contig_genome(
        self, make_bigbed
    ):
        """Test the contig sampling limit.

        Given:
            A policy sampling at most one contig per tile, and a whole-genome
            tile that inverts to three.
        When:
            It is generated.
        Then:
            Its records should come from a single contig. On a real assembly
            with thousands of scaffolds this is what keeps one tile from
            issuing thousands of range queries.
        """
        # Arrange
        tileset = BigBedTileset(
            str(make_bigbed()),
            policy=DEFAULT_POLICY.with_(max_chroms_sampled=1),
        )

        # Act
        ((_, records),) = tileset.tiles([tileset.parse_tile_id("x.0.0")])

        # Assert
        assert len({r["fields"][0] for r in records}) == 1

    # --- properties ---

    @PROPERTY_IO
    @given(chromsizes=strategies.chromsizes_pairs(min_size=1, max_size=5),
           data=st.data())
    def test_tiles_should_never_exceed_the_cap_at_any_position(
        self, make_bigbed, chromsizes, data
    ):
        """Test that the density bound holds everywhere.

        Given:
            Any position in the ladder and any per-request cap.
        When:
            The tile is generated.
        Then:
            It should return no more records than the effective cap, and every
            record should be uniquely identified and in genomic order. An
            unbounded tile is the failure the cap exists to prevent, and it
            cannot be sampled out of a handful of examples.
        """
        # Arrange
        tileset = BigBedTileset(str(make_bigbed(chromsizes=chromsizes)))
        info = tileset.info()
        z = data.draw(st.integers(min_value=0, max_value=info.max_zoom))
        x = data.draw(
            st.integers(min_value=0, max_value=info.canvas(z).n_tiles - 1)
        )
        cap = data.draw(st.integers(min_value=1, max_value=20))

        # Act
        ((_, records),) = tileset.tiles(
            [tileset.parse_tile_id(f"x.{z}.{x},max:{cap}")]
        )

        # Assert
        assert len(records) <= cap
        assert len({r["uid"] for r in records}) == len(records)
        assert [r["xStart"] for r in records] == sorted(
            r["xStart"] for r in records
        )


class TestBigBedAlternateChromsizes:
    """The ``,cos:<uid>`` option, which re-orders the genome per request."""

    @pytest.fixture
    def tileset(self, make_bigbed):
        """A tileset carrying one alternate ordering, registered as ``rev``."""
        return BigBedTileset(
            str(make_bigbed()),
            chromsizes=genome.canonical(),
            chromsizes_alts={
                "rev": Chromsizes.from_pairs(
                    [["c2", 1500], ["c1", 1000], ["c3", 500]]
                )
            },
        )

    def test_tiles_should_reoffset_records_for_a_registered_uid(self, tileset):
        """Test that an alternate ordering moves records along the axis.

        Given:
            An alternate chromsizes putting c2 first, so c1's offset becomes
            1500 rather than 0.
        When:
            The whole-genome tile is requested against it.
        Then:
            Every c1 record should be reported at the new offset. The record
            set is unchanged; only its placement on the genome axis moves.
        """
        # Act
        ((_, records),) = tileset.tiles(
            [tileset.parse_tile_id("x.0.0,cos:rev")]
        )

        # Assert
        c1 = [r for r in records if r["fields"][0] == "c1"]
        assert c1 and all(r["chrOffset"] == 1500 for r in c1)

    def test_tiles_should_fall_back_to_the_default_for_an_unknown_uid(
        self, tileset
    ):
        """Test an unregistered ordering.

        Given:
            A ``cos`` uid the tileset never registered.
        When:
            The tile is generated.
        Then:
            It should serve the default ordering rather than raising: the
            option is declared, so the parser accepts it, and the uid names a
            client-side selection the server may not know about.
        """
        # Act
        ((_, unknown),) = tileset.tiles(
            [tileset.parse_tile_id("x.0.0,cos:nope")]
        )
        ((_, default),) = tileset.tiles([tileset.parse_tile_id("x.0.0")])

        # Assert
        assert unknown == default


class TestBigBedTilesetBounds:
    """What happens off the end of the ladder.

    Pinned, not asserted as desirable, for the same reason as bigwig: the
    legacy bigBed module borrows bigwig's info and bigwig has no bound check at
    all. Only cooler has an established skip contract.
    """

    def test_tiles_should_raise_when_the_zoom_is_past_the_ladder(self, tileset):
        """Test a zoom level above max_zoom.

        Given:
            A zoom one past the deepest the ladder declares.
        When:
            The tile is requested.
        Then:
            It should raise TileOutOfBounds.
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
            It should raise TileOutOfBounds.
        """
        # Act & assert
        with pytest.raises(TileOutOfBounds, match="outside the 2 tiles"):
            tileset.tiles([tileset.parse_tile_id("x.1.4")])
