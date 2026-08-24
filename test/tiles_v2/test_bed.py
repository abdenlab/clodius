"""Tests for clodius.tiles_v2.bed.

The other record tileset, and the only one with two query paths. With a tabix
index a tile reads just its own byte ranges; without one every tile scans the
whole file and relies on a pushed-down polars predicate to do the same
filtering. The two paths must agree on every tile, which is what most of the
parametrization here is for.

It is also the only tileset implementing ``ProvidesRegions``: a paginated
listing in file order, served without a coordinate query at all.

Every fixture is synthesized into a temp directory, so the module runs on a
checkout with no git-LFS payload.
"""

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from clodius.core.coords import GenomicRange
from clodius.core.errors import TileOutOfBounds, TilesetUnavailable
from clodius.core.policies import DEFAULT_POLICY
from clodius.core.tileset import Ladder
from clodius.tiles_v2.bed import (
    BedTileset,
    overlap_predicate,
    to_tile_record,
)

from ..harness import builders, genome, strategies

RECORDS = builders.DEFAULT_RECORDS


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
def unindexed(make_bed, chromsizes):
    """A BedTileset over a plain BED, so every tile scans the file."""
    return BedTileset(make_bed(), chromsizes)


@pytest.fixture
def indexed(make_bed_bgzf, chromsizes):
    """A BedTileset over a BGZF+tabix BED, so tiles read byte ranges."""
    return BedTileset(make_bed_bgzf(), chromsizes)


def matches(chromsizes, ranges, records=RECORDS):
    """Records that ``overlap_predicate`` keeps, evaluated through polars."""
    frame = pl.DataFrame(
        {
            "chrom": [r[0] for r in records],
            "start": [r[1] for r in records],
            "end": [r[2] for r in records],
            "rest": [r[3] for r in records],
        }
    )
    predicate = overlap_predicate(ranges, chromsizes)
    if predicate is None:
        return frame["rest"].to_list()
    return frame.filter(predicate)["rest"].to_list()


class TestOverlapPredicate:
    """The pushdown filter that stands in for an index on the scan path.

    It has to agree with what a tabix range query would return, since the two
    paths serve the same tile ids and any divergence is a tile that changes
    content when someone compresses the file.
    """

    def test_overlap_predicate_should_be_none_when_the_genome_is_covered(
        self, chromsizes, unindexed
    ):
        """Test the whole-genome tile.

        Given:
            Ranges spanning every chromosome in full.
        When:
            A predicate is built.
        Then:
            It should be None, meaning no filter is needed. Emitting a
            tautological disjunction instead would defeat the pushdown for the
            most common tile in the ladder.
        """
        # Arrange
        ranges = list(unindexed.info().canvas(0).invert(0))

        # Act & assert
        assert overlap_predicate(ranges, chromsizes) is None

    def test_overlap_predicate_should_match_nothing_past_the_genome(
        self, chromsizes
    ):
        """Test a tile lying entirely past the last chromosome.

        Given:
            Ranges that are all out of bounds.
        When:
            A predicate is built and applied.
        Then:
            It should keep no records. Returning None here would read as "no
            filter needed" and hand back the whole file.
        """
        # Arrange
        ranges = [GenomicRange(0, None, 0, 1000)]

        # Act & assert
        assert matches(chromsizes, ranges) == []

    def test_overlap_predicate_should_keep_only_overlapping_records(
        self, chromsizes
    ):
        """Test a partial range on one chromosome.

        Given:
            A range covering the first 25 bp of c1. The fixture's first record
            spans 10-20 and overlaps it; its second spans 30-40 and does not.
        When:
            The predicate is applied.
        Then:
            Only the overlapping record should survive.
        """
        # Arrange
        ranges = [GenomicRange(0, "c1", 0, 25)]

        # Act & assert
        assert matches(chromsizes, ranges) == ["a"]

    def test_overlap_predicate_should_exclude_a_record_ending_at_the_start(
        self, chromsizes
    ):
        """Test the half-open boundary, which is the off-by-one that matters.

        Given:
            A range starting exactly where the fixture's first record ends.
        When:
            The predicate is applied.
        Then:
            That record should not match. Both coordinate systems are
            half-open, so touching endpoints do not overlap -- and getting this
            wrong duplicates a feature into two adjacent tiles.
        """
        # Arrange
        ranges = [GenomicRange(0, "c1", 20, 30)]

        # Act & assert
        assert matches(chromsizes, ranges) == []

    def test_overlap_predicate_should_drop_the_coordinate_test_for_a_full_contig(
        self, chromsizes
    ):
        """Test a range covering one chromosome entirely.

        Given:
            A mix of one fully-covered contig and one partial range, and a
            record on the covered contig lying past that contig's declared
            end.
        When:
            The predicate is applied.
        Then:
            Every record on the covered contig should survive regardless of
            position -- the out-of-range one included -- alongside the
            overlapping records from the partial range. That record is what
            makes the fast path observable: with every record inside its
            contig, dropping the coordinate test and keeping it select the
            same rows and the branch cannot be seen from the outside.
        """
        # Arrange
        records = [
            ("c1", 10, 20, "a"),
            ("c1", 30, 40, "b"),
            ("c1", 2000, 2100, "past-the-end"),
            ("c2", 0, 50, "c"),
            ("c2", 100, 150, "d"),
        ]
        ranges = [
            GenomicRange(0, "c1", 0, 1000),
            GenomicRange(1, "c2", 0, 60),
        ]

        # Act & assert
        assert matches(chromsizes, ranges, records) == [
            "a",
            "b",
            "past-the-end",
            "c",
        ]

    def test_overlap_predicate_should_ignore_a_zero_length_range(
        self, chromsizes
    ):
        """Test the degenerate range a tile boundary produces.

        Given:
            Only a zero-length range, which ``invert`` yields whenever a tile
            boundary lands exactly on a chromosome boundary.
        When:
            The predicate is applied.
        Then:
            It should match nothing rather than building a term no record can
            satisfy in a disjunction that then admits everything else.
        """
        # Arrange
        ranges = [GenomicRange(0, "c1", 15, 15)]

        # Act & assert
        assert matches(chromsizes, ranges) == []


class TestToTileRecord:
    """Turning a materialized row into the client's bedlike shape."""

    def row(self, chrom="c1", start=10, end=20, rest="a", digest=0x1234):
        return {
            "chrom": chrom,
            "start": start,
            "end": end,
            "rest": rest,
            "_digest": digest,
        }

    def test_to_tile_record_should_place_the_record_on_the_genome_axis(self):
        """Test the coordinate translation.

        Given:
            A row on a contig whose offset is 1000.
        When:
            It is converted.
        Then:
            Its span should be shifted by that offset, with the offset itself
            reported so the client can undo it.
        """
        # Act
        record = to_tile_record(
            self.row(chrom="c2"), {"c1": 0, "c2": 1000}
        )

        # Assert
        assert record["chrOffset"] == 1000
        assert (record["xStart"], record["xEnd"]) == (1010, 1020)

    def test_to_tile_record_should_split_the_rest_column_into_fields(self):
        """Test the ``bed3+`` schema's trailing column.

        Given:
            A row whose ``rest`` holds two tab-joined columns.
        When:
            It is converted.
        Then:
            ``fields`` should be the coordinates as text followed by those
            columns, which is the record as the client renders it.
        """
        # Act
        record = to_tile_record(self.row(rest="name\t500"), {"c1": 0})

        # Assert
        assert record["fields"] == ["c1", "10", "20", "name", "500"]

    def test_to_tile_record_should_omit_an_empty_rest_column(self):
        """Test a plain BED3.

        Given:
            A row with nothing past the coordinates.
        When:
            It is converted.
        Then:
            ``fields`` should hold only the three coordinates rather than a
            trailing empty string.
        """
        # Act
        record = to_tile_record(self.row(rest=""), {"c1": 0})

        # Assert
        assert record["fields"] == ["c1", "10", "20"]

    def test_to_tile_record_should_return_none_for_an_unplaceable_contig(self):
        """Test a row whose contig the coordinate system does not know.

        Given:
            A row on a contig absent from the offsets -- an unplaced scaffold,
            a naming mismatch, a different assembly, or a header line parsed as
            a record.
        When:
            It is converted.
        Then:
            It should return None rather than raising KeyError, which is not a
            TileError and would escape the server boundary as a 500.
        """
        # Act & assert
        assert to_tile_record(self.row(chrom="cUNKNOWN"), {"c1": 0}) is None

    def test_to_tile_record_should_derive_importance_from_the_digest(self):
        """Test that importance and uid share one hash.

        Given:
            A row carrying a 64-bit digest.
        When:
            It is converted.
        Then:
            ``uid`` should be that digest in hex and ``importance`` the same
            digest scaled into [0, 1). Rank order is identical either way,
            which is why ``top_k`` can thin on the integer and never divide.
        """
        # Act
        record = to_tile_record(self.row(digest=0xABCD), {"c1": 0})

        # Assert
        assert record["uid"] == format(0xABCD, "016x")
        assert record["importance"] == 0xABCD / 2.0**64


class TestBedTilesetDeclarations:
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
        assert BedTileset.datatype == "bedlike"

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
        assert BedTileset.ndim == 1

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
        assert BedTileset.tile_kind == "bedlike"

    def test_density_policy_should_be_subsampled(self):
        """Test how this tileset bounds a single tile.

        Given:
            The tileset class.
        When:
            Its ``density_policy`` is read.
        Then:
            It should be "subsampled", describing 1D bedlike records thinned at request time, matching bigBed -- a BED
            has no aggregation pass to stratify by.
        """
        # Act & assert
        assert BedTileset.density_policy == "subsampled"
    def test_modifiers_should_be_absent(self):
        """Test that the format declares no modifier slot.

        Given:
            The declared modifiers.
        When:
            They are read.
        Then:
            It should be ``None``, so any modifier is rejected at parse time
            rather than silently ignored. ``None`` is a stronger statement
            than an empty spec: ``TileId.parse`` refuses a modifier outright
            against ``None`` and validates one against a spec.
        """
        # Act & assert
        assert BedTileset.modifiers is None

    def test_options_should_be_empty(self):
        """Test that the format recognizes no tile-id options.

        Given:
            The declared option keys.
        When:
            They are read.
        Then:
            It should be an empty set, so an unrecognized ``,key:value`` is
            rejected at parse time.
        """
        # Act & assert
        assert BedTileset.options == frozenset()

class TestBedTilesetConstruction:
    """Index detection and the coordinate system."""

    def test_is_indexed_should_be_false_for_a_plain_bed(self, unindexed):
        """Test index auto-detection when no sibling index exists.

        Given:
            A plain uncompressed BED.
        When:
            The tileset is constructed.
        Then:
            It should report itself unindexed.
        """
        # Act & assert
        assert unindexed.is_indexed is False

    def test_is_indexed_should_be_true_when_a_sibling_index_exists(
        self, indexed
    ):
        """Test index auto-detection from a sibling .tbi.

        Given:
            A BGZF BED with a tabix index beside it.
        When:
            The tileset is constructed.
        Then:
            It should report itself indexed without being told.
        """
        # Act & assert
        assert indexed.is_indexed is True

    def test_is_indexed_should_be_true_for_an_index_given_explicitly(
        self, make_bed_bgzf, chromsizes
    ):
        """Test an index stored away from the file.

        Given:
            An index path passed to the constructor.
        When:
            The tileset is constructed.
        Then:
            It should report itself indexed, so an index kept elsewhere is not
            silently ignored in favour of a whole-file scan.
        """
        # Arrange
        path = make_bed_bgzf()

        # Act
        tileset = BedTileset(path, chromsizes, index_path=f"{path}.tbi")

        # Assert
        assert tileset.is_indexed is True

    def test_info_should_describe_a_quadtree_over_the_genome(self, unindexed):
        """Test the implicit ladder geometry.

        Given:
            A 3000 bp genome and a 1024 bp tile.
        When:
            The info is read.
        Then:
            It should declare the same quadtree bigWig and bigBed do. The
            genome length would be a second convention on one datatype.
        """
        # Act
        info = unindexed.info()

        # Assert
        assert info.ladder is Ladder.IMPLICIT
        assert (info.max_zoom, info.max_width, info.tile_size) == (2, 4096, 1024)
        assert info.max_pos == [4096]

    def test_chromsizes_should_be_the_ones_supplied(self, unindexed, chromsizes):
        """Test the coordinate system.

        Given:
            A tileset constructed with explicit chromsizes, which a BED
            requires since it carries no lengths of its own.
        When:
            They are read back.
        Then:
            They should be the same object.
        """
        # Act & assert
        assert unindexed.chromsizes() is chromsizes

    def test_policy_should_be_the_one_supplied_at_construction(
        self, make_bed, chromsizes
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
        tileset = BedTileset(make_bed(), chromsizes, policy=policy)

        # Assert
        assert tileset.policy is policy

    def test_close_should_be_a_no_op(self, unindexed):
        """Test the lifetime declaration for a stateless reader.

        Given:
            A tileset over oxbow, which opens a source per query and holds no
            persistent handle.
        When:
            It is closed twice and then used.
        Then:
            It should still serve tiles. The method exists because the protocol
            requires it, not because there is anything to release.
        """
        # Arrange
        unindexed.close()
        unindexed.close()

        # Act
        ((_, records),) = unindexed.tiles([unindexed.parse_tile_id("x.0.0")])

        # Assert
        assert len(records) == len(RECORDS)


class TestBedTilesetScanGuard:
    """The refusal to whole-file-scan an unindexed file above a size."""

    def test_tiles_should_refuse_an_unindexed_file_above_the_ceiling(
        self, make_bed, chromsizes
    ):
        """Test the guard on the scan path.

        Given:
            A policy whose unindexed ceiling is below the file's size.
        When:
            A tile is requested.
        Then:
            It should raise TilesetUnavailable naming both the file's size and
            the ceiling, rather than reading the whole file into memory. The
            pattern spans both numbers: matching only the static tail of the
            message passes however wrong the sizes it reports are.
        """
        # Arrange
        tileset = BedTileset(
            make_bed(),
            chromsizes,
            policy=DEFAULT_POLICY.with_(max_unindexed_filesize=1),
        )

        # Act & assert
        with pytest.raises(
            TilesetUnavailable, match=r"is \d+ bytes; files above 1 must be"
        ):
            tileset.tiles([tileset.parse_tile_id("x.0.0")])

    def test_tiles_should_not_apply_the_ceiling_to_an_indexed_file(
        self, make_bed_bgzf, chromsizes
    ):
        """Test that the guard is about scanning, not about size.

        Given:
            The same ceiling against an indexed file.
        When:
            A tile is requested.
        Then:
            It should be served. An indexed tile reads its own byte ranges, so
            the file's total size does not bound the work.
        """
        # Arrange
        tileset = BedTileset(
            make_bed_bgzf(),
            chromsizes,
            policy=DEFAULT_POLICY.with_(max_unindexed_filesize=1),
        )

        # Act
        ((_, records),) = tileset.tiles([tileset.parse_tile_id("x.0.0")])

        # Assert
        assert len(records) == len(RECORDS)

    def test_tiles_should_serve_an_unindexed_file_under_the_ceiling(
        self, make_bed, chromsizes
    ):
        """Test the ordinary scan path.

        Given:
            A policy whose ceiling is above the file's size.
        When:
            A tile is requested twice.
        Then:
            Both should be served, and both should carry every record in the
            file. The size check is memoized after it first passes, so a
            second tile must not re-stat the file and must not start refusing
            -- but ``first == second`` alone is satisfied by two empty
            results, so a tileset that served nothing at all would pass.
        """
        # Arrange
        tileset = BedTileset(
            make_bed(),
            chromsizes,
            policy=DEFAULT_POLICY.with_(max_unindexed_filesize=10_000),
        )
        tid = tileset.parse_tile_id("x.0.0")

        # Act
        first = tileset.tiles([tid])
        second = tileset.tiles([tid])

        # Assert
        ((_, records),) = first
        assert len(records) == len(RECORDS)
        assert first == second


class TestBedTilesetTiles:
    """Tile generation, on both query paths."""

    def test_tiles_should_return_one_entry_per_requested_id(self, unindexed):
        """Test batch shape.

        Given:
            Three tile ids.
        When:
            They are generated in one call.
        Then:
            The result should pair each id with its records, in request order.
            The whole-genome id is included so at least one entry is non-empty:
            asserting only that the ids round-trip is satisfied by
            ``[(tid, []) for tid in ids]``.
        """
        # Arrange
        ids = [unindexed.parse_tile_id(t) for t in ("x.2.0", "x.2.1", "x.0.0")]

        # Act
        result = unindexed.tiles(ids)

        # Assert
        assert [tid for tid, _ in result] == ids
        assert len(result[-1][1]) == len(RECORDS)

    @pytest.mark.parametrize("fixture", ["unindexed", "indexed"])
    def test_tiles_should_return_the_overlapping_records(
        self, fixture, request
    ):
        """Test the ordinary in-genome case on both paths.

        Given:
            A tile covering the whole genome at the coarsest zoom.
        When:
            It is generated.
        Then:
            It should return every record, in genomic order.
        """
        # Arrange
        tileset = request.getfixturevalue(fixture)

        # Act
        ((_, records),) = tileset.tiles([tileset.parse_tile_id("x.0.0")])

        # Assert
        assert [r["fields"][0] for r in records] == [
            "c1",
            "c1",
            "c2",
            "c2",
            "c3",
        ]

    @pytest.mark.parametrize("z,x", [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)])
    def test_tiles_should_agree_between_the_indexed_and_scanned_paths(
        self, indexed, unindexed, z, x
    ):
        """Test that compressing a file does not change what it serves.

        Given:
            The same records as a plain BED and as a BGZF+tabix BED.
        When:
            The same tile is generated from each.
        Then:
            The records should be identical. The two paths filter completely
            differently -- a tabix range query against a polars predicate --
            and nothing else in the suite compares them.
        """
        # Arrange
        tid = f"x.{z}.{x}"

        # Act
        ((_, from_scan),) = unindexed.tiles([unindexed.parse_tile_id(tid)])
        ((_, from_index),) = indexed.tiles([indexed.parse_tile_id(tid)])

        # Assert
        assert from_scan == from_index

    @pytest.mark.parametrize("fixture", ["unindexed", "indexed"])
    def test_tiles_should_return_nothing_when_the_tile_is_past_the_genome(
        self, fixture, request
    ):
        """Test the tile positions HiGlass requests routinely at every zoom.

        Given:
            A tile lying entirely past the end of the genome, which every zoom
            level has because ``max_width`` always exceeds the genome length.
        When:
            It is generated, on both paths.
        Then:
            It should return nothing. oxbow reads an empty region list as *no
            region filter* and scans the whole file, so the indexed path used
            to return every record in it.
        """
        # Arrange
        tileset = request.getfixturevalue(fixture)
        z, x = genome.past_genome_tile(tileset.info())

        # Act
        ((_, records),) = tileset.tiles([tileset.parse_tile_id(f"x.{z}.{x}")])

        # Assert
        assert records == []

    def test_tiles_should_skip_records_on_a_contig_not_in_the_chromsizes(
        self, make_bed, chromsizes
    ):
        """Test a BED carrying a contig the coordinate system does not know.

        Given:
            A BED with an unplaced contig absent from the supplied chromsizes.
        When:
            A tile covering the genome is generated.
        Then:
            It should return only the placeable records rather than raising
            KeyError out of the request handler.
        """
        # Arrange
        path = make_bed(
            records=RECORDS + [("cUNKNOWN", 0, 10, "z")], name="extra.bed"
        )
        tileset = BedTileset(path, chromsizes)

        # Act
        ((_, records),) = tileset.tiles([tileset.parse_tile_id("x.0.0")])

        # Assert
        assert len(records) == len(RECORDS)
        assert all(r["fields"][0] != "cUNKNOWN" for r in records)

    def test_tiles_should_not_let_an_unknown_contig_consume_a_cap_slot(
        self, make_bed, chromsizes
    ):
        """Test the order of filtering and capping.

        Given:
            A BED whose unplaceable records outnumber the per-tile cap, and a
            cap below the count of placeable ones.
        When:
            A tile is generated.
        Then:
            It should still return a full tile. Capping before filtering would
            spend the whole budget on rows that are then discarded, silently
            shortening the tile to nothing.
        """
        # Arrange
        noise = [("cUNKNOWN", i * 10, i * 10 + 5, f"z{i}") for i in range(50)]
        tileset = BedTileset(
            make_bed(records=RECORDS + noise, name="noisy.bed"),
            chromsizes,
            policy=DEFAULT_POLICY.with_(max_entries_per_tile=3),
        )

        # Act
        ((_, records),) = tileset.tiles([tileset.parse_tile_id("x.0.0")])

        # Assert
        assert len(records) == 3

    def test_tiles_should_cap_records_at_the_policy_limit(
        self, make_bed, chromsizes
    ):
        """Test the density cap.

        Given:
            A tileset whose policy caps entries per tile below the record
            count.
        When:
            A tile covering every record is generated.
        Then:
            It should return no more than the cap, in genomic order.
        """
        # Arrange
        tileset = BedTileset(
            make_bed(),
            chromsizes,
            policy=DEFAULT_POLICY.with_(max_entries_per_tile=2),
        )

        # Act
        ((_, records),) = tileset.tiles([tileset.parse_tile_id("x.0.0")])

        # Assert
        assert len(records) == 2
        assert [r["xStart"] for r in records] == sorted(
            r["xStart"] for r in records
        )

    def test_tiles_should_return_every_record_when_the_policy_is_unlimited(
        self, make_bed, chromsizes
    ):
        """Test an uncapped policy.

        Given:
            A policy with no per-tile entry cap.
        When:
            A tile covering every record is generated.
        Then:
            It should return all of them. ``top_k`` is skipped entirely on this
            path, so it is a separate branch rather than a large cap.
        """
        # Arrange
        tileset = BedTileset(
            make_bed(),
            chromsizes,
            policy=DEFAULT_POLICY.with_(max_entries_per_tile=None),
        )

        # Act
        ((_, records),) = tileset.tiles([tileset.parse_tile_id("x.0.0")])

        # Assert
        assert len(records) == len(RECORDS)

    def test_tiles_should_keep_a_surviving_record_at_the_next_zoom(
        self, make_bed, chromsizes
    ):
        """Test that thinning is coherent across zoom levels.

        Given:
            A cap tight enough to thin both a coarse tile and the finer tile
            nested inside it.
        When:
            Both are generated.
        Then:
            Every record the coarse tile kept from the finer tile's range
            should also survive there. The digest is zoom-independent, so a
            feature does not flicker in and out as the client zooms.
        """
        # Arrange
        tileset = BedTileset(
            make_bed(),
            chromsizes,
            policy=DEFAULT_POLICY.with_(max_entries_per_tile=2),
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

    @pytest.mark.xfail(
        raises=TypeError,
        strict=True,
        reason="oxbow parses a BED end of 0 as null; to_tile_record then "
        "raises TypeError instead of a TileError (#TBD)",
    )
    def test_tiles_should_place_a_record_ending_at_zero(self, make_bed):
        """Test a zero-length record at the start of a contig.

        Given:
            A BED holding the single record ``c1 0 0``, which is a legal
            zero-length feature.
        When:
            The tile covering it is generated.
        Then:
            It should be placed like any other record. It is not: oxbow reads
            the ``end`` column as null when its value is 0 -- ``c1 5 5`` parses
            correctly, so the trigger is the value and not the zero length --
            and ``to_tile_record`` reaches ``offset + end`` on that null. The
            resulting bare ``TypeError`` is exactly the failure the same
            function's contig guard exists to prevent: it is not a
            ``TileError``, so it escapes the server boundary as a 500.
        """
        # Arrange
        tileset = BedTileset(
            make_bed(records=[("c1", 0, 0, "r0")]), genome.canonical()
        )

        # Act & assert
        tileset.tiles([tileset.parse_tile_id("x.0.0")])

    # --- properties ---

    @PROPERTY_IO
    @given(data=st.data())
    def test_tiles_should_only_return_records_the_tile_covers(
        self, make_bed, data
    ):
        """Test that a tile's records lie inside its own span.

        Given:
            Any position in the ladder, over drawn records to which two more
            are added: one ending exactly where the tile begins and one
            beginning exactly where it ends.
        When:
            The tile is generated.
        Then:
            Every record should overlap the tile's span on the genome axis, be
            uniquely identified, and come back in genomic order -- which
            excludes both abutting records, since the span is half-open.

            The two anchored records are what make the boundary reachable.
            Half-open intersection is wrong in one direction at a time, and a
            drawn record lands exactly on a tile edge so rarely that the
            property never sees the case: flipping ``end > start`` to ``>=``
            leaves it green across every example.
        """
        # Arrange
        chromsizes = genome.canonical()
        drawn = data.draw(strategies.bed_records(genome.CANONICAL_CHROMSIZES))
        tileset = BedTileset(make_bed(records=drawn), chromsizes)
        info = tileset.info()
        z = data.draw(st.integers(min_value=0, max_value=info.max_zoom))
        canvas = info.canvas(z)
        x = data.draw(st.integers(min_value=0, max_value=canvas.n_tiles - 1))
        lo, hi = canvas.tile_span(x)
        spans = [gr for gr in canvas.invert(x) if gr.name is not None]
        abutting = []
        if spans:
            head, tail = spans[0], spans[-1]
            if head.start > 0:
                abutting.append((head.name, head.start - 1, head.start, "before"))
            length = dict(genome.CANONICAL_CHROMSIZES)[tail.name]
            if tail.end < length:
                abutting.append((tail.name, tail.end, tail.end + 1, "after"))
        order = {name: i for i, (name, _) in enumerate(genome.CANONICAL_CHROMSIZES)}
        placed = sorted(drawn + abutting, key=lambda r: (order[r[0]], r[1]))
        tileset = BedTileset(make_bed(records=placed), chromsizes)

        # Act
        ((_, records),) = tileset.tiles([tileset.parse_tile_id(f"x.{z}.{x}")])

        # Assert
        assert all(r["xStart"] < hi and r["xEnd"] > lo for r in records)
        assert len({r["uid"] for r in records}) == len(records)
        assert [r["xStart"] for r in records] == sorted(
            r["xStart"] for r in records
        )

    @PROPERTY_IO
    @given(
        cap=st.integers(min_value=1, max_value=10),
        z=st.integers(min_value=0, max_value=2),
        records=strategies.bed_records(genome.CANONICAL_CHROMSIZES),
    )
    def test_tiles_should_never_exceed_the_cap_at_any_zoom(
        self, make_bed, cap, z, records
    ):
        """Test that the density bound holds across the ladder.

        Given:
            Any zoom and any per-tile cap.
        When:
            Every tile at that zoom is generated.
        Then:
            None should exceed the cap, and every record they serve should be
            one the uncapped tileset serves. A bare count bound holds for any
            implementation whatsoever, including one inventing records, so the
            uncapped set is what gives the second half something to fail on.
            An unbounded tile is the failure the cap exists to prevent.
        """
        # Arrange
        path = make_bed(records=records)
        uncapped = BedTileset(
            path,
            genome.canonical(),
            policy=DEFAULT_POLICY.with_(max_entries_per_tile=None),
        )
        every_uid = {
            r["uid"]
            for _, rs in uncapped.tiles(
                [uncapped.parse_tile_id("x.0.0")]
            )
            for r in rs
        }
        tileset = BedTileset(
            path,
            genome.canonical(),
            policy=DEFAULT_POLICY.with_(max_entries_per_tile=cap),
        )
        n_tiles = tileset.info().canvas(z).n_tiles

        # Act
        result = tileset.tiles(
            [tileset.parse_tile_id(f"x.{z}.{x}") for x in range(n_tiles)]
        )

        # Assert
        assert all(len(records) <= cap for _, records in result)
        served = {r["uid"] for _, rs in result for r in rs}
        assert len(served) <= len(records)
        assert served <= every_uid


class TestBedTilesetRegions:
    """The paginated listing, which no other tileset implements."""

    def test_regions_should_return_the_page_in_file_order(
        self, make_bed, chromsizes
    ):
        """Test the listing order.

        Given:
            A BED whose records are written out of genomic order, so file
            order, coordinate order and name order are three different
            sequences.
        When:
            The first page is requested.
        Then:
            It should come back in file order, which is what a listing means --
            there is no coordinate query behind it. The default fixture is
            written in genomic order and so cannot distinguish the three; a
            listing that quietly sorted would pass against it.
        """
        # Arrange
        scrambled = [
            ("c2", 100, 150, "d"),
            ("c1", 30, 40, "b"),
            ("c3", 5, 25, "e"),
            ("c1", 10, 20, "a"),
            ("c2", 0, 50, "c"),
        ]
        tileset = BedTileset(make_bed(records=scrambled), chromsizes)

        # Act
        rows, _ = tileset.regions(0, 10)

        # Assert
        assert [r["fields"][3] for r in rows] == ["d", "b", "e", "a", "c"]

    def test_regions_should_report_more_when_a_page_is_full(self, unindexed):
        """Test paging over the record listing.

        Given:
            A BED holding more records than the page limit.
        When:
            The first page is requested.
        Then:
            It should return exactly the limit and report that more follow.
            The reader takes ``limit + 1`` rows to answer that without a count.
        """
        # Act
        rows, has_next = unindexed.regions(0, 2)

        # Assert
        assert len(rows) == 2
        assert has_next is True

    def test_regions_should_report_no_more_on_the_last_page(self, unindexed):
        """Test the end of the listing.

        Given:
            An offset far enough in that fewer rows remain than the limit.
        When:
            The page is requested.
        Then:
            It should return the remainder and report that nothing follows.
        """
        # Act
        rows, has_next = unindexed.regions(3, 5)

        # Assert
        assert len(rows) == 2
        assert has_next is False

    def test_regions_should_omit_the_importance_field(self, unindexed):
        """Test the listing's wire shape.

        Given:
            A page of regions.
        When:
            A row is inspected.
        Then:
            It should carry the RegionRow keys only. Importance ranks records
            for thinning; a listing does not thin, so reporting it would
            advertise a ranking that means nothing here.
        """
        # Act
        (row, *_), _ = unindexed.regions(0, 1)

        # Assert
        assert sorted(row) == ["chrOffset", "fields", "uid", "xEnd", "xStart"]

    def test_regions_should_skip_records_on_an_unknown_contig(
        self, make_bed, chromsizes
    ):
        """Test the paginated listing against an unknown contig.

        Given:
            A BED with a contig absent from the supplied chromsizes.
        When:
            A page of regions is requested.
        Then:
            It should return the placeable rows rather than raising KeyError,
            which is not a TileError and would surface as a 500.
        """
        # Arrange
        path = make_bed(
            records=[("cUNKNOWN", 0, 10, "z")] + RECORDS, name="extra.bed"
        )
        tileset = BedTileset(path, chromsizes)

        # Act
        rows, has_next = tileset.regions(0, 20)

        # Assert
        assert len(rows) == len(RECORDS)
        assert has_next is False


@pytest.mark.pinned
class TestBedTilesetBounds:
    """What happens off the end of the ladder.

    Pinned, not asserted as desirable. Only cooler has an established skip
    contract; the bedlike types raise until the boundary layer decides how an
    out-of-range id is surfaced.
    """

    def test_tiles_should_raise_when_the_zoom_is_past_the_ladder(
        self, unindexed
    ):
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
            unindexed.tiles([unindexed.parse_tile_id("x.3.0")])

    def test_tiles_should_raise_when_the_position_is_past_the_canvas(
        self, unindexed
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
            unindexed.tiles([unindexed.parse_tile_id("x.1.4")])
