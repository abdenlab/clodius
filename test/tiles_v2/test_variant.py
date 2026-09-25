"""Tests for clodius.tiles_v2.variant.

A VCF names its contigs in the header, so the tileset's coordinate system
usually comes from the file itself -- but it can be given one instead, to
serve against a different assembly or to cover a header with no ``##contig``
lines. Then the two can disagree, and a record on a contig the chromsizes do
not name has no position on the genome-spanning axis.

The rule is the same one the sibling BED and BEDPE converters follow: the
converter drops the record and says so in its return type, rather than the
call sites each remembering to test membership first. A bare ``KeyError`` is
not a ``TileError``, so it escapes the server boundary as a 500 -- and a rule
held at three call sites is a rule a fourth caller reintroduces.

``regions`` pages through the file for a client enumerating it, and there the
filter has to run *before* the slice: the next-page probe and the offset both
count rows, so filtering afterwards makes a full page look short and makes
consecutive pages overlap.

Fixtures are synthesized into a temp directory, so the module runs on a
checkout with no git-LFS payload.
"""

import pysam
import pytest

from clodius.core.coords import Chromsizes
from clodius.core.policies import TilePolicy
from clodius.tiles_v2.variant import VcfTileset, to_bedlike

#: The chromsizes the tileset is served against, which name only ``c1``.
CHROMSIZES = Chromsizes.from_pairs([("c1", 1000)])

#: Header contigs, which deliberately include one the chromsizes omit.
HEADER_CONTIGS = [("c1", 1000), ("cUNKNOWN", 500)]

#: Ten placeable variants with an unplaceable one wedged in at index 2.
VARIANTS = (
    [("c1", 10 + i * 10, f"v{i}") for i in range(2)]
    + [("cUNKNOWN", 5, "z")]
    + [("c1", 10 + i * 10, f"v{i}") for i in range(2, 10)]
)


def write_vcf(path, variants, contigs=None):
    """Write a minimal VCF carrying ``variants`` and return its path."""
    with open(path, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        for name, length in contigs or HEADER_CONTIGS:
            fh.write(f"##contig=<ID={name},length={length}>\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for chrom, pos, name in variants:
            fh.write(f"{chrom}\t{pos}\t{name}\tA\tT\t.\tPASS\t.\n")
    return str(path)


@pytest.fixture
def make_vcf(tmp_path):
    """Build a plain VCF, so every tile scans the file."""

    def build(variants=VARIANTS, name="variants.vcf", contigs=None):
        return write_vcf(tmp_path / name, variants, contigs)

    return build


@pytest.fixture
def make_indexed_vcf(tmp_path):
    """Build a BGZF-compressed, tabix-indexed VCF and return its path."""

    def build(variants=VARIANTS, name="variants.vcf", contigs=None):
        plain = write_vcf(tmp_path / name, variants, contigs)
        gz = plain + ".gz"
        pysam.tabix_compress(plain, gz, force=True)
        pysam.tabix_index(gz, preset="vcf", force=True)
        return gz

    return build


def names(records):
    """The ID column of each served record."""
    return [r["fields"][2] for r in records]


class TestToBedlike:
    """The row-to-record converter, and the contig it cannot place."""

    def test_to_bedlike_should_return_the_record(self):
        """Test the ordinary row, so the guard cannot pass by skipping all.

        Given:
            A row whose contig the offsets carry.
        When:
            It is converted.
        Then:
            It should place the record through that contig's offset.
        """
        # Arrange
        row = {
            "chrom": "c1",
            "pos": 11,
            "id": ["v0"],
            "ref": "A",
            "alt": ["T"],
            "filter": ["PASS"],
            "qual": None,
            "_start": 10,
            "_end": 11,
            "_digest": 7,
        }

        # Act
        record = to_bedlike(row, CHROMSIZES.offsets)

        # Assert
        assert record["xStart"] == CHROMSIZES.offsets["c1"] + 10

    def test_to_bedlike_should_return_none_when_the_contig_is_unknown(self):
        """Test the converter against a contig outside the coordinate system.

        Given:
            A row naming a contig the offsets do not carry.
        When:
            It is converted.
        Then:
            It should return ``None`` rather than raising. A bare ``KeyError``
            is not a ``TileError``, so it would escape the server boundary as
            a 500 -- and holding the rule here is what stops a later call site
            from reintroducing it.
        """
        # Arrange
        row = {
            "chrom": "cUNKNOWN",
            "pos": 6,
            "id": ["z"],
            "ref": "A",
            "alt": ["T"],
            "filter": ["PASS"],
            "qual": None,
            "_start": 5,
            "_end": 6,
            "_digest": 7,
        }

        # Act
        record = to_bedlike(row, CHROMSIZES.offsets)

        # Assert
        assert record is None


class TestVariantTileset:
    """Serving and paging a VCF whose header names more contigs than the
    coordinate system does."""

    def test_tiles_should_skip_a_variant_on_an_unknown_contig(self, make_vcf):
        """Test the tile path against a contig outside the chromsizes.

        Given:
            A VCF carrying a variant on a contig the chromsizes omit.
        When:
            The whole-genome tile is served.
        Then:
            It should return the placeable variants and omit the other, rather
            than failing the tile.
        """
        # Arrange
        tileset = VcfTileset(make_vcf(), chromsizes=CHROMSIZES)

        # Act
        (_, records), = tileset.tiles([tileset.parse_tile_id("u.0.0")])

        # Assert
        assert "z" not in names(records)
        assert records

    def test_tiles_should_serve_an_indexed_file_missing_a_contig(
        self, make_vcf, make_indexed_vcf
    ):
        """Test an index that does not name every contig in the chromsizes.

        Given:
            A VCF declaring two contigs but carrying variants on ``c1`` only,
            written both plain and as BGZF+tabix, served against chromsizes
            naming both.
        When:
            The whole-genome tile is served from each.
        Then:
            Both should return the same variants. Query regions are derived
            from the chromsizes, so the indexed path asked the index for a
            contig it does not carry; the reader raised a ``ComputeError``,
            which is not a ``TileError``, so one ordinary tile failed the
            entire batch.
        """
        # Arrange. The index lists only contigs it saw records on, so `c2`
        # must be in the chromsizes and absent from the data for the tile to
        # ask for something the index cannot answer.
        two_contigs = [("c1", 1000), ("c2", 1500)]
        both = Chromsizes.from_pairs(two_contigs)
        c1_only = [v for v in VARIANTS if v[0] == "c1"]
        indexed = VcfTileset(
            make_indexed_vcf(c1_only, name="c1only.vcf", contigs=two_contigs),
            chromsizes=both,
        )
        scanning = VcfTileset(
            make_vcf(c1_only, name="c1plain.vcf", contigs=two_contigs),
            chromsizes=both,
        )

        # Act
        (_, from_index), = indexed.tiles([indexed.parse_tile_id("u.0.0")])
        (_, from_scan), = scanning.tiles([scanning.parse_tile_id("u.0.0")])

        # Assert
        assert names(from_index) == names(from_scan)
        assert from_index

    @pytest.mark.parametrize("cap,expected", [(0, 0), (3, 3), (None, 10)])
    def test_tiles_should_return_at_most_the_capped_number_of_records(
        self, make_vcf, cap, expected
    ):
        """Test the record cap, which a VCF tile is subject to like any other.

        Given:
            A VCF carrying ten placeable variants, under a policy capping
            records at zero, at three, and at no limit.
        When:
            The whole-genome tile is served.
        Then:
            It should return no more than the cap. The tile read every variant
            at every setting, so a server configured to serve none emitted an
            unbounded tile -- the precise failure the cap exists to prevent.
        """
        # Arrange
        tileset = VcfTileset(
            make_vcf(),
            chromsizes=CHROMSIZES,
            policy=TilePolicy(max_records=cap),
        )

        # Act
        (_, records), = tileset.tiles([tileset.parse_tile_id("u.0.0")])

        # Assert
        assert len(records) == expected

    def test_regions_should_report_a_next_page_when_one_remains(
        self, make_vcf
    ):
        """Test the next-page probe against a window holding an unplaceable
        row.

        Given:
            A VCF of ten placeable variants with an unplaceable one at index
            2, and a page of five requested from the start.
        When:
            The page is listed.
        Then:
            It should report that more follow. The probe reads ``limit + 1``
            rows; counting it after the unplaceable row is dropped makes a full
            page look short, and a client paging on ``has_next`` stops with
            variants it was never shown.
        """
        # Arrange
        tileset = VcfTileset(make_vcf(), chromsizes=CHROMSIZES)

        # Act
        rows, has_next = tileset.regions(0, 5)

        # Assert
        assert len(rows) == 5
        assert has_next is True

    def test_regions_should_not_repeat_a_variant_across_consecutive_pages(
        self, make_vcf
    ):
        """Test that the offset and the page range over the same population.

        Given:
            The same VCF, listed as two consecutive pages of five.
        When:
            Both pages are listed.
        Then:
            They should share no variant. ``offset`` indexes rows in the file;
            filtering after the slice makes the page a subset of that window,
            so the next offset lands short and the pages overlap.
        """
        # Arrange
        tileset = VcfTileset(make_vcf(), chromsizes=CHROMSIZES)

        # Act
        first, _ = tileset.regions(0, 5)
        second, _ = tileset.regions(5, 5)

        # Assert
        assert len(first) == 5 and len(second) == 5
        assert set(names(first)) & set(names(second)) == set()
