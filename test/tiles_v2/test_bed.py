"""Tests for clodius.tiles_v2.bed.

Every fixture here is synthesized into a temp directory, so the whole module
runs without any git-LFS payload.
"""

import pytest

from clodius.core.coords import Chromsizes
from clodius.tiles_v2.bed import BedTileset

# Long enough that the quadtree has more than one level, so there is a tile
# position past the end of the genome to ask for. Total 3000 bp gives
# max_zoom 2 and a 4096 bp canvas, whose last tile at that zoom starts at 3072.
TINY = [["c1", 1000], ["c2", 1500], ["c3", 500]]

RECORDS = [
    ("c1", 10, 20, "a"),
    ("c1", 30, 40, "b"),
    ("c2", 0, 50, "c"),
    ("c2", 100, 150, "d"),
    ("c3", 5, 25, "e"),
]


def write_bed(path, records=RECORDS):
    """A plain uncompressed BED holding ``records``."""
    path.write_text(
        "".join(f"{c}\t{s}\t{e}\t{name}\n" for c, s, e, name in records)
    )
    return path


def write_indexed_bed(path, records=RECORDS):
    """A BGZF-compressed BED with a sibling tabix index."""
    pysam = pytest.importorskip("pysam")
    plain = path.parent / "plain.bed"
    write_bed(plain, records)
    pysam.tabix_compress(str(plain), str(path), force=True)
    pysam.tabix_index(str(path), preset="bed", force=True)
    return path


@pytest.fixture
def chromsizes():
    """The three-chromosome coordinate system the fixtures are placed on."""
    return Chromsizes.from_pairs(TINY)


@pytest.fixture
def unindexed(tmp_path, chromsizes):
    """A BedTileset over a plain BED, so every tile scans the file."""
    return BedTileset(write_bed(tmp_path / "tiny.bed"), chromsizes)


@pytest.fixture
def indexed(tmp_path, chromsizes):
    """A BedTileset over a BGZF+tabix BED, so tiles read byte ranges."""
    return BedTileset(write_indexed_bed(tmp_path / "tiny.bed.gz"), chromsizes)


def past_genome_tile(tileset):
    """A tile position at the far end of the canvas, past the last contig."""
    info = tileset.info()
    canvas = info.canvas(info.max_zoom)
    x = canvas.n_tiles - 1
    total = sum(length for _, length in info.chromsizes)
    assert canvas.tile_span(x)[0] >= total, (
        "fixture genome fits inside the canvas' last tile, so there is no "
        "past-genome position to test"
    )
    return x


class TestBedTileset:
    """Behavior of the BED tileset."""

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
            It should report itself indexed.
        """
        # Act & assert
        assert indexed.is_indexed is True

    def test_tiles_should_return_the_overlapping_records(self, unindexed):
        """Test the ordinary in-genome case.

        Given:
            A tile covering the whole genome at the coarsest zoom.
        When:
            It is generated.
        Then:
            It should return every record, in genomic order.
        """
        # Arrange
        tid = unindexed.parse_tile_id("x.0.0")

        # Act
        ((_, records),) = unindexed.tiles([tid])

        # Assert
        assert [r["fields"][0] for r in records] == [
            "c1",
            "c1",
            "c2",
            "c2",
            "c3",
        ]

    @pytest.mark.parametrize("fixture", ["unindexed", "indexed"])
    def test_tiles_should_return_nothing_when_the_tile_is_past_the_genome(
        self, fixture, request
    ):
        """Test the tile positions HiGlass requests routinely at every zoom.

        Given:
            A tile lying entirely past the end of the genome, which every
            zoom level has because ``max_width`` always exceeds the genome
            length.
        When:
            It is generated, on both the indexed and unindexed paths.
        Then:
            It should return nothing. oxbow reads an empty region list as *no
            region filter* and scans the whole file, so the indexed path used
            to return every record in it.
        """
        # Arrange
        tileset = request.getfixturevalue(fixture)
        x = past_genome_tile(tileset)
        tid = tileset.parse_tile_id(f"x.{tileset.info().max_zoom}.{x}")

        # Act
        ((_, records),) = tileset.tiles([tid])

        # Assert
        assert records == []

    def test_tiles_should_skip_records_on_a_contig_not_in_the_chromsizes(
        self, tmp_path, chromsizes
    ):
        """Test a BED carrying a contig the coordinate system does not know.

        Given:
            A BED with an unplaced contig absent from the supplied
            chromsizes.
        When:
            a tile covering the genome is generated.
        Then:
            It should return only the placeable records rather than raising
            KeyError out of the request handler.
        """
        # Arrange
        path = write_bed(
            tmp_path / "extra.bed", RECORDS + [("cUNKNOWN", 0, 10, "z")]
        )
        tileset = BedTileset(path, chromsizes)
        tid = tileset.parse_tile_id("x.0.0")

        # Act
        ((_, records),) = tileset.tiles([tid])

        # Assert
        assert len(records) == len(RECORDS)
        assert all(r["fields"][0] != "cUNKNOWN" for r in records)

    def test_regions_should_skip_records_on_an_unknown_contig(
        self, tmp_path, chromsizes
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
        path = write_bed(
            tmp_path / "extra.bed", [("cUNKNOWN", 0, 10, "z")] + RECORDS
        )
        tileset = BedTileset(path, chromsizes)

        # Act
        rows, has_next = tileset.regions(0, 20)

        # Assert
        assert len(rows) == len(RECORDS)
        assert has_next is False

    def test_regions_should_report_more_when_a_page_is_full(self, unindexed):
        """Test paging over the record listing.

        Given:
            A BED holding more records than the page limit.
        When:
            The first page is requested.
        Then:
            It should return the page and report that more follow.
        """
        # Act
        rows, has_next = unindexed.regions(0, 2)

        # Assert
        assert len(rows) == 2
        assert has_next is True

    def test_tiles_should_cap_records_at_the_policy_limit(
        self, tmp_path, chromsizes
    ):
        """Test the density cap.

        Given:
            A tileset whose policy caps entries per tile below the record
            count.
        When:
            A tile covering every record is generated.
        Then:
            It should return no more than the cap.
        """
        # Arrange
        from clodius.core.policies import DEFAULT_POLICY

        tileset = BedTileset(
            write_bed(tmp_path / "tiny.bed"),
            chromsizes,
            policy=DEFAULT_POLICY.with_(max_entries_per_tile=2),
        )
        tid = tileset.parse_tile_id("x.0.0")

        # Act
        ((_, records),) = tileset.tiles([tid])

        # Assert
        assert len(records) == 2
