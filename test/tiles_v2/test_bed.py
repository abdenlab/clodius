"""Tests for clodius.tiles_v2.bed.

The only tileset with two query paths, and the only one implementing
``ProvidesRegions``. With a tabix index a tile reads just its own byte ranges;
without one every tile scans the whole file behind a pushed-down polars
predicate. The two must agree on every tile, because otherwise compressing a
file changes what it serves -- and they did not, until the out-of-bounds
early return landed: oxbow reads an empty region list as *no filter at all*
and scans everything.

A BED file carries no chromosome lengths, so the coordinate system comes from
outside it and the two can disagree. A record on a contig the chromsizes do
not name has no position on the genome-spanning axis, so the only question is
whether it is skipped or takes the request down with it.

The genome here totals 3000 bp deliberately. ``TILE_SIZE`` is 1024, so the
quadtree extent is 4096 and the last tile at max zoom spans [3072, 4096) --
entirely past the end of the genome. That is the tile the two paths used to
disagree about, and the only window that produces one is 2048 < total <= 3072.
Rounding this genome up to 4096 would silently delete that coverage.

Every fixture is synthesized into a temp directory, so the module runs on a
checkout with no git-LFS payload.
"""

import pysam
import pytest

from clodius.core.coords import Chromsizes
from clodius.core.policies import TilePolicy
from clodius.tiles_v2.bed import BedTileset, to_tile_record

CHROMSIZES = Chromsizes.from_pairs([("c1", 1000), ("c2", 1500), ("c3", 500)])

#: Placeable records, in file order -- which is deliberately neither genomic
#: order nor name order, so a listing that sorts is distinguishable from one
#: that does not.
RECORDS = [
    ("c2", 900, 950, "d"),
    ("c1", 10, 20, "a"),
    ("c3", 5, 15, "e"),
    ("c1", 300, 400, "b"),
    ("c2", 50, 60, "c"),
]

#: A contig the chromsizes above do not name.
UNPLACEABLE = ("cUNKNOWN", 0, 10, "z")


def write_bed(path, records):
    """Write records to a plain BED and return its path."""
    with open(path, "w") as fh:
        for record in records:
            fh.write("\t".join(str(field) for field in record) + "\n")
    return str(path)


@pytest.fixture
def make_bed(tmp_path):
    """Build a plain BED, so every tile scans the file."""

    def build(records=RECORDS, name="records.bed"):
        return write_bed(tmp_path / name, records)

    return build


@pytest.fixture
def make_bed_bgzf(tmp_path):
    """Build a BGZF-compressed BED with a tabix index, so tiles read ranges."""

    def build(records=RECORDS, name="records.bed"):
        plain = write_bed(tmp_path / name, sorted(records))
        gz = plain + ".gz"
        pysam.tabix_compress(plain, gz, force=True)
        pysam.tabix_index(gz, preset="bed", force=True)
        return gz

    return build


def names(records):
    """The fourth BED column of each record, which is its label here."""
    return [r["fields"][3] for r in records]


def test_to_tile_record_should_return_the_record():
    """Test the ordinary row, so the guard cannot pass by skipping all.

    Given:
        A row on a contig the chromsizes name.
    When:
        It is converted.
    Then:
        It should carry the record's position offset onto the
        genome-spanning axis.
    """
    # Arrange
    row = {"chrom": "c2", "start": 50, "end": 60, "rest": "c", "_digest": 1}

    # Act
    record = to_tile_record(row, CHROMSIZES.offsets)

    # Assert
    assert (record["chrOffset"], record["xStart"]) == (1000, 1050)


def test_to_tile_record_should_return_none_when_the_contig_is_unknown():
    """Test a row the coordinate system cannot place.

    Given:
        A row on a contig absent from the offsets.
    When:
        It is converted.
    Then:
        It should return None rather than raising ``KeyError``, which is not a
        ``TileError`` and so would escape the server boundary as a 500.
    """
    # Arrange
    row = {
        "chrom": "cUNKNOWN",
        "start": 0,
        "end": 10,
        "rest": "z",
        "_digest": 1,
    }

    # Act
    record = to_tile_record(row, CHROMSIZES.offsets)

    # Assert
    assert record is None


def test_tiles_should_agree_between_the_indexed_and_scanning_paths(
    make_bed, make_bed_bgzf
):
    """Test the two query paths against each other across the whole ladder.

    Given:
        The same records written as a plain BED and as a BGZF+tabix BED.
    When:
        Every tile in the ladder is requested from each.
    Then:
        The record sets should match tile for tile. Compressing a file must
        not change what it serves, and it did: for a tile past the end of the
        genome the indexed path handed oxbow an empty region list, which it
        reads as *no filter*, and served the entire file.
    """
    # Arrange
    scanning = BedTileset(make_bed(), CHROMSIZES)
    indexed = BedTileset(make_bed_bgzf(), CHROMSIZES)
    info = scanning.info()

    # Act & assert
    for z in range(info.max_zoom + 1):
        for x in range(info.canvas(z).n_tiles):
            tile_id = f"u.{z}.{x}"
            (_, left), = scanning.tiles([scanning.parse_tile_id(tile_id)])
            (_, right), = indexed.tiles([indexed.parse_tile_id(tile_id)])
            assert sorted(names(left)) == sorted(names(right)), tile_id


@pytest.mark.parametrize("indexed", [False, True])
def test_tiles_should_be_empty_past_the_end_of_the_genome(
    make_bed, make_bed_bgzf, indexed
):
    """Test the tile that lies entirely beyond the last chromosome.

    Given:
        A tileset whose quadtree extent exceeds its genome, which is every
        tileset, and the last tile at max zoom.
    When:
        It is requested.
    Then:
        It should hold no records. This is not an exotic position: the extent
        always overshoots, so a client walking the top zoom asks for it every
        time.
    """
    # Arrange
    build = make_bed_bgzf if indexed else make_bed
    tileset = BedTileset(build(), CHROMSIZES)
    info = tileset.info()
    last = info.canvas(info.max_zoom).n_tiles - 1

    # Act
    (_, records), = tileset.tiles(
        [tileset.parse_tile_id(f"u.{info.max_zoom}.{last}")]
    )

    # Assert
    assert records == []


def test_tiles_should_return_nothing_when_the_cap_is_zero(make_bed):
    """Test the record cap on the path that never calls the shared helper.

    Given:
        A policy capping records at zero.
    When:
        The whole-genome tile is requested.
    Then:
        It should return nothing. This path caps by pushing ``top_k`` into
        polars rather than by calling ``take_most_important``, so the guard
        restored there does not cover it -- testing ``cap is not None`` rather
        than the truthiness of the cap is what keeps zero from meaning
        unlimited.
    """
    # Arrange
    tileset = BedTileset(
        make_bed(), CHROMSIZES, policy=TilePolicy(max_records=0)
    )

    # Act
    (_, records), = tileset.tiles([tileset.parse_tile_id("u.0.0")])

    # Assert
    assert records == []


def test_regions_should_return_the_records_in_file_order(make_bed):
    """Test that a listing is a listing, not a query.

    Given:
        A BED whose file order is neither genomic nor alphabetical.
    When:
        A page of regions is requested.
    Then:
        It should come back in file order. ``regions`` pages through a file
        for a client that wants to enumerate it; sorting would make the page
        boundaries depend on content and silently reorder a listing the client
        is walking by offset.
    """
    # Arrange
    tileset = BedTileset(make_bed(), CHROMSIZES)

    # Act
    rows, has_next = tileset.regions(0, 20)

    # Assert
    assert names(rows) == [r[3] for r in RECORDS]
    assert has_next is False


def test_regions_should_skip_a_record_on_an_unknown_contig(make_bed):
    """Test the paginated listing against an unknown contig.

    Given:
        A BED carrying a contig absent from the chromsizes.
    When:
        A page of regions is requested.
    Then:
        It should return the placeable rows rather than raising. ``regions``
        queries no coordinate range, so the filter the tile path applies
        before converting a row never runs here -- which is why closing this
        inside ``to_tile_record`` is what fixes both paths.
    """
    # Arrange
    path = make_bed([UNPLACEABLE] + RECORDS, name="extra.bed")
    tileset = BedTileset(path, CHROMSIZES)

    # Act
    rows, has_next = tileset.regions(0, 20)

    # Assert
    assert len(rows) == len(RECORDS)
    assert all(r["fields"][0] != "cUNKNOWN" for r in rows)


def test_tiles_should_not_let_an_unknown_contig_consume_a_cap_slot(make_bed):
    """Test the record cap against rows that cannot be placed.

    Given:
        A BED where unplaceable records outnumber the placeable ones, served
        under a cap smaller than the placeable count.
    When:
        The whole-genome tile is requested.
    Then:
        It should return a full tile of placeable records. The cap ranks rows
        by digest, so an unplaceable row that survives the ranking and is
        dropped afterwards silently shortens the tile -- the filter has to run
        before the cap, not after it.
    """
    # Arrange
    noise = [("cUNKNOWN", i * 10, i * 10 + 5, f"z{i}") for i in range(50)]
    path = make_bed(noise + RECORDS, name="noisy.bed")
    tileset = BedTileset(
        path, CHROMSIZES, policy=TilePolicy(max_records=3)
    )

    # Act
    (_, records), = tileset.tiles([tileset.parse_tile_id("u.0.0")])

    # Assert
    assert len(records) == 3
    assert all(r["fields"][0] != "cUNKNOWN" for r in records)
