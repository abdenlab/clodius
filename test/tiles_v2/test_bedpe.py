"""Tests for clodius.tiles_v2.bedpe.

Two tilesets over one file: rectangles, where anchor 1 must overlap the x
range and anchor 2 the y range, and links, where a policy decides which of the
two anchors has to be in view. Both share the reading path, and that path has
a shape the sibling BED module does not: when a tile's ranges are all past the
end of the genome, an empty seek list is indistinguishable from *no* seek at
all, so the reader is asked to scan the whole file rather than to read
nothing.

A quadtree extent always exceeds the genome, so the off-the-end tile is not an
edge case a client has to go looking for -- it is the last tile of every zoom.
And scanning is not merely slow here: the size ceiling that guards it raises
``TilesetUnavailable``, which is a sibling of ``TileError`` rather than a
subclass, so the per-tile boundary does not catch it and one routine tile
fails the whole request.

The fixtures are synthesized into a temp directory, so the module runs on a
checkout with no git-LFS payload.
"""

import os

import pysam
import pytest

from clodius.core.coords import Chromsizes
from clodius.core.errors import TilesetUnavailable
from clodius.core.policies import LinkPolicy
from clodius.core.policies import TilePolicy
from clodius.tiles_v2.bedpe import (
    BedpeLinksTileset,
    BedpeTileset,
    to_paired,
)

CHROMSIZES = Chromsizes.from_pairs([("c1", 1000), ("c2", 1500)])

TILE_SIZE = 4

#: Four intra-chromosomal pairs on c1, spread across the first 400 bp.
RECORDS = [
    ("c1", 10, 20, "c1", 100, 110, "r0"),
    ("c1", 30, 40, "c1", 200, 210, "r1"),
    ("c1", 50, 60, "c1", 300, 310, "r2"),
    ("c1", 70, 80, "c1", 400, 410, "r3"),
]

#: Pairs whose first anchor names a contig the chromsizes do not. Twenty of
#: them against four placeable records, so a cap applied before the contig
#: filter cannot leave the placeable set intact by luck.
UNPLACEABLE = [
    ("cUNKNOWN", i * 10, i * 10 + 10, "c1", 100, 110, f"z{i}")
    for i in range(20)
]


def write_bedpe(path, records):
    """Write records to a plain BEDPE and return its path."""
    with open(path, "w") as fh:
        for record in records:
            fh.write("\t".join(str(field) for field in record) + "\n")
    return str(path)


@pytest.fixture
def make_bedpe(tmp_path):
    """Build a plain BEDPE, so every tile scans the file."""

    def build(records=RECORDS, name="pairs.bedpe"):
        return write_bedpe(tmp_path / name, records)

    return build


@pytest.fixture
def make_indexed_bedpe(tmp_path):
    """Build a BGZF-compressed, tabix-indexed BEDPE and return its path."""

    def build(records=RECORDS, name="pairs.bedpe"):
        plain = write_bedpe(tmp_path / name, records)
        gz = plain + ".gz"
        pysam.tabix_compress(plain, gz, force=True)
        pysam.tabix_index(gz, preset="bed", force=True)
        return gz

    return build


#: A single contig, so a zoom-3 tile spans 128 bp and anchors can be placed
#: in known tiles by hand. At tile 1, [128, 256): ``near`` has both anchors
#: inside, ``straddle`` one, ``spanner`` neither but a hull that crosses.
POLICY_CHROMSIZES = Chromsizes.from_pairs([("c1", 1000)])

POLICY_RECORDS = [
    ("c1", 10, 20, "c1", 180, 190, "straddle"),
    ("c1", 10, 20, "c1", 300, 310, "spanner"),
    ("c1", 140, 150, "c1", 200, 210, "near"),
]


def names(records):
    """The trailing label column of each served record."""
    return [r["fields"][6] for r in records]


class TestToPaired:
    """The row-to-record converter, and the contig it cannot place."""

    def test_to_paired_should_return_the_record(self):
        """Test the ordinary row, so the guard cannot pass by skipping all.

        Given:
            A row whose two anchors both name contigs the offsets carry.
        When:
            It is converted.
        Then:
            It should place both anchors through their own contig offsets.
        """
        # Arrange
        row = {
            "chrom": "c2",
            "start": 10,
            "end": 20,
            "chrom2": "c1",
            "start2": 30,
            "end2": 40,
            "rest2": "",
            "_digest": 7,
        }
        offsets = CHROMSIZES.offsets

        # Act
        record = to_paired(row, offsets)

        # Assert
        assert record["xStart"] == offsets["c2"] + 10
        assert record["yStart"] == offsets["c1"] + 30

    @pytest.mark.parametrize("column", ["chrom", "chrom2"])
    def test_to_paired_should_return_none_when_an_anchor_is_unplaceable(
        self, column
    ):
        """Test the converter against a contig on either side of the pair.

        Given:
            A row naming a contig the offsets do not carry, on one anchor.
        When:
            It is converted.
        Then:
            It should return ``None`` rather than raising. A bare ``KeyError``
            is not a ``TileError``, so it escapes the server boundary as a 500
            -- and holding the rule in the converter is what keeps a third
            caller from reintroducing it.
        """
        # Arrange
        row = {
            "chrom": "c1",
            "start": 10,
            "end": 20,
            "chrom2": "c1",
            "start2": 30,
            "end2": 40,
            "rest2": "",
            "_digest": 7,
        }
        row[column] = "cUNKNOWN"

        # Act
        record = to_paired(row, CHROMSIZES.offsets)

        # Assert
        assert record is None


class TestBedpeTileset:
    """Rectangles, where both anchors are constrained by the tile."""

    def test_tiles_should_return_the_pairs_in_view(self, make_bedpe):
        """Test the ordinary tile, so the screens cannot pass by emptying all.

        Given:
            A BEDPE of four intra-chromosomal pairs, and the tile covering the
            whole genome on both axes.
        When:
            It is served.
        Then:
            It should return every pair.
        """
        # Arrange
        tileset = BedpeTileset(
            make_bedpe(), CHROMSIZES, tile_size=TILE_SIZE
        )

        # Act
        (_, records), = tileset.tiles([tileset.parse_tile_id("u.0.0.0")])

        # Assert
        assert sorted(names(records)) == ["r0", "r1", "r2", "r3"]

    def test_tiles_should_be_empty_past_the_end_of_the_genome(
        self, make_bedpe
    ):
        """Test the last tile of a zoom, which every quadtree canvas has.

        Given:
            The final tile position at the deepest zoom, whose ranges lie
            entirely past the end of the genome.
        When:
            It is served.
        Then:
            It should return no records, without consulting the file. An empty
            seek list reads to the reader as "no restriction", which scans
            everything to match nothing -- and trips the unindexed-size
            ceiling on any file large enough to have one.
        """
        # Arrange
        tileset = BedpeTileset(
            make_bedpe(), CHROMSIZES, tile_size=TILE_SIZE
        )
        canvas = tileset.info().canvas(tileset.info().max_zoom)
        last = canvas.n_tiles - 1

        # Act
        (_, records), = tileset.tiles(
            [tileset.parse_tile_id(f"u.{tileset.info().max_zoom}.{last}.{last}")]
        )

        # Assert
        assert records == []

    def test_tiles_should_not_scan_a_file_above_the_ceiling_for_an_off_the_end_tile(
        self, make_bedpe
    ):
        """Test that the off-the-end tile does not fail the whole request.

        Given:
            An unindexed BEDPE under a policy whose scan ceiling the file
            exceeds, and the final tile of the deepest zoom.
        When:
            It is served.
        Then:
            It should return no records rather than raising.
            ``TilesetUnavailable`` is a sibling of ``TileError``, not a
            subclass, so the per-tile boundary does not convert it -- one
            routine tile would take every other tile in the batch with it.
        """
        # Arrange
        tileset = BedpeTileset(
            make_bedpe(),
            CHROMSIZES,
            tile_size=TILE_SIZE,
            policy=TilePolicy(max_scan_bytes=1),
        )
        z = tileset.info().max_zoom
        last = tileset.info().canvas(z).n_tiles - 1

        # Act
        (_, records), = tileset.tiles(
            [tileset.parse_tile_id(f"u.{z}.{last}.{last}")]
        )

        # Assert
        assert records == []

class TestBedpeLinksTileset:
    """Links, where the tile constrains a single axis."""

    def test_tiles_should_be_empty_past_the_end_of_the_genome(
        self, make_bedpe
    ):
        """Test the one-dimensional screen, which reads one axis not two.

        Given:
            A links tileset and the final tile position at the deepest zoom.
        When:
            It is served.
        Then:
            It should return no records. The rectangles tileset screens two
            coordinate slots and this one screens a single slot, so a screen
            written against a fixed arity would miss one of them.
        """
        # Arrange
        tileset = BedpeLinksTileset(
            make_bedpe(), CHROMSIZES, tile_size=TILE_SIZE
        )
        z = tileset.info().max_zoom
        last = tileset.info().canvas(z).n_tiles - 1

        # Act
        (_, records), = tileset.tiles(
            [tileset.parse_tile_id(f"u.{z}.{last}")]
        )

        # Assert
        assert records == []

    @pytest.mark.parametrize(
        "policy,expected",
        [
            (LinkPolicy.BOTH, ["near"]),
            (LinkPolicy.EITHER, ["near", "straddle"]),
            (LinkPolicy.HULL, ["near", "spanner", "straddle"]),
        ],
        ids=["both", "either", "hull"],
    )
    def test_tiles_should_select_the_links_the_policy_asks_for(
        self, make_bedpe, policy, expected
    ):
        """Test the three link policies against a tile that separates them.

        Given:
            A BEDPE holding one pair with both anchors in the tile, one with a
            single anchor in it, and one with neither but a hull that crosses
            it, served under each policy.
        When:
            That tile is served.
        Then:
            Each policy should return a different set. The three differ only
            in which anchors have to be in view, so a tile where all three
            agree -- which is most tiles -- cannot tell them apart, and only
            ``BOTH`` had any coverage before.
        """
        # Arrange
        tileset = BedpeLinksTileset(
            make_bedpe(POLICY_RECORDS, name="policies.bedpe"),
            POLICY_CHROMSIZES,
            link_policy=policy,
            tile_size=TILE_SIZE,
        )

        # Act
        (_, payload), = tileset.tiles([tileset.parse_tile_id("u.3.1")])

        # Assert
        assert sorted(names(payload)) == expected

    def test_tiles_should_serve_an_indexed_file_missing_a_contig(
        self, make_bedpe, make_indexed_bedpe
    ):
        """Test an index that does not carry every contig the tile asks for.

        Given:
            A BEDPE whose pairs all sit on one contig, served against
            chromsizes naming two, written both plain and as BGZF+tabix.
        When:
            The whole-genome tile is served from each.
        Then:
            Both should return the same links. The seek list is built from the
            chromsizes, so the indexed path named a contig the index does not
            carry and the reader raised past the per-tile boundary.
        """
        # Arrange
        indexed = BedpeLinksTileset(
            make_indexed_bedpe(RECORDS, name="c1only.bedpe"),
            CHROMSIZES,
            link_policy=LinkPolicy.BOTH,
            tile_size=TILE_SIZE,
        )
        scanning = BedpeLinksTileset(
            make_bedpe(RECORDS, name="c1plain.bedpe"),
            CHROMSIZES,
            link_policy=LinkPolicy.BOTH,
            tile_size=TILE_SIZE,
        )

        # Act
        (_, from_index), = indexed.tiles([indexed.parse_tile_id("u.0.0")])
        (_, from_scan), = scanning.tiles([scanning.parse_tile_id("u.0.0")])

        # Assert
        assert names(from_index) == names(from_scan)
        assert from_index

    def test_tiles_should_serve_an_indexed_file_above_the_scan_ceiling(
        self, make_indexed_bedpe
    ):
        """Test the ceiling against a file that does not need scanning.

        Given:
            A BGZF-compressed, tabix-indexed BEDPE larger than the policy's
            scan ceiling, served under the default link policy.
        When:
            An in-bounds tile is served.
        Then:
            It should return the links in view. The default policy seeks
            nowhere -- either anchor may match -- so every in-bounds tile
            reached the ceiling, and an indexed file was refused with a message
            saying it must be indexed. ``TilesetUnavailable`` is a sibling of
            ``TileError``, not a subclass, so that refusal failed the whole
            batch rather than one tile.
        """
        # Arrange
        path = make_indexed_bedpe()
        assert os.path.getsize(path) > 100
        tileset = BedpeLinksTileset(
            path,
            CHROMSIZES,
            policy=TilePolicy(max_scan_bytes=100),
            tile_size=TILE_SIZE,
        )

        # Act
        (_, records), = tileset.tiles([tileset.parse_tile_id("u.0.0")])

        # Assert
        assert names(records) == ["r0", "r1", "r2", "r3"]

    def test_tiles_should_refuse_an_unindexed_file_above_the_scan_ceiling(
        self, make_bedpe
    ):
        """Test the control, so the case above does not disable the ceiling.

        Given:
            A plain BEDPE larger than the policy's scan ceiling.
        When:
            An in-bounds tile is served.
        Then:
            It should raise ``TilesetUnavailable``. Scanning is the cost the
            ceiling exists to refuse, and an unindexed file leaves no other
            way to read it.
        """
        # Arrange
        path = make_bedpe()
        assert os.path.getsize(path) > 50
        tileset = BedpeLinksTileset(
            path,
            CHROMSIZES,
            policy=TilePolicy(max_scan_bytes=50),
            tile_size=TILE_SIZE,
        )

        # Act & assert
        with pytest.raises(TilesetUnavailable):
            tileset.tiles([tileset.parse_tile_id("u.0.0")])

    def test_tiles_should_return_the_links_in_view(self, make_bedpe):
        """Test the ordinary tile, so the screen cannot pass by emptying all.

        Given:
            A links tileset over four pairs, and the whole-genome tile.
        When:
            It is served.
        Then:
            It should return every pair.
        """
        # Arrange
        tileset = BedpeLinksTileset(
            make_bedpe(), CHROMSIZES, tile_size=TILE_SIZE
        )

        # Act
        (_, records), = tileset.tiles([tileset.parse_tile_id("u.0.0")])

        # Assert
        assert sorted(names(records)) == ["r0", "r1", "r2", "r3"]

    def test_tiles_should_not_let_an_unknown_contig_consume_a_cap_slot(
        self, make_bedpe
    ):
        """Test the record cap against pairs that cannot be placed.

        Given:
            A links tileset under the default ``EITHER`` policy, over a BEDPE
            carrying twenty pairs whose first anchor is on an unnamed contig
            and whose second is in view, under a cap of four. ``EITHER`` is a
            disjunction, so unlike the rectangles path -- where the conjunction
            excludes an unknown contig before the cap ever sees it -- these
            rows do reach the cap.
        When:
            The whole-genome tile is served.
        Then:
            It should return the four placeable pairs. Capping before the
            contig filter lets the unplaceable rows win slots and then be
            dropped, silently shortening the tile.
        """
        # Arrange
        path = make_bedpe(RECORDS + UNPLACEABLE, name="extra.bedpe")
        tileset = BedpeLinksTileset(
            path,
            CHROMSIZES,
            tile_size=TILE_SIZE,
            policy=TilePolicy(max_records=4),
        )

        # Act
        (_, records), = tileset.tiles([tileset.parse_tile_id("u.0.0")])

        # Assert
        assert sorted(names(records)) == ["r0", "r1", "r2", "r3"]
