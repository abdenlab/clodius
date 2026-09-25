"""Tests for the record cap in clodius.tiles_v2.gxf.

A GFF tileset caps by *gene*, not by row: thinning row by row would sever
exons from their transcripts and serve a gene missing pieces. That makes it
the one tileset the shared ``take_most_important`` helper does not reach, so
the cap-of-zero rule -- a server configured to serve no records must not emit
an unbounded tile -- is stated here separately rather than inherited.

The shape worth naming is a tile whose rows carry no gene- or pseudogene-typed
feature, which is ordinary outside GENCODE. It leaves the gene index empty,
which a bare ``len(spans) <= cap`` comparison reads as "comfortably under the
cap" and answers by returning every row unthinned. Nothing reaches the client
that way today, because gene assembly discards rows with no gene to root them
at -- but the thinning stage is the one that states the policy, and it should
state it whether or not a later stage happens to cover for it.

Fixtures are synthesized into a temp directory, so the module runs on a
checkout with no git-LFS payload.
"""

import pytest

from clodius.core.coords import Chromsizes
from clodius.core.policies import TilePolicy
from clodius.tiles_v2.gxf import GffGenesTileset

CHROMSIZES = Chromsizes.from_pairs([("c1", 1000)])

#: Two genes, each with an exon, so a cap has something to rank.
GENES = [
    ("c1", "test", "gene", 1, 100, "ID=g1;Name=alpha"),
    ("c1", "test", "exon", 11, 20, "ID=e1;Parent=g1"),
    ("c1", "test", "gene", 201, 500, "ID=g2;Name=beta"),
    ("c1", "test", "exon", 211, 220, "ID=e2;Parent=g2"),
]

#: Five genes carrying no ``ID=``, which GFF3 permits. The dialect reads no
#: id from them, so anything keyed on that id counts none of them -- while
#: gene assembly synthesizes one and emits all five.
GENES_NO_ID = [
    row
    for i in range(5)
    for row in (
        ("c1", "test", "gene", 1 + i * 100, 50 + i * 100, f"Name=g{i}"),
        (
            "c1",
            "test",
            "exon",
            11 + i * 100,
            20 + i * 100,
            f"Parent=gene_{1 + i * 100}_{50 + i * 100}",
        ),
    )
]

#: The same five genes, each carrying an ``ID=``. The control.
GENES_WITH_ID = [
    row
    for i in range(5)
    for row in (
        ("c1", "test", "gene", 1 + i * 100, 50 + i * 100, f"ID=g{i};Name=g{i}"),
        ("c1", "test", "exon", 11 + i * 100, 20 + i * 100, f"Parent=g{i}"),
    )
]

#: Rows typed only ``exon``, so the gene index stays empty.
CHILD_ONLY = [
    ("c1", "test", "exon", 11, 20, "ID=e1;Parent=t1"),
    ("c1", "test", "exon", 31, 40, "ID=e2;Parent=t1"),
]


def write_gff(path, rows):
    """Write rows to a plain GFF3 and return its path."""
    with open(path, "w") as fh:
        fh.write("##gff-version 3\n")
        for chrom, source, kind, start, end, attrs in rows:
            fh.write(
                f"{chrom}\t{source}\t{kind}\t{start}\t{end}\t.\t+\t.\t{attrs}\n"
            )
    return str(path)


@pytest.fixture
def make_gff(tmp_path):
    """Build a plain GFF3, so every tile scans the file."""

    def build(rows, name="annot.gff"):
        return write_gff(tmp_path / name, rows)

    return build


class TestGxfTileset:
    """The gene cap, at the boundary the shared helper does not cover."""

    def test_tiles_should_return_the_genes_when_the_cap_is_none(
        self, make_gff
    ):
        """Test the uncapped tile, so the cap tests cannot pass by emptying all.

        Given:
            An annotation carrying two genes, under a policy naming no cap.
        When:
            The whole-genome tile is served.
        Then:
            It should return both genes.
        """
        # Arrange
        tileset = GffGenesTileset(
            make_gff(GENES), CHROMSIZES, policy=TilePolicy(max_records=None)
        )

        # Act
        (_, records), = tileset.tiles([tileset.parse_tile_id("u.0.0")])

        # Assert
        assert len(records) == 2

    def test_tiles_should_return_one_gene_when_the_cap_is_one(self, make_gff):
        """Test that the cap ranks rather than truncating arbitrarily.

        Given:
            The same annotation, under a policy capping genes at one.
        When:
            The whole-genome tile is served.
        Then:
            It should return the longer-spanning gene. Selection is by span,
            the same measure served as ``importance``, so the gene that
            survives is the one the client would have ranked highest anyway.
        """
        # Arrange
        tileset = GffGenesTileset(
            make_gff(GENES), CHROMSIZES, policy=TilePolicy(max_records=1)
        )

        # Act
        (_, records), = tileset.tiles([tileset.parse_tile_id("u.0.0")])

        # Assert
        assert len(records) == 1
        assert "beta" in records[0]["fields"]

    def test_tiles_should_return_nothing_when_the_cap_is_zero(self, make_gff):
        """Test a cap of zero against a tile that has genes to drop.

        Given:
            An annotation carrying two genes, under a policy capping genes at
            zero.
        When:
            The whole-genome tile is served.
        Then:
            It should return no records. A server configured to serve none
            must not emit a tile, and the slice implementing the cap inverts
            at zero to mean "everything" if it is reached.
        """
        # Arrange
        tileset = GffGenesTileset(
            make_gff(GENES), CHROMSIZES, policy=TilePolicy(max_records=0)
        )

        # Act
        (_, records), = tileset.tiles([tileset.parse_tile_id("u.0.0")])

        # Assert
        assert records == []

    def test_tiles_should_apply_the_cap_when_no_gene_row_carries_an_id(
        self, make_gff
    ):
        """Test a positive cap against genes the dialect reads no id from.

        Given:
            Five ``ID=``-less genes, which GFF3 permits, under a policy
            capping genes at two.
        When:
            The whole-genome tile is served.
        Then:
            It should return two genes. Counting by id counted none of them
            while gene assembly synthesized one per row and emitted all five,
            so every positive cap passed the tile through whole.
        """
        # Arrange
        tileset = GffGenesTileset(
            make_gff(GENES_NO_ID), CHROMSIZES, policy=TilePolicy(max_records=2)
        )

        # Act
        (_, records), = tileset.tiles([tileset.parse_tile_id("u.0.0")])

        # Assert
        assert len(records) == 2

    def test_tiles_should_apply_the_cap_when_every_gene_row_carries_an_id(
        self, make_gff
    ):
        """Test the control for the ``ID=``-less case above.

        Given:
            The same five genes, each carrying an ``ID=``, under the same cap
            of two.
        When:
            The whole-genome tile is served.
        Then:
            It should also return two genes -- so the case above pins the
            missing id rather than anything about the cap itself.
        """
        # Arrange
        tileset = GffGenesTileset(
            make_gff(GENES_WITH_ID),
            CHROMSIZES,
            policy=TilePolicy(max_records=2),
        )

        # Act
        (_, records), = tileset.tiles([tileset.parse_tile_id("u.0.0")])

        # Assert
        assert len(records) == 2

    def test_tiles_should_return_nothing_when_no_row_carries_a_gene(
        self, make_gff
    ):
        """Test the file shape whose gene index is empty.

        Given:
            An annotation of exons with no gene- or pseudogene-typed row,
            under a policy capping genes at zero.
        When:
            The whole-genome tile is served.
        Then:
            It should return no records. The cap comparison reads an empty
            gene index as under the cap and passes every row through
            unthinned; gene assembly then discards them for want of a gene to
            root at, so this pins the tile rather than the stage -- the stage's
            own guard is what keeps the two from having to agree.
        """
        # Arrange
        tileset = GffGenesTileset(
            make_gff(CHILD_ONLY), CHROMSIZES, policy=TilePolicy(max_records=0)
        )

        # Act
        (_, records), = tileset.tiles([tileset.parse_tile_id("u.0.0")])

        # Assert
        assert records == []
