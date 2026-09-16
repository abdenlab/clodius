import os.path as op

import clodius.tiles.beddb as hgbe

testdir = op.realpath(op.dirname(op.dirname(__file__)))


def test_get_tiles():
    filename = op.join(testdir, "data", "corrected.geneListwithStrand.bed.multires")

    hgbe.tiles(filename, ["x.1.0", "x.1.1"])
    # TODO: Do something with the return value


def test_list_items():
    filename = op.join(testdir, "data", "gene_annotations.short.db")

    items = hgbe.list_items(filename, 0, 100000000, max_entries=100)
    assert "name" not in items[0]
    # TODO: Do something with the return value


def test_name_in_tile():
    filename = op.join(testdir, "data", "geneAnnotationsExonUnions.1000.bed.v3.beddb")

    tiles = hgbe.tiles(filename, ["x.1.0", "x.1.1"])

    assert "name" in tiles[0][1][0]


def test_tileset_info():
    filename = op.join(testdir, "data", "geneAnnotationsExonUnions.1000.bed.v3.beddb")

    tileset_info = hgbe.tileset_info(filename)

    assert len(tileset_info["chromsizes"]) > 4
