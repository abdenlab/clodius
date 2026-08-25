import os.path as op

import clodius.tiles.fasta as ctf

from .harness.lfs import requires_lfs

fasta_filename = op.join("data", "GCA_000350705.1_Esch_coli_KTE11_V1_genomic.short.fna")
fai_filename = op.join(
    "data", "GCA_000350705.1_Esch_coli_KTE11_V1_genomic.short.fna.fai"
)

# Guarded per test rather than per module, because the two fixtures are not in
# the same state on a plain checkout: the ``.fna`` is LFS-backed and arrives as
# a ~130-byte pointer, while the ``.fai`` is three lines of real content checked
# in directly. That combination is the worst case for fasta -- a valid index
# describing 2320 bp of sequence that is not there -- and ``fetch_sequence``
# spins rather than raising, hanging a bare ``uv run pytest`` indefinitely.
# ``tileset_info`` reads only the index and is unaffected, so it keeps running.
requires_sequence = requires_lfs(fasta_filename, fai_filename)
requires_index = requires_lfs(fai_filename)


@requires_index
def test_tileset_info():
    tsinfo = ctf.tileset_info(fai_filename)

    assert "max_zoom" in tsinfo
    assert "max_width" in tsinfo


@requires_sequence
def test_multivec_tiles():
    tiles = ctf.multivec_tiles(
        fasta_filename, index_filename=fai_filename, tile_ids=["x.0.0"]
    )

    assert "shape" in tiles[0][1]


@requires_sequence
def test_sequence_tiles():

    tsinfo = ctf.tileset_info(fai_filename)

    tiles = ctf.sequence_tiles(
        fasta_filename, index_filename=fai_filename, tile_ids=["x.2.0"]
    )
    assert len(tiles[0][1]["sequence"]) == ctf.TILE_SIZE

    tiles = ctf.sequence_tiles(
        fasta_filename, index_filename=fai_filename, tile_ids=["x.0.0"]
    )
    assert len(tiles[0][1]["sequence"]) == tsinfo["max_pos"][0]
