"""Cross-type conformance checks.

Check #1 of the suite sketched in section 2.4 of
_scratch/clodius-contracts-and-interface.md: **a dense tile must contain exactly
the number of bins its tileset_info advertises.**

This is the property that catches grid-reconciliation bugs (section 1.5b). Every
dense type has to squeeze a ragged, chromosome-respecting bin grid onto the
client's uniform lattice, and each one does it differently. When that arithmetic
drifts, the symptom is a tile of the wrong length -- or, worse, the right length
with the data shifted by a bin.

Types are added here as fixtures become available. Missing fixtures skip rather
than fail, since data/ lives in git LFS.
"""

import base64
import os.path as op

import numpy as np
import pytest

import clodius.tiles.bigwig as bigwig
import clodius.tiles.cooler as cooler
import clodius.tiles.fasta as fasta

BIGWIG = op.join(
    "data", "wgEncodeCaltechRnaSeqHuvecR1x75dTh1014IlnaPlusSignalRep2.bigWig"
)
COOLER = op.join("data", "Dixon2012-J1-NcoI-R1-filtered.100kb.multires.cool")
FASTA = op.join("data", "GCA_000350705.1_Esch_coli_KTE11_V1_genomic.short.fna")
FASTA_FAI = FASTA + ".fai"

COOLER_BINS_PER_TILE = 256


def requires(*paths):
    missing = [p for p in paths if not op.exists(p)]
    return pytest.mark.skipif(
        bool(missing), reason=f"missing LFS fixture(s): {missing}"
    )


def decode(payload):
    """Decode a dense payload into a flat array."""
    assert "error" not in payload, payload.get("error")
    return np.frombuffer(base64.b64decode(payload["dense"]), dtype=payload["dtype"])


def n_bins(payload):
    """Bins in a dense tile. ``size`` is values-per-bin (2 for minMax, 4 for
    whisker); it is absent on payloads that hand-roll their formatting."""
    return len(decode(payload)) // payload.get("size", 1)


# --- bigwig -----------------------------------------------------------------
# The interesting cases are tiles that span a chromosome boundary: those are
# where the ragged data grid over-represents and bins must be dropped to
# resynchronize. Tiles deep inside one chromosome never exercise it.


@requires(BIGWIG)
@pytest.mark.parametrize(
    "z,x",
    [
        (0, 0),  # whole genome: every boundary at once, plus past-genome padding
        (1, 0),
        (1, 1),
        (2, 1),  # previously emitted 1024 bins with the tail shifted by one
        (3, 1),
        (4, 2),
        (5, 31),
        (8, 128),
        (12, 2048),
        (10, 100),  # interior of a chromosome: no boundary crossed
    ],
)
def test_bigwig_tile_has_advertised_bin_count(z, x):
    tile_size = bigwig.tileset_info(BIGWIG)["tile_size"]
    (_, payload) = bigwig.tiles(BIGWIG, [f"x.{z}.{x}"])[0]

    assert n_bins(payload) == tile_size


@requires(BIGWIG)
@pytest.mark.parametrize(
    "mode,values_per_bin",
    [
        ("mean", 1),
        ("min", 1),
        ("max", 1),
        ("std", 1),
        ("sum", 1),
        ("minMax", 2),
        ("whisker", 4),
    ],
)
def test_bigwig_bin_count_independent_of_aggregation_mode(mode, values_per_bin):
    """Range modes change values-per-bin, never the bin count."""
    tile_size = bigwig.tileset_info(BIGWIG)["tile_size"]
    (_, payload) = bigwig.tiles(BIGWIG, [f"x.0.0.{mode}"])[0]

    assert payload["size"] == values_per_bin
    assert len(decode(payload)) == tile_size * values_per_bin
    assert n_bins(payload) == tile_size


@requires(BIGWIG)
def test_bigwig_tiles_are_uniform_length_across_zoom_levels():
    """Sweep every zoom level at the boundary-crossing edge of the genome."""
    info = bigwig.tileset_info(BIGWIG)
    tile_size, max_zoom = info["tile_size"], info["max_zoom"]

    lengths = set()
    for z in range(0, min(max_zoom, 13) + 1):
        for x in {0, 1, max(0, 2**z - 1)}:
            (_, payload) = bigwig.tiles(BIGWIG, [f"x.{z}.{x}"])[0]
            lengths.add(n_bins(payload))

    assert lengths == {tile_size}


# --- cooler -----------------------------------------------------------------


@requires(COOLER)
@pytest.mark.parametrize("z,x,y", [(0, 0, 0), (1, 0, 0), (1, 1, 1), (2, 1, 2)])
def test_cooler_tile_is_square_and_full(z, x, y):
    """cooler rasterizes by absolute position into a fixed 256x256 buffer, so
    length is correct by construction -- collisions and gaps are its failure
    mode, not length. Pinned so that stays true."""
    (_, payload) = cooler.tiles(COOLER, [f"a.{z}.{x}.{y}"])[0]

    assert len(decode(payload)) == COOLER_BINS_PER_TILE**2


# --- fasta ------------------------------------------------------------------


@requires(FASTA, FASTA_FAI)
def test_fasta_tile_is_full_away_from_the_genome_end():
    info = fasta.tileset_info(FASTA_FAI)
    (_, payload) = fasta.sequence_tiles(FASTA, [f"x.{info['max_zoom']}.0"], FASTA_FAI)[0]

    assert len(payload["sequence"]) == info["tile_size"]


@requires(FASTA, FASTA_FAI)
def test_fasta_tile_is_short_at_the_genome_end():
    """Documents current behavior rather than endorsing it.

    fasta does no grid reconciliation: intervals past the last contig
    contribute nothing, so the final tiles are short instead of padded. Every
    other dense type pads or corrects. See section 1.5b.
    """
    info = fasta.tileset_info(FASTA_FAI)
    z = info["max_zoom"]
    last_x = (info["max_width"] // info["tile_size"]) - 1

    (_, payload) = fasta.sequence_tiles(FASTA, [f"x.{z}.{last_x}"], FASTA_FAI)[0]

    assert len(payload["sequence"]) < info["tile_size"]
