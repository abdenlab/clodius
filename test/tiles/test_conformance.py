"""Cross-type conformance checks against the legacy tile modules.

**A dense tile must contain exactly the number of bins its tileset_info
advertises.** This is the property that catches grid-reconciliation bugs. Every
dense type has to squeeze a ragged, chromosome-respecting bin grid onto the
client's uniform lattice, and each one does it differently. When that
arithmetic drifts, the symptom is a tile of the wrong length -- or, worse, the
right length with the data shifted by a bin.

These pin ``clodius.tiles.*``, not ``clodius.tiles_v2.*``. They are the
reference the rewrites are compared against, which is why they live beside the
modules they cover rather than under ``test/core/``.

Types are added here as fixtures become available. Missing fixtures skip rather
than fail, since ``data/`` lives in git LFS.
"""

import os.path as op
import warnings

import pytest

import clodius.tiles.bigwig as bigwig
import clodius.tiles.cooler as cooler
import clodius.tiles.fasta as fasta

from ..harness.lfs import requires_lfs, unavailable
from ..harness.wire import decode, n_bins, square

BIGWIG = op.join(
    "data", "wgEncodeCaltechRnaSeqHuvecR1x75dTh1014IlnaPlusSignalRep2.bigWig"
)
COOLER = op.join("data", "Dixon2012-J1-NcoI-R1-filtered.100kb.multires.cool")
FASTA = op.join("data", "GCA_000350705.1_Esch_coli_KTE11_V1_genomic.short.fna")
FASTA_FAI = FASTA + ".fai"

ALL_FIXTURES = (BIGWIG, COOLER, FASTA, FASTA_FAI)


# --- fixture availability ---------------------------------------------------
# Not skipped, so "everything skipped" is visibly distinct from "everything
# passed". Without this, a checkout with no LFS payloads reports a green file
# in which zero assertions ran.


def test_unavailable_should_name_every_missing_or_unsmudged_fixture():
    """Test that the suite says out loud when it is running on stubs.

    Given:
        The LFS-backed fixtures this module depends on.
    When:
        Each is checked for content.
    Then:
        It should pass, emitting a warning naming any that are absent or
        un-smudged so a green run is not mistaken for a covered one, and it
        should name only fixtures this module actually declares.

        Passing loudly is the point. Skipping here would put this test in the
        same skip count as the 26 below it, which is exactly the signal it
        exists to distinguish -- and the assertion after a ``pytest.skip`` is
        unreachable, so the previous form could not fail either way.
    """
    # Act
    missing = unavailable(*ALL_FIXTURES)

    # Assert
    if missing:
        warnings.warn(
            "LFS payloads not materialized, so every check below is skipped: "
            f"{missing}. Run `git lfs pull` to cover them.",
            stacklevel=1,
        )
    assert set(missing) <= set(ALL_FIXTURES)


# --- bigwig -----------------------------------------------------------------
# The interesting cases are tiles that span a chromosome boundary: those are
# where the ragged data grid over-represents and bins must be dropped to
# resynchronize. Tiles deep inside one chromosome never exercise it.


@requires_lfs(BIGWIG)
@pytest.mark.parametrize(
    "z,x",
    [
        (0, 0),  # whole genome: every boundary at once, plus past-genome pad
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
def test_tiles_should_return_the_advertised_bin_count_for_bigwig(z, x):
    """Test that a bigwig tile is exactly as long as tileset_info promises.

    Given:
        A bigwig and a tile position, including positions whose tile spans a
        chromosome boundary.
    When:
        The tile is generated.
    Then:
        It should carry exactly ``tile_size`` bins.
    """
    # Arrange
    tile_size = bigwig.tileset_info(BIGWIG)["tile_size"]

    # Act
    (_, payload) = bigwig.tiles(BIGWIG, [f"x.{z}.{x}"])[0]

    # Assert
    assert n_bins(payload) == tile_size


@requires_lfs(BIGWIG)
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
def test_tiles_should_keep_the_bin_count_when_the_range_mode_changes(
    mode, values_per_bin
):
    """Test that range modes change values-per-bin, never the bin count.

    Given:
        A bigwig and each of its aggregation and range modes.
    When:
        A tile is generated in that mode.
    Then:
        It should carry ``tile_size`` bins, each holding as many values as
        the mode declares.
    """
    # Arrange
    tile_size = bigwig.tileset_info(BIGWIG)["tile_size"]

    # Act
    (_, payload) = bigwig.tiles(BIGWIG, [f"x.0.0.{mode}"])[0]

    # Assert
    assert payload["size"] == values_per_bin
    assert len(decode(payload)) == tile_size * values_per_bin
    assert n_bins(payload) == tile_size


@requires_lfs(BIGWIG)
def test_tiles_should_return_a_uniform_length_at_every_zoom_level():
    """Test bin-count stability across the whole resolution ladder.

    Given:
        A bigwig, swept at the first, second and last position of every zoom
        up to 13. The cap is deliberate: this reads a real multi-gigabase
        bigWig from ``data/``, and the zooms past 13 hold tens of thousands of
        tiles each. Positions 0 and 1 sit at the start of the genome where the
        first contig boundary falls, and ``2**z - 1`` is the tile that
        overhangs the end -- the three places the reconciler has to drop or
        pad bins.
    When:
        Tiles are generated.
    Then:
        It should return the same bin count everywhere.
    """
    # Arrange
    info = bigwig.tileset_info(BIGWIG)
    tile_size, max_zoom = info["tile_size"], info["max_zoom"]

    # Act
    lengths = set()
    for z in range(0, min(max_zoom, 13) + 1):
        for x in {0, 1, max(0, 2**z - 1)}:
            (_, payload) = bigwig.tiles(BIGWIG, [f"x.{z}.{x}"])[0]
            lengths.add(n_bins(payload))

    # Assert
    assert lengths == {tile_size}


# --- cooler -----------------------------------------------------------------


@requires_lfs(COOLER)
@pytest.mark.parametrize("z,x,y", [(0, 0, 0), (1, 0, 0), (1, 1, 1), (2, 1, 2)])
def test_tiles_should_return_a_square_full_tile_for_cooler(z, x, y):
    """Test that a cooler tile fills its fixed buffer.

    Given:
        A multires cooler and a tile position inside its ladder.
    When:
        The tile is generated.
    Then:
        It should be a square whose side is the ``bins_per_dimension`` the
        info advertises. cooler rasterizes by absolute position into a fixed
        buffer, so length is correct by construction -- collisions and gaps
        are its failure mode, not length. Pinned so that stays true.

        The side is read off the info rather than written down here. Asserting
        against a local ``256`` compares the payload with a constant this test
        chose, not with the number the tileset told the client to expect --
        which is the only thing that makes a wrong length detectable.
    """
    # Arrange
    side = cooler.tileset_info(COOLER)["bins_per_dimension"]

    # Act
    (_, payload) = cooler.tiles(COOLER, [f"a.{z}.{x}.{y}"])[0]

    # Assert
    assert square(payload).shape == (side, side)


@requires_lfs(COOLER)
def test_tiles_should_skip_the_tile_when_the_zoom_level_is_past_the_ladder():
    """Test the guard at the top of the resolution ladder.

    Given:
        A multires cooler and a zoom level one past its last resolution,
        which indexes ``sorted_resolutions`` out of range.
    When:
        The tile is requested.
    Then:
        It should be skipped rather than raising IndexError. The guard read
        ``>`` where it needed ``>=``, so the boundary case fell through.
    """
    # Arrange
    z = len(cooler.tileset_info(COOLER)["resolutions"])

    # Act
    result = cooler.tiles(COOLER, [f"a.{z}.0.0"])

    # Assert
    assert result == []


# --- fasta ------------------------------------------------------------------


@requires_lfs(FASTA, FASTA_FAI)
def test_sequence_tiles_should_be_full_away_from_the_genome_end():
    """Test that an interior sequence tile is a full tile.

    Given:
        A fasta and a tile position away from the end of the genome.
    When:
        The sequence tile is generated.
    Then:
        It should be exactly ``tile_size`` bases long.
    """
    # Arrange
    info = fasta.tileset_info(FASTA_FAI)

    # Act
    (_, payload) = fasta.sequence_tiles(
        FASTA, [f"x.{info['max_zoom']}.0"], FASTA_FAI
    )[0]

    # Assert
    assert len(payload["sequence"]) == info["tile_size"]


@requires_lfs(FASTA, FASTA_FAI)
def test_sequence_tiles_should_be_short_at_the_genome_end():
    """Test the current end-of-genome behavior, rather than endorsing it.

    Given:
        A fasta and the last tile position in its coordinate space.
    When:
        The sequence tile is generated.
    Then:
        It should come back short. fasta does no grid reconciliation:
        intervals past the last contig contribute nothing, so the final tiles
        are short instead of padded. Every other dense type pads or corrects.
    """
    # Arrange
    info = fasta.tileset_info(FASTA_FAI)
    z = info["max_zoom"]
    last_x = (info["max_width"] // info["tile_size"]) - 1

    # Act
    (_, payload) = fasta.sequence_tiles(
        FASTA, [f"x.{z}.{last_x}"], FASTA_FAI
    )[0]

    # Assert
    expected = info["max_pos"][0] - last_x * info["tile_size"]
    assert len(payload["sequence"]) == expected
    assert 0 < expected < info["tile_size"]
