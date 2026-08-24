"""Regression tests for clodius.tiles.bigwig's bin accounting.

``clodius/tiles/bigwig.py`` serves a *uniform* lattice -- ``tile_size`` bins of
``binsize`` bp each -- out of a *ragged* one, since each chromosome contributes
``ceil(length / binsize)`` bins while occupying only ``length`` bp. Every
chromosome boundary therefore over-represents by up to a bin, and the module
drops a trailing bin whenever that drift reaches a full one.

Two defects in that accounting were fixed on this branch. The bin count came
from ``ceil(end / binsize) - floor(start / binsize)``, which over-counts by one
whenever the interval starts mid-bin, and the drop test read ``>`` where it
needed ``>=``, so at exactly one bin of drift the surplus bin stayed. They
produce three distinct failure modes, and only one of them changes a length:

* 1025 bins where the tileset info promises 1024
* 1024 bins with a real bin replaced by padding, losing data outright
* 1024 bins with the correct number of real bins placed one bin to the left

``test/tiles/test_conformance.py`` covers all three against a real bigWig from
``data/``, which is an un-smudged git-LFS pointer on a plain checkout -- so it
skips, and has skipped for the whole life of the fix. Everything here builds its
own bigWig, and the three genomes below were found by searching for one instance
of each mode.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from clodius.tiles import bigwig as mod

from ..harness import builders
from ..harness.wire import decode

TILE_SIZE = 1024

#: One genome per pre-fix failure mode, each found by differential search
#: against the old accounting. The lengths are not adjustable: each depends on
#: exactly where its contig boundaries fall relative to the bin grid.
EMITS_AN_EXTRA_BIN = (
    [["c0", 764], ["c1", 613], ["c2", 324], ["c3", 370], ["c4", 50], ["c5", 91], ["c6", 20]],
    0,
    0,
)
DISCARDS_A_REAL_BIN = (
    [["c0", 1122], ["c1", 333], ["c2", 979], ["c3", 805], ["c4", 4]],
    1,
    1,
)
SHIFTS_THE_TAIL_LEFT = (
    [["c0", 955], ["c1", 846], ["c2", 277], ["c3", 920], ["c4", 63], ["c5", 684], ["c6", 486]],
    2,
    1,
)


def build(tmp_path, chromsizes, name="tiny.bw"):
    """A bigWig over ``chromsizes``, with contig ``i`` carrying signal ``i + 1``."""
    return str(builders.build_bigwig(tmp_path / name, chromsizes=chromsizes))


def tile_at(path, z, x, chromsizes):
    """One tile's values, addressed by zoom and position rather than by span."""
    info = mod.tileset_info(path)
    span = info["max_width"] // 2**z
    return mod.get_bigwig_tile(path, z, x * span, (x + 1) * span, chromsizes)


def expected_signal(chromsizes, bp):
    """The signal ``build`` writes at an absolute genomic position.

    ``None`` past the end of the genome, where a tile carries padding.
    """
    acc = 0
    for i, (_, length) in enumerate(chromsizes):
        if acc <= bp < acc + length:
            return float(i + 1)
        acc += length
    return None


class TestAbs2Genomic:
    """Splitting an absolute span into per-chromosome intervals.

    Shared with ``clodius/tiles/vcf.py``, which imports it directly, so a change
    here moves tile boundaries in two modules.
    """

    @pytest.fixture
    def chromsizes(self):
        """A three-contig genome as the pandas Series the module expects."""

        return pd.Series(
            [1000, 1500, 500], index=["c1", "c2", "c3"], dtype=int
        )

    def test_abs2genomic_should_return_one_interval_inside_a_chromosome(
        self, chromsizes
    ):
        """Test a span that crosses no boundary.

        Given:
            A span entirely within the first contig.
        When:
            It is split.
        Then:
            It should yield that one interval, in contig-relative coordinates.
        """
        # Act & assert
        assert list(mod.abs2genomic(chromsizes, 100, 400)) == [(0, 100, 400)]

    def test_abs2genomic_should_split_a_span_at_every_boundary(
        self, chromsizes
    ):
        """Test a span crossing two boundaries.

        Given:
            A span running from inside the first contig to inside the third.
        When:
            It is split.
        Then:
            It should yield the tail of the first, the whole of the second, and
            the head of the third -- each restarting from that contig's own
            zero.
        """
        # Act
        intervals = list(mod.abs2genomic(chromsizes, 900, 2600))

        # Assert
        assert intervals == [(0, 900, 1000), (1, 0, 1500), (2, 0, 100)]

    def test_abs2genomic_should_emit_a_synthetic_contig_past_the_genome(
        self, chromsizes
    ):
        """Test a span extending beyond the last contig.

        Given:
            A span whose end lies past the genome, which every implicit ladder
            produces because ``max_width`` always exceeds the genome length.
        When:
            It is split.
        Then:
            The real contig should be clamped at its own length and the
            overshoot handed back under a contig id one past the last. That id
            indexes nothing, which is how the caller recognizes padding -- the
            span is not clamped away and not attributed to a real chromosome.
        """
        # Act
        intervals = list(mod.abs2genomic(chromsizes, 2900, 3500))

        # Assert
        assert intervals == [(2, 400, 500), (3, 0, 500)]


class TestTilesetInfo:
    """The quadtree the module advertises."""

    def test_tileset_info_should_declare_a_power_of_two_canvas(self, tmp_path):
        """Test the ladder geometry.

        Given:
            A 3000 bp genome at the module's 1024 bp tile size.
        When:
            The info is read.
        Then:
            It should declare max_zoom 2 and a 4096 bp canvas -- the smallest
            power-of-two multiple of the tile size covering the genome.
        """
        # Arrange
        path = build(tmp_path, [["c1", 1000], ["c2", 1500], ["c3", 500]])

        # Act
        info = mod.tileset_info(path)

        # Assert
        assert info["tile_size"] == TILE_SIZE
        assert info["max_zoom"] == 2
        assert info["max_width"] == 4096


class TestBinCount:
    """The length half of the contract."""

    def test_get_bigwig_tile_should_not_emit_an_extra_bin(self, tmp_path):
        """Test the pre-fix mode that changed a length.

        Given:
            A genome whose contig boundaries accumulate exactly one bin of
            drift by the end of the whole-genome tile.
        When:
            That tile is generated.
        Then:
            It should carry ``tile_size`` bins. The drop test read ``>``, so at
            exactly one bin of drift the surplus bin was kept and the tile came
            back 1025 bins long -- one more than the tileset info promises,
            which the client reads as a whole different resolution.
        """
        # Arrange
        chromsizes, z, x = EMITS_AN_EXTRA_BIN
        path = build(tmp_path, chromsizes)

        # Act
        values = tile_at(path, z, x, mod.get_chromsizes(path))

        # Assert
        assert len(values) == TILE_SIZE

    @pytest.mark.parametrize(
        "case",
        [EMITS_AN_EXTRA_BIN, DISCARDS_A_REAL_BIN, SHIFTS_THE_TAIL_LEFT],
        ids=["extra-bin", "lost-bin", "shifted-tail"],
    )
    def test_get_bigwig_tile_should_hold_the_bin_count_at_every_position(
        self, tmp_path, case
    ):
        """Test the bin count across a whole ladder, not just one tile.

        Given:
            Any of the three genomes, swept at every zoom and position.
        When:
            Each tile is generated.
        Then:
            All should carry ``tile_size`` bins. A tile deep inside one contig
            crosses no boundary and can never exercise this; the sweep is what
            reaches the ones that do.
        """
        # Arrange
        chromsizes, _, _ = case
        path = build(tmp_path, chromsizes)
        cs = mod.get_chromsizes(path)
        info = mod.tileset_info(path)

        # Act
        lengths = {
            len(tile_at(path, z, x, cs))
            for z in range(info["max_zoom"] + 1)
            for x in range(2**z)
        }

        # Assert
        assert lengths == {TILE_SIZE}

    @pytest.mark.parametrize(
        "mode,values_per_bin",
        [
            ("mean", 1),
            ("sum", 1),
            ("minMax", 2),
            ("whisker", 4),
        ],
    )
    def test_tiles_should_keep_the_bin_count_when_the_range_mode_changes(
        self, tmp_path, mode, values_per_bin
    ):
        """Test that a display mode widens a bin without adding bins.

        Given:
            A tile requested in each aggregation and range mode.
        When:
            It is generated.
        Then:
            It should report the mode's values per bin and ship that many times
            ``tile_size`` values. Range modes run the same accounting path
            with a wider array, so an off-by-one there is multiplied rather
            than exposed.
        """
        # Arrange
        path = build(tmp_path, [["c1", 1000], ["c2", 1500], ["c3", 500]])

        # Act
        ((_, payload),) = mod.tiles(path, [f"x.0.0.{mode}"])

        # Assert
        assert payload["size"] == values_per_bin
        assert len(decode(payload)) == TILE_SIZE * values_per_bin


class TestBinPlacement:
    """The content half, which the bin count cannot see.

    Two of the three pre-fix modes returned a tile of exactly the right length.
    What made them wrong was *where* the values sat: the lattice is uniform, so
    bin ``i`` of zoom ``z`` covers ``[i * binsize, (i + 1) * binsize)`` in
    genome coordinates and must report the contig living there.
    """

    def assert_placed_at(self, path, chromsizes, z, x):
        """Every real bin of tile ``x`` reports the contig at its own position.

        Returns the number of real -- that is, in-genome -- bins the tile
        carried, which is what separates "a real bin was discarded" from "the
        tail shifted left": both leave the tile the right length.
        """
        cs = mod.get_chromsizes(path)
        info = mod.tileset_info(path)
        binsize = (info["max_width"] // 2**z) // TILE_SIZE
        real = 0
        for i, value in enumerate(tile_at(path, z, x, cs)):
            bp = (x * TILE_SIZE + i) * binsize
            want = expected_signal(chromsizes, bp)
            if want is None:
                continue
            real += 1
            assert value == want, (
                f"zoom {z} tile {x} bin {i} covers bp {bp} on contig "
                f"{want:.0f} but reports {value}"
            )
        return real

    def assert_placed(self, path, chromsizes, z):
        """Every real bin at zoom ``z`` reports the contig at its own position."""
        for x in range(2**z):
            self.assert_placed_at(path, chromsizes, z, x)

    def test_get_bigwig_tile_should_not_discard_a_real_bin(self, tmp_path):
        """Test the pre-fix mode that lost data at a fixed length.

        Given:
            A genome ending in a 4 bp contig, so the drift crosses a bin
            boundary right where the real data runs out.
        When:
            The tile at that boundary is generated.
        Then:
            Every real bin should carry its own contig's signal, and the tile
            should hold 598 of them against a length of 1024 -- so it is
            partly past the genome, which is the situation in which a chunk of
            one bin can lose it. The pre-fix module fails the placement
            assertion here before any count is taken, so the count is not what
            catches the regression; it is what keeps this test distinct from
            the tail-shift one below, whose tile is fully backed at 1024.

            The single position the constant names is what is generated, not
            every tile at that zoom: the constant was found by differential
            search precisely because ``x`` is where the mode shows, and
            sweeping the zoom makes that field dead data.
        """
        # Arrange
        chromsizes, z, x = DISCARDS_A_REAL_BIN
        path = build(tmp_path, chromsizes)

        # Act
        real = self.assert_placed_at(path, chromsizes, z, x)

        # Assert
        assert real == 598

    def test_get_bigwig_tile_should_not_shift_the_tail_left(self, tmp_path):
        """Test the pre-fix mode that was invisible to every length check.

        Given:
            A seven-contig genome whose accumulated drift crosses a bin mid-tile
            at the deepest zoom.
        When:
            That tile is generated.
        Then:
            Every real bin should report the contig at its own genomic
            position, across all 1024 of them -- this tile is fully inside the
            genome, unlike the one above. The pre-fix tile had the right
            length and the right number of real bins, with the tail sitting
            one bin to the left, which draws as a track offset by one bin and
            nothing else. Only placement can see it.
        """
        # Arrange
        chromsizes, z, x = SHIFTS_THE_TAIL_LEFT
        path = build(tmp_path, chromsizes)

        # Act
        real = self.assert_placed_at(path, chromsizes, z, x)

        # Assert
        assert real == 1024

    @given(
        lengths=st.lists(
            st.integers(min_value=1, max_value=1200),
            min_size=2,
            max_size=7,
        ).filter(lambda ls: sum(ls) > 1200)
    )
    @settings(max_examples=25, deadline=None)
    def test_get_bigwig_tile_should_place_every_bin_at_its_own_position(
        self, tmp_path_factory, lengths
    ):
        """Test bin placement over arbitrary contig geometry.

        Given:
            Any genome of two to seven contigs, at any zoom in its ladder.
        When:
            Every tile is generated.
        Then:
            Bin ``i`` should carry the signal of the contig containing bp
            ``i * binsize``, exactly -- no tolerance. This is the whole contract
            the reconciler implements, and it holds without slack: the drift
            budget is spent resynchronizing, not absorbed. The pre-fix module
            fails it on roughly two thirds of randomly drawn genomes.
        """
        # Arrange
        chromsizes = [[f"c{i}", n] for i, n in enumerate(lengths)]
        path = build(
            tmp_path_factory.mktemp("bw"), chromsizes, name="drift.bw"
        )
        info = mod.tileset_info(path)

        # Act & assert
        for z in range(info["max_zoom"] + 1):
            self.assert_placed(path, chromsizes, z)

    def test_get_bigwig_tile_should_pad_past_the_genome_with_nan(
        self, tmp_path
    ):
        """Test the region past the last contig.

        Given:
            The whole-genome tile over a 3000 bp genome on a 4096 bp canvas.
        When:
            It is generated.
        Then:
            Its real bins should end exactly where the genome does and the
            remainder should be NaN. Padding with zero instead would draw a
            flat track past the end of the assembly rather than nothing.
        """
        # Arrange
        chromsizes = [["c1", 1000], ["c2", 1500], ["c3", 500]]
        path = build(tmp_path, chromsizes)

        # Act
        values = tile_at(path, 0, 0, mod.get_chromsizes(path))

        # Assert
        real = 3000 // 4
        assert not np.isnan(values[:real]).any()
        assert np.isnan(values[real:]).all()
