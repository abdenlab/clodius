"""Tests for clodius.core.coords.

The differential tests pin the new implementation against the three legacy
``abs2genomic`` functions it is meant to replace. They should stay in place
until the last caller has migrated, at which point the legacy functions (and
these comparisons) can be deleted.
"""

import os.path as op
import random

import pandas as pd
import pytest

from clodius.tiles.chromosomes import chromsizes_as_array
from clodius.core.coords import (
    Chromsizes,
    GenomicRange,
    natsorted,
)
from clodius.core.tileset import _quadtree_depth as quadtree_depth
from clodius.tiles.bigwig import TILE_SIZE as BIGWIG_TILE_SIZE
from clodius.tiles.bigwig import abs2genomic as abs2genomic_bigwig
from clodius.tiles.bigwig import get_quadtree_depth as quadtree_depth_bigwig
from clodius.tiles.bigwig import natsorted as natsorted_bigwig
from clodius.tiles.cooler import get_quadtree_depth as quadtree_depth_cooler
from clodius.tiles.multivec import abs2genomic as abs2genomic_multivec
from clodius.tiles.utils import abs2genomic as abs2genomic_utils
from clodius.tiles.utils import get_quadtree_depth as quadtree_depth_utils
from clodius.tiles.utils import natsorted as natsorted_utils

testdir = op.realpath(op.dirname(op.dirname(__file__)))

TINY = [["c1", 100], ["c2", 200], ["c3", 50]]


def as_tuples(intervals):
    return [(iv.cid, iv.start, iv.end) for iv in intervals]


def legacy_as_tuples(gen):
    """Normalize legacy output, which leaks numpy scalars and floats.

    Zero-length intervals are dropped. Legacy derives its last chromosome index
    from the span's *exclusive* end, so an end landing exactly on a boundary --
    including the end of the genome -- trails an empty interval, flagged
    out-of-bounds in the whole-genome case. That is the second intentional
    difference, after the int cast.
    """
    return [
        (int(cid), int(start), int(end))
        for cid, start, end in gen
        if int(end) > int(start)
    ]


# --- basic semantics --------------------------------------------------------


def test_within_single_chromosome():
    cs = Chromsizes.from_pairs(TINY)
    assert as_tuples(cs.invert((10, 50))) == [(0, 10, 50)]


def test_spanning_two_chromosomes():
    cs = Chromsizes.from_pairs(TINY)
    assert as_tuples(cs.invert((90, 150))) == [(0, 90, 100), (1, 0, 50)]


def test_boundary_belongs_to_following_chromosome():
    """Half-open: a position exactly on a boundary starts the next chrom."""
    cs = Chromsizes.from_pairs(TINY)
    assert as_tuples(cs.invert((100, 150))) == [(1, 0, 50)]


def test_offsets_are_ints():
    cs = Chromsizes.from_pairs(TINY)
    assert cs.offsets == {"c1": 0, "c2": 100, "c3": 300}
    assert all(isinstance(v, int) for v in cs.offsets.values())


# --- the out-of-bounds tail -------------------------------------------------


def test_past_end_is_flagged_not_dropped():
    """The tail past the last chromosome must survive.

    It is what fills the trailing NaN bins of the final tiles, where a
    tileset's power-of-two max_width exceeds the genome length.
    """
    cs = Chromsizes.from_pairs(TINY)
    ivs = list(cs.invert((250, 470)))

    assert len(ivs) == 3
    assert as_tuples(ivs) == [(1, 150, 200), (2, 0, 50), (3, 0, 120)]
    assert [iv.is_out_of_bounds for iv in ivs] == [False, False, True]
    assert ivs[-1].name is None
    assert ivs[-1].cid == len(cs)


def test_whole_genome_has_no_tail():
    """A span ending exactly at the end of the genome is entirely in bounds.

    Legacy trails an empty out-of-bounds interval here, because it locates the
    last chromosome from the exclusive end.
    """
    cs = Chromsizes.from_pairs(TINY)
    ivs = list(cs.invert((0, cs.total_length)))

    assert len(ivs) == len(cs)
    assert not any(iv.is_out_of_bounds for iv in ivs)
    assert as_tuples(ivs)[-1] == (len(cs) - 1, 0, TINY[-1][1])


def test_boundary_aligned_end_stops_short():
    cs = Chromsizes.from_pairs(TINY)
    assert as_tuples(cs.invert((0, TINY[0][1]))) == [(0, 0, TINY[0][1])]


def test_empty_span_covers_nothing():
    cs = Chromsizes.from_pairs(TINY)
    assert list(cs.invert((50, 50))) == []


def test_entirely_past_end():
    cs = Chromsizes.from_pairs(TINY)
    ivs = list(cs.invert((360, 400)))

    assert len(ivs) == 1
    assert ivs[0].is_out_of_bounds
    assert (ivs[0].start, ivs[0].end) == (10, 50)


# --- coordinates are always ints --------------------------------------------


def test_fractional_input_truncates_to_int():
    """tabix/vcf/bedfile pass ``x * max_width / 2**z``, which is fractional.

    Legacy leaked those floats through (bigwig always, utils on all but its
    final interval). Every caller cast at the point of use.
    """
    cs = Chromsizes.from_pairs(TINY)
    ivs = list(cs.invert((90.5, 150.7)))

    assert as_tuples(ivs) == [(0, 90, 100), (1, 0, 50)]
    for iv in ivs:
        assert isinstance(iv.start, int) and isinstance(iv.end, int)


# --- rejected input ---------------------------------------------------------


def test_negative_start_raises():
    """Legacy indexed backwards off the end of the offsets array instead."""
    cs = Chromsizes.from_pairs(TINY)
    with pytest.raises(ValueError, match="negative start"):
        list(cs.invert((-5, 100)))


def test_reversed_span_raises():
    cs = Chromsizes.from_pairs(TINY)
    with pytest.raises(ValueError, match="precedes start"):
        list(cs.invert((200, 100)))


# --- differential against the three legacy implementations ------------------


def _chromsizes_cases():
    cases = [("tiny", TINY)]
    for name in ("chm13v1.chrom.sizes", "hg38.chrom.sizes"):
        path = op.join(testdir, "data", name)
        if op.exists(path):
            cases.append((name, chromsizes_as_array(path)))
    return cases


@pytest.mark.parametrize("label,pairs", _chromsizes_cases())
def test_matches_legacy_implementations(label, pairs):
    """Same intervals as all three legacy functions, once both are int-cast.

    The cast is the one intentional difference; this asserts the underlying
    interval arithmetic is unchanged.
    """
    cs = Chromsizes.from_pairs(pairs)
    lengths = list(cs.lengths)
    series = pd.Series(lengths, index=list(cs.names))
    total = cs.total_length

    spans = [(0, total), (0, total * 2), (total, total + 1000), (0, 1)]
    rng = random.Random(0)
    for _ in range(500):
        start = rng.randint(0, int(total * 1.3))
        spans.append((start, start + rng.randint(0, int(total * 0.4))))
    for _ in range(200):
        start = rng.uniform(0, total)
        spans.append((start, start + rng.uniform(0, total * 0.2)))

    for start, end in spans:
        expected = as_tuples(cs.invert((start, end)))
        assert expected == legacy_as_tuples(
            abs2genomic_utils(lengths, start, end)
        ), f"utils differs on ({start}, {end})"
        assert expected == legacy_as_tuples(
            abs2genomic_multivec(lengths, start, end)
        ), f"multivec differs on ({start}, {end})"
        assert expected == legacy_as_tuples(
            abs2genomic_bigwig(series, start, end)
        ), f"bigwig differs on ({start}, {end})"


def test_interval_is_hashable():
    assert len({GenomicRange(0, "c1", 0, 10), GenomicRange(0, "c1", 0, 10)}) == 1


# --- quadtree_depth ---------------------------------------------------------


@pytest.mark.parametrize(
    "total,tile_size,expected",
    [
        (1, 1024, 0),  # smaller than one tile
        (1024, 1024, 0),  # exactly one tile
        (1025, 1024, 1),  # spills into a second
        (2048, 1024, 1),
        (2049, 1024, 2),
        (4096, 1024, 2),
    ],
)
def test_quadtree_depth(total, tile_size, expected):
    assert quadtree_depth(total, tile_size) == expected


@pytest.mark.parametrize("bad", [0, -1])
def test_quadtree_depth_rejects_nonpositive(bad):
    with pytest.raises(ValueError):
        quadtree_depth(bad, 1024)
    with pytest.raises(ValueError):
        quadtree_depth(1024, bad)


@pytest.mark.parametrize("label,pairs", _chromsizes_cases())
def test_quadtree_depth_matches_legacy_call_sites(label, pairs):
    """All three legacy versions share this formula and differ only in how they
    derive tile_size_bp. Each is checked with its own derivation."""
    lengths = [p[1] for p in pairs]
    total = sum(lengths)

    # tiles/utils.py takes tile_size_bp directly
    for tile_size in (256, 1024, 4096):
        assert quadtree_depth(total, tile_size) == quadtree_depth_utils(
            lengths, tile_size
        )

    # tiles/bigwig.py hardcodes TILE_SIZE
    assert quadtree_depth(total, BIGWIG_TILE_SIZE) == quadtree_depth_bigwig(lengths)

    # tiles/cooler.py uses 256 * binsize
    for binsize in (1000, 5000, 100000):
        assert quadtree_depth(total, 256 * binsize) == quadtree_depth_cooler(
            lengths, binsize
        )


# --- natsorted --------------------------------------------------------------


def test_natsorted_orders_numerically_then_xym():
    names = ["chr10", "chr2", "chrM", "chr1", "chrX", "chrY"]
    assert natsorted(names) == ["chr1", "chr2", "chr10", "chrX", "chrY", "chrM"]


def test_natsorted_puts_underscored_contigs_last():
    """Unplaced/alt contigs sort after plain names, and among themselves are
    compared on the segment after the first underscore."""
    names = ["chr1_KI270706v1_random", "chr2", "chr1", "chrUn_GL000195v1"]
    result = natsorted(names)

    assert result[:2] == ["chr1", "chr2"]
    assert set(result[2:]) == {"chr1_KI270706v1_random", "chrUn_GL000195v1"}


NAME_SETS = {
    "plain": ["chr1", "chr10", "chr2", "chrX", "chrY", "chrM", "chr22"],
    "unprefixed": ["1", "10", "2", "X", "Y", "MT", "22"],
    "underscored": ["chr1_KI270706v1_random", "chr1", "chrUn_GL000195v1", "chr2_x"],
    "mixed": ["scaffold_9", "scaffold_10", "contig1", "contig02", "CHR1", "chr1"],
}


@pytest.mark.parametrize("label", sorted(NAME_SETS))
@pytest.mark.parametrize("seed", range(8))
def test_natsorted_matches_legacy(label, seed):
    """Order defines the absolute coordinate space, so it must not shift.

    Checked from shuffled input, since a comparator bug can hide when the input
    is already sorted.
    """
    names = list(NAME_SETS[label])
    random.Random(seed).shuffle(names)

    assert natsorted(names) == natsorted_utils(names) == natsorted_bigwig(names)


@pytest.mark.parametrize("label,pairs", _chromsizes_cases())
def test_natsorted_matches_legacy_on_real_assemblies(label, pairs):
    names = [p[0] for p in pairs]
    for seed in range(8):
        shuffled = list(names)
        random.Random(seed).shuffle(shuffled)
        assert (
            natsorted(shuffled)
            == natsorted_utils(shuffled)
            == natsorted_bigwig(shuffled)
        )
