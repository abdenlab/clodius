"""Tests for clodius.core.coords.

The differential tests pin the new implementation against the three legacy
``abs2genomic`` functions it is meant to replace. They should stay in place
until the last caller has migrated, at which point the legacy functions (and
these comparisons) can be deleted.
"""

import math
import os.path as op

import pandas as pd
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from clodius.chromosomes import chromsizes_as_array
from clodius.core.coords import (
    Chromsizes,
    GenomicRange,
    natsorted,
    reconcile_sequential,
)
from clodius.tiles.bigwig import abs2genomic as abs2genomic_bigwig
from clodius.tiles.bigwig import natsorted as natsorted_bigwig
from clodius.tiles.multivec import abs2genomic as abs2genomic_multivec
from clodius.tiles.utils import abs2genomic as abs2genomic_utils
from clodius.tiles.utils import natsorted as natsorted_utils

TINY = [["c1", 100], ["c2", 200], ["c3", 50]]

NAME_SETS = {
    "plain": ["chr1", "chr10", "chr2", "chrX", "chrY", "chrM", "chr22"],
    "unprefixed": ["1", "10", "2", "X", "Y", "MT", "22"],
    "underscored": [
        "chr1_KI270706v1_random",
        "chr1",
        "chrUn_GL000195v1",
        "chr2_x",
    ],
    "mixed": [
        "scaffold_9",
        "scaffold_10",
        "contig1",
        "contig02",
        "CHR1",
        "chr1",
    ],
}

# Hypothesis and pytest parametrization both bind function-scoped fixtures,
# which trips a health check that does not apply here: the parameters are
# plain values, not stateful fixtures.
PROPERTY = settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def as_tuples(intervals):
    return [(iv.cid, iv.start, iv.end) for iv in intervals]


def legacy_as_tuples(gen):
    """Normalize legacy output, which leaks numpy scalars and floats."""
    return [(int(cid), int(start), int(end)) for cid, start, end in gen]


def chromsizes_cases():
    """``(label, pairs)`` for the tiny synthetic genome plus any real ones."""
    cases = [("tiny", TINY)]
    for name in ("chm13v1.chrom.sizes", "hg38.chrom.sizes"):
        path = op.join("data", name)
        if op.exists(path):
            cases.append((name, chromsizes_as_array(path)))
    return cases


def outward_bin_counts(intervals, binsize):
    """Bins each interval yields from an outward-rounding fetcher.

    ``ceil(end / b) - floor(start / b)``, which is what multivec's
    ``bin_slice`` and cooler's ``_bin_count`` compute, and what over-supplies
    by up to nearly two bins per chunk.
    """
    counts = []
    for iv in intervals:
        if iv.is_out_of_bounds:
            counts.append(math.ceil((iv.end - iv.start) / binsize))
        else:
            counts.append(
                math.ceil(iv.end / binsize) - (iv.start // binsize)
            )
    return counts


class TestChromsizes:
    """Behavior of the coordinate system that defines the tiling axis."""

    def test_offsets_should_return_int_cumulative_starts(self):
        """Test that offsets are the running sum of preceding lengths.

        Given:
            A three-chromosome coordinate system.
        When:
            Its offsets are read.
        Then:
            It should map each name to the int sum of the lengths before it.
        """
        # Arrange
        cs = Chromsizes.from_pairs(TINY)

        # Act
        offsets = cs.offsets

        # Assert
        assert offsets == {"c1": 0, "c2": 100, "c3": 300}
        assert all(isinstance(v, int) for v in offsets.values())

    def test_invert_should_return_one_range_when_span_is_inside_a_chromosome(
        self,
    ):
        """Test inversion of a span that crosses no boundary.

        Given:
            A span lying wholly inside the first chromosome.
        When:
            The span is inverted.
        Then:
            It should yield a single range on that chromosome.
        """
        # Arrange
        cs = Chromsizes.from_pairs(TINY)

        # Act
        intervals = cs.invert((10, 50))

        # Assert
        assert as_tuples(intervals) == [(0, 10, 50)]

    def test_invert_should_split_when_span_crosses_a_boundary(self):
        """Test inversion of a span crossing a chromosome boundary.

        Given:
            A span starting in the first chromosome and ending in the second.
        When:
            The span is inverted.
        Then:
            It should yield one range per chromosome touched.
        """
        # Arrange
        cs = Chromsizes.from_pairs(TINY)

        # Act
        intervals = cs.invert((90, 150))

        # Assert
        assert as_tuples(intervals) == [(0, 90, 100), (1, 0, 50)]

    def test_invert_should_start_the_next_chromosome_when_start_is_a_boundary(
        self,
    ):
        """Test the half-open convention at an exact boundary.

        Given:
            A span starting exactly on a chromosome boundary.
        When:
            The span is inverted.
        Then:
            It should place that position on the following chromosome.
        """
        # Arrange
        cs = Chromsizes.from_pairs(TINY)

        # Act
        intervals = cs.invert((100, 150))

        # Assert
        assert as_tuples(intervals) == [(1, 0, 50)]

    def test_invert_should_truncate_to_int_when_span_is_fractional(self):
        """Test that fractional spans yield integral coordinates.

        Given:
            A fractional span, as tabix, vcf and bedfile produce from
            ``x * max_width / 2**z``.
        When:
            The span is inverted.
        Then:
            It should yield int coordinates, truncated toward zero.
        """
        # Arrange
        cs = Chromsizes.from_pairs(TINY)

        # Act
        intervals = list(cs.invert((90.5, 150.7)))

        # Assert
        assert as_tuples(intervals) == [(0, 90, 100), (1, 0, 50)]
        assert all(
            isinstance(iv.start, int) and isinstance(iv.end, int)
            for iv in intervals
        )

    def test_invert_should_keep_the_tail_when_span_passes_the_last_chromosome(
        self,
    ):
        """Test that the out-of-genome tail survives inversion.

        Given:
            A span extending past the end of the last chromosome.
        When:
            The span is inverted.
        Then:
            It should yield the tail as a trailing range rather than drop it.
        """
        # Arrange
        cs = Chromsizes.from_pairs(TINY)

        # Act
        intervals = list(cs.invert((250, 470)))

        # Assert
        assert as_tuples(intervals) == [(1, 150, 200), (2, 0, 50), (3, 0, 120)]

    def test_invert_should_flag_the_tail_when_span_passes_the_last_chromosome(
        self,
    ):
        """Test that the out-of-genome tail is marked out of bounds.

        Given:
            A span extending past the end of the last chromosome.
        When:
            The span is inverted.
        Then:
            It should flag only the trailing range, which is what fills the
            NaN bins of the final tiles.
        """
        # Arrange
        cs = Chromsizes.from_pairs(TINY)

        # Act
        intervals = list(cs.invert((250, 470)))

        # Assert
        assert [iv.is_out_of_bounds for iv in intervals] == [
            False,
            False,
            True,
        ]
        assert intervals[-1].name is None
        assert intervals[-1].cid == len(cs)

    def test_invert_should_yield_a_zero_length_tail_when_span_is_the_genome(
        self,
    ):
        """Test the boundary case of a span covering exactly the genome.

        Given:
            A span running from zero to the total genome length.
        When:
            The span is inverted.
        Then:
            It should end with a zero-length out-of-bounds range.
        """
        # Arrange
        cs = Chromsizes.from_pairs(TINY)

        # Act
        intervals = list(cs.invert((0, cs.total_length)))

        # Assert
        assert intervals[-1].is_out_of_bounds
        assert (intervals[-1].start, intervals[-1].end) == (0, 0)
        assert not any(iv.is_out_of_bounds for iv in intervals[:-1])

    def test_invert_should_return_one_range_when_span_is_entirely_past_the_end(
        self,
    ):
        """Test inversion of a span with no in-genome part at all.

        Given:
            A span lying wholly past the end of the last chromosome.
        When:
            The span is inverted.
        Then:
            It should yield a single out-of-bounds range.
        """
        # Arrange
        cs = Chromsizes.from_pairs(TINY)

        # Act
        intervals = list(cs.invert((360, 400)))

        # Assert
        assert len(intervals) == 1
        assert intervals[0].is_out_of_bounds
        assert (intervals[0].start, intervals[0].end) == (10, 50)

    def test_invert_should_raise_when_start_is_negative(self):
        """Test rejection of a negative start position.

        Given:
            A span with a negative start, which legacy indexed backwards off
            the end of the offsets array instead.
        When:
            The span is inverted.
        Then:
            It should raise ValueError.
        """
        # Arrange
        cs = Chromsizes.from_pairs(TINY)

        # Act & assert
        with pytest.raises(ValueError, match="negative start"):
            list(cs.invert((-5, 100)))

    def test_invert_should_raise_when_end_precedes_start(self):
        """Test rejection of a reversed span.

        Given:
            A span whose end lies before its start.
        When:
            The span is inverted.
        Then:
            It should raise ValueError.
        """
        # Arrange
        cs = Chromsizes.from_pairs(TINY)

        # Act & assert
        with pytest.raises(ValueError, match="precedes start"):
            list(cs.invert((200, 100)))

    @pytest.mark.parametrize("label,pairs", chromsizes_cases())
    @PROPERTY
    @example(start=0, width=0)
    @example(start=0, width=1)
    @given(
        start=st.integers(min_value=0, max_value=10**10),
        width=st.integers(min_value=0, max_value=10**10),
    )
    def test_invert_should_match_legacy_implementations(
        self, label, pairs, start, width
    ):
        """Test the differential invariant against all three legacy functions.

        Given:
            Any non-negative half-open span over a real or synthetic genome,
            including spans past the end of it.
        When:
            The span is inverted.
        Then:
            It should yield the same intervals as ``abs2genomic`` in
            ``tiles.utils``, ``tiles.multivec`` and ``tiles.bigwig``, once
            both sides are int-cast.
        """
        # Arrange
        cs = Chromsizes.from_pairs(pairs)
        lengths = list(cs.lengths)
        series = pd.Series(lengths, index=list(cs.names))
        end = start + width

        # Act
        result = as_tuples(cs.invert((start, end)))

        # Assert
        assert result == legacy_as_tuples(
            abs2genomic_utils(lengths, start, end)
        )
        assert result == legacy_as_tuples(
            abs2genomic_multivec(lengths, start, end)
        )
        assert result == legacy_as_tuples(
            abs2genomic_bigwig(series, start, end)
        )

    @pytest.mark.parametrize("label,pairs", chromsizes_cases())
    def test_invert_should_match_legacy_at_the_genome_boundaries(
        self, label, pairs
    ):
        """Test the differential invariant at the spans that matter most.

        Given:
            The whole genome, twice the genome, and the span starting exactly
            at its end -- the cases the shrinker is least likely to find.
        When:
            Each span is inverted.
        Then:
            It should agree with all three legacy implementations.
        """
        # Arrange
        cs = Chromsizes.from_pairs(pairs)
        lengths = list(cs.lengths)
        series = pd.Series(lengths, index=list(cs.names))
        total = cs.total_length
        spans = [(0, total), (0, total * 2), (total, total + 1000), (0, 1)]

        for start, end in spans:
            # Act
            result = as_tuples(cs.invert((start, end)))

            # Assert
            assert result == legacy_as_tuples(
                abs2genomic_utils(lengths, start, end)
            ), f"utils differs on ({start}, {end})"
            assert result == legacy_as_tuples(
                abs2genomic_multivec(lengths, start, end)
            ), f"multivec differs on ({start}, {end})"
            assert result == legacy_as_tuples(
                abs2genomic_bigwig(series, start, end)
            ), f"bigwig differs on ({start}, {end})"


def test___hash___should_treat_equal_ranges_as_one():
    """Test that a genomic range is usable as a set member and cache key.

    Given:
        Two ranges constructed with identical fields.
    When:
        Both are put in a set.
    Then:
        It should hold one element.
    """
    # Act
    unique = {GenomicRange(0, "c1", 0, 10), GenomicRange(0, "c1", 0, 10)}

    # Assert
    assert len(unique) == 1


def test_natsorted_should_order_numerically_then_x_y_m():
    """Test the conventional chromosome ordering.

    Given:
        Numeric and sex/mitochondrial chromosome names out of order.
    When:
        They are natsorted.
    Then:
        It should order numerically, then X, Y, M.
    """
    # Arrange
    names = ["chr10", "chr2", "chrM", "chr1", "chrX", "chrY"]

    # Act
    result = natsorted(names)

    # Assert
    assert result == ["chr1", "chr2", "chr10", "chrX", "chrY", "chrM"]


def test_natsorted_should_sort_underscored_contigs_last():
    """Test placement of unplaced and alt contigs.

    Given:
        A mix of plain names and names containing an underscore.
    When:
        They are natsorted.
    Then:
        It should place every underscored name after every plain one.
    """
    # Arrange
    names = ["chr1_KI270706v1_random", "chr2", "chr1", "chrUn_GL000195v1"]

    # Act
    result = natsorted(names)

    # Assert
    assert result[:2] == ["chr1", "chr2"]
    assert set(result[2:]) == {"chr1_KI270706v1_random", "chrUn_GL000195v1"}


@pytest.mark.parametrize("label", sorted(NAME_SETS))
@PROPERTY
@given(data=st.data())
def test_natsorted_should_match_legacy_when_input_is_shuffled(label, data):
    """Test the differential invariant on name ordering.

    Given:
        Any permutation of a set of chromosome names, since a comparator bug
        can hide when the input is already sorted.
    When:
        The names are natsorted.
    Then:
        It should agree with both legacy implementations, because the order
        defines the absolute coordinate space and must not shift.
    """
    # Arrange
    names = data.draw(st.permutations(NAME_SETS[label]))

    # Act
    result = natsorted(names)

    # Assert
    assert result == natsorted_utils(names) == natsorted_bigwig(names)


@pytest.mark.parametrize("label,pairs", chromsizes_cases())
@PROPERTY
@given(data=st.data())
def test_natsorted_should_match_legacy_on_real_assemblies(label, pairs, data):
    """Test the differential invariant on real assembly name sets.

    Given:
        Any permutation of a real assembly's chromosome names.
    When:
        The names are natsorted.
    Then:
        It should agree with both legacy implementations.
    """
    # Arrange
    names = data.draw(st.permutations([p[0] for p in pairs]))

    # Act
    result = natsorted(names)

    # Assert
    assert result == natsorted_utils(names) == natsorted_bigwig(names)


def test_reconcile_sequential_should_concatenate_when_no_drift_accumulates():
    """Test the base case where the segmented grid already fits the lattice.

    Given:
        Chunks whose bin counts exactly cover their spans.
    When:
        They are reconciled.
    Then:
        It should return every bin in order.
    """
    # Arrange
    chunks = [([1, 2], 20.0), ([3, 4], 20.0)]

    # Act
    out = reconcile_sequential(chunks, 10.0, expected_bins=4)

    # Assert
    assert list(out) == [1, 2, 3, 4]


def test_reconcile_sequential_should_raise_when_the_count_is_wrong():
    """Test that a miscount is an error rather than a silent truncation.

    Given:
        Chunks that supply more bins than the caller expects.
    When:
        They are reconciled against an expected bin count.
    Then:
        It should raise ValueError rather than clamp.
    """
    # Arrange
    chunks = [([1, 2, 3], 30.0)]

    # Act & assert
    with pytest.raises(ValueError, match="expected 2"):
        reconcile_sequential(chunks, 10.0, expected_bins=2)


@pytest.mark.parametrize("threshold", [0.5, 1.0])
@pytest.mark.parametrize("binsize", [1, 3, 17, 101, 1024, 100_000])
@PROPERTY
@given(
    # Both regimes matter. Chromosomes far longer than a tile give one or two
    # chunks; chromosomes shorter than a tile give dozens, which is where the
    # drift budget is under the most pressure.
    lengths=st.lists(
        st.integers(min_value=1, max_value=250_000_000),
        min_size=1,
        max_size=60,
    )
    | st.lists(
        st.integers(min_value=1, max_value=5_000), min_size=1, max_size=60
    ),
    tile_size=st.sampled_from([16, 256, 1024]),
    x=st.integers(min_value=0, max_value=4096),
)
def test_reconcile_sequential_should_produce_expected_bins_for_outward_fetchers(
    threshold, binsize, lengths, tile_size, x
):
    """Test that outward-rounding fetchers still land on the lattice.

    Given:
        Any genome, binsize and tile position, with per-chunk bin counts
        computed the way multivec and cooler compute them --
        ``ceil(end / b) - floor(start / b)``, which over-supplies by up to
        nearly two bins per chunk because chromosome offsets are not
        multiples of the binsize.
    When:
        The chunks are reconciled against the tile's bin count.
    Then:
        It should return exactly ``tile_size`` bins, not raise.
    """
    # Arrange
    cs = Chromsizes(
        tuple(f"c{i}" for i in range(len(lengths))), tuple(lengths)
    )
    span = binsize * tile_size
    intervals = list(cs.invert((x * span, (x + 1) * span)))
    counts = outward_bin_counts(intervals, binsize)
    chunks = [
        (list(range(n)), iv.end - iv.start)
        for iv, n in zip(intervals, counts)
    ]

    # Act
    out = reconcile_sequential(
        chunks, binsize, expected_bins=tile_size, threshold=threshold
    )

    # Assert
    assert len(out) == tile_size
