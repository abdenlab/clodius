"""Tests for clodius.core.coords.

The differential tests pin the new implementation against the three legacy
``abs2genomic`` functions it is meant to replace. They should stay in place
until the last caller has migrated, at which point the legacy functions (and
these comparisons) can be deleted.
"""

import io
import math
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from clodius.core.coords import (
    Canvas,
    Chromsizes,
    GenomicRange,
    bin_count,
    natsorted,
    reconcile_sequential,
    reconcile_sequential_2d,
)
from clodius.core.errors import TileOutOfBounds, TilesetUnavailable
from clodius.tiles.bigwig import abs2genomic as abs2genomic_bigwig
from clodius.tiles.bigwig import natsorted as natsorted_bigwig
from clodius.tiles.multivec import abs2genomic as abs2genomic_multivec
from clodius.tiles.utils import abs2genomic as abs2genomic_utils
from clodius.tiles.utils import natsorted as natsorted_utils

from ..harness import strategies
from ..harness.genome import MINIMAL_CHROMSIZES

TINY = MINIMAL_CHROMSIZES

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
    """``(label, pairs)`` for the tiny synthetic genome plus two real ones.

    Sourced from bioframe's bundled assembly tables rather than ``data/``.
    The previous form guarded a CWD-relative path with ``op.exists``, so on a
    fresh clone -- or simply when pytest ran from another directory -- the
    real-assembly cases vanished and every differential test below silently
    degraded to a 350 bp genome while still reporting green.
    """
    cases = [("tiny", TINY)]
    for label, kwargs in (
        ("hg38", {"roles": "all"}),  # 194 contigs, incl. unplaced and random
        ("hs1", {}),  # 25 contigs, T2T
    ):
        cases.append((label, Chromsizes.from_assembly(label, **kwargs).to_pairs()))
    return cases


def outward_bin_counts(intervals, binsize):
    """Bins each interval yields from an outward-rounding fetcher.

    ``ceil(end / b) - floor(start / b)``, which is what multivec's
    ``bin_slice`` and :func:`~clodius.core.coords.bin_count` compute, and
    what over-supplies by up to nearly two bins per chunk.

    Not a call to ``bin_count`` itself. The two genuinely differ on an
    out-of-bounds interval: this takes the naive ``ceil(span / b)``, which is
    what ``clodius.tiles_v2.multivec`` pads with, while ``bin_count`` rounds
    outward there as everywhere else. The 1D and 2D reconcilers are fed by
    different conventions and both are correct for their caller. Reusing
    ``bin_count`` here would silently change which convention these tests
    describe.
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


class TestChromsizesConstruction:
    """The constructors and the views derived from them."""

    def test___init___should_coerce_lengths_to_int(self):
        """Test that lengths are normalized on the way in.

        Given:
            Names and lengths where one length is a string and one a numpy
            integer, as a file reader and a dataframe respectively produce.
        When:
            A coordinate system is constructed.
        Then:
            It should store every length as a Python int, so downstream
            arithmetic never mixes types.
        """
        # Act
        cs = Chromsizes(("c1", "c2"), ("100", np.int64(200)))

        # Assert
        assert cs.lengths == (100, 200)
        assert all(type(n) is int for n in cs.lengths)

    def test___init___should_raise_when_the_counts_disagree(self):
        """Test rejection of mismatched sequences.

        Given:
            Two names but only one length.
        When:
            A coordinate system is constructed.
        Then:
            It should raise reporting both counts.
        """
        # Act & assert
        with pytest.raises(ValueError, match="2 names but 1 lengths"):
            Chromsizes(("c1", "c2"), (100,))

    def test___len___should_count_the_chromosomes(self):
        """Test the chromosome count.

        Given:
            A three-chromosome coordinate system.
        When:
            Its length is taken.
        Then:
            It should be three.
        """
        # Act & assert
        assert len(Chromsizes.from_pairs(TINY)) == 3

    def test___repr___should_name_the_count_and_total(self):
        """Test the debugging representation.

        Given:
            A coordinate system totalling more than a thousand bases, so the
            thousands separator is actually exercised.
        When:
            It is rendered.
        Then:
            It should name the chromosome count and the separated total.
        """
        # Arrange
        cs = Chromsizes.from_pairs([["c1", 1000], ["c2", 2500]])

        # Act & assert
        assert repr(cs) == "<Chromsizes 2 chroms, 3,500 bp>"

    def test_total_length_should_sum_every_chromosome(self):
        """Test the genome length.

        Given:
            The three-chromosome coordinate system.
        When:
            Its total length is read.
        Then:
            It should be the sum of the lengths, as an int.
        """
        # Act
        total = Chromsizes.from_pairs(TINY).total_length

        # Assert
        assert total == 350
        assert type(total) is int

    def test_total_length_should_be_zero_when_there_are_no_chromosomes(self):
        """Test the empty genome.

        Given:
            A coordinate system with no chromosomes.
        When:
            Its length, total and offsets are read.
        Then:
            They should be zero, zero and empty rather than raising.
        """
        # Arrange
        cs = Chromsizes.from_pairs([])

        # Act & assert
        assert (len(cs), cs.total_length, cs.offsets) == (0, 0, {})

    def test_from_pairs_should_preserve_the_given_order(self):
        """Test that this constructor does not sort.

        Given:
            Pairs deliberately out of natural order, with string lengths.
        When:
            A coordinate system is built from them.
        Then:
            It should keep the given order and coerce the lengths, since the
            order defines the tiling coordinate space.
        """
        # Act
        cs = Chromsizes.from_pairs([["c10", "5"], ["c2", "7"]])

        # Assert
        assert (cs.names, cs.lengths) == (("c10", "c2"), (5, 7))

    def test_from_pairs_should_accept_an_exhausted_iterator(self):
        """Test the emptiness branch on a non-sized argument.

        Given:
            An exhausted iterator rather than an empty list.
        When:
            A coordinate system is built from it.
        Then:
            It should produce an empty coordinate system. Branching on the
            argument's truthiness would make this raise where an empty list
            succeeds.
        """
        # Act & assert
        assert len(Chromsizes.from_pairs(iter([]))) == 0

    def test_from_series_should_read_names_from_the_index(self):
        """Test construction from a pandas Series.

        Given:
            A series of lengths indexed by chromosome name.
        When:
            A coordinate system is built from it.
        Then:
            It should take names from the index and lengths from the values,
            in series order.
        """
        # Arrange
        series = pd.Series({"c2": 200, "c1": 100})

        # Act & assert
        assert Chromsizes.from_series(series).to_pairs() == [
            ["c2", 200],
            ["c1", 100],
        ]

    def test_to_pairs_should_round_trip_through_from_pairs(self):
        """Test the pairs round trip.

        Given:
            A three-chromosome coordinate system.
        When:
            It is converted to pairs and rebuilt.
        Then:
            It should produce identical names and lengths.
        """
        # Arrange
        cs = Chromsizes.from_pairs(TINY)

        # Act
        rebuilt = Chromsizes.from_pairs(cs.to_pairs())

        # Assert
        assert (rebuilt.names, rebuilt.lengths) == (cs.names, cs.lengths)

    def test_to_series_should_round_trip_through_from_series(self):
        """Test the series round trip.

        Given:
            A three-chromosome coordinate system.
        When:
            It is converted to a series and rebuilt.
        Then:
            It should produce identical names and lengths.
        """
        # Arrange
        cs = Chromsizes.from_pairs(TINY)

        # Act
        rebuilt = Chromsizes.from_series(cs.to_series())

        # Assert
        assert (rebuilt.names, rebuilt.lengths) == (cs.names, cs.lengths)

    def test_from_assembly_should_load_a_bundled_assembly(self):
        """Test construction from a named assembly.

        Given:
            The yeast assembly, whose table ships with bioframe and needs no
            network.
        When:
            A coordinate system is built from it.
        Then:
            It should carry the assembly's chromosomes and total length.
        """
        # Act
        cs = Chromsizes.from_assembly("sacCer3")

        # Assert
        assert cs.names[0] == "chrI"
        assert cs.total_length == 12157105

    def test_from_assembly_should_widen_the_genome_when_roles_are_all(self):
        """Test the roles argument.

        Given:
            A human assembly requested with default roles and again with all.
        When:
            A coordinate system is built each way.
        Then:
            The second should carry many more contigs, including the unplaced
            ones a default request omits.
        """
        # Act
        default = Chromsizes.from_assembly("hg38")
        every = Chromsizes.from_assembly("hg38", roles="all")

        # Assert
        assert len(every) > len(default)
        assert any(n.startswith("chrUn_") for n in every.names)

    def test_from_assembly_should_raise_when_the_assembly_is_unknown(self):
        """Test rejection of an unknown assembly name.

        Given:
            An assembly name bioframe does not ship.
        When:
            A coordinate system is built from it.
        Then:
            It should raise.
        """
        # Act & assert
        with pytest.raises(ValueError):
            Chromsizes.from_assembly("not-an-assembly")

    def test_from_file_should_read_a_two_column_table(self):
        """Test the ordinary file path.

        Given:
            A two-column tab-separated sizes table.
        When:
            A coordinate system is built from it.
        Then:
            It should carry every row with its length as an int.
        """
        # Act & assert
        assert Chromsizes.from_file(
            io.StringIO("c1\t100\nc2\t200\n")
        ).to_pairs() == [["c1", 100], ["c2", 200]]

    def test_from_file_should_accept_a_path(self, tmp_path):
        """Test the path branch alongside the handle branch.

        Given:
            The same table written to a file.
        When:
            A coordinate system is built from the path.
        Then:
            It should equal the one built from a handle, so both forms of the
            argument behave alike.
        """
        # Arrange
        path = tmp_path / "sizes.chrom.sizes"
        path.write_text("c1\t100\nc2\t200\n")

        # Act & assert
        assert Chromsizes.from_file(str(path)).to_pairs() == [
            ["c1", 100],
            ["c2", 200],
        ]

    def test_from_file_should_apply_natural_genomic_ordering(self):
        """Test that this constructor sorts, unlike from_assembly.

        Given:
            A sizes table whose rows are deliberately out of order.
        When:
            A coordinate system is built from it.
        Then:
            It should reorder them naturally. Note this diverges from
            from_assembly, which preserves provider order -- so the same
            genome loaded the two ways yields different coordinate spaces and
            therefore different tiles.
        """
        # Act
        cs = Chromsizes.from_file(
            io.StringIO("chr2\t200\nchr1\t100\nchrM\t50\nchr1_alt\t7\n")
        )

        # Assert
        assert cs.names == ("chr1", "chr2", "chrM", "chr1_alt")

    @pytest.mark.pinned
    def test_from_file_should_mis_order_roman_numeral_assemblies(self):
        """Test the documented limit of the natural ordering.

        Given:
            A sizes table using Roman numerals, as yeast assemblies do.
        When:
            A coordinate system is built from it.
        Then:
            It should place chrV first and chrXVI before chrX, because the
            ordering keys on a digit run and a letter rule rather than parsing
            numerals. Pinned as a limitation: a Roman-numeral genome must not
            be routed through this constructor.
        """
        # Act
        cs = Chromsizes.from_file(
            io.StringIO("chrX\t10\nchrXVI\t20\nchrV\t30\n")
        )

        # Assert
        assert cs.names == ("chrV", "chrXVI", "chrX")

    def test_from_file_should_keep_non_standard_contigs(self):
        """Test that nothing is silently discarded.

        Given:
            A sizes table carrying a scaffold and a viral contig that a
            filtering reader would drop.
        When:
            A coordinate system is built from it.
        Then:
            It should retain every row, since dropping one shortens the
            tiling axis and moves every tile boundary after it.
        """
        # Act & assert
        assert (
            len(
                Chromsizes.from_file(
                    io.StringIO("chr1\t100\nscaffold_7\t20\nchrEBV\t5\n")
                )
            )
            == 3
        )

    def test_from_file_should_return_an_empty_system_for_an_empty_file(self):
        """Test the empty table.

        Given:
            An empty sizes file.
        When:
            A coordinate system is built from it.
        Then:
            It should be empty rather than raising.
        """
        # Act & assert
        assert len(Chromsizes.from_file(io.StringIO(""))) == 0

    def test_from_file_should_raise_when_a_contig_is_duplicated(self):
        """Test rejection of a repeated chromosome name.

        Given:
            A sizes table naming the same chromosome twice with different
            lengths.
        When:
            A coordinate system is built from it.
        Then:
            It should raise a tile error naming the duplicate. Collapsing it
            would keep one length silently, leaving the genome short by the
            other -- and total length defines the tiling axis, so that
            produces wrong tiles rather than an error.
        """
        # Act & assert
        with pytest.raises(TilesetUnavailable, match="duplicate chromosome"):
            Chromsizes.from_file(
                io.StringIO("chr1\t100\nchr2\t200\nchr1\t999\n")
            )

    def test_from_file_should_raise_a_tile_error_when_the_table_is_malformed(
        self,
    ):
        """Test that a parser failure is translated.

        Given:
            A sizes table with only one column.
        When:
            A coordinate system is built from it.
        Then:
            It should raise a tile error rather than leaking the reader's own
            exception type, which the server boundary cannot render.
        """
        # Act & assert
        with pytest.raises(TilesetUnavailable, match="could not read"):
            Chromsizes.from_file(io.StringIO("chr1\n"))


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

    @pytest.mark.parametrize(
        "span",
        [
            lambda total: (0, total),
            lambda total: (0, total * 2),
            lambda total: (total, total + 1000),
            lambda total: (0, 1),
        ],
        ids=["whole-genome", "twice-the-genome", "past-the-end", "first-base"],
    )
    @pytest.mark.parametrize("label,pairs", chromsizes_cases())
    def test_invert_should_match_legacy_at_the_genome_boundaries(
        self, label, pairs, span
    ):
        """Test the differential invariant at the spans that matter most.

        Given:
            The whole genome, twice the genome, the span starting exactly at
            its end, and the first base -- the cases the shrinker is least
            likely to find. Each is its own case rather than a loop, so a
            divergence reports which span diverged instead of one red test
            with an f-string to read.
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
        start, end = span(total)

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


    def test_invert_should_yield_a_zero_width_range_inside_a_chromosome(self):
        """Test a degenerate span away from any boundary.

        Given:
            A zero-width span in the middle of the first chromosome.
        When:
            It is inverted.
        Then:
            It should yield one range with equal start and end.
        """
        # Act & assert
        assert as_tuples(Chromsizes.from_pairs(TINY).invert((50, 50))) == [
            (0, 50, 50)
        ]

    def test_invert_should_open_the_next_chromosome_on_a_boundary(self):
        """Test a degenerate span landing exactly on a boundary.

        Given:
            A zero-width span at the boundary between two chromosomes.
        When:
            It is inverted.
        Then:
            It should yield one zero-width range at the start of the following
            chromosome, matching the half-open convention.
        """
        # Act & assert
        assert as_tuples(Chromsizes.from_pairs(TINY).invert((100, 100))) == [
            (1, 0, 0)
        ]

    def test_invert_should_flag_a_zero_width_span_past_the_genome(self):
        """Test a degenerate span beyond the last chromosome.

        Given:
            A zero-width span well past the end of the genome.
        When:
            It is inverted.
        Then:
            It should yield one out-of-bounds range measured from the genome
            end.
        """
        # Arrange
        cs = Chromsizes.from_pairs(TINY)

        # Act
        result = list(cs.invert((400, 400)))

        # Assert
        assert [r.is_out_of_bounds for r in result] == [True]
        assert as_tuples(result) == [(3, 50, 50)]

    def test_invert_should_work_on_an_empty_coordinate_system(self):
        """Test inversion when the genome has no chromosomes.

        Given:
            An empty coordinate system, which reading an empty sizes table
            produces.
        When:
            Any span is inverted.
        Then:
            It should yield a single out-of-bounds range rather than raising.
            The cumulative-sum array must be integral for this to work: an
            empty float cumsum could not be narrowed and the call failed
            outright.
        """
        # Act
        result = list(Chromsizes.from_pairs([]).invert((0, 10)))

        # Assert
        assert [r.is_out_of_bounds for r in result] == [True]
        assert as_tuples(result) == [(0, 0, 10)]

    @given(data=st.data())
    @PROPERTY
    def test_invert_should_preserve_the_total_width_of_the_span(self, data):
        """Test that inversion loses no base pairs.

        Given:
            Any genome and any non-negative span, including spans wider than
            the genome and spans starting past its end.
        When:
            The span is inverted.
        Then:
            The widths of the yielded ranges should sum to exactly the span's
            width, so no base is dropped or double-counted at a boundary.
        """
        # Arrange
        pairs = data.draw(
            st.lists(
                st.integers(min_value=1, max_value=10_000),
                min_size=1,
                max_size=8,
            ).map(lambda ls: [[f"c{i}", n] for i, n in enumerate(ls)])
        )
        cs = Chromsizes.from_pairs(pairs)
        total = cs.total_length
        start = data.draw(st.integers(min_value=0, max_value=total * 2))
        end = data.draw(st.integers(min_value=start, max_value=total * 2 + 1))

        # Act
        result = list(cs.invert((start, end)))

        # Assert
        assert sum(r.end - r.start for r in result) == end - start

    @given(data=st.data())
    @PROPERTY
    def test_invert_should_yield_a_gapless_chain_of_ranges(self, data):
        """Test that the yielded ranges tile the span without gaps.

        Given:
            Any genome and any span, including one ending exactly at the
            genome's end -- which yields a trailing zero-width out-of-bounds
            range that has no chromosome to transform against.
        When:
            The span is inverted and each in-bounds range transformed back to
            absolute coordinates.
        Then:
            The in-bounds ranges should form a contiguous chain starting at
            the span's start and ending where the span ends or the genome
            does, whichever comes first, with consecutive chromosome ids.
        """
        # Arrange
        pairs = data.draw(
            st.lists(
                st.integers(min_value=1, max_value=10_000),
                min_size=1,
                max_size=8,
            ).map(lambda ls: [[f"c{i}", n] for i, n in enumerate(ls)])
        )
        cs = Chromsizes.from_pairs(pairs)
        total = cs.total_length
        start = data.draw(st.integers(min_value=0, max_value=total - 1))
        end = data.draw(st.integers(min_value=start, max_value=total))

        # Act
        result = list(cs.invert((start, end)))

        # Assert
        absolute = [cs.transform(r) for r in result if not r.is_out_of_bounds]
        assert absolute[0][0] == start
        assert absolute[-1][1] == min(end, total)
        for (_, prev_end), (next_start, _) in zip(absolute, absolute[1:]):
            assert prev_end == next_start
        assert [r.cid for r in result] == list(
            range(result[0].cid, result[0].cid + len(result))
        )


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


@pytest.mark.pinned
@pytest.mark.xfail(
    strict=True,
    reason="_natcmp is not antisymmetric across naming conventions: "
    "cmp('chr1','1') and cmp('1','chr1') both return 1 (#TBD)",
)
def test_natsorted_should_order_a_prefixed_name_against_its_bare_spelling():
    """Test the comparator across the two naming conventions.

    Given:
        A prefixed name and its bare spelling, in each input order.
    When:
        Each list is natsorted.
    Then:
        Both should give the same list, as they do for any two names within
        one convention. They do not: ``_natcmp`` reports *both* orderings as
        "greater", so it is not a valid comparator here and ``sorted`` simply
        keeps whichever came first.

        No real assembly mixes ``chr1`` with a bare ``1``, which is why the
        ordering properties draw from one convention at a time -- but a
        comparator that contradicts itself is a latent hazard wherever contig
        names are merged from two sources, and merging is exactly what
        ``Chromsizes.from_file`` does.
    """
    # Act
    forward = natsorted(["chr1", "1"])
    backward = natsorted(["1", "chr1"])

    # Assert
    assert forward == backward


@PROPERTY
@given(names=strategies.conventional_chrom_names())
def test_natsorted_should_be_invariant_under_the_input_order(names):
    """Test that the ordering depends only on the set of names.

    Given:
        Any shuffled list of distinct conventional chromosome names.
    When:
        It is natsorted, and its reverse is natsorted.
    Then:
        Both should give the same list. A comparator that is not a total order
        yields different results from different starting permutations, which
        shows up as a genome whose contig offsets depend on the order its
        source file happened to list them in -- and therefore as tiles that
        move when a file is rewritten.
    """
    # Act
    forward = natsorted(names)
    backward = natsorted(list(reversed(names)))

    # Assert
    assert forward == backward


@PROPERTY
@given(names=strategies.conventional_chrom_names())
def test_natsorted_should_reach_a_fixed_point_in_one_pass(names):
    """Test that sorting an already-sorted list changes nothing.

    Given:
        Any shuffled list of distinct conventional chromosome names.
    When:
        It is natsorted twice.
    Then:
        The second pass should be a no-op, and the result should be a
        permutation of the input -- so nothing is dropped or duplicated.
    """
    # Act
    once = natsorted(names)
    twice = natsorted(once)

    # Assert
    assert once == twice
    assert sorted(once) == sorted(names)


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


class TestBinCount:
    """Bin accounting, which decides whether blocks in a strip agree on shape.

    Every dense 2D tileset shapes its fetched blocks with this, and the
    reconcilers below consume those shapes, so an off-by-one here surfaces as
    a misshapen tile rather than as a wrong value.
    """

    @pytest.mark.parametrize(
        "start,end,expected",
        [
            (0, 1000, 250),
            (0, 150, 38),
            (50, 150, 26),
            (1, 2, 1),
            (0, 0, 0),
            (50, 54, 2),
        ],
        ids=[
            "whole-contig",
            "aligned-start",
            "unaligned-start",
            "sub-bin",
            "empty",
            "one-bin-wide-across-two",
        ],
    )
    def test_bin_count_should_round_outward_from_both_ends(
        self, start, end, expected
    ):
        """Test the overlap-based count against hand-computed values.

        Given:
            An interval on a contig and a 4 bp resolution.
        When:
            Its bin count is taken.
        Then:
            It should span ``floor(start / 4)`` to ``ceil(end / 4)``, because a
            fetcher returns every bin the region *overlaps*. The last case is
            the one that matters: ``50-54`` is exactly one bin wide but
            straddles a boundary, so it touches two.
        """
        # Arrange
        interval = GenomicRange(0, "c1", start, end)

        # Act
        result = bin_count(interval, 4)

        # Assert
        assert result == expected

    def test_bin_count_should_count_an_out_of_bounds_interval_the_same_way(
        self,
    ):
        """Test that the formula does not branch on boundedness.

        Given:
            An interval with no chromosome name -- padding past the last
            contig -- starting mid-bin.
        When:
            Its bin count is taken.
        Then:
            It should round outward exactly as an in-bounds interval does. The
            count is the shape the reconciler expects for the padding block, so
            a shorter count there is the same shape error as anywhere else.
        """
        # Arrange
        interval = GenomicRange(0, None, 50, 150)

        # Act
        result = bin_count(interval, 4)

        # Assert
        assert result == 26

    @pytest.mark.pinned
    def test_bin_count_should_exceed_the_convention_multivec_pads_with(self):
        """Test the divergence between the 1D and 2D padding conventions.

        Given:
            An out-of-bounds interval starting mid-bin.
        When:
            Its bin count is compared with the naive ``ceil(span / binsize)``
            that ``clodius.tiles_v2.multivec`` uses to size the same padding.
        Then:
            They should differ by one. Both are correct for their own
            reconciler, and nothing in either module says so -- the next person
            to notice the duplication will unify them and silently change the
            shape one of the two paths produces.
        """
        # Arrange
        interval = GenomicRange(0, None, 50, 150)
        naive = math.ceil((interval.end - interval.start) / 4)

        # Act
        result = bin_count(interval, 4)

        # Assert
        assert result == 26
        assert naive == 25

    @PROPERTY
    @given(
        start=st.integers(min_value=0, max_value=10_000),
        span=st.integers(min_value=0, max_value=10_000),
        binsize=st.integers(min_value=1, max_value=500),
    )
    def test_bin_count_should_never_undercount_the_span(
        self, start, span, binsize
    ):
        """Test the relationship to the naive count.

        Given:
            Any interval and bin size.
        When:
            Its bin count is compared with ``ceil(span / binsize)``.
        Then:
            It should be at least as large, and at most one larger. An
            undercount makes a fetched block narrower than the reconciler
            expects, which is a shape error rather than a wrong value.
        """
        # Arrange
        interval = GenomicRange(0, "c1", start, start + span)
        naive = -(-span // binsize)

        # Act
        result = bin_count(interval, binsize)

        # Assert
        assert naive <= result <= naive + 1


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


class TestGenomicRange:
    """The per-chromosome interval every inversion yields."""

    def test___init___should_expose_the_four_fields_as_given(self):
        """Test construction.

        Given:
            A chromosome id, name, start and end.
        When:
            A range is constructed.
        Then:
            It should expose all four unchanged.
        """
        # Act
        gr = GenomicRange(cid=1, name="c2", start=5, end=9)

        # Assert
        assert (gr.cid, gr.name, gr.start, gr.end) == (1, "c2", 5, 9)

    def test___setattr___should_raise_when_a_field_is_assigned(self):
        """Test immutability.

        Given:
            A constructed range.
        When:
            One of its fields is assigned.
        Then:
            It should raise, since ranges are yielded from a generator and
            shared by callers.
        """
        # Arrange
        gr = GenomicRange(cid=0, name="c1", start=0, end=10)

        # Act & assert
        with pytest.raises(AttributeError):
            gr.start = 5

    def test_is_out_of_bounds_should_return_false_when_the_range_is_named(self):
        """Test the in-genome case.

        Given:
            A range carrying a chromosome name.
        When:
            Its out-of-bounds flag is read.
        Then:
            It should be false.
        """
        # Act & assert
        assert GenomicRange(0, "c1", 0, 10).is_out_of_bounds is False

    def test_is_out_of_bounds_should_return_true_when_the_range_is_unnamed(
        self,
    ):
        """Test the padding inversion yields past the genome.

        Given:
            A range whose name is None.
        When:
            Its out-of-bounds flag is read.
        Then:
            It should be true, which is how a caller knows to pad the interval
            rather than fetch it.
        """
        # Act & assert
        assert GenomicRange(3, None, 0, 10).is_out_of_bounds is True

    def test_as_tuple_should_return_name_start_and_end(self):
        """Test the tuple form.

        Given:
            A range on a named chromosome.
        When:
            It is converted to a tuple.
        Then:
            It should return name, start and end, dropping the chromosome id.
        """
        # Act & assert
        assert GenomicRange(0, "c1", 5, 9).as_tuple() == ("c1", 5, 9)

    def test_as_tuple_should_raise_when_the_range_is_out_of_bounds(self):
        """Test the tuple form on padding.

        Given:
            A range whose name is None.
        When:
            It is converted to a tuple.
        Then:
            It should raise, since there is no chromosome to name.
        """
        # Act & assert
        with pytest.raises(ValueError, match="Out-of-bounds range"):
            GenomicRange(3, None, 0, 10).as_tuple()

    def test_to_ucsc_should_use_the_closed_convention_by_default(self):
        """Test the default rendering.

        Given:
            A range covering four bases, stored zero-based half-open.
        When:
            It is rendered with no convention argument.
        Then:
            It should return the one-based fully-closed form.
        """
        # Act & assert
        assert GenomicRange(0, "c1", 5, 9).to_ucsc() == "c1:6-9"

    def test_to_ucsc_should_shift_only_the_start_when_closed(self):
        """Test the one-based fully-closed convention.

        Given:
            The same four-base range.
        When:
            It is rendered as one-based fully-closed.
        Then:
            It should shift the start by one and leave the end alone.
            Converting half-open to closed decrements the end while
            zero-based to one-based increments it, so the two cancel.
        """
        # Act & assert
        assert GenomicRange(0, "c1", 5, 9).to_ucsc(coords="11") == "c1:6-9"

    def test_to_ucsc_should_emit_stored_coordinates_when_half_open(self):
        """Test the zero-based half-open convention.

        Given:
            The same four-base range.
        When:
            It is rendered as zero-based half-open.
        Then:
            It should emit the stored coordinates verbatim, describing the
            same four bases as the closed form.
        """
        # Act & assert
        assert GenomicRange(0, "c1", 5, 9).to_ucsc(coords="01") == "c1:5-9"

    def test_to_ucsc_should_reverse_a_zero_width_range_when_closed(self):
        """Test the interval the closed convention cannot express.

        Given:
            A zero-width range, the shape inversion yields whenever a tile
            boundary lands exactly on a chromosome boundary.
        When:
            It is rendered as one-based fully-closed.
        Then:
            It should return a reversed string, a closed interval having no
            way to denote emptiness. Reachable in ordinary tiling, not
            hypothetical.
        """
        # Act & assert
        assert GenomicRange(0, "c1", 5, 5).to_ucsc(coords="11") == "c1:6-5"

    def test_to_ucsc_should_raise_when_the_convention_is_unrecognized(self):
        """Test rejection of an unknown convention.

        Given:
            A valid range.
        When:
            It is rendered with a convention that is neither supported form.
        Then:
            It should raise rather than silently picking one.
        """
        # Act & assert
        with pytest.raises(ValueError, match="Invalid coordinate convention"):
            GenomicRange(0, "c1", 5, 9).to_ucsc(coords="10")

    def test_to_ucsc_should_raise_when_the_range_is_out_of_bounds(self):
        """Test that the name check precedes the convention check.

        Given:
            An out-of-bounds range and an invalid convention.
        When:
            It is rendered.
        Then:
            It should report the out-of-bounds range, not the convention.
        """
        # Act & assert
        with pytest.raises(ValueError, match="Out-of-bounds range"):
            GenomicRange(3, None, 0, 10).to_ucsc(coords="10")

    @given(
        start=st.integers(min_value=0, max_value=10**9),
        width=st.integers(min_value=0, max_value=10**6),
    )
    @PROPERTY
    def test_to_ucsc_should_differ_only_in_the_start_between_conventions(
        self, start, width
    ):
        """Test the relationship between the two conventions.

        Given:
            Any range on a named chromosome.
        When:
            It is rendered under both conventions.
        Then:
            The half-open form should parse back to exactly the stored
            coordinates, and the closed form should agree on the chromosome
            and the end while starting exactly one higher.
        """
        # Arrange
        gr = GenomicRange(0, "c1", start, start + width)

        # Act
        half_open = gr.to_ucsc(coords="01")
        closed = gr.to_ucsc(coords="11")

        # Assert
        name, _, span = half_open.partition(":")
        lo, _, hi = span.partition("-")
        assert (name, int(lo), int(hi)) == gr.as_tuple()
        c_name, _, c_span = closed.partition(":")
        c_lo, _, c_hi = c_span.partition("-")
        assert (c_name, int(c_hi), int(c_lo)) == (name, int(hi), int(lo) + 1)

    def test___hash___should_treat_equal_ranges_as_one(self):
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


class TestChromsizesTransform:
    """Mapping a genomic range into the concatenated coordinate space."""

    def test_transform_should_leave_the_first_chromosome_unshifted(self):
        """Test the zero-offset case.

        Given:
            A range on the first chromosome.
        When:
            It is transformed.
        Then:
            It should come back unchanged, that chromosome starting at zero.
        """
        # Act & assert
        assert Chromsizes.from_pairs(TINY).transform(
            GenomicRange(0, "c1", 10, 20)
        ) == (10, 20)

    def test_transform_should_add_the_chromosome_offset(self):
        """Test the ordinary case.

        Given:
            A range on the second chromosome, which starts at absolute 100.
        When:
            It is transformed.
        Then:
            It should shift both coordinates by that offset.
        """
        # Act & assert
        assert Chromsizes.from_pairs(TINY).transform(
            GenomicRange(1, "c2", 10, 20)
        ) == (110, 120)

    def test_transform_should_raise_when_the_chromosome_is_unknown(self):
        """Test rejection of a contig outside the coordinate system.

        Given:
            A range naming a chromosome the system does not carry.
        When:
            It is transformed.
        Then:
            It should raise naming the unknown chromosome.
        """
        # Act & assert
        with pytest.raises(KeyError, match="cX"):
            Chromsizes.from_pairs(TINY).transform(
                GenomicRange(9, "cX", 0, 10)
            )

    def test_transform_should_raise_when_the_range_is_out_of_bounds(self):
        """Test rejection of padding.

        Given:
            A range whose name is None.
        When:
            It is transformed.
        Then:
            It should raise, padding having no chromosome to offset from.
        """
        # Act & assert
        with pytest.raises(ValueError, match="Out-of-bounds range"):
            Chromsizes.from_pairs(TINY).transform(
                GenomicRange(3, None, 0, 10)
            )

    @pytest.mark.pinned
    def test_transform_should_ignore_the_chromosome_id(self):
        """Test which field the offset is keyed on.

        Given:
            A range whose chromosome id contradicts its name.
        When:
            It is transformed.
        Then:
            It should offset by the name's chromosome and never consult the
            id, so a hand-built range with a mismatched id is silently
            accepted. There is no public way to detect the mismatch.
        """
        # Act & assert
        assert Chromsizes.from_pairs(TINY).transform(
            GenomicRange(99, "c2", 0, 10)
        ) == (100, 110)

    def test_transform_should_not_clamp_to_the_chromosome_length(self):
        """Test the absence of a containment check.

        Given:
            A range whose end runs past its chromosome's length.
        When:
            It is transformed.
        Then:
            It should return the shifted coordinates unclamped, so a caller
            building ranges by hand gets no protection from overrunning.
        """
        # Act & assert
        assert Chromsizes.from_pairs(TINY).transform(
            GenomicRange(0, "c1", 0, 500)
        ) == (0, 500)


    @given(data=st.data())
    @PROPERTY
    def test_transform_should_invert_a_range_lying_inside_one_chromosome(
        self, data
    ):
        """Test the round trip for a range that crosses no boundary.

        Given:
            Any genome and any range strictly inside one of its chromosomes.
        When:
            The range is transformed to absolute coordinates and inverted
            back.
        Then:
            It should yield exactly the original range.
        """
        # Arrange
        pairs = data.draw(
            st.lists(
                st.integers(min_value=2, max_value=10_000),
                min_size=1,
                max_size=8,
            ).map(lambda ls: [[f"c{i}", n] for i, n in enumerate(ls)])
        )
        cs = Chromsizes.from_pairs(pairs)
        cid = data.draw(st.integers(min_value=0, max_value=len(pairs) - 1))
        length = pairs[cid][1]
        start = data.draw(st.integers(min_value=0, max_value=length - 1))
        end = data.draw(st.integers(min_value=start, max_value=length - 1))
        gr = GenomicRange(cid, pairs[cid][0], start, end)

        # Act
        result = list(cs.invert(cs.transform(gr)))

        # Assert
        assert as_tuples(result) == [(cid, start, end)]

def canvas(**overrides):
    """A quadtree canvas over the tiny genome, with fields overridable."""
    base = {
        "z": 4,
        "binsize": 1.0,
        "tile_size": 32,
        "max_width": 512,
        "chromsizes": Chromsizes.from_pairs(TINY),
    }
    return Canvas(**{**base, **overrides})


class TestCanvas:
    """The genome-spanning uniform lattice at one zoom level."""

    def test___init___should_expose_the_lattice_geometry_as_given(self):
        """Test construction.

        Given:
            A zoom, binsize, tile size, extent and coordinate system.
        When:
            A canvas is constructed.
        Then:
            It should expose each unchanged.
        """
        # Arrange
        cs = Chromsizes.from_pairs(TINY)

        # Act
        result = Canvas(
            z=4, binsize=1.0, tile_size=32, max_width=512, chromsizes=cs
        )

        # Assert
        assert (result.z, result.binsize, result.tile_size) == (4, 1.0, 32)
        assert (result.max_width, result.chromsizes) == (512, cs)

    def test___init___should_default_to_no_coordinate_system(self):
        """Test the non-genomic case.

        Given:
            A canvas constructed without a coordinate system.
        When:
            Its chromsizes are read.
        Then:
            It should be None, a non-genomic tileset having none.
        """
        # Act & assert
        assert (
            Canvas(z=0, binsize=1.0, tile_size=8, max_width=8).chromsizes
            is None
        )

    def test___setattr___should_raise_when_a_field_is_assigned(self):
        """Test immutability.

        Given:
            A constructed canvas, handed out per tile and whose derived
            geometry is only sound because it cannot change.
        When:
            One of its fields is assigned.
        Then:
            It should raise.
        """
        # Arrange
        subject = canvas()

        # Act & assert
        with pytest.raises(AttributeError):
            subject.z = 1

    @pytest.mark.parametrize(
        "field,value,match",
        [
            ("binsize", 0.0, "binsize must be positive"),
            ("binsize", -1.0, "binsize must be positive"),
            ("tile_size", 0, "tile_size must be positive"),
            ("tile_size", -8, "tile_size must be positive"),
            ("max_width", -1, "max_width must be non-negative"),
        ],
        ids=["binsize-0", "binsize-neg", "tile-0", "tile-neg", "width-neg"],
    )
    def test___init___should_reject_a_degenerate_lattice(
        self, field, value, match
    ):
        """Test the construction guards.

        Given:
            A geometry with a non-positive binsize or tile size, or a negative
            extent.
        When:
            A canvas is constructed.
        Then:
            It should raise naming the offending field. Without the guard the
            failure surfaced later as ZeroDivisionError from a property, or
            not at all.
        """
        # Act & assert
        with pytest.raises(ValueError, match=match):
            canvas(**{field: value})

    def test_span_should_run_from_zero_to_the_extent(self):
        """Test the canvas extent.

        Given:
            A canvas of a given extent.
        When:
            Its span is read.
        Then:
            It should run from zero to that extent.
        """
        # Act & assert
        assert canvas().span == (0, 512)

    @pytest.mark.parametrize(
        "binsize,max_width,expected",
        [
            (16.0, 4096, 256),
            (2.5, 1000, 400),  # fractional binsize, integral result
            (3.0, 10, 3),  # partial trailing bin discarded
            (1000.0, 10, 0),  # binsize wider than the canvas
        ],
        ids=["exact", "fractional", "partial", "oversized"],
    )
    def test_n_bins_should_divide_the_extent_by_the_binsize(
        self, binsize, max_width, expected
    ):
        """Test the bin count.

        Given:
            A canvas whose binsize divides its extent exactly, fractionally,
            or not at all.
        When:
            Its bin count is read.
        Then:
            It should truncate rather than round up, discarding any partial
            trailing bin.
        """
        # Act
        n_bins = canvas(binsize=binsize, max_width=max_width).n_bins

        # Assert
        assert n_bins == expected
        assert type(n_bins) is int

    def test_n_tiles_should_be_two_to_the_zoom_on_a_quadtree(self):
        """Test the identity the implicit ladder rests on.

        Given:
            A quadtree canvas, where the extent is the tile size times two to
            the max zoom.
        When:
            Its tile count is read.
        Then:
            It should equal two to the zoom.
        """
        # Act & assert
        assert canvas().n_tiles == 16

    def test_n_tiles_should_count_a_partial_trailing_tile(self):
        """Test that a partly filled tile still counts.

        Given:
            A canvas holding a bin count the tile size does not divide.
        When:
            Its tile count is read.
        Then:
            It should round up, the trailing bins still needing a tile.
        """
        # Act & assert
        assert canvas(tile_size=3, max_width=400).n_tiles == 134

    def test_n_tiles_should_be_zero_when_the_canvas_holds_no_bins(self):
        """Test the degenerate canvas.

        Given:
            A canvas whose binsize exceeds its extent, so it holds no bins.
        When:
            Its tile count is read.
        Then:
            It should be zero.
        """
        # Act & assert
        assert canvas(binsize=1000.0, max_width=10).n_tiles == 0

    def test_tile_span_should_return_the_absolute_range_of_the_tile(self):
        """Test the tile window.

        Given:
            A quadtree canvas of sixteen tiles.
        When:
            The spans of the first and last tiles are requested.
        Then:
            They should start at zero and end exactly at the canvas edge.
        """
        # Arrange
        subject = canvas()

        # Act & assert
        assert (subject.tile_span(0), subject.tile_span(15)) == (
            (0, 32),
            (480, 512),
        )

    def test_tile_span_should_stay_contiguous_when_the_width_is_fractional(
        self,
    ):
        """Test truncation at a fractional tile width.

        Given:
            A canvas whose tile width is seven and a half bases.
        When:
            Three consecutive tile spans are requested.
        Then:
            Each should start exactly where the previous ended, so truncation
            leaves no gap even though no tile is a whole number wide.
        """
        # Arrange
        subject = canvas(binsize=2.5, tile_size=3, max_width=1000)

        # Act & assert
        assert [subject.tile_span(x) for x in range(3)] == [
            (0, 7),
            (7, 15),
            (15, 22),
        ]

    def test_tile_span_should_not_bounds_check_the_position(self):
        """Test the absence of a guard on this accessor.

        Given:
            A canvas of sixteen tiles.
        When:
            The span of a position far past the last is requested.
        Then:
            It should return a range past the canvas rather than raising.
            Only inversion bounds-checks, so a caller reaching for this
            directly gets no protection -- no tileset does today.
        """
        # Act & assert
        assert canvas().tile_span(999) == (31968, 32000)

    def test_invert_should_return_the_genomic_ranges_of_the_tile(self):
        """Test the ordinary inversion.

        Given:
            A quadtree canvas over the three-chromosome genome.
        When:
            The first tile is inverted.
        Then:
            It should yield one range covering that tile's span on the first
            chromosome.
        """
        # Act & assert
        assert as_tuples(canvas().invert(0)) == [(0, 0, 32)]

    def test_invert_should_split_when_the_tile_crosses_a_boundary(self):
        """Test a tile spanning two chromosomes.

        Given:
            The same canvas and a tile whose span crosses the first boundary.
        When:
            It is inverted.
        Then:
            It should yield the tail of the first chromosome then the head of
            the second.
        """
        # Act & assert
        assert as_tuples(canvas().invert(3)) == [(0, 96, 100), (1, 0, 28)]

    def test_invert_should_accept_a_tile_past_the_genome(self):
        """Test the padding tiles a low zoom always produces.

        Given:
            A tile lying past the 350 bp genome but inside the 512 bp canvas.
        When:
            It is inverted.
        Then:
            It should yield an out-of-bounds range rather than raising. That
            padding is what fills the trailing NaN bins of a low-zoom tile, so
            rejecting it would break every such tile.
        """
        # Act
        result = list(canvas().invert(12))

        # Assert
        assert [r.is_out_of_bounds for r in result] == [True]

    def test_invert_should_accept_the_last_tile(self):
        """Test the inclusive upper end of the accepted domain.

        Given:
            A canvas of sixteen tiles.
        When:
            The last position is inverted.
        Then:
            It should yield ranges rather than raising, pinning that the guard
            rejects at the tile count rather than one below it.
        """
        # Act & assert
        assert list(canvas().invert(15))

    @pytest.mark.parametrize("x", [16, 999, -1], ids=["one-past", "far", "neg"])
    def test_invert_should_raise_when_the_position_is_outside_the_canvas(
        self, x
    ):
        """Test the bounds guard.

        Given:
            A canvas of sixteen tiles and a position outside them.
        When:
            It is inverted.
        Then:
            It should raise a tile error naming the count and the zoom, which
            the server boundary can render per tile -- rather than reaching
            coordinate arithmetic and raising a plain ValueError.
        """
        # Act & assert
        with pytest.raises(TileOutOfBounds, match="16 tiles at zoom 4"):
            canvas().invert(x)

    def test_invert_should_raise_before_the_result_is_consumed(self):
        """Test that the bounds error is eager.

        Given:
            A canvas and an out-of-range position.
        When:
            Inversion is called and the result never iterated.
        Then:
            It should raise at call time. The underlying coordinate inversion
            is a generator whose errors surface only on iteration, so a caller
            that checks bounds without consuming would otherwise miss this.
        """
        # Act & assert
        with pytest.raises(TileOutOfBounds):
            canvas().invert(99)

    def test_invert_should_raise_when_the_canvas_has_no_chromsizes(self):
        """Test inversion on a non-genomic canvas.

        Given:
            A canvas built without a coordinate system.
        When:
            A tile is inverted.
        Then:
            It should raise, there being nothing to invert against.
        """
        # Act & assert
        with pytest.raises(ValueError, match="no chromsizes"):
            canvas(chromsizes=None).invert(0)

    def test_invert_should_report_the_missing_chromsizes_first(self):
        """Test the precedence between the two guards.

        Given:
            A canvas with no coordinate system and an out-of-range position.
        When:
            It is inverted.
        Then:
            It should report the missing coordinate system rather than the
            bounds, that check running first.
        """
        # Act & assert
        with pytest.raises(ValueError, match="no chromsizes"):
            canvas(chromsizes=None).invert(999)

    @given(
        binsize=st.integers(min_value=1, max_value=4096),
        tile_size=st.sampled_from([1, 16, 256, 1024]),
        max_width=st.integers(min_value=1, max_value=10**9),
    )
    @PROPERTY
    def test_n_tiles_should_be_the_smallest_count_covering_every_bin(
        self, binsize, tile_size, max_width
    ):
        """Test the relationship between bins and tiles.

        Given:
            Any canvas geometry with an integer binsize, so the property is
            about the ceiling rule rather than float representation.
        When:
            Its bin and tile counts are read.
        Then:
            The tiles should cover every bin, one fewer should not, and a
            canvas holds no tiles exactly when it holds no bins.
        """
        # Arrange
        subject = canvas(
            binsize=float(binsize), tile_size=tile_size, max_width=max_width
        )

        # Act
        n_bins, n_tiles = subject.n_bins, subject.n_tiles

        # Assert
        assert n_bins == max_width // binsize
        assert n_tiles * tile_size >= n_bins
        if n_bins:
            assert (n_tiles - 1) * tile_size < n_bins
        else:
            assert n_tiles == 0

    @given(
        binsize=st.integers(min_value=1, max_value=1024),
        tile_size=st.sampled_from([1, 16, 256]),
        max_width=st.integers(min_value=1, max_value=10**6),
    )
    @PROPERTY
    def test_tile_span_should_tile_the_canvas_contiguously(
        self, binsize, tile_size, max_width
    ):
        """Test that the lattice has no gaps or overlaps.

        Given:
            Any canvas geometry with an integer binsize.
        When:
            Every tile span is requested in order.
        Then:
            The first should start at zero and each should begin exactly where
            the previous ended. This asserts contiguity rather than full
            coverage: when the binsize does not divide the extent the trailing
            partial bin is discarded and the last tile ends short of the
            canvas, which no ladder in use exercises.
        """
        # Arrange
        subject = canvas(
            binsize=float(binsize), tile_size=tile_size, max_width=max_width
        )

        # Act
        spans = [subject.tile_span(x) for x in range(subject.n_tiles)]

        # Assert
        if spans:
            assert spans[0][0] == 0
            assert all(hi >= lo for lo, hi in spans)
            for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
                assert prev_end == next_start


def drop_reference(chunks, binsize, threshold=1.0):
    """An independent reference for which chunks lose their trailing bin.

    Derived as target-tracking rather than drift-accumulating: a chunk sheds
    its last bin once the bins kept so far exceed what the distance actually
    walked entitles the caller to. Dividing the implementation's drift
    condition through by the binsize gives the same rule, but this form is
    stated in exact rational arithmetic, so it also catches float error
    accumulating across a long chunk sequence.

    Exact, not ``limit_denominator``: a float is exactly representable as a
    Fraction, and rounding to a bounded denominator silently flattens a
    sub-nanobase span to zero and flips the comparison.
    """
    kept_total, walked, drops = 0, Fraction(0), set()
    b, t = Fraction(binsize), Fraction(threshold)
    for i, (values, span) in enumerate(chunks):
        walked += Fraction(span)
        if len(values) == 0:
            continue
        kept_total += len(values)
        if Fraction(kept_total) >= walked / b + t:
            kept_total -= 1
            drops.add(i)
    return drops


def reconcile_reference(chunks, binsize, threshold=1.0):
    """The bins :func:`reconcile_sequential` should keep, in order."""
    drops = drop_reference(chunks, binsize, threshold)
    out = []
    for i, (values, _) in enumerate(chunks):
        if len(values) == 0:
            continue
        out.extend(list(values)[:-1] if i in drops else list(values))
    return out


def labelled_chunks(counts_and_spans, binsize):
    """``(values, span)`` chunks whose bins are globally distinct integers."""
    built, k = [], 0
    for n, span_bp in counts_and_spans:
        built.append((list(range(k, k + n)), float(span_bp)))
        k += n
    return built


def block_grid(row_labels, col_labels):
    """Blocks whose every cell carries the identity of its row and column.

    Built as an outer sum, so a cell's value names exactly which row label and
    which column label produced it. That is what lets the separability
    property compare a 2D result against two independent 1D results.
    """
    return [
        [
            np.add.outer(
                np.asarray(rows, dtype=float), np.asarray(cols, dtype=float)
            )
            for cols in col_labels
        ]
        for rows in row_labels
    ]


def test_reconcile_sequential_should_drop_the_bin_of_the_drifting_chunk():
    """Test which bin a matured drift costs.

    Given:
        Two chunks whose bins over-represent their spans, so drift reaches a
        whole bin on the second.
    When:
        They are reconciled.
    Then:
        It should drop the trailing bin of the chunk whose drift matured and
        no other, so the tile keeps its leading data and loses its last.
    """
    # Act
    result = reconcile_sequential([([1, 2, 3], 25.0), ([4, 5, 6], 25.0)], 10.0)

    # Assert
    assert list(result) == [1, 2, 3, 4, 5]


def test_reconcile_sequential_should_drop_at_most_one_bin_per_chunk():
    """Test the per-chunk drop budget.

    Given:
        A first chunk that over-supplies by more than two whole bins.
    When:
        The chunks are reconciled.
    Then:
        It should spend at most one bin on that chunk, deferring the rest.
    """
    # Act
    result = reconcile_sequential([([1, 2, 3], 5.0), ([4, 5, 6], 45.0)], 10.0)

    # Assert
    assert list(result) == [1, 2, 4, 5, 6]


def test_reconcile_sequential_should_take_a_deferred_drop_on_the_next_chunk():
    """Test that unspent drift carries forward.

    Given:
        A first chunk over-supplying by more than one bin, followed by a chunk
        holding a single bin.
    When:
        They are reconciled.
    Then:
        The second chunk should surrender its only bin, showing the deferred
        drop is taken at the next opportunity rather than lost.
    """
    # Act
    result = reconcile_sequential([([1, 2, 3], 5.0), ([9], 5.0)], 10.0)

    # Assert
    assert list(result) == [1, 2]


@pytest.mark.parametrize(
    "threshold,expected",
    [(1.0, [1, 2, 3]), (0.5, [1, 3, 4])],
    ids=["floor", "midpoint"],
)
def test_reconcile_sequential_should_drop_a_different_bin_at_each_threshold(
    threshold, expected
):
    """Test that the threshold changes which bin goes, not how many.

    Given:
        Two chunks each over-supplying by half a bin.
    When:
        They are reconciled at the floor and the midpoint threshold.
    Then:
        Both should return the same number of bins but drop a different one.
        This is why a bin-count assertion cannot detect a threshold change --
        only an assertion on which bins survive can.
    """
    # Act
    result = reconcile_sequential(
        [([1, 2], 15.0), ([3, 4], 15.0)], 10.0, threshold=threshold
    )

    # Assert
    assert list(result) == expected


def test_reconcile_sequential_should_count_the_span_of_an_empty_chunk():
    """Test that a chunk supplying no bins still moves the drift.

    Given:
        Three chunks where the middle supplies no bins, and the same sequence
        with that chunk removed.
    When:
        Both are reconciled.
    Then:
        The empty chunk should contribute no bins of its own yet still advance
        the walked distance, so its presence changes what the later chunks
        keep.
    """
    # Act
    with_empty = reconcile_sequential(
        [([1, 2, 3], 25.0), ([], 10.0), ([4, 5], 15.0)], 10.0
    )
    without = reconcile_sequential([([1, 2, 3], 25.0), ([4, 5], 15.0)], 10.0)

    # Assert
    assert list(with_empty) == [1, 2, 3, 4, 5]
    assert list(without) == [1, 2, 3, 4]


def test_reconcile_sequential_should_preserve_a_second_dimension():
    """Test the multi-statistic tile shape.

    Given:
        Chunks of two-column arrays, as bigwig's minMax and whisker modes
        produce.
    When:
        They are reconciled.
    Then:
        It should drop a whole row and keep both columns, so the per-bin
        statistic count survives reconciliation.
    """
    # Arrange
    block = np.arange(6, dtype=float).reshape(3, 2)

    # Act
    result = reconcile_sequential([(block, 25.0), (block, 25.0)], 10.0)

    # Assert
    assert result.shape == (5, 2)


def test_reconcile_sequential_should_preserve_an_integer_dtype():
    """Test that reconciliation does not widen the values.

    Given:
        Integer-valued chunks where the drop takes a chunk's only bin.
    When:
        They are reconciled.
    Then:
        It should return an integer array. Slicing a bare Python list down to
        empty would otherwise promote the whole concatenation to float.
    """
    # Act
    result = reconcile_sequential([([7], 0.0), ([1, 2], 20.0)], 10.0)

    # Assert
    assert result.dtype.kind == "i"


def test_reconcile_sequential_should_consume_any_iterable():
    """Test that the chunks argument need not be a sequence.

    Given:
        The chunks supplied as a one-shot generator.
    When:
        They are reconciled.
    Then:
        It should return the same bins as the equivalent list, since the
        function walks the chunks more than once internally.
    """
    # Arrange
    chunks = [([1, 2, 3], 25.0), ([4, 5, 6], 25.0)]

    # Act
    result = reconcile_sequential((c for c in chunks), 10.0)

    # Assert
    assert list(result) == [1, 2, 3, 4, 5]


def test_reconcile_sequential_should_return_an_empty_array_for_no_chunks():
    """Test the degenerate input.

    Given:
        No chunks at all and an expected count of zero.
    When:
        They are reconciled.
    Then:
        It should return an empty array rather than raising.
    """
    # Act & assert
    assert len(reconcile_sequential([], 10.0, expected_bins=0)) == 0


def test_reconcile_sequential_should_not_raise_when_no_count_is_expected():
    """Test that the count check is opt-in.

    Given:
        Chunks supplying more bins than their spans justify, and no expected
        count.
    When:
        They are reconciled.
    Then:
        It should return the drifted bins without raising, the assertion
        applying only when a caller states what it expects.
    """
    # Act & assert
    assert len(reconcile_sequential([([1, 2, 3, 4], 10.0)], 10.0)) == 3


@given(
    chunks=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=5),
            st.integers(min_value=0, max_value=20_000),
        ),
        min_size=1,
        max_size=12,
    ),
    binsize=st.sampled_from([1.0, 3.0, 10.0, 1024.0, 2.5, 17.0]),
    threshold=st.sampled_from([1.0, 0.5, 0.25]),
)
@PROPERTY
def test_reconcile_sequential_should_match_an_exact_arithmetic_reference(
    chunks, binsize, threshold
):
    """Test the drop policy against an independently derived reference.

    Given:
        Any chunk sequence, binsize and threshold, with each chunk's bins
        labelled so every one is identifiable and its span a whole number of
        base pairs.
    When:
        The chunks are reconciled.
    Then:
        It should keep exactly the bins the reference keeps, in the same
        order. The reference is float-free, so this catches floating-point
        error accumulating across a long chunk sequence as well as a wrong
        drop.

        Spans are integral because that is what a caller can produce: every
        span is an interval width on integer genomic coordinates. Far below
        one base pair the implementation's float subtraction and the
        reference's exact comparison do diverge at the threshold, but nothing
        can reach that regime -- measured at zero divergences over 20,000
        randomized cases with integral spans, and 57 once spans below a
        picobase are admitted.
    """
    # Arrange
    built = labelled_chunks(chunks, binsize)

    # Act
    result = reconcile_sequential(built, binsize, threshold=threshold)

    # Assert
    assert list(result) == reconcile_reference(built, binsize, threshold)


@given(
    chunks=st.lists(
        st.tuples(
            st.integers(min_value=1, max_value=5),
            st.integers(min_value=0, max_value=200),
        ),
        min_size=1,
        max_size=10,
    ),
    binsize=st.sampled_from([1.0, 10.0, 2.5]),
)
@PROPERTY
def test_reconcile_sequential_should_only_ever_drop_trailing_bins(
    chunks, binsize
):
    """Test that reconciliation never reorders or reaches inside a chunk.

    Given:
        Any chunk sequence whose bins are distinct ascending integers.
    When:
        The chunks are reconciled.
    Then:
        The output should preserve input order, and each chunk's surviving
        bins should be a prefix of that chunk, one bin shorter at most. A
        reconciler that reversed the chunks, dropped from the head, or took
        two bins from one chunk would fail here while still returning the
        right total.
    """
    # Arrange
    built = labelled_chunks(chunks, binsize)

    # Act
    result = list(reconcile_sequential(built, binsize))

    # Assert
    kept = set(result)
    concatenated = [v for values, _ in built for v in values]
    assert result == [v for v in concatenated if v in kept]
    for values, _ in built:
        present = [v for v in values if v in kept]
        assert present == values[: len(present)]
        assert len(values) - 1 <= len(present) <= len(values)


@given(
    counts=st.lists(
        st.integers(min_value=1, max_value=5), min_size=1, max_size=8
    ),
    binsize=st.sampled_from([1.0, 10.0, 4.0]),
    threshold=st.sampled_from([1.0, 0.5]),
)
@PROPERTY
def test_reconcile_sequential_should_drop_nothing_when_already_on_the_lattice(
    counts, binsize, threshold
):
    """Test that a well-behaved fetcher is never penalized.

    Given:
        Chunks whose bin counts exactly match their spans divided by the
        binsize, so no drift accumulates.
    When:
        They are reconciled.
    Then:
        It should return the plain concatenation, dropping nothing.
    """
    # Arrange
    built = labelled_chunks([(n, n * binsize) for n in counts], binsize)
    expected = [v for values, _ in built for v in values]

    # Act
    result = reconcile_sequential(built, binsize, threshold=threshold)

    # Assert
    assert list(result) == expected


def test_reconcile_sequential_2d_should_assemble_when_no_drift_accumulates():
    """Test the undrifted assembly.

    Given:
        A two-by-two grid of three-by-three blocks whose spans match their bin
        counts exactly.
    When:
        The blocks are reconciled.
    Then:
        It should return the plain block assembly at full size.
    """
    # Arrange
    blocks = [[np.ones((3, 3)) for _ in range(2)] for _ in range(2)]

    # Act
    result = reconcile_sequential_2d(blocks, [30.0, 30.0], [30.0, 30.0], 10.0)

    # Assert
    assert result.shape == (6, 6)


def test_reconcile_sequential_2d_should_drop_a_row_from_the_drifting_strip():
    """Test that a drifting row axis costs rows and not columns.

    Given:
        A grid whose row spans over-represent while its column spans do not.
    When:
        The blocks are reconciled.
    Then:
        Every block in the drifting row strip should lose its last row, and no
        column should be touched.
    """
    # Arrange
    blocks = [[np.ones((3, 3)) for _ in range(2)] for _ in range(2)]

    # Act
    result = reconcile_sequential_2d(blocks, [25.0, 25.0], [30.0, 30.0], 10.0)

    # Assert
    assert result.shape == (5, 6)


def test_reconcile_sequential_2d_should_drop_a_column_from_the_drifting_strip():
    """Test the mirror image of the row case.

    Given:
        A grid whose column spans over-represent while its row spans do not.
    When:
        The blocks are reconciled.
    Then:
        It should lose a column and keep every row, confirming the two axes
        are corrected independently rather than together.
    """
    # Arrange
    blocks = [[np.ones((3, 3)) for _ in range(2)] for _ in range(2)]

    # Act
    result = reconcile_sequential_2d(blocks, [30.0, 30.0], [25.0, 25.0], 10.0)

    # Assert
    assert result.shape == (6, 5)


def test_reconcile_sequential_2d_should_drop_both_when_both_axes_drift():
    """Test the corner case where both axes mature together.

    Given:
        A grid drifting on both axes at once.
    When:
        The blocks are reconciled.
    Then:
        It should shed a row and a column, the corner block losing both.
    """
    # Arrange
    blocks = [[np.ones((3, 3)) for _ in range(2)] for _ in range(2)]

    # Act
    result = reconcile_sequential_2d(blocks, [25.0, 25.0], [25.0, 25.0], 10.0)

    # Assert
    assert result.shape == (5, 5)


def test_reconcile_sequential_2d_should_remove_a_strip_supplying_no_bins():
    """Test a strip whose interval contributes nothing.

    Given:
        A grid whose middle row strip holds zero-row blocks, as a tile
        boundary landing exactly on a chromosome boundary produces.
    When:
        The blocks are reconciled.
    Then:
        That strip should vanish from the row axis rather than being assembled
        as zero-width, while its span still counts toward the row drift.
    """
    # Arrange
    blocks = [
        [np.ones((2, 2)), np.ones((2, 2))],
        [np.ones((0, 2)), np.ones((0, 2))],
        [np.ones((2, 2)), np.ones((2, 2))],
    ]

    # Act
    result = reconcile_sequential_2d(
        blocks, [20.0, 10.0, 20.0], [20.0, 20.0], 10.0
    )

    # Assert
    assert result.shape == (4, 4)


def test_reconcile_sequential_2d_should_raise_when_the_shape_is_wrong():
    """Test the shape assertion.

    Given:
        A grid assembling to one shape and an expected shape of another.
    When:
        The blocks are reconciled.
    Then:
        It should raise naming both rather than clamping, since silent
        truncation is what hid the original accumulator bug.
    """
    # Arrange
    blocks = [[np.ones((3, 3)) for _ in range(2)] for _ in range(2)]

    # Act & assert
    with pytest.raises(ValueError, match="expected"):
        reconcile_sequential_2d(
            blocks, [30.0, 30.0], [30.0, 30.0], 10.0, expected_shape=(5, 5)
        )


def test_reconcile_sequential_2d_should_return_the_tile_when_shape_matches():
    """Test the satisfied assertion.

    Given:
        A grid and the shape it actually assembles to.
    When:
        The blocks are reconciled.
    Then:
        It should return that tile unchanged.
    """
    # Arrange
    blocks = [[np.ones((3, 3)) for _ in range(2)] for _ in range(2)]

    # Act
    result = reconcile_sequential_2d(
        blocks, [30.0, 30.0], [30.0, 30.0], 10.0, expected_shape=(6, 6)
    )

    # Assert
    assert result.shape == (6, 6)


@pytest.mark.parametrize(
    "blocks", [[], [[]]], ids=["no-strips", "strip-with-no-blocks"]
)
def test_reconcile_sequential_2d_should_return_an_empty_tile_for_empty_grids(
    blocks,
):
    """Test the degenerate grids.

    Given:
        A grid with no strips at all, and one whose strip carries no blocks.
    When:
        They are reconciled.
    Then:
        Both should return an empty tile. The second form would otherwise
        index into a missing block and raise a bare IndexError, which no
        caller could translate.
    """
    # Act
    result = reconcile_sequential_2d(blocks, [0.0], [], 10.0)

    # Assert
    assert result.shape == (0, 0)


def test_reconcile_sequential_2d_should_return_an_empty_tile_when_an_axis_is_bare():
    """Test a grid where one axis supplies nothing.

    Given:
        A grid whose only column strip carries no columns.
    When:
        The blocks are reconciled.
    Then:
        It should return an empty tile. Assembling the surviving axis anyway
        hands numpy a row of no arrays, which it rejects with an error naming
        its own internals rather than anything a caller can act on.
    """
    # Act
    result = reconcile_sequential_2d([[np.ones((2, 0))]], [20.0], [0.0], 10.0)

    # Assert
    assert result.shape == (0, 0)


@given(
    row_counts=st.lists(
        st.integers(min_value=0, max_value=4), min_size=1, max_size=5
    ),
    col_counts=st.lists(
        st.integers(min_value=0, max_value=4), min_size=1, max_size=5
    ),
    row_spans=st.lists(
        st.integers(min_value=0, max_value=60), min_size=5, max_size=5
    ),
    col_spans=st.lists(
        st.integers(min_value=0, max_value=60), min_size=5, max_size=5
    ),
    binsize=st.sampled_from([1.0, 10.0, 2.5]),
)
@PROPERTY
def test_reconcile_sequential_2d_should_match_the_1d_reconciler_on_each_axis(
    row_counts, col_counts, row_spans, col_spans, binsize
):
    """Test that the two axes are corrected independently.

    Given:
        Any block grid whose every cell is the outer sum of a row label and a
        column label, so each cell names the row and column that produced it.
    When:
        The grid is reconciled, and separately the row labels and column
        labels are each reconciled with the one-dimensional reconciler using
        the same spans.
    Then:
        The result should equal the outer sum of the two one-dimensional
        results. The surviving rows are exactly the 1D row result, the
        surviving columns exactly the 1D column result, and assembly preserves
        position -- which is the whole claim that a 2D reconciler needs no
        arithmetic of its own.
    """
    # Arrange
    rows = [float(s) for s in row_spans[: len(row_counts)]]
    cols = [float(s) for s in col_spans[: len(col_counts)]]
    row_labels, k = [], 0
    for c in row_counts:
        row_labels.append(list(range(k, k + c)))
        k += c
    col_labels, k = [], 0
    for c in col_counts:
        col_labels.append([v * 1000 for v in range(k, k + c)])
        k += c

    # Act
    result = reconcile_sequential_2d(
        block_grid(row_labels, col_labels), rows, cols, binsize
    )
    kept_rows = reconcile_sequential(list(zip(row_labels, rows)), binsize)
    kept_cols = reconcile_sequential(list(zip(col_labels, cols)), binsize)

    # Assert
    if not any(row_counts) or not any(col_counts):
        # Every strip on one axis supplied no bins, so there is nothing to
        # assemble and the tile is empty on both axes. A strip that merely
        # drops to zero rows is different -- it still carries the other axis's
        # extent, and falls through to the outer-product check below.
        assert result.shape == (0, 0)
        return
    expected = np.add.outer(
        np.asarray(kept_rows, dtype=float), np.asarray(kept_cols, dtype=float)
    )
    assert result.shape == expected.shape
    np.testing.assert_array_equal(result, expected)


@given(
    row_counts=st.lists(
        st.integers(min_value=1, max_value=4), min_size=1, max_size=4
    ),
    col_counts=st.lists(
        st.integers(min_value=1, max_value=4), min_size=1, max_size=4
    ),
    row_spans=st.lists(
        st.integers(min_value=0, max_value=50), min_size=4, max_size=4
    ),
    col_spans=st.lists(
        st.integers(min_value=0, max_value=50), min_size=4, max_size=4
    ),
    binsize=st.sampled_from([1.0, 10.0]),
)
@PROPERTY
def test_reconcile_sequential_2d_should_transpose_when_the_axes_are_swapped(
    row_counts, col_counts, row_spans, col_spans, binsize
):
    """Test that neither axis is privileged.

    Given:
        Any block grid, and the same grid with every block transposed and the
        two axes' roles exchanged.
    When:
        Both are reconciled.
    Then:
        The second result should be the transpose of the first. An
        implementation that treated rows and columns differently -- reading
        counts off the wrong strip, say -- would break this while still
        producing plausible shapes.
    """
    # Arrange
    rows = [float(s) for s in row_spans[: len(row_counts)]]
    cols = [float(s) for s in col_spans[: len(col_counts)]]
    row_labels = [list(range(c)) for c in row_counts]
    col_labels = [[v * 1000 for v in range(c)] for c in col_counts]
    blocks = block_grid(row_labels, col_labels)
    swapped = [
        [blocks[r][c].T for r in range(len(row_labels))]
        for c in range(len(col_labels))
    ]

    # Act
    straight = reconcile_sequential_2d(blocks, rows, cols, binsize)
    flipped = reconcile_sequential_2d(swapped, cols, rows, binsize)

    # Assert
    np.testing.assert_array_equal(flipped, straight.T)
