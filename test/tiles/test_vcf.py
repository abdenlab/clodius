"""Tests for clodius.tiles.vcf.

``generic_regions`` was rewritten on this branch. It previously returned a
*dict* when the offset ran past the end of the data and a *pair* otherwise,
which every caller unpacks; it raised ``StopIteration`` when the offset landed
exactly at the end; and it answered "is there another page" by materializing a
whole second page of up to ``limit`` records, where ``limit`` is capped at
10000 on the wire.

The first half of this module tests it directly against a plain iterator. The
second half runs the same four cases through ``regions``, which is the only
caller and the only place the paging meets a real pysam fetcher -- the fixture
for that in ``data/`` is an un-smudged git-LFS pointer on a plain checkout, so
the VCF here is synthesized instead.
"""

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from clodius.tiles import vcf as mod
from clodius.tiles.vcf import generic_regions

from ..harness import builders, genome

VARIANTS = builders.DEFAULT_VARIANTS


@pytest.fixture
def chromsizes():
    """The canonical genome as the pandas Series the module expects."""
    return pd.Series(
        [length for _, length in genome.CANONICAL_CHROMSIZES],
        index=[name for name, _ in genome.CANONICAL_CHROMSIZES],
        dtype=int,
    )


@pytest.fixture
def vcf(tmp_path):
    """A BGZF+tabix VCF holding five variants across three contigs."""
    return str(builders.build_vcf(tmp_path / "tiny.vcf.gz"))


def test_generic_regions_should_return_a_page_and_flag_more_when_data_follows():
    """Test the ordinary paging case.

    Given:
        An iterator holding more records than the page limit.
    When:
        The first page is requested.
    Then:
        It should return that page and report that another follows, having
        consumed exactly one record beyond the page to decide it. That single
        lookahead is the whole point of the change: at the wire's limit of
        10000 the old form materialized a second full page -- 10000 records
        parsed and discarded -- to answer a yes/no question. Reading the
        iterator afterwards is what turns that from a comment into a test.
    """
    # Arrange
    fetcher = iter(range(10))

    # Act
    rows, has_next = generic_regions(fetcher, 0, 4)

    # Assert
    assert rows == [0, 1, 2, 3]
    assert has_next is True
    assert next(fetcher) == 5


def test_generic_regions_should_report_no_more_when_the_page_exhausts_the_data():
    """Test the last-page case.

    Given:
        An iterator holding exactly one page of records.
    When:
        That page is requested.
    Then:
        It should return the records and report that nothing follows.
    """
    # Arrange
    fetcher = iter(range(4))

    # Act
    rows, has_next = generic_regions(fetcher, 0, 4)

    # Assert
    assert rows == [0, 1, 2, 3]
    assert has_next is False


def test_generic_regions_should_return_an_empty_page_when_offset_lands_at_end():
    """Test the offset that consumes exactly all the data.

    Given:
        An offset equal to the number of available records.
    When:
        A page is requested.
    Then:
        It should return an empty page as a pair, not a dict. Both callers
        unpack a pair, so the dict this path used to return was a ValueError
        waiting to happen.
    """
    # Arrange
    fetcher = iter(range(4))

    # Act
    result = generic_regions(fetcher, 4, 10)

    # Assert
    assert result == ([], False)


def test_generic_regions_should_return_an_empty_page_when_offset_past_end():
    """Test the offset that runs past the end of the data.

    Given:
        An offset larger than the number of available records.
    When:
        A page is requested.
    Then:
        It should return an empty page as a pair.
    """
    # Arrange
    fetcher = iter(range(4))

    # Act
    result = generic_regions(fetcher, 100, 10)

    # Assert
    assert result == ([], False)



PROPERTY = settings(max_examples=200)


@PROPERTY
@given(
    n=st.integers(min_value=0, max_value=50),
    offset=st.integers(min_value=0, max_value=60),
    limit=st.integers(min_value=1, max_value=20),
)
def test_generic_regions_should_return_the_slice_the_offset_and_limit_name(
    n, offset, limit
):
    """Test paging as a slice of the underlying sequence.

    Given:
        Any record count, offset and page limit.
    When:
        A page is requested.
    Then:
        It should be exactly ``records[offset : offset + limit]``. Paging is a
        slice, and a slice is the kind of thing four examples cannot pin: the
        interesting cases are the offset landing on, one before and one past
        the end, at every page size.
    """
    # Arrange
    records = list(range(n))

    # Act
    rows, _ = generic_regions(iter(records), offset, limit)

    # Assert
    assert rows == records[offset : offset + limit]


@PROPERTY
@given(
    n=st.integers(min_value=0, max_value=50),
    offset=st.integers(min_value=0, max_value=60),
    limit=st.integers(min_value=1, max_value=20),
)
def test_generic_regions_should_report_a_next_page_exactly_when_records_remain(
    n, offset, limit
):
    """Test the has-next flag against what is left in the sequence.

    Given:
        Any record count, offset and page limit.
    When:
        A page is requested.
    Then:
        The flag should be true exactly when the slice does not reach the end.
        The flag is what a client loops on, so reporting it wrongly either
        truncates a listing or spins forever.
    """
    # Arrange
    records = list(range(n))

    # Act
    _, has_next = generic_regions(iter(records), offset, limit)

    # Assert
    assert has_next is (offset + limit < n)


@pytest.mark.xfail(
    strict=True,
    reason="a limit of 0 yields an empty page and reports no successor, "
    "conflating an empty page with exhausted data (#TBD)",
)
def test_generic_regions_should_report_a_next_page_when_the_limit_is_zero():
    """Test the degenerate page size.

    Given:
        Ten records and a page limit of zero.
    When:
        The first page is requested.
    Then:
        It should return nothing and report that a next page follows, because
        ten records still remain. It reports the opposite: ``grouper`` yields
        no chunk at all for ``n=0``, ``next`` raises ``StopIteration``, and the
        handler at ``vcf.py:43`` reads that as "the offset landed exactly at
        the end of the data" -- an empty *page* mistaken for exhausted *data*.
        A client paging on the flag stops with everything still unread.
    """
    # Act
    rows, has_next = generic_regions(iter(range(10)), 0, 0)

    # Assert
    assert rows == []
    assert has_next is True

class TestRegions:
    """The paging, exercised through its only caller.

    ``generic_regions`` takes any iterator, so the unit tests above run on
    ``range``. What they cannot show is that a pysam fetcher -- which is
    consumed lazily, cannot be rewound, and raises rather than returning empty
    once exhausted -- reaches the same four outcomes.
    """

    def test_regions_should_return_a_page_and_flag_more_when_data_follows(
        self, vcf, chromsizes
    ):
        """Test the ordinary paging case.

        Given:
            A VCF holding more variants than the page limit.
        When:
            The first page is requested.
        Then:
            It should return that page, in file order, and report that another
            follows.
        """
        # Act
        rows, has_next = mod.regions(vcf, chromsizes, 0, 3)

        # Assert
        assert [r["uid"] for r in rows] == ["rs1", "rs2", "rs3"]
        assert has_next is True

    def test_regions_should_report_no_more_on_the_last_page(
        self, vcf, chromsizes
    ):
        """Test the last-page case.

        Given:
            An offset far enough in that fewer variants remain than the limit.
        When:
            That page is requested.
        Then:
            It should return the remainder and report that nothing follows.
        """
        # Act
        rows, has_next = mod.regions(vcf, chromsizes, 3, 3)

        # Assert
        assert [r["uid"] for r in rows] == ["rs4", "rs5"]
        assert has_next is False

    def test_regions_should_return_an_empty_page_when_offset_lands_at_end(
        self, vcf, chromsizes
    ):
        """Test the offset that consumes exactly every variant.

        Given:
            An offset equal to the number of variants in the file.
        When:
            A page is requested.
        Then:
            It should return an empty page as a pair. The offset loop used to
            fall through here and call ``next`` on an exhausted grouper, which
            raises StopIteration out of the request handler.
        """
        # Act
        result = mod.regions(vcf, chromsizes, len(VARIANTS), 3)

        # Assert
        assert result == ([], False)

    def test_regions_should_return_an_empty_page_when_offset_past_end(
        self, vcf, chromsizes
    ):
        """Test the offset that runs past the end.

        Given:
            An offset larger than the number of variants.
        When:
            A page is requested.
        Then:
            It should return an empty page as a pair. This path used to return
            a four-key dict, which the caller then unpacked into two names.
        """
        # Act
        result = mod.regions(vcf, chromsizes, len(VARIANTS) + 10, 3)

        # Assert
        assert result == ([], False)

    def test_regions_should_place_each_variant_on_the_genome_axis(
        self, vcf, chromsizes
    ):
        """Test the coordinate translation.

        Given:
            A variant at position 50 on c2, whose offset is 1000.
        When:
            The page holding it is requested.
        Then:
            Its span should be shifted by that offset, and reported zero-based
            -- the file writes one-based positions and pysam converts.
        """
        # Act
        rows, _ = mod.regions(vcf, chromsizes, 2, 1)

        # Assert
        (row,) = rows
        assert row["uid"] == "rs3"
        assert (row["chrOffset"], row["xStart"], row["xEnd"]) == (1000, 1049, 1050)


class TestTilesetInfo:
    """The ladder a VCF advertises, which is derived purely from chromsizes."""

    def test_tileset_info_should_span_the_supplied_chromsizes(
        self, chromsizes
    ):
        """Test the declared extent.

        Given:
            A VCF and the chromsizes the browser supplies, since the file
            carries no coordinate system the server can trust.
        When:
            The info is read.
        Then:
            It should run to the genome length on both ends and advertise the
            max tile width past which a tile is refused rather than thinned.
        """
        # Act
        info = mod.tileset_info(vcf, chromsizes)

        # Assert
        assert info["max_width"] == genome.CANONICAL_TOTAL
        assert info["max_pos"] == [genome.CANONICAL_TOTAL]
        assert info["max_tile_width"] == 100000
