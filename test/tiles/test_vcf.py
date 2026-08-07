"""Tests for clodius.tiles.vcf."""

from clodius.tiles.vcf import generic_regions


def test_generic_regions_should_return_a_page_and_flag_more_when_data_follows():
    """Test the ordinary paging case.

    Given:
        An iterator holding more records than the page limit.
    When:
        The first page is requested.
    Then:
        It should return that page and report that another follows.
    """
    # Arrange
    fetcher = iter(range(10))

    # Act
    rows, has_next = generic_regions(fetcher, 0, 4)

    # Assert
    assert rows == [0, 1, 2, 3]
    assert has_next is True


def test_generic_regions_should_report_no_more_when_the_page_exhausts_the_data():
    """Test the last-page case.

    Given:
        An iterator holding exactly one page of records.
    When:
        That page is requested.
    Then:
        It should return the records and report that nothing follows -- using
        one lookahead record rather than materializing a whole extra page,
        which at the wire's limit of 10000 meant parsing 10000 records to
        answer a yes/no question.
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
