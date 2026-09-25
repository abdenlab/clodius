"""Tests for clodius.tiles_v2._index.

A tile's query regions come from the tileset's chromsizes; an index only knows
the contigs it was built over. Screening the first against the second is what
keeps a single-chromosome file servable against a whole assembly, so the two
things this module promises are worth pinning directly: that it reports the
contigs a range query may name, and that it says so in a way a caller can act
on when it cannot tell.

``None`` is the load-bearing case. It means "do not screen" rather than "no
contigs" -- a format whose index this cannot read must end up exactly where it
was before, not with every query dropped.
"""

import pysam
import pytest

from clodius.core.coords import GenomicRange
from clodius.tiles_v2._index import indexed_contigs, screen_regions


@pytest.fixture
def indexed_bed(tmp_path):
    """A BGZF-compressed, tabix-indexed BED carrying one contig."""
    plain = tmp_path / "one.bed"
    plain.write_text("c1\t0\t5\tr0\nc1\t10\t15\tr1\n")
    gz = str(plain) + ".gz"
    pysam.tabix_compress(str(plain), gz, force=True)
    pysam.tabix_index(gz, preset="bed", force=True)
    return gz


def ranges(*names):
    """One in-bounds range per contig name."""
    return [GenomicRange(i, name, 0, 10) for i, name in enumerate(names)]


def test_indexed_contigs_should_report_the_contigs_the_index_carries(
    indexed_bed,
):
    """Test the set a range query may legally name.

    Given:
        A tabix-indexed BED whose records all sit on one contig.
    When:
        Its index is read.
    Then:
        It should report that contig alone. A tabix index lists what it saw,
        not what a header or a chromsizes file declares, which is the whole
        reason a screen is needed.
    """
    # Act
    contigs = indexed_contigs(indexed_bed)

    # Assert
    assert contigs == frozenset({"c1"})


def test_indexed_contigs_should_return_none_when_there_is_no_index(tmp_path):
    """Test a file whose index cannot be read at all.

    Given:
        A plain, unindexed BED.
    When:
        Its index is read.
    Then:
        It should return ``None``, not an empty set. The two mean opposite
        things to a caller: ``None`` leaves the regions alone, while an empty
        set would drop every one of them and serve an empty tile.
    """
    # Arrange
    plain = tmp_path / "plain.bed"
    plain.write_text("c1\t0\t5\tr0\n")

    # Act & assert
    assert indexed_contigs(str(plain)) is None


def test_screen_regions_should_drop_a_contig_the_index_does_not_carry():
    """Test the screen that keeps an unanswerable query from being issued.

    Given:
        Ranges on three contigs and an index carrying one.
    When:
        They are screened.
    Then:
        Only the carried contig should survive. The dropped ones have no
        records by definition, so this changes nothing a client sees -- it
        only removes the queries that would have raised.
    """
    # Act
    kept = screen_regions(ranges("c1", "c2", "c3"), frozenset({"c1"}))

    # Assert
    assert [gr.name for gr in kept] == ["c1"]


def test_screen_regions_should_keep_everything_when_the_index_is_unknown():
    """Test the fallback the whole design rests on.

    Given:
        Ranges on three contigs and no knowledge of the index.
    When:
        They are screened.
    Then:
        All three should survive. A format this cannot introspect has to
        behave exactly as it did before the screen existed.
    """
    # Act
    kept = screen_regions(ranges("c1", "c2", "c3"), None)

    # Assert
    assert [gr.name for gr in kept] == ["c1", "c2", "c3"]
