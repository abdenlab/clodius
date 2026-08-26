"""Tests for clodius.tiles.files.

Two functions mapping a filename to a filetype and a filetype to a datatype.
They are pure and total, so the whole contract is a table -- which is why these
are parametrized rather than a run of assertions in one body.

There are three implementations of this pair in the tree: this one,
``clodius.tiles.utils``, and ``clodius.utils``, the last driven by a
``FILETYPES`` table rather than a chain of comparisons. Nothing in the repo
calls any of them, so all three are public API for downstream consumers, and a
consumer that picks the wrong one gets a different answer. The last two tests
here are what keeps that divergence visible until the three are consolidated.
"""

import pytest

import clodius.tiles.files as hgfi
import clodius.tiles.utils as hgut
import clodius.utils as cu

#: Every filename spelling the registries claim to recognize, plus one they
#: should not. Case is varied deliberately: ``infer_filetype`` lowercases the
#: extension it extracts, so the mixed-case spellings are contract, not
#: incidental.
FILETYPES = [
    ("matrix.mcool", "cooler"),
    ("matrix.cool", "cooler"),
    ("matrix.hic", "hic"),
    ("matrix.HiC", "hic"),
    ("a.bw", "bigwig"),
    ("a.bigwig", "bigwig"),
    ("a.bigWig", "bigwig"),
    ("series.htime", "time-interval-json"),
    ("vector.hitile", "hitile"),
    ("matrix.txt", None),
    ("no-extension", None),
]

DATATYPES = [
    ("cooler", "matrix"),
    ("hic", "matrix"),
    ("bigwig", "vector"),
    ("time-interval-json", "time-interval"),
    ("hitile", "vector"),
    ("not-a-filetype", None),
]


@pytest.mark.parametrize("filename,expected", FILETYPES, ids=lambda v: str(v))
def test_infer_filetype_should_map_an_extension_to_its_filetype(
    filename, expected
):
    """Test the extension table, including the unrecognized case.

    Given:
        A filename carrying a recognized extension in one of its accepted
        spellings, or an extension the registry does not know.
    When:
        Its filetype is inferred.
    Then:
        It should be the documented filetype, or None. Returning None rather
        than raising is what lets a caller fall back to an explicit filetype
        argument, so the None cases are contract rather than an accident of
        falling off the end.
    """
    # Act
    result = hgfi.infer_filetype(filename)

    # Assert
    assert result == expected


@pytest.mark.parametrize("filetype,expected", DATATYPES, ids=lambda v: str(v))
def test_infer_datatype_should_map_a_filetype_to_its_datatype(
    filetype, expected
):
    """Test the datatype table, including the unrecognized case.

    Given:
        A recognized filetype, or a string that is not one.
    When:
        Its datatype is inferred.
    Then:
        It should be the documented datatype, or None. ``hic`` and ``cooler``
        share ``matrix``: they are different containers for the same thing, and
        a client renders them with one track type.
    """
    # Act
    result = hgfi.infer_datatype(filetype)

    # Assert
    assert result == expected


@pytest.mark.parametrize(
    "filename", ["matrix.hic", "matrix.mcool", "a.bigwig", "vector.hitile"]
)
def test_infer_filetype_should_agree_across_the_three_registries(filename):
    """Test that the duplicated implementations return one answer.

    Given:
        A lowercase filename recognized by all three copies of this function.
    When:
        Each is asked.
    Then:
        They should agree. They did not for ``.hic`` when it was first added:
        only this module was wired, so a downstream caller reaching for either
        of the other two got None for a file clodius could serve.
    """
    # Act
    answers = {
        hgfi.infer_filetype(filename),
        hgut.infer_filetype(filename),
        cu.infer_filetype(filename),
    }

    # Assert
    assert len(answers) == 1


@pytest.mark.pinned
def test_infer_filetype_should_disagree_across_registries_on_case():
    """Test the divergence that adding .hic did not introduce and did not fix.

    Given:
        A filename whose extension is spelled in mixed case.
    When:
        Each of the three registries is asked.
    Then:
        The two in ``clodius.tiles`` should recognize it and the one in
        ``clodius.utils`` should not, because it compares the filename against
        a lowercased suffix without lowercasing the filename. The same is true
        of ``.MCOOL``, so this predates ``.hic`` and is recorded rather than
        repaired here -- the fix is consolidating the three, not patching one.
    """
    # Act & assert
    assert hgfi.infer_filetype("matrix.HiC") == "hic"
    assert hgut.infer_filetype("matrix.HiC") == "hic"
    assert cu.infer_filetype("matrix.HiC") is None
    assert cu.infer_filetype("matrix.MCOOL") is None
