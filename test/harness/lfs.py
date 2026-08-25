"""Detection of un-smudged git-LFS pointers under ``data/``.

``op.exists`` is not enough to decide whether a fixture is usable: the pointer
exists, so a bare existence check lets the suite run against a ~130-byte text
stub. Each format then fails in its own confusing way -- bigwig raises
``BBIReadError``, cooler raises ``OSError``, gzip raises ``BadGzipFile`` on the
literal bytes ``b've'`` (the start of ``version https://git-lfs``), and fasta
spins forever in ``fetch_sequence`` when the ``.fna`` is a pointer and its
``.fai`` sibling is real.

This module is the single place that knows the difference: every guarded read
of ``data/`` goes through :func:`requires_lfs`, including
``test/tiles/test_conformance.py``, which was the only file guarding its reads
at all and did so with a private copy of this logic.
"""

import os.path as op

import pytest

# An un-smudged LFS file is a short text stub, not the payload it stands for.
LFS_POINTER_MAGIC = b"version https://git-lfs"
LFS_POINTER_MAX_BYTES = 1024


def is_lfs_pointer(path):
    """Whether ``path`` is an un-smudged git-LFS pointer rather than content."""
    try:
        if op.getsize(path) > LFS_POINTER_MAX_BYTES:
            return False
        with open(path, "rb") as f:
            return f.read(len(LFS_POINTER_MAGIC)) == LFS_POINTER_MAGIC
    except OSError:
        return False


def unavailable(*paths):
    """The subset of ``paths`` that is missing or is an un-smudged pointer."""
    return [p for p in paths if not op.exists(p) or is_lfs_pointer(p)]


def requires_lfs(*paths):
    """Skip marker naming whichever of ``paths`` cannot be read."""
    missing = unavailable(*paths)
    return pytest.mark.skipif(
        bool(missing),
        reason=f"missing or un-smudged LFS fixture(s): {missing}. "
        "Run 'git lfs pull' to cover them.",
    )
