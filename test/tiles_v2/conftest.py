"""Fixtures for ``clodius.tiles_v2`` tests.

Factory fixtures take parameters rather than returning one fixed file, so a
test needing an unknown contig or a one-resolution ladder passes it in instead
of asking for a second fixture.

Scope follows build cost. bed, bigWig, bigBed and multivec all build in 1-3 ms,
so they are function-scoped over ``tmp_path``. A cooler costs several hundred ms
and a .hic around 175, which is why ``shared_mcool`` and ``shared_hic`` are
session-scoped -- note that mixing one with a function-scoped fixture in a
single test is a ``ScopeMismatch`` error, not a warning, so a test needing a
*modified* file must build its own.
"""

import pytest

from ..harness import builders, genome


@pytest.fixture
def chromsizes():
    """The canonical 3000 bp coordinate system the fixtures are placed on."""
    return genome.canonical()


@pytest.fixture
def make_bed(tmp_path):
    """``make_bed(records=..., name=...) -> Path`` for a plain BED."""

    def _make(records=builders.DEFAULT_RECORDS, name="tiny.bed"):
        return builders.build_bed(tmp_path / name, records)

    return _make


@pytest.fixture
def make_bed_bgzf(tmp_path):
    """``make_bed_bgzf(records=..., name=...) -> Path`` for BGZF + tabix."""

    def _make(records=builders.DEFAULT_RECORDS, name="tiny.bed.gz"):
        return builders.build_bed_bgzf(tmp_path / name, records)

    return _make


@pytest.fixture
def make_bigwig(tmp_path):
    """``make_bigwig(chromsizes=..., values=...) -> Path``."""

    def _make(
        chromsizes=genome.CANONICAL_CHROMSIZES, values=None, name="tiny.bw"
    ):
        return builders.build_bigwig(tmp_path / name, chromsizes, values)

    return _make


@pytest.fixture
def make_bigbed(tmp_path):
    """``make_bigbed(chromsizes=..., step=..., width=...) -> Path``."""

    def _make(
        chromsizes=genome.CANONICAL_CHROMSIZES,
        step=200,
        width=50,
        name="tiny.bb",
    ):
        return builders.build_bigbed(tmp_path / name, chromsizes, step, width)

    return _make


@pytest.fixture
def make_mv5(tmp_path):
    """``make_mv5(chromsizes=..., resolutions=..., n_rows=...) -> Path``."""

    def _make(name="tiny.mv5", **kwargs):
        return builders.build_mv5(tmp_path / name, **kwargs)

    return _make


@pytest.fixture(scope="session")
def shared_mcool(tmp_path_factory):
    """One multi-resolution cooler for the whole session (~350 ms to build)."""
    path = tmp_path_factory.mktemp("cooler") / "tiny.mcool"
    return builders.build_mcool(path)


@pytest.fixture
def make_mcool(tmp_path):
    """``make_mcool(chromsizes=..., resolutions=..., weights=...) -> Path``.

    Session-scoped ``shared_mcool`` covers the default fixture; this is for the
    tests that need a *different* cooler -- a balancing column, a single-entry
    ladder -- and so cannot share one.
    """

    def _make(name="tiny.mcool", **kwargs):
        return builders.build_mcool(tmp_path / name, **kwargs)

    return _make


@pytest.fixture(scope="session")
def shared_hic(tmp_path_factory):
    """One multi-resolution .hic for the whole session (~175 ms to build)."""
    path = tmp_path_factory.mktemp("hic") / "tiny.hic"
    return builders.build_hic(path)


@pytest.fixture
def make_hic(tmp_path):
    """``make_hic(chromsizes=..., resolutions=..., seed=...) -> Path``.

    Session-scoped ``shared_hic`` covers the default fixture; this is for the
    tests that need a *different* file -- a single-entry ladder, a genome whose
    contig order discriminates -- and so cannot share one.
    """

    def _make(name="tiny.hic", **kwargs):
        return builders.build_hic(tmp_path / name, **kwargs)

    return _make
