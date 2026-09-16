"""Shared test plumbing."""

import pytest

from test import fixtures


@pytest.fixture(scope="session")
def fixture_path():
    """Locate a test fixture, skipping the test when it cannot be had.

    ``fixture_path("exons.bed")`` for a committed one,
    ``fixture_path("alignment/multichrom.bam")`` for one listed in
    ``fixtures.toml`` -- see :mod:`test.fixtures`.
    """
    return fixtures.require
