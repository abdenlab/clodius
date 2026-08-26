"""Fixtures for ``clodius.core`` tests.

Thin wrappers over ``test.harness``; the genomes and builders themselves live
there so Hypothesis strategies can import them.
"""

import pytest

from ..harness import genome


@pytest.fixture
def canonical_chromsizes():
    """The 3000 bp tiling genome (``max_zoom`` 2, four tiles at that zoom)."""
    return genome.canonical()


@pytest.fixture
def minimal_chromsizes():
    """The 350 bp coordinate-arithmetic genome (collapses to ``max_zoom`` 0)."""
    return genome.minimal()
