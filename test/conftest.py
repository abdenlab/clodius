"""Suite-wide pytest configuration.

Two jobs: register Hypothesis profiles, and make a test that touches an
un-smudged git-LFS payload skip cleanly instead of failing with a format-
specific parse error.
"""

import pytest
from hypothesis import HealthCheck, settings

from .harness.lfs import lfs_available, unavailable

# Hypothesis's default deadline is 200 ms per example. That is already tight
# for the existing 200-example run against hg38's ~455 contigs, and a single
# `cooler.create_cooler` call costs ~117 ms on its own, so every profile
# disables it rather than trading real coverage for wall-clock superstition.
settings.register_profile("pure", max_examples=200, deadline=None)
settings.register_profile(
    "io",
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
settings.register_profile("ci", max_examples=50, deadline=None)
settings.load_profile("pure")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "lfs(*paths): skip unless every path holds real content rather than "
        "an un-smudged git-LFS pointer",
    )


def pytest_collection_modifyitems(config, items):
    """Skip anything marked ``lfs`` whose payloads are not materialized.

    A belt-and-braces net beneath ``harness.lfs.requires_lfs``: a test that
    forgets the decorator but carries the marker still skips rather than
    failing on stub bytes.
    """
    for item in items:
        for mark in item.iter_markers(name="lfs"):
            if not lfs_available(*mark.args):
                item.add_marker(
                    pytest.mark.skip(
                        reason="un-smudged LFS fixture(s): "
                        f"{unavailable(*mark.args)}; run 'git lfs pull'"
                    )
                )
