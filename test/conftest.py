"""Suite-wide pytest configuration: Hypothesis profiles.

Only ``pure`` is ever loaded. A per-module ``settings(...)`` alias -- ``PROPERTY``
and ``PROPERTY_IO`` in ``test/tiles_v2``, ``PROPERTY`` in ``test/core`` -- is how
a test departs from it, because the guide requires the decorator on the test
regardless of what profile happens to be active. An ``io`` profile registered
here was deleted for that reason: it described the right budget for the
file-backed property tests and no mechanism ever applied it to them.

Skipping on un-smudged git-LFS payloads is handled by
``harness.lfs.requires_lfs`` at each call site rather than by a collection
hook here. A ``@pytest.mark.lfs`` net used to sit beneath it as belt-and-
braces; it was removed because no test ever carried the marker, so it was a
second mechanism that could only ever agree with the first or do nothing.
"""

from hypothesis import settings

# Hypothesis's default deadline is 200 ms per example. That is already tight
# for the existing 200-example run against hg38's ~455 contigs, and a single
# `cooler.create_cooler` call costs ~117 ms on its own, so every profile
# disables it rather than trading real coverage for wall-clock superstition.
settings.register_profile("pure", max_examples=200, deadline=None)
settings.register_profile("ci", max_examples=50, deadline=None)
settings.load_profile("pure")
