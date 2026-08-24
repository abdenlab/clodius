"""Hypothesis strategies shared across test modules.

These live in an importable module rather than a ``conftest.py`` because
``@given`` is evaluated at decoration time and cannot take a fixture. Without a
shared home, each module reinvents ``chromsizes_pairs`` with its own bounds and
the property tests stop being comparable.

The chromosome-length strategy deliberately unions two regimes -- lengths far
longer than a tile and lengths shorter than one. ``test/core/test_coords.py``
discovered that the short regime is where bin-drift reconciliation actually
breaks; a single wide range almost never draws it.
"""

from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

#: Conventional chromosome names, split by naming convention. ``natsorted`` is
#: only a valid total order *within* one of these -- on arbitrary strings its
#: comparator fails both permutation invariance and idempotence, and across the
#: two conventions it is not even antisymmetric: ``_natcmp("chr1", "1")`` and
#: ``_natcmp("1", "chr1")`` both return 1, so which one sorts first depends on
#: which came first in the input. No real assembly mixes the two, which is why
#: the strategy draws from one at a time; the mixed case is pinned by
#: ``test_natsorted_should_order_a_prefixed_name_against_its_bare_spelling``.
PREFIXED_NAMES = (
    [f"chr{i}" for i in range(1, 23)]
    + ["chrX", "chrY", "chrM"]
    + [
        "chr1_KI270706v1_random",
        "chr2_KI270715v1_random",
        "chrUn_GL000195v1",
        "chrUn_KI270757v1",
    ]
)
BARE_NAMES = [f"{i}" for i in range(1, 23)] + ["X", "Y", "MT"]

CONVENTIONAL_NAMES = PREFIXED_NAMES + BARE_NAMES


def chrom_names():
    """A single conventional chromosome name."""
    return st.sampled_from(CONVENTIONAL_NAMES)


def conventional_chrom_names(min_size=1, max_size=25):
    """A shuffled list of distinct names from one naming convention.

    The domain on which ``natsorted`` is permutation-invariant and idempotent.
    Drawing from a single convention is not a convenience: mixing ``chr1`` with
    a bare ``1`` breaks the comparator's antisymmetry, and no assembly does it.
    """
    return st.sampled_from([PREFIXED_NAMES, BARE_NAMES]).flatmap(
        lambda pool: st.lists(
            st.sampled_from(pool),
            unique=True,
            min_size=min_size,
            max_size=min(max_size, len(pool)),
        )
    ).flatmap(st.permutations)


def chromsizes_pairs(min_size=1, max_size=8):
    """``[(name, length), ...]`` with unique names and positive lengths.

    Lengths draw from both the long and short regimes; see the module
    docstring.
    """
    lengths = st.one_of(
        st.integers(min_value=1, max_value=250_000_000),
        st.integers(min_value=1, max_value=5_000),
    )
    return st.lists(
        lengths, min_size=min_size, max_size=max_size
    ).map(lambda ls: [[f"c{i}", n] for i, n in enumerate(ls)])


def bed_records(chromsizes, min_size=0, max_size=12):
    """``[(chrom, start, end, name), ...]`` placed inside ``chromsizes``.

    Sorted by ``(contig order, start)``, which the BGZF+tabix builder requires
    and a plain BED shares so the two query paths stay comparable.

    Drawing the records is the point. A property test over a fixed five-record
    file explores only the tile address, which on the canonical genome is a
    domain of 7 or 30 values -- Hypothesis reports "Stopped because nothing
    left to do" and the budget buys nothing. It is also how a half-open
    interval bug survives: with records this sparse, no drawn tile has an edge
    landing exactly on one.
    """
    order = {name: i for i, (name, _) in enumerate(chromsizes)}

    def one(draw):
        name, length = draw(st.sampled_from(list(chromsizes)))
        start = draw(st.integers(min_value=0, max_value=max(0, int(length) - 1)))
        # ``end`` never draws 0. oxbow parses a BED ``end`` of 0 as null rather
        # than as the integer -- ``c1 0 0`` yields ``end=None`` while ``c1 5 5``
        # parses fine -- and the null then reaches arithmetic in
        # ``to_tile_record``. That is an upstream defect with a clodius
        # robustness gap behind it; it is pinned by
        # ``test_tiles_should_raise_a_type_error_on_a_record_ending_at_zero``
        # rather than drawn here, so this strategy explores tile geometry
        # instead of rediscovering one parser bug every run.
        end = draw(st.integers(min_value=max(start, 1), max_value=int(length)))
        return (name, start, end)

    @st.composite
    def _records(draw):
        n = draw(st.integers(min_value=min_size, max_value=max_size))
        raw = [one(draw) for _ in range(n)]
        raw.sort(key=lambda r: (order[r[0]], r[1]))
        return [
            (chrom, start, end, f"r{i}")
            for i, (chrom, start, end) in enumerate(raw)
        ]

    return _records()


def genomic_spans(max_coord=10**10):
    """A half-open ``(start, end)`` with ``start <= end``.

    Deliberately unbounded by any genome, so spans past the end are drawn.
    """
    return st.tuples(
        st.integers(min_value=0, max_value=max_coord),
        st.integers(min_value=0, max_value=max_coord),
    ).map(lambda p: (min(p), max(p)))


def uids():
    """A tileset uid containing none of the tile-id grammar's separators."""
    return st.text(
        alphabet=st.characters(blacklist_characters=".,:"), max_size=12
    )


def zoom_and_position(max_zoom=6):
    """``(z, x)`` with ``0 <= z <= max_zoom`` and ``0 <= x < 2**z``.

    The dependent draw matters: ``n_tiles`` varies per zoom, so drawing the two
    independently spends most examples out of bounds.
    """
    return st.integers(min_value=0, max_value=max_zoom).flatmap(
        lambda z: st.tuples(
            st.just(z), st.integers(min_value=0, max_value=2**z - 1)
        )
    )


def value_arrays(min_side=0, max_side=64, allow_nan=True):
    """1-D float64 arrays, optionally admitting NaN."""
    return hnp.arrays(
        dtype="float64",
        shape=st.integers(min_value=min_side, max_value=max_side),
        elements=st.floats(
            allow_nan=allow_nan, allow_infinity=False, width=32
        ),
    )
