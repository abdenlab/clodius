"""Tests for clodius.core.policies.

Only the two functions whose guards this branch restores: the record cap, and
the digest the cap ranks on. The rest of the module -- ``TilePolicy``,
``LinkPolicy``, ``reconcile``, ``reconcile_2d`` -- is covered when the full
suite lands.

A cap is a safety limit, so its edges are the whole point: zero and ``None``
sit next to each other in the signature and mean opposite things, and the
slice that implements the limit reads ``ranked[-cap:]``, which silently
inverts at zero.
"""

import hashlib

import pytest

from clodius.core.policies import (
    TilePolicy,
    stable_importance,
    take_most_important,
)

#: Importance values as this implementation produces them today, recorded so
#: that a change of digest, encoding, slice or divisor has to be deliberate.
#: A same-process double call cannot do this job: it proves only that the
#: function is not random, and a ``hash(key)`` implementation -- which
#: reshuffles on every interpreter restart -- passes it green.
GOLDEN = {
    "": 0.8285759000573307,
    "chr1:100-200": 0.14890028489753604,
    "contig-\u00e9\u4e2d": 0.5237986561842263,
}


def identity(x):
    """Rank a record by its own value."""
    return x


class TestTilePolicy:
    """The limits themselves, validated where all four capping sites pass."""

    @pytest.mark.parametrize(
        "field", ["max_span", "max_records", "max_scan_bytes"]
    )
    def test___init___should_raise_when_a_limit_is_negative(self, field):
        """Test the guard covering every site that reads a limit.

        Given:
            A policy constructed with one limit set negative.
        When:
            It is constructed.
        Then:
            It should raise ``ValueError``. A negative ``max_records`` meant
            four different things at the four capping sites, and one of them
            -- polars' ``top_k`` -- raised ``OverflowError``, which is not a
            ``TilesetError`` and so escaped the per-tile boundary and failed
            the whole batch.
        """
        # Act & assert
        with pytest.raises(ValueError, match=field):
            TilePolicy(**{field: -1})

    @pytest.mark.parametrize(
        "field", ["max_span", "max_records", "max_scan_bytes"]
    )
    def test___init___should_accept_a_limit_of_zero(self, field):
        """Test that the guard rejects negatives without rejecting zero.

        Given:
            A policy constructed with one limit set to zero.
        When:
            It is constructed.
        Then:
            It should succeed. Zero is a meaningful setting -- "serve nothing"
            -- and rejecting it would break the very configuration the cap
            exists to honour.
        """
        # Act
        policy = TilePolicy(**{field: 0})

        # Assert
        assert getattr(policy, field) == 0

    def test_with__should_raise_when_the_change_is_negative(self):
        """Test that deriving a policy re-validates rather than bypassing.

        Given:
            A valid policy.
        When:
            A copy is derived with a negative cap.
        Then:
            It should raise ``ValueError``. ``dataclasses.replace`` runs
            ``__init__``, so the guard is inherited rather than restated --
            but only a test says so.
        """
        # Arrange
        policy = TilePolicy()

        # Act & assert
        with pytest.raises(ValueError, match="max_records"):
            policy.with_(max_records=-1)


def test_take_most_important_should_return_everything_when_under_the_cap():
    """Test the pass-through case.

    Given:
        Fewer records than the cap.
    When:
        They are thinned.
    Then:
        It should return all of them, in their original order.
    """
    # Act
    result = take_most_important([1, 2, 3], 10, identity)

    # Assert
    assert result == [1, 2, 3]


def test_take_most_important_should_return_everything_when_cap_is_exact():
    """Test the inclusive boundary.

    Given:
        Exactly as many records as the cap.
    When:
        They are thinned.
    Then:
        It should return all of them, pinning that the comparison is
        inclusive rather than off by one.
    """
    # Act
    result = take_most_important([1, 2, 3], 3, identity)

    # Assert
    assert result == [1, 2, 3]


def test_take_most_important_should_keep_the_ranked_records_in_input_order():
    """Test that thinning preserves genomic order rather than rank order.

    Given:
        More records than the cap, arranged so the two survivors appear in the
        input in the opposite order to their rank.
    When:
        They are thinned.
    Then:
        It should return the highest-ranked ones in input order -- ``[4, 5]``,
        not ``[5, 4]``. The arrangement is what makes that observable: with
        the survivors already in descending rank, input order and rank order
        are the same list and reversing the output changes nothing.
    """
    # Act
    result = take_most_important([4, 5, 1, 2, 3], 2, identity)

    # Assert
    assert result == [4, 5]


def test_take_most_important_should_return_nothing_when_cap_is_zero():
    """Test the guard against an unbounded tile.

    Given:
        A non-empty record list and a cap of zero, as a server configured to
        serve no records would supply.
    When:
        The records are thinned.
    Then:
        It should return nothing. Without the guard ``ranked[-0:]`` is the
        whole list, so the cap emits everything it exists to suppress.
    """
    # Act
    result = take_most_important([1, 2, 3], 0, identity)

    # Assert
    assert result == []


def test_take_most_important_should_return_nothing_when_cap_is_negative():
    """Test the negative cap, a distinct branch from zero and from None.

    Given:
        A non-empty record list and a cap of minus one.
    When:
        The records are thinned.
    Then:
        It should return nothing.
    """
    # Act
    result = take_most_important([1, 2, 3], -1, identity)

    # Assert
    assert result == []


def test_take_most_important_should_return_everything_when_cap_is_none():
    """Test that None means no limit, matching TilePolicy.

    Given:
        A record list and a cap of None, which is what ``TilePolicy`` supplies
        when ``max_records`` is unset.
    When:
        The records are thinned.
    Then:
        It should return all of them rather than raising on the length
        comparison.
    """
    # Act
    result = take_most_important([1, 2, 3], None, identity)

    # Assert
    assert result == [1, 2, 3]


def test_take_most_important_should_return_nothing_when_there_are_no_records():
    """Test the empty input.

    Given:
        An empty record list and a positive cap.
    When:
        The records are thinned.
    Then:
        It should return an empty list rather than raising. No branch of the
        current implementation is falsified by this -- both the pass-through
        and the ranked path reach ``[]`` -- so it stands as a no-raise smoke
        test rather than as coverage of the cap.
    """
    # Act & assert
    assert take_most_important([], 5, identity) == []


@pytest.mark.parametrize("key", sorted(GOLDEN), ids=lambda k: repr(k))
def test_stable_importance_should_equal_its_recorded_value(key):
    """Test the property the whole ranking rests on.

    Given:
        A feature key -- empty, ASCII, and non-ASCII -- with the value this
        implementation produced when the test was written.
    When:
        Its importance is derived.
    Then:
        It should equal that value exactly. The ranking has to be identical
        across processes, workers and releases, not merely within one call:
        a tile cached by one worker and a tile served by another must agree on
        which records survived thinning, and a feature that survives at one
        zoom must survive at the next.
    """
    # Act
    result = stable_importance(key)

    # Assert
    assert result == GOLDEN[key]


def test_stable_importance_should_distinguish_different_keys():
    """Test that the digest discriminates rather than merely returning.

    Given:
        A thousand distinct feature keys.
    When:
        Their importance is derived.
    Then:
        Every value should differ. A ranking that collapses keys into a few
        buckets thins arbitrarily within each one, which is the failure a
        constant return value or a truncated digest produces.
    """
    # Act
    values = {stable_importance(f"chr1:{i}-{i + 10}") for i in range(1000)}

    # Assert
    assert len(values) == 1000


def test_stable_importance_should_return_a_value_in_the_unit_interval():
    """Test the documented range.

    Given:
        An assortment of feature keys.
    When:
        Their importance is derived.
    Then:
        It should fall in ``[0, 1)``, which is what lets the value stand in
        for a precomputed score.
    """
    # Act
    values = [stable_importance(f"chr1:{i}-{i + 10}") for i in range(64)]

    # Assert
    assert all(0.0 <= v < 1.0 for v in values)


def test_stable_importance_should_return_a_value_when_md5_is_restricted(
    monkeypatch,
):
    """Test the digest against a build that refuses md5 as a security hash.

    Given:
        A ``hashlib.md5`` that raises unless told the digest is not being used
        for security, as it does on a FIPS-enforcing build.
    When:
        A key's importance is derived.
    Then:
        It should return the same value every other host does. Asserting only
        that some number comes back would also pass for an implementation that
        catches the error and falls back to another digest -- the most likely
        way someone "fixes" a FIPS failure -- which silently gives those hosts
        a different ranking from the rest of the fleet.
    """
    # Arrange
    real_md5 = hashlib.md5

    def fips_md5(data=b"", *, usedforsecurity=True):
        if usedforsecurity:
            raise ValueError("[digital envelope routines] unsupported")
        return real_md5(data, usedforsecurity=False)

    monkeypatch.setattr(hashlib, "md5", fips_md5)

    # Act
    result = stable_importance("chr1:100-200")

    # Assert
    assert result == GOLDEN["chr1:100-200"]


def test_take_most_important_should_return_a_new_list():
    """Test that the caller's own sequence is never handed back.

    Given:
        Records supplied as a tuple, under each cap that passes them through
        unthinned.
    When:
        They are thinned.
    Then:
        It should be a new list every time. The bigBed path hands this result
        straight out as the tile payload, so returning the caller's sequence
        would alias a tileset's own state into a response.
    """
    # Arrange
    records = (1, 2, 3)

    # Act
    results = [take_most_important(records, cap, identity) for cap in (None, 10, 3)]

    # Assert
    assert all(isinstance(r, list) and r == [1, 2, 3] for r in results)


def test_take_most_important_should_keep_one_record_per_cap_slot():
    """Test that the cap counts records, not distinct importances.

    Given:
        Three records of which two rank identically, and a cap of one.
    When:
        They are thinned.
    Then:
        It should return exactly one record. Selecting by value rather than by
        position returns both of the tied records here -- twice the cap, from
        an implementation that looks correct and passes every other test in
        this file.
    """
    # Act
    result = take_most_important([3, 3, 1], 1, identity)

    # Assert
    assert result == [3]


def test_take_most_important_should_accept_an_unhashable_record():
    """Test the shape the only caller actually passes.

    Given:
        Records as dicts, ranked by their ``importance`` key, as the bigBed
        tileset supplies them.
    When:
        They are thinned to two.
    Then:
        It should return the two highest. A value-keyed implementation raises
        ``TypeError: unhashable type: 'dict'`` here, so this is the same
        defect as above reaching the real call site.
    """
    # Arrange
    records = [
        {"uid": "a", "importance": 0.1},
        {"uid": "b", "importance": 0.9},
        {"uid": "c", "importance": 0.5},
    ]

    # Act
    result = take_most_important(records, 2, lambda r: r["importance"])

    # Assert
    assert [r["uid"] for r in result] == ["b", "c"]


def test_take_most_important_should_keep_the_first_of_equally_ranked_records():
    """Test the tie-break, which the cap reaches on any unranked format.

    Given:
        Four records of identical importance and a cap of two.
    When:
        They are thinned.
    Then:
        It should keep the first two. This pins observed behavior rather than
        a settled contract: the heap is fed in input order and breaks a tie
        toward what it saw first. It answered ``["c", "d"]`` while the
        selection was a stable sort sliced from the tail -- the records
        returned changed, their count and their relative order did not.
    """
    # Act
    result = take_most_important(["a", "b", "c", "d"], 2, lambda r: 1.0)

    # Assert
    assert result == ["a", "b"]
