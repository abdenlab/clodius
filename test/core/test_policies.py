"""Tests for clodius.core.policies."""

import pytest

from clodius.core.policies import (
    DEFAULT_POLICY,
    TilePolicy,
    stable_importance,
    take_most_important,
)


def identity(x):
    return x


class TestTilePolicy:
    """Behavior of the server limits lifted out of the tileset modules."""

    def test_with__should_return_a_copy_carrying_the_change(self):
        """Test derivation of a policy variant.

        Given:
            The default policy.
        When:
            A variant is derived with one field changed.
        Then:
            It should carry the new value and leave the original untouched.
        """
        # Arrange
        original = DEFAULT_POLICY

        # Act
        derived = original.with_(max_chroms_sampled=8)

        # Assert
        assert derived.max_chroms_sampled == 8
        assert original.max_chroms_sampled == 128

    def test_with__should_accept_none_for_a_documented_no_limit_field(self):
        """Test that "no limit" is expressible, as the class docstring says.

        Given:
            The default policy.
        When:
            A variant is derived setting a cap to None.
        Then:
            It should accept it rather than reject the annotation.
        """
        # Act
        derived = TilePolicy().with_(max_entries_per_tile=None)

        # Assert
        assert derived.max_entries_per_tile is None


def test_stable_importance_should_be_deterministic_for_the_same_key():
    """Test that importance does not re-roll between requests.

    Given:
        The same record key.
    When:
        Its importance is derived twice.
    Then:
        It should be identical, so a feature does not flicker between zooms.
    """
    # Act
    first, second = stable_importance("chr1:100-200"), stable_importance(
        "chr1:100-200"
    )

    # Assert
    assert first == second
    assert 0.0 <= first < 1.0


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


def test_take_most_important_should_keep_the_ranked_records_in_input_order():
    """Test that thinning preserves genomic order rather than rank order.

    Given:
        More records than the cap.
    When:
        They are thinned.
    Then:
        It should return the highest-ranked ones, still in input order.
    """
    # Act
    result = take_most_important([5, 1, 4, 2, 3], 2, identity)

    # Assert
    assert result == [5, 4]


@pytest.mark.parametrize("cap", [0, -1])
def test_take_most_important_should_return_nothing_when_cap_is_nonpositive(
    cap,
):
    """Test the guard against an unbounded tile.

    Given:
        A non-empty record list and a cap of zero or less, as a server
        configured to serve no records would supply.
    When:
        The records are thinned.
    Then:
        It should return nothing. Without the guard ``ranked[-0:]`` is the
        whole list, so the cap emits everything it exists to suppress.
    """
    # Act
    result = take_most_important([1, 2, 3], cap, identity)

    # Assert
    assert result == []


def test_take_most_important_should_return_everything_when_cap_is_none():
    """Test that None means no limit, matching TilePolicy.

    Given:
        A record list and a cap of None.
    When:
        The records are thinned.
    Then:
        It should return all of them.
    """
    # Act
    result = take_most_important([1, 2, 3], None, identity)

    # Assert
    assert result == [1, 2, 3]
