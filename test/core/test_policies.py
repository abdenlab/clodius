"""Tests for clodius.core.policies.

This module is pure -- no file I/O -- so nothing here touches ``data/``.

Note the test names for ``with_`` carry two underscores before ``should``:
the naming rule concatenates literally, and the method's own name ends in one.
The guide's ``test___init___should_...`` example is the same rule applied to a
dunder.
"""

import subprocess
import sys

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from clodius.core.policies import (
    DEFAULT_POLICY,
    DensityPolicy,
    GridPolicy,
    TilePolicy,
    stable_importance,
    take_most_important,
)

#: Every field with its documented default, asserted in one comparison so a
#: typo'd default fails with a readable diff rather than five separate rows.
DEFAULTS = {
    "max_tile_width": None,
    "max_entries_per_tile": 1024,
    "max_query_size": 1_000_000,
    "max_unindexed_filesize": 20_000_000,
    "max_zoom_diff": 3,
    "max_chroms_sampled": 128,
}

#: The fields whose annotation admits None, meaning "no limit".
CAPPED_FIELDS = [
    "max_entries_per_tile",
    "max_query_size",
    "max_unindexed_filesize",
    "max_zoom_diff",
    "max_chroms_sampled",
]

RECORD_LISTS = st.lists(st.integers(), max_size=50)
CAPS = st.one_of(st.none(), st.integers(min_value=-5, max_value=60))


def identity(x):
    return x


class TestTilePolicy:
    """The server limits lifted out of the tileset modules."""

    def test___init___should_carry_the_documented_defaults(self):
        """Test the shipped defaults.

        Given:
            No constructor arguments.
        When:
            A policy is instantiated.
        Then:
            It should carry every documented default, since these are what a
            caller supplying no policy gets.
        """
        # Act
        policy = TilePolicy()

        # Assert
        assert {f: getattr(policy, f) for f in DEFAULTS} == DEFAULTS

    def test___init___should_carry_explicit_values(self):
        """Test that supplied values override the defaults.

        Given:
            An explicit value for every field.
        When:
            A policy is instantiated with them.
        Then:
            It should carry each supplied value.
        """
        # Arrange
        supplied = dict.fromkeys(DEFAULTS, 7)

        # Act
        policy = TilePolicy(**supplied)

        # Assert
        assert {f: getattr(policy, f) for f in DEFAULTS} == supplied

    @pytest.mark.parametrize("field", CAPPED_FIELDS)
    def test___init___should_accept_none_for_a_capped_field(self, field):
        """Test that "no limit" is expressible on every capped field.

        Given:
            None for one of the fields whose annotation admits it.
        When:
            A policy is instantiated with that field unset.
        Then:
            It should hold None, which the class docstring defines as no limit.
        """
        # Act
        policy = TilePolicy(**{field: None})

        # Assert
        assert getattr(policy, field) is None

    def test___setattr___should_raise_when_a_field_is_assigned(self):
        """Test that a policy cannot be mutated after construction.

        Given:
            A constructed policy.
        When:
            One of its fields is assigned.
        Then:
            It should raise, so variants can only be derived through with_ and
            a policy stays safe to share.
        """
        # Arrange
        policy = TilePolicy()

        # Act & assert
        with pytest.raises(AttributeError):
            policy.max_entries_per_tile = 5

    def test___eq___should_compare_by_field_values(self):
        """Test policy equality.

        Given:
            Two policies constructed with identical values, and a third
            differing in one field.
        When:
            They are compared.
        Then:
            The identical pair should compare equal and the third unequal.
        """
        # Act & assert
        assert TilePolicy() == TilePolicy()
        assert TilePolicy() != TilePolicy(max_entries_per_tile=8)

    def test___hash___should_collapse_equal_policies(self):
        """Test that a policy can key a cache.

        Given:
            Two equal policies.
        When:
            Both are used as keys in one dict.
        Then:
            It should hold a single entry.
        """
        # Act
        keyed = {TilePolicy(): "a", TilePolicy(): "b"}

        # Assert
        assert len(keyed) == 1

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

    def test_with__should_carry_the_untouched_fields_across(self):
        """Test that derivation changes only what it is asked to.

        Given:
            A policy with every field set to a non-default value.
        When:
            A variant is derived changing one field.
        Then:
            It should carry the new value for that field and the original's
            values for all the others.
        """
        # Arrange
        original = TilePolicy(**dict.fromkeys(DEFAULTS, 7))

        # Act
        derived = original.with_(max_query_size=99)

        # Assert
        assert derived.max_query_size == 99
        assert derived.max_entries_per_tile == 7
        assert derived.max_chroms_sampled == 7

    def test_with__should_apply_several_changes_at_once(self):
        """Test a multi-field derivation.

        Given:
            The default policy.
        When:
            A variant is derived changing two fields together.
        Then:
            It should carry both new values.
        """
        # Act
        derived = DEFAULT_POLICY.with_(
            max_entries_per_tile=8, max_unindexed_filesize=99
        )

        # Assert
        assert derived.max_entries_per_tile == 8
        assert derived.max_unindexed_filesize == 99

    def test_with__should_return_an_equal_copy_when_nothing_changes(self):
        """Test the no-op derivation.

        Given:
            The default policy.
        When:
            A variant is derived with no changes.
        Then:
            It should be equal to the original but a distinct object.
        """
        # Act
        derived = DEFAULT_POLICY.with_()

        # Assert
        assert derived == DEFAULT_POLICY
        assert derived is not DEFAULT_POLICY

    def test_with__should_raise_when_the_field_is_not_declared(self):
        """Test that a mistyped knob fails loudly.

        Given:
            The default policy.
        When:
            A variant is derived naming a field the policy does not declare.
        Then:
            It should raise TypeError, rather than silently dropping the
            change and serving with the default limit.
        """
        # Act & assert
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            DEFAULT_POLICY.with_(max_entries=5)


def test_default_policy_should_equal_a_freshly_constructed_policy():
    """Test the shipped default instance.

    Given:
        The module's default policy.
    When:
        It is compared with a freshly constructed one.
    Then:
        It should be equal, confirming the default applies today's hardcoded
        behavior unmodified.
    """
    # Act & assert
    assert DEFAULT_POLICY == TilePolicy()


class TestGridPolicy:
    """How chromosome-segmented rasters land on a genome-spanning grid."""

    def test_grid_policy_should_declare_only_the_implemented_strategy(self):
        """Test the declared vocabulary.

        Given:
            The grid policy enum.
        When:
            Its members are iterated.
        Then:
            It should yield exactly one member, valued "sequential", since
            that is the only resampling strategy implemented.
        """
        # Act & assert
        assert [m.value for m in GridPolicy] == ["sequential"]

    def test_grid_policy_should_reject_the_removed_scatter_strategy(self):
        """Test that the unimplemented alternative stays out.

        Given:
            The grid policy enum.
        When:
            It is looked up by the value "scatter".
        Then:
            It should raise ValueError, guarding against the positional
            alternative creeping back in undeclared.
        """
        # Act & assert
        with pytest.raises(ValueError):
            GridPolicy("scatter")

    def test_grid_policy_should_compare_equal_to_its_wire_string(self):
        """Test the string mixin.

        Given:
            The sequential member.
        When:
            It is compared with its plain string value.
        Then:
            It should compare equal, since tilesets expose it as a class
            attribute and downstream code treats it as a string.
        """
        # Act & assert
        assert GridPolicy.SEQUENTIAL == "sequential"


class TestDensityPolicy:
    """What to do when a tile holds more features than it can show."""

    def test_density_policy_should_declare_the_documented_strategies(self):
        """Test the declared vocabulary and its order.

        Given:
            The density policy enum.
        When:
            Its members are iterated.
        Then:
            It should yield stratified, subsampled and refused in declaration
            order.
        """
        # Act & assert
        assert [m.value for m in DensityPolicy] == [
            "stratified",
            "subsampled",
            "refused",
        ]

    @pytest.mark.parametrize("member", list(DensityPolicy), ids=lambda m: m.value)
    def test_density_policy_should_compare_equal_to_its_wire_string(
        self, member
    ):
        """Test the string mixin on every member.

        Given:
            Each declared density policy.
        When:
            It is compared with its documented string value.
        Then:
            It should compare equal.
        """
        # Act & assert
        assert member == member.value


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


def test_stable_importance_should_return_a_value_in_the_unit_interval():
    """Test the advertised output range.

    Given:
        A representative record key.
    When:
        Its importance is derived.
    Then:
        It should fall in [0, 1), the range the docstring advertises.
    """
    # Act
    result = stable_importance("chr1:100-200")

    # Assert
    assert 0.0 <= result < 1.0


@pytest.mark.integration
def test_stable_importance_should_survive_a_process_restart():
    """Test that the ranking is stable across processes.

    Given:
        The same record key evaluated in two fresh interpreters started with
        different hash seeds.
    When:
        Its importance is derived in each.
    Then:
        It should be identical, proving the value comes from a content digest
        rather than Python's per-process string hash -- so a server restart or
        a second worker does not reshuffle which features survive thinning.
    """
    # Arrange
    program = (
        "from clodius.core.policies import stable_importance;"
        "print(repr(stable_importance('chr1:100-200')))"
    )

    # Act
    outputs = [
        subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            check=True,
        ).stdout.strip()
        for seed in ("0", "1")
    ]

    # Assert
    assert outputs[0] == outputs[1]


def test_stable_importance_should_distinguish_different_keys():
    """Test that the ranking is not collapsed to a constant.

    Given:
        Several distinct record keys.
    When:
        Their importances are derived.
    Then:
        Each should differ, so thinning discriminates between records.
    """
    # Arrange
    keys = [f"chr1:{i}-{i + 10}" for i in range(20)]

    # Act
    values = [stable_importance(k) for k in keys]

    # Assert
    assert len(set(values)) == len(keys)


@pytest.mark.parametrize("key", ["", "chr1:100-200", "contig-é中"])
def test_stable_importance_should_accept_any_key(key):
    """Test the empty and non-ASCII key paths.

    Given:
        An empty key, an ordinary one, and one carrying non-ASCII characters.
    When:
        Each importance is derived.
    Then:
        It should return a value in range rather than raising, since the
        digest encodes explicitly rather than relying on a default codec.
    """
    # Act
    result = stable_importance(key)

    # Assert
    assert 0.0 <= result < 1.0


def test_stable_importance_should_spread_across_the_interval():
    """Test that thinning is not biased toward one end of the range.

    Given:
        A thousand distinct record keys.
    When:
        Their importances are derived.
    Then:
        They should be distinct and spread across the interval, so the cap
        keeps a representative sample rather than a clustered one.
    """
    # Arrange
    keys = [f"chr1:{i}-{i + 10}" for i in range(1000)]

    # Act
    values = [stable_importance(k) for k in keys]

    # Assert
    assert len(set(values)) == len(keys)
    assert min(values) < 0.05
    assert max(values) > 0.95
    assert 0.45 < sum(values) / len(values) < 0.55


@given(key=st.text())
@settings(max_examples=200)
def test_stable_importance_should_stay_in_range_for_any_key(key):
    """Test the output range over arbitrary keys.

    Given:
        Any text key.
    When:
        Its importance is derived.
    Then:
        It should always fall in [0, 1) and never raise.
    """
    # Act
    result = stable_importance(key)

    # Assert
    assert 0.0 <= result < 1.0


@given(key=st.text())
@settings(max_examples=200)
def test_stable_importance_should_be_repeatable_for_any_key(key):
    """Test determinism over arbitrary keys.

    Given:
        Any text key.
    When:
        Its importance is derived twice.
    Then:
        It should be identical, so tiles stay reproducible and cacheable for
        every feature key rather than only the ones a fixture happens to use.
    """
    # Act & assert
    assert stable_importance(key) == stable_importance(key)


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


def test_take_most_important_should_return_nothing_when_there_are_no_records():
    """Test the empty input.

    Given:
        An empty record list and a positive cap.
    When:
        The records are thinned.
    Then:
        It should return an empty list rather than raising.
    """
    # Act & assert
    assert take_most_important([], 5, identity) == []


def test_take_most_important_should_return_a_list_for_any_sequence():
    """Test that the caller's own sequence is never handed back.

    Given:
        Records supplied as a tuple.
    When:
        They are thinned with no limit.
    Then:
        It should return a list holding the same records, so a caller cannot
        mutate the result into the input.
    """
    # Act
    result = take_most_important((1, 2, 3), None, identity)

    # Assert
    assert result == [1, 2, 3]
    assert isinstance(result, list)


@pytest.mark.pinned
def test_take_most_important_should_break_ties_toward_later_records():
    """Test the tie-break among equally important records.

    Given:
        Four records all carrying the same importance, and a cap of two.
    When:
        They are thinned.
    Then:
        It should keep the last two, since the ranking sort is stable and the
        cap slices the tail. This pins observed behavior -- if earlier records
        should win instead, this is the assertion to invert alongside the fix.
    """
    # Arrange
    records = [("a", 1), ("b", 1), ("c", 1), ("d", 1)]

    # Act
    result = take_most_important(records, 2, lambda r: r[1])

    # Assert
    assert result == [("c", 1), ("d", 1)]


def test_take_most_important_should_accept_the_ranking_by_keyword():
    """Test the parameter name, which a live caller depends on.

    Given:
        A record list and a ranking function.
    When:
        Thinning is invoked passing the function as the ``importance``
        keyword, the way the bigbed tileset calls it.
    Then:
        It should thin normally, pinning the name as part of the signature.
    """
    # Act
    result = take_most_important([5, 1, 4], 2, importance=identity)

    # Assert
    assert result == [5, 4]


@given(records=RECORD_LISTS, cap=CAPS)
@settings(max_examples=200)
def test_take_most_important_should_return_a_subsequence(records, cap):
    """Test that thinning only ever removes.

    Given:
        Any record list and any cap, including None and non-positive values.
    When:
        The records are thinned.
    Then:
        Every returned record should appear in the input, in the same relative
        order -- so thinning never reorders, duplicates or invents a record.
    """
    # Act
    result = take_most_important(records, cap, identity)

    # Assert
    remaining = list(records)
    for item in result:
        assert item in remaining
        remaining = remaining[remaining.index(item) + 1 :]


@given(records=RECORD_LISTS, cap=st.integers(min_value=1, max_value=60))
@settings(max_examples=200)
def test_take_most_important_should_return_the_capped_count(records, cap):
    """Test the length law.

    Given:
        Any record list and any positive cap.
    When:
        The records are thinned.
    Then:
        It should return exactly as many records as the smaller of the input
        length and the cap.
    """
    # Act
    result = take_most_important(records, cap, identity)

    # Assert
    assert len(result) == min(len(records), cap)


@given(records=RECORD_LISTS, cap=CAPS)
@settings(max_examples=200)
def test_take_most_important_should_be_repeatable(records, cap):
    """Test determinism.

    Given:
        Any record list and any cap.
    When:
        The records are thinned twice with the same arguments.
    Then:
        It should return the same records both times, unlike a sampler.
    """
    # Act & assert
    assert take_most_important(records, cap, identity) == take_most_important(
        records, cap, identity
    )


@given(
    records=st.lists(st.integers(), max_size=50, unique=True),
    cap=st.integers(min_value=1, max_value=60),
)
@settings(max_examples=200)
def test_take_most_important_should_select_the_highest_ranked(records, cap):
    """Test that selection really is by importance.

    Given:
        Any list of distinct integers and any positive cap, ranked by their
        own value. Distinctness matters: "the top cap" is only well defined
        when no two records tie.
    When:
        The records are thinned.
    Then:
        It should return exactly the cap largest values, so selection is by
        rank rather than by position.
    """
    # Act
    result = take_most_important(records, cap, identity)

    # Assert
    assert set(result) == set(sorted(records)[-cap:])
