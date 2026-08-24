"""Tests for clodius.core.tileid."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from clodius.core.errors import (
    MalformedTileId,
    TileError,
    UnsupportedModifier,
    UnsupportedOption,
)
from clodius.core.tileid import ModifierSpec, TileId

AGGREGATION = ModifierSpec(
    values=frozenset({"mean", "max"}), default="mean", kind="aggregation"
)

# The grammar's separators are the only characters a uid or option key may not
# contain; everything else is fair game, including unicode.
UIDS = st.text(alphabet=st.characters(blacklist_characters=".,:"), max_size=8)
OPTION_KEYS = st.text(
    alphabet=st.characters(blacklist_characters=".,:"), min_size=1, max_size=6
)
# A colon is legal inside a value -- the split takes the first one only.
OPTION_VALUES = st.text(
    alphabet=st.characters(blacklist_characters=".,"), max_size=6
)
COORDS = st.integers(min_value=0, max_value=2**40)
GRAMMAR_ALPHABET = ".,:-+0123456789abc "


class TestModifierSpec:
    """Behavior of the declarative modifier-slot description."""

    def test___init___should_default_to_a_closed_set_with_no_default(self):
        """Test the declaration defaults.

        Given:
            Only a set of accepted values.
        When:
            A spec is constructed.
        Then:
            It should declare no default, the generic kind label, and a closed
            set -- so a tileset that says nothing more gets the strict
            reading.
        """
        # Act
        spec = ModifierSpec(values=frozenset({"significant"}))

        # Assert
        assert spec.default is None
        assert spec.kind == "modifier"
        assert spec.allow_unknown is False

    def test_validate_should_return_none_when_no_default_is_declared(self):
        """Test the unmodified case on a spec with no default.

        Given:
            A spec declaring no default.
        When:
            None is validated.
        Then:
            It should return None rather than inventing a value.
        """
        # Arrange
        spec = ModifierSpec(values=frozenset({"significant"}))

        # Act & assert
        assert spec.validate(None) is None

    def test_validate_should_accept_a_declared_value(self):
        """Test the plain accept path on a closed set.

        Given:
            A closed spec declaring mean and max.
        When:
            One of the declared values is validated.
        Then:
            It should return it unchanged.
        """
        # Act & assert
        assert AGGREGATION.validate("max") == "max"

    def test_validate_should_prefer_the_declared_set_when_open(self):
        """Test that an open set still recognizes its own sentinels.

        Given:
            An open spec whose declared values include the empty string.
        When:
            The empty string is validated.
        Then:
            It should be returned, because the declared-set branch runs first.
            The empty string is the arrangement rather than something more
            natural like ``"default"`` because it is the only input on which
            the two branches disagree: for every other declared value both
            return it unchanged, so deleting the declared-set branch entirely
            leaves the answer identical. Open admission alone rejects the
            empty string -- see
            ``test_validate_should_raise_when_an_open_value_is_empty`` -- so a
            spec that declares it and still returns it can only have taken the
            branch this test is named for.
        """
        # Arrange
        spec = ModifierSpec(
            values=frozenset({""}), allow_unknown=True, kind="transform"
        )

        # Act & assert
        assert spec.validate("") == ""

    def test_validate_should_raise_when_an_open_value_is_empty(self):
        """Test that open does not admit a blank modifier.

        Given:
            An open spec of a named kind.
        When:
            The empty string is validated.
        Then:
            It should raise UnsupportedModifier naming that kind, since an
            empty column name resolves to nothing at fetch time.
        """
        # Arrange
        spec = ModifierSpec(
            values=frozenset({"default"}), allow_unknown=True, kind="transform"
        )

        # Act & assert
        with pytest.raises(UnsupportedModifier, match="empty transform"):
            spec.validate("")

    def test_validate_should_return_the_default_when_no_modifier_is_given(
        self,
    ):
        """Test the unmodified case.

        Given:
            A spec declaring a default.
        When:
            None is validated.
        Then:
            It should return the default.
        """
        # Act & assert
        assert AGGREGATION.validate(None) == "mean"

    def test_validate_should_raise_when_the_value_is_not_declared(self):
        """Test rejection of an unrecognized modifier on a closed set.

        Given:
            A spec declaring a closed set of values.
        When:
            A value outside it is validated.
        Then:
            It should raise UnsupportedModifier.
        """
        # Act & assert
        with pytest.raises(UnsupportedModifier, match="aggregation"):
            AGGREGATION.validate("median")

    def test_validate_should_accept_an_unknown_value_when_the_set_is_open(
        self,
    ):
        """Test the open-set case, which cooler's column names require.

        Given:
            A spec allowing values outside its declared set.
        When:
            An undeclared non-empty value is validated.
        Then:
            It should pass it through for the instance to resolve.
        """
        # Arrange
        spec = ModifierSpec(
            values=frozenset({"default"}), allow_unknown=True, kind="transform"
        )

        # Act & assert
        assert spec.validate("KR") == "KR"


class TestTileId:
    """Behavior of the single tile-id parser for the whole layer."""

    def test___init___should_default_the_optional_fields(self):
        """Test direct construction.

        Given:
            Only a uid, zoom and position.
        When:
            A tile id is constructed.
        Then:
            It should default to no modifier, no options and an empty original
            string.
        """
        # Act
        tid = TileId(uid="abc", z=3, pos=(4,))

        # Assert
        assert tid.modifier is None
        assert tid.options == ()
        assert tid.raw == ""

    def test___setattr___should_raise_when_a_field_is_assigned(self):
        """Test immutability.

        Given:
            A constructed tile id.
        When:
            One of its fields is assigned.
        Then:
            It should raise, keeping the record safe to share across a batch.
        """
        # Arrange
        tid = TileId(uid="abc", z=3, pos=(4,))

        # Act & assert
        with pytest.raises(AttributeError):
            tid.z = 9

    def test___hash___should_collapse_identical_tile_ids(self):
        """Test that a tile id can key a cache.

        Given:
            Two tile ids built from identical field values.
        When:
            Both are put in a set.
        Then:
            It should hold one member, which is what lets a batch dedupe.
        """
        # Arrange
        first = TileId(uid="abc", z=3, pos=(4,), raw="abc.3.4")
        second = TileId(uid="abc", z=3, pos=(4,), raw="abc.3.4")

        # Act & assert
        assert len({first, second}) == 1

    @pytest.mark.pinned
    def test___eq___should_distinguish_by_the_request_string(self):
        """Test that the verbatim request participates in identity.

        Given:
            Two tile ids with identical coordinates but different original
            strings, as ``abc.3.4`` and ``abc.3.4,`` produce.
        When:
            They are compared.
        Then:
            It should report them unequal. Identity here is per request
            string rather than per tile, because ``raw`` is the key the
            response is built under and a client that asked for two spellings
            is owed two entries. The consequence is that a batch carrying both
            does not collapse to one and a cache keyed on this holds both --
            recorded as a deliberate trade rather than an accident, and open
            for revisiting if batch dedupe ever matters more than the response
            key.
        """
        # Arrange
        first = TileId(uid="abc", z=3, pos=(4,), raw="abc.3.4")
        second = TileId(uid="abc", z=3, pos=(4,), raw="abc.3.4,")

        # Act & assert
        assert first != second

    def test_z_and_pos_should_lead_with_the_zoom_in_one_dimension(self):
        """Test the flattened coordinate tuple for a vector tileset.

        Given:
            A 1D tile id constructed directly.
        When:
            Its flattened coordinates are read.
        Then:
            It should return the zoom followed by the single position.
        """
        # Act & assert
        assert TileId(uid="a", z=3, pos=(4,)).z_and_pos == (3, 4)

    def test_z_and_pos_should_keep_both_coordinates_in_two_dimensions(self):
        """Test the flattened coordinate tuple for a matrix tileset.

        Given:
            A 2D tile id constructed directly.
        When:
            Its flattened coordinates are read.
        Then:
            It should return the zoom followed by both positions in order.
        """
        # Act & assert
        assert TileId(uid="a", z=3, pos=(4, 5)).z_and_pos == (3, 4, 5)

    def test_option_should_return_the_value_when_the_key_is_present(self):
        """Test the ordinary option lookup.

        Given:
            A tile id carrying one option.
        When:
            That key is read.
        Then:
            It should return its value.
        """
        # Arrange
        tid = TileId(uid="a", z=0, pos=(0,), options=(("cos", "hg19"),))

        # Act & assert
        assert tid.option("cos") == "hg19"

    def test_option_should_return_the_default_when_the_key_is_absent(self):
        """Test option lookup for a key the tile id does not carry.

        Given:
            A tile id carrying no options, constructed directly so a parser
            regression cannot cascade into the option tests.
        When:
            An absent key is read with a default.
        Then:
            It should return the default.
        """
        # Arrange
        tid = TileId(uid="abc", z=3, pos=(4,))

        # Act & assert
        assert tid.option("cos", "fallback") == "fallback"

    def test_option_should_return_none_when_no_default_is_given(self):
        """Test lookup of an absent key with no fallback.

        Given:
            A tile id carrying no options.
        When:
            An absent key is read without a default.
        Then:
            It should return None.
        """
        # Act & assert
        assert TileId(uid="a", z=0, pos=(0,)).option("cos") is None

    def test_option_should_return_the_first_of_a_repeated_key(self):
        """Test a key the grammar admits more than once.

        Given:
            A tile id carrying the same option key twice with different
            values, which the grammar accepts without complaint.
        When:
            That key is read.
        Then:
            It should return the first value.
        """
        # Arrange
        tid = TileId(
            uid="a", z=0, pos=(0,), options=(("cos", "one"), ("cos", "two"))
        )

        # Act & assert
        assert tid.option("cos") == "one"

    def test_parse_should_split_uid_zoom_and_position(self):
        """Test the base grammar.

        Given:
            A well-formed 1D tile id.
        When:
            It is parsed.
        Then:
            It should expose the uid, zoom, position and the original string.
        """
        # Act
        tid = TileId.parse("abc.3.4", ndim=1)

        # Assert
        assert (tid.uid, tid.z, tid.pos, tid.raw) == ("abc", 3, (4,), "abc.3.4")

    def test_parse_should_read_both_coordinates_when_the_tileset_is_2d(self):
        """Test the arity the tileset declares.

        Given:
            A 2D tileset and a tile id with two coordinates.
        When:
            It is parsed.
        Then:
            It should expose both.
        """
        # Act
        tid = TileId.parse("abc.3.4.5", ndim=2)

        # Assert
        assert (tid.z, tid.pos) == (3, (4, 5))

    def test_parse_should_accept_the_zero_boundary(self):
        """Test the smallest legal coordinates.

        Given:
            A tile id whose zoom and position are both zero, sitting exactly
            on the non-negative boundary.
        When:
            It is parsed.
        Then:
            It should accept it, since the guards reject below zero rather
            than at it.
        """
        # Act
        tid = TileId.parse("abc.0.0", ndim=1)

        # Assert
        assert (tid.z, tid.pos) == (0, (0,))

    def test_parse_should_accept_an_empty_uid(self):
        """Test that the grammar places no constraint on the uid slot.

        Given:
            A tile id whose uid is empty.
        When:
            It is parsed.
        Then:
            It should accept it with an empty uid.
        """
        # Act
        tid = TileId.parse(".3.4", ndim=1)

        # Assert
        assert tid.uid == ""

    def test_parse_should_raise_when_there_are_too_few_dotted_parts(self):
        """Test rejection of a truncated tile id.

        Given:
            A 2D tileset and a tile id carrying one coordinate.
        When:
            It is parsed.
        Then:
            It should raise MalformedTileId.
        """
        # Act & assert
        with pytest.raises(MalformedTileId, match="dotted parts"):
            TileId.parse("abc.3.4", ndim=2)

    def test_parse_should_raise_when_there_are_two_trailing_parts(self):
        """Test the cap on the modifier slot.

        Given:
            A 1D tile id carrying two dotted parts after the position.
        When:
            It is parsed.
        Then:
            It should raise MalformedTileId, at most one modifier being
            allowed.
        """
        # Act & assert
        with pytest.raises(MalformedTileId, match="trailing parts"):
            TileId.parse("abc.3.4.mean.extra", ndim=1, modifiers=AGGREGATION)

    def test_parse_should_raise_when_a_2d_id_reaches_a_1d_tileset(self):
        """Test the arity mismatch, whose error names the wrong thing.

        Given:
            A 1D tileset declaring no modifier, handed a 2D-shaped tile id.
        When:
            It is parsed.
        Then:
            It should raise UnsupportedModifier complaining about a modifier
            '5' rather than MalformedTileId -- at this layer an extra
            coordinate is indistinguishable from a modifier, so a client
            debugging an arity mismatch is told about a modifier it never
            sent.
        """
        # Act & assert
        with pytest.raises(UnsupportedModifier, match="'5'"):
            TileId.parse("abc.3.4.5", ndim=1)

    def test_parse_should_raise_when_a_position_is_not_an_integer(self):
        """Test rejection of a non-numeric coordinate.

        Given:
            A tile id whose position is not a number.
        When:
            It is parsed.
        Then:
            It should raise MalformedTileId.
        """
        # Act & assert
        with pytest.raises(MalformedTileId, match="non-integer"):
            TileId.parse("abc.3.x", ndim=1)

    def test_parse_should_raise_when_the_zoom_is_not_an_integer(self):
        """Test that the zoom slot is checked as well as the coordinates.

        Given:
            A tile id whose zoom slot is not a number.
        When:
            It is parsed.
        Then:
            It should raise MalformedTileId.
        """
        # Act & assert
        with pytest.raises(MalformedTileId, match="non-integer position"):
            TileId.parse("abc.x.4", ndim=1)

    def test_parse_should_raise_when_the_zoom_level_is_negative(self):
        """Test rejection of a negative zoom.

        Given:
            A tile id with a negative zoom level.
        When:
            It is parsed.
        Then:
            It should raise MalformedTileId, which the server boundary can
            render, rather than letting a ValueError escape from downstream
            coordinate arithmetic.
        """
        # Act & assert
        with pytest.raises(MalformedTileId, match="negative zoom"):
            TileId.parse("abc.-1.5", ndim=1)

    def test_parse_should_raise_when_a_position_is_negative(self):
        """Test rejection of a negative coordinate.

        Given:
            A tile id with a negative tile position.
        When:
            It is parsed.
        Then:
            It should raise MalformedTileId rather than reaching
            ``Chromsizes.invert``, whose ValueError is not a TileError.
        """
        # Act & assert
        with pytest.raises(MalformedTileId, match="negative tile position"):
            TileId.parse("abc.3.-1", ndim=1)

    def test_parse_should_raise_when_the_second_coordinate_is_negative(self):
        """Test that every coordinate is guarded, not just the first.

        Given:
            A 2D tile id whose second coordinate is negative.
        When:
            It is parsed.
        Then:
            It should raise MalformedTileId.
        """
        # Act & assert
        with pytest.raises(MalformedTileId, match="negative tile position"):
            TileId.parse("abc.3.4.-5", ndim=2)

    def test_parse_should_raise_when_the_arity_is_negative(self):
        """Test the guard against a tileset that mis-declares its shape.

        Given:
            A well-formed tile id and a negative declared arity.
        When:
            It is parsed.
        Then:
            It should raise MalformedTileId rather than a bare IndexError from
            the coordinate slice, which is not a TileError and so would escape
            the server boundary as a 500.
        """
        # Act & assert
        with pytest.raises(MalformedTileId, match="negative arity"):
            TileId.parse("abc.3.4", ndim=-1)

    def test_parse_should_raise_when_a_digit_run_is_absurdly_long(self):
        """Test that the interpreter's own limit is translated.

        Given:
            A tile id whose coordinate is a digit run longer than CPython's
            integer-conversion limit.
        When:
            It is parsed.
        Then:
            It should raise MalformedTileId, so nothing outside the tile error
            hierarchy escapes and one absurd request cannot fail the batch
            with an untranslatable error.
        """
        # Arrange
        absurd = "9" * 5000

        # Act & assert
        with pytest.raises(MalformedTileId):
            TileId.parse(f"abc.3.{absurd}", ndim=1)

    def test_parse_should_fill_in_the_declared_default_modifier(self):
        """Test that an omitted modifier is resolved, not left empty.

        Given:
            A tileset whose modifier spec declares a default.
        When:
            A tile id with no modifier slot is parsed.
        Then:
            It should carry the default, which is why the re-serialized form
            can differ from the request string.
        """
        # Act
        tid = TileId.parse("abc.3.4", ndim=1, modifiers=AGGREGATION)

        # Assert
        assert tid.modifier == "mean"

    def test_parse_should_accept_a_declared_modifier(self):
        """Test the ordinary modifier path.

        Given:
            A tileset declaring mean and max.
        When:
            A tile id ending in one of them is parsed.
        Then:
            It should carry that value.
        """
        # Act
        tid = TileId.parse("abc.3.4.max", ndim=1, modifiers=AGGREGATION)

        # Assert
        assert tid.modifier == "max"

    def test_parse_should_carry_an_unknown_modifier_when_the_set_is_open(self):
        """Test the open modifier set cooler needs.

        Given:
            An open spec, as a tileset whose modifiers name per-file columns
            declares.
        When:
            A tile id carrying an undeclared value is parsed.
        Then:
            It should carry it through for the instance to resolve, since the
            grammar cannot know which columns a given file holds.
        """
        # Arrange
        transform = ModifierSpec(
            values=frozenset({"default", "None"}),
            default="default",
            kind="transform",
            allow_unknown=True,
        )

        # Act
        tid = TileId.parse("abc.3.4.KR", ndim=1, modifiers=transform)

        # Assert
        assert tid.modifier == "KR"

    def test_parse_should_raise_when_a_modifier_is_undeclared(self):
        """Test rejection of a modifier on a tileset that declares none.

        Given:
            A tileset with no modifier spec.
        When:
            A tile id carrying a modifier is parsed.
        Then:
            It should raise UnsupportedModifier.
        """
        # Act & assert
        with pytest.raises(UnsupportedModifier, match="declares none"):
            TileId.parse("abc.3.4.mean", ndim=1)

    def test_parse_should_raise_when_the_modifier_slot_is_blank(self):
        """Test that an open set still rejects an empty modifier.

        Given:
            An open spec and a tile id whose modifier slot is empty.
        When:
            It is parsed.
        Then:
            It should raise UnsupportedModifier reporting an empty slot, since
            open does not mean blank is meaningful.
        """
        # Arrange
        transform = ModifierSpec(
            values=frozenset({"default"}),
            kind="transform",
            allow_unknown=True,
        )

        # Act & assert
        with pytest.raises(UnsupportedModifier, match="empty transform"):
            TileId.parse("abc.3.4.", ndim=1, modifiers=transform)

    def test_parse_should_record_several_options_in_order(self):
        """Test the option tail.

        Given:
            A tile id carrying two options and a tileset accepting any key.
        When:
            It is parsed.
        Then:
            It should record both pairs in the order written.
        """
        # Act
        tid = TileId.parse("abc.3.4,cos:hg19,max:5", ndim=1)

        # Assert
        assert tid.options == (("cos", "hg19"), ("max", "5"))

    def test_parse_should_split_an_option_at_the_first_colon(self):
        """Test an option value containing a colon.

        Given:
            A tile id whose option value itself contains a colon.
        When:
            It is parsed.
        Then:
            It should split at the first colon only, keeping the value whole
            -- a genomic locus is a plausible option value.
        """
        # Act
        tid = TileId.parse("abc.3.4,cos:a:b", ndim=1)

        # Assert
        assert tid.option("cos") == "a:b"

    def test_parse_should_accept_an_empty_option_value(self):
        """Test an option with nothing after the colon.

        Given:
            A tile id whose option value is empty.
        When:
            It is parsed.
        Then:
            It should record an empty value rather than rejecting the option.
        """
        # Act
        tid = TileId.parse("abc.3.4,cos:", ndim=1)

        # Assert
        assert tid.option("cos") == ""

    def test_parse_should_accept_a_trailing_separator(self):
        """Test a tile id ending in a bare comma.

        Given:
            A tile id ending in the option separator with nothing after it.
        When:
            It is parsed.
        Then:
            It should accept it with no options -- unlike a doubled separator,
            which is rejected. The asymmetry is worth seeing.
        """
        # Act
        tid = TileId.parse("abc.3.4,", ndim=1)

        # Assert
        assert tid.options == ()

    def test_parse_should_raise_when_an_option_is_not_a_key_value_pair(self):
        """Test rejection of a malformed option.

        Given:
            A tile id whose option carries no colon.
        When:
            It is parsed.
        Then:
            It should raise MalformedTileId.
        """
        # Act & assert
        with pytest.raises(MalformedTileId, match="not 'key:value'"):
            TileId.parse("abc.3.4,bogus", ndim=1, options=None)

    def test_parse_should_raise_when_the_separator_is_doubled(self):
        """Test a tile id carrying an empty option chunk.

        Given:
            A tile id with two consecutive option separators.
        When:
            It is parsed.
        Then:
            It should raise MalformedTileId for the empty chunk.
        """
        # Act & assert
        with pytest.raises(MalformedTileId, match="not 'key:value'"):
            TileId.parse("abc.3.4,,cos:x", ndim=1)

    def test_parse_should_raise_when_the_option_set_is_empty(self):
        """Test that an empty option set rejects every option.

        Given:
            A tileset declaring an empty set of recognized option keys.
        When:
            A tile id carrying any option is parsed.
        Then:
            It should raise UnsupportedOption.
        """
        # Act & assert
        with pytest.raises(UnsupportedOption):
            TileId.parse("abc.3.4,cos:x", ndim=1, options=frozenset())

    def test_parse_should_accept_any_option_when_the_set_is_none(self):
        """Test the open-option case.

        Given:
            A tileset declaring None for its recognized option keys.
        When:
            A tile id carrying an arbitrary option is parsed.
        Then:
            It should accept it.
        """
        # Act
        tid = TileId.parse("abc.3.4,anything:1", ndim=1, options=None)

        # Assert
        assert tid.option("anything") == "1"

    def test_parse_should_raise_when_the_option_key_is_unrecognized(self):
        """Test rejection against a non-empty recognized set.

        Given:
            A tileset recognizing exactly one option key.
        When:
            A tile id carrying a different key is parsed.
        Then:
            It should raise UnsupportedOption listing what is recognized.
        """
        # Act & assert
        with pytest.raises(UnsupportedOption, match="bogus"):
            TileId.parse(
                "abc.3.4,bogus:1", ndim=1, options=frozenset({"cos"})
            )

    def test_parse_should_check_option_shape_before_recognition(self):
        """Test the precedence between the two option errors.

        Given:
            A tileset recognizing one key, handed a chunk that is both
            malformed and unrecognized.
        When:
            It is parsed.
        Then:
            It should raise MalformedTileId, the shape check running first.
        """
        # Act & assert
        with pytest.raises(MalformedTileId):
            TileId.parse("abc.3.4,bogus", ndim=1, options=frozenset({"cos"}))

    def test_parse_should_check_options_before_the_dotted_shape(self):
        """Test that the option tail is validated first.

        Given:
            A tile id whose head has too few dotted parts and whose option is
            unrecognized.
        When:
            It is parsed against an empty recognized set.
        Then:
            It should raise UnsupportedOption, since the tail is parsed before
            the head is counted.
        """
        # Act & assert
        with pytest.raises(UnsupportedOption):
            TileId.parse("nope,bad:1", ndim=1, options=frozenset())

    @given(
        uid=UIDS,
        z=COORDS,
        pos=COORDS,
        modifier=st.sampled_from(sorted(AGGREGATION.values)),
        options=st.lists(
            st.tuples(OPTION_KEYS, OPTION_VALUES), max_size=3
        ),
    )
    @settings(max_examples=200)
    def test_parse_should_recover_the_components_it_was_built_from(
        self, uid, z, pos, modifier, options
    ):
        """Test the parser against structurally generated tile ids.

        Given:
            Any tile id assembled from a uid free of the grammar's
            separators, a non-negative zoom and position, a declared modifier,
            and option pairs whose keys avoid colon and comma.
        When:
            The assembled string is parsed.
        Then:
            It should recover exactly the components that built it, and
            re-serialize to the identical string.
        """
        # Arrange
        tail = "".join(f",{k}:{v}" for k, v in options)
        raw = f"{uid}.{z}.{pos}.{modifier}{tail}"

        # Act
        tid = TileId.parse(raw, ndim=1, modifiers=AGGREGATION, options=None)

        # Assert
        assert (tid.uid, tid.z, tid.pos, tid.modifier) == (
            uid,
            z,
            (pos,),
            modifier,
        )
        assert tid.options == tuple(options)
        assert tid.format() == raw

    @given(text=st.text(alphabet=GRAMMAR_ALPHABET, max_size=30))
    @settings(max_examples=200)
    def test_parse_should_raise_only_tile_errors_for_any_input(self, text):
        """Test that the parser is total over its own alphabet.

        Given:
            Any text drawn from an alphabet rich in the grammar's separators,
            digits and minus signs.
        When:
            It is parsed against a fixed arity.
        Then:
            It should either return a tile id whose original string is the
            input verbatim and whose coordinates are non-negative, or raise a
            TileError. Nothing else may escape -- an untranslatable exception
            at this layer takes down a whole batch rather than one tile.
        """
        # Act
        try:
            tid = TileId.parse(text, ndim=1, modifiers=AGGREGATION)
        except TileError:
            return

        # Assert
        assert tid.raw == text
        assert tid.z >= 0
        assert len(tid.pos) == 1
        assert all(n >= 0 for n in tid.pos)

    @given(text=st.text(max_size=30))
    @settings(max_examples=200)
    def test_parse_should_raise_only_tile_errors_for_arbitrary_text(
        self, text
    ):
        """Test totality over unrestricted text.

        Given:
            Any text at all, including unicode digits and whitespace, which
            Python's int() accepts more readily than the grammar suggests.
        When:
            It is parsed against a fixed arity.
        Then:
            It should either parse or raise a TileError, never anything else.
        """
        # Act & assert
        try:
            TileId.parse(text, ndim=1, modifiers=AGGREGATION)
        except TileError:
            pass

    def test_format_should_append_the_modifier(self):
        """Test re-serialization of the modifier slot.

        Given:
            A tile id carrying a modifier and no options.
        When:
            It is re-serialized.
        Then:
            It should append the modifier as a fourth dotted part.
        """
        # Arrange
        tid = TileId(uid="abc", z=3, pos=(4,), modifier="mean")

        # Act & assert
        assert tid.format() == "abc.3.4.mean"

    def test_format_should_append_every_option(self):
        """Test re-serialization of the option tail.

        Given:
            A tile id carrying two options.
        When:
            It is re-serialized.
        Then:
            It should append each pair comma-separated and colon-joined, in
            order.
        """
        # Arrange
        tid = TileId(
            uid="abc", z=3, pos=(4,), options=(("cos", "hg19"), ("max", "5"))
        )

        # Act & assert
        assert tid.format() == "abc.3.4,cos:hg19,max:5"

    def test_format_should_differ_from_the_request_when_a_default_is_filled(
        self,
    ):
        """Test the normalization that makes raw and format diverge.

        Given:
            A tile id with no modifier slot, parsed against a spec declaring a
            default.
        When:
            It is re-serialized.
        Then:
            It should carry the filled-in default and so differ from the
            request string -- which is exactly why the original is what gets
            echoed back to the client.
        """
        # Arrange
        raw = "abc.3.4"

        # Act
        tid = TileId.parse(raw, ndim=1, modifiers=AGGREGATION)

        # Assert
        assert tid.format() == "abc.3.4.mean"
        assert tid.raw == raw

    def test_format_should_round_trip_a_parsed_tile_id(self):
        """Test re-serialization.

        Given:
            A tile id with a modifier and an option.
        When:
            It is parsed and re-formatted.
        Then:
            It should reproduce the original string.
        """
        # Arrange
        raw = "abc.3.4.mean,cos:xyz"

        # Act
        tid = TileId.parse(
            raw, ndim=1, modifiers=AGGREGATION, options=frozenset({"cos"})
        )

        # Assert
        assert tid.format() == raw

    @given(
        uid=UIDS,
        z=COORDS,
        pos=COORDS,
        modifier=st.sampled_from(sorted(AGGREGATION.values)),
    )
    @settings(max_examples=200)
    def test_format_should_reach_a_fixed_point_in_one_pass(
        self, uid, z, pos, modifier
    ):
        """Test that re-serialization converges rather than drifting.

        Given:
            Any structurally valid tile id, with an optional trailing
            separator that the parser silently drops.
        When:
            It is parsed, re-serialized, and that output parsed again.
        Then:
            The second parse should agree on every component and re-serialize
            identically. A plain ``parse(s).format() == s`` identity does not
            hold -- a filled-in default modifier and integer normalization
            both make the output differ from the request -- so the property
            worth pinning is that normalizing twice changes nothing more than
            normalizing once.
        """
        # Arrange
        raw = f"{uid}.{z}.{pos}.{modifier},"

        # Act
        once = TileId.parse(raw, ndim=1, modifiers=AGGREGATION, options=None)
        twice = TileId.parse(
            once.format(), ndim=1, modifiers=AGGREGATION, options=None
        )

        # Assert
        assert twice.format() == once.format()
        assert (twice.uid, twice.z, twice.pos, twice.modifier) == (
            once.uid,
            once.z,
            once.pos,
            once.modifier,
        )

    def test___str___should_reproduce_the_request_string(self):
        """Test that the verbatim request survives.

        Given:
            A tile id parsed from a string whose zoom carries padding.
        When:
            It is interpolated into a string.
        Then:
            It should reproduce the request exactly, padding included, since
            the client matches responses by exact string.
        """
        # Arrange
        raw = "abc. 3 .4"

        # Act
        tid = TileId.parse(raw, ndim=1)

        # Assert
        assert f"{tid}" == raw

    def test___str___should_fall_back_to_the_serialized_form(self):
        """Test a tile id that was never parsed from a request.

        Given:
            A tile id constructed directly, with no original string recorded.
        When:
            It is interpolated into a string.
        Then:
            It should fall back to re-serializing itself.
        """
        # Arrange
        tid = TileId(uid="abc", z=3, pos=(4,))

        # Act & assert
        assert f"{tid}" == "abc.3.4"

