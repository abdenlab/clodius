"""Tests for clodius.core.tileid."""

import pytest

from clodius.core.errors import (
    MalformedTileId,
    UnsupportedModifier,
    UnsupportedOption,
)
from clodius.core.tileid import ModifierSpec, TileId

AGGREGATION = ModifierSpec(
    values=frozenset({"mean", "max"}), default="mean", kind="aggregation"
)


class TestModifierSpec:
    """Behavior of the declarative modifier-slot description."""

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
        assert tid.z_and_pos == (3, 4, 5)

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

    def test_option_should_return_the_default_when_the_key_is_absent(self):
        """Test option lookup for a key the tile id does not carry.

        Given:
            A parsed tile id with no options.
        When:
            An absent key is read with a default.
        Then:
            It should return the default.
        """
        # Arrange
        tid = TileId.parse("abc.3.4", ndim=1)

        # Act & assert
        assert tid.option("cos", "fallback") == "fallback"

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
        assert str(tid) == raw
