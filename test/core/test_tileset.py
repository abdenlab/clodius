"""Tests for clodius.core.tileset.

Two guards, both about a declaration meaning what it says. ``BaseTileset``
hands its ``options`` set to the parser, where ``None`` is the wire for
"accept anything" -- so a tileset that declares it accepts no options must not
have that collapse into the opposite. ``TilesetInfo`` caches derivations off
its fields, which is only sound while the fields cannot move.
"""

import pytest
from pydantic import ValidationError

from clodius.core.coords import Chromsizes
from clodius.core.errors import MalformedTileId, UnsupportedOption
from clodius.core.tileid import ModifierSpec
from clodius.core.tileset import BaseTileset, TilesetInfo

CHROMSIZES = Chromsizes.from_pairs([("c1", 100), ("c2", 200)])


class ClosedTileset(BaseTileset):
    """A tileset declaring that it recognizes no options at all."""

    ndim = 1


class CosTileset(BaseTileset):
    """A tileset declaring the one option the bigwig path reads."""

    ndim = 1
    options = frozenset({"cos"})


class MatrixTileset(BaseTileset):
    """A 2D tileset that also declares an aggregation modifier."""

    ndim = 2
    modifiers = ModifierSpec(values=frozenset({"mean", "sum"}))


class UndeclaredTileset(BaseTileset):
    """A tileset that forgets to declare its arity, as BaseTileset does."""


class TestBaseTileset:
    """The declared shape a tile id is parsed against."""

    def test_parse_tile_id_should_return_the_parsed_id(self):
        """Test the ordinary case, so the guards cannot pass by rejecting all.

        Given:
            A tileset and a well-formed 1D tile id carrying no options.
        When:
            The id is parsed.
        Then:
            It should carry the uid, zoom and position it names.
        """
        # Arrange
        tileset = ClosedTileset()

        # Act
        tid = tileset.parse_tile_id("abc.3.4")

        # Assert
        assert (tid.uid, tid.z, tid.pos) == ("abc", 3, (4,))

    def test_parse_tile_id_should_raise_when_no_option_is_declared(self):
        """Test the empty option set, which must reject rather than admit.

        Given:
            A tileset declaring an empty ``options`` frozenset, and a tile id
            carrying an option.
        When:
            The id is parsed.
        Then:
            It should raise ``UnsupportedOption``. An empty set declares that
            no key is recognized; passing it on as ``None`` tells the parser
            to accept every key, which is the opposite.
        """
        # Arrange
        tileset = ClosedTileset()

        # Act & assert
        with pytest.raises(UnsupportedOption, match="bogus"):
            tileset.parse_tile_id("abc.3.4,bogus:1")

    def test_parse_tile_id_should_accept_a_declared_option(self):
        """Test that declaring an option still admits it.

        Given:
            A tileset declaring ``cos``, and a tile id carrying it.
        When:
            The id is parsed.
        Then:
            It should return the option's value, so the fix above rejects only
            what is undeclared.
        """
        # Arrange
        tileset = CosTileset()

        # Act
        tid = tileset.parse_tile_id("abc.3.4,cos:hg38")

        # Assert
        assert tid.option("cos") == "hg38"

    def test_parse_tile_id_should_raise_when_the_option_is_not_declared(self):
        """Test a declared set against a key outside it.

        Given:
            A tileset declaring ``cos`` only, and a tile id carrying another
            key.
        When:
            The id is parsed.
        Then:
            It should raise ``UnsupportedOption``. This passes with or without
            the guard -- a non-empty frozenset survives ``or None`` unchanged
            -- so it is the control for the empty-set case above, not a second
            instance of it.
        """
        # Arrange
        tileset = CosTileset()

        # Act & assert
        with pytest.raises(UnsupportedOption, match="bogus"):
            tileset.parse_tile_id("abc.3.4,bogus:1")


    def test_parse_tile_id_should_return_the_declared_modifier(self):
        """Test that the tileset's modifier spec reaches the parser.

        Given:
            A tileset declaring a spec that includes ``mean``, and a tile id
            carrying it.
        When:
            The id is parsed.
        Then:
            It should read ``mean`` as the modifier. Asserting that an
            *undeclared* modifier is rejected would not pin this: a tileset
            passing no spec at all rejects every modifier too, for a different
            reason and with the same exception.
        """
        # Arrange
        tileset = MatrixTileset()

        # Act
        tid = tileset.parse_tile_id("abc.3.4.5.mean")

        # Assert
        assert tid.modifier == "mean"

    def test_parse_tile_id_should_read_as_many_positions_as_declared(self):
        """Test that the tileset's arity reaches the parser.

        Given:
            A tileset declaring two coordinate slots, and a well-formed 2D id.
        When:
            The id is parsed.
        Then:
            It should read both coordinates. The arity is what separates a
            coordinate from a modifier, so a tileset whose declaration does not
            reach the parser has its last coordinate read as a modifier.
        """
        # Arrange
        tileset = MatrixTileset()

        # Act
        tid = tileset.parse_tile_id("abc.3.4.5")

        # Assert
        assert tid.pos == (4, 5)

    def test_parse_tile_id_should_raise_when_the_arity_is_undeclared(self):
        """Test a tileset that forgets to declare its shape.

        Given:
            A subclass that declares no ``ndim``.
        When:
            a tile id is parsed against it.
        Then:
            It should raise ``MalformedTileId``, not ``AttributeError``. The
            distinction is the whole point: only a ``TilesetError`` can be
            rendered at the server boundary, and anything else escapes as a
            500 -- the same failure the parser's own arity guard exists to
            prevent, one level up.
        """
        # Arrange
        tileset = UndeclaredTileset()

        # Act & assert
        with pytest.raises(MalformedTileId, match="arity"):
            tileset.parse_tile_id("abc.3.4")


class TestTilesetInfo:
    """The served description of a tileset, and its immutability."""

    def test___setattr___should_raise_when_a_field_is_assigned(self):
        """Test that the model cannot be mutated after construction.

        Given:
            A tileset info built over a quadtree ladder.
        When:
            One of its fields is assigned.
        Then:
            It should raise. The canvas and zoom-count derivations are cached
            off these fields, so a mutated info would keep serving the
            derivations of the values it no longer holds.
        """
        # Arrange
        info = TilesetInfo.quadtree(CHROMSIZES, 256)

        # Act & assert
        with pytest.raises(ValidationError):
            info.tile_size = 999

    def test___setattr___should_raise_when_an_undeclared_field_is_assigned(
        self,
    ):
        """Test that ``extra="allow"`` does not leave a way in.

        Given:
            A tileset info, whose model config admits extra fields at
            construction so a tileset can carry type-specific keys.
        When:
            A name the model does not declare is assigned.
        Then:
            It should raise, so the extras are a construction-time affordance
            rather than a mutable side channel.
        """
        # Arrange
        info = TilesetInfo.quadtree(CHROMSIZES, 256)

        # Act & assert
        with pytest.raises(ValidationError):
            info.mirror_tiles = "false"

    def test___init___should_accept_an_undeclared_field(self):
        """Test the affordance the tilesets actually use.

        Given:
            A type-specific key supplied at construction, as the cooler
            tileset supplies ``mirror_tiles``.
        When:
            The info is built.
        Then:
            It should carry the key, so freezing the model did not close the
            door the tilesets come through.
        """
        # Act
        info = TilesetInfo.quadtree(CHROMSIZES, 256, mirror_tiles="false")

        # Assert
        assert info.to_dict()["mirror_tiles"] == "false"

    def test___setattr___should_raise_on_a_derived_copy(self):
        """Test that deriving a variant does not launder mutability back in.

        Given:
            A copy taken with ``model_copy(update=...)``, the documented way to
            derive a variant and the one the bigwig path uses.
        When:
            A field is assigned on the copy.
        Then:
            It should raise. Asserting only that the copy carries the new value
            would pin pydantic rather than this model: ``model_copy`` behaves
            identically on a mutable model, so such a test passes with
            ``frozen=True`` removed.
        """
        # Arrange
        info = TilesetInfo.quadtree(CHROMSIZES, 256)
        padded = info.model_copy(update={"max_pos": [info.max_width]})

        # Act & assert
        with pytest.raises(ValidationError):
            padded.tile_size = 999
