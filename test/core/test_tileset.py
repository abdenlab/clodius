"""Tests for clodius.core.tileset.

Two guards, both about a declaration meaning what it says. ``BaseTileset``
hands its ``options`` set to the parser, where ``None`` is the wire for
"accept anything" -- so a tileset that declares it accepts no options must not
have that collapse into the opposite. ``TilesetInfo`` caches derivations off
its fields, which is only sound while the fields cannot move.
"""

from collections.abc import Hashable

import pytest
from pydantic import ValidationError

from clodius.core.coords import Chromsizes
from clodius.core.errors import (
    TileOutOfBounds,
    TilesetError,
    TilesetUnavailable,
    UnsupportedOption,
)
from clodius.core.tileid import ModifierSpec
from clodius.core.tileset import BaseTileset, DatasetInfo, TilesetInfo

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


class ServingTileset(BaseTileset):
    """A 1D tileset whose tile payload is just its own position.

    ``refuse`` names positions that raise a ``TileError``; ``explode`` makes
    every position raise something that is not one.
    """

    ndim = 1
    datatype = "bedlike"

    def __init__(self, refuse=frozenset(), explode=False):
        self._refuse = refuse
        self._explode = explode

    def _tile(self, tid):
        if self._explode:
            raise OSError("the file is gone")
        if tid.pos[0] in self._refuse:
            raise TileOutOfBounds(f"tile position {tid.pos[0]} refused")
        return [tid.pos[0]]


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
            It should raise ``TilesetUnavailable``, not ``AttributeError``.
            Only a ``TilesetError`` can be rendered at the server boundary and
            anything else escapes as a 500; and the arity is the tileset's own
            declaration rather than part of the request, so ``MalformedTileId``
            would blame the client for a fault it cannot fix and invite it to
            retry a well-formed id forever.
        """
        # Arrange
        tileset = UndeclaredTileset()

        # Act & assert
        with pytest.raises(TilesetUnavailable, match="ndim"):
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


class TestBaseTilesetTiles:
    """The per-tile error boundary, which lives here and not in ten copies.

    ``clodius.core.errors`` states it as a library-wide contract: a
    ``TileError`` is caught, converted to a dict, and returned as that tile's
    payload, leaving the other responses intact. Anything else is a
    whole-request failure and propagates.
    """

    def test_tiles_should_return_one_entry_per_requested_id(self):
        """Test the shape the protocol requires of every tileset.

        Given:
            A tileset over three tile ids, none of which refuse.
        When:
            They are served.
        Then:
            It should return one pair per id, carrying the id alongside its
            payload. The client matches responses to requests by raw tile id
            string, so a dropped entry is an unanswerable request.
        """
        # Arrange
        tileset = ServingTileset()
        ids = [tileset.parse_tile_id(f"abc.0.{x}") for x in (0, 1, 2)]

        # Act
        served = tileset.tiles(ids)

        # Assert
        assert [tid.pos[0] for tid, _ in served] == [0, 1, 2]
        assert [payload for _, payload in served] == [[0], [1], [2]]

    @pytest.mark.parametrize("options", [None, {}])
    def test_tiles_should_serve_the_batch_when_no_option_is_passed(
        self, options
    ):
        """Test the two spellings of "no options", which must both serve.

        Given:
            A tileset inheriting the default ``tiles``, and a batch passing
            either no options or an empty mapping.
        When:
            The batch is served.
        Then:
            It should serve every tile. An empty mapping is not a request for
            anything, so refusing it would break every caller that passes the
            parameter through unconditionally.
        """
        # Arrange
        tileset = ServingTileset()
        ids = [tileset.parse_tile_id("abc.0.0")]

        # Act
        served = tileset.tiles(ids, options)

        # Assert
        assert [payload for _, payload in served] == [[0]]

    def test_tiles_should_raise_when_an_option_is_passed_it_cannot_read(self):
        """Test the promise the base makes on behalf of everything under it.

        Given:
            A tileset inheriting the default ``tiles``, which understands no
            options, and a batch carrying one.
        When:
            The batch is served.
        Then:
            It should raise a ``TilesetError`` rather than serve. Options
            arrive once for the whole batch, so a bad one is a whole-batch
            failure -- and the default silently discarded them, serving tiles
            that ignored what was asked for.
        """
        # Arrange
        tileset = ServingTileset()
        ids = [tileset.parse_tile_id("abc.0.0")]

        # Act & assert
        with pytest.raises(TilesetError):
            tileset.tiles(ids, {"aggregation": "mean"})

    def test_tiles_should_return_a_refusal_as_that_tiles_payload(self):
        """Test that one refusal does not take the batch down with it.

        Given:
            A batch in which one tile raises ``TileOutOfBounds`` and the rest
            serve normally.
        When:
            The batch is served.
        Then:
            It should return an error payload in the refusing tile's slot and
            ordinary payloads in the others. Letting the exception escape would
            answer a batch of sixteen with a single failure, for a condition
            that is about one tile.
        """
        # Arrange
        tileset = ServingTileset(refuse={1})
        ids = [tileset.parse_tile_id(f"abc.0.{x}") for x in (0, 1, 2)]

        # Act
        served = tileset.tiles(ids)

        # Assert
        assert served[0][1] == [0] and served[2][1] == [2]
        assert served[1][1]["error"]

    def test_tiles_should_raise_when_the_failure_is_not_per_tile(self):
        """Test that the catch stays narrow.

        Given:
            A tileset whose ``_tile`` raises something that is not a
            ``TileError`` -- the shape an unreadable file takes.
        When:
            A batch is served.
        Then:
            It should propagate. Answering sixteen tiles with sixteen cheerful
            error payloads and a 200 would be a lie when the honest answer is
            that the dataset cannot be served at all.
        """
        # Arrange
        tileset = ServingTileset(explode=True)
        ids = [tileset.parse_tile_id("abc.0.0")]

        # Act & assert
        with pytest.raises(OSError):
            tileset.tiles(ids)


class TestDatasetInfo:
    """The base half of the info pair, and the promise it has to share."""

    def test___setattr___should_raise_when_a_field_is_assigned(self):
        """Test that the base makes the same immutability promise as the
        subclass.

        Given:
            A dataset info, the type ``Dataset.info()`` is annotated with.
        When:
            One of its fields is assigned.
        Then:
            It should raise. Both types are public and the subclass is frozen,
            so a base that accepted assignment would let code written against
            the published contract type-check and then fail only on a
            ``TilesetInfo`` -- which is exactly how the multivec row-metadata
            assignment survived review.
        """
        # Arrange
        info = DatasetInfo(min_pos=[0], max_pos=[100])

        # Act & assert
        with pytest.raises(ValidationError):
            info.max_pos = [200]

    def test_with__should_return_a_copy_carrying_the_changes(self):
        """Test the published way to derive a variant of a frozen info.

        Given:
            A dataset info and a field to change.
        When:
            ``with_`` is called.
        Then:
            It should return a new info carrying the change, leaving the
            original untouched.
        """
        # Arrange
        info = DatasetInfo(min_pos=[0], max_pos=[100])

        # Act
        derived = info.with_(max_pos=[200])

        # Assert
        assert derived.max_pos == [200]
        assert info.max_pos == [100]

    def test_with__should_raise_when_the_change_is_invalid(self):
        """Test the validation ``model_copy`` skips.

        Given:
            A value the constructor rejects.
        When:
            ``with_`` is called with it.
        Then:
            It should raise. ``model_copy(update=...)`` accepts it silently, so
            a derive path built on that would serve a value the type promises
            cannot exist.
        """
        # Arrange
        info = DatasetInfo(min_pos=[0], max_pos=[100])

        # Act & assert
        with pytest.raises(ValidationError):
            info.with_(max_width=-1)


class TestTilesetInfoDerivation:
    """Deriving a variant of an info that caches off its own fields."""

    def test_with__should_rebuild_the_cached_coordinate_system(self):
        """Test that a derived info does not keep the old coordinate system.

        Given:
            An info whose ``coordinate_system`` has already been computed, and
            a replacement set of chromsizes in a different order.
        When:
            ``with_`` is called with the new chromsizes.
        Then:
            It should serve canvases built from the new ordering.
            ``model_copy`` carries the cache across untouched, so the copy
            reports the new chromsizes through ``to_dict`` while placing every
            record by the old ones -- a wrong answer indistinguishable from a
            right one.
        """
        # Arrange
        info = TilesetInfo.quadtree(CHROMSIZES, 256)
        assert info.coordinate_system.names == ("c1", "c2")
        reversed_pairs = Chromsizes.from_pairs(
            [("c2", 200), ("c1", 100)]
        ).to_pairs()

        # Act
        derived = info.with_(chromsizes=reversed_pairs)

        # Assert
        assert derived.coordinate_system.names == ("c2", "c1")

    def test___hash___should_not_be_advertised(self):
        """Test that the type does not claim a capability no instance has.

        Given:
            A tileset info, whose ``min_pos``/``max_pos`` are lists.
        When:
            It is tested for hashability.
        Then:
            It should report unhashable. Pydantic synthesizes a hash for every
            frozen model, which would otherwise make ``isinstance(info,
            Hashable)`` true while ``hash(info)`` raised.
        """
        # Arrange
        info = TilesetInfo.quadtree(CHROMSIZES, 256)

        # Act & assert
        assert not isinstance(info, Hashable)
