"""Tests for the parse guards in clodius.core.tileid.

``TileId.parse`` sits at the server boundary: it is the first thing a request
string meets, and the errors it raises are the ones a server renders into a
per-tile error payload. Anything it lets through unvalidated becomes a
coordinate somewhere downstream, and anything it raises that is not a
``TilesetError`` escapes the boundary as a 500 and takes the whole batch with
it -- so the class of the exception is as much of the contract as the
rejection.
"""

import pytest

from clodius.core.errors import MalformedTileId, TilesetError
from clodius.core.tileid import ModifierSpec, TileId


def test_parse_should_return_the_parsed_id():
    """Test the ordinary case, so the guards cannot pass by rejecting all.

    Given:
        A well-formed 2D tile id.
    When:
        It is parsed.
    Then:
        It should carry the uid, zoom and both positions it names.
    """
    # Act
    tid = TileId.parse("abc.3.4.5", ndim=2)

    # Assert
    assert (tid.uid, tid.z, tid.pos) == ("abc", 3, (4, 5))


def test_parse_should_raise_when_the_zoom_is_negative():
    """Test the negative zoom level.

    Given:
        A tile id whose zoom is ``-1``.
    When:
        It is parsed.
    Then:
        It should raise ``MalformedTileId``. A zoom is an index into the
        resolution ladder, so a negative one indexes it backwards -- the
        request does not fail, it resolves to some other level's tile.
    """
    # Act & assert
    with pytest.raises(MalformedTileId, match="negative zoom level"):
        TileId.parse("abc.-1.0", ndim=1)


def test_parse_should_raise_when_a_position_is_negative():
    """Test a negative coordinate, a distinct slot from the zoom.

    Given:
        A tile id whose position is ``-4``.
    When:
        It is parsed.
    Then:
        It should raise ``MalformedTileId`` rather than carrying the negative
        through to ``Chromsizes.invert``, which raises a plain ``ValueError``
        the boundary cannot render.
    """
    # Act & assert
    with pytest.raises(MalformedTileId, match="negative tile position"):
        TileId.parse("abc.3.-4", ndim=1)


@pytest.mark.parametrize("ndim", [0, -1])
def test_parse_should_raise_when_the_declared_arity_is_invalid(ndim):
    """Test a tileset that mis-declares its own shape.

    Given:
        A declared arity below one.
    When:
        A tile id is parsed against it.
    Then:
        It should raise ``MalformedTileId``. The defect being pinned is the
        exception's class, not the rejection: a bare ``ValueError`` here is
        not a ``TilesetError``, so it escapes the server boundary as a 500
        instead of a per-tile error.
    """
    # Act & assert
    with pytest.raises(MalformedTileId, match="arity"):
        TileId.parse("abc.3.4", ndim=ndim)


@pytest.mark.parametrize(
    "tile_id,ndim",
    [
        ("abc.-1.0", 1),
        ("abc.3.-4", 1),
        ("abc.3.4", 0),
        ("abc.3", 1),
        ("abc.3.x", 1),
    ],
)
def test_parse_should_raise_a_tileset_error_for_any_rejection(tile_id, ndim):
    """Test that every rejection is renderable at the boundary.

    Given:
        A tile id or declared arity the parser rejects.
    When:
        It is parsed.
    Then:
        It should raise a ``TilesetError`` subclass, which is the only thing
        the server catches. This is the invariant the individual guards are
        instances of, asserted once over all of them -- so no row here is the
        sole coverage of any rejection, and trimming one loses nothing but
        this restatement.
    """
    # Act & assert
    with pytest.raises(TilesetError):
        TileId.parse(tile_id, ndim=ndim)


def test_parse_should_raise_when_a_coordinate_is_not_ascii_decimal():
    """Test the spellings a bare int conversion would accept.

    Given:
        A coordinate written with a sign, a digit separator, a superscript, or
        a fullwidth digit.
    When:
        The id is parsed.
    Then:
        It should raise ``MalformedTileId`` for all of them. Two of these are
        wrong answers rather than crashes -- ``int("1_0")`` is 10 and
        ``int("\uff11")`` is 1 -- so the client receives another tile's data
        under the key it asked for. The superscript is the crash: ``isdigit``
        admits it and ``int`` rejects it, and the resulting ``ValueError`` is
        not a ``TilesetError``, so it escapes the boundary as a 500.
    """
    # Act & assert
    for tile_id in (
        "abc.3.+5",
        "abc.3.1_0",
        "abc.+3.5",
        "abc.3.\u00b2",
        "abc.3.\uff11",
    ):
        with pytest.raises(MalformedTileId):
            TileId.parse(tile_id, ndim=1)


def test_parse_should_raise_when_there_are_too_few_positions():
    """Test an id shorter than the declared arity.

    Given:
        A tile id carrying one dotted part fewer than the tileset declares.
    When:
        It is parsed.
    Then:
        It should raise ``MalformedTileId``. Without the length check the
        coordinate slice simply comes up short and the id parses to an empty
        position tuple -- an under-dimensioned tile, served rather than
        refused.
    """
    # Act & assert
    with pytest.raises(MalformedTileId, match="dotted parts"):
        TileId.parse("abc.3", ndim=1)


def test_parse_should_raise_when_a_coordinate_is_not_a_number():
    """Test a non-numeric part in a coordinate slot.

    Given:
        A tile id whose position is a word.
    When:
        It is parsed.
    Then:
        It should raise ``MalformedTileId`` rather than letting the conversion
        raise a bare ``ValueError`` past the server boundary.
    """
    # Act & assert
    with pytest.raises(MalformedTileId, match="dotted parts"):
        TileId.parse("abc.3.x", ndim=1)


def test_parse_should_read_a_trailing_part_as_the_modifier():
    """Test the disambiguation the declared arity exists to provide.

    Given:
        A tileset declaring one coordinate slot and an aggregation modifier,
        and an id carrying both.
    When:
        The id is parsed.
    Then:
        The numeric part should be the coordinate and the trailing word the
        modifier. Nothing in the id itself says which slot is which: the arity
        fixes the coordinates, and whatever follows is the modifier.
    """
    # Arrange
    spec = ModifierSpec(values=frozenset({"mean", "sum"}))

    # Act
    tid = TileId.parse("abc.3.4.mean", ndim=1, modifiers=spec)

    # Assert
    assert (tid.pos, tid.modifier) == ((4,), "mean")


def test_parse_should_preserve_the_request_string_verbatim():
    """Test the round-trip the response key depends on.

    Given:
        A well-formed id carrying both a modifier and an option.
    When:
        It is parsed.
    Then:
        ``raw`` should equal the input byte for byte. The client matches
        responses to requests by exact string, so a normalized or truncated
        echo is a tile the client asked for and cannot find.
    """
    # Arrange
    spec = ModifierSpec(values=frozenset({"mean"}))
    tile_id = "abc.3.4.mean,cos:hg38"

    # Act
    tid = TileId.parse(
        tile_id, ndim=1, modifiers=spec, options=frozenset({"cos"})
    )

    # Assert
    assert tid.raw == tile_id
