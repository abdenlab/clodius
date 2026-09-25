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

from clodius.core.errors import (
    MalformedTileId,
    TilesetError,
    TilesetUnavailable,
    UnsupportedModifier,
    UnsupportedOption,
)
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
        It should raise ``TilesetUnavailable`` rather than ``MalformedTileId``.
        The defect being pinned is the exception's class, not the rejection.
        The arity is the tileset's own declaration and not part of the
        request, so a client told its well-formed id was malformed retries
        forever against a fault only the server can fix.
    """
    # Act & assert
    with pytest.raises(TilesetUnavailable, match="arity"):
        TileId.parse("abc.3.4", ndim=ndim)


@pytest.mark.parametrize("ndim", [0, -1])
def test_parse_should_not_raise_a_client_error_when_the_arity_is_invalid(ndim):
    """Test that a server-side misdeclaration is not blamed on the client.

    Given:
        A declared arity below one.
    When:
        A tile id is parsed against it.
    Then:
        It should not raise ``MalformedTileId``, whose contract is that the
        request itself is bad. Asserted separately from the class above
        because the two errors share a base: a test for the positive class
        alone would keep passing if the negative one were reintroduced
        alongside it.
    """
    # Act & assert
    with pytest.raises(TilesetError) as excinfo:
        TileId.parse("abc.3.4", ndim=ndim)
    assert not isinstance(excinfo.value, MalformedTileId)


@pytest.mark.parametrize(
    "tile_id,ndim",
    [
        ("abc.-1.0", 1),
        ("abc.3.-4", 1),
        ("abc.3.4", 0),
        ("abc.3", 1),
        ("abc.3.x", 1),
        # These three have no dedicated test of their own. Without them every
        # row above duplicates one, and this test cannot fail unless a more
        # specific one fails first.
        ("abc.3.4.mean", 1),
        ("abc.3.4.a.b", 1),
        ("abc.3.4,bogus", 1),
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
        instances of, asserted once over all of them. The last three rows are
        the only coverage of their rejection; the rest restate a dedicated
        test and are kept so the invariant is asserted over the whole set.
    """
    # Act & assert
    with pytest.raises(TilesetError):
        TileId.parse(tile_id, ndim=ndim)


@pytest.mark.parametrize(
    "tile_id",
    [
        "abc.3.+5",
        "abc.3.1_0",
        "abc.+3.5",
        "abc.3.\u00b2",
        "abc.3.\uff11",
    ],
)
def test_parse_should_raise_when_a_coordinate_is_not_ascii_decimal(tile_id):
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
    with pytest.raises(MalformedTileId):
        TileId.parse(tile_id, ndim=1)


def test_parse_should_raise_when_the_modifier_is_not_declared():
    """Test a modifier the tileset's spec does not list.

    Given:
        A spec accepting ``mean`` and ``sum``, and an id carrying ``avg``.
    When:
        It is parsed.
    Then:
        It should raise ``UnsupportedModifier`` naming what it does accept.
        The class is the contract: it is a ``MalformedTileId``, and so a
        sibling of ``TileError``, which means the request is refused outright
        rather than served as one tile carrying an error.
    """
    # Arrange
    spec = ModifierSpec(values=frozenset({"mean", "sum"}))

    # Act & assert
    with pytest.raises(UnsupportedModifier, match="mean"):
        TileId.parse("abc.3.4.avg", ndim=1, modifiers=spec)


def test_parse_should_raise_when_the_tileset_declares_no_modifiers():
    """Test a modifier against a tileset that accepts none at all.

    Given:
        A tileset declaring no modifier spec, and an id carrying one.
    When:
        It is parsed.
    Then:
        It should raise ``UnsupportedModifier``. A distinct cause from a spec
        that merely excludes the value, and a distinct branch: without it the
        trailing part is read as a coordinate the tileset never declared.
    """
    # Act & assert
    with pytest.raises(UnsupportedModifier):
        TileId.parse("abc.3.4.mean", ndim=1, modifiers=None)


def test_parse_should_raise_when_an_open_modifier_slot_is_empty():
    """Test the sentinel an ``allow_unknown`` spec must still refuse.

    Given:
        A spec that accepts unlisted values -- cooler's shape, where the
        modifier is a column name validated at fetch time -- and an id whose
        modifier slot is empty.
    When:
        It is parsed.
    Then:
        It should raise ``UnsupportedModifier``. Without this branch the empty
        string is accepted and the tileset looks up a column named ``""``
        much later, where the failure is no longer a tile-id problem.
    """
    # Arrange
    spec = ModifierSpec(values=frozenset({"default"}), allow_unknown=True)

    # Act & assert
    with pytest.raises(UnsupportedModifier, match="empty"):
        TileId.parse("abc.3.4.", ndim=1, modifiers=spec)


def test_parse_should_raise_when_an_option_is_not_key_value():
    """Test an option chunk that is not a pair at all.

    Given:
        An id whose option chunk carries no colon.
    When:
        It is parsed.
    Then:
        It should raise ``MalformedTileId`` and not ``UnsupportedOption``:
        the fault is the syntax, not an unrecognized key, and nothing else
        distinguishes the two.
    """
    # Act & assert
    with pytest.raises(MalformedTileId) as excinfo:
        TileId.parse("abc.3.4,bogus", ndim=1)
    assert not isinstance(excinfo.value, UnsupportedOption)


def test_parse_should_raise_when_a_second_part_trails_the_modifier():
    """Test an id with more trailing parts than the modifier slot holds.

    Given:
        A 1D id carrying two parts after its position.
    When:
        It is parsed.
    Then:
        It should raise ``MalformedTileId``. Only one trailing part is legal;
        without the length check a second is silently discarded and the client
        is served a tile it did not ask for.
    """
    # Act & assert
    with pytest.raises(MalformedTileId):
        TileId.parse("abc.3.4.a.b", ndim=1)


def test_parse_should_accept_the_zeroth_zoom_and_position():
    """Test the boundary the two negative guards sit on.

    Given:
        An id naming zoom zero and position zero.
    When:
        It is parsed.
    Then:
        It should parse. Every other success case here uses a non-zero zoom,
        so writing either guard as ``<= 0`` rather than ``< 0`` would pass the
        whole suite while refusing the top tile of every tileset -- the one a
        client asks for first.
    """
    # Act
    tid = TileId.parse("abc.0.0", ndim=1)

    # Assert
    assert (tid.z, tid.pos) == (0, (0,))


@pytest.mark.parametrize(
    "tile_id,expected",
    [("abc.-0.0", (0, (0,))), ("abc.00.007", (0, (7,))), ("abc.0.7", (0, (7,)))],
)
def test_parse_should_read_an_unnormalized_id_as_the_tile_it_names(
    tile_id, expected
):
    """Test the spellings that denote a tile without being its canonical id.

    Given:
        Ids carrying a negative zero or leading zeros.
    When:
        They are parsed.
    Then:
        Each should give the tile its digits name -- ``abc.-0.0`` is tile 0
        and ``abc.00.007`` is tile 7, the same tile as ``abc.0.7``. The answer
        is correct either way because ``raw`` is echoed back verbatim; only
        the cache key duplicates. Pinned because ``_is_int`` admits ``-0``
        while the guard below rejects anything negative, and tightening
        either one starts refusing tile 0.
    """
    # Act
    tid = TileId.parse(tile_id, ndim=1)

    # Assert
    assert (tid.z, tid.pos) == expected


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
