"""Tests for the taxonomy in clodius.core.errors.

The shape of this hierarchy is load-bearing rather than decorative.
``BaseTileset.tiles`` catches exactly ``TileError`` and converts it into that
tile's payload; everything else propagates and fails the whole request. So a
class's *parent* decides whether a failure costs one tile or all sixteen, and
reparenting one silently changes that with no other test noticing.

The three groups the module docstring draws are asserted here directly: the
per-tile errors, the request errors raised before ``tiles()`` is ever called,
and ``TilesetUnavailable``, which is a sibling of ``TileError`` rather than a
subclass precisely so the boundary cannot swallow it.
"""

import pytest

from clodius.core.errors import (
    MalformedTileId,
    TileError,
    TileOutOfBounds,
    TilesetError,
    TilesetUnavailable,
    TileTooLarge,
    TileTooWide,
    UnsupportedModifier,
    UnsupportedOption,
)

#: Errors the per-tile boundary must render into a payload.
PER_TILE = [TileOutOfBounds, TileTooWide, TileTooLarge]

#: Errors that must escape it. A client fault or a server fault, but never
#: something one tile's payload slot can carry.
WHOLE_REQUEST = [
    TilesetUnavailable,
    MalformedTileId,
    UnsupportedModifier,
    UnsupportedOption,
]


@pytest.mark.parametrize("error", PER_TILE, ids=lambda e: e.__name__)
def test___init___should_produce_a_tile_error_for_a_per_tile_failure(error):
    """Test the classes the per-tile boundary is supposed to catch.

    Given:
        An error describing a failure of one tile.
    When:
        Its ancestry is examined.
    Then:
        It should be a ``TileError``. That is the only class ``tiles()``
        catches, so one that drifts out of this group stops being renderable
        and starts failing the whole batch instead.
    """
    # Act & assert
    assert issubclass(error, TileError)


@pytest.mark.parametrize("error", WHOLE_REQUEST, ids=lambda e: e.__name__)
def test___init___should_not_produce_a_tile_error_for_a_whole_request_failure(
    error,
):
    """Test the classes the per-tile boundary must let through.

    Given:
        An error describing a bad request or an unservable tileset.
    When:
        Its ancestry is examined.
    Then:
        It should NOT be a ``TileError``, while still being a
        ``TilesetError``. ``TilesetUnavailable`` in particular is a sibling by
        design: a file that cannot be opened is not something to report
        sixteen times over, once per tile.
    """
    # Act & assert
    assert not issubclass(error, TileError)
    assert issubclass(error, TilesetError)


@pytest.mark.parametrize(
    "error", [UnsupportedModifier, UnsupportedOption], ids=lambda e: e.__name__
)
def test___init___should_group_the_id_rejections_under_malformed(error):
    """Test the two rejections a caller may want to catch together.

    Given:
        An error raised while parsing a tile id.
    When:
        Its ancestry is examined.
    Then:
        It should be a ``MalformedTileId``, so a server can answer every bad
        id the same way without listing the subclasses.
    """
    # Act & assert
    assert issubclass(error, MalformedTileId)


def test_to_dict_should_name_the_error_and_its_type():
    """Test the payload a refused tile is served as.

    Given:
        A ``TileError`` carrying a message.
    When:
        It is converted to a tile payload.
    Then:
        It should carry both the message and the class name. The client
        branches on ``error_type``, so the key matters as much as the text --
        and nothing else in the suite reads it.
    """
    # Arrange
    error = TileOutOfBounds("tile position 9 is outside the 4 tiles at zoom 2")

    # Act
    payload = error.to_dict()

    # Assert
    assert payload == {
        "error": "tile position 9 is outside the 4 tiles at zoom 2",
        "error_type": "TileOutOfBounds",
    }
