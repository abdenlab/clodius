"""Errors for the tile-serving layer.

Three groups::

    TilesetError
    |
    +-- TilesetUnavailable           the dataset cannot be served at all
    |
    +-- MalformedTileId              the request is bad; raised before tiles()
    |   +-- UnsupportedModifier
    |   +-- UnsupportedOption
    |
    +-- TileError                    this one tile cannot be served
        +-- TileOutOfBounds
        +-- TileTooWide
        +-- TileTooLarge

A :class:`TileError` is caught by ``tiles()``, converted into a dict, and
returned as that tile's payload, leaving other tile responses intact.

All other errors propagate to the server and poison the whole response.
"""

from clodius.core.tile import ErrorTilePayload


class TilesetError(Exception):
    """Base for every error the tile layer raises deliberately."""


class TilesetUnavailable(TilesetError):
    """The tileset cannot be served at all.

    An unreadable file, a missing index, a header that will not parse. Fatal to
    the whole request, not to one tile.
    """


class MalformedTileId(TilesetError):
    """A tile id could not be parsed against the tileset's declared shape."""


class UnsupportedModifier(MalformedTileId):
    """The tile id carries a modifier this tileset does not declare."""


class UnsupportedOption(MalformedTileId):
    """The tile id carries a ``,key:value`` option this tileset does not declare."""


class TileError(TilesetError):
    """Base for failures that affect one tile, leaving the batch servable."""

    def to_dict(self) -> ErrorTilePayload:
        """This error as a tile payload."""
        return {"error": str(self), "error_type": type(self).__name__}


class TileOutOfBounds(TileError):
    """The requested position does not exist at this zoom level."""


class TileTooWide(TileError):
    """The tile spans more of the coordinate space than policy allows."""


class TileTooLarge(TileError):
    """Serving the tile would require reading more data than policy allows.

    This is based on an estimated result size, not the coordinate span.
    """
