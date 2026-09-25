"""Tile identifier parsing.

Grammar::

    tile_id  := uid "." z ( "." coord )* ( "." modifier )? ( "," option )*
    option   := key ":" value

``uid`` must round-trip verbatim into the response key: the client matches
responses to requests by exact string.

``z`` is an index into the tileset's resolution ladder, coarsest first.

Modifiers are determined by a ``ModifierSpec``.
"""

from __future__ import annotations

from dataclasses import dataclass
from clodius.core.errors import (
    MalformedTileId,
    TilesetUnavailable,
    UnsupportedModifier,
    UnsupportedOption,
)

# Separator between the dotted tile position and any ``key:value`` options.
TILE_OPTIONS_CHAR = ","


def _is_int(text: str) -> bool:
    """Whether a dotted part is a coordinate rather than a modifier.

    A leading ``-`` counts, so that a negative coordinate is read as the
    coordinate it is and rejected by `TileId.parse` with a message
    naming the problem, rather than falling through to the modifier slot.

    ASCII decimal digits only, which is narrower than both of the obvious
    spellings. ``int`` would accept ``+5`` and ``1_0``; ``str.isdigit`` is the
    opposite error, admitting superscripts that ``int`` then rejects with a
    bare ``ValueError`` -- not a
    `clodius.core.errors.TilesetError`, so it escapes the server
    boundary as a 500; and ``str.isdecimal`` alone still admits fullwidth
    digits, where ``int`` succeeds and ``abc.3.１`` silently denotes tile 1.

    This does not close every alias. Leading zeros and a negative zero are
    not normalized, so a tile has more than one spelling: ``abc.-0.0`` and
    ``abc.0.0`` both denote tile 0, and ``abc.00.007`` and ``abc.0.7`` both
    denote tile 7. The answer is correct either way, since ``raw`` is echoed
    back verbatim; only the cache key duplicates.
    """
    return text.isascii() and (
        text.isdecimal() or (text[:1] == "-" and text[1:].isdecimal())
    )


@dataclass(frozen=True, slots=True)
class ModifierSpec:
    """What a tileset accepts in the modifier slot after the coordinates.

    A spec lets a tileset declare what modifers it accepts when a tile is
    requested. The slot is type-specific and overloaded across the codebase:

    ==========  =========================================  ===========
    tileset     values                                     meaning
    ==========  =========================================  ===========
    bigwig      mean/min/max/std/sum                       aggregation
    bigwig      minMax/whisker                             range mode (changes dimensionality!)
    bigbed      significant                                range mode
    cooler      default/None/<column name>                 transform
    ==========  =========================================  ===========
    """

    values: frozenset[str]
    default: str | None = None
    # Free-form label for what the slot signifies ('aggregation', 'transform', ...).
    kind: str = "modifier"
    # When set, `values` lists the recognized sentinels but any other non-empty
    # string is accepted too and the tileset validates meaning at fetch time.
    allow_unknown: bool = False

    def validate(self, value: str | None) -> str | None:
        if value is None:
            return self.default
        if value in self.values:
            return value
        if self.allow_unknown:
            if not value:
                raise UnsupportedModifier(f"empty {self.kind}")
            return value
        raise UnsupportedModifier(
            f"{value!r} is not a valid {self.kind}; expected one of "
            f"{sorted(self.values)}"
        )


@dataclass(frozen=True, slots=True)
class TileId:
    """A parsed tile identifier.

    Immutable and hashable so it can be used as a key.
    """

    uid: str
    z: int
    pos: tuple[int, ...]
    modifier: str | None = None
    # ``key:value`` options as a tuple of pairs, so the whole record stays
    # hashable. Use :meth:`option` to read one.
    options: tuple[tuple[str, str], ...] = ()
    # The original string, verbatim. MUST be used as the response key.
    raw: str = ""

    @property
    def z_and_pos(self) -> tuple[int, ...]:
        return (self.z, *self.pos)

    def option(self, key: str, default: str | None = None) -> str | None:
        for k, v in self.options:
            if k == key:
                return v
        return default

    @classmethod
    def parse(
        cls,
        tile_id: str,
        *,
        ndim: int,
        modifiers: ModifierSpec | None = None,
        options: frozenset[str] | None = None,
    ) -> TileId:
        """Parse ``tile_id`` against a tileset's declared shape.

        Parameters
        ----------
        ndim
            Number of coordinate slots after ``z``: 1 for vectors and 1D
            annotations, 2 for matrices and 2D annotations.
        modifiers
            What the trailing modifier slot may contain, if anything.
        options
            Recognized ``,key:value`` keys. ``None`` accepts any; an empty set
            rejects all.

        Notes
        -----
        ``ndim`` and ``modifiers`` are required to disambiguate coordinate
        slots from modifiers: with the arity fixed, a trailing non-numeric part
        is unambiguously the modifier.
        """
        # Checked before the id itself, and reported against the tileset
        # rather than the request. The arity is the tileset's own declaration,
        # so a client told its well-formed id was malformed retries forever
        # against a fault only the server can fix -- the same reasoning
        # `BaseTileset.parse_tile_id` applies to a missing `ndim`. Fatal to the
        # tileset, not renderable per tile: `TilesetUnavailable` is a sibling
        # of `TileError`, not a subclass.
        #
        # Without the check nothing raises loudly. `ndim=0` parses a
        # coordinate-less id without complaint, or reads the position as a
        # modifier; `ndim<0` slices its way to an `IndexError`. Neither is a
        # `TilesetError`, so neither reaches the boundary as anything a client
        # can read.
        if ndim < 1:
            raise TilesetUnavailable(
                f"invalid arity {ndim} declared for {tile_id!r}; a tile id "
                f"needs at least one coordinate"
            )

        head, _, opt_str = tile_id.partition(TILE_OPTIONS_CHAR)

        parsed_options: list[tuple[str, str]] = []
        if opt_str:
            for chunk in opt_str.split(TILE_OPTIONS_CHAR):
                key, sep, value = chunk.partition(":")
                if not sep:
                    raise MalformedTileId(
                        f"option {chunk!r} in {tile_id!r} is not 'key:value'"
                    )
                if options is not None and key not in options:
                    raise UnsupportedOption(
                        f"{key!r} is not a recognized option; expected one of "
                        f"{sorted(options)}"
                    )
                parsed_options.append((key, value))

        parts = head.split(".")
        expected = 1 + 1 + ndim  # uid + z + coords
        if len(parts) < expected or not all(
            _is_int(p) for p in parts[1:expected]
        ):
            raise MalformedTileId(
                f"{tile_id!r} does not match uid.z{'.pos' * ndim}; got "
                f"{len(parts)} dotted parts ({parts!r})"
            )

        uid = parts[0]
        numbers = [int(p) for p in parts[1:expected]]

        # Validated here rather than downstream. A negative position otherwise
        # reaches `Chromsizes.invert`, which raises a plain `ValueError` -- and
        # a `ValueError` is not a `TilesetError`, so the server boundary cannot
        # render it into a per-tile error payload and a single malformed id
        # takes down the whole batch.
        if numbers[0] < 0:
            raise MalformedTileId(
                f"negative zoom level {numbers[0]} in {tile_id!r}"
            )
        if any(n < 0 for n in numbers[1:]):
            raise MalformedTileId(f"negative tile position in {tile_id!r}")

        modifier_parts = parts[expected:]
        if len(modifier_parts) > 1:
            raise MalformedTileId(
                f"{tile_id!r} has {len(modifier_parts)} trailing parts after the "
                f"{ndim}D position; expected at most one modifier"
            )
        raw_modifier = modifier_parts[0] if modifier_parts else None

        if raw_modifier is not None and modifiers is None:
            raise UnsupportedModifier(
                f"{tile_id!r} carries modifier {raw_modifier!r} but this tileset "
                f"declares none"
            )
        modifier = modifiers.validate(raw_modifier) if modifiers else None

        return cls(
            uid=uid,
            z=numbers[0],
            pos=tuple(numbers[1:]),
            modifier=modifier,
            options=tuple(parsed_options),
            raw=tile_id,
        )

    def __str__(self) -> str:
        return self.raw or self.format()

    def format(self) -> str:
        """Re-serialize. Prefer :attr:`raw` when echoing a request back."""
        parts = [self.uid, str(self.z), *map(str, self.pos)]
        if self.modifier is not None:
            parts.append(self.modifier)
        out = ".".join(parts)
        for key, value in self.options:
            out += f"{TILE_OPTIONS_CHAR}{key}:{value}"
        return out
