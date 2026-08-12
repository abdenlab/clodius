"""Tile identifier parsing.

One parser, replacing the ~12 ad-hoc ``tile_id.split(".")`` sites.

Grammar::

    tile_id  := uid "." z ( "." coord )* ( "." modifier )? ( "," option )*
    option   := key ":" value

``uid`` must round-trip verbatim into the response key -- the client matches
responses to requests by exact string. :attr:`TileId.raw` preserves it, which is
what lets us retire ``cooler``'s ``transform_id_to_original_id`` bookkeeping.

``z`` is an index into the tileset's resolution ladder, coarsest first. It means
the same thing for every tileset; only the ladder's *serialization* varies. See
:class:`~clodius.core.tileset.Ladder`.
"""

from __future__ import annotations

from dataclasses import dataclass

from clodius.core.errors import (
    MalformedTileId,
    UnsupportedModifier,
    UnsupportedOption,
)

# Separator between the dotted tile position and any ``key:value`` options.
# Mirrors ``clodius.utils.TILE_OPTIONS_CHAR``; kept here so the parser is
# self-contained.
TILE_OPTIONS_CHAR = ","


@dataclass(frozen=True, slots=True)
class ModifierSpec:
    """What a tileset accepts in the modifier slot after the coordinates.

    The slot is type-specific and overloaded across the codebase:

    ==========  =========================================  ===========
    tileset     values                                     meaning
    ==========  =========================================  ===========
    bigwig      mean/min/max/std/sum                       aggregation
    bigwig      minMax/whisker                             range mode (changes dimensionality!)
    bigbed      significant                                range mode
    cooler      default/None/<column name>                 transform (open set)
    ==========  =========================================  ===========

    Declaring it as data is what lets a server ask the tileset what it accepts
    instead of hardcoding that bigwig means aggregation and cooler means
    normalization.

    Closed vs open sets are a real distinction, not a convenience. bigwig's
    modifiers are pybigtools statistics -- a fixed vocabulary. Cooler's are the
    *name of a bin table column*, which is per-file and unbounded, so the spec
    can only check that the modifier is well-formed; whether the column exists
    is a question for the instance, not the grammar.
    """

    values: frozenset[str]
    default: str | None = None
    # Free-form label for what the slot signifies ('aggregation', 'transform', ...).
    kind: str = "modifier"
    # When set, `values` lists the recognized sentinels but any other non-empty
    # string is accepted too. The tileset validates meaning at fetch time.
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


def _parse_options(
    opt_str: str, tile_id: str, recognized: frozenset[str] | None
) -> tuple[tuple[str, str], ...]:
    """The ``,key:value`` tail of a tile id, validated against ``recognized``.

    ``recognized`` of ``None`` accepts any key; an empty set rejects all.
    """
    if not opt_str:
        return ()

    parsed = []
    for chunk in opt_str.split(TILE_OPTIONS_CHAR):
        key, sep, value = chunk.partition(":")
        if not sep:
            raise MalformedTileId(
                f"option {chunk!r} in {tile_id!r} is not 'key:value'"
            )
        if recognized is not None and key not in recognized:
            raise UnsupportedOption(
                f"{key!r} is not a recognized option; expected one of "
                f"{sorted(recognized)}"
            )
        parsed.append((key, value))
    return tuple(parsed)


@dataclass(frozen=True, slots=True)
class TileId:
    """A parsed tile identifier.

    Immutable and hashable, so it can key a cache directly.

    Identity is per *request string*, not per tile: ``raw`` participates in
    equality and hashing, so ``"abc.3.4"`` and ``"abc.3.4,"`` are two distinct
    records for one tile. That is deliberate rather than incidental -- ``raw``
    is the key the response is built under, and a client that asked for two
    spellings is owed two entries -- but it does mean a batch carrying both
    does not collapse to one, and a cache keyed on this holds both. Pinned by
    ``test___eq___should_distinguish_by_the_request_string``.
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

        ``ndim`` and ``modifiers`` are required to disambiguate the positional
        slots -- which is precisely why parsing cannot be a free function and has
        to sit next to the declarative capability data.

        Parameters
        ----------
        ndim
            Number of coordinate slots after ``z`` (1 for vectors, 2 for matrices).
        modifiers
            What the trailing modifier slot may contain, if anything.
        options
            Recognized ``,key:value`` keys. ``None`` accepts any; an empty set
            rejects all.
        """
        head, _, opt_str = tile_id.partition(TILE_OPTIONS_CHAR)
        parsed_options = _parse_options(opt_str, tile_id, options)

        parts = head.split(".")
        expected = 1 + 1 + ndim  # uid + z + coords
        if len(parts) < expected:
            raise MalformedTileId(
                f"{tile_id!r} has {len(parts)} dotted parts; a {ndim}D tile "
                f"needs at least {expected} (uid.z{'.pos' * ndim})"
            )

        uid = parts[0]
        try:
            numbers = [int(p) for p in parts[1:expected]]
        except ValueError as exc:
            raise MalformedTileId(
                f"non-integer position in {tile_id!r}: {exc}"
            ) from exc

        # Validated here rather than downstream. A negative position otherwise
        # reaches `Chromsizes.invert`, which raises a plain `ValueError` -- and
        # a `ValueError` is not a `TileError`, so the server boundary cannot
        # render it into a per-tile error payload and a single malformed id
        # takes down the whole batch.
        if numbers[0] < 0:
            raise MalformedTileId(
                f"negative zoom level {numbers[0]} in {tile_id!r}"
            )
        if any(n < 0 for n in numbers[1:]):
            raise MalformedTileId(
                f"negative tile position in {tile_id!r}"
            )

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
            options=parsed_options,
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
