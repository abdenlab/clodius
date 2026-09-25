from __future__ import annotations

import math
from functools import cached_property
from enum import Enum
from collections.abc import Mapping
from typing import Any, ClassVar, Protocol, Self, Sequence, runtime_checkable

from clodius.core.coords import TileCanvas, Chromsizes
from clodius.core.tile import AnnotationRecord, TileKind
from clodius.core.policies import TilePolicy
from clodius.core.tileid import ModifierSpec, TileId
from clodius.core.errors import (
    TileError,
    TileOutOfBounds,
    TilesetUnavailable,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    model_validator,
)


@runtime_checkable
class Dataset(Protocol):
    """Anything a HiGlass server can serve, tiled or not.

    ``chromsizes`` and ``time_interval`` are datasets but NOT tilesets: the
    client fetches them once in full and does its own up/downsampling, so they
    have no resolution ladder and no ``tiles()``.
    """

    # HiGlass datatype string ('vector', 'matrix', 'bedlike', 'chromsizes', ...).
    datatype: ClassVar[str]

    def info(self) -> DatasetInfo:
        """Backs ``GET /tileset_info/?d=<uid>``."""
        ...

    def close(self) -> None: ...


@runtime_checkable
class Tileset(Dataset, Protocol):
    """A dataset served tile-by-tile through a resolution ladder."""

    # Number of coordinate slots after ``z``.
    ndim: ClassVar[int]

    # What ``tiles()`` returns, so a server knows what it is holding.
    datatype: ClassVar[str]

    # What the modifier slot accepts, or None if the type declares none.
    modifiers: ClassVar[ModifierSpec | None]

    # Recognized ``,key:value`` option keys. Empty means options are rejected.
    options: ClassVar[frozenset[str]]

    # Limits applied when serving.
    policy: TilePolicy

    def info(self) -> TilesetInfo: ...

    def tiles(
        self,
        ids: Sequence[TileId],
        options: Mapping[str, Any] | None = None,
    ) -> list[tuple[TileId, TileKind]]:
        """Fetch a batch of tiles.

        Parameters
        ----------
        ids : Sequence[TileId]
            One or more tile IDs for fetching. Batching makes it possible for
            adjacent requests to be coalesced into a single fetch pass.
        options : Mapping, optional
            Per-request settings that apply to the whole batch, from the
            ``options`` object of a ``POST /tiles/`` body.

        Returns
        -------
        list[tuple[TileId, TileKind]]
            Pairs of tile IDs and matched payloads. A tile that cannot be
            served carries a :class:`TileError` in the payload slot.

        Notes
        -----
        Implementations MUST return one entry per requested ID, in any order.
        The client and server match responses to requests by exact raw tile ID
        string, which is why the TileId travels with its payload.

        Failures to fetch individual tiles should be caught and return a
        TileError as response payload. A whole-batch failure should raise
        TilesetUnavailable and propagate to the server. Malformed ``options``
        are a whole-batch failure: they arrive once, for every tile at once.
        """
        ...

    def close(self) -> None:
        """Release file handles."""
        ...


@runtime_checkable
class ProvidesChromsizes(Protocol):
    """Backs ``GET /chrom-sizes/?id=<uid>``."""

    def chromsizes(self) -> Chromsizes: ...


@runtime_checkable
class ProvidesRegions(Protocol):
    """Backs a paginated annotation-listing endpoint.

    This is a resgen extension served by resgen-server's ``/regions/``
    view and consumed by the resgen-app ``RegionsList`` component.

        GET api/v1/regions/?d=<uid>&o=<offset>&l=<limit>
        -> {"offset": int, "limit": int, "results": [AnnotationRecord], "next": bool}
    """

    def regions(
        self, offset: int, limit: int
    ) -> tuple[list[AnnotationRecord], bool]: ...


class BaseTileset:
    """Optional convenience base: context-manager support and defaults."""

    modifiers: ClassVar[ModifierSpec | None] = None
    options: ClassVar[frozenset[str]] = frozenset()
    # Annotated, deliberately not assigned. `Tileset` is a runtime-checkable
    # Protocol, so `isinstance` is an attribute-presence test and `ndim` is one
    # of the attributes it tests -- a concrete default here would make a
    # subclass that forgot to declare one pass the only published "is this
    # servable" check. The absence is caught in `parse_tile_id` instead, where
    # it can be reported as the server-side misdeclaration it is.
    ndim: ClassVar[int]

    tile_size: int
    policy: TilePolicy

    def tiles(
        self,
        ids: Sequence[TileId],
        options: Mapping[str, Any] | None = None,
    ) -> list[tuple[TileId, TileKind]]:
        """One entry per requested id, always.

        The per-tile boundary `clodius.core.errors` describes, in one place.
        A refusal lands in that tile's payload slot rather than aborting the
        batch. Only errors that are genuinely per-tile are caught -- an
        unreadable file still raises, because retrying the other fifteen tiles
        against it is pointless.

        A tileset whose `_tile` needs more than a tile id -- a batch-wide
        option, a shared reader -- overrides this; every other one inherits it.
        A tileset inheriting this default understands no options at all, so a
        non-empty ``options`` is refused rather than dropped: the protocol says
        malformed options are a whole-batch failure, and silently serving
        tiles that ignore what was asked for is the worse of the two answers.
        """
        if options:
            raise TilesetUnavailable(
                f"{type(self).__name__} accepts no tile options; got "
                f"{sorted(options)}"
            )
        out = []
        for tid in ids:
            try:
                out.append((tid, self._tile(tid)))
            except TileError as exc:
                out.append((tid, exc.to_dict()))
        return out

    def _tile(self, tid: TileId) -> TileKind:
        """One tile's payload, or raise a `TileError` refusing it."""
        raise NotImplementedError

    def parse_tile_id(self, tile_id: str) -> TileId:
        """Parse against this tileset's declared shape."""
        # `getattr` with a default is load-bearing: `ndim: ClassVar[int]`
        # creates an annotation and no attribute, so `self.ndim` would raise
        # AttributeError. The arity is the tileset's own declaration and not
        # part of the request, so a missing one is `TilesetUnavailable` -- a
        # client told its well-formed id was malformed retries forever against
        # a fault only the server can fix.
        ndim = getattr(self, "ndim", None)
        if ndim is None:
            raise TilesetUnavailable(
                f"{type(self).__name__} declares no ndim, so a tile id "
                f"cannot be parsed against it"
            )
        return TileId.parse(
            tile_id,
            ndim=ndim,
            modifiers=self.modifiers,
            # Passed through unchanged. `self.options or None` would collapse
            # an empty frozenset to `None` -- "accept any option" -- which is
            # the opposite of what an empty set declares.
            options=self.options,
        )

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _quadtree_depth(total_length: int, tile_size_bp: int) -> int:
    """
    Number of zoom levels to cover ``total_length`` with a specific quadtree.

    The quadtree starts with tiles of length ``tile_size_bp`` at the highest
    resolution and a power-of-two size ladder until the entire coordinate space
    is covered by a single tile.

    Parameters
    ----------
    total_length : int
        Total length of the coordinate space to cover.
    tile_size_bp : int
        Size of a single tile in base pairs.

    Returns
    -------
    int
        The minimum quadtree depth required to cover the coordinate space with
        tiles of the specified size.
    """
    if total_length <= 0:
        raise ValueError(f"total_length must be positive, got {total_length}")
    if tile_size_bp <= 0:
        raise ValueError(f"tile_size_bp must be positive, got {tile_size_bp}")

    min_tile_cover = math.ceil(total_length / tile_size_bp)
    return int(math.ceil(math.log2(min_tile_cover)))


class Ladder(str, Enum):
    """How a tileset serializes its resolution ladder."""

    # Quadtree, power-of-two spaced, derived from ``max_zoom`` + ``tile_size``.
    IMPLICIT = "implicit"
    # Arbitrary, enumerated in ``resolutions``.
    EXPLICIT = "explicit"


class DatasetInfo(BaseModel):
    """Fields common to anything servable, tiled or not.

    Frozen, like :class:`TilesetInfo`. Both are public and `Dataset.info()`
    is annotated with this one, so code written against the base's contract
    would otherwise type-check and then fail only on the subclass. Derive a
    variant with :meth:`with_`.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    # `frozen=True` makes pydantic synthesize a hash over the fields, and
    # `min_pos`/`max_pos` are lists -- so the type would advertise Hashable
    # while every instance raised TypeError.
    __hash__ = None

    def with_(self, **changes: Any) -> Self:
        """A copy with ``changes`` applied, re-validated.

        Unlike ``model_copy(update=...)`` this runs the constructor, so a
        misspelled key or an out-of-range value is rejected rather than served
        verbatim, and any cached derivation is rebuilt from the new fields
        instead of surviving the copy. Mirrors `TilePolicy.with_`.
        """
        return type(self)(**{**self.model_dump(exclude_none=True), **changes})

    min_pos: list[int]
    max_pos: list[int]
    max_width: int | None = None
    chromsizes: list[tuple[str, int]] | None = None

    @field_validator("max_width")
    @classmethod
    def _max_width_positive(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("max_width must be > 0")
        return v

    def to_dict(self) -> dict[str, Any]:
        """The info as the client should receive it.

        Unset optionals are dropped rather than serialized as ``null``, because
        `null` is not interchangeable with *absent* on the wire.
        """
        return self.model_dump(exclude_none=True)


class TilesetInfo(DatasetInfo):
    """A :class:`DatasetInfo` that also carries a resolution ladder.

    Frozen. The `coordinate_system` cached on it is only sound because the
    field it reads cannot change underneath it.

    Derive a variant with :meth:`with_`, which re-validates and rebuilds the
    cache. ``model_copy(update=...)`` does neither: a misspelled key is served
    verbatim, an out-of-range value is accepted where the constructor would
    reject it, and a copy updating ``chromsizes`` keeps the coordinate system
    derived from the old ones.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    # Inherited from DatasetInfo in intent, restated because pydantic
    # re-synthesizes a hash for every frozen model.
    __hash__ = None

    # --- ladder, implicit form (power-of-two) ---
    max_zoom: int | None = None
    tile_size: int | None = None  # in bins

    # --- ladder, explicit form ---
    resolutions: list[int] | None = None

    @field_validator("max_zoom")
    @classmethod
    def _max_zoom_non_negative(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("max_zoom must be >= 0")
        return v

    @field_validator("tile_size")
    @classmethod
    def _tile_size_positive(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("tile_size must be > 0")
        return v

    @model_validator(mode="after")
    def _implicit_ladder_is_a_quadtree(self) -> TilesetInfo:
        """An implicit ladder must span exactly ``tile_size * 2**max_zoom``.

        That is what makes every level's resolution a whole number of base
        pairs, and it is what :meth:`quadtree` constructs. Precomputed files
        satisfy it too -- ``cli/aggregate`` writes the same product. Asserted
        rather than rounded away, so a file that breaks it is loud.
        """
        if self.resolutions:
            return self
        if (
            self.max_width is None
            or self.tile_size is None
            or self.max_zoom is None
        ):
            return self
        expected = self.tile_size * 2**self.max_zoom
        if self.max_width != expected:
            raise ValueError(
                f"implicit ladder is not a quadtree: max_width "
                f"{self.max_width} != tile_size {self.tile_size} * 2**max_zoom "
                f"{self.max_zoom} ({expected})"
            )
        return self

    @classmethod
    def quadtree(
        cls,
        chromsizes: Chromsizes,
        tile_size: int,
        *,
        ndim: int = 1,
        **fields,
    ) -> TilesetInfo:
        """Build the info for a quadtree tileset over ``chromsizes``.

        Parameters
        ----------
        chromsizes : Chromsizes
            Coordinate system to cover.
        tile_size : int
            Size of a single tile in base pairs at the highest resolution.
        ndim : int, optional
            Number of coordinate axes, which sets the length of ``min_pos`` and
            ``max_pos``.
        **fields
            Additional model fields, overriding any of the derived ones.

        Returns
        -------
        TilesetInfo
        """
        total = chromsizes.total_length
        max_zoom = _quadtree_depth(total, tile_size)
        return cls(
            **{
                "min_pos": [0] * ndim,
                "max_pos": [total] * ndim,
                "max_width": tile_size * 2**max_zoom,
                "tile_size": tile_size,
                "max_zoom": max_zoom,
                "chromsizes": chromsizes.to_pairs(),
                **fields,
            }
        )

    @property
    def ladder(self) -> Ladder:
        return Ladder.EXPLICIT if self.resolutions else Ladder.IMPLICIT

    @property
    def num_zoom_levels(self) -> int:
        if self.resolutions:
            return len(self.resolutions)
        if self.max_zoom is None:
            raise ValueError(
                "tileset declares neither resolutions nor max_zoom"
            )
        return self.max_zoom + 1

    def resolution_for(self, z: int) -> int:
        """Base pairs per bin at zoom index ``z``.

        Raises :class:`TileOutOfBounds` if ``z`` is outside the ladder.
        """
        if z < 0:
            raise TileOutOfBounds(f"negative zoom level {z}")

        if self.resolutions:
            ladder = sorted(self.resolutions, reverse=True)
            if z >= len(ladder):
                raise TileOutOfBounds(
                    f"zoom {z} exceeds ladder of {len(ladder)} resolutions"
                )
            return ladder[z]

        if self.max_width is None or self.tile_size is None:
            raise ValueError(
                "Quadtree zoom ladder needs both max_width and tile_size; got "
                f"max_width={self.max_width}, tile_size={self.tile_size}"
            )
        if self.max_zoom is not None and z > self.max_zoom:
            raise TileOutOfBounds(f"zoom {z} exceeds max_zoom {self.max_zoom}")

        # Check that a whole number resolution is derived.
        binsize, remainder = divmod(self.max_width, 2**z * self.tile_size)
        if remainder:
            raise ValueError(
                "Malformed quadtree - "
                f"zoom {z} does not divide the coordinate space evenly: "
                f"max_width {self.max_width} / 2**{z} is not a multiple of "
                f"tile_size {self.tile_size}"
            )
        return binsize

    def tile_width(self, z: int) -> int:
        """Width of one tile at zoom ``z``, in coordinate-space units (bp).

        Raises :class:`TileOutOfBounds` if ``z`` is outside the ladder.
        """
        if self.tile_size is None:
            raise ValueError("tileset declares no tile_size")
        return self.resolution_for(z) * self.tile_size

    @cached_property
    def coordinate_system(self) -> Chromsizes | None:
        """The ``chromsizes``; ``None`` for non-genomic tilesets."""
        if self.chromsizes is None:
            return None
        return Chromsizes.from_pairs(self.chromsizes)

    def canvas(self, z: int):
        """The scale mapping for the grid of tiles at zoom ``z``.

        The canvas knows its bin size, tile size, how many bins and tiles it
        holds, and can invert tile positions into genomic intervals.

        Raises :class:`TileOutOfBounds` if ``z`` is outside the ladder.

        Notes
        -----
        For a quadtree ladder, the canvas has a fixed size across all zoom
        levels, determined by the amount of padding needed to cover the
        coordinate system at the highest resolution.

        For an explicit ladder, the canvas size is calculated at each zoom
        level independently and depends on the padding required to cover the
        coordinate system for each tile size in the ladder.
        """
        if self.tile_size is None:
            raise ValueError("tileset declares no tile_size")

        binsize = self.resolution_for(z)

        if self.ladder is Ladder.EXPLICIT:
            if self.coordinate_system is None:
                raise ValueError(
                    "An explicit ladder needs chromsizes to derive its extent"
                )
            tile_span = binsize * self.tile_size
            extent = math.ceil(self.coordinate_system.total_length / tile_span)
            extent = int(extent * tile_span)
        else:
            # An implicit ladder is a quadtree: max_width is tile_size * 2**max_zoom
            if self.max_width is None:
                raise ValueError("Quadtree tileset declares no max_width")
            extent = self.max_width

        return TileCanvas(
            z=z,
            binsize=binsize,
            tile_size=self.tile_size,
            max_width=extent,
            chromsizes=self.coordinate_system,
        )
