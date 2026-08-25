from __future__ import annotations

import math
from functools import cached_property
from enum import Enum
from typing import ClassVar, Protocol, Sequence, runtime_checkable

from clodius.core.coords import Canvas, Chromsizes
from clodius.core.payloads import TileKind, RegionRow
from clodius.core.policies import TilePolicy
from clodius.core.tileid import ModifierSpec, TileId
from clodius.core.errors import TileOutOfBounds
from pydantic import BaseModel, ConfigDict, field_validator


@runtime_checkable
class Dataset(Protocol):
    """Anything a HiGlass server can serve, tiled or not.

    ``chromsizes`` and ``time_interval`` are datasets but NOT tilesets: the
    client fetches them once in full and does its own up/downsampling, so they
    have no resolution ladder and no ``tiles()``. They are the only two such
    modules in ``clodius/tiles/``.
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

    # --- declarative capability surface -------------------------------------

    # Number of coordinate slots after ``z``. 1 for vectors, 2 for matrices.
    # Note bedpe and bed2ddb serve BOTH today, inferring from tile-id arity.
    ndim: ClassVar[int]

    # What ``tiles()`` returns, so a server knows what it is holding.
    tile_kind: ClassVar[TileKind]

    # What the modifier slot accepts, or None if the type declares none.
    modifiers: ClassVar[ModifierSpec | None]

    # Recognized ``,key:value`` option keys. Empty means options are rejected.
    options: ClassVar[frozenset[str]]

    # Limits applied when serving. Instance-level: a server may vary it per
    # tileset.
    policy: TilePolicy

    # --- the two core methods -----------------------------------------------

    def info(self) -> TilesetInfo:
        """Narrows :meth:`Dataset.info` to guarantee a resolution ladder."""
        ...

    def tiles(self, ids: Sequence[TileId]) -> list[tuple[TileId, object]]:
        """Fetch a batch of tiles.

        Batched (rather than one call per tile) so scan-oriented backends can
        coalesce adjacent requests -- see ``utils.partition_by_adjacent_tiles``.

        Returns pairs rather than a dict so the boundary can echo
        ``tile_id.raw`` as the response key, making the round-trip requirement
        structural instead of per-module bookkeeping.

        Implementations MUST NOT let one id's failure discard another's --
        that is the current bug in ``vcf.py:195``, ``bam.py:628`` and
        ``bam_pysam.py:359``, all of which ``return`` from inside the per-tile
        loop.

        Returning one entry per requested id is the default, but it is not
        universal: where a format's predecessor established that an
        out-of-range id is *skipped*, the rewrite preserves that and the result
        is shorter than the request. ``clodius/tiles/cooler.py`` did, so
        :class:`~clodius.tiles_v2.cooler.CoolerTileset` does; ``mrmatrix.py``
        raises and ``bigwig.py`` has no bound check at all, so no general rule
        is imposed here. A caller must key on the returned ids rather than
        zipping against the request.
        """
        ...

    def close(self) -> None:
        """Release file handles. ``cooler.mats`` holds h5py handles open
        forever today; this is what gives them a defined lifetime."""
        ...


@runtime_checkable
class ProvidesChromsizes(Protocol):
    """Backs ``GET /chrom-sizes/?id=<uid>``.

    Implemented today by: chromsizes (get_tsv_chromsizes), bigwig.chromsizes,
    bigbed.chromsizes. Also derivable from gff (gff_chromsizes:21), fasta (.fai)
    and pileup (_chromsizes_from_fasta:457), which currently keep it private.
    """

    def chromsizes(self) -> Chromsizes: ...


@runtime_checkable
class ProvidesRegions(Protocol):
    """Backs a paginated annotation-listing endpoint.

    NOT part of vanilla HiGlass. This is a resgen extension: served by
    resgen-server's ``/regions/`` view and consumed by the resgen-app
    ``RegionsList`` component. Worth keeping as a declared capability so the
    extension has a defined seam, but nothing in stock HiGlass calls it.

    Wire contract, confirmed against both ends::

        GET api/v1/regions/?d=<uid>&o=<offset>&l=<limit>
        -> {"offset": int, "limit": int, "results": [RegionRow], "next": bool}

    ``o`` defaults to 0 and is rejected above 1000; ``l`` defaults to 20 and is
    capped at 10000. The view also accepts ``cs`` and ``s`` params but ignores
    them. Only ``bed``/``bedfile`` and ``vcf`` are dispatched; anything else
    raises.

    The envelope is assembled by the *view*, not by clodius -- so the tuple
    below is the right internal shape, not an approximation of the wire.

    Implemented today by ``bedfile.regions:337`` and ``vcf.regions:41``, both
    delegating to ``vcf.generic_regions:19``.
    """

    def regions(
        self, offset: int, limit: int
    ) -> tuple[list[RegionRow], bool]: ...


class BaseTileset:
    """Optional convenience base: context-manager support and defaults.

    Implementing :class:`Tileset` does not require subclassing this -- it is a
    ``Protocol``, so structural typing applies.
    """

    modifiers: ClassVar[ModifierSpec | None] = None
    options: ClassVar[frozenset[str]] = frozenset()

    policy: TilePolicy

    def parse_tile_id(self, tile_id: str) -> TileId:
        """Parse against this tileset's declared shape."""
        return TileId.parse(
            tile_id,
            ndim=self.ndim,  # type: ignore[attr-defined]
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


def quadtree_depth(total_length: int, tile_size_bp: int) -> int:
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

    # Exact integer arithmetic, not ceil(log2(...)) over a float. The float
    # form is wrong above roughly 2**48: quadtree_depth(2**49 + 1, 1) returned
    # 49, and 2**49 is less than 2**49 + 1, so the top tile did not cover the
    # space it is defined to cover. No genome reaches that, but the exact form
    # costs nothing and removes the bound.
    min_tile_cover = -(-total_length // tile_size_bp)
    return (min_tile_cover - 1).bit_length()


class Ladder(str, Enum):
    """How a tileset serializes its resolution ladder.

    ``z`` always means "index into the ladder, coarsest first". Only the
    serialization differs -- and it is a property of the *file*, not the
    filetype: cooler uses IMPLICIT for legacy single-res ``.cool`` and EXPLICIT
    for ``.mcool``.
    """

    # Power-of-two spaced, derived from ``max_zoom`` + ``tile_size``.
    IMPLICIT = "implicit"
    # Arbitrary, enumerated in ``resolutions``.
    #
    # ``resolutions`` is serialized in **ascending** order -- finest first,
    # the opposite of what ``z`` indexes. That is what legacy cooler emits and
    # what a client that does not re-sort will assume. :meth:`resolution_for`
    # re-sorts internally, so the wire order has no effect on serving; it is
    # fixed here only so two tilesets do not emit the same field two ways.
    EXPLICIT = "explicit"


class DatasetInfo(BaseModel):
    """Fields common to anything servable, tiled or not.

    Not every dataset HiGlass consumes is a tileset. ``chromsizes`` and
    ``time_interval`` are fetched once in their entirety and rendered
    client-side -- chromsizes via the dedicated ``chrom-sizes?id=`` endpoint --
    so they have no resolution ladder and no ``tiles()``. For those two the
    entire dataset content *is* the info response.

    Per-type extensions (``aggregation_modes``, ``row_infos``, ``header``,
    ``max_per_tile``, ``start_value``, ...) are permitted via ``extra="allow"``
    during migration. The plan is per-type subclasses that declare their own
    fields, with ``extra`` flipped to ``"forbid"`` as each type migrates.
    """

    model_config = ConfigDict(extra="allow")

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


class TilesetInfo(DatasetInfo):
    """A :class:`DatasetInfo` that also carries a resolution ladder.

    This is the model for anything served through ``tiles()``. The ladder
    accessors live here rather than on the base so that asking a non-tiled
    dataset for its zoom levels is a type error, not a runtime surprise.

    Immutable after construction. :attr:`coordinate_system` and
    :attr:`_descending_ladder` are cached, and ``canvas()`` feeds the former
    straight into tile-position inversion -- so a post-construction assignment
    to ``chromsizes`` would silently serve tiles built against a stale genome
    layout rather than raising. Per-type extras still go through the
    constructor, since ``extra="allow"`` accepts them as keyword arguments.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    # --- ladder, implicit form (power-of-two) ---
    max_zoom: int | None = None
    tile_size: int | None = None

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

    # --- ladder access -------------------------------------------------------

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

    def resolution_for(self, z: int) -> float:
        """Base pairs per bin at zoom index ``z``.

        Absorbs both serializations so callers never branch on
        ``"resolutions" in info``. This retires the sniff at
        ``cooler.generate_tiles:660``, the four separate ladder implementations,
        and ``sequence_logos``' ``del tsinfo["max_zoom"]`` conversion hack.

        The implicit formula is exactly ``cooler.py:675``, which already sits in
        the ``else`` branch of the very sniff this replaces::

            resolution = (max_width / 2**z) / tile_size

        Raises
        ------
        TileOutOfBounds
            If ``z`` is outside the ladder. Note the current cooler behavior is
            to skip such tiles silently; the boundary decides which to surface.
        """
        if z < 0:
            raise TileOutOfBounds(f"negative zoom level {z}")

        ladder = self._descending_ladder
        if ladder:
            if z >= len(ladder):
                raise TileOutOfBounds(
                    f"zoom {z} exceeds ladder of {len(ladder)} resolutions"
                )
            return float(ladder[z])

        if self.max_width is None or self.tile_size is None:
            raise ValueError(
                "implicit ladder needs both max_width and tile_size; got "
                f"max_width={self.max_width}, tile_size={self.tile_size}"
            )
        if self.max_zoom is not None and z > self.max_zoom:
            raise TileOutOfBounds(f"zoom {z} exceeds max_zoom {self.max_zoom}")

        # TODO: decide rounding. This is exact for power-of-two ladders (every
        # current implicit tileset) but goes fractional if max_width is not a
        # multiple of tile_size * 2**z.
        return (self.max_width / 2**z) / self.tile_size

    def tile_span(self, z: int) -> float:
        """Width of one tile at zoom ``z``, in coordinate-space units.

        Replaces the ``tsinfo["max_width"] / 2**z`` idiom inlined in bedfile,
        bedpe, gff, vcf, bam and tabix.
        """
        if self.max_width is None:
            raise ValueError("tileset declares no max_width")
        return self.max_width / 2**z

    def canvas(self, z: int):
        """The uniform lattice at zoom ``z``.

        The entry point for all grid geometry: a
        :class:`~clodius.core.coords.Canvas` knows its bin size, how many bins
        and tiles it holds, and inverts tile positions back to genomic
        intervals.
        Callers get the zoom-level arithmetic done once instead of recomputing
        ``max_width / 2**z``.

        Needs ``tile_size`` and ``max_width`` even on an explicit ladder, where
        ``resolutions`` alone fixes the bin sizes: those two describe the
        *geometry* (how wide a tile is, how far the lattice extends) rather than
        the ladder. A tileset that does not advertise them natively should
        derive them -- see :class:`~clodius.tiles_v2.cooler.CoolerTileset`,
        where tile size is a format constant and the extent follows from the
        coarsest resolution.

        Raises
        ------
        TileOutOfBounds
            If ``z`` lies outside the resolution ladder.
        """
        if self.tile_size is None:
            raise ValueError("tileset declares no tile_size")

        binsize = self.resolution_for(z)

        if self.ladder is Ladder.EXPLICIT:
            # For an explicit ladder the extent is a property of the *zoom*, not
            # of the tileset: it is however many tiles are needed to cover the
            # genome at this resolution. Only a power-of-two ladder (what
            # `cooler zoomify` emits) makes it invariant, because such a ladder
            # is a quadtree. On the 4DN standard set it varies non-monotonically
            # -- 5.12 Gb, 3.84, 3.20, 3.33, ... -- so there is no tileset-level
            # value to inherit, and inheriting one from the coarsest level
            # inflates n_tiles at every finer zoom.
            if self.coordinate_system is None:
                raise ValueError(
                    "an explicit ladder needs chromsizes to derive its extent"
                )
            tile_span = binsize * self.tile_size
            extent = math.ceil(self.coordinate_system.total_length / tile_span)
            extent = int(extent * tile_span)
        else:
            # An implicit ladder is a quadtree: max_width is tile_size * 2**max_zoom
            # and is genuinely invariant, so the tileset-level value stands.
            # `resolution_for` above already rejected a missing max_width, so
            # no guard is needed here.
            extent = self.max_width

        return Canvas(
            z=z,
            binsize=binsize,
            tile_size=self.tile_size,
            max_width=extent,
            chromsizes=self.coordinate_system,
        )

    @cached_property
    def _descending_ladder(self) -> tuple[int, ...]:
        """``resolutions`` coarsest-first, which is the order ``z`` indexes.

        Cached because ``resolution_for`` is called once per tile and
        ``canvas()`` calls it again. Empty on an implicit ladder.
        """
        return tuple(sorted(self.resolutions or (), reverse=True))

    @cached_property
    def coordinate_system(self) -> Chromsizes | None:
        """The wire ``chromsizes`` field as a :class:`Chromsizes`.

        Cached because ``canvas()`` is called per tile and building this
        recomputes cumulative offsets. ``None`` for non-genomic tilesets, which
        have no chromosomes to invert tile positions into.
        """
        if self.chromsizes is None:
            return None
        return Chromsizes.from_pairs(self.chromsizes)
