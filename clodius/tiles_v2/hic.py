from __future__ import annotations

from functools import cached_property

import hictkpy
import numpy as np

from clodius.core import (
    DEFAULT_POLICY,
    BaseTileset,
    Chromsizes,
    DenseTile,
    GenomicRange,
    GridPolicy,
    ModifierSpec,
    TileError,
    TileId,
    TileKind,
    TileOutOfBounds,
    TilePolicy,
    TilesetInfo,
    bin_count,
    reconcile_sequential_2d,
)

TILE_SIZE = 256

#: No normalization. Never listed by ``avail_normalizations`` -- it is the
#: absence of a weight vector rather than one of them -- so it has to be
#: recognized separately from whatever a file carries.
NO_NORMALIZATION = "NONE"

#: The juicer normalization vocabulary. Closed, unlike cooler's transform slot:
#: cooler's modifier names a *bin table column*, which is per-file and
#: unbounded, while these are fixed by the .hic format. ``ModifierSpec``'s
#: docstring draws exactly this distinction, so the spec can reject a
#: misspelled name at parse time instead of deferring every case to fetch time.
#: Whether a given file carries a given vector is still an instance question,
#: answered by :func:`resolve_normalization`.
NORMALIZATIONS = (
    NO_NORMALIZATION,
    "VC",
    "VC_SQRT",
    "KR",
    "SCALE",
    "GW_VC",
    "GW_KR",
    "GW_SCALE",
    "INTER_VC",
    "INTER_KR",
    "INTER_SCALE",
)

HIC_NORMALIZATION = ModifierSpec(
    values=frozenset({"default", *NORMALIZATIONS}),
    default="default",
    kind="transform",
)


def resolve_normalization(available, transform: str | None) -> str:
    """Translate a transform modifier into a hictkpy normalization name.

    Parameters
    ----------
    available : Sequence[str]
        The normalizations this file carries, from
        ``hictkpy.File.avail_normalizations``. ``NONE`` is never among them.
    transform : str or None
        The tile-id transform modifier, or None if no modifier was present.

    Returns
    -------
    str
        A normalization name to hand to ``File.fetch``.

    Raises
    ------
    TileError
        If the requested normalization is not stored in this file.

    Notes
    -----
    Takes the list of names rather than an open file so the whole contract is
    exercisable from literals. A synthesized ``.hic`` carries no normalization
    vectors at all -- ``hictkpy.hic.FileWriter`` stores pixels and leaves KR
    and VC to juicer -- so a fixture cannot cover the interesting cases.

    ``default`` and a missing modifier both resolve to ``NONE``, which is where
    this parts company with :func:`~clodius.tiles_v2.cooler.resolve_balance`.
    Cooler's fallback to ``weight`` is not a policy choice: ``weight`` is the
    only candidate a cooler ever has. A ``.hic`` typically carries four
    vectors, so silently electing one of them would be a rendering decision the
    client did not make. ``transforms`` in ``tileset_info`` advertises what is
    there; a client that wants normalization asks for it by name.
    """
    if transform is None or transform == "default":
        return NO_NORMALIZATION
    if transform == NO_NORMALIZATION:
        return NO_NORMALIZATION
    if transform not in available:
        raise TileError(
            f"no {transform!r} normalization in this .hic; available: "
            f"{sorted(available)}"
        )
    return transform


def shared_normalizations(listings) -> list[str]:
    """The normalizations every resolution carries, in the finest's order.

    Parameters
    ----------
    listings : Sequence[Sequence[str]]
        One ``avail_normalizations`` result per resolution, finest first.

    Returns
    -------
    list[str]
        The intersection, ordered by ``listings[0]``.

    Notes
    -----
    Offered only if stored at *every* resolution, because a client may request
    one at any zoom and a dropdown entry that fails on some levels is worse
    than an absent one.

    Ordered by the finest resolution's own listing rather than by iterating the
    intersection set. ``clodius/tiles/cooler.py`` builds its transform list
    with ``for t in <set>``, whose order varies with the process hash seed, so
    its dropdown can reorder between server restarts.

    Pure, and separate from :meth:`HicTileset.info`, for the same reason
    :func:`resolve_normalization` is: ``hictkpy.hic.FileWriter`` writes no
    normalization vectors, so no synthesizable fixture can exercise either
    rule through a real file.
    """
    if not listings:
        return []
    shared = set(listings[0]).intersection(*map(set, listings[1:]))
    return [n for n in listings[0] if n in shared]


def fetch_block(
    f: hictkpy.File,
    row: GenomicRange,
    col: GenomicRange,
    binsize: float,
    normalization: str,
):
    """One block of the matrix, shaped for the reconciler.

    A ``.hic`` stores the upper triangle only, and hictkpy refuses a query that
    lands wholly below the diagonal rather than mirroring it -- ``fetch("c2:..",
    "c1:..")`` raises ``overlaps with the lower-triangle of the matrix``. The
    ordering test is on ``(cid, start)``: equal is the diagonal and is fine, so
    only a strictly greater row is reflected, and the transpose of the mirrored
    block is the block that was asked for.

    ``query_type="BED"`` takes a tab-separated ``chrom<TAB>start<TAB>end``,
    zero-based half-open -- the same convention :class:`GenomicRange` uses
    internally. The default ``"UCSC"`` would need the one-based fully-closed
    spelling and turns a boundary interval into a block one bin narrow, which
    is the kind of error that produces a plausible tile rather than a loud one.
    """
    shape = (bin_count(row, binsize), bin_count(col, binsize))
    if row.is_out_of_bounds or col.is_out_of_bounds:
        # Past the last chromosome: padding, not missing data. Same contract as
        # cooler's -- the caller supplies a correctly shaped NaN block.
        return np.full(shape, np.nan, dtype=np.float32)

    if (row.cid, row.start) > (col.cid, col.start):
        return fetch_block(f, col, row, binsize, normalization).T

    block = f.fetch(
        _bed(row),
        _bed(col),
        normalization=normalization,
        query_type="BED",
    ).to_numpy(query_span="full")
    return block.astype(np.float32)


def _bed(interval: GenomicRange) -> str:
    name, start, end = interval.as_tuple()
    return f"{name}\t{start}\t{end}"


class HicTileset(BaseTileset):
    """A .hic served as a 2D matrix tileset."""

    datatype = "matrix"
    ndim = 2
    tile_kind = TileKind.DENSE
    modifiers = HIC_NORMALIZATION
    options = frozenset()

    grid_policy = GridPolicy.SEQUENTIAL

    def __init__(self, path, policy: TilePolicy = DEFAULT_POLICY):
        self._path = path
        self._file = None
        self._policy = policy
        self._info = None

    # --- resource lifetime --------------------------------------------------

    @property
    def file(self) -> hictkpy.MultiResFile:
        """The open multi-resolution handle, opened on first read.

        Raises
        ------
        ValueError
            If the path cannot be read as a ``.hic``.

        Notes
        -----
        hictkpy reports a missing file, a ``.mcool``, and a text file renamed
        ``.hic`` identically -- a bare ``RuntimeError`` out of the C++ layer,
        carrying the path but nothing a caller can dispatch on. Translating it
        means a caller does not have to know which extension module refused the
        file, and does not have to catch ``RuntimeError``, which anything at
        all may raise.

        This is close to, but not the same as, what ``CoolerTileset.file``
        does. That one raises ``ValueError`` for a *readable* file of the wrong
        shape -- an ``.mcool`` with no ``resolutions`` group -- and lets h5py's
        ``OSError`` through for a path it cannot open at all. Which exception
        an unreadable tileset should raise is unsettled across the three
        implementations; ``tiles_v2/bed.py`` raises ``TilesetUnavailable``.
        """
        if self._file is None:
            try:
                self._file = hictkpy.MultiResFile(str(self._path))
            except RuntimeError as exc:
                raise ValueError(
                    f"{self._path} is not a readable .hic file"
                ) from exc
        return self._file

    def close(self) -> None:
        # hictkpy exposes no explicit close; dropping the handle is what
        # releases the underlying file, and the next read reopens.
        self._file = None

    # --- ProvidesChromsizes -------------------------------------------------

    def chromsizes(self) -> Chromsizes:
        """The genome, without the ``All`` pseudo-chromosome.

        ``include_ALL`` is passed explicitly even though ``False`` is its
        default: a ``.hic`` carries a genome-wide ``All`` entry alongside the
        real contigs, and inheriting its exclusion from a default would leave
        nothing at the call site to say that the entry exists at all.
        """
        chroms = self.file.chromosomes(include_ALL=False)
        return Chromsizes(
            tuple(chroms.keys()), tuple(int(v) for v in chroms.values())
        )

    # --- the protocol -------------------------------------------------------

    @property
    def policy(self) -> TilePolicy:
        return self._policy

    @cached_property
    def resolutions(self) -> tuple[int, ...]:
        """The stored resolutions, ascending.

        Cached for the same reason cooler's is: ``_build_info`` walks it
        repeatedly, and each access otherwise re-reads the file's index.
        """
        return tuple(sorted(int(r) for r in self.file.resolutions()))

    def info(self) -> TilesetInfo:
        if self._info is None:
            self._info = self._build_info()
        return self._info

    def tiles(self, ids):
        """Tiles for ``ids``, omitting any position the ladder does not hold.

        Matches :meth:`~clodius.tiles_v2.cooler.CoolerTileset.tiles`. Both
        formats are explicit-ladder matrix tilesets and a client should not
        have to branch on which one it is talking to, so a zoom above the
        ladder or a position past the canvas yields a short list rather than an
        error. Cooler inherited that contract from ``clodius/tiles/cooler.py``;
        here it is chosen, since ``.hic`` has no predecessor.

        Only ``TileOutOfBounds`` is swallowed. An unavailable normalization is
        a ``TileError`` and still propagates: a client asking for a vector the
        file does not carry has made a different kind of mistake than one
        asking for a tile off the end of the genome.
        """
        out = []
        for tid in ids:
            try:
                out.append((tid, self._tile(tid)))
            except TileOutOfBounds:
                continue
        return out

    # --- internals ----------------------------------------------------------

    def _at(self, resolution) -> hictkpy.File:
        # int() is load-bearing: canvas.binsize is a float, and MultiResFile is
        # keyed by the integer resolution.
        return self.file[int(resolution)]

    def _build_info(self) -> TilesetInfo:
        resolutions = self.resolutions
        chromsizes = self.chromsizes()

        # `resolutions` is ascending, so element 0 is the finest and supplies
        # the order. `NONE` needs no filtering out: it is the absence of a
        # weight vector and is never listed.
        transforms = shared_normalizations(
            [self._at(r).avail_normalizations() for r in resolutions]
        )

        # No max_width, for the reason CoolerTileset._build_info records: on an
        # explicit ladder the extent belongs to the zoom level, and Canvas
        # derives it per level.
        #
        # No mirror_tiles either. That flag tells the client to reflect a tile
        # across the diagonal, and `fetch_block` has already done so -- hictkpy
        # refuses a lower-triangle query outright, so the reflection happens
        # here or not at all.
        return TilesetInfo(
            min_pos=[1, 1],
            max_pos=[chromsizes.total_length, chromsizes.total_length],
            resolutions=list(resolutions),
            tile_size=TILE_SIZE,
            chromsizes=chromsizes.to_pairs(),
            transforms=[{"name": n, "value": n} for n in transforms],
        )

    def _tile(self, tid: TileId):
        x, y = tid.pos
        canvas = self.info().canvas(tid.z)
        binsize = canvas.binsize
        # Zero-length intervals -- a tile boundary landing exactly on a
        # chromosome boundary -- contribute no bins and no bp, so they only
        # cost a `fetch` round-trip.
        col_intervals = [iv for iv in canvas.invert(x) if iv.end > iv.start]
        row_intervals = [iv for iv in canvas.invert(y) if iv.end > iv.start]

        f = self._at(canvas.binsize)
        normalization = resolve_normalization(
            f.avail_normalizations(), tid.modifier
        )

        blocks = [
            [
                fetch_block(f, row, col, binsize, normalization)
                for col in col_intervals
            ]
            for row in row_intervals
        ]

        # Reconcile the blocks into a single 2D array.
        out = reconcile_sequential_2d(
            blocks,
            row_spans=[iv.end - iv.start for iv in row_intervals],
            col_spans=[iv.end - iv.start for iv in col_intervals],
            binsize=binsize,
            expected_shape=(canvas.tile_size, canvas.tile_size),
        )
        return DenseTile(out.ravel()).to_dict()
