from __future__ import annotations

import math
from functools import cached_property

import cooler
import h5py
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
    reconcile_sequential_2d,
)

TILE_SIZE = 256

COOLER_TRANSFORM = ModifierSpec(
    values=frozenset({"default", "None"}),
    default="default",
    kind="transform",
    allow_unknown=True,
)
TRANSFORM_LABELS = {"weight": "ICE"}

# Present in every bin table; everything else is a candidate weight column.
BIN_COORDINATE_COLUMNS = ("chrom", "start", "end")


def resolve_balance(clr: cooler.Cooler, transform: str | None):
    """Translate a transform modifier into a balance argument for cooler.

    Parameters
    ----------
    clr : cooler.Cooler
        The cooler to serve tiles from.
    transform : str or None
        The tile-id transform modifier, or None if no modifier was present.

    Returns
    -------
    str or bool
        The column name to balance with, or False if no balancing is desired.

    Raises
    ------
    TileError
        If the requested transform is not available in this cooler.

    Notes
    -----
    The following ``transform`` values are recognized:
    - no modifier, or ``default``: use the ``weight`` column if present, else
      no balancing.
    - String literal `"None"``:  do not balance.
    - Anything else: the name of a bin table column to balance with.
    """
    if transform == "None":
        return False
    if transform is None or transform == "default":
        return "weight" if "weight" in clr.bins().columns else False
    if transform not in clr.bins().columns:
        available = sorted(
            c for c in clr.bins().columns if c not in ("chrom", "start", "end")
        )
        raise TileError(
            f"no balancing column {transform!r} in this cooler; available: {available}"
        )
    return transform


def fetch_block(
    clr: cooler.Cooler,
    row: GenomicRange,
    col: GenomicRange,
    binsize: float,
    balance: str | bool,
):
    shape = (_bin_count(row, binsize), _bin_count(col, binsize))
    if row.is_out_of_bounds or col.is_out_of_bounds:
        # Past the last chromosome: padding, not missing data. Same contract as
        # the 1D path -- the caller supplies a correctly shaped NaN block.
        return np.full(shape, np.nan, dtype=np.float32)

    block = clr.matrix(balance=balance).fetch(row.as_tuple(), col.as_tuple())
    return block.astype(np.float32)


def _bin_count(interval: GenomicRange, binsize: float) -> int:
    """Bins a fetch of this interval returns.

    Outward rounding -- ``floor(start / b)`` to ``ceil(end / b)`` -- because
    cooler returns every bin that *overlaps* the region. That is one more than
    ``ceil(span / b)`` whenever the interval starts mid-bin, and getting it wrong
    makes blocks in the same strip disagree on shape. Same formula as
    the chromosome-relative bin slice ``floor(start/b) .. ceil(end/b)``.
    """
    return math.ceil(interval.end / binsize) - math.floor(
        interval.start / binsize
    )


class CoolerTileset(BaseTileset):
    """An .mcool served as a 2D matrix tileset."""

    datatype = "matrix"
    ndim = 2
    tile_kind = TileKind.DENSE
    modifiers = COOLER_TRANSFORM
    options = frozenset()

    grid_policy = GridPolicy.SEQUENTIAL

    def __init__(self, path, policy: TilePolicy = DEFAULT_POLICY):
        self._path = path
        self._file = None
        self._policy = policy
        self._info = None

    # --- resource lifetime --------------------------------------------------

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self._path, "r")
            if "resolutions" not in self._file:
                raise ValueError(
                    f"{self._path} has no 'resolutions' group. Legacy "
                    "multi-resolution cooler files are served by cooler.py."
                )
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    # --- ProvidesChromsizes -------------------------------------------------

    def chromsizes(self) -> Chromsizes:
        clr = self._cooler(self.resolutions[0])
        return Chromsizes(
            tuple(clr.chromnames), tuple(int(v) for v in clr.chromsizes.values)
        )

    # --- the protocol -------------------------------------------------------

    @property
    def policy(self) -> TilePolicy:
        return self._policy

    @cached_property
    def resolutions(self) -> tuple[int, ...]:
        """The stored resolutions, ascending.

        Cached: `_build_info` walks it repeatedly and each access otherwise
        re-reads the h5py group.
        """
        return tuple(sorted(int(r) for r in self.file["resolutions"].keys()))

    def info(self) -> TilesetInfo:
        if self._info is None:
            self._info = self._build_info()
        return self._info

    def tiles(self, ids):
        """Tiles for ``ids``, omitting any position the ladder does not hold.

        Skipping rather than raising is the pre-existing contract for this
        format specifically. ``clodius/tiles/cooler.py`` -- the direct
        predecessor -- ``continue``s past a zoom above the ladder (twice, once
        per ladder form) and filters out-of-bounds positions before generating
        anything, so the caller gets a short list rather than an error. The
        ``>=`` boundary fix in this PR exists precisely so the equality case
        reaches that skip instead of falling through to an ``IndexError``, and
        ``test/tiles/test_conformance.py`` asserts the resulting empty list.

        It does not generalize across formats, so it is not lifted into
        ``BaseTileset``: ``tiles/mrmatrix.py`` raises ``ValueError`` for the
        same condition and ``tiles/bigwig.py`` has no bound check at all, which
        is an accident rather than a contract. The other ``tiles_v2`` tilesets
        keep propagating ``TileOutOfBounds`` until each one's own legacy
        behavior is established.

        Only ``TileOutOfBounds`` is swallowed. A malformed transform is a
        ``TileError`` and still propagates, because a client asking for a
        balancing column that does not exist has made a different kind of
        mistake than one asking for a tile off the end of the genome.
        """
        out = []
        for tid in ids:
            try:
                out.append((tid, self._tile(tid)))
            except TileOutOfBounds:
                continue
        return out

    # --- internals ----------------------------------------------------------

    def _cooler(self, resolution) -> cooler.Cooler:
        # int() is load-bearing: canvas.binsize is a float, and the resolution
        # groups are named "8000", not "8000.0".
        return cooler.Cooler(self.file["resolutions"][str(int(resolution))])

    def _build_info(self) -> TilesetInfo:
        resolutions = self.resolutions
        clr = self._cooler(resolutions[0])
        chromsizes = self.chromsizes()

        # Any non-coordinate bin column is a usable weight column, so advertise
        # them all rather than a hardcoded four. Offered only if present at
        # *every* resolution, since the client may request one at any zoom.
        #
        # Ordered by the bin table's own column order, not by iterating a set:
        # the legacy module does `for t in <set>`, whose order varies with the
        # process hash seed, so its dropdown could reorder between restarts.
        per_resolution = [
            set(self._cooler(r).bins().columns) for r in resolutions
        ]
        shared = set.intersection(*per_resolution) if per_resolution else set()
        transforms = [
            c
            for c in clr.bins().columns
            if c not in BIN_COORDINATE_COLUMNS and c in shared
        ]

        # No max_width: for an explicit ladder the extent belongs to the zoom
        # level, not the tileset, and Canvas derives it per level. Emitting a
        # single value taken from the coarsest resolution is only correct for a
        # power-of-two ladder; on the 4DN standard set it over-reports n_tiles
        # at every finer zoom (200 vs 121 at z=6).
        # `mirror_tiles` is passed as a constructor keyword rather than
        # assigned afterwards: TilesetInfo is frozen, and its cached
        # derivations are only sound because it is.
        extras = {}
        if clr.info.get("storage-mode") == "square":
            extras["mirror_tiles"] = "false"

        return TilesetInfo(
            min_pos=[1, 1],
            max_pos=[chromsizes.total_length, chromsizes.total_length],
            resolutions=list(resolutions),
            tile_size=TILE_SIZE,
            chromsizes=chromsizes.to_pairs(),
            transforms=[
                {"name": TRANSFORM_LABELS.get(c, c), "value": c}
                for c in transforms
            ],
            **extras,
        )

    def _tile(self, tid: TileId):
        x, y = tid.pos
        canvas = self.info().canvas(tid.z)
        binsize = canvas.binsize
        # Zero-length intervals -- a tile boundary landing exactly on a
        # chromosome boundary -- contribute no bins and no bp, so they only
        # cost a `matrix().fetch` round-trip.
        col_intervals = [iv for iv in canvas.invert(x) if iv.end > iv.start]
        row_intervals = [iv for iv in canvas.invert(y) if iv.end > iv.start]

        clr = self._cooler(canvas.binsize)
        balance = resolve_balance(clr, tid.modifier)

        blocks = [
            [
                fetch_block(clr, row, col, binsize, balance)
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
