from __future__ import annotations

import numpy as np
import pybigtools

from clodius.core import (
    DEFAULT_POLICY,
    BaseTileset,
    Chromsizes,
    DenseTile,
    GenomicRange,
    GridPolicy,
    ModifierSpec,
    TileId,
    TileKind,
    TilePolicy,
    TilesetInfo,
    natsorted,
    quadtree_depth,
    reconcile_sequential,
)

TILE_SIZE = 1024

AGGREGATION_MODES = {
    "mean": "Mean",
    "min": "Min",
    "max": "Max",
    "std": "Standard Deviation",
    "sum": "Sum",
}

# Range modes return several values per bin instead of one.
RANGE_MODES = {
    "minMax": ("Min-Max", ("min", "max")),
    "whisker": ("Whisker", ("min", "max", "mean", "std")),
}


def fetch(f, interval: GenomicRange, binsize: float, stats: tuple[str, ...]):
    """One interval's bins, uncovered positions left as NaN.

    `fillna=None` is passed explicitly rather than omitted. pybigtools 0.3.0
    resolves an unset `fillna` to `None` and warns; passing it gets the same
    NaN semantics the legacy module has without the DeprecationWarning.

    NaN and zero are not interchangeable here: the client draws a gap where a
    bin has no data and a zero-height bar where it has zero signal, and
    `DenseTile.to_dict` picks float32 over float16 precisely when NaN is
    present.
    """
    n_bins = int(np.ceil((interval.end - interval.start) / binsize))
    shape = (n_bins, len(stats)) if len(stats) > 1 else (n_bins,)

    if interval.is_out_of_bounds:
        return np.full(shape, np.nan)

    out = np.zeros(shape)
    args = (*interval.as_tuple(), n_bins)
    try:
        if len(stats) > 1:
            for k, stat in enumerate(stats):
                out[:, k] = f.values(*args, stat, fillna=None)
        else:
            out[:] = f.values(*args, stats[0], fillna=None)
    except Exception as exc:  # noqa: BLE001 -- matches current behavior
        if "No chromomsome with name" not in str(exc):
            raise
        out[:] = np.nan  # supported chromosome absent from the file, e.g. chrM
    return out


class BigWigTileset(BaseTileset):
    """A bigWig served as a 1D vector tileset."""

    datatype = "vector"
    ndim = 1
    tile_kind = TileKind.DENSE
    modifiers = ModifierSpec(
        values=frozenset(
            {"mean", "min", "max", "std", "sum", "minMax", "whisker"}
        ),
        default="mean",
        kind="display mode",
    )
    options = frozenset({"cos"})

    grid_policy = GridPolicy.SEQUENTIAL
    # Floor, matching the fixed legacy path (`tiles/bigwig.py`, `offset >=
    # binsize`) and MultivecTileset. Midpoint changes which bins are dropped
    # without changing how many, so `expected_bins` cannot catch the
    # difference -- and byte-for-byte comparability against legacy is what
    # this rewrite is being validated by.
    grid_threshold = 1.0

    def __init__(
        self,
        path,
        chromsizes: Chromsizes | None = None,
        chromsizes_alts: dict[str, Chromsizes] | None = None,
        policy: TilePolicy = DEFAULT_POLICY,
    ):
        self._path = path
        self._file = None
        self._policy = policy
        if chromsizes is not None:
            self._chromsizes = chromsizes
        else:
            chroms = self.file.chroms()
            names = natsorted(chroms.keys())
            self._chromsizes = Chromsizes(
                tuple(names), tuple(chroms[n] for n in names)
            )
        self._info = self._info_for(self._chromsizes)

        # Alternate pre-defined chromsizes orderings.
        # The client can select these per tile via `,cos:<uid>`.
        self._chromsizes_alts = chromsizes_alts or {}

    # --- resource lifetime --------------------------------------------------

    @property
    def file(self):
        if self._file is None:
            self._file = pybigtools.open(self._path)
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    # --- ProvidesChromsizes -------------------------------------------------

    def chromsizes(self) -> Chromsizes:
        return self._chromsizes

    # --- the protocol -------------------------------------------------------

    @property
    def policy(self) -> TilePolicy:
        return self._policy

    def info(self) -> TilesetInfo:
        return self._info

    def tiles(self, ids):
        return [(tid, self._tile(tid)) for tid in ids]

    # --- internals ----------------------------------------------------------

    def _info_for(self, chromsizes: Chromsizes) -> TilesetInfo:
        """Tileset info under a given set of chromsizes."""
        max_zoom = quadtree_depth(chromsizes.total_length, TILE_SIZE)
        max_width = TILE_SIZE * 2**max_zoom

        return TilesetInfo(
            min_pos=[0],
            max_pos=[max_width],
            max_width=max_width,
            tile_size=TILE_SIZE,
            max_zoom=max_zoom,
            chromsizes=chromsizes.to_pairs(),
            aggregation_modes=[
                {"name": label, "value": key}
                for key, label in AGGREGATION_MODES.items()
            ],
            range_modes=[
                {"name": label, "value": key}
                for key, (label, _) in RANGE_MODES.items()
            ],
        )

    def _tile(self, tid: TileId):
        uid = tid.option("cos")
        if uid is not None and uid in self._chromsizes_alts:
            chromsizes = self._chromsizes_alts[uid]
            info = self._info_for(chromsizes)
        else:
            info = self._info

        stats = (
            RANGE_MODES[tid.modifier][1]
            if tid.modifier in RANGE_MODES
            else (tid.modifier,)
        )

        canvas = info.canvas(tid.z)
        chunks = [
            (
                fetch(self.file, ival, canvas.binsize, stats),
                ival.end - ival.start,
            )
            # A tile boundary landing exactly on a chromosome boundary yields
            # a zero-length interval. It contributes no bins and no bp, so
            # skipping it changes nothing except the wasted backend query.
            for ival in canvas.invert(tid.pos[0])
            if ival.end > ival.start
        ]

        values = reconcile_sequential(
            chunks,
            canvas.binsize,
            expected_bins=canvas.tile_size,
            threshold=self.grid_threshold,
        )
        return DenseTile(values, size=len(stats)).to_dict()
