"""Multivec (.mv5) rewritten against clodius.core.

The third prototype, and the first to combine an **explicit ladder** with the
**SEQUENTIAL** policy -- bigwig is sequential with an implicit ladder, cooler is
explicit with 2D blocks. It is also the first with a *shaped* payload: each bin
carries a vector of per-row values, so a tile is ``(n_rows, tile_size)`` rather
than a flat run.

Two things it does differently from the other dense types, preserved here:

- **Padding is zeros, not NaN.** Where bigwig fills the band past the last
  chromosome with NaN (unmappable), multivec fills it with 0 (no signal). Both
  are defensible; they are simply different, and the client draws them
  differently.
- **The payload omits ``size``/``min_value``/``max_value``**, emitting only
  ``dense``/``dtype``/``shape``. That is ``DenseTileShapedPayload``; see
  ``DenseTile.to_dict(stats=False)``.
"""

from __future__ import annotations

import json
import math

import h5py
import numpy as np

from clodius.core import (
    DEFAULT_POLICY,
    BaseTileset,
    Chromsizes,
    DenseTile,
    GenomicRange,
    GridPolicy,
    TileId,
    TileKind,
    TilePolicy,
    TilesetInfo,
    reconcile_sequential,
)


def bin_slice(start: int, end: int, binsize: int) -> tuple[int, int]:
    """Chromosome-relative bin range covering ``[start, end)``.

    Outward rounding, so the returned bins fully cover the interval.
    """
    return (int(start) // binsize, math.ceil(int(end) / binsize))


def fetch(
    grp: h5py.Group, interval: GenomicRange, binsize: int, n_rows: int
) -> np.ndarray:
    """One interval's bins, shaped ``(n_bins, n_rows)``.

    Reads a slice of the stored per-chromosome array. Unlike bigwig, which asks
    its reader for a bin count, the bins here are materialized in the file, so
    the slice bounds *are* the bin count.
    """
    if interval.is_out_of_bounds:
        # Past the last chromosome. Zeros rather than NaN -- multivec's
        # convention, meaning "no signal" rather than "unmappable".
        n_bins = math.ceil((interval.end - interval.start) / binsize)
        return np.zeros((n_bins, n_rows))

    lo, hi = bin_slice(interval.start, interval.end, binsize)
    return grp[interval.name][lo:hi]


class MultivecTileset(BaseTileset):
    """A multi-resolution multivec served as a stack of 1D vectors."""

    datatype = "multivec"
    ndim = 1
    tile_kind = TileKind.DENSE
    modifiers = None
    options = frozenset()

    grid_policy = GridPolicy.SEQUENTIAL
    # Floor, matching what the legacy module intends. Note the legacy
    # accumulator never decrements after a drop, so once its drift crosses the
    # threshold it drops from every subsequent chromosome -- an over-correction
    # masked by a hard `[: shape[0]]` clamp. See the notes at the bottom.
    grid_threshold = 1.0

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
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    # --- ProvidesChromsizes -------------------------------------------------

    def chromsizes(self) -> Chromsizes:
        chroms = self.file["chroms"]
        names = tuple(
            n.decode("utf8") if isinstance(n, bytes) else n
            for n in chroms["name"][:]
        )
        return Chromsizes(names, tuple(int(v) for v in chroms["length"][:]))

    # --- the protocol -------------------------------------------------------

    @property
    def policy(self) -> TilePolicy:
        return self._policy

    @property
    def resolutions(self) -> tuple[int, ...]:
        """The stored resolutions, ascending.

        Ascending to match cooler and the `Ladder.EXPLICIT` contract, which
        fixes the serialized order so two tilesets do not emit the same
        `tileset_info` field two ways. `z` still indexes the ladder
        coarsest-first; `TilesetInfo.resolution_for` re-sorts.
        """
        return tuple(sorted(int(r) for r in self.file["resolutions"]))

    def info(self) -> TilesetInfo:
        if self._info is None:
            self._info = self._build_info()
        return self._info

    def tiles(self, ids):
        return [(tid, self._tile(tid)) for tid in ids]

    # --- internals ----------------------------------------------------------

    def _build_info(self) -> TilesetInfo:
        chromsizes = self.chromsizes()
        resolutions = self.resolutions
        tile_size = int(self.file["info"].attrs["tile-size"])
        n_rows = self._n_rows(max(resolutions))

        # Row metadata is passed as constructor keywords rather than assigned
        # afterwards: TilesetInfo is frozen, and its cached derivations are
        # only sound because it is.
        return TilesetInfo(
            min_pos=[0],
            max_pos=[chromsizes.total_length],
            resolutions=list(resolutions),
            tile_size=tile_size,
            chromsizes=chromsizes.to_pairs(),
            shape=[tile_size, n_rows],
            **self._row_metadata(),
        )

    def _n_rows(self, resolution: int) -> int:
        grp = self.file[f"resolutions/{resolution}/values"]
        first = next(iter(grp))
        return int(grp[first].shape[1])

    def _row_metadata(self) -> dict:
        """``row_infos`` / ``category_infos``, decoded from wherever they live.

        The legacy module checks three locations with two encodings and two
        JSON-decode fallbacks (``multivec.py:295-315``); this keeps the same
        tolerance without the nesting.
        """
        out = {}
        info_group = self.file.get("info")
        if info_group is not None:
            for key in ("row_infos", "category_infos"):
                if key in info_group:
                    out[key] = _decode_json(info_group[key][()])

        if "row_infos" not in out:
            coarsest = max(self.resolutions)
            attrs = self.file["resolutions"][str(coarsest)].attrs
            if "row_infos" in attrs:
                out["row_infos"] = [_decode_json(r) for r in attrs["row_infos"]]
        return out

    def _tile(self, tid: TileId):
        info = self.info()
        canvas = info.canvas(tid.z)
        binsize = int(canvas.binsize)
        n_rows = info.shape[1]
        grp = self.file[f"resolutions/{binsize}/values"]

        chunks = [
            (fetch(grp, gr, binsize, n_rows), gr.end - gr.start)
            # A tile boundary landing exactly on a chromosome boundary yields
            # a zero-length interval: no bins, no bp, only a wasted slice.
            for gr in canvas.invert(tid.pos[0])
            if gr.end > gr.start
        ]

        dense = reconcile_sequential(
            chunks,
            binsize,
            expected_bins=canvas.tile_size,
            threshold=self.grid_threshold,
        )
        # Transposed to (n_rows, tile_size): the client reads a stack of tracks,
        # not a run of vectors.
        stacked = dense.T
        return DenseTile(stacked, shape=stacked.shape).to_dict(stats=False)


def _decode_json(raw):
    """Row metadata is stored as bytes or str, JSON or plain."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf8")
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


# --- Notes ------------------------------------------------------------------
#
# 1. The legacy accumulator is broken in a way bigwig's was not. At
#    `multivec.py:203` it drops a bin without decrementing
#    `current_binned_data_position`, so drift only ever grows: once it crosses
#    the threshold it keeps dropping from every later chromosome. The damage is
#    hidden by `np.concatenate(arrays)[: shape[0]]` at `:235`, plus zero-padding
#    in `get_single_tile` when the result is short. So the tile is always the
#    right *length* and quietly the wrong *content*.
#
#    Using `expected_bins` here is the opposite bet: assert the count instead of
#    clamping it, so an accounting error is loud.
#
# 2. Zeros vs NaN for padding is a real semantic split across the dense types,
#    not an implementation detail -- see the module docstring.
#
# 3. `shape` is emitted as `[tile_size, n_rows]` in tileset_info but the payload
#    ships `(n_rows, tile_size)`. That transposition is in the legacy module
#    too; it is confusing but load-bearing.
