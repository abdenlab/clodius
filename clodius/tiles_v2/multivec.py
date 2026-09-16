"""Multivec (.mv5)"""

from __future__ import annotations

import json
import math

import h5py
import numpy as np

from clodius.core.coords import Chromsizes, GenomicRange
from clodius.core.errors import MalformedTileId
from clodius.core.tile import DenseTile, DenseTilePayload
from clodius.core.policies import TilePolicy, reconcile
from clodius.core.tileid import TileId
from clodius.core.tileset import BaseTileset, TilesetInfo


def bin_slice(start: int, end: int, binsize: int) -> tuple[int, int]:
    """Chromosome-relative bin range answering ``[start, end)``.

    The stored bins are fixed at each chromosome's start, while tile edges fall
    wherever the concatenated coordinate space puts them, so from the second
    chromosome on the two grids can be out of phase -- by a constant amount
    within a chromosome. The shift cannot be removed without re-aggregating,
    but rounding the start edge to the *nearest* stored bin keeps it within
    half a bin rather than systematically displacing the data rightward by up
    to a whole one. Same reasoning as ``cooler._first_bin``.

    The end rounds outward, so the bins still cover the interval and a ragged
    final bin is still offered to the grid reconciler.
    """
    lo = math.floor(int(start) / binsize + 0.5)
    hi = math.ceil(int(end) / binsize)
    if end > start and lo >= hi:
        lo = hi - 1
    return (lo, max(lo, hi))


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

    ndim = 1
    datatype = "multivec"
    modifiers = None
    options = frozenset()

    def __init__(
        self,
        path,
        policy: TilePolicy | None = None,
        tile_size: int | None = None,
    ):
        self._path = path
        self._file = None
        self._info = None
        self.policy = policy or None
        self.tile_size = tile_size or int(self.file["info"].attrs["tile-size"])

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
    def resolutions(self) -> tuple[int, ...]:
        return tuple(
            sorted((int(r) for r in self.file["resolutions"]), reverse=True)
        )

    def info(self) -> TilesetInfo:
        if self._info is None:
            self._info = self._build_info()
        return self._info

    def tiles(self, ids, options=None) -> list[tuple[TileId, DenseTilePayload]]:
        # Parsed once, not per tile: it is one setting for the whole batch, and
        # a bad one should fail the request rather than fifteen tiles over.
        aggregation = parse_row_aggregation(options, self.info().shape[1])
        return [(tid, self._tile(tid, aggregation)) for tid in ids]

    # --- internals ----------------------------------------------------------

    def _build_info(self) -> TilesetInfo:
        chromsizes = self.chromsizes()
        resolutions = self.resolutions
        tile_size = self.tile_size
        n_rows = self._n_rows(resolutions[0])

        info = TilesetInfo(
            min_pos=[0],
            max_pos=[chromsizes.total_length],
            resolutions=list(resolutions),
            tile_size=tile_size,
            chromsizes=chromsizes.to_pairs(),
            shape=[tile_size, n_rows],
        )
        for field, value in self._row_metadata().items():
            setattr(info, field, value)
        return info

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
            attrs = self.file["resolutions"][str(self.resolutions[0])].attrs
            if "row_infos" in attrs:
                out["row_infos"] = [_decode_json(r) for r in attrs["row_infos"]]
        return out

    def _tile(self, tid: TileId, aggregation=None) -> DenseTilePayload:
        info = self.info()
        canvas = info.canvas(tid.z)
        binsize = int(canvas.binsize)
        n_rows = info.shape[1]
        grp = self.file[f"resolutions/{binsize}/values"]

        chunks = [
            (fetch(grp, gr, binsize, n_rows), gr.end - gr.start)
            for gr in canvas.invert(tid.pos[0])
        ]

        dense = reconcile(
            chunks,
            binsize,
            expected_bins=canvas.tile_size,
        )
        if aggregation is None:
            # Transposed to (n_rows, tile_size): the client reads a stack of
            # tracks, not a run of vectors.
            stacked = dense.T
        else:
            # One output row per group, in the order asked for. `shape` has to
            # describe what is actually sent -- the client unflattens with it --
            # so it is (n_groups, tile_size) here while tileset_info keeps
            # advertising the file's full row count.
            groups, func = aggregation
            stacked = np.stack(
                [func(dense[:, rows], axis=1) for rows in groups]
            )
        return DenseTile(stacked, shape=stacked.shape).to_dict(stats=False)


# Row aggregation, the one per-request option in the wild -----------------------
#
# `HorizontalMultivecTrack` with `selectRowsAggregationMethod: "server"` sends
# `{"aggGroups": [...], "aggFunc": "mean"}` in the POST body. `aggGroups` both
# selects and *reorders* rows: an entry is one row index, or a list of indices
# to combine into a single output row. The tile then has one row per group.
#
# Client-side aggregation -- the default -- sends nothing, and the tile is the
# whole matrix.

AGG_FUNCS = {
    "sum": np.sum,
    "mean": np.mean,
    "median": np.median,
    "std": np.std,
    "var": np.var,
    "min": np.amin,
    "max": np.amax,
}


def parse_row_aggregation(options, n_rows: int):
    """Validate ``{"aggGroups", "aggFunc"}`` against a tileset of ``n_rows``.

    Returns ``(groups, func)`` or ``None`` when the request asks for no
    aggregation. Raises :class:`MalformedTileId` -- a whole-batch failure,
    since options arrive once for every tile in the request.
    """
    if not options:
        return None
    groups = options.get("aggGroups")
    func_name = options.get("aggFunc")
    if groups is None and func_name is None:
        return None
    if groups is None or func_name is None:
        raise MalformedTileId(
            "row aggregation needs both 'aggGroups' and 'aggFunc'; got "
            f"{sorted(options)}"
        )
    if func_name not in AGG_FUNCS:
        raise MalformedTileId(
            f"{func_name!r} is not a valid aggFunc; expected one of "
            f"{sorted(AGG_FUNCS)}"
        )

    normalized = []
    for group in groups:
        rows = [group] if isinstance(group, int) else list(group)
        if not rows:
            raise MalformedTileId("an aggGroups entry is empty")
        for row in rows:
            if not isinstance(row, int) or not 0 <= row < n_rows:
                raise MalformedTileId(
                    f"row {row!r} is out of range for a tileset of {n_rows} "
                    "rows"
                )
        normalized.append(rows)
    return normalized, AGG_FUNCS[func_name]


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
#
# Two things multivec does differently from the other dense types, preserved here:
# - **Padding is zeros, not NaN.** Where bigwig fills the band past the last
#   chromosome with NaN (unmappable), multivec fills it with 0 (no signal). Both
#   are defensible; they are simply different, and the client draws them
#   differently.
# - **The payload omits ``size``/``min_value``/``max_value``**, emitting only
#   ``dense``/``dtype``/``shape`` -- see ``DenseTile.to_dict(stats=False)``.
