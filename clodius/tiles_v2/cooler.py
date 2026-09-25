from __future__ import annotations

import math
from contextlib import ExitStack

import cooler
import h5py
import numpy as np
from cooler.core import CSRReader, DirectRangeQuery2D, FillLowerRangeQuery2D
from cooler.util import open_hdf5

from clodius.core.coords import Chromsizes, GenomicRange
from clodius.core.errors import TileError, TilesetUnavailable
from clodius.core.tile import DenseTile, TileKind
from clodius.core.policies import (
    TilePolicy,
    reconcile_2d,
)
from clodius.core.tileid import ModifierSpec, TileId
from clodius.core.tileset import BaseTileset, TilesetInfo

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
    """One rectangle of the matrix, through the public cooler API."""
    shape = (_bin_count(row, binsize), _bin_count(col, binsize))
    if _is_absent(row, clr.chromnames) or _is_absent(col, clr.chromnames):
        # Padding past the last chromosome, or a chromosome the served
        # chromsizes declare but the file does not hold. Neither is missing
        # data: the tile is simply empty there.
        return np.full(shape, np.nan, dtype=np.float32)
    if 0 in shape:
        return np.empty(shape, dtype=np.float32)

    block = clr.matrix(balance=balance).fetch(
        _query(row, binsize), _query(col, binsize)
    )
    return block.astype(np.float32)


def _is_absent(interval: GenomicRange, names) -> bool:
    """Whether the cooler holds no bins for this interval at all."""
    return interval.is_out_of_bounds or interval.name not in names


def _query(interval: GenomicRange, binsize: float) -> tuple[str, int, int]:
    """Fetch the canvas interval, snapped to the nearest source bin start."""
    name, _, end = interval.as_tuple()
    return (name, int(_first_bin(interval, binsize) * binsize), end)


def bin_offsets(clr: cooler.Cooler) -> dict[str, int]:
    """First bin id of each chromosome, in the cooler's own chromosome order.

    Two reads can be merged into one only if they land on adjacent bins here.
    Adjacency on the canvas does not imply it: the chromosome order served to
    the client need not be the file's.
    """
    offsets, total = {}, 0
    binsize = int(clr.binsize)
    for name, length in clr.chromsizes.items():
        offsets[name] = total
        total += math.ceil(int(length) / binsize)
    return offsets


def _window(
    interval: GenomicRange, binsize: float, offsets: dict[str, int]
) -> tuple[int, int] | None:
    """Bin ids ``[lo, hi)`` that a fetch of this interval returns.

    None where the cooler has no bins to read: past the last chromosome, or on
    a chromosome the served chromsizes declare but the file does not hold.
    """
    if _is_absent(interval, offsets):
        return None
    lo = offsets[interval.name] + _first_bin(interval, binsize)
    return (lo, lo + _bin_count(interval, binsize))


def _merge(windows) -> tuple[int, int] | None:
    """One bin range covering all the windows, or None if they leave a gap.

    Windows must arrive in ascending order. Consecutive tiles may overlap by a
    bin, since each snaps its own start edge to the nearest bin boundary. Empty
    windows are skipped rather than treated as a gap: they read nothing, and
    each block is sliced by its own window, so covering the span an absent
    chromosome sits in is harmless.
    """
    lo = hi = None
    for w in windows:
        if w is None:
            continue
        if lo is None:
            lo, hi = w
        elif lo <= w[0] <= hi:
            hi = max(hi, w[1])
        else:
            return None
    return None if lo is None else (lo, hi)


def _runs(positions, strip):
    """Split sorted tile positions into runs readable as a single fetch."""
    run: list[int] = []
    for p in positions:
        if run and _merge([w for q in run + [p] for w in strip(q)]) is not None:
            run.append(p)
        else:
            if run:
                yield run
            run = [p]
    if run:
        yield run


class BlockReader:
    """Reads one 2D block of a cooler per (row, col) interval pair.

    A tile position maps to one genomic interval per axis, or to several when
    it straddles a chromosome boundary, and each pairing of the two axes'
    intervals is read separately through the public ``Cooler.matrix()`` API.
    """

    def __init__(self, clr: cooler.Cooler, canvas, balance: str | bool) -> None:
        self.clr = clr
        self.canvas = canvas
        self.balance = balance
        self.binsize = canvas.binsize

    def prefetch(self, positions) -> None:
        """Read ahead for the given ``(x, y)`` tiles. Nothing to do here."""

    def block(self, row: GenomicRange, col: GenomicRange) -> np.ndarray:
        """The submatrix of one (row, col) interval pair."""
        return fetch_block(self.clr, row, col, self.binsize, self.balance)

    def close(self) -> None:
        pass

    def __enter__(self) -> BlockReader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class BatchedBlockReader(BlockReader):
    """Reads 2D blocks of a cooler, batching adjacent tile queries.

    :meth:`prefetch` replaces the fetches of a group of tiles with a single
    rectangular bbox wherever their bin ranges are contiguous, and
    :meth:`block` slices them back apart. Anything not prefetched is read on
    demand, so the batching is only an optimization.

    Because pixels are stored CSR, merging fetches of horizontally-adjacent
    bounding boxes eliminates multiple reads of the same matrix rows from disk.
    For bounding boxes crossing the diagonal in symmetric-upper mode, vertical
    consolidation also eliminates redundant I/O.

    As an additional optimization, fetches go through cooler's internal query
    engines rather than ``Cooler.matrix()`` to read the internal index only
    once for the whole batch.
    """

    field = "count"
    chunksize = 10_000_000
    # Weight columns with these names are divisive rather than multiplicative.
    divisive = frozenset({"KR", "VC", "VC_SQRT"})

    def __init__(self, clr: cooler.Cooler, canvas, balance: str | bool) -> None:
        super().__init__(clr, canvas, balance)
        self._offsets = bin_offsets(clr)
        self._strips: dict[int, list[tuple[int, int] | None]] = {}
        self._blocks: list[
            tuple[tuple[int, int], tuple[int, int], np.ndarray]
        ] = []
        self._stack = ExitStack()
        h5 = self._stack.enter_context(open_hdf5(clr.store, **clr.open_kws))
        self._grp = h5[clr.root]
        self._reader = CSRReader(
            self._grp["pixels"], self._grp["indexes/bin1_offset"][:]
        )
        self._query = (
            FillLowerRangeQuery2D
            if clr.storage_mode == "symmetric-upper"
            else DirectRangeQuery2D
        )

    def close(self) -> None:
        self._stack.close()

    def strip(self, pos: int) -> list[tuple[int, int] | None]:
        """Bin windows of one tile position, one per genomic interval."""
        if pos not in self._strips:
            self._strips[pos] = [
                _window(iv, self.binsize, self._offsets)
                for iv in self.canvas.invert(pos)
            ]
        return self._strips[pos]

    def prefetch(self, positions) -> None:
        """Read what the given ``(x, y)`` tiles need, in as few fetches as possible."""
        cols_by_row: dict[int, set[int]] = {}
        for x, y in positions:
            cols_by_row.setdefault(y, set()).add(x)

        col_runs = {
            y: tuple(tuple(r) for r in _runs(sorted(xs), self.strip))
            for y, xs in cols_by_row.items()
        }
        for rows in _row_groups(col_runs, self.strip):
            rspan = _merge([w for y in rows for w in self.strip(y)])
            if rspan is None:
                continue
            for run in col_runs[rows[0]]:
                cspan = _merge([w for x in run for w in self.strip(x)])
                if cspan is not None:
                    self._read(rspan, cspan)

    def block(self, row: GenomicRange, col: GenomicRange) -> np.ndarray:
        """The submatrix of one (row, col) interval pair."""
        rw = _window(row, self.binsize, self._offsets)
        cw = _window(col, self.binsize, self._offsets)
        if rw is None or cw is None:
            # Nothing stored for this interval pair; see :func:`_window`.
            shape = (
                _bin_count(row, self.binsize),
                _bin_count(col, self.binsize),
            )
            return np.full(shape, np.nan, dtype=np.float32)
        if rw[0] == rw[1] or cw[0] == cw[1]:
            return np.empty((rw[1] - rw[0], cw[1] - cw[0]), dtype=np.float32)

        for (r0, r1), (c0, c1), block in self._blocks:
            if r0 <= rw[0] and rw[1] <= r1 and c0 <= cw[0] and cw[1] <= c1:
                return block[rw[0] - r0 : rw[1] - r0, cw[0] - c0 : cw[1] - c0]
        return self._read(rw, cw)

    def _read(
        self, rspan: tuple[int, int], cspan: tuple[int, int]
    ) -> np.ndarray:
        bbox = (*rspan, *cspan)
        block = self._query(
            self._reader, self.field, bbox, self.chunksize
        ).to_array()
        if self.balance:
            block = block * np.outer(*self._bias(rspan, cspan))
        block = block.astype(np.float32)
        self._blocks.append((rspan, cspan, block))
        return block

    def _bias(self, rspan, cspan) -> tuple[np.ndarray, np.ndarray]:
        weights = self._grp["bins"][self.balance]
        bias1 = weights[rspan[0] : rspan[1]]
        bias2 = bias1 if rspan == cspan else weights[cspan[0] : cspan[1]]
        if self.balance in self.divisive:
            bias1, bias2 = 1 / bias1, 1 / bias2
        return bias1, bias2


def _row_groups(col_runs, strip):
    """Group tile rows that can share a fetch (request the same columns)."""
    group: list[int] = []
    for y in sorted(col_runs):
        if (
            group
            and col_runs[y] == col_runs[group[0]]
            and _merge([w for q in group + [y] for w in strip(q)]) is not None
        ):
            group.append(y)
        else:
            if group:
                yield group
            group = [y]
    if group:
        yield group


def _first_bin(interval: GenomicRange, binsize: float) -> int:
    """Source bin whose start is nearest the tile start.

    Tile edges fall wherever the concatenated coordinate space puts them. From
    the second chromosome on, the source and target bin grids are out of phase.
    The phase shift is constant within a chromosome. We can't remove the shift
    without re-aggregating or interpolating, but we can round the tile start
    edge to the nearest source bin boundary to limit the shift to within
    +/-0.5 of a bin instead of systematically right-shifted up to +1 bin.
    """
    # floor(x + 0.5) rather than round, which rounds halves to even.
    lo = math.floor(interval.start / binsize + 0.5)
    last = math.ceil(interval.end / binsize)
    if interval.end > interval.start and lo >= last:
        return last - 1
    return lo


def _bin_count(interval: GenomicRange, binsize: float) -> int:
    """Bins a fetch of this interval returns.

    From the nearest bin boundary at or near ``start`` out to the last bin
    overlapping ``end``, because cooler returns every bin overlapping the range.
    Blocks in the same strip must agree on shape, so this and :func:`_window`
    have to describe the same window.
    """
    return max(
        0,
        math.ceil(interval.end / binsize) - _first_bin(interval, binsize),
    )


class CoolerTileset(BaseTileset):
    """An .mcool served as a 2D matrix tileset.

    Parameters
    ----------
    batched : bool, optional [default: True]
        Consolidate fetch operations for sub-batches of tiles to perform as few
        reads as the disk layout allows. If False, perform fetches for each
        tile independently.
    """

    ndim = 2
    datatype = "matrix"
    modifiers = COOLER_TRANSFORM
    options = frozenset()

    def __init__(
        self,
        path,
        chromsizes: Chromsizes | None = None,
        policy: TilePolicy | None = None,
        tile_size: int = TILE_SIZE,
        batched: bool = True,
    ):
        self._path = path
        self._file = None
        self._info = None
        if chromsizes is not None:
            self._chromsizes = chromsizes
        else:
            clr = self._cooler(self.resolutions[0])
            self._chromsizes = Chromsizes(
                tuple(clr.chromnames),
                tuple(int(v) for v in clr.chromsizes.values),
            )
        self.policy = policy or TilePolicy()
        self.tile_size = tile_size
        self.reader = BatchedBlockReader if batched else BlockReader

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
        return self._chromsizes

    @property
    def resolutions(self) -> tuple[int, ...]:
        return tuple(sorted(int(r) for r in self.file["resolutions"].keys()))

    def info(self) -> TilesetInfo:
        if self._info is None:
            self._info = self._build_info()
        return self._info

    def tiles(self, ids, options=None) -> list[tuple[TileId, TileKind]]:
        """One entry per requested id; a refusal rides in the payload slot.

        Overrides `BaseTileset.tiles` because tiles are batched per zoom
        level and transform, and `_tile` reads through a shared reader.

        Overriding also means the base's option refusal is not inherited, so
        it is restated here: this tileset reads no batch options, and the
        protocol makes a malformed one a whole-batch failure rather than
        something to serve tiles in spite of.
        """
        if options:
            raise TilesetUnavailable(
                f"{type(self).__name__} accepts no tile options; got "
                f"{sorted(options)}"
            )

        # Batched per zoom level and transform, since those decide which cooler
        # and which weights a tile is read from.
        ids = list(ids)
        batches: dict[tuple[int, str | None], list[int]] = {}
        for i, tid in enumerate(ids):
            batches.setdefault((tid.z, tid.modifier), []).append(i)

        payloads: dict[int, TileKind] = {}
        for (z, modifier), indices in batches.items():
            # Guarded like any other refusal. A zoom past the ladder and a
            # transform the file does not carry are both ordinary client
            # requests, and both raise TileError by contract -- outside a
            # try they would fail the whole request, and because `batches`
            # is keyed on (z, modifier), one bad id would take the *other*
            # zoom groups down with it.
            try:
                canvas = self.info().canvas(z)
                clr = self._cooler(canvas.binsize)
                balance = resolve_balance(clr, modifier)
            except TileError as exc:
                for i in indices:
                    payloads[i] = exc.to_dict()
                continue

            # One entry per requested id; a refusal rides in the payload slot.
            #
            # Positions are screened before the prefetch rather than inside
            # the serving loop, because the prefetch reads every position in
            # the batch up front: one off-lattice id would raise there and
            # take its fifteen well-formed neighbours down with it, which is
            # the whole failure this guard exists to prevent. `invert` is the
            # canonical check, so the bounds test is not restated here.
            servable = []
            for i in indices:
                try:
                    for pos in ids[i].pos:
                        canvas.invert(pos)
                except TileError as exc:
                    payloads[i] = exc.to_dict()
                else:
                    servable.append(i)

            # Nothing to read: the reader's constructor materializes the whole
            # bin1 offset index, which is tens of megabytes on a fine-binned
            # genome, and a raise there would discard the error payloads just
            # recorded for this group.
            if not servable:
                continue

            # The catch stays narrow. An unopenable file is a whole-request
            # failure, and answering with sixteen cheerful error payloads
            # would be a lie -- which is what the cooler test asserts.
            with self.reader(clr, canvas, balance) as reader:
                reader.prefetch(ids[i].pos for i in servable)
                for i in servable:
                    try:
                        payloads[i] = self._tile(ids[i], reader)
                    except TileError as exc:
                        payloads[i] = exc.to_dict()
        return [(tid, payloads[i]) for i, tid in enumerate(ids)]

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
        #
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

    def _tile(self, tid: TileId, reader: BlockReader | None = None):
        """One tile's payload, from a reader opened for the whole batch.

        Widens `BaseTileset._tile` rather than contradicting it. The reader is
        genuinely required -- `tiles()` opens one per zoom and modifier and
        shares it across that group -- but a signature that adds a *required*
        parameter is not an implementation of the hook it appears to override.
        """
        if reader is None:
            raise TypeError(
                f"{type(self).__name__}._tile needs a reader; call tiles(), "
                f"which opens one per batch"
            )

        x, y = tid.pos
        canvas = reader.canvas
        binsize = canvas.binsize
        col_intervals = list(canvas.invert(x))
        row_intervals = list(canvas.invert(y))

        blocks = [
            [reader.block(row, col) for col in col_intervals]
            for row in row_intervals
        ]

        # Reconcile the blocks into a single 2D array.
        out = reconcile_2d(
            blocks,
            row_spans=[iv.end - iv.start for iv in row_intervals],
            col_spans=[iv.end - iv.start for iv in col_intervals],
            binsize=binsize,
            expected_shape=(canvas.tile_size, canvas.tile_size),
        )
        return DenseTile(out.ravel()).to_dict()
