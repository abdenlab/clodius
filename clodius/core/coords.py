from __future__ import annotations

import math
import re
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Iterable, Iterator, Literal, Sequence

import bioframe
import numpy as np


@dataclass(frozen=True, slots=True)
class GenomicRange:
    """A range on a single chromosome."""

    cid: int
    name: str | None
    start: int
    end: int

    @property
    def is_out_of_bounds(self) -> bool:
        return self.name is None

    def as_tuple(self) -> tuple[str, int, int]:
        if self.name is None:
            raise ValueError("Out-of-bounds range")
        return (self.name, self.start, self.end)


class Chromsizes:
    """An ordered chromosome name -> length mapping, with derived views.

    Ordering matters. It defines the absolute coordinate space for tiling.
    """

    __slots__ = ("_names", "_lengths", "_offsets", "_bounds")

    def __init__(self, names: tuple[str, ...], lengths: tuple[int, ...]):
        if len(names) != len(lengths):
            raise ValueError(
                f"got {len(names)} names but {len(lengths)} lengths"
            )
        self._names = tuple(names)
        self._lengths = tuple(int(x) for x in lengths)
        self._offsets: dict[str, int] | None = None
        self._bounds = None

    @classmethod
    def from_pairs(cls, pairs) -> Chromsizes:
        names, lengths = zip(*pairs) if pairs else ((), ())
        return cls(tuple(names), tuple(int(x) for x in lengths))

    @classmethod
    def from_series(cls, series) -> Chromsizes:
        return cls(tuple(series.index), tuple(int(x) for x in series.values))

    @classmethod
    def from_assembly(
        cls,
        assembly: str,
        roles: list[str] | Literal["all"] | None,
        units: list[str] | Literal["all"] | None,
    ) -> Chromsizes:
        return cls.from_series(
            bioframe.assembly_info(assembly, roles, units).chromsizes
        )

    @classmethod
    def from_file(cls, path_or_handle) -> Chromsizes:
        cs = bioframe.read_chromsizes(
            path_or_handle, filter_chroms=False
        ).to_dict()
        key = cmp_to_key(_natcmp)
        return cls.from_pairs(sorted(cs.items(), key=lambda x: key(x[0])))

    def to_pairs(self) -> list[list]:
        return [
            [name, length] for name, length in zip(self._names, self._lengths)
        ]

    def to_series(self):
        import pandas as pd

        return pd.Series(self._lengths, index=self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __repr__(self) -> str:
        return f"<Chromsizes {len(self)} chroms, {self.total_length:,} bp>"

    @property
    def _boundaries(self):
        """``[0, len0, len0+len1, ..., total]``"""
        if self._bounds is None:
            self._bounds = np.concatenate(
                [[0], np.cumsum(self._lengths)], dtype=int
            )
        return self._bounds

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def lengths(self) -> tuple[int, ...]:
        return self._lengths

    @property
    def total_length(self) -> int:
        """Total length of the concatenated genome."""
        return sum(self._lengths)

    @property
    def offsets(self) -> dict[str, int]:
        """Cumulative start offset of each chromosome."""
        if self._offsets is None:
            offsets, running = {}, 0
            for name, length in zip(self._names, self._lengths):
                offsets[name] = running
                running += length
            self._offsets = offsets
        return self._offsets

    def transform(self, gr: GenomicRange) -> tuple[int, int]:
        """Convert a genomic range to absolute coordinates.

        Parameters
        ----------
        gr : GenomicRange
            A genomic range to transform.

        Returns
        -------
        tuple[int, int]
            A half-open absolute range in the concatenated coordinate space.

        Notes
        -----
        Raises KeyError if the chromosome is unknown, ValueError if the range is
        out-of-bounds.

        Examples
        --------
        e.g. with {"chr1": 100, "chr2": 200}
        ``(chr1, 10, 20) --> (10, 20)`
        ``(chr2, 10, 20) --> (110, 120)`
        """
        if gr.name is None:
            raise ValueError("Out-of-bounds range")
        offset = self.offsets.get(gr.name)
        if offset is None:
            raise KeyError(f"Unknown chromosome {gr.name!r}")
        return (offset + gr.start, offset + gr.end)

    def invert(self, span: tuple[int, int]) -> Iterator[GenomicRange]:
        """Split a half-open canvas range into per-chromosome intervals.

        Parameters
        ----------
        span : tuple[int, int]
            A half-open absolute range in the concatenated coordinate space.

        Returns
        -------
        Iterator[GenomicRange]
            A generator of :class:`GenomicRange` objects

        Notes
        -----
        Chromosome boundary behavior:
        * Half-open convention: a position exactly on a boundary belongs to
          the following chromosome.
        * Negative start raises.
        * If end extends past the last chromosome, a final interval is returned
          and flagged :attr:`~GenomicRange.is_out_of_bounds`.`

        Examples
        --------
        e.g. with {"chr1": 100, "chr2": 200}:
        ``(90, 150) --> [(chr1, 90, 100), (chr2, 0, 50)]``
        """
        start, end = span
        if start < 0:
            raise ValueError(f"negative start position: {start}")
        if end < start:
            raise ValueError(f"end {end} precedes start {start}")

        bounds = self._boundaries
        n = len(self._names)

        cid_lo, cid_hi = np.searchsorted(bounds, [start, end], side="right") - 1
        cid_lo, cid_hi = int(cid_lo), int(cid_hi)

        pos = start - bounds[cid_lo]

        for cid in range(cid_lo, cid_hi):
            # cid_hi is at most n, so the loop never reaches an invalid index
            yield GenomicRange(
                cid=cid,
                name=self._names[cid],
                start=int(pos),
                end=int(self._lengths[cid]),
            )
            pos = 0

        yield GenomicRange(
            cid=cid_hi,
            name=self._names[cid_hi] if cid_hi < n else None,
            start=int(pos),
            end=int(end - bounds[cid_hi]),
        )


_DIGITS = re.compile(r"(\d+)", re.U)


def _natcmp(x: str, y: str) -> int:
    if x.find("_") >= 0:
        x_parts = x.split("_")
        if y.find("_") >= 0:
            # chr_1 vs chr_2 -- compare the parts after the underscore
            return _natcmp(x_parts[1], y.split("_")[1])
        # chr_1 vs chr1 -- the plain name comes first
        return 1
    if y.find("_") >= 0:
        return -1

    for key in ("m", "y", "x"):
        # Deliberately reversed: whichever name matches first sorts later.
        if key in y.lower():
            return -1
        if key in x.lower():
            return 1

    x_parts = tuple(int(a) if a.isdigit() else a for a in _DIGITS.split(x) if a)
    y_parts = tuple(int(a) if a.isdigit() else a for a in _DIGITS.split(y) if a)

    try:
        if x_parts < y_parts:
            return -1
        if x_parts > y_parts:
            return 1
        return 0
    except TypeError:
        # Mixed int/str at the same position, e.g. ('chr', 1) vs ('chr', 'a')
        return 1


def natsorted(names: Iterable[str]) -> list[str]:
    """
    Sort chromosome names in natural genomic order.

    The rules, in order:

    1. A name containing ``_`` (an unplaced/alt contig such as
       ``chr1_KI270706v1_random``) is compared on the segment *after* the first
       underscore. Underscored names always sort after plain ones.
    2. Names containing ``m``, ``y`` or ``x`` are pushed to the end, in that
       reversed priority, giving ... chrX, chrY, chrM.
    3. Otherwise, split into digit and non-digit runs and compare as a tuple, so
       ``chr2`` precedes ``chr10``.

    Rule 2 is a substring test over the whole lowercased name, so it is only
    correct for conventional ``chrN`` naming -- ``chromosome1`` would trip the
    ``m`` test. Rule 1 saves the common case, since ``_random`` contigs are
    compared on their accession segment instead.
    """
    return sorted(names, key=cmp_to_key(_natcmp))


@dataclass(frozen=True, slots=True)
class Canvas:
    """
    The genome-spanning uniform lattice at one zoom level.

    A zoom level is the natural unit of a grid: it defines a complete partition
    of ``[0, max_width)`` into bins of exactly ``binsize``, ignoring chromosome
    boundaries. A tile is a window of ``tile_size`` consecutive bins cut from
    it, which is why :meth:`tile` takes only ``x`` -- the lattice is already
    fixed by the zoom.

    Obtained from :meth:`TilesetInfo.canvas`, so the resolution-ladder logic
    stays in one place and callers never recompute ``max_width / 2**z``.

    ``binsize`` is integral for any tileset that actually bins. An implicit
    ladder is a quadtree -- ``max_width = tile_size * 2**max_zoom`` with a
    fixed ``tile_size``.

    ``max_width`` is the extent of *this zoom level*, which for an explicit
    ladder is not a tileset-wide constant. See :meth:`TilesetInfo.canvas`.
    """

    z: int
    binsize: float
    tile_size: int
    # Extent of the coordinate space this lattice spans.
    max_width: int
    # Needed to invert tile positions back into genomic intervals.
    chromsizes: Chromsizes | None = None

    @property
    def span(self) -> tuple[int, int]:
        """Absolute ``[start, end)`` the whole canvas covers."""
        return (0, self.max_width)

    @property
    def n_bins(self) -> int:
        """Bins across the full canvas."""
        return int(self.max_width / self.binsize)

    @property
    def n_tiles(self) -> int:
        """Tiles needed to cover the canvas at this zoom."""
        return math.ceil(self.n_bins / self.tile_size)

    def tile_span(self, x: int) -> tuple[int, int]:
        """Absolute ``[start, end)`` covered by tile ``x``."""
        width = self.binsize * self.tile_size
        return (int(x * width), int((x + 1) * width))

    def invert(self, x: int) -> Iterator[GenomicRange]:
        """Genomic intervals covered by tile ``x``."""
        if self.chromsizes is None:
            raise ValueError(
                "canvas has no chromsizes, so tile positions cannot be "
                "inverted to genomic intervals"
            )
        return self.chromsizes.invert(self.tile_span(x))

    def tile(self, x: int) -> TileGrid:
        """The window covering tile ``x`` at this zoom.

        Only needed by the SCATTER policy, which places data by absolute
        position and so needs tile-relative bin indices. SEQUENTIAL callers
        want :meth:`invert` instead.
        """
        return TileGrid(
            origin=self.tile_span(x)[0],
            binsize=self.binsize,
            tile_size=self.tile_size,
        )

    def bin_of(self, abs_pos) -> int | np.ndarray:
        """Canvas-global bin index containing ``abs_pos``.

        Global rather than tile-relative -- subtract ``x * tile_size`` for the
        latter, or use :meth:`TileGrid.bin_of`. Accepts arrays as well as
        scalars.
        """
        idx = np.asarray(abs_pos) // self.binsize
        return idx.astype(int) if idx.ndim else int(idx)

    def tile_of(self, abs_pos) -> int:
        """Which tile index contains ``abs_pos``."""
        return int(np.asarray(abs_pos) // (self.binsize * self.tile_size))


@dataclass(frozen=True, slots=True)
class TileGrid:
    """
    One tile's window onto a :class:`Canvas`.

    Bin ``i`` covers ``[origin + i * binsize, origin + (i + 1) * binsize)``.
    Exactly ``tile_size`` bins. Construct via :meth:`Canvas.tile` rather than
    directly, so the zoom-level arithmetic happens once.
    """

    # Absolute bp coordinate where this tile begins.
    origin: int
    binsize: float
    tile_size: int

    @property
    def span(self) -> tuple[int, int]:
        """Absolute ``[start, end)`` this tile covers."""
        return (self.origin, self.origin + int(self.binsize * self.tile_size))

    def bin_of(self, abs_pos):
        """Index *within this tile* of the bin containing ``abs_pos``.

        This is cooler's ``(genome_start1 - start1) // binsize``
        (``cooler.py:340``) -- the whole of the SCATTER policy.

        Accepts arrays as well as scalars, since cooler applies it to a column
        of pixel positions at once.
        """
        idx = (np.asarray(abs_pos) - self.origin) // self.binsize
        return idx.astype(int) if idx.ndim else int(idx)

    def bin_bounds(self, i: int) -> tuple[int, int]:
        """Absolute ``[start, end)`` covered by bin ``i``."""
        return (
            self.origin + int(i * self.binsize),
            self.origin + int((i + 1) * self.binsize),
        )


def reconcile_sequential(
    chunks: Iterable[tuple[Sequence, float]],
    binsize: float,
    *,
    expected_bins: int | None = None,
    threshold: float = 1.0,
) -> np.ndarray:
    """
    Concatenate ragged per-chromosome bins onto the uniform lattice.

    The SEQUENTIAL policy, factored out of ``bigwig.get_bigwig_tile:268`` and
    ``multivec.get_tile:112``. It is Bresenham's error accumulation with the sign
    flipped: carry the running over-representation, and once it reaches a whole
    bin, *drop* one bin rather than adding a step.

    Parameters
    ----------
    chunks
        ``(values, span_bp)`` per interval, in order. ``values`` is that
        interval's bins -- 1D, or 2D as ``(n_bins, n_dim)`` for bigwig's minMax
        and whisker modes. ``span_bp`` is the interval's true width, which is
        what the bins over-represent.
    binsize
        Nominal bp per bin on the uniform lattice.
    expected_bins
        If given, raise unless exactly this many bins come out.

        Deliberately an assertion rather than a clamp. multivec currently ends
        with ``np.concatenate(arrays)[: shape[0]]``, and that silent truncation
        is what hid its broken accumulator. Padding is not needed here: the
        out-of-genome interval is yielded by
        :meth:`Chromsizes.abs2genomic` and filled to the right length by the
        per-interval fetcher, so the count comes out right on its own.
    threshold
        When to spend accumulated drift, as a fraction of a bin. ``1.0`` floors
        (truncates; worst-case shift ~1 bin), ``0.5`` rounds to nearest (~0.5
        bin). See section 1.5b -- midpoint is a likely future default, but it
        changes which bins are dropped and therefore what renders.

    Returns
    -------
    np.ndarray
        The concatenated bins, ``expected_bins`` long when that is given.
    """
    chunks = list(chunks)
    spans = [span for _, span in chunks]
    counts = [len(values) for values, _ in chunks]
    drops = _bresenham_drop_indices(spans, counts, binsize, threshold)

    kept = []
    for i, (values, _) in enumerate(chunks):
        if counts[i] == 0:
            continue
        kept.append(values[:-1] if i in drops else values)

    out = np.concatenate(kept) if kept else np.asarray([])

    if expected_bins is not None and len(out) != expected_bins:
        raise ValueError(
            f"grid reconciliation produced {len(out)} bins, expected "
            f"{expected_bins}. The chromosome-aligned and genome-aligned grids "
            f"have not been reconciled correctly."
        )
    return out


def _bresenham_drop_indices(
    spans, bin_counts, binsize, threshold=1.0
) -> set[int]:
    """Which chunks lose their trailing bin.

    Split out from :func:`reconcile_sequential` so the policy can be tested and
    compared without fetching any data -- everything it needs is interval widths
    and bin counts. That is how the floor-vs-midpoint threshold comparison in
    section 1.5b was measured.

    ``bp_walked`` is how far along the tile's span we have actually travelled;
    ``bp_claimed`` is how far the bins emitted so far claim to reach, since the
    client treats every bin as exactly ``binsize`` wide. Their difference is the
    accumulated drift.
    """
    drops = set()
    bp_walked = bp_claimed = 0.0

    for i, (span, n) in enumerate(zip(spans, bin_counts)):
        bp_walked += span
        if n == 0:
            continue
        bp_claimed += binsize * n
        if bp_claimed - bp_walked >= binsize * threshold:
            bp_claimed -= binsize
            drops.add(i)

    return drops


def reconcile_sequential_2d(
    blocks: Sequence[Sequence[np.ndarray]],
    row_spans: Sequence[float],
    col_spans: Sequence[float],
    binsize: float,
    *,
    expected_shape: tuple[int, int] | None = None,
    threshold: float = 1.0,
) -> np.ndarray:
    """Assemble a 2D grid of blocks onto the canvas, correcting drift per axis.

    The 2D case is **separable**: the two axes over-represent independently, so
    this is :func:`reconcile_sequential` run once per axis. A block loses its
    last row if its row interval's drift matured, its last column if its column
    interval's did, and both if both. Nothing about the correction couples the
    axes -- which is why a 2D reconciler needs no new arithmetic, only new
    bookkeeping.

    Parameters
    ----------
    blocks
        ``blocks[r][c]`` is the 2D array for row interval ``r`` crossed with
        column interval ``c``, shaped ``(row_bins[r], col_bins[c])``. Every row
        must have the same number of columns.

        Which genomic axis is "row" is the caller's choice and is not inferred
        -- get it wrong and the tile comes out transposed. Blocks for intervals
        past the end of the genome are the caller's responsibility too: supply a
        NaN array of the right shape, exactly as the 1D path does, since that
        padding is what fills the trailing cells of low-zoom tiles.
    row_spans, col_spans
        True bp width of each interval, which is what the bins over-represent.
    expected_shape
        If given, raise unless the assembled tile is exactly this shape. An
        assertion rather than a clamp, for the reason given in
        :func:`reconcile_sequential`.

    Returns
    -------
    np.ndarray
        The assembled 2D tile.
    """
    if not blocks:
        return np.zeros((0, 0))

    row_counts = [blocks[r][0].shape[0] for r in range(len(blocks))]
    col_counts = [blocks[0][c].shape[1] for c in range(len(blocks[0]))]

    row_drops = _bresenham_drop_indices(
        row_spans, row_counts, binsize, threshold
    )
    col_drops = _bresenham_drop_indices(
        col_spans, col_counts, binsize, threshold
    )

    # Strips contributing no bins drop out entirely; np.block cannot assemble a
    # ragged grid, so they must go from both axes rather than being zero-width.
    keep_rows = [r for r in range(len(blocks)) if row_counts[r]]
    keep_cols = [c for c in range(len(blocks[0])) if col_counts[c]]

    grid = [
        [_trim(blocks[r][c], r in row_drops, c in col_drops) for c in keep_cols]
        for r in keep_rows
    ]

    out = np.block(grid) if grid else np.zeros((0, 0))

    if expected_shape is not None and out.shape != expected_shape:
        raise ValueError(
            f"grid reconciliation produced {out.shape}, expected "
            f"{expected_shape}. The chromosome-segmented and genome-spanning "
            f"grids have not been reconciled correctly."
        )
    return out


def _trim(block: np.ndarray, drop_row: bool, drop_col: bool) -> np.ndarray:
    """Shed a block's last row and/or column."""
    if drop_row:
        block = block[:-1, :]
    if drop_col:
        block = block[:, :-1]
    return block
