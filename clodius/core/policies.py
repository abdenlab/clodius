from __future__ import annotations
import hashlib
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Iterable, Sequence, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class TilePolicy:
    """Limits applied when serving tiles.

    ``None`` means "no limit". Frozen -- derive variants with :meth:`with_`.
    """

    # Refuse a tile spanning more than this many base pairs.
    # This should enforced if set but it is also advertised to the client as
    # ``max_tile_width`` in tileset_info, so that it can stop requesting.
    max_span: int | None = None

    # Cap on records returned per tile, after ranking by importance.
    max_records: int | None = 8192

    # Refuse to scan an unindexed file larger than this.
    max_scan_bytes: int = 20_000_000

    def with_(self, **changes) -> TilePolicy:
        """Return a copy with ``changes`` applied."""
        return replace(self, **changes)


class LinkPolicy(str, Enum):
    """Which links a 1D links tile returns."""

    # Both anchors overlap the tile span: interactions entirely in view.
    BOTH = "both"

    # At least one anchor overlaps the tile span: interactions touching the region.
    EITHER = "either"

    # The hull [min(start), max(end)) overlaps the tile span: this includes
    # links that cross over the range with both anchors outside it.
    HULL = "hull"


def reconcile(
    chunks: Iterable[tuple[Sequence, float]],
    binsize: float,
    *,
    expected_bins: int | None = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Reconcile consecutive binned queries with a concatenated genome canvas.

    Needed for queries that span chromosome boundaries, because retrieved
    source data bin within chromosome bounds, where trailing bins may be
    "shorter" than the nominal bin size, while the target coordinate system
    bins a concatenated genome without respecting chromosome boundaries.

    Returns a concatenation of the chunks provided with a subset of trailing
    bins removed to minimize the cumulative drift of the source binned signal
    from the target coordinate system. The selection of trailing bins to drop
    is done using Bresenham's line drawing algorithm.

    Parameters
    ----------
    chunks
        ``(values, span_bp)`` per interval, in order. ``values`` is 1D binned
        raster or 2D (n_bins, n_tracks) for a multiplex one (e.g. min & max).
        ``span_bp`` is the corresponding interval's true width in bp.
    binsize
        Nominal bp per bin on the concatenated genome lattice.
    expected_bins: int, optional
        If given, raise unless exactly this many bins come out.
    threshold : float, optional
        Accumulated error per span, as a fraction of a bin, at which to drop a
        trailing bin.

    Returns
    -------
    np.ndarray
        The reconciled concatenated signal.

    Notes
    -----
    A threshold of ``1.0`` means a drop occurs once a full bin of drift has
    built up, so cumulative error stays within ``[0, binsize)``.

    A threshold of ``0.5`` means a drop occurs as soon as the error reaches
    half a bin, so the error stays in ``[-binsize/2, binsize/2)``.
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


def reconcile_2d(
    blocks: Sequence[Sequence[np.ndarray]],
    row_spans: Sequence[float],
    col_spans: Sequence[float],
    binsize: float,
    *,
    expected_shape: tuple[int, int] | None = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Reconcile a grid of 2D binned queries with a concatenated genome canvas.

    Needed for queries that span chromosome boundaries, because retrieved
    source data bin within chromosome bounds, where trailing bins may be
    "shorter" than the nominal bin size, while the target coordinate system
    bins a concatenated genome without respecting chromosome boundaries.

    Returns an assembled 2D tile with a subset of trailing bin rows/cols
    removed to minimize the cumulative drift of the source binned signal
    from the target 2D coordinate system.

    Parameters
    ----------
    blocks
        ``blocks[r][c]`` is the 2D array for row interval ``r`` crossed with
        column interval ``c``, shaped ``(row_bins[r], col_bins[c])``. Every row
        must have the same number of columns.
    row_spans, col_spans
        True bp width of each interval.
    binsize
        Nominal bp per bin on the concatenated genome lattice.
    expected_shape: tuple[int, int], optional
        If given, raise unless the assembled tile is exactly this shape.
    threshold : float, optional
        Accumulated error per span, as a fraction of a bin, at which to drop a
        trailing bin.

    Returns
    -------
    np.ndarray
        The assembled 2D tile.

    Notes
    -----
    The two axes over-represent independently, so we apply the Bresenham
    algorithm once per axis. A block loses its last row if its row drift
    matures and its last column if its column drift does.

    Which genomic axis is "row" is the caller's choice and is not inferred
    -- get it wrong and the tile comes out transposed. Blocks for intervals
    past the end of the genome are the caller's responsibility too: supply a
    NaN array of the right shape, exactly as the 1D path does, since that
    padding is what fills the trailing cells of low-zoom tiles.
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


def _bresenham_drop_indices(spans, bin_counts, binsize, threshold) -> set[int]:
    """
    Determine which consecutive spans lose their trailing bin to minimize drift.

    Parameters
    ----------
    spans : Sequence[float]
        True bp width of each span, in order.
    bin_counts : Sequence[int]
        Number of bins emitted for each span. Spans with zero bins are skipped
        and can never be dropped.
    binsize : float
        Nominal bp per bin on the uniform lattice. Each emitted bin claims this
        much width regardless of the span's true width.
    threshold : float, optional
        Accumulated error per span, as a fraction of a bin, at which to drop a
        trailing bin.

    Returns
    -------
    set[int]
        Indices of the spans whose trailing bin should be dropped.
    """
    # bp_walked: how far along the tile's span we have actually travelled
    # bp_claimed: how far the bins emitted so far claim to reach, since the
    # client treats every bin as exactly `binsize` wide
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


def _trim(block: np.ndarray, drop_row: bool, drop_col: bool) -> np.ndarray:
    """Shed a block's last row and/or column."""
    if drop_row:
        block = block[:-1, :]
    if drop_col:
        block = block[:, :-1]
    return block


def stable_importance(key: str) -> float:
    """A deterministic pseudo-importance in ``[0, 1)``, derived from content.

    A stand-in for a real precomputed ranking, for formats that have none. It is
    still arbitrary -- but *stably* arbitrary, which is the part that matters:

    - the same feature gets the same value on every request, so tiles are
      reproducible and cacheable;
    - the ranking is identical at every zoom, so a feature that survives
      thinning at one level survives at the next, instead of flickering.

    Takes a second digest, of the key it is handed. Callers that pass a uid
    which is itself a hexdigest therefore hash twice; slicing the caller's
    digest instead would be cheaper but would change every importance value,
    and so which records survive thinning.
    """
    # `usedforsecurity=False` marks this as a bucketing hash rather than a
    # security primitive, so it keeps working on a FIPS-enforcing build where
    # md5 is otherwise refused.
    digest = hashlib.md5(key.encode("utf8"), usedforsecurity=False).hexdigest()
    return int(digest[:8], 16) / 0x1_0000_0000


def take_most_important(
    records: Sequence[T],
    cap: int | None,
    importance: Callable[[T], float],
) -> list[T]:
    """The ``cap`` most important records, in their original order.

    ``cap`` of ``None`` means no limit, matching :class:`TilePolicy`. A ``cap``
    of zero or less returns nothing: ``ranked[-0:]`` is the whole list, so
    without this guard a server configured to serve no records would emit an
    unbounded tile -- the precise failure the cap exists to prevent.

    Deterministic, unlike ``random.choices``, which additionally samples *with
    replacement* and so can return the same record twice while dropping another
    entirely.

    Order is preserved rather than sorted by importance: the client positions
    records by coordinate, and keeping genomic order makes the output easier to
    diff against the unthinned set.
    """
    if cap is None:
        return list(records)
    if cap <= 0:
        return []
    if len(records) <= cap:
        return list(records)

    ranked = sorted(range(len(records)), key=lambda i: importance(records[i]))
    keep = set(ranked[-cap:])
    return [r for i, r in enumerate(records) if i in keep]
