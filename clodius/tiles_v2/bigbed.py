"""BigBed rewritten against clodius.core.

The first *record* prototype: it returns a variable-length list of features, not
a grid of values. So none of `Canvas`'s bin machinery applies -- only
`invert(x)`, to turn a tile position into genomic ranges to query.

The legacy module does not really have a tileset_info of its own. It is::

    ti = hgbw.tileset_info(bbpath, chromsizes)   # bigwig's, wholesale
    ti["range_modes"] = range_modes              # patch one field

so it inherits bigwig's quadtree *and* bigwig's `aggregation_modes`
(mean/min/max/std/sum) -- binning statistics, advertised by a tileset that never
bins. This builds its own instead. The quadtree geometry is kept, because
changing `max_width` would move every tile boundary; the spurious
`aggregation_modes` is dropped, and `range_modes` is emitted as a *list* to match
bigwig rather than the bare dict the legacy patch leaves behind.

Density is `SUBSAMPLED`: no aggregation pass exists for bigBed, so a tile fetches
everything overlapping its range and thins to a cap.
"""

from __future__ import annotations
import hashlib

import numpy as np
import pybigtools

from clodius.core.coords import Chromsizes, GenomicRange, natsorted
from clodius.core.policies import (
    DensityPolicy,
    stable_importance,
    take_most_important,
)
from clodius.core.payloads import BedlikeTile, TileKind
from clodius.core.policies import DEFAULT_POLICY, TilePolicy
from clodius.core.tileid import ModifierSpec, TileId
from clodius.core.tileset import BaseTileset, TilesetInfo, quadtree_depth

TILE_SIZE = 1024

# The legacy module accepts this and then never reads it: `range_mode` is a
# parameter of `get_bigbed_tile` that appears nowhere in its body. Kept so the
# modifier still parses and the dropdown still renders.
RANGE_MODES = {"significant": "Significant"}

BIGBED_RANGE = ModifierSpec(
    values=frozenset(RANGE_MODES),
    default="significant",
    kind="range mode",
)

# Chromosome ranges sampled per tile when a genome has very many contigs.
MAX_RANGES_PER_TILE = 128
# Records returned per tile, overridable per request via `,max:N`.
DEFAULT_MAX_RECORDS = 100


def max_records(tid: TileId) -> int:
    """`,max:N` from the tile id, else the default."""
    raw = tid.option("max")
    if raw is None:
        return DEFAULT_MAX_RECORDS
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_RECORDS


def downsample_ranges(
    ranges: list[GenomicRange], tid: TileId, n: int
) -> list[GenomicRange]:
    """Genomic ranges to query, sampled when a genome has very many contigs."""
    spans = np.array([gr.end - gr.start for gr in ranges], dtype=float)
    rng = np.random.default_rng(abs(hash((tid.pos[0], tid.z))) % (2**32))
    chosen = rng.choice(len(ranges), n, p=spans / spans.sum(), replace=False)
    return [ranges[i] for i in sorted(chosen)]


def fetch_records(f, gr: GenomicRange) -> list[tuple]:
    """Raw bigBed records overlapping a genomic range, chromosome name prepended."""
    if gr.is_out_of_bounds:
        return []
    chrom, start, end = gr.as_tuple()
    return [(chrom,) + record for record in f.records(chrom, start, end)]


def to_bedlike(record: tuple, chrom_offset: int) -> BedlikeTile:
    """One raw record as the client's bedlike shape."""
    uid = hashlib.md5("".join(map(str, record)).encode("utf8")).hexdigest()
    return {
        "uid": uid,
        "chrOffset": chrom_offset,
        "xStart": chrom_offset + record[1],
        "xEnd": chrom_offset + record[2],
        # Derived from the record's own digest rather than random.random().
        # Stable across requests and identical at every zoom, so thinning is
        # reproducible and coherent -- see clodius.core.density.
        "importance": stable_importance(uid),
        "fields": record,
    }


class BigBedTileset(BaseTileset):
    """A bigBed served as a 1D annotation tileset."""

    datatype = "bedlike"
    ndim = 1
    tile_kind = TileKind.BEDLIKE
    modifiers = BIGBED_RANGE
    options = frozenset({"cos", "min", "max"})

    density_policy = DensityPolicy.SUBSAMPLED

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
        self._chromsizes_alts = chromsizes_alts or {}
        self._info = self._info_for(self._chromsizes)

    @property
    def file(self):
        if self._file is None:
            # Accepts a path directly. The legacy module calls `bbpath.seek(0)`
            # on whatever it is handed, so `tiles(<str>, ...)` raises
            # AttributeError -- masked because the test that would catch it is
            # skipped as obsolete (test/tiles/bigbed_test.py:9).
            self._file = pybigtools.open(self._path)
        return self._file

    def close(self) -> None:
        self._file = None

    def chromsizes(self) -> Chromsizes:
        return self._chromsizes

    @property
    def policy(self) -> TilePolicy:
        return self._policy

    def info(self) -> TilesetInfo:
        return self._info

    def tiles(self, ids):
        return [(tid, self._tile(tid)) for tid in ids]

    def _info_for(self, chromsizes: Chromsizes) -> TilesetInfo:
        max_zoom = quadtree_depth(chromsizes.total_length, TILE_SIZE)
        max_width = TILE_SIZE * 2**max_zoom
        return TilesetInfo(
            min_pos=[0],
            max_pos=[max_width],
            max_width=max_width,
            tile_size=TILE_SIZE,
            max_zoom=max_zoom,
            chromsizes=chromsizes.to_pairs(),
            # A list, matching bigwig. The legacy patch assigns the raw dict, so
            # `range_modes` arrives as an object from bigbed and an array from
            # bigwig -- same field, two JSON types.
            range_modes=[
                {"name": label, "value": key}
                for key, label in RANGE_MODES.items()
            ],
        )

    def _tile(self, tid: TileId) -> list[BedlikeTile]:
        uid = tid.option("cos")
        if uid is not None and uid in self._chromsizes_alts:
            chromsizes = self._chromsizes_alts[uid]
        else:
            chromsizes = self._chromsizes

        canvas = self._info_for(chromsizes).canvas(tid.z)

        ranges = [
            gr for gr in canvas.invert(tid.pos[0]) if not gr.is_out_of_bounds
        ]
        if len(ranges) > MAX_RANGES_PER_TILE:
            ranges = downsample_ranges(ranges, tid, MAX_RANGES_PER_TILE)

        records = []
        for gr in ranges:
            records.extend(fetch_records(self.file, gr))

        offsets = chromsizes.offsets
        rows = [to_bedlike(r, offsets[r[0]]) for r in records]
        return take_most_important(
            rows, max_records(tid), importance=lambda r: r["importance"]
        )


# --- Notes ------------------------------------------------------------------
#
# 1. Three defects in the legacy module are fixed here rather than reproduced,
#    because reproducing them would mean writing them on purpose:
#      - `bbpath.seek(0)` on a str, so `tiles(<path>, ...)` cannot run at all
#      - `nr.choice(..., replace=128)` samples contigs WITH replacement
#      - `random.choices(intervals, k=100)` samples records WITH replacement,
#        so a tile can contain the same feature twice while dropping another
#    None are reachable by the byte-comparison used for the dense prototypes,
#    since the legacy output is not reproducible run to run.
#
# 2. `importance` is derived from the record digest instead of `random.random()`.
#    Still arbitrary, but stable: identical across requests (cacheable) and
#    identical across zoom levels (a feature does not flicker in and out as you
#    zoom). This is the cheap half of the repair; the real fix is a
#    precomputation pass, which is a tile-generation concern.
#
# 3. Two wire changes, both deliberate:
#      - `aggregation_modes` dropped. Inherited from bigwig by accident; a
#        bigBed cannot compute a mean.
#      - `range_modes` emitted as a list rather than a dict, matching bigwig.
#    The quadtree geometry is deliberately NOT changed -- `max_width` sets tile
#    boundaries, so touching it would move every tile.
