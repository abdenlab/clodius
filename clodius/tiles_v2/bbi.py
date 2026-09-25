from __future__ import annotations

import hashlib


import numpy as np
import pybigtools

from clodius.core.coords import Chromsizes, GenomicRange, natsorted
from clodius.core.policies import (
    LinkPolicy,
    TilePolicy,
    reconcile,
    stable_importance,
    take_most_important,
)
from clodius.core.tile import (
    Annotation2DRecord,
    AnnotationRecord,
    DenseTile,
    DenseTilePayload,
)
from clodius.core.tileid import ModifierSpec, TileId
from clodius.core.tileset import BaseTileset, TilesetInfo

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

# Chromosome ranges queried per tile when a genome has very many contigs.
# An implementation detail of this reader, not a policy: it bounds how many
# range queries one tile issues, not how many records it returns.
MAX_RANGES_PER_TILE = 128


class BBITileset(BaseTileset):
    """Shared plumbing: the file handle, the chromsizes and the info.

    Neither concrete tileset differs in how a BBI file is opened, what its
    coordinate space is, or how alternate chromosome orderings are selected --
    only in what it reads out of the file.
    """

    ndim = 1

    def __init__(
        self,
        path,
        chromsizes: Chromsizes | None = None,
        chromsizes_alts: dict[str, Chromsizes] | None = None,
        policy: TilePolicy | None = None,
        tile_size: int = TILE_SIZE,
    ):
        self._path = path
        self._file = None
        self._handle = None
        self.policy = policy or TilePolicy()
        self.tile_size = tile_size
        if chromsizes is not None:
            self._chromsizes = chromsizes
        else:
            chroms = self.file.chroms()
            names = natsorted(chroms.keys())
            self._chromsizes = Chromsizes(
                tuple(names), tuple(chroms[n] for n in names)
            )
        # Alternate pre-defined chromsizes orderings. The client can select
        # these per tile via `,cos:<uid>`.
        self._chromsizes_alts = chromsizes_alts or {}
        self._info = self._build_info(self._chromsizes)

    # --- resource lifetime --------------------------------------------------

    @property
    def file(self):
        if self._file is None:
            # pybigtools dispatches on the extension, so a bigInteract (or any
            # other bigBed flavour) has to be handed an open file instead.
            suffix = str(self._path).lower()
            if not suffix.endswith((".bw", ".bigwig", ".bb", ".bigbed")):
                self._handle = open(self._path, "rb")
                self._file = pybigtools.open(self._handle)
                return self._file
            # Accepts a path directly. The legacy bigBed module calls
            # `bbpath.seek(0)` on whatever it is handed, so `tiles(<str>, ...)`
            # raises AttributeError -- masked because the test that would catch
            # it is skipped as obsolete (test/tiles/bigbed_test.py:9).
            self._file = pybigtools.open(self._path)
        return self._file

    def close(self) -> None:
        self._file = None
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()
            self._handle = None

    # --- ProvidesChromsizes -------------------------------------------------

    def chromsizes(self) -> Chromsizes:
        return self._chromsizes

    # --- the protocol -------------------------------------------------------

    def info(self) -> TilesetInfo:
        return self._info

    # --- internals ----------------------------------------------------------

    def _serves_nothing(self) -> bool:
        """Whether the policy refuses every record before any I/O.

        The cap is also enforced downstream, in `take_most_important`, but by
        then the tile has been fetched and every record digested -- seconds of
        work on a zoom-0 tile, to serve an empty list.
        """
        cap = self.policy.max_records
        return cap is not None and cap <= 0

    def _info_for(self, chromsizes: Chromsizes) -> TilesetInfo:
        """Tileset info under a given set of chromsizes.

        The tileset's own chromsizes are the case by far the most tiles take,
        and the info for those was built once at construction. Rebuilding it
        runs a full quadtree validation and hands back an instance with a cold
        `coordinate_system` cache, so `canvas()` then rebuilds a whole
        `Chromsizes` -- half a millisecond per tile on a heavily scaffolded
        assembly, against under a microsecond for the cached info.
        """
        if chromsizes is self._chromsizes:
            return self._info
        return self._build_info(chromsizes)

    def _build_info(self, chromsizes: Chromsizes) -> TilesetInfo:
        info = TilesetInfo.quadtree(
            chromsizes, self.tile_size, ndim=self.ndim, **self._info_extras()
        )
        # Range padded out to the full quadtree extent, as legacy does. One
        # entry per axis: a 2D track reads the y extent from ``max_pos[1]``.
        #
        # Derived rather than assigned: TilesetInfo is frozen, and `max_width`
        # is computed inside `quadtree`, so there is nothing to pass in up
        # front. `with_` re-validates and drops the cache; `model_copy` does
        # neither.
        return info.with_(max_pos=[info.max_width] * self.ndim)

    def _info_extras(self) -> dict:
        """Type-specific info fields, if any."""
        return {}

    def _chromsizes_for(self, tid: TileId) -> Chromsizes:
        """The ordering this tile asked for, via ``,cos:<uid>``."""
        uid = tid.option("cos")
        if uid is not None and uid in self._chromsizes_alts:
            return self._chromsizes_alts[uid]
        return self._chromsizes


# --- signal -----------------------------------------------------------------


def fetch(f, interval: GenomicRange, binsize: int, stats: tuple[str, ...]):
    n_bins = int(np.ceil((interval.end - interval.start) / binsize))
    shape = (n_bins, len(stats)) if len(stats) > 1 else (n_bins,)

    if interval.is_out_of_bounds:
        return np.full(shape, np.nan)

    out = np.zeros(shape)
    args = (*interval.as_tuple(), n_bins)
    try:
        if len(stats) > 1:
            for k, stat in enumerate(stats):
                out[:, k] = f.values(*args, stat, fillna=0)
        else:
            out[:] = f.values(*args, stats[0], fillna=0)
    except Exception as exc:  # noqa: BLE001 -- matches current behavior
        if "No chromomsome with name" not in str(exc):
            raise
        out[:] = np.nan  # supported chromosome absent from the file, e.g. chrM
    return out


class BBISignalTileset(BBITileset):
    """A bigWig or bigBed served as a 1D vector tileset."""

    datatype = "vector"
    modifiers = ModifierSpec(
        values=frozenset(
            {"mean", "min", "max", "std", "sum", "minMax", "whisker"}
        ),
        default="mean",
        kind="display mode",
    )
    options = frozenset({"cos"})

    def _info_extras(self) -> dict:
        return {
            "aggregation_modes": [
                {"name": label, "value": key}
                for key, label in AGGREGATION_MODES.items()
            ],
            "range_modes": [
                {"name": label, "value": key}
                for key, (label, _) in RANGE_MODES.items()
            ],
        }

    def _tile(self, tid: TileId) -> DenseTilePayload:
        chromsizes = self._chromsizes_for(tid)
        info = self._info_for(chromsizes)

        # A range mode returns several values per bin instead of one, which is
        # what `size` on the payload announces.
        stats = (
            RANGE_MODES[tid.modifier][1]
            if tid.modifier in RANGE_MODES
            else (tid.modifier,)
        )

        canvas = info.canvas(tid.z)
        chunks = [
            (fetch(self.file, gr, canvas.binsize, stats), gr.end - gr.start)
            for gr in canvas.invert(tid.pos[0])
        ]

        values = reconcile(
            chunks, canvas.binsize, expected_bins=canvas.tile_size
        )
        return DenseTile(values, size=len(stats)).to_dict()


# --- annotations ------------------------------------------------------------


def downsample_ranges(
    ranges: list[GenomicRange], tid: TileId, n: int
) -> list[GenomicRange]:
    """Genomic ranges to query, sampled when a genome has very many contigs."""
    spans = np.array([gr.end - gr.start for gr in ranges], dtype=float)
    rng = np.random.default_rng(abs(hash((tid.pos[0], tid.z))) % (2**32))
    chosen = rng.choice(len(ranges), n, p=spans / spans.sum(), replace=False)
    return [ranges[i] for i in sorted(chosen)]


def fetch_records(f, gr: GenomicRange) -> list[tuple]:
    """Raw records overlapping a genomic range, chromosome name prepended."""
    if gr.is_out_of_bounds:
        return []
    chrom, start, end = gr.as_tuple()
    return [(chrom,) + record for record in f.records(chrom, start, end)]


def to_bedlike(record: tuple, chrom_offset: int) -> AnnotationRecord:
    """One raw record as the client's bedlike shape."""
    # `usedforsecurity=False` for the same reason `stable_importance` carries
    # it: a bucketing hash, not a security primitive. Without it this call
    # raises first on a FIPS-enforcing build, so the flag downstream never
    # gets the chance to help.
    uid = hashlib.md5(
        "".join(map(str, record)).encode("utf8"), usedforsecurity=False
    ).hexdigest()
    return {
        "uid": uid,
        "chrOffset": chrom_offset,
        "xStart": chrom_offset + record[1],
        "xEnd": chrom_offset + record[2],
        # Derived from the record's own digest rather than random.random().
        # Stable across requests and identical at every zoom, so thinning is
        # reproducible and coherent.
        "importance": stable_importance(uid),
        "fields": record,
    }


class BBIAnnotationTileset(BBITileset):
    """A bigBed or bigWig served as a 1D annotation tileset."""

    datatype = "bedlike"
    modifiers = None
    options = frozenset({"cos"})

    def _tile(self, tid: TileId) -> list[AnnotationRecord]:
        if self._serves_nothing():
            return []

        chromsizes = self._chromsizes_for(tid)
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
        # `take_most_important` owns what a cap of None means; restating it
        # here would leave two surfaces encoding one rule.
        return take_most_important(
            rows, self.policy.max_records, importance=lambda r: r["importance"]
        )


# --- interactions -------------------------------------------------------------
#
# A bigInteract is a bigBed with the `interact` schema (bed5+13). The first
# three columns are the *hull* of the interaction; the two anchors are in the
# custom fields. Indices below are into a record with the chromosome prepended,
# as `fetch_records` returns it.
#
# https://genome.ucsc.edu/goldenpath/help/interact.html

INTERACT_FIELDS = 18

VALUE = 5
SOURCE_CHROM, SOURCE_START, SOURCE_END = 8, 9, 10
TARGET_CHROM, TARGET_START, TARGET_END = 13, 14, 15


def to_interaction(
    record: tuple, offsets: dict[str, int]
) -> Annotation2DRecord | None:
    """One interact record as a 2D annotation, or None if it is unplaceable."""
    if len(record) < INTERACT_FIELDS:
        raise ValueError(
            f"expected an interact record of {INTERACT_FIELDS} fields, got "
            f"{len(record)}; this bigBed is not bed5+13"
        )

    x_offset = offsets.get(record[SOURCE_CHROM])
    y_offset = offsets.get(record[TARGET_CHROM])
    if x_offset is None or y_offset is None:
        return None

    uid = hashlib.md5(
        "".join(map(str, record)).encode("utf8"), usedforsecurity=False
    ).hexdigest()
    try:
        importance = float(record[VALUE])
    except (TypeError, ValueError):
        importance = stable_importance(uid)

    return {
        "uid": uid,
        "xStart": x_offset + int(record[SOURCE_START]),
        "xEnd": x_offset + int(record[SOURCE_END]),
        "yStart": y_offset + int(record[TARGET_START]),
        "yEnd": y_offset + int(record[TARGET_END]),
        "chrOffset": x_offset,
        "xChrOffset": x_offset,
        "yChrOffset": y_offset,
        "importance": importance,
        "fields": [str(f) for f in record],
    }


class BBIInteractionTileset(BBITileset):
    """A bigInteract served as paired intervals.

    Prefer :class:`BBIInteraction2DTileset` or
    :class:`BBIInteractionLinksTileset`, which differ in how a tile selects.
    """

    datatype = "2d-rectangle-domains"
    modifiers = None
    options = frozenset({"cos"})

    def _interactions(
        self, chromsizes: Chromsizes, canvas, x: int
    ) -> list[Annotation2DRecord]:
        """Every interaction whose hull meets tile column ``x``."""
        ranges = [gr for gr in canvas.invert(x) if not gr.is_out_of_bounds]
        records = []
        for gr in ranges:
            records.extend(fetch_records(self.file, gr))

        offsets = chromsizes.offsets
        rows = [to_interaction(r, offsets) for r in records]
        return [r for r in rows if r is not None]

    def _capped(
        self, rows: list[Annotation2DRecord]
    ) -> list[Annotation2DRecord]:
        return take_most_important(
            rows, self.policy.max_records, importance=lambda r: r["importance"]
        )


class BBIInteraction2DTileset(BBIInteractionTileset):
    """A bigInteract served as 2D rectangles."""

    ndim = 2

    def _tile(self, tid: TileId) -> list[Annotation2DRecord]:
        if self._serves_nothing():
            return []

        chromsizes = self._chromsizes_for(tid)
        canvas = self._info_for(chromsizes).canvas(tid.z)
        x, y = tid.pos

        # The query is by hull, which also catches interactions that merely
        # pass over the column, so both anchors are checked here.
        x_lo, x_hi = canvas.tile_span(x)
        y_lo, y_hi = canvas.tile_span(y)
        rows = [
            r
            for r in self._interactions(chromsizes, canvas, x)
            if r["xStart"] < x_hi
            and r["xEnd"] > x_lo
            and r["yStart"] < y_hi
            and r["yEnd"] > y_lo
        ]
        return self._capped(rows)


class BBIInteractionLinksTileset(BBIInteractionTileset):
    """A bigInteract served as 1D links, for an arc-style track."""

    ndim = 1
    datatype = "bedlike"

    def __init__(
        self, *args, link_policy: LinkPolicy = LinkPolicy.EITHER, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.link_policy = LinkPolicy(link_policy)

    def _tile(self, tid: TileId) -> list[Annotation2DRecord]:
        if self._serves_nothing():
            return []

        chromsizes = self._chromsizes_for(tid)
        canvas = self._info_for(chromsizes).canvas(tid.z)
        lo, hi = canvas.tile_span(tid.pos[0])

        def anchors_in_view(r):
            return (
                r["xStart"] < hi and r["xEnd"] > lo,
                r["yStart"] < hi and r["yEnd"] > lo,
            )

        rows = []
        for r in self._interactions(chromsizes, canvas, tid.pos[0]):
            x_in, y_in = anchors_in_view(r)
            match self.link_policy:
                case LinkPolicy.BOTH:
                    keep = x_in and y_in
                case LinkPolicy.EITHER:
                    keep = x_in or y_in
                case _:
                    keep = True
            if keep:
                rows.append(r)
        return self._capped(rows)


# --- Notes ------------------------------------------------------------------
#
# 1. Three defects in the legacy bigBed module are fixed here rather than
#    reproduced, because reproducing them would mean writing them on purpose:
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
