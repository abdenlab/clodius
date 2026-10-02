from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
# Load-bearing beyond the array maths, and nothing enforces it: `BBITileset.
# reading` exempts a path-backed reader from the read lock, and the measured
# precondition for that exemption is that `numpy` has been imported in this
# process. Without it, eight threads on one path-backed reader refuse 2800 of
# 3200 queries as `Already borrowed`; with it, 0 of 3200. See `reading` for
# the measurement and for what is deliberately not claimed about why.
import numpy as np
import pybigtools

from clodius.core.coords import Chromsizes, GenomicRange, natsorted
from clodius.core.errors import TileError
from clodius.core.policies import (
    LinkPolicy,
    TilePolicy,
    reconcile,
    stable_importance,
    take_most_important,
)
from clodius.core.source import BinaryHandle, SourceLike
from clodius.core.tile import (
    Annotation2DRecord,
    AnnotationRecord,
    DenseTile,
    DenseTilePayload,
)
from clodius.core.tileid import ModifierSpec, TileId
from clodius.core.tileset import TilesetInfo
from clodius.tiles_v2._backed import FileBacked

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

# The suffixes `pybigtools.open` dispatches on, in the exact spellings it
# accepts. Case-sensitive, and deliberately not case-folded: `pybigtools`
# refuses `signal.BW` as an invalid file type, so folding the comparison sends
# a perfectly good bigWig down the path branch to be rejected.
#
# Recognition is not free, and it is worth being exact about which direction
# costs what. A path ending in anything else is handed over as a live handle,
# which opens any BBI file whatever it is named -- so *failing* to recognize a
# suffix costs a different open, not a slower one. Recognizing one commits to
# the flavour it names: `pybigtools.open(<path>)` picks the parser from the
# extension while `pybigtools.open(<handle>)` sniffs the bytes, so a bigWig
# named `.bb` is refused by the path branch and served fine by the handle
# branch. Being unrecognized is therefore strictly safer than being
# mis-recognized, and a misnamed file is the one case this fast path cannot
# serve.
_PYBIGTOOLS_PATH_SUFFIXES = (
    ".bw",
    ".bigwig",
    ".bigWig",
    ".bb",
    ".bigbed",
    ".bigBed",
)

# What `pybigtools` raises when two queries overlap on one reader: it holds a
# borrow on the ``BBIReader`` -- a PyO3 `RefCell` -- for the duration of a
# query. On the reader, not on the Python handle, which is why a path-backed
# reader can produce it too and why `BBITileset.reading`'s exemption for that
# route rests on a measured property of the process rather than on nothing
# being shared.
#
# `reading` serializes the route that produces it in practice, so this is the
# belt to that braces -- a residual case must reach a client as that tile's
# refusal rather than as a 500, and `RuntimeError` is not a `TileError`, so
# `BaseTileset.tiles` would otherwise let it through the per-tile boundary and
# abort the whole batch.
_BORROW_ERROR = "Already borrowed"


class BBITileset(FileBacked[pybigtools.BBIReader]):
    """Shared plumbing: the reader, the chromsizes and the info.

    Neither concrete tileset differs in how a BBI file is opened, what its
    coordinate space is, or how alternate chromosome orderings are selected --
    only in what it reads out of the file.

    **A factory-backed tileset's reads do not scale with a worker pool.** One
    tileset holds one reader and `reading` serializes every read through it,
    so the lock is the whole of that tileset's read concurrency. Measured
    flat from 1 to 8 threads -- 177-213 queries/s and 632-648 tiles/s -- with
    4.2-7.8x recoverable by giving each thread its own reader. Throughput is
    the metric that shows it; per-call latency does not, because `RLock`
    barging lets one thread run uncontended while the others starve. A
    path-backed tileset is not affected: `reading` does not lock that route.

    Lifting the ceiling needs a bounded pool of readers rather than one,
    which is filed against the oxbow epic rather than solved here. The lock
    is the correct minimal fix in the meantime -- removing it refuses tiles
    rather than slowing them.

    Parameters
    ----------
    source : str, bytes, os.PathLike, Source, or callable
        Where the file's bytes come from: a filesystem path, or a
        zero-argument callable returning a freshly opened, seekable binary
        handle each time it is called (``lambda: fs.open(url, "rb")`` and
        ``lambda: open(p, "rb")`` both qualify). An already-open file is
        refused, because the obvious repair -- wrapping it as
        ``lambda: handle`` -- returns the same exhausted handle on every call.
        A path whose suffix ``pybigtools`` recognizes is handed over unchanged;
        any other source is opened here, and :meth:`close` closes what it
        opened.
    chromsizes : Chromsizes, optional
        The coordinate system to serve against. Read from the file's own
        header when omitted, which costs one open at construction.
    chromsizes_alts : dict of str to Chromsizes, optional
        Alternate orderings a tile may select with ``,cos:<uid>``. Each one's
        info is built at construction, so they are fixed configuration rather
        than something a request can grow.
    policy : TilePolicy, optional
        Limits applied when serving. Defaults to `TilePolicy`'s own defaults.
    tile_size : int, optional [default: 1024]
        Bins per tile.
    """

    ndim = 1

    def __init__(
        self,
        source: SourceLike,
        chromsizes: Chromsizes | None = None,
        chromsizes_alts: dict[str, Chromsizes] | None = None,
        policy: TilePolicy | None = None,
        tile_size: int = TILE_SIZE,
    ):
        super().__init__(source)
        # Serializes reads through a handle this tileset owns. Separate from
        # the base's `_lock`, which guards open and close: one lock for both
        # would mean holding it across a read, so `_opened`'s slow path and
        # every query would contend on one lock and `close` would wait behind
        # all of them rather than on the open it actually races.
        #
        # `close` is overridden to take this one as well, deliberately -- see
        # there. Acquiring it waits on the read that holds it; merging the two
        # locks would have made every read and every open one queue.
        #
        # Reentrant. `reading()` is published, so a caller may nest it --
        # hold the reader across a helper, compare two ranges -- and a plain
        # `Lock` made that a permanent hang with no exception and no timeout,
        # for a factory-backed source only. Reentrancy costs nothing here:
        # both read helpers return before the block ends, so a nested acquire
        # on one thread cannot produce two overlapping queries.
        #
        # Allocated here rather than on the base because `pybigtools`' borrow
        # is BBI's constraint; HDF5 serializes its own calls and never takes
        # this.
        self._read_lock = threading.RLock()
        with self._configuring():
            self.policy = policy or TilePolicy()
            self.tile_size = tile_size
            if chromsizes is not None:
                self._chromsizes = chromsizes
            else:
                chroms = self._opened().chroms()
                names = natsorted(chroms.keys())
                self._chromsizes = Chromsizes(
                    tuple(names), tuple(chroms[n] for n in names)
                )
            # Alternate pre-defined chromsizes orderings. The client can
            # select these per tile via `,cos:<uid>`.
            self._chromsizes_alts = chromsizes_alts or {}
            self._info = self._build_info(self._chromsizes)
            # Built eagerly, for the same reason `self._info` is: rebuilding
            # runs a full quadtree validation and hands back a cold
            # `coordinate_system`, which `canvas()` then pays to rebuild. The
            # alternates are fixed at construction, so this memo is bounded by
            # configuration and cannot grow with request volume.
            self._info_alts = {
                uid: self._build_info(cs)
                for uid, cs in self._chromsizes_alts.items()
            }

    # --- ProvidesChromsizes -------------------------------------------------

    def chromsizes(self) -> Chromsizes:
        return self._chromsizes

    # --- the protocol -------------------------------------------------------

    @contextmanager
    def reading(self) -> Iterator[pybigtools.BBIReader]:
        """The reader, serialized when this tileset owns the handle.

        `pybigtools` holds a borrow on a Python handle for the duration of a
        query, so two threads reading through one handle-backed reader raise
        `RuntimeError: Already borrowed` -- and a tile server answers requests
        on a thread pool, so the source shape this module exists to support is
        the one that breaks.

        Held around a single reader call rather than a whole batch. The borrow
        lasts one query, and a batch is mostly work that never touches the
        reader -- a digest per record, a sort, a base64 encode -- so a
        batch-wide lock put every concurrent request behind all of it.

        **Every read a block starts must also finish inside the block.** The
        borrow is released when the query *returns*, which is what makes one
        call the right scope -- but ``records()`` hands back a lazy
        ``BigBedEntriesIterator``, and *advancing* one is a read. So the
        eagerness of `fetch` and `fetch_records` is load-bearing here, not
        incidental: `fetch` writes into a preallocated array, and
        `fetch_records` materializes its list before the block ends.

        Measured on a handle-backed reader, five threads x twenty queries,
        taking the iterator inside the block and draining it after: **76 of
        100 queries died** with ``pyo3_runtime.PanicException: ... BadData``,
        and of the 24 that returned, every one returned a *truncated* list --
        72, 172, 196 and 372 records against a truth of 900. The truncation
        raises nothing, so it reaches a client as a well-formed wrong tile,
        and the panic is not an `Exception`, so no boundary in this package
        catches it. Draining inside the block, same workload: 0 of 100
        errors, every count correct. Streaming records instead of
        accumulating them is the obvious next optimization in
        `BBIAnnotationTileset._tile`, and it is the one change this contract
        forbids.

        **This serializes a factory-backed tileset's reads process-wide** --
        see the ceiling in `BBITileset`'s own docstring, which is a real
        limit for a latency-bound source and is filed against the oxbow epic.

        A path-backed reader is exempt, and the honest reason is narrower
        than the structural one this docstring used to give. The borrow is on
        the ``BBIReader`` object -- a PyO3 ``RefCell`` -- and not on the
        Python handle, so there is a borrow to contend on *either* route: the
        same eight-thread workload against a path-backed reader in an
        interpreter where `numpy` has not been imported refuses **2800 of
        3200** queries as `Already borrowed`. What was measured instead is
        that in this process the overlap does not occur: with `numpy`
        imported, a thread asking to wake 90 ms into a 165 ms path-backed
        `values()` got the interpreter back only as that query returned
        (3 of 3 runs; earliest 2 ms before it returned), and the same
        workload refuses 0 of 3200.

        So the exemption rests on a precondition of the process rather than a
        property of the library, `bbi.py` happens to satisfy it at module
        scope, and nothing enforces it -- see the note at ``import numpy``.
        No mechanism linking the two measurements is asserted here. The two
        routes cost the same per query; what the path route does not pay is
        this lock.

        The test is `_needs_handle()` and NOT whether a handle is currently
        published. `_handle` is `None` until a first open *completes*, and
        construction hands back what the chromsizes read opened, so a freshly
        constructed tileset is cold -- keying on the published handle let
        every thread of a cold tileset's first batch take the unlocked
        branch, share the one reader `_opened` then publishes, and race.
        `_needs_handle()` is a property of the source, so it answers before
        anything is open and does not change under us.

        The reader is resolved before the lock is taken, not inside it. That
        is scope hygiene and nothing more: `_opened()` takes the base's
        `_lock` and double-checks, so exactly one thread performs the open
        and the rest block on that lock either way. Hoisting it changes which
        lock they park on, not how long -- a cold eight-thread first batch
        against a factory sleeping 50-100 ms per open measured the same
        inside and outside, crossing in both directions.

        Entering this costs a generator per genomic range, measured at
        0.41-0.52 us and accepted: a tile spanning many contigs issues one
        per range, which is nothing beside the query it guards. Recorded
        because it has now been independently rediscovered four times.
        """
        reader = self._opened()
        if not self._needs_handle():
            yield reader
            return
        with self._read_lock:
            yield reader

    def close(self) -> None:
        """Release the reader, waiting out a read already in flight.

        The base deliberately does not hold the read lock, so without this a
        `close` landing mid-query called ``BBIReader.close()`` while
        `pybigtools` held its borrow -- and that raises. Measured on a
        factory-backed tileset whose handle costs 10 ms a read, closing 50 ms
        into one request, 5 of 5 trials: `close` raised
        `RuntimeError: Already borrowed`, which is not a `TilesetError` and
        is what `BaseTileset.__exit__` would have propagated out of a ``with``
        block, and the in-flight request died with ``read of closed file``.
        With the lock taken here, the same 5 of 5: `close` returns and the
        request is *served*, for a wait of 37-46 ms -- the query it waited
        for. Under ordinary local load, 8 threads on ~1 ms queries, the wait
        went from a median of 0.11 ms to 0.15 ms and the requests that saw a
        closed file fell from 49 to 8 across 12 trials.

        This waits on the read lock, which is not the same as waiting on
        every in-flight query -- the reason the two locks are separate in the
        first place. Reads are already serialized through `_read_lock`, so
        one query holds it; the rest are queued on it, and `RLock` does not
        promise this call a place at the front of that queue. The bound is
        the queue, and what was measured is the query.

        It does not close the window entirely: a request that resolved the
        reader before this call took the lock still holds the reference, and
        reading through a reference a caller already has is the race
        `_opened` disclaims rather than one this can fix. The 8 residual
        errors above are that case.
        """
        with self._read_lock:
            super().close()

    def info(self) -> TilesetInfo:
        return self._info

    # --- internals ----------------------------------------------------------

    def _needs_handle(self) -> bool:
        # `pybigtools.open` dispatches on the extension, so a bigInteract (or
        # any other bigBed flavour) has to be handed an open file instead. A
        # factory-backed source has no path to dispatch on at all.
        path = self._src.path
        return path is None or not path.endswith(
            _PYBIGTOOLS_PATH_SUFFIXES
        )

    def _reader_open(self, target: str | BinaryHandle) -> pybigtools.BBIReader:
        # Takes a path directly, where the suffix allows one. The legacy
        # bigBed module calls `bbpath.seek(0)` on whatever it is handed, so
        # `tiles(<path>, ...)` raises AttributeError there -- masked because
        # the test that would catch it is skipped as obsolete
        # (test/tiles/bigbed_test.py:9).
        return pybigtools.open(target)

    def _validate(self, reader: pybigtools.BBIReader) -> None:
        """Read the header, so a file that opens but cannot be read is refused.

        The hook `FileBacked._validate` exists for, which BBI had left as the
        base's no-op: `_open` calls this with the reader open and nothing
        published, so a file that `pybigtools.open` accepts but whose header
        does not parse is rejected with nothing cached -- rather than cached
        as a reader whose every later use fails differently from its first,
        which is the bug the publication order was designed around.

        One `chroms()` per *open*, not per construction, measured at 1.3 us.

        A second motivation was offered for it and is recorded here as not
        reproduced, because it is the kind of claim this module has been
        wrong about before. A reviewer measured a rare race (1 trial in 120
        to 360) attributed to `pybigtools` taking an *exclusive* borrow on a
        reader's first call, which this call would absorb under `_lock`. Four
        attempts to reproduce it found nothing: 1,360 trials of an
        eight-thread cold first batch through the real tileset, 0 refusals;
        and a direct probe says there is no window to race, because a
        path-backed `values()` does not yield the interpreter until it
        returns (see `reading`). That probe ran against a warm page cache, so
        it does not rule out a window behind real I/O. The call stands on the
        refusal above, which needs no race to justify it.
        """
        reader.chroms()

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

        Every ordering this tileset can be asked for -- its own and each
        `,cos:<uid>` alternate -- had its info built once at construction.
        Rebuilding runs a full quadtree validation and hands back an instance
        with a cold `coordinate_system` cache, so `canvas()` then rebuilds a
        whole `Chromsizes` -- half a millisecond per tile on a heavily
        scaffolded assembly, against under a microsecond for the cached info.
        """
        if chromsizes is self._chromsizes:
            return self._info
        for uid, alt in self._chromsizes_alts.items():
            if chromsizes is alt:
                return self._info_alts[uid]
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


@contextmanager
def _borrow_as_tile_error() -> Iterator[None]:
    """Translate a `pybigtools` borrow error into a `TileError`.

    Written once because the two read helpers had drifted apart: one caught
    `Exception` and one `RuntimeError`, and each named a flavour in its
    message that was wrong whenever a tileset served the other one --
    `BBISignalTileset` and `BBIAnnotationTileset` both accept a bigWig or a
    bigBed, so the message reached a client naming a format its file is not.

    `RuntimeError` is the narrow catch, which is what PyO3 raises. A
    `TileError` stops at `BaseTileset.tiles`' per-tile boundary; a
    `RuntimeError` does not, and takes the whole batch down as a 500.

    A context manager rather than a function the caller re-raises after: a
    translation split across two statements at the call site can be
    half-applied, and dropping the re-raise swallows every ``RuntimeError``
    that is not a borrow, silently. One ``with`` cannot be.

    Above the section banners deliberately: it serves `fetch` in the signal
    section and `fetch_records` in the annotation one, so filing it under
    either is what let the two drift apart in the first place.
    """
    try:
        yield
    except RuntimeError as exc:
        if _BORROW_ERROR in str(exc):
            raise TileError(
                "this BBI file is being read concurrently; retry the tile"
            ) from exc
        raise


def _bin_shape(
    interval: GenomicRange, binsize: int, stats: tuple[str, ...]
) -> tuple[int, ...]:
    """The shape a range's values occupy in a tile.

    One home, because the pad and the data have to agree: `reconcile` is given
    ``expected_bins`` and refuses a grid whose bin count is short, so a
    one-sided edit to either derivation is a `ValueError` out of the tile
    loop rather than a cosmetic drift. A range mode contributes a second axis,
    which is what `size` on the payload announces.
    """
    n_bins = int(np.ceil((interval.end - interval.start) / binsize))
    return (n_bins, len(stats)) if len(stats) > 1 else (n_bins,)


def _empty_bins(
    interval: GenomicRange, binsize: int, stats: tuple[str, ...]
):
    """The NaN pad a range past the end of the genome contributes to a tile.

    Built by `BBISignalTileset._tile` rather than by `fetch`, so a range that
    never touches the reader does not enter the read lock to do nothing. The
    pad is not optional: `reconcile` is given ``expected_bins`` and refuses a
    short grid, and the padding is what fills the trailing cells of a
    low-zoom tile.
    """
    return np.full(_bin_shape(interval, binsize, stats), np.nan)


# --- signal -----------------------------------------------------------------


def fetch(f, interval: GenomicRange, binsize: int, stats: tuple[str, ...]):
    """Binned values for one in-bounds range.

    ``interval`` must be in bounds. A range past the end of the genome
    contributes `_empty_bins` and no I/O, and `BBISignalTileset._tile` -- the
    only caller -- decides which of the two a range gets before it takes the
    read lock. This used to re-test it and return the pad itself, from an arm
    the caller's own `else` had already made unreachable.
    """
    shape = _bin_shape(interval, binsize, stats)
    out = np.zeros(shape)
    args = (*interval.as_tuple(), shape[0])
    try:
        with _borrow_as_tile_error():
            if len(stats) > 1:
                for k, stat in enumerate(stats):
                    out[:, k] = f.values(*args, stat, fillna=0)
            else:
                out[:] = f.values(*args, stats[0], fillna=0)
    except TileError:
        # Explicit, so the borrow translation's precedence over the broad
        # catch below is stated rather than resting on the accident that a
        # `TileError`'s text does not contain the chromosome needle.
        raise
    except Exception as exc:  # noqa: BLE001 -- matches current behavior
        if "No chromomsome with name" not in str(exc):
            raise
        out[:] = np.nan  # supported chromosome absent from the file, e.g. chrM
    return out


class BBISignalTileset(BBITileset):
    """A bigWig or bigBed served as a 1D vector tileset.

    See `BBITileset` for the parameters every BBI tileset shares,
    ``source`` among them.
    """

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
        # Entered per query, not per tile: the borrow lasts one call, and a
        # tile spanning many contigs issues one per range. Holding it across
        # the whole comprehension put every other request behind all of them.
        #
        # A range past the end of the genome contributes a NaN pad and no
        # I/O, so it skips the seam rather than taking the read lock to do
        # nothing -- measured at 554 of 2075 entries on this module's own
        # fixture, since the quadtree pads to a power of two. It does NOT
        # skip the chunk: `reconcile` below is given `expected_bins` and
        # refuses a short grid, and the pad is what fills the trailing cells
        # of a low-zoom tile. The two sibling tilesets filter these ranges
        # out entirely, which they can because they accumulate records.
        chunks = []
        for gr in canvas.invert(tid.pos[0]):
            if gr.is_out_of_bounds:
                values = _empty_bins(gr, canvas.binsize, stats)
            else:
                with self.reading() as f:
                    values = fetch(f, gr, canvas.binsize, stats)
            chunks.append((values, gr.end - gr.start))

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
    with _borrow_as_tile_error():
        return [(chrom,) + record for record in f.records(chrom, start, end)]


def to_bedlike(
    record: tuple, offsets: dict[str, int]
) -> AnnotationRecord | None:
    """One raw record as the client's bedlike shape.

    ``None`` for a contig absent from ``offsets``: it has no place on the
    canvas, so there is nothing to convert it to. The converter owns the
    lookup rather than the call site, matching `to_interaction` below and the
    three converters outcome 8 fixed.
    """
    chrom_offset = offsets.get(record[0])
    if chrom_offset is None:
        return None

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
    """A bigBed or bigWig served as a 1D annotation tileset.

    See `BBITileset` for the parameters every BBI tileset shares,
    ``source`` among them.
    """

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
            # Per query rather than around the loop: see `_tile` in
            # `BBISignalTileset` for why the wider block is the wrong scope.
            with self.reading() as f:
                records.extend(fetch_records(f, gr))

        offsets = chromsizes.offsets
        rows = [
            row for r in records if (row := to_bedlike(r, offsets)) is not None
        ]
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

    Prefer `BBIInteraction2DTileset` or `BBIInteractionLinksTileset`, which
    differ in how a tile selects.

    See `BBITileset` for the parameters every BBI tileset shares,
    ``source`` among them.
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
            # Per query rather than around the loop: see `_tile` in
            # `BBISignalTileset` for why the wider block is the wrong scope.
            with self.reading() as f:
                records.extend(fetch_records(f, gr))

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
    """A bigInteract served as 2D rectangles.

    See `BBITileset` for the parameters every BBI tileset shares,
    ``source`` among them.
    """

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
    """A bigInteract served as 1D links, for an arc-style track.

    Parameters
    ----------
    link_policy : LinkPolicy, optional [default: EITHER]
        Which interactions a tile keeps: those with both anchors in view,
        either one, or every interaction whose hull crosses it.

    See `BBITileset` for the parameters every BBI tileset shares, ``source``
    among them.
    """

    ndim = 1
    datatype = "bedlike"

    def __init__(
        self,
        source: SourceLike,
        chromsizes: Chromsizes | None = None,
        chromsizes_alts: dict[str, Chromsizes] | None = None,
        policy: TilePolicy | None = None,
        tile_size: int = TILE_SIZE,
        *,
        link_policy: LinkPolicy = LinkPolicy.EITHER,
    ):
        # Spelled out rather than swept into `*args`/`**kwargs`, so every
        # parameter this tileset takes is visible to `inspect.signature` and
        # to `help`. It was the one BBI class publishing a caller `*args`
        # where its five siblings publish five named parameters, and the
        # docstring pointing at `BBITileset` named four of them that
        # introspection would not admit existed.
        #
        # Coerced before the base constructor opens anything: a bad value
        # raises here, where there is no reader and no handle to release.
        #
        # Private behind a read-only property, matching `bedpe.py` and
        # `bed2ddb.py`, the other two link tilesets. A bare public attribute
        # let a value assigned after construction skip this coercion, and
        # `_tile`'s match then fell through to the hull arm -- serving every
        # interaction crossing the tile under the id of whatever was asked
        # for, with nothing raised.
        self._link_policy = LinkPolicy(link_policy)
        # By keyword, not position: this subclass duplicates its base's
        # parameter list, and a positional forward drops a parameter added to
        # the base later without failing.
        super().__init__(
            source,
            chromsizes=chromsizes,
            chromsizes_alts=chromsizes_alts,
            policy=policy,
            tile_size=tile_size,
        )

    @property
    def link_policy(self) -> LinkPolicy:
        return self._link_policy

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
            match self._link_policy:
                case LinkPolicy.BOTH:
                    keep = x_in and y_in
                case LinkPolicy.EITHER:
                    keep = x_in or y_in
                case LinkPolicy.HULL:
                    # Everything the index returned already meets the hull.
                    keep = True
                case _:
                    # Unreachable while `LinkPolicy` has three members and the
                    # property coerces. Named rather than left to fall
                    # through, because a fourth member would otherwise leave
                    # `keep` unbound and surface as `UnboundLocalError` from
                    # inside the record loop instead of as a refusal.
                    raise ValueError(
                        f"unhandled link policy {self._link_policy!r}"
                    )
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
