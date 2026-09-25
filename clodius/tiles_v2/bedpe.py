from __future__ import annotations

import os
from typing import ClassVar, Sequence

import oxbow as ox
import polars as pl

from clodius.core.coords import Chromsizes, GenomicRange
from clodius.core.errors import TilesetUnavailable
from clodius.core.tile import Annotation2DRecord
from clodius.core.policies import TilePolicy, LinkPolicy
from clodius.core.tileid import TileId
from clodius.core.tileset import BaseTileset, TilesetInfo
from clodius.tiles_v2._exprs import known_chroms
from clodius.tiles_v2._index import indexed_contigs, screen_regions

TILE_SIZE = 1024

HASH_SEED = 0x1F4B_5C0D

# oxbow has no BEDPE reader. `bed3+` gives chrom/start/end plus everything else
# as one tab-joined `rest`, and anchor 2 is split back out of `rest` in
# `_with_anchor2`. Passing custom field definitions instead would type the
# columns properly but consumes the `+`, dropping the trailing BEDPE fields that
# `fields` has to carry. Isolated in one place so it can be deleted if oxbow
# grows a native BEDPE schema.
BEDPE_SCHEMA = "bed3+"

# Columns of anchor 2, parsed out of `rest`.
ANCHOR2 = ("chrom2", "start2", "end2")


def _with_anchor2(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Split ``chrom2``/``start2``/``end2`` off the front of ``rest``.

    ``splitn`` with n=4 keeps the remainder intact in the last field, so
    arbitrarily wide BEDPE files survive without knowing the column count.
    """
    return (
        frame.with_columns(
            _anchor2=pl.col("rest").str.splitn("\t", 4),
        )
        .unnest("_anchor2")
        .rename(
            {
                "field_0": "chrom2",
                "field_1": "start2",
                "field_2": "end2",
                "field_3": "rest2",
            }
        )
        .with_columns(
            pl.col("start2").cast(pl.Int64),
            pl.col("end2").cast(pl.Int64),
        )
    )


def _anchor_overlaps(
    ranges: Sequence[GenomicRange],
    chrom_col: str,
    start_col: str,
    end_col: str,
) -> pl.Expr:
    """Whether an anchor overlaps any of ``ranges``.

    Unlike ``bed.overlap_predicate`` this never returns ``None`` for "matches
    everything": the caller combines two of these with AND or OR, and a missing
    term would silently widen the other.
    """
    terms = [
        (pl.col(chrom_col) == gr.name)
        & (pl.col(end_col) > gr.start)
        & (pl.col(start_col) < gr.end)
        for gr in ranges
        if not gr.is_out_of_bounds and gr.end > gr.start
    ]
    if not terms:
        # Entirely past the end of the genome.
        return pl.lit(False)

    expr = terms[0]
    for term in terms[1:]:
        expr = expr | term
    return expr


def to_paired(
    row: dict, offsets: dict[str, int]
) -> Annotation2DRecord | None:
    """One materialized row as the client's bedpe shape.

    ``None`` when either anchor's contig is absent from ``offsets`` -- an
    unplaced contig, a naming mismatch, or a different assembly build. The
    record has no position on the genome-spanning axis, so it is skipped
    rather than raising: a bare ``KeyError`` is not a
    `clodius.core.errors.TileError` and would escape the server
    boundary as a 500. Handled here rather than at the call sites so a third
    caller cannot reintroduce it.
    """
    x_offset = offsets.get(row["chrom"])
    y_offset = offsets.get(row["chrom2"])
    if x_offset is None or y_offset is None:
        return None

    fields = [
        row["chrom"],
        str(row["start"]),
        str(row["end"]),
        row["chrom2"],
        str(row["start2"]),
        str(row["end2"]),
    ]
    if row["rest2"]:
        fields.extend(row["rest2"].split("\t"))

    return {
        "uid": format(row["_digest"], "016x"),
        "xStart": x_offset + row["start"],
        "xEnd": x_offset + row["end"],
        "yStart": y_offset + row["start2"],
        "yEnd": y_offset + row["end2"],
        # The x anchor's offset, under the name bed2ddb uses. Required: the
        # arcs track reads `chrOffset + fields[n]` whenever startField/endField
        # are configured.
        "chrOffset": x_offset,
        "xChrOffset": x_offset,
        "yChrOffset": y_offset,
        "importance": row["_digest"] / 2.0**64,
        "fields": fields,
    }


def _has_sibling_index(path: str) -> bool:
    return any(os.path.exists(path + ext) for ext in (".tbi", ".csi"))


class _BedpeBase(BaseTileset):
    """Shared reading, geometry and thinning for the two BEDPE tilesets.

    Everything here is identical between rectangles and links. The subclasses
    differ only in ``ndim`` and in :meth:`_select`, which is the whole point of
    splitting them: the selection rule is the type, not a runtime branch.
    """

    modifiers = None
    options = frozenset()

    def __init__(
        self,
        path: str | os.PathLike,
        chromsizes: Chromsizes,
        index_path: str | os.PathLike | None = None,
        policy: TilePolicy | None = None,
        tile_size: int = TILE_SIZE,
    ):
        self._path = os.fspath(path)
        self._index_path = os.fspath(index_path) if index_path else None
        self._is_indexed = self._index_path is not None or _has_sibling_index(
            self._path
        )
        self._chromsizes = chromsizes
        self._known_chroms = known_chroms(chromsizes, "chrom", "chrom2")
        # What the index can be asked for; see `clodius.tiles_v2._index`.
        self._index_contigs = (
            indexed_contigs(self._path, self._index_path)
            if self._is_indexed
            else None
        )
        self.policy = policy or TilePolicy()
        self.tile_size = tile_size
        self._info = self._build_info()
        self._checked_size = False

    @property
    def is_indexed(self) -> bool:
        return self._is_indexed

    def chromsizes(self) -> Chromsizes:
        return self._chromsizes

    def info(self) -> TilesetInfo:
        return self._info

    def close(self) -> None:
        pass

    # --- what the subclasses supply -----------------------------------------

    def _select(
        self, axes: list[list[GenomicRange]]
    ) -> tuple[pl.Expr, list[GenomicRange] | None]:
        """The tile's selection predicate, and the anchor-1 ranges to seek.

        ``axes`` holds the genomic ranges each of the tile's coordinate slots
        covers, already inverted by the caller and already screened for a slot
        with nothing in bounds.

        Returning the seek ranges alongside the predicate keeps the two
        consistent: a subclass cannot ask for a seek that its predicate does not
        justify, which is the mistake that silently drops records.
        """
        raise NotImplementedError

    # --- internals ----------------------------------------------------------

    def _build_info(self) -> TilesetInfo:
        # Square and two-dimensional in `min_pos`/`max_pos` for both subclasses.
        # bed2ddb declares `[1, 1]` / `[max_length, max_length]` regardless of
        # which track reads it, and a 1D track takes the x entries and ignores
        # the rest, so this stays uniform rather than varying with `ndim`.
        return TilesetInfo.quadtree(self._chromsizes, self.tile_size, ndim=2)

    def _digest_expr(self) -> pl.Expr:
        """A stable 64-bit digest of the whole record.

        Covers both anchors and the trailing fields, so two records sharing an
        anchor still rank independently. Legacy used the pandas row index, which
        is not a property of the record at all.
        """
        return pl.concat_str(
            ["chrom", "start", "end", "rest"],
            separator="\t",
            ignore_nulls=True,
        ).hash(seed=HASH_SEED)

    def _check_scannable(self) -> None:
        """Refuse to scan an unindexed file above the policy ceiling.

        The `_is_indexed` arm is load-bearing rather than an optimization.
        `BedpeLinksTileset._select` returns no seek for `LinkPolicy.EITHER` --
        the default -- so an indexed file reaches this on every in-bounds tile,
        and without the early return a perfectly well-indexed file above the
        ceiling is refused with a message saying it is not indexed. Same guard
        bed.py and variant.py apply, for the same reason.
        """
        if self._is_indexed or self._checked_size:
            return
        limit = self.policy.max_scan_bytes
        size = os.path.getsize(self._path)
        if limit is not None and size > limit:
            raise TilesetUnavailable(
                f"{self._path} is {size} bytes; files above {limit} must be "
                f"BGZF-compressed and indexed"
            )
        self._checked_size = True

    def _frame(self, seek_to: list[GenomicRange] | None) -> pl.LazyFrame:
        """A lazy frame over the file, seeking when asked and able.

        ``seek_to`` is an anchor-1 restriction the caller has established is
        sound for its predicate, or ``None`` to scan.
        """
        regions = None
        if seek_to is not None and self._is_indexed:
            regions = [
                gr.to_ucsc(coords="01")
                for gr in seek_to
                if not gr.is_out_of_bounds
            ]
            if not regions:
                # Wholly out of bounds. An empty region list would be read as
                # "no restriction" and scan the file.
                regions = None
        if regions is None:
            self._check_scannable()

        frame = ox.from_bed(
            self._path,
            BEDPE_SCHEMA,
            regions=regions,
            index=self._index_path,
        ).pl(lazy=True)
        return _with_anchor2(frame)

    def _tile(self, tid: TileId) -> list[Annotation2DRecord]:
        canvas = self._info.canvas(tid.z)

        # Screened before any frame is built. `max_width` always exceeds the
        # genome length, so a tile past the end of the genome is routine --
        # and there every range is out of bounds, the seek list collapses to
        # "no restriction", and `_check_scannable` raises TilesetUnavailable
        # for an indexed file above the ceiling. That is a *sibling* of
        # TileError, not a subclass, so the per-tile boundary does not catch
        # it and one routine off-the-end tile fails the whole request. Under
        # the ceiling it merely scans the file end to end to match nothing.
        axes = [list(canvas.invert(pos)) for pos in tid.pos]
        if any(all(gr.is_out_of_bounds for gr in axis) for axis in axes):
            return []

        predicate, seek_to = self._select(axes)
        if seek_to is not None and self._is_indexed:
            # Only the seeking policies reach this. A contig absent from
            # the index has no records, so an empty screen is an empty
            # tile -- and querying it would raise past the per-tile
            # boundary as a ComputeError.
            seek_to = screen_regions(seek_to, self._index_contigs)
            if not seek_to:
                return []

        offsets = self._chromsizes.offsets
        frame = (
            self._frame(seek_to)
            .filter(predicate)
            # Filtered before capping: an unknown contig that consumed a cap
            # slot would silently shorten the tile.
            .filter(self._known_chroms)
            .with_columns(_digest=self._digest_expr())
        )

        cap = self.policy.max_records
        if cap is not None:
            frame = frame.top_k(cap, by="_digest")

        records = [
            record
            for row in frame.collect(engine="streaming").to_dicts()
            if (record := to_paired(row, offsets)) is not None
        ]
        records.sort(key=lambda r: (r["xStart"], r["yStart"]))
        return records


class BedpeTileset(_BedpeBase):
    """A BEDPE served as 2D rectangles, for annotations over a contact map.

    One selection rule, and it is not configurable: a record is in tile
    ``(x, y)`` when anchor 1 overlaps the x range and anchor 2 overlaps the y
    range. That is not one option among several -- it is exactly "the drawn
    rectangle intersects the tile", because the drawn rectangle *is*
    ``anchor1 x anchor2``. A record spanning the whole tile is selected by it,
    since the anchors then overlap the ranges even though neither is contained
    in one.

    Not mirrored: a record is emitted at ``(x, y)`` and never also at
    ``(y, x)``. Deliberate, and it is what makes the anchor-1 seek sound -- the
    tile's condition is asymmetric in the same way the index is.
    """

    ndim: ClassVar[int] = 2
    datatype: ClassVar[str] = "2d-rectangle-domains"

    def _select(self, axes):
        x_ranges, y_ranges = axes
        predicate = _anchor_overlaps(
            x_ranges, "chrom", "start", "end"
        ) & _anchor_overlaps(y_ranges, "chrom2", "start2", "end2")
        # Anchor 1 is constrained to the x range, so seeking it is sound; the
        # anchor-2 term trims what comes back.
        return predicate, x_ranges


class BedpeLinksTileset(_BedpeBase):
    """A BEDPE served as 1D links, for an arc-style track.

    Parameters
    ----------
    link_policy :
        Which links a tile returns; see :class:`LinkPolicy`. Defaults to
        ``EITHER``, matching legacy. ``HULL`` is what the arcs renderer's
        geometry implies, but see the notes -- the renderer does not dedupe
        across tiles, so a link is drawn once per tile it spans.

    Everything else is inherited. Note ``ndim`` is 1: this tileset answers
    ``uid.z.x`` and nothing else. A BEDPE to be shown as both links and
    rectangles is registered twice, once per class.
    """

    ndim: ClassVar[int] = 1
    datatype: ClassVar[str] = "bedlike"

    def __init__(
        self, *args, link_policy: LinkPolicy = LinkPolicy.EITHER, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._link_policy = LinkPolicy(link_policy)

    @property
    def link_policy(self) -> LinkPolicy:
        return self._link_policy

    def _select(self, axes):
        (ranges,) = axes
        a1 = _anchor_overlaps(ranges, "chrom", "start", "end")
        a2 = _anchor_overlaps(ranges, "chrom2", "start2", "end2")

        match self._link_policy:
            case LinkPolicy.BOTH:
                # Anchor 1 must be in range, so the seek is sound -- the only
                # policy of the three for which that is true.
                return a1 & a2, ranges
            case LinkPolicy.EITHER:
                # Anchor 2 is not in the index, so a seek would drop exactly the
                # records the disjunction exists to include.
                return a1 | a2, None
            case LinkPolicy.HULL:
                # Restricted to intra-chromosomal pairs: the hull of an
                # inter-chromosomal pair is not a genomic interval, and an arc
                # track has nowhere to draw it. Those records fall back to the
                # EITHER condition rather than being dropped, so nothing
                # disappears relative to the weaker policy.
                intra = pl.col("chrom") == pl.col("chrom2")
                hull_terms = [
                    (pl.col("chrom") == gr.name)
                    & (pl.min_horizontal("start", "start2") < gr.end)
                    & (pl.max_horizontal("end", "end2") > gr.start)
                    for gr in ranges
                    if not gr.is_out_of_bounds and gr.end > gr.start
                ]
                hull = pl.lit(False)
                for term in hull_terms:
                    hull = hull | term
                return (intra & hull) | (~intra & (a1 | a2)), None


# --- Notes ------------------------------------------------------------------
#
# 1. The predicate is applied even on the seek path, unlike bed.py where it was
#    removed as redundant. It is not redundant here: the region query restricts
#    anchor 1 only, so the anchor-2 term is what makes the result correct. The
#    anchor-1 term is the redundant half, kept because splitting the conjunction
#    to drop it costs more clarity than the filter costs time -- polars
#    evaluates it against an already-narrow frame.
#
# 2. `LinkPolicy.HULL` is not the default, despite being what the arcs
#    renderer's geometry implies. `Arcs1DTrack` collects items with
#    `Object.values(this.fetchedTiles).flatMap(t => t.tileData)` and does not
#    dedupe by uid, so a link shipped in each of the N tiles its hull spans is
#    drawn N times -- overplotted rather than visibly wrong, but the cost grows
#    with span. Fixable upstream in one line; until then HULL is opt-in.
#
#    The renderer does handle off-tile endpoints correctly: it projects through
#    the track's scale, clamps the angular sweep to
#    `[max(0, x1), min(trackWidth, x2)]`, and has an explicit
#    `completelyContained` option -- defaulting to false -- whose only purpose is
#    to discard links with an endpoint outside the range. So HULL is safe to
#    ship, and BOTH duplicates that client option server-side.
#
# 3. Not mirrored -- a record is emitted at (x, y), never also at (y, x).
#    Intentional. It is also what makes the anchor-1 seek sound: the tile's
#    condition is asymmetric in the same way the index is.
#
#    A pairix-indexed BEDPE would make a 2D tile a genuine two-range seek rather
#    than "seek anchor 1, then filter anchor 2". Performance, not correctness.
#    See _scratch/clodius-record-tilesets.md section 7.
#
# 4. bigInteract is the format that makes HULL cheap: it stores the hull *as*
#    the record's canonical range, so a hull query is one ordinary bigBed range
#    query against the R-tree instead of a scan. oxbow reads it via
#    `from_bigbed(path, schema="autosql")`. That belongs as a sibling of the
#    bigbed tileset, not here.
#
# 5. `max_records` defaults to 1024 from TilePolicy, where legacy
#    hardcodes 512 for bedpe and 1024 for bedfile. Never justified separately;
#    a 2D tile arguably wants fewer records than a 1D one, so if the difference
#    is deliberate it belongs in the policy as a second field rather than as a
#    constant here. Now that the two are separate classes, they could also carry
#    different defaults -- which is an argument for the split, not against it.
#
#
# What changes relative to ``clodius/tiles/bedpe.py``
# --------------------------------------------------
# Everything ``bed.py`` fixed, for the same reasons: no whole-file pandas
# materialization, ``importance`` from the record digest rather than
# ``random.random()``, thinning by ``top_k`` rather than ``df.sample``, errors
# raised rather than returned as ``{"error": ...}``, limits from
# :class:`TilePolicy`, and ``chromsizes`` required at construction. Plus:

# - **``uid`` is a content digest, not the row number.** Legacy uses ``row["ix"]``,
#   the pandas index, so uids change if the file is re-sorted.
# - **``chrOffset`` is emitted.** Legacy sends only ``xChrOffset``/``yChrOffset``,
#   so an arcs track configured with ``startField``/``endField`` computes
#   ``undefined + n`` and gets NaN. See :class:`Annotation2DRecord`.
# - **The double cache write is gone.** ``bedpe_to_df`` writes a raw DataFrame and
#   then a dict under the same key.
