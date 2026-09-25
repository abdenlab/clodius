from __future__ import annotations

import os
from typing import Sequence

import oxbow as ox
import polars as pl

from clodius.core.coords import Chromsizes, GenomicRange
from clodius.core.errors import TilesetUnavailable
from clodius.core.tile import AnnotationRecord
from clodius.core.policies import TilePolicy
from clodius.core.tileid import TileId
from clodius.core.tileset import BaseTileset, TilesetInfo
from clodius.tiles_v2._exprs import known_chroms

TILE_SIZE = 1024

# Seed for the row digest. Fixed so that importance is reproducible across
# processes; the value itself is arbitrary.
HASH_SEED = 0x1F4B_5C0D

# oxbow's default schema: chrom/start/end plus everything else as one tab-joined
# `rest` string. That is exactly the shape the client wants, since `fields` is
# just the record's columns as text.
BED_SCHEMA = "bed3+"


def overlap_predicate(
    ranges: Sequence[GenomicRange], chromsizes: Chromsizes
) -> pl.Expr | None:
    """A pushdown-friendly filter matching records overlapping ``ranges``.

    Parameters
    ----------
    ranges : Sequence[GenomicRange]
        The tile's genomic ranges, in genome order. May be empty or contain
        out-of-bounds intervals.
    chromsizes : Chromsizes
        The coordinate system to interpret the ranges against.

    Returns
    -------
    pl.Expr or None
        A polars expression that evaluates to True for rows overlapping any of
        the ranges. ``pl.lit(False)`` if the ranges are all out-of-bounds, so
        no rows match. ``None`` if the ranges cover every chromosome in full,
        so no filter is needed.
    """
    covered = {
        gr.name
        for gr in ranges
        if not gr.is_out_of_bounds
        and gr.start == 0
        and gr.end >= chromsizes.lengths[gr.cid]
    }
    if len(covered) == len(chromsizes):
        return None

    terms = []
    for gr in ranges:
        if gr.is_out_of_bounds or gr.end <= gr.start:
            continue
        on_chrom = pl.col("chrom") == gr.name
        if gr.name in covered:
            terms.append(on_chrom)
        else:
            # Half-open intersection: a record ending exactly at gr.start does
            # not overlap.
            terms.append(
                on_chrom
                & (pl.col("end") > gr.start)
                & (pl.col("start") < gr.end)
            )

    if not terms:
        # Entirely past the end of the genome.
        return pl.lit(False)

    expr = terms[0]
    for term in terms[1:]:
        expr = expr | term
    return expr


def to_tile_record(
    row: dict, offsets: dict[str, int]
) -> AnnotationRecord | None:
    """One materialized row as the client's bedlike shape.

    ``None`` when the record's contig is absent from ``offsets`` -- an
    unplaced contig, a naming mismatch, a different assembly build, or a
    header line parsed as a record. The row has no position on the
    genome-spanning axis, so it is skipped rather than raising: a bare
    ``KeyError`` is not a :class:`~clodius.core.errors.TileError` and would
    escape the server boundary as a 500. Handled here rather than at the call
    sites so a third caller cannot reintroduce it.
    """
    chrom, start, end = row["chrom"], row["start"], row["end"]
    offset = offsets.get(chrom)
    if offset is None:
        return None
    rest = row["rest"]
    fields = [chrom, str(start), str(end)]
    if rest:
        fields.extend(rest.split("\t"))
    return {
        "uid": format(row["_digest"], "016x"),
        "chrOffset": offset,
        "xStart": offset + start,
        "xEnd": offset + end,
        "importance": row["_digest"] / 2.0**64,
        "fields": fields,
    }


def _has_sibling_index(path: str) -> bool:
    """Whether oxbow will auto-detect an index next to ``path``."""
    return any(os.path.exists(path + ext) for ext in (".tbi", ".csi"))


class BedTileset(BaseTileset):
    """A plain BED file served as a 1D annotation tileset.

    Parameters
    ----------
    path :
        The BED file. May be uncompressed, gzipped, or BGZF-compressed.
    chromsizes :
        Required. A BED file carries no chromosome lengths, so without these
        there is no coordinate system to tile and no way to place a record on
        the genome-spanning axis.
    index_path :
        A ``.tbi``/``.csi`` for a BGZF-compressed file. When omitted, oxbow
        still auto-detects a sibling index; pass ``index_path`` only to point at
        one stored elsewhere. With an index, a tile reads just its own byte
        ranges; without one, every tile scans the file.
    policy :
        Limits. ``max_records`` caps records per tile;
        ``max_scan_bytes`` refuses to scan an unindexed file above a
        size.
    """

    ndim = 1
    datatype = "bedlike"
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
        self.policy = policy or TilePolicy()
        self.tile_size = tile_size
        self._chromsizes = chromsizes
        self._known_chroms = known_chroms(chromsizes, "chrom")
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
        # oxbow sources are opened per query and own no persistent handle, so
        # there is nothing to release. Declared anyway: the protocol requires it
        # and a caller should not have to know which types are stateless.
        pass

    # --- ProvidesRegions ----------------------------------------------------

    def regions(
        self, offset: int, limit: int
    ) -> tuple[list[AnnotationRecord], bool]:
        """A page of records in file order, plus whether more follow."""
        # Reads ``limit + 1`` rows to answer "is there a next page" without a
        # count. The slice sits above the filter rather than inside the scan
        # node, so it no longer pushes a row count into the reader -- but the
        # streaming engine still stops once the page fills, which bounds the
        # read by matching rows instead of by file rows. Measured at ~47 ms
        # over the pushed-down plan, flat from 250k rows to 4M against 900 ms
        # for a full scan of the same file.
        #
        # Do not "restore" the pushdown by moving the filter back below the
        # slice. Filtered before the slice, not after. `offset`, `limit` and the
        # probe all have to range over the same population: filtering
        # afterwards lets one unplaceable row consume the probe -- `has_next`
        # goes False with records still unread -- and leaves `offset` indexing
        # raw file rows while the page is a filtered subset, so consecutive
        # pages overlap.
        frame = (
            self._scan()
            .filter(self._known_chroms)
            .slice(offset, limit + 1)
            .with_columns(_digest=self._digest_expr())
            .collect(engine="streaming")
        )
        offsets = self._chromsizes.offsets
        rows = [
            record
            for r in frame.to_dicts()
            if (record := to_tile_record(r, offsets)) is not None
        ]
        has_next = len(rows) > limit
        return [
            {
                "uid": r["uid"],
                "xStart": r["xStart"],
                "xEnd": r["xEnd"],
                "fields": r["fields"],
                "chrOffset": r["chrOffset"],
            }
            for r in rows[:limit]
        ], has_next

    # --- internals ----------------------------------------------------------

    def _build_info(self) -> TilesetInfo:
        return TilesetInfo.quadtree(self._chromsizes, self.tile_size)

    def _digest_expr(self) -> pl.Expr:
        """A stable 64-bit digest of the record's own bytes.

        Serves as both ``uid`` and ``importance``: rank order by digest is the
        same whether you compare the integers or the floats they scale to, so
        ``top_k`` never needs the division.
        """
        return pl.concat_str(
            ["chrom", "start", "end", "rest"], separator="\t", ignore_nulls=True
        ).hash(seed=HASH_SEED)

    def _check_scannable(self) -> None:
        """Refuse to scan an unindexed file above the policy ceiling."""
        if self._is_indexed or self._checked_size:
            return
        limit = self.policy.max_scan_bytes
        size = os.path.getsize(self._path)
        if limit is not None and size > limit:
            raise TilesetUnavailable(
                f"{self._path} is {size} bytes; files above "
                f"{limit} must be BGZF-compressed and indexed"
            )
        self._checked_size = True

    def _scan(self, regions: list[str] | None = None) -> pl.LazyFrame:
        self._check_scannable()
        return ox.from_bed(
            self._path,
            BED_SCHEMA,
            regions=regions,
            coords="01",
            index=self._index_path,
        ).pl(lazy=True)

    def _tile(self, tid: TileId) -> list[AnnotationRecord]:
        ranges = list(self._info.canvas(tid.z).invert(tid.pos[0]))
        in_bounds = [gr for gr in ranges if not gr.is_out_of_bounds]

        # Applied before the indexed/unindexed split so the two paths cannot
        # disagree. oxbow reads `regions=[]` as *no region filter* and scans
        # the whole file, while `overlap_predicate` correctly matches nothing
        # -- and a tile entirely past the end of the genome is routine, since
        # `max_width` always exceeds the genome length.
        if not in_bounds:
            return []

        if self._is_indexed:
            frame = self._scan([gr.to_ucsc(coords="01") for gr in in_bounds])
        else:
            frame = self._scan()
            predicate = overlap_predicate(ranges, self._chromsizes)
            if predicate is not None:
                frame = frame.filter(predicate)

        offsets = self._chromsizes.offsets

        # Only on the scan path, and before capping: an unknown contig that
        # consumed a cap slot would silently shorten the tile. The indexed
        # branch queries regions named by `canvas.invert`, i.e. drawn from the
        # very chromsizes this tests against, so there is no row for it to
        # drop -- 0.4-0.8 ms per tile spent proving that.
        if not self._is_indexed:
            frame = frame.filter(self._known_chroms)
        frame = frame.with_columns(_digest=self._digest_expr())

        # Bounded-memory downsampling: `top_k` keeps a heap of `cap` rows, so
        # peak memory does not depend on how many records the tile covers.
        cap = self.policy.max_records
        if cap is not None:
            frame = frame.top_k(cap, by="_digest")

        rows = frame.collect(engine="streaming").to_dicts()
        records = [
            record
            for row in rows
            if (record := to_tile_record(row, offsets)) is not None
        ]
        records.sort(key=lambda r: r["xStart"])
        return records
