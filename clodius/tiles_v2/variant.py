from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, ClassVar, Sequence

import oxbow as ox
import polars as pl

from clodius.core.coords import Chromsizes, GenomicRange
from clodius.core.errors import (
    TilesetUnavailable,
    TileTooWide,
)
from clodius.core.tile import AnnotationRecord
from clodius.core.policies import (
    TilePolicy,
)
from clodius.core.tileid import TileId
from clodius.core.tileset import BaseTileset, TilesetInfo
from clodius.tiles_v2._exprs import known_chroms
from clodius.tiles_v2._index import indexed_contigs, screen_regions

TILE_SIZE = 1024
HASH_SEED = 0x1F4B_5C0D

# Columns needed to place and identify a record. `info` and the genotype block
# are left unprojected: oxbow pushes the projection into the reader, so not
# asking for them means not decoding them.
FIELDS = ["chrom", "pos", "id", "ref", "alt", "qual", "filter"]


def _end_expr(has_end_info: bool) -> pl.Expr:
    """Zero-based, half-open end of a record.

    ``pos + len(ref) - 1`` for an ordinary variant. A symbolic allele
    (``<DEL>``, ``<DUP>``) has a one-character REF and its true extent in
    ``INFO/END``, so that is preferred where the header declares it -- which is
    what pysam's ``rec.stop`` does, and what the ``len(ref)`` arithmetic in
    ``bedfile.py`` cannot.
    """
    from_ref = pl.col("pos") - 1 + pl.col("ref").str.len_bytes()
    if not has_end_info:
        return from_ref
    return (
        pl.when(pl.col("info").struct.field("END").is_not_null())
        .then(pl.col("info").struct.field("END"))
        .otherwise(from_ref)
        .cast(pl.Int64)
    )


# --- the two sources --------------------------------------------------------
#
# oxbow normalizes VCF and BCF to the same columns, and its two readers take the
# same arguments. Only the reader, the compression it expects and the index
# suffixes differ, so they are data rather than branches -- the same shape as
# :class:`~clodius.tiles_v2.bam.AlignmentFormat`.


@dataclass(frozen=True)
class VariantFormat:
    """What differs between a VCF and a BCF."""

    name: str
    reader: Callable
    # VCF is text and may arrive plain, gzipped or BGZF-compressed; BCF is
    # always BGZF. Only a BGZF source can be range-queried either way.
    compression: str
    # Sibling index extensions, most conventional first.
    index_suffixes: tuple[str, ...]

    def open(self, path: str, **kwargs):
        return self.reader(path, compression=self.compression, **kwargs)

    def has_sibling_index(self, path: str) -> bool:
        return any(os.path.exists(path + ext) for ext in self.index_suffixes)


VCF = VariantFormat("vcf", ox.from_vcf, "infer", (".tbi", ".csi"))
# `.csi` only: tabix indexes text, and a BCF is binary.
BCF = VariantFormat("bcf", ox.from_bcf, "bgzf", (".csi",))


def overlap_predicate(ranges: Sequence[GenomicRange]) -> pl.Expr:
    """Records overlapping any of ``ranges``, for the scan path.

    Built on ``chrom``/``pos`` only so polars can push it into the reader.
    ``pos`` is 1-based, and the record starts at ``pos - 1``; the end is
    ``_end``, a derived column, so the test is deliberately loose on that side
    and exact on the start. A record is kept when its start falls in the range,
    or when it began earlier and has not ended -- the latter needs ``_end``, so
    it is applied after the pushdown rather than as part of it.
    """
    terms = [
        (pl.col("chrom") == gr.name)
        & (pl.col("pos") - 1 < gr.end)
        & (pl.col("_end") > gr.start)
        for gr in ranges
        if not gr.is_out_of_bounds and gr.end > gr.start
    ]
    if not terms:
        return pl.lit(False)
    expr = terms[0]
    for term in terms[1:]:
        expr = expr | term
    return expr


def to_bedlike(row: dict, offsets: dict[str, int]) -> AnnotationRecord | None:
    """One materialized row as the client's bedlike shape.

    ``None`` when the record's contig is absent from ``offsets`` -- an
    unplaced contig, a naming mismatch, or a different assembly build. The row
    has no position on the genome-spanning axis, so it is skipped rather than
    raising: a bare ``KeyError`` is not a
    `clodius.core.errors.TileError` and would escape the server
    boundary as a 500. Handled here rather than at the call sites so a third
    caller cannot reintroduce it.
    """
    chrom = row["chrom"]
    offset = offsets.get(chrom)
    if offset is None:
        return None
    ids = row["id"] or []
    alts = row["alt"] or []
    filters = row["filter"] or []

    return {
        # Derived from the record, not from the ID column. VCF IDs are `.` for
        # most variants, so legacy's `uid: rec.id` gives every unnamed record
        # the same identity -- and the client keys on uid.
        "uid": format(row["_digest"], "016x"),
        "xStart": offset + row["_start"],
        "xEnd": offset + row["_end"],
        "chrOffset": offset,
        "importance": row["_digest"] / 2.0**64,
        # Strings throughout, unlike legacy's [str, int, int, str] with the
        # whole re-serialized record (newline included) in the last slot.
        "fields": [
            chrom,
            str(row["pos"]),
            ";".join(ids) if ids else ".",
            row["ref"],
            ",".join(alts) if alts else ".",
            "" if row["qual"] is None else str(row["qual"]),
            ";".join(filters) if filters else ".",
        ],
    }


class VariantTileset(BaseTileset):
    """A VCF or BCF served as 1D variant annotations.

    Prefer :class:`VcfTileset` or :class:`BcfTileset`, which differ only in
    their :class:`VariantFormat`.

    Parameters
    ----------
    path :
        The VCF. BGZF-compressed with a ``.tbi``/``.csi`` for range queries;
        plain or gzipped works too, but oxbow cannot range-query those, so
        every tile scans the file and ``max_scan_bytes`` applies.
    chromsizes :
        Optional. Defaults to the ``##contig`` lines in the VCF header, which
        oxbow exposes as ``chrom_sizes``. Pass explicitly to serve against a
        different assembly, or for a file whose header omits contigs.
    index_path :
        Only needed when the index is not a sibling of the data file.
    policy :
        ``max_span`` is both advertised (as ``max_tile_width``) and enforced.
    """

    ndim: ClassVar[int] = 1
    datatype: ClassVar[str] = "bedlike"
    modifiers = None
    options = frozenset()

    format: ClassVar[VariantFormat] = VCF

    def __init__(
        self,
        path: str | os.PathLike,
        chromsizes: Chromsizes | None = None,
        index_path: str | os.PathLike | None = None,
        policy: TilePolicy | None = None,
        tile_size: int = TILE_SIZE,
    ):
        self._path = os.fspath(path)
        self._index_path = os.fspath(index_path) if index_path else None
        # oxbow refuses a region query on anything but an indexed BGZF source,
        # so this decides between seeking and scanning rather than merely
        # deciding how fast the seek is.
        fmt = type(self).format
        self._is_indexed = (
            self._index_path is not None or fmt.has_sibling_index(self._path)
        )
        self.policy = policy or TilePolicy()
        self.tile_size = tile_size

        # Read the header once, projecting no INFO column: whether END is
        # declared is exactly what this open exists to find out, and asking
        # oxbow for an undeclared INFO field raises.
        header = fmt.open(
            self._path,
            fields=FIELDS,
            info_fields=None,
            samples=None,
            index=self._index_path,
        )
        self._has_end_info = any(
            name == "END" for name, _number, _type in header.info_field_defs
        )
        if chromsizes is None:
            pairs = header.chrom_sizes
            if not pairs:
                raise ValueError(
                    f"{self._path} declares no ##contig lines; pass chromsizes "
                    f"explicitly"
                )
            chromsizes = Chromsizes(
                tuple(n for n, _ in pairs), tuple(int(v) for _, v in pairs)
            )
        self._chromsizes = chromsizes
        self._known_chroms = known_chroms(chromsizes, "chrom")
        # What the index can be asked for; see `clodius.tiles_v2._index`.
        self._index_contigs = (
            indexed_contigs(self._path, self._index_path)
            if self._is_indexed
            else None
        )
        self._info = self._build_info()

    def chromsizes(self) -> Chromsizes:
        return self._chromsizes

    def info(self) -> TilesetInfo:
        return self._info

    def close(self) -> None:
        pass

    # --- ProvidesRegions ----------------------------------------------------

    def regions(
        self, offset: int, limit: int
    ) -> tuple[list[AnnotationRecord], bool]:
        """A page of variants in file order, plus whether more follow.

        The only entry point of the legacy module that anything calls -- resgen's
        ``regions`` view imports ``clodius.tiles.vcf`` for exactly this. That
        version pages by advancing a pysam iterator ``offset`` times, one
        ``next()`` per skipped record; ``slice`` pushes the row count into the
        reader instead.
        """
        # Filtered before the slice, which costs the reader-level row limit:
        # the slice is evaluated above the filter rather than pushed into the
        # scan node. The streaming engine still stops once the page fills, so
        # the read is bounded by matching rows rather than by file rows. Do not
        # "restore" the pushdown by reordering these two.
        #
        # Filtered first so `offset`, `limit` and the next-page probe all
        # range over the same population. Filtering afterwards lets
        # one unplaceable row consume the probe -- `has_next` goes False with
        # records unread -- and leaves `offset` indexing raw file rows while
        # the page is a filtered subset, so consecutive pages overlap.
        frame = (
            self._frame(regions=None)
            .filter(self._known_chroms)
            .slice(offset, limit + 1)
            .collect(engine="streaming")
        )
        offsets = self._chromsizes.offsets
        rows = [
            record
            for r in frame.to_dicts()
            if (record := to_bedlike(r, offsets)) is not None
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
        extra = {}
        if self.policy.max_span is not None:
            # The wire name stays `max_tile_width`: the client computes
            # `max_width / 2**z` and compares against it.
            extra["max_tile_width"] = self.policy.max_span
        return TilesetInfo.quadtree(self._chromsizes, TILE_SIZE, **extra)

    def _source(self, regions=None):
        """An oxbow source with only the columns this module reads."""
        return type(self).format.open(
            self._path,
            fields=FIELDS,
            info_fields=["END"] if self._has_end_info else None,
            samples=None,
            regions=regions,
            index=self._index_path,
        )

    def _frame(self, regions: list[str] | None) -> pl.LazyFrame:
        frame = self._source(regions).pl(lazy=True)
        digest = pl.concat_str(
            ["chrom", "pos", "ref"], separator="\t", ignore_nulls=True
        ).hash(seed=HASH_SEED)
        return frame.with_columns(
            _start=pl.col("pos") - 1,
            _end=_end_expr(self._has_end_info),
            _digest=digest,
        )

    def _check_scannable(self) -> None:
        """Refuse to scan an unindexed file above the policy ceiling.

        Only reachable without an index, where every tile re-reads the file.
        Same guard bed.py applies, and the same reason: the ceiling is a policy
        choice, not a memory limit -- the read itself streams.
        """
        limit = self.policy.max_scan_bytes
        if limit is None:
            return
        size = os.path.getsize(self._path)
        if size > limit:
            raise TilesetUnavailable(
                f"{self._path} is {size} bytes and has no index; files above "
                f"{limit} must be BGZF-compressed and indexed"
            )

    def _check_tile_span(self, tid: TileId) -> None:
        limit = self.policy.max_span
        if limit is None:
            return
        width = self._info.tile_width(tid.z)
        if width > limit:
            raise TileTooWide(
                f"tile spans {int(width)} bp; this tileset serves at most "
                f"{limit}. Zoom in."
            )

    def _tile(self, tid: TileId) -> list[AnnotationRecord]:
        # Before any I/O: the whole point of REFUSED is not reading the data.
        self._check_tile_span(tid)

        ranges = [
            gr
            for gr in self._info.canvas(tid.z).invert(tid.pos[0])
            if not gr.is_out_of_bounds
        ]
        if not ranges:
            return []

        if self._is_indexed:
            # A contig the index does not carry has no variants, so this
            # only drops queries that would have raised out of the reader
            # as a ComputeError and failed the whole batch.
            seekable = screen_regions(ranges, self._index_contigs)
            if not seekable:
                return []
            frame = self._frame([_region(gr) for gr in seekable])
        else:
            self._check_scannable()
            frame = self._frame(None).filter(overlap_predicate(ranges))

        # Bounded-memory downsampling, as bed.py and bedpe.py do it: `top_k`
        # keeps a heap of `cap` rows, so peak memory does not depend on how
        # many variants the tile covers. `_digest` is the same column
        # `to_bedlike` scales into `importance`, so the records the client
        # would have ranked highest are the ones that survive.
        cap = self.policy.max_records
        if cap is not None:
            frame = frame.top_k(cap, by="_digest")

        offsets = self._chromsizes.offsets
        records = [
            record
            for row in frame.collect(engine="streaming").to_dicts()
            if (record := to_bedlike(row, offsets)) is not None
        ]
        records.sort(key=lambda r: r["xStart"])
        return records


def _region(gr: GenomicRange) -> str:
    """A range as the 1-based inclusive string a VCF source expects.

    VCF is a 1-based format and oxbow reads a bare ``chrom:start-end`` in the
    source's own coordinate system, so this is ``"11"`` where the BED modules
    use ``"01"``. Getting it wrong shifts every tile by one base.
    """
    return gr.to_ucsc(coords="11")


class VcfTileset(VariantTileset):
    """A VCF, plain, gzipped or BGZF-compressed.

    Only a BGZF-compressed file with a ``.tbi``/``.csi`` can be range-queried;
    anything else is scanned per tile, under ``max_scan_bytes``.
    """

    format: ClassVar[VariantFormat] = VCF


class BcfTileset(VariantTileset):
    """A BCF, range-queried when a ``.csi`` sits beside it."""

    format: ClassVar[VariantFormat] = BCF


# --- Notes ------------------------------------------------------------------
#
# 1. Errors in the payload slot. `tiles()` returns `(TileId, payload | TileError)`
#    so a refusal cannot take its siblings down with it. This is a protocol
#    change and should be settled before other REFUSED types (bam) copy it. The
#    alternatives considered:
#      - raise: correct for a broken tileset, wrong for a refusal the client is
#        *expected* to trigger by zooming out;
#      - return an `ErrorTile` dict: puts wire-format knowledge back in library
#        code, which is what raising was meant to remove;
#      - never refuse, thin instead: a different density policy, not a fix.
#    The boundary already translates `TileError` into `ErrorTile`; this only
#    changes where it finds one.
#
# 2. The tile span check runs before any I/O. Legacy checks in two places with
#    two comparisons (`>=` in `tiles`, `>` in `single_tile`) and the second is
#    unreachable. One check, one comparison, one source for the number.
#
# 3. No `MAX_QUERY_SIZE` byte estimate. Legacy guards on `est_query_size` from
#    the tabix index -- 450000 here, 1000000 in `tabix.py`, applied only when an
#    index was passed, so an unindexed file skipped it entirely. The width guard
#    already bounds the request, and a byte estimate that silently does nothing
#    for half the inputs is worse than not having one. `TilePolicy` keeps the
#    field for `tabix.py`'s sake.
#
# 4. `INFO/END` is honored when the header declares it, so symbolic alleles
#    (`<DEL>`) get their real extent. `bedfile.py` cannot do this at all, and
#    its `len(t[3])` computes the row count of a column rather than the length
#    of a REF string.
#
# 5. Genotypes are never decoded. `samples=None` (oxbow >= 0.7 default) keeps
#    the per-sample block out of the projection, which for a population VCF is
#    the difference between reading seven columns and reading thousands.
#
# 6. Still unresolved, and not this module's to decide: `clodius/tiles/vcf.py`
#    and the `filetype == "vcf"` branch in `clodius/tiles/bedfile.py` should
#    both go, but the second is the live path in resgen-server, so removing it
#    is a coordinated change with the server's dispatch table.
