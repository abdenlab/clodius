"""GFF3 and GTF served as gene or transcript tiles.

Two :class:`Dialect`s:

===================  =========================  ==============================
                     GFF3                       GTF
===================  =========================  ==============================
feature identity     ``ID``                     implicit, by row type
parent link          ``Parent``                 ``gene_id`` / ``transcript_id``
transcript type      ``mRNA``, ``lnc_RNA``, ...  ``transcript``
===================  =========================  ==============================

Two wire layouts

Both tracks read the ``fields`` field positionally:

  gene annotations (`horizontal-gene-annotations`)
    0 chrom     3 geneName   6 refseqId   9 geneDesc   12 exonStarts
    1 txStart   4 importance 7 geneId    10 cdsStart   13 exonEnds
    2 txEnd     5 strand     8 geneType  11 cdsEnd

  transcripts (`horizontal-transcripts`)
    0 chrom     3 txName     6 --         9 exonStarts  12 stopCodonPos
    1 txStart   4 importance 7 --        10 exonEnds
    2 txEnd     5 strand     8 codingType 11 startCodonPos

The transcripts track is also 1-based: it subtracts one from everything it
reads, so this layout is written 1-based to survive the trip.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Sequence

import oxbow as ox
import polars as pl

from clodius.core.coords import Chromsizes, GenomicRange
from clodius.core.errors import TileError, TilesetUnavailable
from clodius.core.tile import AnnotationRecord
from clodius.core.policies import (
    TilePolicy,
)
from clodius.core.tileid import TileId
from clodius.core.tileset import BaseTileset, TilesetInfo

TILE_SIZE = 1024
TRANSCRIPT_TYPES = frozenset(
    {
        "mRNA",
        "transcript",
        "lnc_RNA",
        "ncRNA",
        "primary_transcript",
        "antisense_RNA",
        "snoRNA",
        "tRNA",
        "rRNA",
        "snRNA",
        "SRP_RNA",
        "RNase_P_RNA",
        "RNase_MRP_RNA",
    }
)

GENE_TYPES = frozenset({"gene"})
PSEUDOGENE_TYPES = frozenset({"pseudogene"})
CHILD_TYPES = frozenset({"exon", "CDS"})


@dataclass(frozen=True)
class Dialect:
    """Everything that differs between GFF3 and GTF.

    ``identity`` returns ``(feature_id, parent_id)`` for a row. That single
    function is the whole format difference: GFF3 states both explicitly in
    ``ID``/``Parent``, while GTF leaves them implicit and they must be derived
    from the row's type and its ``gene_id``/``transcript_id``.
    """

    name: str
    reader: Callable
    attribute_defs: tuple[tuple[str, str], ...]
    identity: Callable[[str, dict], tuple[str | None, str | None]]

    @property
    def attribute_names(self) -> list[str]:
        return [name for name, _ in self.attribute_defs]


def _gff_identity(feature_type: str, attrs: dict):
    parent = attrs.get("Parent")
    # GFF3 permits `Parent=a,b`. Only the first is used, matching legacy, which
    # treats the raw string as a key and so silently fails to match on multi-
    # parent rows rather than attaching to both.
    if parent and "," in parent:
        parent = parent.split(",")[0]
    return attrs.get("ID"), parent


def _gtf_identity(feature_type: str, attrs: dict):
    gene_id = attrs.get("gene_id")
    tx_id = attrs.get("transcript_id")
    if feature_type in GENE_TYPES or feature_type in PSEUDOGENE_TYPES:
        return gene_id, None
    if feature_type in TRANSCRIPT_TYPES:
        return tx_id, gene_id
    # exon / CDS: no identity of their own in GTF, and their parent is the
    # transcript.
    return None, tx_id


GFF3 = Dialect(
    name="gff",
    reader=ox.from_gff,
    attribute_defs=(
        ("ID", "String"),
        ("Parent", "String"),
        ("Name", "String"),
        ("gene_biotype", "String"),
        ("pseudo", "String"),
        ("description", "String"),
    ),
    identity=_gff_identity,
)

GTF = Dialect(
    name="gtf",
    reader=ox.from_gtf,
    attribute_defs=(
        ("gene_id", "String"),
        ("transcript_id", "String"),
        ("gene_name", "String"),
        ("gene_biotype", "String"),
    ),
    identity=_gtf_identity,
)


def _entity(row: dict, attrs: dict, feature_id: str) -> dict:
    """The fields every model class shares."""
    return {
        "id": feature_id,
        "chrom": row["seqid"],
        "start": row["start"],
        "end": row["end"],
        "strand": row["strand"] if row["strand"] in ("+", "-", ".") else None,
        "score": row["score"],
        "phase": row["frame"],
        "attributes": attrs,
    }


@dataclass
class Transcript:
    """One transcript, with the children that were linked to it."""

    row: dict
    attrs: dict
    exons: list[tuple[int, int]] = field(default_factory=list)
    cds: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class Gene:
    """One gene or pseudogene, with its transcripts.

    Plain containers rather than the pydantic classes in
    :mod:`clodius.models.gff_models`: nothing here is validated or serialized,
    and of that module's thirteen transcript classes only ``mRNA`` ever behaved
    differently (it alone carries CDS). No transcript type reaches the wire --
    ``geneType`` comes from the *gene*'s ``gene_biotype`` -- so the taxonomy
    distinguished nothing a client could see.
    """

    row: dict
    attrs: dict
    transcripts: list[Transcript] = field(default_factory=list)
    # A pseudogene's exons hang off the gene directly, with no transcript
    # between them.
    exons: list[tuple[int, int]] = field(default_factory=list)

    @property
    def chrom(self) -> str:
        return self.row["seqid"]

    @property
    def span(self) -> tuple[int, int]:
        """``[start, end)``, converted from GFF's 1-based inclusive."""
        return (self.row["start"] - 1, self.row["end"])

    @property
    def strand(self) -> str:
        return self.row["strand"] if self.row["strand"] in ("+", "-") else "+"

    @property
    def biotype(self) -> str:
        return self.attrs.get("gene_biotype") or self.row["type"]

    @property
    def name(self) -> str:
        return self.attrs.get("Name") or self.attrs.get("gene_name") or ""


def build_genes(rows: list[dict], dialect: Dialect) -> dict[str, Gene]:
    """Link flat feature rows into genes, transcripts and their children.

    Two passes rather than legacy's one. Legacy attaches a child to its parent
    at the moment it sees the child, so a child preceding its parent in the file
    is dropped -- and GTF makes no ordering guarantee at all. Collecting first
    and linking second removes the ordering assumption.
    """
    genes: dict[str, Gene] = {}
    transcripts: dict[str, Transcript] = {}
    tx_to_gene: dict[str, str] = {}
    # A miRNA sits between a primary_transcript and its exons, so a child's
    # parent may be one level further from the gene than usual.
    mirna_to_tx: dict[str, str] = {}
    children: list[tuple[str, str, int, int]] = []

    for row in rows:
        attrs = {
            k: v
            for k, v in (row.get("attributes") or {}).items()
            if v is not None
        }
        feature_type = row["type"]
        feature_id, parent_id = dialect.identity(feature_type, attrs)
        span = (row["start"] - 1, row["end"])

        if feature_type in CHILD_TYPES:
            if parent_id:
                children.append((feature_type, parent_id, *span))
            continue

        if not feature_id:
            feature_id = f"{feature_type}_{row['start']}_{row['end']}"

        if feature_type in GENE_TYPES or feature_type in PSEUDOGENE_TYPES:
            genes[feature_id] = Gene(row=row, attrs=attrs)
        elif feature_type in TRANSCRIPT_TYPES:
            transcripts[feature_id] = Transcript(row=row, attrs=attrs)
            if parent_id:
                tx_to_gene[feature_id] = parent_id
        elif feature_type == "miRNA" and parent_id:
            mirna_to_tx[feature_id] = parent_id
        # Anything else -- region, sequence_feature, mobile_genetic_element --
        # is not part of a gene model and is skipped.

    for kind, parent_id, lo, hi in children:
        parent_id = mirna_to_tx.get(parent_id, parent_id)
        transcript = transcripts.get(parent_id)
        if transcript is not None:
            target = transcript.cds if kind == "CDS" else transcript.exons
            target.append((lo, hi))
        elif parent_id in genes:
            # A pseudogene's exons, which have no transcript in between.
            genes[parent_id].exons.append((lo, hi))

    for tx_id, transcript in transcripts.items():
        gene = genes.get(tx_to_gene.get(tx_id))
        if gene is not None:
            gene.transcripts.append(transcript)

    return genes


def _union(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping intervals, as the beddb pipeline's ``exonU.py`` does.

    A gene's transcripts share exons, so the naive concatenation would draw the
    same block many times over. The gene-annotations track has no notion of
    transcripts -- it draws one row per gene -- so the exons it is given must
    already be a union.
    """
    if not intervals:
        return []
    ordered = sorted(intervals)
    out = [ordered[0]]
    for start, end in ordered[1:]:
        if start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def _record(
    uid: str,
    chrom: str,
    span: tuple[int, int],
    offset: int,
    strand: str,
    fields: list[str],
) -> AnnotationRecord:
    lo, hi = span
    return {
        "uid": uid,
        "xStart": offset + lo,
        "xEnd": offset + hi,
        "chrOffset": offset,
        # Span length stands in for citationCount, which no annotation file
        # carries. Deterministic, and it ranks whole features above fragments.
        "importance": float(hi - lo),
        "strand": strand,
        "fields": fields,
    }


def to_gene_records(
    genes: dict[str, Gene], offsets: dict[str, int]
) -> list[AnnotationRecord]:
    """One record per gene, exons merged across its transcripts."""
    records: list[AnnotationRecord] = []

    for gene_id, gene in genes.items():
        offset = offsets.get(gene.chrom)
        if offset is None:
            continue

        lo, hi = gene.span
        exons = list(gene.exons)
        cds: list[tuple[int, int]] = []
        for transcript in gene.transcripts:
            exons += transcript.exons
            cds += transcript.cds
        exons = _union(exons) or [(lo, hi)]
        # UCSC writes cdsStart == cdsEnd for a non-coding gene.
        cds_start = min(c[0] for c in cds) if cds else lo
        cds_end = max(c[1] for c in cds) if cds else lo

        records.append(
            _record(
                gene_id,
                gene.chrom,
                (lo, hi),
                offset,
                gene.strand,
                [
                    gene.chrom,
                    str(lo),
                    str(hi),
                    gene.name or gene_id,
                    str(float(hi - lo)),
                    gene.strand,
                    gene_id,
                    gene.attrs.get("gene_id") or gene_id,
                    gene.biotype,
                    gene.attrs.get("description") or "",
                    str(cds_start),
                    str(cds_end),
                    ",".join(str(s) for s, _ in exons),
                    ",".join(str(e) for _, e in exons),
                ],
            )
        )

    records.sort(key=lambda r: r["xStart"])
    return records


def to_transcript_records(
    genes: dict[str, Gene], offsets: dict[str, int]
) -> list[AnnotationRecord]:
    """One record per transcript, with its own exons and codon positions.

    Coordinates stay 1-based, as they are in the file: the track subtracts one
    from everything it reads (``txStart = +ts[1] - 1``, exon starts likewise).
    Verified against ``higlass-transcripts/examples/transcripts_test.txt``.
    """
    records: list[AnnotationRecord] = []

    for gene_id, gene in genes.items():
        offset = offsets.get(gene.chrom)
        if offset is None:
            continue

        for transcript in gene.transcripts:
            attrs = transcript.attrs
            tx_id = attrs.get("ID") or attrs.get("transcript_id") or ""
            lo, hi = transcript.row["start"] - 1, transcript.row["end"]
            exons = _union(transcript.exons) or [(lo, hi)]
            start_codon, stop_codon = _codon_starts(transcript.cds, gene.strand)

            records.append(
                _record(
                    tx_id or f"{gene.chrom}_{lo}_{hi}",
                    gene.chrom,
                    (lo, hi),
                    offset,
                    gene.strand,
                    [
                        gene.chrom,
                        str(lo + 1),
                        str(hi),
                        attrs.get("Name") or tx_id,
                        str(float(hi - lo)),
                        gene.strand,
                        gene.attrs.get("gene_id") or gene_id,
                        tx_id,
                        gene.biotype,
                        ",".join(str(s + 1) for s, _ in exons),
                        ",".join(str(e) for _, e in exons),
                        start_codon,
                        stop_codon,
                    ],
                )
            )

    records.sort(key=lambda r: r["xStart"])
    return records


def _codon_starts(cds: list[tuple[int, int]], strand: str) -> tuple[str, str]:
    """Where the start and stop codons begin, 1-based, or ``"."`` for neither.

    A GTF names ``start_codon``/``stop_codon`` outright, but a GFF3 carries only
    CDS, so the two ends of the CDS extent stand in for them -- which end is
    which depends on the strand. Both columns hold the codon's *lowest*
    coordinate; the track picks the end it needs
    (``+ts[11] - 1`` on the plus strand, ``+ts[11] + 2`` on the minus).

    ``"."`` is the track's sentinel for a non-coding transcript, and it draws
    the whole thing as UTR.
    """
    if not cds:
        return (".", ".")
    lowest = min(lo for lo, _ in cds) + 1
    highest = max(hi for _, hi in cds)
    if strand == "-":
        # Translation runs from the high end, so the start codon is up there.
        return (str(highest - 2), str(lowest))
    return (str(lowest), str(highest - 2))


def _has_sibling_index(path: str) -> bool:
    return any(os.path.exists(path + ext) for ext in (".tbi", ".csi"))


class GxfTileset(BaseTileset):
    """A GFF3 or GTF served as gene-model tiles.

    Parameters
    ----------
    path :
        The annotation file. BGZF-compressed with a ``.tbi``/``.csi`` for range
        queries; otherwise every tile scans.
    chromsizes :
        Optional for GFF3 with ``##sequence-region``-derived ``region`` rows,
        which is what legacy's ``gff_chromsizes`` reads. Required for GTF, which
        has no such convention.
    dialect :
        :data:`GFF3` or :data:`GTF`. Prefer :class:`GffTileset` /
        :class:`GtfTileset`.
    policy :
        ``max_records`` caps *genes* per tile, not rows.
    """

    ndim: ClassVar[int] = 1
    datatype: ClassVar[str] = "gene-annotation"
    modifiers = None
    options = frozenset()

    dialect: ClassVar[Dialect] = GFF3

    def __init__(
        self,
        path: str | os.PathLike,
        chromsizes: Chromsizes | None = None,
        index_path: str | os.PathLike | None = None,
        dialect: Dialect | None = None,
        policy: TilePolicy | None = None,
        tile_size: int = TILE_SIZE,
    ):
        self._path = os.fspath(path)
        self._index_path = os.fspath(index_path) if index_path else None
        self._is_indexed = self._index_path is not None or _has_sibling_index(
            self._path
        )
        self._dialect = dialect or type(self).dialect
        self._chromsizes = chromsizes or self._derive_chromsizes()
        self._info = self._build_info()
        self._checked_size = False
        self.policy = policy or TilePolicy()
        self.tile_size = tile_size

    def chromsizes(self) -> Chromsizes:
        return self._chromsizes

    def info(self) -> TilesetInfo:
        return self._info

    def tiles(
        self, ids: Sequence[TileId], options=None
    ) -> list[tuple[TileId, list[AnnotationRecord]]]:
        """One entry per requested id; a refusal rides in the payload slot."""
        out = []
        for tid in ids:
            try:
                out.append((tid, self._tile(tid)))
            except TileError as exc:
                out.append((tid, exc.to_dict()))
        return out

    def close(self) -> None:
        pass

    # --- internals ----------------------------------------------------------

    def _derive_chromsizes(self) -> Chromsizes:
        """Chromsizes from ``region`` rows, as legacy's ``gff_chromsizes`` does.

        A GFF3 written from ``##sequence-region`` directives carries one
        ``region`` feature per sequence. GTF has no equivalent, so this raises
        there rather than guessing from the maximum observed coordinate -- which
        would silently shrink the coordinate space to whatever the file happens
        to annotate.
        """
        frame = (
            self._source(None)
            .pl(lazy=True)
            .filter(pl.col("type") == "region")
            .select("seqid", "end")
            .collect(engine="streaming")
        )
        if frame.is_empty():
            raise ValueError(
                f"{self._path} has no 'region' features to derive chromsizes "
                f"from; pass chromsizes explicitly"
            )
        names = tuple(frame["seqid"])
        return Chromsizes(names, tuple(int(v) for v in frame["end"]))

    def _build_info(self) -> TilesetInfo:
        return TilesetInfo.quadtree(self._chromsizes, TILE_SIZE)

    def _source(self, regions: list[str] | None):
        return self._dialect.reader(
            self._path,
            attribute_defs=list(self._dialect.attribute_defs),
            regions=regions,
            index=self._index_path,
        )

    def _check_scannable(self) -> None:
        if self._checked_size:
            return
        limit = self.policy.max_scan_bytes
        size = os.path.getsize(self._path)
        if limit is not None and size > limit:
            raise TilesetUnavailable(
                f"{self._path} is {size} bytes; files above {limit} must be "
                f"BGZF-compressed and indexed"
            )
        self._checked_size = True

    def _rows(self, ranges: Sequence[GenomicRange]) -> list[dict]:
        """Rows overlapping ``ranges``, seeking when indexed and scanning when not."""
        if self._is_indexed:
            regions = [
                gr.to_ucsc(coords="11")
                for gr in ranges
                if not gr.is_out_of_bounds and gr.end > gr.start
            ]
            if not regions:
                return []
            frame = self._source(regions).pl(lazy=True)
        else:
            self._check_scannable()
            frame = self._source(None).pl(lazy=True).filter(_overlaps(ranges))
        return frame.collect(engine="streaming").to_dicts()

    def _thin(self, rows: list[dict]) -> list[dict]:
        """Cap by gene, keeping every child of the genes kept.

        Row-level thinning would sever exons from their transcripts and produce
        a gene missing pieces. Selection is by span length -- the same measure
        emitted as ``importance`` -- so the genes that survive the cap are the
        ones the client would have ranked highest anyway, at every zoom and
        across requests.
        """
        cap = self.policy.max_records
        if cap is None:
            return rows

        spans: dict[str, int] = {}
        for row in rows:
            attrs = row.get("attributes") or {}
            if row["type"] in GENE_TYPES or row["type"] in PSEUDOGENE_TYPES:
                fid, _ = self._dialect.identity(row["type"], attrs)
                if fid:
                    spans[fid] = row["end"] - row["start"]
        if len(spans) <= cap:
            return rows

        ranked = sorted(spans, key=lambda fid: (spans[fid], fid), reverse=True)
        keep = set(ranked[:cap])
        # A row survives if it belongs to a kept gene, directly or through its
        # transcript. Resolved by walking gene -> transcript -> child.
        kept_tx = set()
        for row in rows:
            attrs = row.get("attributes") or {}
            fid, parent = self._dialect.identity(row["type"], attrs)
            if row["type"] in TRANSCRIPT_TYPES and parent in keep and fid:
                kept_tx.add(fid)

        out = []
        for row in rows:
            attrs = row.get("attributes") or {}
            fid, parent = self._dialect.identity(row["type"], attrs)
            if fid in keep or parent in keep or parent in kept_tx:
                out.append(row)
        return out

    def _records(
        self, genes: dict[str, Gene], offsets: dict[str, int]
    ) -> list[AnnotationRecord]:
        """One record per gene. Overridden to serve transcripts instead.

        The only thing the two differ in: reading, dialect handling, thinning
        and guards are identical either way.
        """
        return to_gene_records(genes, offsets)

    def _tile(self, tid: TileId) -> list[AnnotationRecord]:
        ranges = list(self._info.canvas(tid.z).invert(tid.pos[0]))
        rows = self._rows(ranges)
        genes = build_genes(self._thin(rows), self._dialect)
        return self._records(genes, self._chromsizes.offsets)


def _overlaps(ranges: Sequence[GenomicRange]) -> pl.Expr:
    """Pushdown filter for the scan path. ``start``/``end`` are 1-based inclusive."""
    terms = [
        (pl.col("seqid") == gr.name)
        & (pl.col("end") > gr.start)
        & (pl.col("start") <= gr.end)
        for gr in ranges
        if not gr.is_out_of_bounds and gr.end > gr.start
    ]
    if not terms:
        return pl.lit(False)
    expr = terms[0]
    for term in terms[1:]:
        expr = expr | term
    return expr


class GffGenesTileset(GxfTileset):
    """A GFF3 file served as gene annotations. See :class:`GxfTileset`."""

    dialect: ClassVar[Dialect] = GFF3


class GtfGenesTileset(GxfTileset):
    """A GTF file served as gene annotations. See :class:`GxfTileset`."""

    dialect: ClassVar[Dialect] = GTF


class GxfTranscriptsTileset(GxfTileset):
    """The same file served one record per *transcript*.

    For ``horizontal-transcripts`` (the ``higlass-transcripts`` plugin), which
    draws each transcript on its own row with its own exons and codon markers,
    rather than one row per gene with the exons merged. Same payload kind, same
    reading -- a different ``fields`` layout and a finer granularity.
    """

    datatype: ClassVar[str] = "gene-transcripts"

    def _records(
        self, genes: dict[str, Gene], offsets: dict[str, int]
    ) -> list[AnnotationRecord]:
        return to_transcript_records(genes, offsets)


class GffTranscriptsTileset(GxfTranscriptsTileset):
    """A GFF3 file served as transcripts."""

    dialect: ClassVar[Dialect] = GFF3


class GtfTranscriptsTileset(GxfTranscriptsTileset):
    """A GTF file served as transcripts."""

    dialect: ClassVar[Dialect] = GTF


# --- Notes ------------------------------------------------------------------
#
# 1. Attributes are hoisted by oxbow, not parsed here. Legacy splits the raw
#    attribute string with `dict([x.split("=") for x in row[8].split(";")])`,
#    which raises on any attribute containing `=` in its value and cannot read
#    GTF's `key "value";` syntax at all. `attribute_defs` makes both formats
#    produce the same struct.
#
# 2. Linking is a second pass. Legacy attaches a child to its parent when it
#    first sees the child, so a child appearing before its parent is dropped.
#    GFF3 recommends but does not require parents first; GTF requires nothing.
#
# 3. Coordinates are 1-based inclusive on the wire and stay that way. Both
#    formats are 1-based, oxbow's default `coords="11"` preserves that, and the
#    models store `start`/`end` verbatim, as legacy does. Note this is the one
#    v2 module whose payload is not in absolute genome coordinates -- the gene
#    models carry `chrom` plus chromosome-relative positions.
#
# 4. `datatype = "gene-annotation"` is a guess, as with bam. clodius never
#    declares datatype; the server's tileset record does.
#
# 5. Not carried over: the `settings` dict, `feature_type` selection,
#    `random.random()` importance, and the whole bedlike rendering. All of it
#    lived in `single_tile`, which nothing calls.
#
# 6. The pydantic models are reused unchanged so the wire shape does not move.
#    They are, however, ten near-identical transcript classes differing only in
#    a `type` Literal -- a single class with a `type` field would do, and the
#    dispatch table above already makes them interchangeable. Worth revisiting
#    with `clodius/models/` generally.
#
#
# Thinning preserves gene integrity
# ---------------------------------
# The live legacy path has no cap at all; the dead one caps *rows*, which would cut
# exons away from their transcripts. Here the cap applies to **genes** -- the most
# important N genes and all of their children -- so a thinned tile is still a set
# of complete gene models. See :meth:`GxfTileset._thin`.
