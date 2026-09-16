"""Alignments (indexed SAM, BAM, or CRAM) served as columnar alignment tiles.

A tile is a batch of reads with per-read CIGAR substitutions and MD-derived
variants (mismatches), returned as parallel arrays rather than a list of
objects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, ClassVar, Sequence

import oxbow as ox
import polars as pl

from clodius.core.coords import Chromsizes, natsorted
from clodius.core.errors import TileError, TilesetUnavailable, TileTooWide
from clodius.core.tile import ReadsTile, ReadsTilePayload
from clodius.core.policies import TilePolicy
from clodius.core.tileid import TileId
from clodius.core.tileset import BaseTileset, TilesetInfo

# Legacy's value. Smaller than every other type's 1024 because a read is far
# bigger than an annotation, and this sets how much genome a tile covers.
TILE_SIZE = 256

# Fixed fields worth decoding. `qual` and `tlen` are never read, and `seq` is
# only needed to reconstruct variants.
FIELDS = [
    "qname",
    "flag",
    "rname",
    "pos",
    "mapq",
    "cigar",
    "rnext",
    "pnext",
    "seq",
    "end",
]

# MD carries the reference mismatches; HP is the haplotype to color by.
TAG_DEFS = [("MD", "Z"), ("HP", "i")]

# How far beyond the tile to look for mates in the first pass.
MATE_EXTENSION = 500

# Secondary (0x100) and supplementary (0x800) alignments are excluded: a
# secondary alignment omits `seq`, so variants cannot be reconstructed for it.
_NOT_PRIMARY = 0x100 | 0x800

# --- the three sources ------------------------------------------------------
#
# oxbow normalizes BAM, SAM and CRAM to the same columns and tags, and its three
# readers take the same arguments. Only three things differ, so they are data
# rather than branches -- the same shape as :class:`~clodius.tiles_v2.gxf.Dialect`.


@dataclass(frozen=True)
class AlignmentFormat:
    """What differs between an indexed BAM, SAM and CRAM."""

    name: str
    reader: Callable
    # CRAM carries its own container compression and takes no such argument.
    compression: str | None
    # Sibling index extensions, most conventional first.
    index_suffixes: tuple[str, ...]
    # CRAM stores bases as differences against a reference, so without one the
    # `seq` column comes back empty and no variant can be reconstructed.
    needs_reference: bool = False

    def open(self, path: str, index: str, reference: str | None, **kwargs):
        if self.compression is not None:
            kwargs["compression"] = self.compression
        if self.needs_reference:
            kwargs["reference"] = reference
        return self.reader(path, index=index, **kwargs)


BAM = AlignmentFormat("bam", ox.from_bam, "bgzf", (".bai", ".csi"))
# BGZF-compressed and indexed, like the others. A plain or gzipped SAM cannot be
# range-queried, and scanning a whole alignment file per tile is never the right
# answer -- so it is refused rather than silently served slowly.
SAM = AlignmentFormat("sam", ox.from_sam, "bgzf", (".tbi", ".csi"))
CRAM = AlignmentFormat(
    "cram", ox.from_cram, None, (".crai",), needs_reference=True
)


# --- CIGAR and MD decoding ---------------------------------------------
#
# Read's CIGAR and MD tag and reconstruct what the aligner saw, so a tile can
# ship the mismatches instead of the sequence.


def get_cigar_substitutions(pos, query_length, cigartuples):
    """The read's structural differences from the reference, from its CIGAR.

    Mismatches (``X``), indels (``I``/``D``), skips (``N``) and the clipped
    ends (``S``/``H``). ``M``/``=`` advance the cursor and emit nothing; an
    insertion emits without advancing it, since it consumes no reference.

    Parameters
    ----------
    pos :
        Where the aligned read starts. Emitted positions are in whatever frame
        this is given in -- 0-based and chromosome-relative, as called here.
    query_length :
        Length of the aligned span on the reference, used only to place the
        trailing clip.
    cigartuples :
        ``(op, length)`` pairs, as :func:`parse_cigar_string` returns them.

    Returns
    -------
    list[tuple[int, str, int]]
        ``(start, op, length)`` per difference.
    """
    subs = []
    curr_pos = 0

    cigartuples = cigartuples
    readstart = pos
    readend = pos + query_length

    for ctuple in cigartuples:
        if ctuple[0] == "X":
            subs.append((readstart + curr_pos, "X", ctuple[1]))
            curr_pos += ctuple[1]
        elif ctuple[0] == "I":
            subs.append((readstart + curr_pos, "I", ctuple[1]))
        elif ctuple[0] == "D":
            subs.append((readstart + curr_pos, "D", ctuple[1]))
            curr_pos += ctuple[1]
        elif ctuple[0] == "N":
            subs.append((readstart + curr_pos, "N", ctuple[1]))
            curr_pos += ctuple[1]
        elif ctuple[0] == "M" or ctuple[0] == "=":
            curr_pos += ctuple[1]

    if len(cigartuples):
        first_ctuple = cigartuples[0]
        last_ctuple = cigartuples[-1]

        if first_ctuple[0] == "S":
            subs.append((readstart - first_ctuple[1], "S", first_ctuple[1]))
        if first_ctuple[0] == "H":
            subs.append((readstart - first_ctuple[1], "H", first_ctuple[1]))

        if last_ctuple[0] == "S":
            subs.append((readend + 1, "S", last_ctuple[1]))
        if last_ctuple[0] == "H":
            subs.append((readend, "H", last_ctuple[1]))

    return subs


def parse_cigar_string(cigar):
    """Split a CIGAR string into its operations.

    Parameters
    ----------
    cigar :
        e.g. ``"30M2I50M"``. Anything falsy or not a string yields ``[]``.

    Returns
    -------
    list[tuple[str, int]]
        ``(op, length)`` pairs, operation first.
    """
    if not cigar or not isinstance(cigar, str):
        return []
    parts = []
    curr = 0
    for c in cigar:
        if c.isnumeric():
            curr = curr * 10 + int(c)
        else:
            parts += [(c, curr)]
            curr = 0
    return parts


def reconstruct_ref(seq, md, cigar):
    """Reconstruct a reference sequence that has the insertions from the query.

    The reason we can't exclude the insertions is that they are encoded for in
    the CIGAR string so we would need to use that to remove them.

    Two passes: the CIGAR lays out the alignment, seeding the reference with the
    read's own bases, then the MD tag overwrites the positions where the two
    actually differ. Soft-clipped bases are dropped.

    Parameters
    ----------
    seq :
        The read sequence as stored, soft-clipped bases included.
    md :
        The read's MD tag, which names the reference bases at mismatches and,
        after ``^``, at deletions.
    cigar :
        The read's CIGAR string.

    Returns
    -------
    tuple[str, str]
        The aligned ``(reference, read)`` pair, equal in length and ready for
        :func:`variants_list`. ``-`` marks an insertion in the reference and a
        deletion in the read.
    """
    i_seq = 0
    i_md = 0
    match_count = 0
    deletion = False
    num = 0

    new_seq = ""
    ref_seq = []

    # go through the cigar and remove the ignored bases
    for i_cig in range(len(cigar)):
        if cigar[i_cig].isnumeric():
            # getting the number of bases the upcoming operation applies to
            num = num * 10 + int(cigar[i_cig])
        else:
            op = cigar[i_cig]
            # print("op", num, op, 'iseq:', i_seq)
            if op == "S":
                i_seq += num
            elif op == "I":
                ref_seq += ["-"] * num
                new_seq += seq[i_seq : i_seq + num]
                i_seq += num
            elif op == "M":
                new_seq += seq[i_seq : i_seq + num]
                ref_seq += list(seq[i_seq : i_seq + num])
                i_seq += num
            elif op == "D":
                ref_seq += ["N"] * num
                new_seq += "-" * num

            num = 0
            i_cig += 1

    # print(ref_seq)
    # print(new_seq)

    i_seq = 0

    i_ref = 0

    # let's iterate over the entire md string
    for i_md in range(len(md)):
        # if we encounter a numeric value then we keep track of what it is
        if md[i_md].isnumeric():
            match_count = match_count * 10 + int(md[i_md])
            # We're definitely not in a deletion if we're in a numeric number
            deletion = False
        else:
            # Add the matches that we've gone over
            # If we've been going over a deletion or mismatches, then match_count will be 0
            # ref += seq[i_seq : i_seq + match_count]
            i_ref += match_count
            # print("readding", i_seq, match_count)
            # print(oseq)
            # print('--------')
            # print(ref)
            # print('==========')

            i_seq += match_count
            match_count = 0

            if md[i_md] == "^":
                # We're starting a deletion sequence
                deletion = True
            else:
                # A letter can indicate that we're either encountering a deletion
                # or a mistmatch

                if deletion:
                    # It's a deletion in the reference
                    # ref += md[i_md]
                    ref_seq[i_ref] = md[i_md]
                    i_ref += 1
                else:
                    # It's a mismatch, add the MD letter and skip the sequence letter
                    # ref += md[i_md]
                    ref_seq[i_ref] = md[i_md]
                    i_ref += 1
                    i_seq += 1

    # Add the last match_count stretch
    # ref += seq[i_seq : i_seq + match_count]
    # print("readding", i_seq, match_count)
    # print(oseq)
    # print('--------')
    # print(ref)
    # print('==========')
    return "".join(ref_seq), new_seq


def variants_list(ref, seq):
    """Get a list of variants that are in seq relative to ref.

    Both are aligned strings of equal length, gaps included. Positions count
    ungapped bases, so they index the read and the reference respectively.

    Parameters
    ----------
    ref :
        The aligned reference. ``-`` marks a base the read inserts, ``N`` marks
        a base the read deletes.
    seq :
        The aligned read, same length as ``ref``, with ``-`` where the read
        deletes.

    Returns
    -------
    list[tuple[...]]
        A list of 0-based ``(query_pos, ref_pos, query_base, ref_base)`` tuples,
        one per mismatch.
    """
    variants = []

    assert len(seq) == len(ref)

    ref_pos = 0
    seq_pos = 0

    for i in range(len(seq)):
        if ref[i] == "-":
            seq_pos += 1
            continue
        if seq[i] == "-":
            ref_pos += 1
            continue

        seq_pos += 1
        ref_pos += 1

        if seq[i] != ref[i]:
            variants += [(seq_pos - 1, ref_pos - 1, seq[i], ref[i])]

    return variants


class AlignmentTileset(BaseTileset):
    """An indexed alignment file served as columnar read tiles.

    Prefer :class:`BamTileset`, :class:`SamTileset` or :class:`CramTileset`,
    which differ only in their :class:`AlignmentFormat`.

    Parameters
    ----------
    path :
        The alignment file. Must be BGZF-compressed and indexed -- there is no
        scan fallback, because a whole-file scan for one tile of reads is never
        the right answer. That applies to SAM as much as to BAM.
    chromsizes :
        Optional. Defaults to the header's reference list, natural-sorted the
        way legacy does it. Pass explicitly to fix a different order, which
        changes every absolute coordinate.
    index_path :
        Defaults to ``<path>`` plus the format's conventional suffix -- ``.bai``
        for BAM, ``.tbi`` for SAM, ``.crai`` for CRAM -- falling back to any
        other suffix the format allows that exists on disk.
    reference :
        The FASTA a CRAM was compressed against. Required for CRAM and rejected
        for the others: without it the ``seq`` column decodes empty and no
        mismatch can be reconstructed.
    policy :
        ``max_span`` is advertised (as ``max_tile_width``) *and* enforced
        here, unlike legacy.
    """

    ndim: ClassVar[int] = 1
    datatype: ClassVar[str] = "reads"
    modifiers = None
    options = frozenset()

    format: ClassVar[AlignmentFormat] = BAM

    def __init__(
        self,
        path: str | os.PathLike,
        chromsizes: Chromsizes | None = None,
        index_path: str | os.PathLike | None = None,
        reference: str | os.PathLike | None = None,
        policy: TilePolicy | None = None,
        tile_size: int = TILE_SIZE,
    ):
        fmt = type(self).format
        self._path = os.fspath(path)
        self._index_path = (
            os.fspath(index_path) if index_path else self._default_index_path()
        )
        if not os.path.exists(self._index_path):
            raise TilesetUnavailable(
                f"no index for {self._path}: looked for "
                f"{', '.join(self._path + x for x in fmt.index_suffixes)}. "
                f"A {fmt.name.upper()} must be BGZF-compressed and indexed to "
                "be served."
            )

        if fmt.needs_reference and reference is None:
            raise TilesetUnavailable(
                f"a {fmt.name.upper()} needs the reference it was compressed "
                "against; without it reads decode without sequence."
            )
        if reference is not None and not fmt.needs_reference:
            raise TilesetUnavailable(
                f"a {fmt.name.upper()} carries its own sequences; a reference "
                "is only meaningful for CRAM."
            )
        self._reference = os.fspath(reference) if reference else None
        self.policy = policy or TilePolicy()
        self.tile_size = tile_size

        header = fmt.open(
            self._path,
            self._index_path,
            self._reference,
            fields=FIELDS,
            tag_defs=[],
        )
        if chromsizes is None:
            sizes = dict(header.chrom_sizes)
            # Natural order, matching legacy: chr1, chr2, ..., chr10, not
            # chr1, chr10, chr2. This defines the absolute coordinate space, so
            # it is not cosmetic.
            names = tuple(natsorted(sizes))
            chromsizes = Chromsizes(names, tuple(int(sizes[n]) for n in names))
        self._chromsizes = chromsizes
        self._info = self._build_info()

    # --- the protocol -------------------------------------------------------

    def chromsizes(self) -> Chromsizes:
        return self._chromsizes

    def info(self) -> TilesetInfo:
        return self._info

    def tiles(
        self, ids: Sequence[TileId], options=None
    ) -> list[tuple[TileId, ReadsTilePayload]]:
        """One entry per requested id, always."""
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

    def _build_info(self) -> TilesetInfo:
        extra = {}
        if self.policy.max_span is not None:
            # The client can compute `max_width / 2**z` and compare against it.
            # to pre-emptively reject zoom levels.
            extra["max_tile_width"] = self.policy.max_span
        return TilesetInfo.quadtree(self._chromsizes, self.tile_size, **extra)

    def _default_index_path(self) -> str:
        """The conventional sibling index, or another the format allows."""
        suffixes = type(self).format.index_suffixes
        for suffix in suffixes:
            candidate = self._path + suffix
            if os.path.exists(candidate):
                return candidate
        return self._path + suffixes[0]

    def _source(self, regions: list[str]):
        return type(self).format.open(
            self._path,
            self._index_path,
            self._reference,
            fields=FIELDS,
            tag_defs=TAG_DEFS,
            regions=regions,
        )

    def _fetch(self, regions: list[str]) -> pl.DataFrame:
        """Primary alignments in ``regions``, as a materialized frame."""
        if not regions:
            return pl.DataFrame(
                schema=self._source([]).pl(lazy=True).collect_schema()
            )
        return (
            self._source(regions)
            .pl(lazy=True)
            .filter((pl.col("flag") & _NOT_PRIMARY) == 0)
            .collect(engine="streaming")
        )

    def _with_mates(self, df: pl.DataFrame, chrom: str) -> pl.DataFrame:
        """Add the missing mate of any half-paired read, in one extra query.

        Legacy resolves mates in a ``while`` loop that issues a fresh range
        query per unpaired read and re-derives the outstanding set each pass.
        Here the outstanding mate positions are collected once and fetched as a
        single multi-region query, which is the same answer in one round trip.

        Mates on another chromosome are ignored, as in legacy: the client draws
        a pair as one span, and a span across chromosomes has no extent.
        """
        if df.is_empty():
            return df

        paired = df.filter((pl.col("flag") & 1) != 0)
        if paired.is_empty():
            return df

        firsts = set(paired.filter((pl.col("flag") & 64) != 0)["qname"])
        lasts = set(paired.filter((pl.col("flag") & 128) != 0)["qname"])
        incomplete = firsts ^ lasts
        if not incomplete:
            return df

        wanted = paired.filter(
            pl.col("qname").is_in(list(incomplete))
            & (pl.col("rnext").cast(pl.String) == chrom)
        )
        if wanted.is_empty():
            return df

        regions = sorted(
            {f"{chrom}:{int(p)}-{int(p) + 1}" for p in wanted["pnext"] if p}
        )
        extra = self._fetch(regions).filter(
            pl.col("qname").is_in(list(incomplete))
        )
        if extra.is_empty():
            return df

        # Deduplicate on read identity. Two sources of repeats: oxbow yields a
        # read once per region it matches, and the mate windows overlap; and a
        # mate may already have arrived in the widened first pass. `qname` alone
        # is not identity -- the two ends of a pair share it -- so the flag and
        # position are part of the key.
        return pl.concat([df, extra]).unique(
            subset=["qname", "flag", "pos"], keep="first", maintain_order=True
        )

    def _check_width(self, tid: TileId) -> None:
        limit = self.policy.max_span
        if limit is None:
            return
        width = self._info.tile_width(tid.z)
        if width > limit:
            raise TileTooWide(
                f"tile spans {int(width)} bp; this tileset serves at most "
                f"{limit}. Zoom in."
            )

    def _tile(self, tid: TileId) -> ReadsTilePayload:
        # Before any I/O -- the point of REFUSED is not reading the data.
        self._check_width(tid)

        tile = ReadsTile()
        offsets = self._chromsizes.offsets

        for gr in self._info.canvas(tid.z).invert(tid.pos[0]):
            if gr.is_out_of_bounds or gr.end <= gr.start:
                continue

            # Widened so most mates arrive in the first query. `gr.start` is
            # zero-based; BAM regions are one-based inclusive.
            lo = max(1, gr.start + 1 - MATE_EXTENSION)
            hi = gr.end + MATE_EXTENSION
            df = self._fetch([f"{gr.name}:{lo}-{hi}"])

            # Reads actually overlapping the tile, then their mates.
            df = df.filter(
                (pl.col("pos") - 1 < gr.end) & (pl.col("end") > gr.start)
            )
            df = self._with_mates(df, gr.name)

            # Extend, never assign: legacy's bug is that a second chromosome
            # replaces the first.
            _extend(tile, df, offsets[gr.name])

        # Renames `start`/`end`/`hp` to their wire spellings and refuses a
        # ragged tile.
        return tile.to_dict()


def _extend(tile: ReadsTile, df: pl.DataFrame, chrom_offset: int) -> None:
    """Append one chromosome's reads to the columnar tile, in place."""
    if df.is_empty():
        return

    rows = df.to_dicts()
    n = len(rows)

    flags = [r["flag"] for r in rows]
    is_paired = [f & 1 for f in flags]
    first_seq = [f & 64 for f in flags]
    tags = [r.get("tags") or {} for r in rows]
    mds = [t.get("MD") or "" for t in tags]

    tile.extend(
        readName=(r["qname"] for r in rows),
        id=(
            r["qname"] if not paired else f"{r['qname']}_{1 if first else 2}"
            for r, paired, first in zip(rows, is_paired, first_seq)
        ),
        start=(chrom_offset + r["pos"] - 1 for r in rows),
        end=(chrom_offset + r["end"] for r in rows),
        chrName=(str(r["rname"]) for r in rows),
        chrOffset=[chrom_offset] * n,
        mapq=(r["mapq"] for r in rows),
        strand=("-" if f & 16 else "+" for f in flags),
        is_paired=is_paired,
        first_seq=first_seq,
        last_seq=(f & 128 for f in flags),
        hp=(t.get("HP") or 0 for t in tags),
        md=mds,
        cigars=(
            get_cigar_substitutions(
                r["pos"] - 1,
                r["end"] - r["pos"],
                parse_cigar_string(r["cigar"]),
            )
            for r in rows
        ),
        # One entry per read, including [] for reads without MD. Legacy emits a
        # single [] for the whole tile, so the column is ragged.
        variants=(
            variants_list(*reconstruct_ref(r["seq"], md, r["cigar"]))
            if md
            else []
            for r, md in zip(rows, mds)
        ),
    )


class BamTileset(AlignmentTileset):
    """An indexed BAM (``.bai``/``.csi``)."""

    format: ClassVar[AlignmentFormat] = BAM


class SamTileset(AlignmentTileset):
    """A BGZF-compressed, tabix-indexed SAM (``.tbi``/``.csi``).

    A plain or gzipped SAM is refused: oxbow cannot range-query one.
    """

    format: ClassVar[AlignmentFormat] = SAM


class CramTileset(AlignmentTileset):
    """An indexed CRAM (``.crai``), served against its reference FASTA."""

    format: ClassVar[AlignmentFormat] = CRAM


# --- Notes ------------------------------------------------------------------
#
# 1. No query-size estimate. Legacy reads bytes from the BAI
#    (`est_query_size_ix`) and refuses above a hardcoded `4e6` -- a third
#    distinct limit, after vcf's 450000 and tabix's 1000000. The three were
#    never reconciled, no client knows about any of them, and `max_span` --
#    which legacy discarded and this one enforces -- is the bound that actually
#    matters. So `TilePolicy` no longer carries one.
#
# 2. Mate resolution is one extra query, not a loop of them. Same answer for
#    mates within the widened window plus one round trip for the rest; legacy
#    issues a range query per unpaired read and recomputes the outstanding set
#    each pass. Mates on other chromosomes are ignored in both.
#
# 3. `datatype = "reads"` is a guess. clodius never declares it -- the server's
#    tileset record does -- so nothing here depends on it being right, but it
#    should be checked against what the pileup track expects before the
#    boundary starts using `datatype` to pick a default track.
#
# 4. No unindexed path. Every other v2 module falls back to a scan; a BAM does
#    not, because scanning gigabytes to answer one 16 kb tile is not a
#    degraded answer but a wrong one. Missing index raises at first fetch.
#
# 5. `ColumnarTile` in clodius.core.payloads does not match what this emits: it
#    omits `from` and `to`. `from` is a Python keyword, so a class-syntax
#    TypedDict cannot declare it -- the functional form can. Worth fixing when
#    the payload models are next touched.
