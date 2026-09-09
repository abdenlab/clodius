"""Tile payload shapes, and constructors for the ones with real encoding work.

Two layers here, deliberately:

**Wire shapes** are ``TypedDict``s -- ``DenseTilePayload``, ``BedlikeTile``,
``ColumnarTile`` and friends. They describe exactly what goes over the wire and
cost nothing at runtime. Assertability comes from the conformance suite::

    from pydantic import TypeAdapter
    TypeAdapter(DenseTilePayload).validate_python(payload)

which keeps validation off the hot path -- a bedlike tile carries up to 1024
records and a batch is dozens of tiles, so validating our own output on every
request buys no safety.

**Constructors** are classes, for the payloads that involve real work rather
than a dict literal. Today that is only :class:`DenseTile`, which has to pick a
float width, base64-encode, and compute min/max. The others are assembled
record-by-record at the call site and gain nothing from a wrapper.

The wire shapes cannot be unified -- the client dispatches on track type -- but
they can be named, typed and checked. See section 1.3 of
_scratch/clodius-contracts-and-interface.md.
"""

from __future__ import annotations

import base64
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, NotRequired, TypedDict

import numpy as np


class TileKind(str, Enum):
    """Declared by each tileset so a server knows what it will get back."""

    DENSE = "dense"
    BEDLIKE = "bedlike"
    BEDLIKE_2D = "bedlike_2d"
    PAIRED = "paired"
    GENE_MODELS = "gene_models"
    COLUMNAR = "columnar"
    SUBS = "subs"
    SEQUENCE = "sequence"
    IMAGE = "image"
    POINTS = "points"


# --- A. dense ---------------------------------------------------------------
# Four incompatible encodings exist today. The one below is canonical;
# multivec/hitile/sequence_logos each reimplement it inline and disagree.
# Whether that divergence is client-visible is still open (section 2.6), so the
# variants are modelled faithfully rather than normalized away.


class DenseTilePayload(TypedDict):
    """The canonical dense wire shape, as produced by :class:`DenseTile`.

    ``dtype`` is chosen per tile at runtime -- float32 whenever the data
    contains NaN or exceeds float16's range -- so the client must read it
    rather than assume.
    """

    dense: str  # base64 of the flattened array
    dtype: str  # "float16" | "float32"
    size: int  # values per bin: 1, or 2/4 for bigwig's range modes
    min_value: float | str  # "NaN" when the tile is empty or all-NaN
    max_value: float | str
    shape: NotRequired[list[int]]  # multivec, fasta


class DenseTileShapedPayload(TypedDict):
    """``multivec.tiles:59`` and ``sequence_logos``: no size/min_value/max_value."""

    dense: str
    dtype: str
    shape: list[int]


class DenseTileMinMaxPayload(TypedDict):
    """``hitile.tiles:333``: per-bin mins/maxs alongside the values, always float32.

    The float16 branch is commented out at ``hitile.py:305-322``.
    """

    dense: str
    mins: str
    maxs: str
    dtype: str


_F16 = np.finfo("float16")


@dataclass(frozen=True, slots=True)
class DenseTile:
    """A dense numeric tile, with its wire encoding attached.

    Replaces bare calls to ``clodius.tiles.format.format_dense_tile``, whose
    output this reproduces exactly. The difference is that ``size`` can be
    *declared* rather than recovered from the array shape: a caller that already
    knows how many statistics it asked for per bin (bigwig's minMax is 2,
    whisker is 4) can say so instead of relying on the array being 2D.

    Parameters
    ----------
    values
        ``(n_bins,)`` or ``(n_bins, size)``. Flattened at encode time.
    size
        Values per bin. Inferred from ``values.shape[1]`` when omitted, which is
        what ``format_dense_tile`` always did. Validated against the array when
        given, so a mismatch fails loudly instead of producing a tile the client
        will misread.
    shape
        Emitted as an extra field when set. multivec and fasta send it; bigwig
        and cooler do not.
    """

    values: np.ndarray
    size: int | None = None
    shape: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.size is None:
            return
        actual = self.values.shape[1] if self.values.ndim == 2 else 1
        if self.size != actual:
            raise ValueError(
                f"declared size {self.size} does not match array carrying "
                f"{actual} value(s) per bin (shape {self.values.shape})"
            )

    @property
    def values_per_bin(self) -> int:
        if self.size is not None:
            return self.size
        return self.values.shape[1] if self.values.ndim == 2 else 1

    @property
    def n_bins(self) -> int:
        return self.values.shape[0] if self.values.ndim else 0

    def to_dict(self, *, stats: bool = True) -> DenseTilePayload:
        """Encode to the wire shape.

        ``stats=False`` omits ``size``, ``min_value`` and ``max_value``,
        producing :class:`DenseTileShapedPayload` -- what multivec and
        sequence_logos emit. The float-width choice is identical either way; only
        the emitted keys differ. Whether that divergence is client-visible or
        just drift is still open (main doc section 2.6), so it is reproduced
        rather than normalized.

        float16 is used only when the data is NaN-free and lies entirely inside
        float16's range; anything else falls back to float32. That check is why
        ``dtype`` is per-tile rather than a property of the tileset.
        """
        data = self.values

        if len(data):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", r"All-NaN (slice|axis) encountered"
                )
                min_dense = float(np.nanmin(data))
                max_dense = float(np.nanmax(data))
        else:
            min_dense = max_dense = np.nan

        fits_f16 = (
            not bool(np.isnan(data).any())
            and _F16.min < min_dense < _F16.max
            and _F16.min < max_dense < _F16.max
        )
        dtype = "float16" if fits_f16 else "float32"

        flat = data.flatten() if data.ndim else data

        payload: DenseTilePayload = {
            "dense": base64.b64encode(flat.astype(dtype)).decode("utf-8"),
            "dtype": dtype,
        }
        if stats:
            payload["min_value"] = (
                min_dense if not np.isnan(min_dense) else "NaN"
            )
            payload["max_value"] = (
                max_dense if not np.isnan(max_dense) else "NaN"
            )
            payload["size"] = self.values_per_bin
        if self.shape is not None:
            payload["shape"] = list(self.shape)
        return payload


# --- B/C. bedlike -----------------------------------------------------------


class BedlikeTile(TypedDict):
    """1D annotation record. Users: beddb, bedfile, bigbed, vcf, hibed."""

    uid: str
    xStart: int
    xEnd: int
    chrOffset: int
    importance: float
    fields: list[str]
    name: NotRequired[str]  # beddb version 3
    zoom: NotRequired[int]  # hibed


class Bedlike2DTile(BedlikeTile):
    """2D annotation record, as ``bed2ddb`` and ``bedarcsdb`` emit it.

    Inherits a single ``chrOffset``, which is what those two produce
    (``bed2ddb.py:150``, ``bedarcsdb.py:144``).

    That single offset is **anchor 1's**, not a claim that the anchors share a
    chromosome. Aggregation resolves the two chromosomes independently
    (``cli/aggregate.py:231-242``), so ``xStart``/``yStart`` are correct
    absolute coordinates even for an inter-chromosomal pair; only
    ``chrOffset = xs[0] - start1`` is stored, and anchor 2's offset is dropped.
    A consumer can therefore recover anchor 1's chromosome-relative position but
    not anchor 2's, unless the two happen to coincide. Both sample databases in
    ``test/sample_data`` are entirely intra-chromosomal -- loops and domains
    always are -- which is why the gap is invisible in practice.

    ``bedpe`` does not match this shape at all: it emits ``xChrOffset`` and
    ``yChrOffset`` and no ``chrOffset`` (``bedpe.py:122``). So the producers of
    "2D bedlike" disagree on the key that locates the record, and no single
    TypedDict covers both. :class:`BedpeTile` models the second; which one the
    client actually reads is section 2.6 (open questions).
    """

    yStart: int
    yEnd: int


class PairedTile(Bedlike2DTile):
    """A paired-interval record: two anchors that may be on different chromosomes.

    Extends :class:`Bedlike2DTile` rather than replacing it, so the wire shape
    is a superset of what ``bed2ddb`` already emits. That nesting is what makes
    it safe to serve to existing tracks:

    - ``Annotations2dTrack`` reads ``uid``/``xStart``/``yStart``/``xEnd``/
      ``yEnd``/``fields``/``importance`` and no offsets at all, so the extra
      keys are inert to it.
    - ``Arcs1DTrack`` (higlass-arcs) reads ``xStart``, then ``yStart`` or
      ``xEnd``, and falls back to ``chrOffset + fields[n]`` -- which it uses
      unconditionally when the ``startField``/``endField`` options are set.

    ``chrOffset`` is therefore required, and is the *x* anchor's offset,
    matching ``bed2ddb``. Legacy ``bedpe`` omits it and emits only
    ``xChrOffset``/``yChrOffset`` (``bedpe.py:122``), so a bedpe tileset in an
    arcs track with ``startField`` set computes ``undefined + n`` and yields
    NaN coordinates. Carrying all three keys fixes that without removing
    anything a consumer might read.

    ``xChrOffset``/``yChrOffset`` are kept because they carry what ``chrOffset``
    structurally cannot: anchor 2's chromosome, for inter-chromosomal pairs.
    Nothing reads them today.
    """

    xChrOffset: int
    yChrOffset: int


# --- D. gene models ---------------------------------------------------------


class GeneModelTile(TypedDict):
    """``gff.tiles``. Values are ``model_dump()``ed pydantic models from
    :mod:`clodius.models.gff_models`."""

    genes: list[dict[str, Any]]
    transcripts: dict[str, Any]


# --- E. columnar ------------------------------------------------------------


class ColumnarTile(TypedDict):
    """Alignment reads as parallel arrays. Users: bam, bam_pysam, cram.

    Note the dotted key (``tags.HP``), which is why this cannot be a dataclass.
    """

    id: list[str]
    readName: list[str]
    chrName: list[str]
    chrOffset: list[int]
    md: list[str]
    cigars: list[Any]
    variants: list[Any]
    mapq: list[int]
    strand: list[str]
    is_paired: list[int]
    first_seq: list[int]
    last_seq: list[int]


# --- F. one-offs ------------------------------------------------------------


class SubsTile(TypedDict):
    """``pileup``: one aligned sequence."""

    id: str
    substitutions: list[Any]
    color: int
    extra: NotRequired[dict[str, Any]]


class SequenceTile(TypedDict):
    """``fasta.sequence_tiles``."""

    sequence: str


class RegionRow(TypedDict):
    """One row from :meth:`~clodius.core.tileset.ProvidesRegions.regions`.

    Emitted by ``bedfile.regions:337`` and ``vcf.regions:41``. The resgen-app
    ``RegionsList`` component reads only ``uid`` (React key), ``fields``
    (destructured as ``[chrom, start, end]``, with ``fields.slice(3)`` shown as
    tab-joined detail) and ``xStart``/``xEnd`` (to navigate). ``chrOffset`` is
    emitted by both producers but read by nobody.
    """

    uid: str
    xStart: int
    xEnd: int
    fields: list[str]
    chrOffset: NotRequired[int]


class ErrorTile(TypedDict):
    """What the boundary renders a :class:`~clodius.core.errors.TileError` into.

    Library code should never construct this directly -- raise instead.
    """

    error: str


# What each TileKind validates against in the conformance suite.
# TODO: dense has three live wire variants; collapse once section 2.6 is settled.
PAYLOAD_TYPES: dict[TileKind, Any] = {
    TileKind.DENSE: DenseTilePayload,
    TileKind.BEDLIKE: BedlikeTile,
    TileKind.BEDLIKE_2D: Bedlike2DTile,
    TileKind.PAIRED: PairedTile,
    TileKind.GENE_MODELS: GeneModelTile,
    TileKind.COLUMNAR: ColumnarTile,
    TileKind.SUBS: SubsTile,
    TileKind.SEQUENCE: SequenceTile,
}
