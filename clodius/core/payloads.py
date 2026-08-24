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
they can be named, typed and checked.

Most of the shapes below have no ``tiles_v2`` producer yet. They are declared
anyway: :data:`PAYLOAD_TYPES` is the catalogue the conformance suite indexes,
and a kind with no declared shape is what the suite cannot check.

.. warning::

   Do not add ``from __future__ import annotations`` here. It stringifies every
   annotation, and ``TypedDict`` does not re-evaluate them, so ``NotRequired``
   becomes invisible at runtime: every shape below reported
   ``__optional_keys__ == frozenset()`` and ``DenseTilePayload`` claimed
   ``shape`` was required. Pydantic re-evaluates the strings and gets it right,
   which is what hid the defect -- but anything dispatching on those attributes
   directly demanded keys no producer emits.
"""

import base64
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, NotRequired, TypedDict, overload

import numpy as np


class TileKind(str, Enum):
    """Declared by each tileset so a server knows what it will get back."""

    DENSE = "dense"
    BEDLIKE = "bedlike"
    BEDLIKE_2D = "bedlike_2d"
    GENE_MODELS = "gene_models"
    COLUMNAR = "columnar"
    SUBS = "subs"
    SEQUENCE = "sequence"
    IMAGE = "image"
    POINTS = "points"


# --- A. dense ---------------------------------------------------------------
# Four incompatible encodings exist today. The one below is canonical;
# multivec/hitile/sequence_logos each reimplement it inline and disagree.
# Whether that divergence is client-visible is still open, so the variants are
# modelled faithfully rather than normalized away.


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

# As Python floats. Comparing a Python float against the numpy scalars directly
# casts the operand to float16, which emits `RuntimeWarning: overflow
# encountered in cast` for any value beyond that range -- on the very path
# whose job is to detect exactly those values.
_F16_MIN = float(_F16.min)
_F16_MAX = float(_F16.max)


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

    @overload
    def to_dict(self, *, stats: Literal[True] = ...) -> DenseTilePayload: ...

    @overload
    def to_dict(self, *, stats: Literal[False]) -> DenseTileShapedPayload: ...

    def to_dict(
        self, *, stats: bool = True
    ) -> DenseTilePayload | DenseTileShapedPayload:
        """Encode to the wire shape.

        ``stats=False`` omits ``size``, ``min_value`` and ``max_value``,
        producing :class:`DenseTileShapedPayload` -- what multivec and
        sequence_logos emit. The float-width choice is identical either way; only
        the emitted keys differ. Whether that divergence is client-visible or
        just drift is still open, so it is reproduced rather than normalized.

        float16 is used only when the data is NaN-free and lies entirely inside
        float16's range; anything else falls back to float32. That check is why
        ``dtype`` is per-tile rather than a property of the tileset.
        """
        data = self.values

        if data.size:
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
            and _F16_MIN < min_dense < _F16_MAX
            and _F16_MIN < max_dense < _F16_MAX
        )
        dtype = "float16" if fits_f16 else "float32"

        flat = data.flatten() if data.ndim else data

        payload: Any = {
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
    """2D annotation record. Users: bedpe, bed2ddb."""

    yStart: int
    yEnd: int


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


class ImageTile(TypedDict):
    """``imtiles.get_tiles`` with ``raw=True``: the stored blob, unencoded.

    Without ``raw`` the same module emits a base64 ``dense`` field instead, so
    :data:`PAYLOAD_TYPES` maps ``IMAGE`` to both.
    """

    image: bytes


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


# What each TileKind validates against in the conformance suite. Every member
# of TileKind appears, so a lookup keyed on a tileset's declared kind reports a
# shape rather than raising KeyError.
#
# TODO: dense has three live wire variants; collapse once it is settled whether
# the divergence is client-visible.
_DENSE_VARIANTS = (
    DenseTilePayload | DenseTileShapedPayload | DenseTileMinMaxPayload
)

PAYLOAD_TYPES: dict[TileKind, Any] = {
    TileKind.DENSE: _DENSE_VARIANTS,
    TileKind.BEDLIKE: BedlikeTile,
    TileKind.BEDLIKE_2D: Bedlike2DTile,
    TileKind.GENE_MODELS: GeneModelTile,
    TileKind.COLUMNAR: ColumnarTile,
    TileKind.SUBS: SubsTile,
    TileKind.SEQUENCE: SequenceTile,
    # imtiles emits the raw blob or a base64 `dense` field, depending on the
    # `raw` flag its caller passes.
    TileKind.IMAGE: ImageTile | DenseTilePayload,
    # density rasterizes points into a grid and formats it as a dense tile;
    # there is no distinct points-on-the-wire shape.
    TileKind.POINTS: DenseTilePayload,
}
