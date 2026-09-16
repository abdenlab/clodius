"""Tile payload shapes and constructors.

Wire shapes are ``TypedDict``s.

Some tiles go over the wire as a single JSON object:
- dense
- sequence
- reads (a JSON object of equal-length arrays - i.e., columnar table)

Other tiles go over the wire as a JSON array of record objects:
- annotation
- annotation_2d
- substitutions

Error tiles - objects carrying an `error` property - substitute for either.
"""

from __future__ import annotations

import base64
import warnings
from dataclasses import dataclass, field
from typing import Any, ClassVar, NotRequired, Sequence, TypedDict

import numpy as np


_F16 = np.finfo("float16")


# --- Constructors ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DenseTile:
    """A dense numeric tile, with its wire encoding attached.

    Parameters
    ----------
    values: 1D or 2D ndarray
        ``(n_bins,)`` or ``(n_bins, size)``. Flattened at encode time.
    size: int, optional
        Number of values per bin for a 1D multiplex tile. Inferred from
        ``values.shape[1]`` when omitted. Validated when given.
    shape: tuple[int, ...], optional
        Emitted as an extra field when set.
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

        Parameters
        ----------
        stats : bool, optional [default: True]
            If false, omits ``size``, ``min_value`` and ``max_value``.

        Returns
        -------
        DenseTilePayload

        Notes
        -----
        float16 is used only when the data is NaN-free and lies entirely inside
        float16's range; anything else falls back to float32.
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


@dataclass
class ReadsTile:
    """Alignment reads held as parallel columns.

    Columns rather than records because a tile holds thousands of reads.

    - ``from`` and ``tags.HP`` are not valid Python identifiers, so library
      code uses ``start``/``end``/``hp``.
    - ``start``/``end`` are **chromosome-relative**.

    Extra columns are passed through untouched.
    """

    COLUMNS: ClassVar[tuple[str, ...]] = (
        "id",
        "readName",
        "start",
        "end",
        "chrName",
        "chrOffset",
        "md",
        "cigars",
        "variants",
        "mapq",
        "strand",
        "is_paired",
        "first_seq",
        "last_seq",
        "hp",
    )

    WIRE_NAMES: ClassVar[dict[str, str]] = {
        "start": "from",
        "end": "to",
        "hp": "tags.HP",
    }

    columns: dict[str, list[Any]] = field(
        default_factory=lambda: {k: [] for k in ReadsTile.COLUMNS}
    )

    def extend(self, **columns: Sequence[Any]) -> None:
        """Append one chromosome's worth of reads, column by column."""
        for key, values in columns.items():
            self.columns.setdefault(key, []).extend(values)

    def __len__(self) -> int:
        return len(self.columns["id"])

    def to_dict(self) -> ReadsTilePayload:
        """Encode to the wire shape, validating the length invariant."""
        lengths = {k: len(v) for k, v in self.columns.items()}
        n = lengths.get("id", 0)
        ragged = {k: v for k, v in lengths.items() if v != n}
        if ragged:
            raise ValueError(
                f"reads tile has {n} reads but these columns disagree: "
                f"{ragged}. The client reads every column positionally against "
                f"the first one, so a short column is silently dropped."
            )
        return {self.WIRE_NAMES.get(k, k): v for k, v in self.columns.items()}  # type: ignore[return-value]


# --- Wire formats ------------------------------------------------------------


class ErrorTilePayload(TypedDict):
    """A tile that could not be served, in the slot where its data would go."""

    error: str
    error_type: NotRequired[str]


class DenseTilePayload(TypedDict):
    """The wire shape for dense tiles.

    Goes over the wire as a JSON object.

    ``dtype`` is float32 whenever the data contains NaN or exceeds float16's
    range.

    The shape of dense tiles differs by tileset/format legacy tree in some of
    the optional fields below:

    * ``shape`` advertises the dataset shape ``(n_rows, n_bins)`` so that the
      client can reconstruct the data from the C-order flattened payload.
      It is emitted by multivec, fasta, and sequence_logos.
    * ``size`` only appears to be used client-side by the ``range-track`` plugin,
      for displaying multivalued range modes.
    * ``min_value``/``max_value`` don't appear to be used by any client but are
      emitted in legacy code by a common tile formatter used by many tilesets.
    """

    dense: str  # base64 of the flattened array
    dtype: str  # "float16" | "float32"
    shape: NotRequired[list[int]]  # multivec, fasta
    size: NotRequired[int]  # values per bin: 2 or 4 for bigwig's range modes
    min_value: NotRequired[float | str]  # "NaN" when empty or all-NaN
    max_value: NotRequired[float | str]


class SequenceTilePayload(TypedDict):
    """Raw bases spanned by a tile's bounds."""

    sequence: str


ReadsTilePayload = TypedDict(
    "ReadsTilePayload",
    {
        "id": list[str],
        "readName": list[str],
        # Chromosome-relative coordinates of the aligned span, 0-based and
        # half-open. The client adds `chrOffset`.
        "from": list[int],
        "to": list[int],
        "chrName": list[str],
        "chrOffset": list[int],
        # MD tag verbatim; "" when the read has none.
        "md": list[str],
        # Per read: [(start, op, length), ...] from the CIGAR.
        "cigars": list[Any],
        # Mismatches against the reference, reconstructed from MD. Per read:
        # [(query_pos, ref_pos, query_base, ref_base), ...], with one entry per
        # read and [] for reads without MD.
        "variants": list[Any],
        "mapq": list[int],
        "strand": list[str],
        "is_paired": list[int],
        "first_seq": list[int],
        "last_seq": list[int],
        # Haplotype tag, 0 when absent. Dotted on the wire, not nested.
        "tags.HP": list[int],
    },
)


# --- Row-based record types --------------------------------------------------


class AnnotationRecord(TypedDict):
    """1D annotation record.

    Emitters include ``beddb``, ``bedfile``, ``bigbed``, ``vcf``, ``hibed``.
    """

    uid: str
    xStart: int
    xEnd: int
    fields: list[str]
    chrOffset: int
    importance: NotRequired[float]
    name: NotRequired[str]  # beddb version 3
    strand: NotRequired[str]  # gene annotations, transcripts; else fields[5]
    zoom: NotRequired[int]  # hibed


AnnotationTilePayload = list[AnnotationRecord]


class Annotation2DRecord(TypedDict):
    """2D annotation record

    Emitters include ``bed2ddb`` and ``bedpe``.

    A pair of intervals, x and y, which may be on different chromosomes.
    All four coordinates -- ``xStart``, ``xEnd``, ``yStart``, ``yEnd`` are
    absolute.

    Notes
    -----
    The required ``chrOffset`` field corresponds to the x interval, so the
    client can only use it to invert the x interval to chromosome-relative
    positions unless y is known to be on the same chromosome.
    """

    uid: str
    xStart: int
    xEnd: int
    yStart: int
    yEnd: int
    fields: list[str]
    chrOffset: int  # equivalent to xChrOffset
    importance: NotRequired[float]
    xChrOffset: NotRequired[int]
    yChrOffset: NotRequired[int]


Annotation2DTilePayload = list[Annotation2DRecord]


SubstitutionsRecord = TypedDict(
    "SubstitutionsRecord",
    {
        "id": str,
        # Reference coordinates of the aligned span, absolute.
        "from": int,
        "to": int,
        "substitutions": list[Any],
        "color": int,
        "extra": NotRequired[dict[str, Any]],
    },
)
SubstitutionsRecord.__doc__ = """All substitutions of one query sequence aligned to the reference.

Although this is retrieved by a call to `tiles()`, it is not really a
tile. The substitutions are global against the reference.
"""


SubstitutionsTilePayload = list[SubstitutionsRecord]


TileKind = (
    DenseTilePayload
    | ReadsTilePayload
    | SequenceTilePayload
    | AnnotationTilePayload
    | Annotation2DTilePayload
    | SubstitutionsTilePayload
    | ErrorTilePayload
)
