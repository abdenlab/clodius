"""FASTA served as sequence tiles.

Two tilesets:

- :class:`FastaSequenceTileset` -- ``{"sequence": "ACGT..."}``, datatype
  ``sequence``. resgen's ``fasta_seq`` filetype.
- :class:`FastaMultivecTileset` -- a dense 6-row one-hot encoding, datatype
  ``multivec_singleres_sequence``. resgen's ``fasta`` filetype.
"""

from __future__ import annotations

import os
from typing import ClassVar, Sequence

import numpy as np
import oxbow as ox

from clodius.core.coords import Chromsizes
from clodius.core.errors import TileError, TileTooWide
from clodius.core.tile import DenseTile, SequenceTilePayload, DenseTilePayload
from clodius.core.policies import TilePolicy
from clodius.core.tileid import TileId
from clodius.core.tileset import BaseTileset, TilesetInfo

TILE_SIZE = 1024

# A sequence tile is one base per column, so its payload grows with the span it
# covers -- a tile at zoom 0 would be the whole genome as a string. Unlike an
# alignment, this type cannot be left unbounded, so its default policy carries a
# span limit: eight tiles' worth of bases, which is legacy's three zoom levels.
DEFAULT_SEQUENCE_POLICY = TilePolicy().with_(max_span=TILE_SIZE * 8)

# Row order of the one-hot encoding, as legacy's `convert_bases_to_multivec`
# defines it. The sixth row catches everything else, including IUPAC ambiguity
# codes and soft-masked lowercase that is not acgtn.
BASES = "atgcn"
N_ROWS = len(BASES) + 1

# Lookup from byte value to row index, built once. Legacy does a dict lookup
# and a `.lower()` per base in Python, which for an 8192-base tile is 8192
# dict lookups and 8192 string allocations.
_ROW_OF_BYTE = np.full(256, N_ROWS - 1, dtype=np.uint8)
for _i, _b in enumerate(BASES):
    _ROW_OF_BYTE[ord(_b)] = _i
    _ROW_OF_BYTE[ord(_b.upper())] = _i


def chromsizes_from_fai(fai_path: str) -> Chromsizes:
    """Names and lengths from a ``.fai``, in file order.

    The first two columns; the rest -- byte offset, bases per line, bytes per
    line -- are what oxbow uses to seek and what legacy re-implemented by hand.
    File order matters: it defines the absolute coordinate space.
    """
    names, lengths = [], []
    with open(fai_path) as handle:
        for line in handle:
            if not line.strip():
                continue
            fields = line.split("\t")
            names.append(fields[0])
            lengths.append(int(fields[1]))
    if not names:
        raise ValueError(f"{fai_path} is empty")
    return Chromsizes(tuple(names), tuple(lengths))


def one_hot(sequence: str) -> np.ndarray:
    """``(6, len(sequence))`` one-hot encoding, rows ordered a,t,g,c,n,other."""
    codes = np.frombuffer(sequence.encode("ascii", "replace"), dtype=np.uint8)
    rows = _ROW_OF_BYTE[codes]
    out = np.zeros((N_ROWS, len(rows)), dtype=np.float32)
    out[rows, np.arange(len(rows))] = 1.0
    return out


class _FastaBase(BaseTileset):
    """Shared geometry, guards and sequence fetching.

    Parameters
    ----------
    path :
        The FASTA. Needs a ``.fai``; BGZF-compressed also needs a ``.gzi``.
    chromsizes :
        Optional. Defaults to the ``.fai``, which is what legacy reads through
        ``clodius.tiles.chromsizes``. Pass explicitly to impose a different
        chromosome *order*, which changes every absolute coordinate -- that is
        what legacy's ``chromsizes_fn`` argument is for.
    index_path :
        The ``.fai``, if not a sibling.
    policy :
        ``max_span`` bounds how many bases a tile may carry. Defaults to
        :data:`DEFAULT_SEQUENCE_POLICY`; passing a policy without one leaves
        the tileset unbounded.
    """

    ndim: ClassVar[int] = 1
    modifiers = None
    options = frozenset()

    def __init__(
        self,
        path: str | os.PathLike,
        chromsizes: Chromsizes | None = None,
        index_path: str | os.PathLike | None = None,
        policy: TilePolicy | None = None,
        tile_size: int = TILE_SIZE,
    ):
        self._path = os.fspath(path)
        self._index_path = (
            os.fspath(index_path) if index_path else self._path + ".fai"
        )
        self._chromsizes = chromsizes or chromsizes_from_fai(self._index_path)
        # Before `_build_info`, which advertises the span limit.
        self.policy = policy or DEFAULT_SEQUENCE_POLICY
        self.tile_size = tile_size
        self._info = self._build_info()

    # --- the protocol -------------------------------------------------------

    def chromsizes(self) -> Chromsizes:
        return self._chromsizes

    def info(self) -> TilesetInfo:
        return self._info

    def tiles(
        self, ids: Sequence[TileId], options=None
    ) -> list[tuple[TileId, SequenceTilePayload | DenseTilePayload]]:
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

    def _source(self, regions: list[str] | None):
        return ox.from_fasta(
            self._path, regions=regions, index=self._index_path
        )

    def _build_info(self) -> TilesetInfo:
        extra = {}
        if self.policy.max_span is not None:
            # Advertised so the client stops requesting, rather than collecting
            # refusals. Legacy said the same thing through higlass-server's
            # `MAX_FASTA_TILE_WIDTH`; saying it here means any host gets it.
            extra["max_tile_width"] = self.policy.max_span
        return TilesetInfo.quadtree(
            self._chromsizes,
            TILE_SIZE,
            # Declared by the class, not patched by the server afterwards.
            datatype=type(self).datatype,
            **extra,
        )

    def _check_span(self, tid: TileId) -> None:
        """Refuse a tile carrying more than ``max_span`` bases.

        Before any I/O: the bound is on the payload, not on the query. A
        sequence tile is one base per column, so its span in base pairs is also
        its length in characters.
        """
        limit = self.policy.max_span
        if limit is None:
            return
        span = self._info.tile_width(tid.z)
        if span > limit:
            raise TileTooWide(
                f"tile at zoom {tid.z} would carry {span} bases; this tileset "
                f"serves at most {limit}. Zoom in."
            )

    def _sequence(self, tid: TileId) -> str:
        """The tile's bases, concatenated across any chromosome boundary."""
        self._check_span(tid)

        ranges = [
            gr
            for gr in self._info.canvas(tid.z).invert(tid.pos[0])
            if not gr.is_out_of_bounds and gr.end > gr.start
        ]
        if not ranges:
            return ""

        frame = self._source([gr.to_ucsc(coords="11") for gr in ranges]).pl()
        # One row per requested region, in request order.
        return "".join(frame["sequence"])


class FastaSequenceTileset(_FastaBase):
    """A FASTA served as raw sequence strings, for the higlass-sequence track."""

    datatype: ClassVar[str] = "sequence"

    def _tile(self, tid: TileId) -> SequenceTilePayload:
        return {"sequence": self._sequence(tid)}


class FastaMultivecTileset(_FastaBase):
    """A FASTA served as a dense 6-row one-hot encoding."""

    datatype: ClassVar[str] = "multivec_singleres_sequence"

    def _tile(self, tid: TileId) -> DenseTilePayload:
        sequence = self._sequence(tid)
        values = one_hot(sequence)
        # `shape` is [6, n] and the array is already (6, n); `DenseTile`
        # flattens row-major at encode time, matching legacy's
        # `format_dense_tile(np.array(res).T)` followed by a shape patch.
        return DenseTile(values, shape=(N_ROWS, len(sequence))).to_dict()


# --- Notes ------------------------------------------------------------------
#
# 1. `datatype` is declared per class instead of patched by the caller. Legacy's
#    single `tileset_info` always says `multivec_singleres_sequence`, and
#    resgen overwrites it with `"sequence"` for the `fasta_seq` filetype. Any
#    other host serving `sequence_tiles` would advertise the wrong datatype and
#    the sequence track would not load.
#
# 2. The guard is a span in base pairs, like every other REFUSED type, rather
#    than a count of zoom levels. For sequence the two are the same statement --
#    one base per column, so `TILE_SIZE * 2**zoom_diff` bases is a span -- and
#    the span is the one the client understands.
#
# 3. A tile at the end of the genome is short rather than padded. The last tile
#    covers fewer bases than its span, so `shape` reports what is actually
#    there. Legacy's `fetch_sequence` raises `ValueError` on `end > seq_length`
#    instead, which the tile loop does not catch.
#
# 4. One-hot encoding is a table lookup over a byte array. Legacy allocates a
#    six-element Python list per base and calls `.lower()` on each -- for the
#    8192-base tile the zoom guard permits, that is 8192 dict lookups and as
#    many string allocations, per tile.
#
# 5. Upper and lower case map to the same row. Legacy lowercases first, so
#    soft-masked regions are indistinguishable from unmasked ones. Preserved,
#    but worth noting: the sixth row could carry masking if anyone wanted it.
