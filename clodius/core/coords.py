"""Primitives for source (data) and target (canvas) spatialc coodinates.

A Concatenated Genomic Coordinate System (CGCS) is defined by a
:class:`Chromsizes`, an ordered mapping of contigs and their lengths, which can
map between genomic ranges and CGCS coordinates.

A :class:`TileCanvas` defines the grid of bin-delimited tiles over a
(appropriately padded) CGCS, and can map between genomic ranges and tile
positions.
"""

from __future__ import annotations

import math
import re
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Iterable, Iterator, Literal

import bioframe
import numpy as np

from clodius.core.errors import TileOutOfBounds


@dataclass(frozen=True, slots=True)
class GenomicRange:
    """A range on a single chromosome.

    Internal representation is zero-based, half-open. Chromosome name is
    optional to represent out-of-bounds intervals.
    """

    cid: int
    name: str | None
    start: int
    end: int

    @property
    def is_out_of_bounds(self) -> bool:
        return self.name is None

    def as_tuple(self) -> tuple[str, int, int]:
        if self.name is None:
            raise ValueError("Out-of-bounds range")
        return (self.name, self.start, self.end)

    def to_ucsc(self, coords="11") -> str:
        """Return a UCSC-style string.

        Parameters
        ----------
        coords : {"01", "11"}, default "11"
            Coordinate convention to use. "01" is zero-based, half-open. "11"
            is one-based, fully-closed.

        Returns
        -------
        str
            A string of the form ``chr:start-end``. Raises ValueError if the
            coordinate convention is unrecognized.
        """
        if self.name is None:
            raise ValueError("Out-of-bounds range")
        match coords:
            case "01":
                return f"{self.name}:{self.start}-{self.end}"
            case "11":
                return f"{self.name}:{self.start + 1}-{self.end}"
            case _:
                raise ValueError(f"Invalid coordinate convention: {coords}")


class Chromsizes:
    """An ordered chromosome name -> length mapping, with derived views.

    Ordering matters: it defines a Concantenated Genomic Coordinate System (CGCS).
    """

    __slots__ = ("_names", "_lengths", "_offsets", "_bounds")

    def __init__(self, names: tuple[str, ...], lengths: tuple[int, ...]):
        if len(names) != len(lengths):
            raise ValueError(
                f"got {len(names)} names but {len(lengths)} lengths"
            )
        self._names = tuple(names)
        self._lengths = tuple(int(x) for x in lengths)
        self._offsets: dict[str, int] | None = None
        self._bounds = None

    @classmethod
    def from_pairs(cls, pairs) -> Chromsizes:
        names, lengths = zip(*pairs) if pairs else ((), ())
        return cls(tuple(names), tuple(int(x) for x in lengths))

    @classmethod
    def from_series(cls, series) -> Chromsizes:
        return cls(tuple(series.index), tuple(int(x) for x in series.values))

    @classmethod
    def from_assembly(
        cls,
        assembly: str,
        roles: list[str] | Literal["all"] | None,
        units: list[str] | Literal["all"] | None,
    ) -> Chromsizes:
        return cls.from_series(
            bioframe.assembly_info(assembly, roles, units).chromsizes
        )

    @classmethod
    def from_file(cls, path_or_handle) -> Chromsizes:
        cs = bioframe.read_chromsizes(
            path_or_handle, filter_chroms=False
        ).to_dict()
        key = cmp_to_key(_natcmp)
        return cls.from_pairs(sorted(cs.items(), key=lambda x: key(x[0])))

    def to_pairs(self) -> list[list]:
        return [
            [name, length] for name, length in zip(self._names, self._lengths)
        ]

    def to_series(self):
        import pandas as pd

        return pd.Series(self._lengths, index=self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __repr__(self) -> str:
        return f"<Chromsizes {len(self)} chroms, {self.total_length:,} bp>"

    @property
    def _boundaries(self):
        """``[0, len0, len0+len1, ..., total]``"""
        if self._bounds is None:
            self._bounds = np.concatenate(
                [[0], np.cumsum(self._lengths)], dtype=int
            )
        return self._bounds

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def lengths(self) -> tuple[int, ...]:
        return self._lengths

    @property
    def total_length(self) -> int:
        """Total length of the concatenated genome."""
        return sum(self._lengths)

    @property
    def offsets(self) -> dict[str, int]:
        """Cumulative start offset of each chromosome."""
        if self._offsets is None:
            offsets, running = {}, 0
            for name, length in zip(self._names, self._lengths):
                offsets[name] = running
                running += length
            self._offsets = offsets
        return self._offsets

    def transform(self, gr: GenomicRange) -> tuple[int, int]:
        """Convert a genomic range to canvas coordinates.

        Parameters
        ----------
        gr : GenomicRange
            A genomic range to transform.

        Returns
        -------
        tuple[int, int]
            A half-open absolute range in the concatenated coordinate space.

        Notes
        -----
        Raises KeyError if the chromosome is unknown, ValueError if the range is
        out-of-bounds.

        Examples
        --------
        e.g. with {"chr1": 100, "chr2": 200}
        ``(chr1, 10, 20) --> (10, 20)`
        ``(chr2, 10, 20) --> (110, 120)`
        """
        if gr.name is None:
            raise ValueError("Out-of-bounds range")
        offset = self.offsets.get(gr.name)
        if offset is None:
            raise KeyError(f"Unknown chromosome {gr.name!r}")
        return (offset + gr.start, offset + gr.end)

    def invert(self, span: tuple[int, int]) -> Iterator[GenomicRange]:
        """Split a half-open canvas range into per-chromosome intervals.

        Parameters
        ----------
        span : tuple[int, int]
            A half-open absolute range in the concatenated coordinate space.

        Returns
        -------
        Iterator[GenomicRange]
            A generator of :class:`GenomicRange` objects

        Notes
        -----
        Chromosome boundary behavior:
        * Half-open convention: a position exactly on a boundary belongs to
          the following chromosome -- so a span *ending* on one ends in the
          chromosome before it, and an empty span covers nothing at all.
        * Negative start raises.
        * If end extends past the last chromosome, a final interval is returned
          and flagged :attr:`~GenomicRange.is_out_of_bounds`.`

        Examples
        --------
        e.g. with {"chr1": 100, "chr2": 200}:
        ``(90, 150) --> [(chr1, 90, 100), (chr2, 0, 50)]``
        """
        start, end = span
        if start < 0:
            raise ValueError(f"negative start position: {start}")
        if end < start:
            raise ValueError(f"end {end} precedes start {start}")
        if end == start:
            return

        bounds = self._boundaries
        n = len(self._names)

        # `end - 1` is the last position the span covers
        cid_lo, cid_hi = (
            np.searchsorted(bounds, [start, end - 1], side="right") - 1
        )
        cid_lo, cid_hi = int(cid_lo), int(cid_hi)

        pos = start - bounds[cid_lo]

        for cid in range(cid_lo, cid_hi):
            # cid_hi is at most n, so the loop never reaches an invalid index
            yield GenomicRange(
                cid=cid,
                name=self._names[cid],
                start=int(pos),
                end=int(self._lengths[cid]),
            )
            pos = 0

        # Only when it covers something: a fractional span can truncate to
        # nothing, as can one ending exactly where this chromosome starts.
        tail_end = int(end - bounds[cid_hi])
        if tail_end > int(pos):
            yield GenomicRange(
                cid=cid_hi,
                name=self._names[cid_hi] if cid_hi < n else None,
                start=int(pos),
                end=tail_end,
            )


@dataclass(frozen=True, slots=True)
class TileCanvas:
    """
    Invertible scale map for a logical grid of tiles atop a uniformly binned CGCS.

    Maps genomic coordinates to tile positions and vice versa, at a single
    zoom level.

    Notes
    -----
    Each zoom level defines a complete partition of ``[0, max_width)`` into
    bins of exactly ``binsize``, not respecting chromosome boundaries. Layered
    on top of this bin partition, a tile is a window of ``tile_size``
    consecutive bins cut from it.

    The ``max_width`` is the length of the CGCS plus enough padding so that the
    tile width (``tile_size * binsize``) divides the range evenly into an whole
    number of tiles.

    For any tileset that actually bins the CGCS, ``max_width`` should satisfy
    the following:

    * The ``binsize`` is integral and evenly divides the range.
    * The tile width also evenly divides the range.

    Note that ``max_width`` may not be constant across zoom levels.
    """

    z: int
    binsize: float
    tile_size: int
    max_width: int
    chromsizes: Chromsizes | None = None

    def __post_init__(self):
        # TilesetInfo validates these upstream, but TileCanvas is re-exported
        # from clodius.core and can be constructed directly -- where a zero
        # tile size or binsize surfaced as ZeroDivisionError from a property,
        # and a negative extent constructed silently.
        # Inverted rather than `<= 0`: binsize is a float, and `nan <= 0` is
        # False, so the straightforward spelling admits NaN and lets it surface
        # as a ValueError from `n_bins` -- one step removed from the argument
        # that was actually wrong, which is the failure this guard exists to
        # prevent.
        if not self.binsize > 0:
            raise ValueError(f"binsize must be positive, got {self.binsize}")
        if self.tile_size <= 0:
            raise ValueError(
                f"tile_size must be positive, got {self.tile_size}"
            )
        if self.max_width < 0:
            raise ValueError(
                f"max_width must be non-negative, got {self.max_width}"
            )

    @property
    def span(self) -> tuple[int, int]:
        """Absolute ``[start, end)`` the whole canvas covers."""
        return (0, self.max_width)

    @property
    def n_bins(self) -> int:
        """Bins across the full canvas."""
        return int(self.max_width / self.binsize)

    @property
    def n_tiles(self) -> int:
        """Tiles needed to cover the canvas at this zoom."""
        return math.ceil(self.n_bins / self.tile_size)

    def _check_pos(self, x: int) -> None:
        """Raise unless tile ``x`` exists at this zoom."""
        if x < 0 or x >= self.n_tiles:
            raise TileOutOfBounds(
                f"tile position {x} is outside the {self.n_tiles} tiles at "
                f"zoom {self.z}"
            )

    def tile_span(self, x: int) -> tuple[int, int]:
        """Absolute ``[start, end)`` covered by tile ``x``.

        Raises
        ------
        TileOutOfBounds
            If ``x`` does not exist at this zoom; see :meth:`invert` for what
            counts as in range. The bound is checked here as well as there
            because the tilesets that work in absolute coordinates -- beddb,
            bed2ddb and the bigInteract pair -- call only this one, and an
            unchecked off-lattice position yields a well-formed range far past
            the genome, zero rows, and an empty tile a client cannot tell apart
            from "no annotations here".
        """
        self._check_pos(x)
        width = self.binsize * self.tile_size
        return (int(x * width), int((x + 1) * width))

    def transform(self, gr: GenomicRange) -> Iterator[int]:
        """Tile positions covering a genomic range."""
        if self.chromsizes is None:
            raise ValueError(
                "canvas has no chromsizes, so genomic ranges cannot be "
                "transformed to tile positions"
            )
        start, end = self.chromsizes.transform(gr)
        if end <= start:
            return

        width = self.binsize * self.tile_size
        # `end` is exclusive, so the last tile is the one holding `end - 1`.
        first = max(0, int(start // width))
        last = min(self.n_tiles - 1, int((end - 1) // width))
        yield from range(first, last + 1)

    def invert(self, x: int) -> Iterator[GenomicRange]:
        """Genomic intervals covered by tile ``x``.

        Validation is eager, unlike its sibling :meth:`transform`: this is an
        ordinary function returning a generator, not a generator function, so
        the exceptions below are raised at the call rather than on the first
        iteration. Callers rely on that to screen a position without consuming
        the result -- adding a ``yield`` to this body would silently defer the
        raise and disarm those checks.

        Raises
        ------
        TileOutOfBounds
            If ``x`` does not exist at this zoom. Tiles past the end of the
            *genome* but inside the canvas are in range -- that padding is
            what fills the trailing NaN bins of low-zoom tiles. Only positions
            past the end of the *canvas* are rejected.
        """
        if self.chromsizes is None:
            raise ValueError(
                "canvas has no chromsizes, so tile locations cannot be "
                "inverted to genomic intervals"
            )
        self._check_pos(x)
        return self.chromsizes.invert(self.tile_span(x))


_DIGITS = re.compile(r"(\d+)", re.U)


def _natcmp(x: str, y: str) -> int:
    if x.find("_") >= 0:
        x_parts = x.split("_")
        if y.find("_") >= 0:
            # chr_1 vs chr_2 -- compare the parts after the underscore
            return _natcmp(x_parts[1], y.split("_")[1])
        # chr_1 vs chr1 -- the plain name comes first
        return 1
    if y.find("_") >= 0:
        return -1

    for key in ("m", "y", "x"):
        # Deliberately reversed: whichever name matches first sorts later.
        if key in y.lower():
            return -1
        if key in x.lower():
            return 1

    x_parts = tuple(int(a) if a.isdigit() else a for a in _DIGITS.split(x) if a)
    y_parts = tuple(int(a) if a.isdigit() else a for a in _DIGITS.split(y) if a)

    try:
        if x_parts < y_parts:
            return -1
        if x_parts > y_parts:
            return 1
        return 0
    except TypeError:
        # Mixed int/str at the same position, e.g. ('chr', 1) vs ('chr', 'a')
        return 1


def natsorted(names: Iterable[str]) -> list[str]:
    """
    Sort chromosome names in natural genomic order.

    The rules, in order:

    1. A name containing ``_`` (an unplaced/alt contig such as
       ``chr1_KI270706v1_random``) is compared on the segment *after* the first
       underscore. Underscored names always sort after plain ones.
    2. Names containing ``m``, ``y`` or ``x`` are pushed to the end, in that
       reversed priority, giving ... chrX, chrY, chrM.
    3. Otherwise, split into digit and non-digit runs and compare as a tuple, so
       ``chr2`` precedes ``chr10``.

    Rule 2 is a substring test over the whole lowercased name, so it is only
    correct for conventional ``chrN`` naming -- ``chromosome1`` would trip the
    ``m`` test. Rule 1 saves the common case, since ``_random`` contigs are
    compared on their accession segment instead.
    """
    return sorted(names, key=cmp_to_key(_natcmp))
