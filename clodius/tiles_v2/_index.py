"""What a range query may legally ask an indexed file for.

A tile's query regions are derived from the tileset's chromsizes, but an index
only knows the contigs it was built over. A tabix index lists exactly the
contigs that carry records, so a chr1-only BED served against a whole assembly
is asked for contigs its index has never heard of -- and the reader raises,
through polars, as a ``ComputeError``. That is not a ``TileError``, so the
per-tile boundary cannot render it and one routine tile fails the whole batch.

Screening the regions against this set is a no-op on the records returned: a
contig the index does not list has no records to return. It only removes
queries that were guaranteed to match nothing and raise instead.
"""

from __future__ import annotations

import os

import pysam


def indexed_contigs(
    path: str | os.PathLike, index_path: str | os.PathLike | None = None
) -> frozenset[str] | None:
    """Contigs an indexed range query may name, or ``None`` if undetermined.

    ``None`` means "do not screen" rather than "no contigs" -- the caller
    leaves the regions alone and behaves as it did before. A format whose
    index this cannot read is no worse off than it was.

    Two routes, because the index formats differ. Tabix indexes text and lists
    the contigs actually present; a BCF carries a CSI whose keys come from the
    header, so the screen is a harmless no-op there.
    """
    try:
        return frozenset(pysam.TabixFile(os.fspath(path), index=index_path).contigs)
    except Exception:
        pass

    try:
        with pysam.VariantFile(os.fspath(path), index_filename=index_path) as vf:
            index = vf.index
            if index is not None:
                return frozenset(index.keys())
    except Exception:
        pass

    return None


def screen_regions(ranges, contigs: frozenset[str] | None) -> list:
    """The ranges an indexed query may name, given ``contigs``.

    ``contigs`` of ``None`` screens nothing. An empty result means the tile
    covers no contig the index carries, so it has no records -- the caller
    returns an empty tile rather than issuing a query that would raise.
    """
    if contigs is None:
        return list(ranges)
    return [gr for gr in ranges if gr.name in contigs]
