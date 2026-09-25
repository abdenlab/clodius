"""Polars expressions shared across the record-oriented tilesets.

Kept out of :mod:`clodius.core` deliberately: nothing under ``core`` imports
polars today, and a protocol layer that costs a polars import to touch is a
worse trade than a small module here.
"""

from __future__ import annotations

import polars as pl

from clodius.core.coords import Chromsizes


def known_chroms(chromsizes: Chromsizes, *columns: str) -> pl.Expr:
    """Rows whose contig columns all name a contig in ``chromsizes``.

    Build once per tileset, not per tile: the chromsizes are fixed for the
    tileset's life, and the polars Series behind ``is_in`` is not free to
    rebuild. A row naming an unknown contig has no offset and cannot be placed,
    so it is dropped before the cap rather than after -- one that consumed a
    cap slot would silently shorten the tile.
    """
    names = list(chromsizes.offsets)
    expr = pl.col(columns[0]).is_in(names)
    for column in columns[1:]:
        expr = expr & pl.col(column).is_in(names)
    return expr
