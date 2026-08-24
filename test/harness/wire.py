"""Decoding the wire payloads a dense tileset emits.

``DenseTile.to_dict`` base64-encodes its array and reports the dtype it chose,
so a test cannot read a tile without undoing both. This was copied into
``test/core/test_payloads.py`` and ``test/tiles/test_conformance.py``
independently; the second copy also asserts the payload is not an error, which
is the version worth keeping.

``dtype`` is per-tile, not per-tileset: ``to_dict`` picks float16 whenever the
data is NaN-free and inside float16's range and float32 otherwise, so a tile
overhanging the genome end decodes at a different width than its interior
sibling. Reading the width off the payload is therefore mandatory, not
defensive.
"""

import base64

import numpy as np


def decode(payload):
    """The payload's ``dense`` field as a flat array at its declared dtype."""
    assert "error" not in payload, payload.get("error")
    return np.frombuffer(
        base64.b64decode(payload["dense"]), dtype=payload["dtype"]
    )


def n_bins(payload):
    """Bins in a dense tile.

    ``size`` is values *per bin* -- 2 for bigwig's ``minMax``, 4 for
    ``whisker`` -- so the decoded length is a multiple of the bin count rather
    than equal to it. Payloads that hand-roll their formatting omit ``size``
    entirely, and those carry one value per bin.
    """
    return len(decode(payload)) // payload.get("size", 1)


def square(payload):
    """A 2D dense payload as its square matrix.

    Cooler ships ``tile_size**2`` values with no ``shape`` field, so the side
    has to be recovered from the length.
    """
    flat = decode(payload)
    side = int(round(len(flat) ** 0.5))
    assert side * side == len(flat), f"{len(flat)} values is not square"
    return flat.reshape(side, side)
