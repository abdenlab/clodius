"""Canonical synthetic genomes shared by every test module.

Two genomes, deliberately distinct, because they answer different questions.

``CANONICAL_CHROMSIZES`` is for anything that *tiles*. Its total sits inside a
narrow window that makes a fully-past-genome tile position exist at the deepest
zoom -- see :func:`assert_canonical_geometry`. ``MINIMAL_CHROMSIZES`` is for
coordinate arithmetic that never tiles, where small round numbers make the
expected tuples readable.

Before this module the name ``TINY`` meant both, with different values, in
different files: ``test/core/test_coords.py`` and ``test/core/test_tileset.py``
used the 350 bp genome while ``test/tiles_v2/test_bed.py`` used the 3000 bp one.
Same identifier, same three chromosome names, incompatible geometry.
"""

from clodius.core.coords import Chromsizes

#: Tiling genome. Total 3000 bp; at ``TILE_SIZE`` 1024 this gives ``max_zoom``
#: 2, ``max_width`` 4096, and four tiles at the deepest zoom -- of which
#: ``x=3`` spans ``[3072, 4096)`` and so lies entirely past the genome.
CANONICAL_CHROMSIZES = [["c1", 1000], ["c2", 1500], ["c3", 500]]
CANONICAL_TOTAL = 3000

#: Coordinate-arithmetic genome. Total 350 bp, which collapses to
#: ``max_zoom == 0`` -- a single tile covering everything. Correct for span
#: inversion, useless for anything that asks for a tile.
MINIMAL_CHROMSIZES = [["c1", 100], ["c2", 200], ["c3", 50]]
MINIMAL_TOTAL = 350

#: Ordering genome. The same 3000 bp total and the same geometry window as
#: CANONICAL, but the three orders an implementation could plausibly return are
#: three distinct permutations: file order (c2, c1, c10), natural order
#: (c1, c2, c10), and lexicographic order (c1, c10, c2). CANONICAL cannot tell
#: them apart -- ``c1/c2/c3`` in declaration order is a fixed point of all
#: three -- so an assertion about ordering is unfalsifiable against it, which
#: is how ``natsorted`` survived being replaced by both ``sorted`` and ``list``.
#:
#: Two digits are what does the work. Any single-digit set makes natural and
#: lexicographic order agree.
SHUFFLED_CHROMSIZES = [["c2", 1500], ["c1", 1000], ["c10", 500]]

#: File, natural and lexicographic order of :data:`SHUFFLED_CHROMSIZES`, spelled
#: out so a test asserts a literal rather than re-deriving one from the sorter
#: it is testing.
SHUFFLED_FILE_ORDER = ("c2", "c1", "c10")
SHUFFLED_NATURAL_ORDER = ("c1", "c2", "c10")
SHUFFLED_LEXICOGRAPHIC_ORDER = ("c1", "c10", "c2")

#: The tile width every implicit-ladder tileset declares.
TILE_SIZE = 1024


def canonical():
    """The tiling genome as a :class:`Chromsizes`."""
    return Chromsizes.from_pairs(CANONICAL_CHROMSIZES)


def minimal():
    """The coordinate-arithmetic genome as a :class:`Chromsizes`."""
    return Chromsizes.from_pairs(MINIMAL_CHROMSIZES)


def shuffled():
    """The ordering genome as a :class:`Chromsizes`.

    Note that ``Chromsizes.from_pairs`` preserves the order given, so this
    keeps file order -- which is the point. ``from_file`` sorts.
    """
    return Chromsizes.from_pairs(SHUFFLED_CHROMSIZES)


def assert_canonical_geometry():
    """Guard the narrow window that makes the canonical genome useful.

    A tile lying wholly past the genome exists at ``max_zoom`` only while
    ``TILE_SIZE * 2**(max_zoom - 1) < total <= TILE_SIZE * 2**max_zoom -
    TILE_SIZE``. Below 2049 the ladder collapses to one zoom level and there is
    no out-of-bounds position at any zoom; above 3072 the last tile straddles
    the genome end instead of clearing it. 3000 is not a round number waiting
    to be tidied up.
    """
    assert 2048 < CANONICAL_TOTAL <= 3072, (
        f"CANONICAL_TOTAL={CANONICAL_TOTAL} leaves no wholly-past-genome tile; "
        "it must stay in (2048, 3072]"
    )


def past_genome_tile(info):
    """The deepest-zoom tile position lying entirely past the genome.

    For **implicit** ladders only -- those declaring ``max_zoom`` and
    ``max_width``, where ``max_width`` is a power-of-two multiple of the tile
    size and therefore overshoots the genome. Explicit-ladder tilesets have no
    such position at any zoom; use :func:`overhanging_tile` for those.
    """
    z = info.max_zoom
    canvas = info.canvas(z)
    total = info.coordinate_system.total_length
    for x in range(canvas.n_tiles):
        if canvas.tile_span(x)[0] >= total:
            return z, x
    raise AssertionError(
        f"no wholly-past-genome tile at zoom {z}: the canvas ends at "
        f"{canvas.span[1]} and the genome at {total}"
    )


def overhanging_tile(info):
    """The deepest-zoom tile that starts inside the genome and ends past it.

    For **explicit** ladders -- cooler and multivec -- where
    ``TilesetInfo.canvas`` sizes the extent to hug the genome
    (``ceil(total / tile_span) * tile_span``), so the last tile always starts
    inside it. What exists there is a partial overhang, which is what exercises
    the padding path.
    """
    z = info.num_zoom_levels - 1
    canvas = info.canvas(z)
    total = info.coordinate_system.total_length
    for x in reversed(range(canvas.n_tiles)):
        lo, hi = canvas.tile_span(x)
        if lo < total < hi:
            return z, x
    raise AssertionError(
        f"no overhanging tile at zoom {z}: the genome length {total} falls on "
        "a tile boundary, so no tile is partially padded"
    )
