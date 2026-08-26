"""Error hierarchy for the tile-serving layer.

The convention is **raise internally, translate at the server boundary**.

Library code raises these exceptions. The boundary layer (a HiGlass server, or
the back-compat shims in ``clodius.tiles.*``) catches them and renders them into
whichever ``{"error": ...}`` position the client expects for that endpoint:

- per-tile   -> ``[(tile_id, {"error": str(exc)}), ...]``
- whole-info -> ``{"error": str(exc)}`` in place of a tileset info

Doing the translation at the boundary rather than in each tileset module is what
makes it structurally impossible to drop sibling tiles from a batch, which is the
current bug in ``vcf.py:195``, ``bam.py:628`` and ``bam_pysam.py:359`` (they
``return`` from inside the per-tile loop).
"""


class TileError(Exception):
    """Base for every error the tile layer raises deliberately."""


class TilesetUnavailable(TileError):
    """The tileset cannot be served at all.

    Translated into a whole-``tileset_info`` error payload.

    Current sites: ``bedfile.py:70,90`` (no chromsizes / file >20MB),
    ``bedpe.py:88,101``, ``imtiles.py:8``.
    """


class MalformedTileId(TileError):
    """A tile id could not be parsed against the tileset's declared shape."""


class UnsupportedModifier(TileError):
    """The tile id carries a modifier this tileset does not declare."""


class UnsupportedOption(TileError):
    """The tile id carries a ``,key:value`` option this tileset does not declare."""


class TileOutOfBounds(TileError):
    """The requested position does not exist at this zoom level.

    NOTE: today this is silently swallowed -- ``cooler.generate_tiles`` simply
    ``continue``s, so the caller gets a short result list rather than an error.
    Whether to preserve that or start raising is an open decision; see
    conformance check #5 (batch integrity).
    """


# ``TileTooWide`` and ``TileTooLarge`` land with the PR that enforces
# ``TilePolicy.max_tile_width`` and ``max_query_size``. Nothing raises them
# today, so they are not declared yet.
