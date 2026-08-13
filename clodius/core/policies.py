"""Server policy, lifted out of the tileset modules.

Every field below is a constant hardcoded somewhere in ``clodius/tiles/`` today.
Defaults preserve current behavior, so adopting this is a no-op; the point is
that the knobs become configurable by construction instead of requiring an edit
to library source.

Note the current values are mutually inconsistent in two places, preserved here
as separate fields until someone decides which is right:

- max query size is 1_000_000 in ``tabix.py:282`` but 450_000 in ``vcf.py:152``
- max entries per tile is 1024 (bedfile), 512 (bedpe), 4096 (hibed) for the
  identical bedlike payload
"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class TilePolicy:
    """Limits applied when serving tiles.

    ``None`` means "no limit". Frozen -- derive variants with :meth:`with_`.
    """

    # Refuse tiles spanning more than this much coordinate space.
    # bam.py:559, vcf.py:90 (advertised as ``max_tile_width`` in tileset_info).
    # Declared, not yet consumed: no tiles_v2 tileset enforces a width limit.
    max_tile_width: int | None = None

    # Cap on records returned per tile. bedfile.py:263 (overridable today via
    # the untyped ``settings["MAX_BEDFILE_ENTRIES"]``). Read by bed and bigbed.
    max_entries_per_tile: int | None = 1024

    # Estimated bytes a tabix/bai range query may touch. tabix.py:282.
    # Declared, not yet consumed: no tiles_v2 tileset estimates query size.
    max_query_size: int | None = 1_000_000

    # Refuse to whole-file-load an unindexed file larger than this.
    # bedfile.py:89, bedpe.py:100. Read by bed.
    max_unindexed_filesize: int | None = 20_000_000

    # fasta.py:174 -- refuse sequence tiles more than N levels out.
    # Declared, not yet consumed: fasta has no tiles_v2 tileset.
    max_zoom_diff: int | None = 3

    # bigbed.py:222 -- chromosomes sampled per tile when a genome has many.
    # Read by bigbed.
    max_chroms_sampled: int | None = 128

    def with_(self, **changes) -> TilePolicy:
        """Return a copy with ``changes`` applied."""
        return replace(self, **changes)


# Applied when a caller supplies nothing. Matches today's hardcoded behavior.
DEFAULT_POLICY = TilePolicy()


class GridPolicy(str, Enum):
    """
    How to resample chromosome-segmented rasters onto a genome-spanning grid.
    """

    # Concatenate chromosome segments in order, dropping trailing bins whenever
    # the accumulated over-representation reaches a threshold.
    #
    # The only policy implemented today. A positional SCATTER alternative --
    # place each chromosome-level bin at the global index its absolute start
    # falls in -- lands with the PR that needs it, along with the tile-relative
    # grid machinery it requires.
    SEQUENTIAL = "sequential"


class DensityPolicy(str, Enum):
    """How a tileset keeps a tile from returning unboundedly many records."""

    # Importance is precomputed at aggregation time and each feature assigned a
    # zoom level; a tile returns features with `zoomLevel <= z`. Zoom is a
    # priority threshold, and density is bounded by construction. The designed
    # model. beddb, bed2ddb, hibed.
    STRATIFIED = "stratified"

    # No aggregation step exists for the format, so everything overlapping the
    # range is fetched and thinned to a cap at request time. bedfile, bedpe,
    # gff, bigbed.
    #
    # This is STRATIFIED without the precomputation, which is why these types
    # emit `importance = random.random()`: there is no ranking to report. Left
    # unseeded it also re-rolls per request, so a feature can appear at one zoom
    # and vanish at the next. `stable_importance` is the cheap repair.
    SUBSAMPLED = "subsampled"

    # No thinning. Guard on tile width or estimated query size and return an
    # error when the request is too large; the client is expected not to ask
    # until zoomed in far enough. bam, vcf.
    REFUSED = "refused"


def stable_importance(key: str) -> float:
    """A deterministic pseudo-importance in ``[0, 1)``, derived from content.

    A stand-in for a real precomputed ranking, for formats that have none. It is
    still arbitrary -- but *stably* arbitrary, which is the part that matters:

    - the same feature gets the same value on every request, so tiles are
      reproducible and cacheable;
    - the ranking is identical at every zoom, so a feature that survives
      thinning at one level survives at the next, instead of flickering.

    Uses the same digest the record's ``uid`` already comes from, so no new
    hashing is introduced.
    """
    digest = hashlib.md5(key.encode("utf8")).hexdigest()
    return int(digest[:8], 16) / 0x1_0000_0000


def take_most_important(
    records: Sequence[T],
    cap: int | None,
    importance: Callable[[T], float],
) -> list[T]:
    """The ``cap`` most important records, in their original order.

    ``cap`` of ``None`` means no limit, matching :class:`TilePolicy`. A ``cap``
    of zero or less returns nothing: ``ranked[-0:]`` is the whole list, so
    without this guard a server configured to serve no records would emit an
    unbounded tile -- the precise failure the cap exists to prevent.

    Deterministic, unlike ``random.choices``, which additionally samples *with
    replacement* and so can return the same record twice while dropping another
    entirely.

    Order is preserved rather than sorted by importance: the client positions
    records by coordinate, and keeping genomic order makes the output easier to
    diff against the unthinned set.
    """
    if cap is None:
        return list(records)
    if cap <= 0:
        return []
    if len(records) <= cap:
        return list(records)

    ranked = sorted(range(len(records)), key=lambda i: importance(records[i]))
    keep = set(ranked[-cap:])
    return [r for i, r in enumerate(records) if i in keep]
