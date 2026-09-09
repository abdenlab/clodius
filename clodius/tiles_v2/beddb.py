"""beddb (.multires.db, 1D) rewritten against clodius.core.

The 1D sibling of ``tiles_v2/bed2ddb.py`` and the reference **STRATIFIED**
tileset: features are ranked and assigned a ``zoomLevel`` at aggregation time,
and a tile returns those with ``zoomLevel <= z``. Zoom is a priority threshold,
so density is bounded by construction -- no request-time cap, no subsampling,
and the ``importance`` on the wire is the real ranking that produced the zoom
levels rather than a stand-in for one.

This is the model ``bedfile``, ``bigbed`` and ``gff`` approximate with
``random.random()``. Worth reading alongside ``tiles_v2/bed.py`` to see what the
approximation costs.

Four schema versions
--------------------

The format has drifted, and a reader has to handle all of it. Versions are
distinguished by the presence of columns rather than declared consistently: a
``version`` column may be absent (meaning 1), an integer, or the *string*
``"3t"``.

=========  ==================================================================
version    how a tile is selected
=========  ==================================================================
1          ``zoomLevel <= z``, R-tree over ``(rStartPos, rEndPos)``
2          ``rStartZoomLevel <= z AND rEndZoomLevel >= 0`` -- the zoom range
           moves *into* the R-tree, so one index answers both the interval and
           the zoom question
3          as 2, plus a ``name`` column echoed into the payload
"3t"        no range query at all: a ``tiles`` table maps a linear tile id to
           interval ids, precomputed
=========  ==================================================================

``"3t"``'s tile id is ``sum(2**k for k in range(z)) + x`` -- the breadth-first
index of a node in a complete binary tree, i.e. ``2**z - 1 + x``. Only the
quadtree's own structure makes that well defined, which is worth noting: it is
the one place the implicit-ladder assumption is load-bearing rather than
descriptive.

:class:`_Selector` isolates the four, so the rest of the module is version-blind.

Fixed here
----------

- **The post-filter is gone.** Legacy re-tests each row in Python with
  ``x_start < tile_x_end and x_end >= tile_x_start`` after the query already
  selected on position. Harmless for one tile, but it silently disagrees with
  the SQL at the edges (the query is closed on both ends, the filter half-open
  on one), and it is what breaks ``bed2ddb``'s 1D path outright.
- **``"3t"`` no longer double-filters.** That version's ``tiles`` table *is* the
  tile assignment; applying a coordinate test afterwards can only drop records
  the aggregation step deliberately placed there.
- **One connection per tileset, not per tile**, and ``tileset_info`` read once.
  Legacy calls ``tileset_info`` inside ``get_1D_tiles``, so a 16-tile batch
  opens 32 connections and re-parses the info table 16 times.
- **The dead ``extra_zoom`` loop is gone.** ``extra_zoom = 0`` immediately
  before ``for j in range(2**extra_zoom)``, with ``new_rows = {}`` assigned and
  overwritten by ``new_rows = []`` on the next line.
"""

from __future__ import annotations

import os
from typing import ClassVar, Sequence

import apsw
import sosqlite

from clodius.core.coords import Chromsizes
from clodius.core.payloads import BedlikeTile, RegionRow, TileKind
from clodius.core.policies import DEFAULT_POLICY, DensityPolicy, TilePolicy
from clodius.core.tileid import TileId
from clodius.core.tileset import BaseTileset, TilesetInfo

_VFS = sosqlite.SmartOpenVFS(name="so-vfs-v2-beddb")


def to_bedlike(row: tuple, name: str | None = None) -> BedlikeTile:
    """One database row as the client's bedlike shape."""
    uid = row[5]
    if isinstance(uid, bytes):
        uid = uid.decode("utf8")
    record: BedlikeTile = {
        "uid": uid,
        "xStart": row[0],
        "xEnd": row[1],
        "chrOffset": row[2],
        "importance": row[3],
        "fields": row[4].split("\t"),
    }
    if name is not None:
        record["name"] = name
    return record


class _Selector:
    """Builds the version-appropriate query for one beddb file.

    Exists so the version check happens once at construction rather than per
    tile, and so the four variants sit next to each other instead of as four
    ``if version ==`` blocks repeated in ``get_1D_tiles`` and ``list_items``.
    """

    _BASE = "SELECT startPos, endPos, chrOffset, importance, fields, uid"

    def __init__(self, version):
        self.version = version
        self.has_name = version in (3, "3t")
        self.columns = self._BASE + (", name" if self.has_name else "")

    @property
    def precomputed_tiles(self) -> bool:
        """Whether tiles are stored rather than derived from a range query."""
        return self.version == "3t"

    def tile_query(self, z: int, lo: float, hi: float) -> str:
        if self.version == "3t":
            # Breadth-first index of node (z, x) in a complete binary tree.
            # `lo` carries the tile position for this variant; see `_tile`.
            raise RuntimeError("use tile_query_3t for precomputed tiles")
        return (
            f"{self.columns} FROM intervals, position_index "
            f"WHERE intervals.id = position_index.id "
            f"AND {self._zoom_clause(z)} "
            f"AND rEndPos > {lo} AND rStartPos < {hi}"
        )

    def tile_query_3t(self, z: int, x: int) -> str:
        tile_id = 2**z - 1 + x
        return (
            f"{self.columns} FROM intervals, tiles "
            f"WHERE tiles.id = {tile_id} AND tiles.intervalId = intervals.id"
        )

    def range_query(self, lo: int, hi: int) -> str:
        """Every record overlapping ``[lo, hi)``, regardless of zoom level.

        Backs :meth:`BedDbTileset.regions`. ``"3t"`` has no zoom column at all,
        so it falls back to the v1 clause -- which is what legacy does too, by
        never special-casing ``"3t"`` in ``list_items``.
        """
        zoom_clause = self._zoom_clause(1_000_000)
        return (
            f"{self.columns} FROM intervals, position_index "
            f"WHERE intervals.id = position_index.id "
            f"AND {zoom_clause} "
            f"AND rEndPos > {lo} AND rStartPos < {hi}"
        )

    def _zoom_clause(self, z: int) -> str:
        if self.version in (2, 3):
            # The zoom range lives in the R-tree, so this is answered by the
            # index rather than by a scan over `intervals`.
            return f"rStartZoomLevel <= {z} AND rEndZoomLevel >= 0"
        return f"zoomLevel <= {z}"


class BedDbTileset(BaseTileset):
    """A multires beddb served as 1D annotations.

    Parameters
    ----------
    path :
        The ``.multires.db``/``.beddb`` file. May be a URI: connections go
        through a smart_open VFS, so S3 and HTTP work.
    policy :
        Accepted for uniformity. Nothing in it applies -- density is bounded by
        the file's own stratification, and every query is indexed.
    """

    datatype: ClassVar[str] = "bedlike"
    ndim: ClassVar[int] = 1
    tile_kind: ClassVar[TileKind] = TileKind.BEDLIKE
    modifiers = None
    options = frozenset()

    density_policy: ClassVar[DensityPolicy] = DensityPolicy.STRATIFIED

    def __init__(
        self,
        path: str | os.PathLike,
        policy: TilePolicy = DEFAULT_POLICY,
    ):
        self._path = os.fspath(path)
        self._policy = policy
        self._conn: apsw.Connection | None = None
        self._info, self._selector = self._read_header()

    # --- resource lifetime --------------------------------------------------

    @property
    def conn(self) -> apsw.Connection:
        if self._conn is None:
            self._conn = apsw.Connection(
                self._path, vfs=_VFS.name, flags=apsw.SQLITE_OPEN_READONLY
            )
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # --- the protocol -------------------------------------------------------

    @property
    def policy(self) -> TilePolicy:
        return self._policy

    @property
    def version(self):
        return self._selector.version

    def chromsizes(self) -> Chromsizes:
        return self._info.coordinate_system

    def info(self) -> TilesetInfo:
        return self._info

    def tiles(self, ids: Sequence[TileId]):
        return [(tid, self._tile(tid)) for tid in ids]

    # --- ProvidesRegions ----------------------------------------------------

    def regions(self, offset: int, limit: int) -> tuple[list[RegionRow], bool]:
        """A page of records over the whole coordinate space.

        Legacy's ``list_items`` takes explicit start/end and applies
        ``max_entries`` as a bare ``LIMIT`` with no offset, so a caller cannot
        page. This matches the :class:`ProvidesRegions` contract instead, and
        reads ``limit + 1`` rows to answer "is there more" without a count.
        """
        span = self._info.max_pos[0]
        query = self._selector.range_query(0, span)
        query += f" LIMIT {limit + 1} OFFSET {offset}"
        rows = list(self.conn.cursor().execute(query))

        has_next = len(rows) > limit
        page = [to_bedlike(r) for r in rows[:limit]]
        return [
            {
                "uid": r["uid"],
                "xStart": r["xStart"],
                "xEnd": r["xEnd"],
                "fields": r["fields"],
                "chrOffset": r["chrOffset"],
            }
            for r in page
        ], has_next

    # --- internals ----------------------------------------------------------

    def _read_header(self) -> tuple[TilesetInfo, _Selector]:
        """Read ``tileset_info`` once, and settle the schema version.

        The table's columns vary by version, so this reads them by name rather
        than by index. Legacy indexes positionally (``row[5]``, ``row[6]``,
        ``row[7]``) and gets away with it only because every variant happens to
        keep the first eight columns in the same order.
        """
        cursor = self.conn.cursor().execute("SELECT * FROM tileset_info")
        row = cursor.fetchone()
        columns = [d[0] for d in cursor.getdescription()]
        header = dict(zip(columns, row))

        version = header.get("version", 1)
        if isinstance(version, str) and version.isdigit():
            version = int(version)

        names = tuple(header["chrom_names"].split("\t"))
        lengths = tuple(int(v) for v in header["chrom_sizes"].split("\t"))

        info = TilesetInfo(
            # One-based, as legacy declares it. Not reinterpreted here: it
            # ships with precomputed files. bed2ddb does the same.
            min_pos=[1],
            max_pos=[header["max_length"]],
            max_width=int(header["max_width"]),
            tile_size=int(header["tile_size"]),
            max_zoom=int(header["max_zoom"]),
            chromsizes=[[n, v] for n, v in zip(names, lengths)],
            assembly=header.get("assembly"),
            header=header.get("header", ""),
            version=version,
        )
        return info, _Selector(version)

    def _tile(self, tid: TileId) -> list[BedlikeTile]:
        selector = self._selector

        if selector.precomputed_tiles:
            # The stored assignment IS the answer. Legacy runs this query and
            # then applies the same coordinate post-filter as the other
            # versions, which can only discard records the aggregation step
            # deliberately put in this tile.
            query = selector.tile_query_3t(tid.z, tid.pos[0])
        else:
            lo, hi = self._info.canvas(tid.z).tile_span(tid.pos[0])
            query = selector.tile_query(tid.z, lo, hi)

        rows = self.conn.cursor().execute(query)
        records = [
            to_bedlike(r, name=r[6] if selector.has_name else None) for r in rows
        ]
        records.sort(key=lambda r: r["xStart"])
        return records


# --- Notes ------------------------------------------------------------------
#
# 1. Half-open overlap (`rEndPos > lo AND rStartPos < hi`) where legacy uses
#    `rEndPos >= lo AND rStartPos <= hi`. The closed form puts a record abutting
#    a tile edge into both adjacent tiles. Consistent now with bed.py, bedpe.py
#    and bed2ddb.py; it is a real behaviour change at exactly one base pair per
#    edge.
#
# 2. `name` is emitted for versions 3 and "3t". Legacy emits it for 3 only,
#    even though "3t" selects the column -- `to_add["name"] = r[6]` is guarded
#    by `if version == 3`, so a "3t" file fetches the name and drops it. Treated
#    as an oversight rather than a decision, since the column would not be in
#    the projection otherwise.
#
# 3. No `max_entries_per_tile`. STRATIFIED bounds density at aggregation time,
#    so a cap here would drop records the file ranked deliberately. `TilePolicy`
#    is accepted only so every tileset takes one.
#
# 4. Version detection is by column presence, and "3t" is a *string* where the
#    others are integers. `_read_header` normalizes a numeric string to int but
#    leaves "3t" alone, so `version` stays comparable to what legacy wrote.
#
# 5. `regions()` is a resgen extension (see `ProvidesRegions`). Legacy's
#    `list_items` has no offset, so it cannot page -- it applies `max_entries`
#    as a bare LIMIT. Reimplemented here against the documented contract rather
#    than reproduced.
#
# 6. `range_query` keeps legacy's "all zoom levels" trick: pass a zoom so large
#    that the clause is vacuously true, rather than omit the clause. Legacy uses
#    100000, this uses 1_000_000; both are arbitrary.
#
#    For v1 the clause is provably droppable -- `zoomLevel` is a plain column on
#    `intervals`, and with or without it the sample returns all 100 records.
#    For v2/v3 it is unverified: `rStartZoomLevel`/`rEndZoomLevel` are R-tree
#    dimensions, and whether `rEndZoomLevel >= 0` is a real filter (are negative
#    end levels meaningful?) or only narrows the index scan cannot be settled
#    without a v2/v3 file. There is no such sample in the repo.
#
#    So the sentinel stays until one turns up. Dropping it for v1 alone would
#    make the two branches diverge for a reason nobody could check.
