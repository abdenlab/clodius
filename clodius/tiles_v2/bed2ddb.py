from __future__ import annotations

import os
from typing import Iterator, Sequence

import apsw
import sosqlite

from clodius.core.coords import Chromsizes
from clodius.core.tile import Annotation2DRecord
from clodius.core.policies import TilePolicy, LinkPolicy
from clodius.core.tileid import TileId
from clodius.core.tileset import BaseTileset, TilesetInfo

# A VFS that opens through smart_open, so a tileset can live on S3/HTTP rather
# than the local filesystem. Module-level because registering it is global and
# idempotent per name.
_VFS = sosqlite.SmartOpenVFS(name="so-vfs-v2")

# The columns every query projects, in the order `to_bedlike2d` expects.
_COLUMNS = "fromX, toX, fromY, toY, chrOffset, importance, fields, uid"


def to_bedlike2d(row: tuple) -> Annotation2DRecord:
    """One database row as the client's 2D bedlike shape.

    ``uid`` arrives as ``bytes`` or ``str`` depending on how the file was
    written, so it is normalized here rather than at four call sites.

    Note this is :class:`Annotation2DRecord`, not the ``Annotation2DRecord`` the BEDPE
    tilesets emit: the format stores a single ``chrOffset`` (anchor 1's) and
    never records anchor 2's chromosome, so ``yChrOffset`` cannot be filled in
    honestly. See the class docstring in clodius.core.payloads.
    """
    uid = row[7]
    if isinstance(uid, bytes):
        uid = uid.decode("utf8")
    return {
        "uid": uid,
        "xStart": row[0],
        "xEnd": row[1],
        "yStart": row[2],
        "yEnd": row[3],
        "chrOffset": row[4],
        "importance": row[5],
        "fields": row[6].split("\t"),
    }


def overlaps(prefix: str, lo: float, hi: float) -> str:
    """R-tree overlap test for one axis, as a SQL fragment.

    ``prefix`` is ``"X"`` or ``"Y"``. Half-open on the right to match every
    other v2 module; legacy uses ``>=``/``<=`` on both ends, which admits a
    record abutting the tile's far edge.
    """
    return f"rTo{prefix} > {lo} AND rFrom{prefix} < {hi}"


class _Bed2ddbBase(BaseTileset):
    """Shared connection handling, geometry and row mapping."""

    modifiers = None
    options = frozenset()

    # No request-time thinning: the zoom level IS the density control.

    def __init__(
        self,
        path: str | os.PathLike,
        policy: TilePolicy | None = None,
        tile_size: int | None = None,
    ):
        self._path = os.fspath(path)
        self._conn: apsw.Connection | None = None
        self._info = self._build_info()
        self.policy = policy or TilePolicy()
        self.tile_size = tile_size or self._info.tile_size

    # --- resource lifetime --------------------------------------------------

    @property
    def conn(self) -> apsw.Connection:
        """A read-only connection, opened once and reused.

        Legacy opens a fresh connection per tile *and* a second one for
        ``tileset_info`` inside it, so a batch of 16 tiles opens 32 handles and
        re-reads the info table 16 times. Holding one is why `close` exists.
        """
        if self._conn is None:
            self._conn = apsw.Connection(
                self._path, vfs=_VFS.name, flags=apsw.SQLITE_OPEN_READONLY
            )
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def chromsizes(self) -> Chromsizes:
        return self._info.coordinate_system

    def info(self) -> TilesetInfo:
        return self._info

    def tiles(self, ids: Sequence[TileId], options=None):
        return [(tid, self._tile(tid)) for tid in ids]

    # --- what the subclasses supply -----------------------------------------

    def _where(self, canvas, tid: TileId) -> str:
        """The tile's selection clause, over the joined intervals + R-tree."""
        raise NotImplementedError

    # --- internals ----------------------------------------------------------

    def _build_info(self) -> TilesetInfo:
        row = (
            self.conn.cursor().execute("SELECT * FROM tileset_info").fetchone()
        )
        (
            _zoom_step,
            max_length,
            assembly,
            chrom_names,
            chrom_sizes,
            tile_size,
            max_zoom,
            max_width,
            *_,
        ) = row

        names = tuple(chrom_names.split("\t"))
        lengths = tuple(int(v) for v in chrom_sizes.split("\t"))

        return TilesetInfo(
            # One-based, as legacy and beddb declare it. See module docstring.
            min_pos=[1, 1],
            max_pos=[max_length, max_length],
            # REAL columns in SQLite; the client expects numbers it can do
            # integer tile arithmetic with.
            max_width=int(max_width),
            tile_size=int(tile_size),
            max_zoom=int(max_zoom),
            chromsizes=[[n, v] for n, v in zip(names, lengths)],
            assembly=assembly,
        )

    def _query(self, where: str) -> Iterator[tuple]:
        return self.conn.cursor().execute(
            f"SELECT {_COLUMNS} FROM intervals, position_index "
            f"WHERE intervals.id = position_index.id AND {where}"
        )

    def _tile(self, tid: TileId) -> list[Annotation2DRecord]:
        canvas = self._info.canvas(tid.z)
        rows = self._query(self._where(canvas, tid))

        # No post-filter. The R-tree clause IS the selection -- legacy re-tests
        # in Python and gets it wrong for 1D (see module docstring). Anything
        # the query cannot express belongs in the query, not after it.
        records = [to_bedlike2d(r) for r in rows]
        records.sort(key=lambda r: (r["xStart"], r["yStart"]))
        return records


class Bed2ddbTileset(_Bed2ddbBase):
    """A bed2ddb served as 2D rectangles.

    One rule, not configurable: the stored rectangle ``[fromX, toX) x [fromY,
    toY)`` intersects the tile. Identical in meaning to
    :class:`~clodius.tiles_v2.bedpe.BedpeTileset`, and here it is a single
    R-tree box query rather than a seek plus a filter.
    """

    ndim = 2
    datatype = "2d-rectangle-domains"

    def _where(self, canvas, tid: TileId) -> str:
        x0, x1 = canvas.tile_span(tid.pos[0])
        y0, y1 = canvas.tile_span(tid.pos[1])
        return (
            f"zoomLevel <= {tid.z} "
            f"AND {overlaps('X', x0, x1)} "
            f"AND {overlaps('Y', y0, y1)}"
        )


class Bed2ddbLinksTileset(_Bed2ddbBase):
    """A bed2ddb served as 1D links, for an arc-style track.

    Parameters
    ----------
    link_policy :
        Which links a tile returns; see
        :class:`~clodius.tiles_v2.bedpe.LinkPolicy`. Defaults to ``EITHER``,
        which is what legacy's SQL intends before its post-filter discards half
        the result.

    Unlike the BEDPE version, all three policies are indexed queries here --
    including ``HULL``, because the R-tree covers both anchors.
    """

    ndim = 1
    datatype = "bedlike"

    def __init__(
        self, *args, link_policy: LinkPolicy = LinkPolicy.EITHER, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._link_policy = LinkPolicy(link_policy)

    @property
    def link_policy(self) -> LinkPolicy:
        return self._link_policy

    def _where(self, canvas, tid: TileId) -> str:
        lo, hi = canvas.tile_span(tid.pos[0])
        zoom = f"zoomLevel <= {tid.z}"
        on_x = overlaps("X", lo, hi)
        on_y = overlaps("Y", lo, hi)

        if self._link_policy is LinkPolicy.BOTH:
            return f"{zoom} AND {on_x} AND {on_y}"

        if self._link_policy is LinkPolicy.EITHER:
            return f"{zoom} AND (({on_x}) OR ({on_y}))"

        # HULL. A link's hull overlaps the range in exactly two ways:
        #
        #   1. some anchor overlaps it            -> the EITHER clause
        #   2. the hull strictly contains it, both anchors outside
        #      -> min(fromX, fromY) < lo AND max(toX, toY) > hi
        #
        # Case 2 has four sub-cases by which anchor supplies the min and the
        # max, but the two where they come from the *same* anchor mean that
        # anchor spans the range, which case 1 already caught. So only the two
        # cross terms are new, and each is a plain conjunctive box the R-tree
        # can answer.
        spans_xy = f"rFromX < {lo} AND rToY > {hi}"
        spans_yx = f"rFromY < {lo} AND rToX > {hi}"
        return (
            f"{zoom} AND (({on_x}) OR ({on_y}) OR ({spans_xy}) OR ({spans_yx}))"
        )


# --- Notes ------------------------------------------------------------------
#
# 1. No `max_records`. STRATIFIED means the aggregation step already
#    bounded tile density by assigning zoom levels, so capping here would drop
#    records the file went to the trouble of ranking. `TilePolicy` is still
#    accepted for uniformity, and `max_scan_bytes` is irrelevant since
#    everything is indexed.
#
# 2. `importance` is read from the file rather than synthesized. It is the value
#    that decided each record's zoom level, so it is the only `importance` in
#    the codebase that means anything. The SUBSAMPLED types emit
#    `random.random()` or, in tiles_v2, a content digest -- both stand-ins for
#    this.
#
# 3. Half-open overlap, where legacy uses `rToX >= lo AND rFromX <= hi`. The
#    closed form admits a record ending exactly at the tile start and one
#    starting exactly at the tile end, so a record lands in two adjacent tiles
#    when it belongs in one. Consistent now with bed.py and bedpe.py.
#
# 4. One connection per tileset, not per tile. Legacy opens a connection for
#    every tile and another for `tileset_info` inside it -- 32 handles for a
#    16-tile batch, with the info table parsed 16 times. `close()` gives that
#    lifetime a name; `mats` in cooler.py is the same problem never solved.
#
# 5. `bedarcsdb.py` reads the same schema and emits the same payload, differing
#    only in its query. It is dispatched nowhere in resgen-server, so it is
#    either dead or reached through another host. If it is live, it is
#    `Bed2ddbLinksTileset` with a particular `LinkPolicy` and should be deleted
#    in favour of it.
#
# 6. `assembly` is passed through to tileset_info because the file carries it
#    and legacy emits it. It is the only v2 tileset that can: every other type
#    infers coordinates from chromsizes handed in at construction.
#
#
#  Deliberately kept
# -----------------
# ``min_pos`` is ``[1, 1]``, matching legacy and ``beddb``, where the BEDPE
# prototypes use ``[0, 0]``. The value is one-based for no reason anyone recorded,
# but it ships with precomputed files and is not ours to reinterpret.
