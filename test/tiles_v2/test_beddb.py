"""Tests for clodius.tiles_v2.beddb.

The format has four schema versions, and ``"3t"`` is the one that does not
answer a tile with a range query. Its tiles are precomputed: a ``tiles`` table
maps a linear node id to the intervals assigned to it, and the reader derives
that id arithmetically as ``2**z - 1 + x`` -- the breadth-first index of a node
in a complete binary tree.

That derivation is total over the integers and injective only over positions
that exist. Given an off-lattice ``x`` it lands on a different node and returns
that node's intervals, so the tile is populated, well-formed, and wrong --
which is worse than the empty tile an unchecked range query would produce,
because nothing about it looks like a failure. And ``2**z`` on a
request-controlled exponent raises a bare ``ValueError`` past Python's
integer-conversion limit, which is not a ``TilesetError`` and so escapes the
per-tile boundary and fails the whole batch.

The other three versions reach a bounds-checked accessor on the way to their
query. This one has to be checked before the branch, which is why the check
lives above it rather than inside the arm that needs it.

The fixture is synthesized with sqlite3 rather than committed: one interval per
node of a three-level tree, each labelled with the node it belongs to, so a
tile serving someone else's records names them.
"""

import sqlite3

import pytest

from clodius.core.errors import TilesetError
from clodius.tiles_v2.beddb import BedDbTileset

#: Every node of a complete three-level binary tree, mapped to the linear id
#: ``2**z - 1 + x`` the reader derives. Written out rather than computed, so
#: the test does not reimplement the arithmetic it is checking.
LAYOUT = {
    (0, 0): 0,
    (1, 0): 1,
    (1, 1): 2,
    (2, 0): 3,
    (2, 1): 4,
    (2, 2): 5,
    (2, 3): 6,
}

TILE_SIZE = 1024
MAX_ZOOM = 2
MAX_WIDTH = 4096


def write_3t_beddb(path):
    """Write a ``"3t"`` beddb holding one labelled interval per tree node."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE tileset_info ("
        " zoom_step INT, max_length INT, assembly TEXT, chrom_names TEXT,"
        " chrom_sizes TEXT, tile_size REAL, max_zoom INT, max_width REAL,"
        " version TEXT, header TEXT)"
    )
    conn.execute(
        "INSERT INTO tileset_info VALUES (?,?,?,?,?,?,?,?,?,?)",
        (1, 1000, "test", "c1", "1000", TILE_SIZE, MAX_ZOOM, MAX_WIDTH, "3t", ""),
    )
    conn.execute(
        "CREATE TABLE intervals ("
        " id INT PRIMARY KEY, zoomLevel INT, importance REAL, startPos INT,"
        " endPos INT, chrOffset INT, uid TEXT, name TEXT, fields TEXT)"
    )
    conn.execute("CREATE TABLE tiles (id INT, intervalId INT)")
    for i, ((z, x), node) in enumerate(LAYOUT.items()):
        label = f"node_z{z}_x{x}"
        conn.execute(
            "INSERT INTO intervals VALUES (?,?,?,?,?,?,?,?,?)",
            (
                i,
                z,
                1.0,
                x * 10,
                x * 10 + 5,
                0,
                f"u{i}",
                label,
                f"c1\t{x * 10}\t{x * 10 + 5}\t{label}",
            ),
        )
        conn.execute("INSERT INTO tiles VALUES (?,?)", (node, i))
    conn.commit()
    conn.close()
    return str(path)


#: Four records at known zoom levels, spread so a tile can pick one out.
RANGE_RECORDS = [
    (0, 0, 10, 20, "a"),
    (1, 1, 1030, 1040, "b"),
    (2, 2, 2060, 2070, "c"),
    (3, 2, 3080, 3090, "d"),
]


def write_range_beddb(path, version):
    """Write a beddb at schema version 1, 2 or 3 over the same records.

    ``version`` of ``None`` writes v1, which has no ``version`` column at all;
    v2 and v3 move the zoom range into the R-tree, and v3 adds ``name`` and
    stores its version as text.
    """
    conn = sqlite3.connect(str(path))
    columns = (
        "zoom_step INT, max_length INT, assembly TEXT, chrom_names TEXT,"
        " chrom_sizes TEXT, tile_size REAL, max_zoom INT, max_width REAL,"
        " header TEXT"
    )
    values = [1, 4000, "test", "c1", "4000", TILE_SIZE, MAX_ZOOM, MAX_WIDTH, ""]
    if version is not None:
        columns += ", version TEXT"
        values.append(version)
    conn.execute(f"CREATE TABLE tileset_info ({columns})")
    conn.execute(
        f"INSERT INTO tileset_info VALUES ({','.join('?' * len(values))})",
        values,
    )

    name_col = ", name TEXT" if version == "3" else ""
    conn.execute(
        "CREATE TABLE intervals ("
        " id INT PRIMARY KEY, zoomLevel INT, importance REAL, startPos INT,"
        f" endPos INT, chrOffset INT, uid TEXT, fields TEXT{name_col})"
    )
    if version is None:
        conn.execute(
            "CREATE VIRTUAL TABLE position_index USING"
            " rtree(id, rStartPos, rEndPos)"
        )
    else:
        conn.execute(
            "CREATE VIRTUAL TABLE position_index USING"
            " rtree(id, rStartPos, rEndPos, rStartZoomLevel, rEndZoomLevel)"
        )

    for rid, zoom, start, end, label in RANGE_RECORDS:
        extra = (label,) if version == "3" else ()
        placeholders = "?,?,?,?,?,?,?,?" + (",?" if extra else "")
        conn.execute(
            f"INSERT INTO intervals VALUES ({placeholders})",
            (rid, zoom, 1.0, start, end, 0, f"u{rid}",
             f"c1\t{start}\t{end}\t{label}") + extra,
        )
        if version is None:
            conn.execute(
                "INSERT INTO position_index VALUES (?,?,?)", (rid, start, end)
            )
        else:
            conn.execute(
                "INSERT INTO position_index VALUES (?,?,?,?,?)",
                (rid, start, end, zoom, MAX_ZOOM),
            )
    conn.commit()
    conn.close()
    return str(path)


@pytest.fixture(scope="module")
def range_beddbs(tmp_path_factory):
    """One beddb per range-querying schema version, over identical records."""
    root = tmp_path_factory.mktemp("beddb_versions")
    return {
        1: write_range_beddb(root / "v1.beddb", None),
        2: write_range_beddb(root / "v2.beddb", 2),
        3: write_range_beddb(root / "v3.beddb", "3"),
    }


@pytest.fixture(scope="module")
def beddb_3t(tmp_path_factory):
    """A precomputed-tile beddb, the one version with no range query."""
    return write_3t_beddb(tmp_path_factory.mktemp("beddb") / "annot.beddb")


def labels(payload):
    """The node label of each served record."""
    return [record["name"] for record in payload]


class TestBedDbTileset:
    """The bounds check the precomputed-tile branch used to skip."""

    @pytest.mark.parametrize("version", [1, 2, 3])
    @pytest.mark.parametrize("tile_id,expected", [
        ("u.0.0", ["a"]),
        ("u.2.0", ["a"]),
        ("u.2.2", ["c"]),
    ])
    def test_tiles_should_agree_across_the_range_querying_versions(
        self, range_beddbs, version, tile_id, expected
    ):
        """Test that three schemas answer one tile the same way.

        Given:
            The same records written at schema versions 1, 2 and 3.
        When:
            The same tile is served from each.
        Then:
            All three should return the same records. Version 1 filters zoom
            with a column predicate while 2 and 3 move it into the R-tree, so
            these are three different queries reaching one answer -- and only
            the fourth variant had any coverage.
        """
        # Arrange
        tileset = BedDbTileset(range_beddbs[version])

        # Act
        (_, payload), = tileset.tiles([tileset.parse_tile_id(tile_id)])

        # Assert
        assert sorted(r["fields"][3] for r in payload) == expected

    def test_version_should_normalize_a_version_stored_as_text(
        self, range_beddbs
    ):
        """Test the v3 header, which stores its version as a string.

        Given:
            A beddb whose ``version`` column holds the text ``"3"``, as real
            v3 files do.
        When:
            The version is read and a tile served.
        Then:
            It should report the integer 3 and emit ``name`` on every record.
            Left as text, the version compares equal to nothing the selector
            branches on and the file is read as v1 -- silently dropping the
            column v3 exists to add.
        """
        # Arrange
        tileset = BedDbTileset(range_beddbs[3])

        # Act
        (_, payload), = tileset.tiles([tileset.parse_tile_id("u.0.0")])

        # Assert
        assert tileset.version == 3
        assert all("name" in record for record in payload)

    @pytest.mark.parametrize("version", [1, 2])
    def test_tiles_should_omit_the_name_before_version_three(
        self, range_beddbs, version
    ):
        """Test the control for the name column above.

        Given:
            A beddb at a schema version with no ``name`` column.
        When:
            A tile is served.
        Then:
            No record should carry a ``name`` key. Selecting one regardless
            would raise on a file that does not have it.
        """
        # Arrange
        tileset = BedDbTileset(range_beddbs[version])

        # Act
        (_, payload), = tileset.tiles([tileset.parse_tile_id("u.0.0")])

        # Assert
        assert all("name" not in record for record in payload)

    def test_version_should_report_the_precomputed_schema(self, beddb_3t):
        """Test that the fixture exercises the branch under test.

        Given:
            The synthesized beddb.
        When:
            Its schema version is read.
        Then:
            It should be ``"3t"``. The other three versions reach a
            bounds-checked accessor on their own, so a fixture that fell back
            to one of them would leave every test below passing vacuously.
        """
        # Act
        tileset = BedDbTileset(beddb_3t)

        # Assert
        assert tileset.version == "3t"

    @pytest.mark.parametrize("z,x", [(0, 0), (1, 1), (2, 0), (2, 3)])
    def test_tiles_should_serve_the_records_stored_for_that_node(
        self, beddb_3t, z, x
    ):
        """Test that each in-range position reads its own node.

        Given:
            A tile position that exists at its zoom.
        When:
            The tile is served.
        Then:
            It should return the interval labelled with that node. Without
            this the bounds tests below would pass against a build that served
            nothing at all.
        """
        # Arrange
        tileset = BedDbTileset(beddb_3t)

        # Act
        (_, payload), = tileset.tiles([tileset.parse_tile_id(f"u.{z}.{x}")])

        # Assert
        assert labels(payload) == [f"node_z{z}_x{x}"]

    @pytest.mark.parametrize("z,x", [(0, 3), (1, 5), (2, 4)])
    def test_tiles_should_refuse_a_position_that_does_not_exist_at_its_zoom(
        self, beddb_3t, z, x
    ):
        """Test the off-lattice position, which used to land on another node.

        Given:
            A tile position past the last tile at its zoom -- ``0.3``, which
            the id arithmetic maps onto node ``(2, 0)``.
        When:
            The tile is served.
        Then:
            It should return a refusal rather than records. Unchecked, the
            derivation is not injective over positions that do not exist, so
            the tile came back populated with a different node's intervals and
            a client had nothing to tell it apart from its own.
        """
        # Arrange
        tileset = BedDbTileset(beddb_3t)

        # Act
        (_, payload), = tileset.tiles([tileset.parse_tile_id(f"u.{z}.{x}")])

        # Assert
        assert isinstance(payload, dict)
        assert payload["error_type"] == "TileOutOfBounds"

    def test_tiles_should_refuse_a_zoom_past_the_ladder_without_failing_the_batch(
        self, beddb_3t
    ):
        """Test a request-controlled exponent, which used to kill the request.

        Given:
            A batch holding one servable tile and one at a zoom far past the
            ladder.
        When:
            The batch is served.
        Then:
            It should answer both, the bad one with a refusal. ``2**z`` on an
            unchecked zoom raises a bare ``ValueError`` past Python's
            integer-conversion limit -- not a ``TileError``, so the per-tile
            boundary could not convert it and the servable tile was lost with
            it.
        """
        # Arrange
        tileset = BedDbTileset(beddb_3t)
        ids = [
            tileset.parse_tile_id("u.0.0"),
            tileset.parse_tile_id("u.99999999.0"),
        ]

        # Act
        served = tileset.tiles(ids)

        # Assert
        assert labels(served[0][1]) == ["node_z0_x0"]
        assert isinstance(served[1][1], dict)

    def test_parse_tile_id_should_reject_a_negative_position(self, beddb_3t):
        """Test the other half of the off-lattice range, rejected earlier.

        Given:
            A tile id carrying a negative position.
        When:
            It is parsed.
        Then:
            It should raise a ``TilesetError``. A negative position is caught
            in the id rather than at the canvas, so the bounds check above
            only has to cover the upper end.
        """
        # Arrange
        tileset = BedDbTileset(beddb_3t)

        # Act & assert
        with pytest.raises(TilesetError):
            tileset.parse_tile_id("u.0.-1")
