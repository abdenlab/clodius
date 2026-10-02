"""Tests for clodius.tiles_v2.multivec.

A multivec is a stack of 1D vectors over a fixed bin grid, and the thing that
makes it one is its row metadata: ``row_infos`` names the rows and
``category_infos`` groups them, and a state or segmentation multivec is
unreadable without them. They are not declared fields of ``TilesetInfo`` --
they ride in as extras -- so they have to go in at construction, because the
model is frozen and assigning to it raises a ``pydantic`` error that is not a
``TilesetError`` and escapes the server boundary as a 500.

That failure lands in ``info()``, which ``tiles()`` calls on its first line,
so it takes ``tileset_info`` and every tile with it. The file without row
metadata is the only one that survives it -- which is why both shapes are
built here, and why the metadata-free case alone would pin nothing.

The fixtures are synthesized with h5py, so the module runs on a checkout with
no git-LFS payload.
"""

import json

import h5py
import numpy as np
import pytest
from pydantic import ValidationError

from clodius.core.errors import MalformedTileId
from clodius.core.policies import TilePolicy
from clodius.core.tileset import TilesetInfo
from clodius.tiles_v2.multivec import MultivecTileset

from ..source_helpers import recording_factory

#: Two contigs at 1000 and 600 bp, over a 100 bp base resolution.
CHROMS = [("c1", 1000), ("c2", 600)]
RESOLUTIONS = [100, 200]
N_ROWS = 3
TILE_SIZE = 4

#: What a state multivec carries: one entry per row, as JSON.
ROW_INFOS = ["enhancer", "promoter", "quiescent"]


def write_multivec(path, row_infos=None):
    """A minimal multires multivec, optionally carrying row metadata."""
    with h5py.File(path, "w") as f:
        chroms = f.create_group("chroms")
        chroms.create_dataset(
            "name", data=np.array([n.encode() for n, _ in CHROMS])
        )
        chroms.create_dataset(
            "length", data=np.array([v for _, v in CHROMS], dtype="i8")
        )

        info = f.create_group("info")
        info.attrs["tile-size"] = TILE_SIZE
        if row_infos is not None:
            info.create_dataset(
                "row_infos", data=np.bytes_(json.dumps(row_infos))
            )

        resolutions = f.create_group("resolutions")
        for res in RESOLUTIONS:
            values = resolutions.create_group(f"{res}/values")
            for name, length in CHROMS:
                n_bins = -(-length // res)
                values.create_dataset(
                    name, data=np.arange(n_bins * N_ROWS, dtype="f8").reshape(
                        n_bins, N_ROWS
                    )
                )
    return path


@pytest.fixture(scope="module")
def stateful_mv5(tmp_path_factory):
    """A multivec carrying row metadata, the ordinary case."""
    path = tmp_path_factory.mktemp("multivec") / "states.mv5"
    return write_multivec(str(path), row_infos=ROW_INFOS)


@pytest.fixture(scope="module")
def bare_mv5(tmp_path_factory):
    """A multivec carrying no row metadata, which never exercised the bug."""
    path = tmp_path_factory.mktemp("multivec") / "bare.mv5"
    return write_multivec(str(path))


class TestMultivecTileset:
    """Serving a multivec whose info carries metadata the model never declared."""

    def test___init___should_hold_a_policy_when_given_none(self, bare_mv5):
        """Test the attribute the ``Tileset`` protocol declares non-optional.

        Given:
            A multivec tileset constructed with no policy.
        When:
            Its policy is read.
        Then:
            It should be a ``TilePolicy``. Collapsing a falsy policy to
            ``None`` -- the same ``X or None`` construct the options slot was
            fixed for -- hands back a value contradicting the protocol, and
            every read of it is an ``AttributeError`` rather than a
            ``TilesetError``, so it escapes the boundary as a 500.
        """
        # Act
        tileset = MultivecTileset(bare_mv5)

        # Assert
        assert isinstance(tileset.policy, TilePolicy)

    def test___init___should_hold_the_policy_it_is_given(self, bare_mv5):
        """Test that supplying a policy still works, so the default is a default.

        Given:
            A multivec tileset constructed with an explicit policy.
        When:
            Its policy is read.
        Then:
            It should be the object given, not a substituted default.
        """
        # Arrange
        policy = TilePolicy(max_records=7)

        # Act
        tileset = MultivecTileset(bare_mv5, policy=policy)

        # Assert
        assert tileset.policy is policy

    def test_tiles_should_raise_when_the_aggregation_function_is_invalid(
        self, bare_mv5
    ):
        """Test a batch option that names something this tileset cannot do.

        Given:
            A batch carrying an ``aggFunc`` that is not one of the supported
            functions.
        When:
            The batch is served.
        Then:
            It should raise ``MalformedTileId`` rather than return an error
            payload per tile. Options arrive once for the whole request, so a
            bad one is a whole-batch failure -- and nothing else in the suite
            can tell those two answers apart.
        """
        # Arrange
        tileset = MultivecTileset(bare_mv5)
        ids = [tileset.parse_tile_id("u.0.0"), tileset.parse_tile_id("u.0.1")]

        # Act & assert
        with pytest.raises(MalformedTileId, match="aggFunc"):
            tileset.tiles(ids, {"aggGroups": [[0, 1]], "aggFunc": "bogus"})

    def test_tiles_should_answer_every_id_when_one_is_off_the_canvas(
        self, bare_mv5
    ):
        """Test the per-tile boundary inside this tileset's own override.

        Given:
            A batch holding one servable tile and one position far past the
            end of the canvas.
        When:
            The batch is served.
        Then:
            It should answer both, the bad one with a ``TileOutOfBounds``
            payload. The override predates the shared boundary, so it has to
            carry its own -- the earlier comprehension had none and lost the
            whole batch to one bad position.
        """
        # Arrange
        tileset = MultivecTileset(bare_mv5)
        ids = [
            tileset.parse_tile_id("u.0.0"),
            tileset.parse_tile_id("u.0.9999"),
        ]

        # Act
        served = tileset.tiles(ids)

        # Assert
        assert len(served) == 2
        assert "error" not in served[0][1]
        assert served[1][1]["error_type"] == "TileOutOfBounds"

    def test_tiles_should_raise_when_an_option_is_not_recognized(
        self, bare_mv5
    ):
        """Test an option key this tileset does not read.

        Given:
            A multivec tileset and a batch carrying an option it knows nothing
            about.
        When:
            The batch is served.
        Then:
            It should raise ``MalformedTileId`` for the whole batch. This
            class overrides ``tiles()`` to hoist its row-aggregation parsing,
            so it does not inherit the base's refusal -- and the unrecognized
            key was silently discarded, serving tiles that ignored the
            request. Options arrive once for every tile, so this is a
            whole-batch failure rather than a per-tile payload.
        """
        # Arrange
        tileset = MultivecTileset(bare_mv5)
        ids = [tileset.parse_tile_id("u.0.0")]

        # Act & assert
        with pytest.raises(MalformedTileId):
            tileset.tiles(ids, {"bogus": 1})

    def test_info_should_carry_the_row_metadata(self, stateful_mv5):
        """Test the field that makes a state multivec readable.

        Given:
            A multivec carrying ``row_infos``, as every state or segmentation
            multivec does.
        When:
            Its info is requested.
        Then:
            It should serve the row metadata. ``row_infos`` is not a declared
            field, so it has to be passed at construction -- assigning it
            afterwards raises on the frozen model, and that raise is not a
            ``TilesetError``, so it reaches the client as a 500 rather than a
            renderable error.
        """
        # Arrange
        tileset = MultivecTileset(stateful_mv5)

        # Act
        served = tileset.info().to_dict()

        # Assert
        assert served["row_infos"] == ROW_INFOS

    def test_info_should_omit_the_metadata_when_the_file_carries_none(
        self, bare_mv5
    ):
        """Test that the metadata is conditional rather than always emitted.

        Given:
            A multivec carrying no row metadata.
        When:
            Its info is requested.
        Then:
            ``row_infos`` should be absent. This is the shape that kept
            working while every file with metadata was dead, so without it the
            test above can pass against a build that serves the key
            unconditionally.
        """
        # Arrange
        tileset = MultivecTileset(bare_mv5)

        # Act
        served = tileset.info().to_dict()

        # Assert
        assert "row_infos" not in served

    def test_info_should_return_a_frozen_model(self, stateful_mv5):
        """Test that the metadata did not arrive by unfreezing the model.

        Given:
            The info of a multivec carrying row metadata.
        When:
            One of its fields is assigned.
        Then:
            It should raise ``ValidationError``. The class matters: a bare
            ``Exception`` here also passes for an ``AttributeError`` or a
            ``TypeError``, so the test would stay green against a model that
            is broken in some other way. Relaxing the model config would make
            the previous test pass too, and would give back the mutable info
            whose cached derivations were the reason for freezing it.
        """
        # Arrange
        info = MultivecTileset(stateful_mv5).info()

        # Assert
        assert isinstance(info, TilesetInfo)

        # Act & assert
        with pytest.raises(ValidationError):
            info.tile_size = 999

    def test_tiles_should_serve_a_file_carrying_row_metadata(
        self, stateful_mv5
    ):
        """Test the path ``info()`` gates.

        Given:
            A multivec carrying row metadata.
        When:
            A tile is served.
        Then:
            It should return a dense payload with one row per track.
            ``tiles()`` calls ``info()`` on its first line, so a file whose
            info cannot be built serves no tiles at all -- the whole dataset
            is dark, not just its description.
        """
        # Arrange
        tileset = MultivecTileset(stateful_mv5)
        tid = tileset.parse_tile_id("u.0.0")

        # Act
        (_, payload), = tileset.tiles([tid])

        # Assert
        assert payload["shape"] == [N_ROWS, TILE_SIZE]

    def test_tiles_should_agree_between_a_path_and_a_file_like_source(
        self, stateful_mv5
    ):
        """Test the two source shapes against each other across the ladder.

        Given:
            The same multivec addressed by path and by a callable that opens
            a fresh binary handle.
        When:
            Every tile in the ladder is requested from each.
        Then:
            The payloads should match tile for tile, and none should be an
            error. How the caller spelled the source is not part of the
            coordinate system, so a remote tileset serving even one tile
            differently would be serving a different dataset under the same
            id -- and asserting only that the two agree is satisfied by both
            refusing every tile.
        """
        # Arrange
        compared = 0

        # Act & assert
        with (
            MultivecTileset(stateful_mv5) as by_path,
            MultivecTileset(lambda: open(stateful_mv5, "rb")) as by_handle,
        ):
            info = by_path.info()
            assert info == by_handle.info()
            for z in range(len(by_path.resolutions)):
                for x in range(info.canvas(z).n_tiles):
                    tile_id = f"u.{z}.{x}"
                    (_, left), = by_path.tiles(
                        [by_path.parse_tile_id(tile_id)]
                    )
                    (_, right), = by_handle.tiles(
                        [by_handle.parse_tile_id(tile_id)]
                    )
                    assert left == right, tile_id
                    assert "error" not in left, tile_id
                    compared += 1
        assert compared

    def test_tiles_should_reopen_the_source_when_the_tileset_was_closed(
        self, stateful_mv5
    ):
        """Test that a closed tileset is reusable rather than spent.

        Given:
            A factory-backed tileset that has served a tile and been closed.
        When:
            The same tile is requested again.
        Then:
            It should be served identically, from a second handle. ``close``
            nulls the cached file so the property re-opens, which is what
            makes ``with`` followed by reuse work.
        """
        # Arrange
        factory, opened = recording_factory(stateful_mv5)
        tileset = MultivecTileset(factory)
        tile_id = tileset.parse_tile_id("u.0.0")
        (_, before), = tileset.tiles([tile_id])

        # Act
        tileset.close()
        (_, after), = tileset.tiles([tile_id])
        tileset.close()

        # Assert
        assert before == after
        assert len(opened) == 2
        assert all(handle.closed for handle in opened)

    def test___init___should_refuse_an_already_open_file(self, stateful_mv5):
        """Test that the one wrong shape fails at construction.

        Given:
            An open file rather than a callable that opens one.
        When:
            A tileset is constructed from it.
        Then:
            It should raise ``TypeError``. h5py would accept the handle, so
            this backend alone would appear to work -- and the same argument
            given to an oxbow tileset is a source that cannot be reopened.
        """
        # Act & assert
        with open(stateful_mv5, "rb") as handle:
            with pytest.raises(TypeError, match="not an already open file"):
                MultivecTileset(handle)

    def test_file_should_close_a_handle_it_could_not_open_as_hdf5(
        self, tmp_path
    ):
        """Test that a failed open does not strand the handle behind it.

        Given:
            A factory-backed tileset over a file that is not HDF5, accessed
            several times.
        When:
            Each access fails.
        Then:
            Every handle opened should be closed. Storing the handle before
            the open succeeded meant each retry overwrote the only reference
            to the previous one, so the tileset leaked a handle per attempt
            while its own contract promised to close what it opened.
        """
        # Arrange
        path = tmp_path / "notan.mv5"
        path.write_bytes(b"this is not HDF5")
        factory, opened = recording_factory(str(path))
        # `tile_size` keeps the constructor from reading the file itself.
        tileset = MultivecTileset(factory, tile_size=4)

        # Act
        for _ in range(3):
            with pytest.raises(OSError):
                tileset.file

        # Assert
        assert len(opened) == 3
        assert all(handle.closed for handle in opened)

    def test_close_should_release_both_the_file_and_the_handle(
        self, stateful_mv5
    ):
        """Test the ownership rule for a caller-supplied factory.

        Given:
            A factory-backed tileset, used once so that it opens.
        When:
            It is closed.
        Then:
            Both the HDF5 file and the handle underneath it should be
            released. ``h5py.File.close()`` does not close a Python file
            object it was handed, so closing only one of the two leaks a
            descriptor per tileset.
        """
        # Arrange
        factory, opened = recording_factory(stateful_mv5)
        tileset = MultivecTileset(factory)
        hdf5 = tileset.file

        # Act
        tileset.close()

        # Assert
        assert not hdf5
        assert opened and all(handle.closed for handle in opened)

    def test___init___should_close_the_handle_when_the_file_is_not_a_multivec(
        self, tmp_path
    ):
        """Test the ownership rule on the failure the guard did not cover.

        Given:
            A factory over a file that is valid HDF5 but carries no ``info``
            group, so the ``tile-size`` read in the constructor raises.
        When:
            Construction fails.
        Then:
            The handle it opened should already be closed. The constructor
            raised, so the caller never receives an object to call ``close``
            on -- and for a factory-backed source the orphan is the caller's
            remote connection, not a local descriptor.
        """
        # Arrange
        path = tmp_path / "plain.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("x", data=[1])
        factory, opened = recording_factory(path)

        # Act
        with pytest.raises(KeyError):
            MultivecTileset(factory)

        # Assert
        assert opened and all(handle.closed for handle in opened)

    def test_file_should_open_a_path_through_the_native_hdf5_driver(
        self, stateful_mv5
    ):
        """Test that a path still reaches h5py as a path.

        Given:
            A tileset constructed from a path, and the same file from a
            factory.
        When:
            Each opens its HDF5 file.
        Then:
            The path-backed one should use HDF5's native ``sec2`` driver and
            the factory-backed one ``fileobj``. See the same assertion, and
            the descriptor cost behind it, in ``test_cooler.py``.
        """
        # Arrange & act
        with (
            MultivecTileset(stateful_mv5) as by_path,
            MultivecTileset(lambda: open(stateful_mv5, "rb")) as by_handle,
        ):
            # Assert
            assert by_path.file.driver == "sec2"
            assert by_handle.file.driver == "fileobj"
