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

from clodius.core.policies import TilePolicy
from clodius.core.tileset import TilesetInfo
from clodius.tiles_v2.multivec import MultivecTileset

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

    def test_info_should_return_a_frozen_model(self, stateful_mv5):
        """Test that the metadata did not arrive by unfreezing the model.

        Given:
            The info of a multivec carrying row metadata.
        When:
            One of its fields is assigned.
        Then:
            It should raise. Relaxing the model config would make the previous
            test pass too, and would give back the mutable info whose cached
            derivations were the reason for freezing it.
        """
        # Arrange
        info = MultivecTileset(stateful_mv5).info()

        # Assert
        assert isinstance(info, TilesetInfo)

        # Act & assert
        with pytest.raises(Exception):
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
