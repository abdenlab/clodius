"""Shared teardown for the legacy tile modules.

``clodius.tiles.cooler`` keeps a module-global handle cache. Nothing evicts
from it, so without this every cooler read in the suite leaks an open HDF5
handle for the process lifetime and two test modules populate the same dict.
"""

import pytest

import clodius.tiles.cooler as tiles_cooler


@pytest.fixture(autouse=True)
def clear_cooler_mats():
    """Close and drop every cached cooler handle after each test.

    ``make_mats`` stores ``[h5py.File, info]`` keyed on the file path and
    ``tileset_info``/``tiles`` read straight from it. Contamination is bounded
    today only because ``tmp_path_factory`` hands each module-scoped fixture a
    unique path, so no two tests collide on a key -- an accident of the fixture
    scoping rather than anything the cache guarantees.
    """
    yield
    for entry in tiles_cooler.mats.values():
        entry[0].close()
    tiles_cooler.mats.clear()
