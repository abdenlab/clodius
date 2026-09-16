import os.path as op
import clodius.tiles.nplabels as ctn
import h5py

testdir = op.realpath(op.dirname(op.dirname(__file__)))


def test_tiles():
    filename = op.join(testdir, "data", "labels.h5")
    f = h5py.File(filename, "r")

    array = f["labels"]

    tiles = ctn.tiles(array, 0, 0, None)

    assert tiles[0]["x"] == 0
    assert tiles[-1]["x"] == 86992

    tiles = ctn.tiles(array, 2, 1, None)

    assert tiles[0]["x"] == 32768
    assert tiles[-1]["x"] == 65535
