import clodius.tiles.time_interval as hgti
import os.path as op

testdir = op.realpath(op.dirname(op.dirname(__file__)))


def test_tileset_info():
    filename = op.join(testdir, "data", "sample_htime.json")

    hgti.tileset_info(filename)
    # TODO: Make assertions about info returned.
