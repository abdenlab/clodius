import os.path as op
import functools as ft
import clodius.tiles.density as hgde
import clodius.tiles.utils as hgut

testdir = op.realpath(op.dirname(op.dirname(__file__)))


def test_bundled_tiles():
    filename = op.join(testdir, "data", "points_density.h5")

    tiles = hgut.bundled_tiles_wrapper_2d(
        ["c.2.0.0", "c.2.1.1", "c.2.0.1", "c.2.1.0"], ft.partial(hgde.tiles, filename)
    )
    assert len(tiles) == 4
