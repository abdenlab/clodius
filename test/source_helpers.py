"""Shared doubles for the `clodius.core.source.Source` contract.

Every tileset backed by a `Source` is tested the same four ways -- over a
path, over a factory, over a factory whose handles are counted, and over a
handle that refuses something the contract requires -- so the doubles that
express those live here rather than once per backend. They were four
byte-identical copies across `test/core/test_source.py`,
`test/tiles_v2/test_cooler.py`, `test/tiles_v2/test_multivec.py` and
`test/tiles_v2/test_bbi.py` before this module existed.

Imported the way `test/core/mcool_fixture.py` already is: a plain module in
the test package, `from ..source_helpers import ...`.
"""

import io
import itertools


def handle_factory(path):
    """A zero-argument factory handing out a fresh handle on every call.

    A function rather than an inline ``lambda`` at each call site: a lambda
    written inside a loop or a parametrize list closes over the loop variable,
    so every factory would open whichever path the loop finished on. The
    binding happens here, in a parameter.
    """
    return recording_factory(path)[0]


def recording_factory(path, opener=lambda path: open(path, "rb")):
    """A factory that keeps every handle it hands out, for leak checks.

    ``opener`` is what makes this one helper rather than two: pass a counting
    or otherwise instrumented handle class to observe reads, and the default
    gives a plain binary handle.
    """
    opened = []

    def factory():
        handle = opener(path)
        opened.append(handle)
        return handle

    return factory, opened


def ladder(tileset):
    """Every tile id in a tileset's ladder, coarsest level first.

    Arity-aware, because a tileset is not always 1D: a 2D tileset is asked
    for the full ``n_tiles ** 2`` grid at each zoom, not its diagonal.
    """
    info = tileset.info()
    levels = getattr(info, "num_zoom_levels", None)
    if levels is None:
        levels = len(tileset.resolutions)
    for z in range(levels):
        n_tiles = info.canvas(z).n_tiles
        for pos in itertools.product(range(n_tiles), repeat=tileset.ndim):
            yield "u.{}.{}".format(z, ".".join(str(p) for p in pos))


class Unseekable(io.RawIOBase):
    """A handle that cannot report its length, as a pure stream cannot."""

    def readable(self):
        return True

    def seekable(self):
        return False

    def seek(self, *args):
        raise OSError("not seekable")


class CountingHandle(io.RawIOBase):
    """A binary handle that counts how many times it was read.

    The public-observable replacement for proving absence of I/O by deleting
    the file, which only works for a path-backed source. Both ``read`` and
    ``readinto`` count, because a reader may drive either.
    """

    def __init__(self, path):
        self._inner = open(path, "rb")
        self.reads = 0

    def read(self, size=-1):
        self.reads += 1
        return self._inner.read(size)

    def readinto(self, buffer):
        self.reads += 1
        return self._inner.readinto(buffer)

    def seek(self, offset, whence=io.SEEK_SET):
        return self._inner.seek(offset, whence)

    def tell(self):
        return self._inner.tell()

    def seekable(self):
        return True

    def readable(self):
        return True

    def close(self):
        self._inner.close()
        super().close()
