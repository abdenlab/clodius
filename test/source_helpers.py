"""Shared doubles for the `clodius.core.source.Source` contract.

Every tileset backed by a `Source` is tested the same four ways -- over a
path, over a factory, over a factory whose handles are counted, and over a
handle that refuses something the contract requires -- so the doubles that
express those live here rather than once per backend. Before this module
existed `test/tiles_v2/test_cooler.py` and `test/tiles_v2/test_multivec.py`
held byte-identical copies, `test/core/test_source.py` held the same body
under a different name, and `test/tiles_v2/test_bbi.py` -- which needs them
most -- held none and was about to grow its own.

Imported the way `test/core/mcool_fixture.py` already is: a plain module in
the test package, `from ..source_helpers import ...`.
"""

import io
import itertools
import threading

#: Every bounded wait in a concurrency test is capped at this, in seconds.
JOIN_TIMEOUT = 10.0


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


def released_together(targets, timeout=JOIN_TIMEOUT):
    """Run each callable in ``targets`` on its own thread, released as one.

    Every thread is already alive and waiting on the barrier before any of
    them runs, which is what makes the release simultaneous and also what
    keeps a thread started late from being starved of the GIL by the others.

    Returns ``(alive, failures)``: whether each thread was still running when
    its bounded join expired, and whatever each one raised. Both are returned
    rather than asserted so the calling test says in its own Assert phase that
    the threads finished -- a hung thread must fail, not stall the suite.

    Daemon threads, which is what makes that promise good: a non-daemon thread
    still wedged on the deadlock these tests exist to detect fails the test
    and then hangs the interpreter forever at exit, because `threading`'s
    shutdown hook joins it with no timeout. The report would never be printed.

    `BaseException`, not `Exception`: `pybigtools` can surface a
    `pyo3_runtime.PanicException`, whose mro is `PanicException` ->
    `BaseException`, so a narrower catch would let the one failure these
    tests most need to see kill its thread silently.

    Here rather than in one backend's module because both `test_bbi.py` and
    `test_cooler.py` test the same double-checked lock. `test_cooler.py` had
    its own copy with non-daemon threads and an unbounded join, which is
    exactly the shape that hangs the suite instead of failing it.
    """
    barrier = threading.Barrier(len(targets), timeout=timeout)
    failures = []

    def run(target):
        try:
            barrier.wait()
            target()
        except BaseException as exc:  # noqa: BLE001 -- reported, not handled
            failures.append(exc)

    threads = [
        threading.Thread(target=run, args=(target,), daemon=True)
        for target in targets
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout)
    return [thread.is_alive() for thread in threads], failures


def ladder(tileset):
    """Every tile id in a tileset's ladder, coarsest level first.

    Arity-aware, because a tileset is not always 1D: a 2D tileset is asked
    for the full ``n_tiles ** 2`` grid at each zoom, not its diagonal.
    """
    info = tileset.info()
    for z in range(info.num_zoom_levels):
        n_tiles = info.canvas(z).n_tiles
        for pos in itertools.product(range(n_tiles), repeat=tileset.ndim):
            yield "u.{}.{}".format(z, ".".join(str(p) for p in pos))
