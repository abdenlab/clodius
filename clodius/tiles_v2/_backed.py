"""The open-and-release protocol a file-backed tileset shares.

The HDF5 and BBI tilesets differ in which reader they hand their bytes to and
agree completely on how they open and close one. That agreement used to be two
hand-written copies, and every gap in the protocol was a gap in one copy or
the other -- a publication order that a concurrent `close` could slip between,
a constructor that orphaned its handle, an unlocked check-then-act, a reader
dropped for the garbage collector rather than closed. It lives here instead,
so a fix lands once and the next backend inherits it.

That includes the whole of construction, which subclasses used to hand-write:
a subclass does its header read inside :meth:`FileBacked._configuring`, which
releases what the read opened whether the block raises or returns. Both halves
used to be the subclass's to remember, and the release half was three
near-identical copies with a byte-identical rationale comment.

There is no test module for this protocol. It is pinned once per backend
instead, in `test/tiles_v2/test_bbi.py` and `test/tiles_v2/test_cooler.py`,
which is the same duplication this module removed from the source. A suite
parametrized over ``(tileset_cls, fixture, source_shape)`` is owed, and is
the right place for the tests the BBI module currently carries on this
class's behalf; it is deferred to the oxbow slices rather than forgotten.

Lives in ``tiles_v2`` rather than beside `clodius.core.source.Source`, which
it builds on, because a subclass names its reader's type and ``clodius.core``
must not depend on ``h5py`` or ``pybigtools``.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO

from clodius.core.source import Source, SourceLike
from clodius.core.tileset import BaseTileset


class FileBacked[_R](BaseTileset):
    """A tileset whose bytes are one file, read through one reader object.

    ``_R`` is the reader a subclass opens -- an ``h5py.File``, a
    ``pybigtools.BBIReader``. It is unbounded deliberately: the only thing
    this class asks of a reader is a ``close()``, and a ``Closeable`` protocol
    bound would not hold for ``pybigtools.BBIReader``, which reaches ``close``
    through ``__getattr__`` on its Python wrapper and so satisfies the shape
    per instance rather than per class.

    Opens lazily, on first use of the reader, so constructing a tileset over
    an unreachable source is not itself an error. :meth:`reading` is the only
    public way in; the reader itself is not published, because for BBI it
    carries a serialization constraint a caller holding the bare object has no
    way to honor.

    *Registration is free.* A subclass that reads a header has opened the
    source by the time construction returns, and :meth:`_configuring` closes
    again to hand the descriptor -- or the caller's remote connection -- back.
    All three backends do: a server registers tilesets it may never serve, and
    a factory-backed one held open is a live remote connection per registered
    dataset. `close` is idempotent and the reader reopens, so a served tileset
    pays one more open and an unserved one pays none.

    Two limits on that, both measured. Construction still *reads* the header
    where it needs one, so a source that is not this tileset's format is
    refused at registration rather than at the first request -- but a caller
    who supplies every header-derived parameter (``chromsizes`` for BBI and
    cooler) skips the only read, and the refusal then moves to the first
    request. And "free" ends at the first serve: nothing reaps a reader once
    it has been used, so steady-state descriptor count equals the number of
    *served* tilesets rather than zero. Measured at 300 served-and-retained
    tilesets holding 300 descriptors, released only by `close`.

    **Ownership.** Where the reader accepts a path, a path-backed source is
    handed the path string, exactly as a reader that had never heard of
    `Source` would do -- :meth:`_needs_handle` decides which sources those
    are. For HDF5 that is not merely equivalent to handing over a handle
    opened from it: HDF5 recognizes that the same file is already open
    read-only in the process and shares one descriptor across every tileset,
    which it cannot do for a Python file object. That reasoning is HDF5's and
    does not generalize: ``pybigtools`` shares nothing between readers of one
    path either way -- measured at one descriptor per reader for both routes,
    and at the same cost per open (0.017 ms each way), so for BBI the path
    branch buys correct suffix dispatch and exemption from the read lock
    rather than any saving. Only a source the reader cannot take as a path
    produces a handle, and that handle is this tileset's to close, because
    this tileset is what opened it. :meth:`close` releases both it and the
    reader, which do not release each other: neither ``h5py.File.close()``
    nor ``BBIReader.close()`` closes a Python file object it was handed.

    **Lifecycle.** :meth:`close` is idempotent, and a tileset reopens on next
    use -- `BaseTileset.__exit__` closes, and a tileset used again after its
    ``with`` block gets a second handle, owned on the same terms as the first.
    """

    _src: Source
    _reader: _R | None
    _handle: IO[bytes] | None
    _lock: threading.Lock

    def __init__(self, source: SourceLike) -> None:
        self._src = Source.coerce(source)
        self._reader = None
        self._handle = None
        self._lock = threading.Lock()

    @contextmanager
    def reading(self) -> Iterator[_R]:
        """The reader, under whatever serialization this backend needs.

        The only public route to the reader. What it yields may carry a
        constraint the tileset is responsible for -- `BBITileset` serializes
        reads through a handle it owns, because ``pybigtools`` holds a borrow
        on it for the length of a query -- and a caller handed the bare reader
        has no way to honor it. That is why the reader is not published.

        This base imposes no serialization: HDF5 serializes its own calls.
        In-package HDF5 callers therefore read through :meth:`_opened`
        directly rather than entering this, which costs a generator per
        access on a property that `_build_info` reads repeatedly.
        """
        yield self._opened()

    def close(self) -> None:
        """Release the reader and any handle this tileset opened."""
        with self._lock:
            try:
                if self._reader is not None:
                    self._reader.close()
            finally:
                # `finally`, because a failed reader close must not strand the
                # handle: leaving `_reader` set would make every retry raise at
                # the same point and the handle would never be reached again.
                self._reader = None
                handle, self._handle = self._handle, None
                if handle is not None:
                    handle.close()

    def _opened(self) -> _R:
        """The open reader, opening it on first access.

        A method rather than a property, and private rather than public. The
        name says what it returns: this is the reader, which is an
        ``h5py.File`` as often as it is a ``pybigtools.BBIReader``, so naming
        it for either one misdescribes it half the time.
        """
        # Double-checked: the unsynchronized read is the whole-lifetime common
        # case, and the lock is what keeps a concurrent first access from
        # calling the factory N times and leaving N-1 handles that `close` can
        # no longer reach. A caller's handle has no finalizer to fall back on.
        # Bound once, and returned from the binding. Reading the attribute
        # again on the way out is a second, unsynchronized load, and a `close`
        # landing between the two hands the caller `None` through a signature
        # that promises a reader. The read inside the lock makes the result
        # non-`None` by construction.
        #
        # What this does NOT make safe is closing a tileset while another
        # thread is reading through the reference it already has: this path is
        # deliberately unlocked, so the reader this returns can be closed
        # while a caller is still reading through it, which surfaces as the
        # backend's own closed-file error.
        # Widening the lock to cover the caller's use would serialize every
        # reader to fix a caller error, so it is not done -- a tileset closed
        # concurrently with a request is the caller's race, not this class's.
        #
        # `close` also waits on an open already in flight, since both take
        # `_lock`. For a factory with no timeout of its own that wait is
        # unbounded; a server that needs a bounded shutdown owns that timeout.
        reader = self._reader
        if reader is None:
            with self._lock:
                if self._reader is None:
                    self._open()
                reader = self._reader
        return reader

    @contextmanager
    def _configuring(self) -> Iterator[None]:
        """Run construction work, then release whatever it opened.

        A subclass that reads a header does it inside this, after
        ``super().__init__(source)``::

            super().__init__(source)
            with self._configuring():
                self.tile_size = tile_size or self._opened()["info"].attrs[...]

        Both exits release. On a raise that is cleanup; on a normal return it
        is the "registration is free" property in the class docstring, which
        every backend used to implement by hand with a trailing
        ``self.close()`` under a byte-identical comment.

        A context manager rather than a hook the base calls: a hook takes no
        arguments, so every subclass had to copy its constructor arguments
        onto ``self`` to read them back one statement later, and the
        requirement to do that *before* ``super().__init__`` was unwritten
        anywhere. Entering a block keeps the arguments as locals, where they
        already are.

        `close` is idempotent and null-safe, so this covers whatever the
        block does -- including a parameter a subclass adds later, and work
        the subclass does around rather than inside a hook.

        `BaseException`, not `Exception`: a KeyboardInterrupt or a
        CancelledError landing between the open and the end of construction
        orphans what was opened exactly as a ValueError does, and the handler
        re-raises, so widening it costs nothing.
        """
        try:
            yield
        except BaseException:
            # Suppressed, not propagated: this is cleanup for a failure that
            # already has an exception, and letting a failing `close` replace
            # it hands the caller the teardown error with the real cause
            # demoted to `__context__`. A server renders a `TilesetUnavailable`
            # as a refusal and an `OSError` as a 500, so which one escapes
            # decides what the operator sees.
            try:
                self.close()
            except Exception:
                pass
            raise
        # A second, deliberately separate suppression. The release on this path
        # is an optimization rather than a correctness requirement, so it must
        # not be able to fail a constructor that otherwise succeeded -- and
        # `close` detaches `_handle` before closing it, so a raise here would
        # also lose the only reference to the handle. Written out rather than
        # folded into a `finally` with the arm above, because one shared
        # suppression would let a test of either path cover both.
        try:
            self.close()
        except Exception:
            pass

    def _open(self) -> None:
        """Open the source and publish it, or release what was opened."""
        # See the ownership note in the class docstring for why a path the
        # reader accepts is not merely equivalent to a handle opened from it.
        #
        # `Source.path` is handed over unchanged -- the identical string
        # object the caller passed, which is what makes the path branch the
        # same call it was before this module existed.
        handle = self._src.open() if self._needs_handle() else None
        target = handle if handle is not None else self._src.path
        if target is None:
            # `_needs_handle` said a path would be there and there is none, so
            # the override narrowed a contract its docstring states it must
            # not. Raised here rather than left to the reader: `None` reaches
            # `h5py` as "expected str, bytes or os.PathLike object", which
            # sends the caller looking for a bug in the path they passed.
            # This is the seam, so this is where it has to fail.
            raise TypeError(
                f"{type(self).__name__}._needs_handle() returned False for a "
                "pathless source; a source with no path has only a handle to "
                "offer. An override may widen that test but MUST NOT narrow "
                "it."
            )
        try:
            reader = self._reader_open(target)
        except BaseException:
            if handle is not None:
                handle.close()
            raise
        try:
            self._validate(reader)
        except BaseException:
            # The reader first: it reads from the handle lazily, so closing
            # the handle under it turns a clean refusal into a read error.
            # `finally`, so a reader whose close raises does not strand a
            # handle nothing has published yet and nothing can reach again.
            #
            # The reader's close failure is suppressed rather than allowed
            # out: `_validate` has already raised the refusal the caller
            # needs, and a `try/finally` here would propagate the teardown
            # error instead and never reach the `raise` below.
            try:
                try:
                    reader.close()
                except Exception:
                    pass
            finally:
                if handle is not None:
                    handle.close()
            raise
        # One statement, so `close` can never observe one without the other.
        # Publishing last is also what keeps the refusal above repeatable: a
        # rejected file is not cached, so the next access raises the same
        # `TilesetError` rather than a bare `KeyError` off a bad cached file.
        self._handle, self._reader = handle, reader

    def _needs_handle(self) -> bool:
        """Whether the reader must be handed a live handle, not a path.

        True for a factory-backed source, which has no path to hand over. A
        reader that accepts only *some* paths -- pybigtools dispatches on the
        file extension -- widens this. An override MUST NOT narrow it: a
        pathless source has nothing but a handle to offer, and `_open` has no
        path to fall back on.
        """
        return self._src.path is None

    def _reader_open(self, target: str | IO[bytes]) -> _R:
        """Open ``target``: a path string, or a handle this tileset owns."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _reader_open"
        )

    def _validate(self, reader: _R) -> None:
        """Refuse a file that opened but is not this tileset's format.

        Called with the reader open and nothing published, so raising here
        releases both the reader and any handle rather than caching a file
        whose every later use would fail differently from its first.
        """
