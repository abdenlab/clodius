"""The open-and-release protocol a file-backed tileset shares.

The HDF5 and BBI tilesets differ in which reader they hand their bytes to and
agree completely on how they open and close one. That agreement used to be two
hand-written copies, and every gap in the protocol was a gap in one copy or
the other -- a publication order that a concurrent `close` could slip between,
a constructor that orphaned its handle, an unlocked check-then-act, a reader
dropped for the garbage collector rather than closed. It lives here instead,
so a fix lands once and the next backend inherits it.

That includes the constructor guard, which subclasses used to hand-write: a
subclass does its construction work in :meth:`FileBacked._configure`, which
`__init__` wraps, so a raise after the open releases what the open produced
without each subclass remembering to arrange it.

Lives in ``tiles_v2`` rather than beside `clodius.core.source.Source`, which
it builds on, because a subclass names its reader's type and ``clodius.core``
must not depend on ``h5py`` or ``pybigtools``.
"""

from __future__ import annotations

import threading
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

    Opens lazily, on first access to :attr:`file`, so constructing a tileset
    over an unreachable source is not itself an error.

    Whether *registration* is free is the subclass's choice, not this class's.
    A subclass whose :meth:`_configure` reads a header has opened the source
    by the time construction returns, and may close again to hand the
    descriptor -- or the caller's remote connection -- back; `BBITileset`
    does. `CoolerTileset` and `MultivecTileset` keep theirs open, so for those
    two a registered tileset holds its source until it is closed.

    **Ownership.** Where the reader accepts a path, a path-backed source is
    handed the path string, exactly as a reader that had never heard of
    `Source` would do -- :meth:`_needs_handle` decides which sources those
    are. For HDF5 that is not merely equivalent to handing over a handle
    opened from it: HDF5 recognizes that the same file is already open
    read-only in the process and shares one descriptor across every tileset,
    which it cannot do for a Python file object. That reasoning is HDF5's and
    does not generalize: ``pybigtools`` shares nothing between readers of one
    path either way -- measured at one descriptor per reader for both routes
    -- so for BBI the path branch is a one-time saving of a few microseconds
    rather than a per-descriptor one. Only a source the reader cannot take as
    a path produces a handle, and that
    handle is this tileset's to close, because this tileset is what opened it.
    :meth:`close` releases both it and the reader, which do not release each
    other: neither ``h5py.File.close()`` nor ``BBIReader.close()`` closes a
    Python file object it was handed.

    **Lifecycle.** :meth:`close` is idempotent, and a tileset reopens on next
    use -- `BaseTileset.__exit__` closes, and a tileset used again after its
    ``with`` block gets a second handle, owned on the same terms as the first.
    """

    _reader: _R | None
    _handle: IO[bytes] | None
    _lock: threading.Lock
    _read_lock: threading.Lock

    def __init__(self, source: SourceLike) -> None:
        self._src = Source.coerce(source)
        self._reader = None
        self._handle = None
        self._lock = threading.Lock()
        # Separate from `_lock`, which guards open and close. A reader lock
        # has to be takeable around a *use* of the reader, and a use goes
        # through `reader`, which takes `_lock` when it has to open -- so
        # sharing one non-reentrant lock between the two would deadlock the
        # first serve after a close.
        self._read_lock = threading.Lock()
        # The guard subclasses used to hand-write, three times. `close` is
        # idempotent and null-safe, so it covers whatever a subclass does in
        # `_configure` -- including a subclass that adds a parameter later.
        #
        # `BaseException`, not `Exception`: a KeyboardInterrupt or a
        # CancelledError landing between the open and the end of construction
        # orphans what was opened exactly as a ValueError does, and the
        # handler re-raises, so widening it costs nothing.
        try:
            self._configure()
        except BaseException:
            self.close()
            raise

    @property
    def file(self) -> _R:
        """The open reader, opening it on first access."""
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
        # deliberately unlocked, so the returned reader can be closed under
        # its reader, which surfaces as the backend's own closed-file error.
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

    def _open(self) -> None:
        """Open the source and publish it, or release what was opened."""
        # See the ownership note in the class docstring for why a path the
        # reader accepts is not merely equivalent to a handle opened from it.
        #
        # `Source.path` rather than `Source.for_reader()`: the latter is
        # oxbow's union and returns the *factory* for a pathless source, which
        # is a shape no reader here accepts. Reading `path` keeps the value
        # and `_reader_open`'s annotation in agreement, so a subclass that
        # narrows `_needs_handle` fails at the seam instead of handing a
        # callable to a reader.
        handle = self._src.open() if self._needs_handle() else None
        target = handle if handle is not None else self._src.path
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
            # `finally`, for the same reason `close` uses one -- a reader
            # whose close raises must not strand a handle nothing has
            # published yet, and so nothing can ever reach again.
            try:
                reader.close()
            finally:
                if handle is not None:
                    handle.close()
            raise
        # One statement, so `close` can never observe one without the other.
        # Publishing last is also what keeps the refusal above repeatable: a
        # rejected file is not cached, so the next access raises the same
        # `TilesetError` rather than a bare `KeyError` off a bad cached file.
        self._handle, self._reader = handle, reader

    def _configure(self) -> None:
        """Construction work that may read through :attr:`file`.

        Called by `__init__` inside a guard that closes what this opened if
        it raises, so a subclass neither writes that guard nor leaves the
        caller's handle orphaned when the file turns out not to be its
        format.

        An implementation that opens the source to read a header MAY close it
        before returning, which makes registration free at the cost of a
        second open for a tileset that is actually served. That is a per
        backend trade and this class does not make it.
        """

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
