"""The open-and-release protocol the HDF5-backed tilesets share.

`CoolerTileset` and `MultivecTileset` differ in how they read an HDF5 file and
agree completely on how they open and close one. That agreement used to be two
hand-written copies, and every gap in the protocol was a gap in one copy or the
other -- a publication order that a concurrent `close` could slip between, a
constructor that orphaned its handle, an unlocked check-then-act. It lives here
instead, so a fix lands once and the next HDF5 backend inherits it.

Lives in ``tiles_v2`` rather than beside `clodius.core.source.Source`, which it
builds on, because ``clodius.core`` must not depend on ``h5py``.
"""

from __future__ import annotations

import threading
from typing import IO

import h5py

from clodius.core.source import Source, SourceLike
from clodius.core.tileset import BaseTileset


class H5Backed(BaseTileset):
    """A tileset whose bytes are one HDF5 file.

    Opens lazily, on first access to :attr:`file`, so constructing a tileset
    over an unreachable source is not itself an error and a server can register
    one without paying for a fetch.

    **Ownership.** A path-backed source is handed to ``h5py`` as the path
    string, exactly as a reader that had never heard of `Source` would do, so
    HDF5 opens it natively through ``sec2`` and recognizes that the same file is
    already open elsewhere in the process -- one descriptor per file rather than
    one per tileset. Only a factory-backed source produces a handle, and that
    handle is this tileset's to close, because this tileset is what called the
    factory. :meth:`close` releases both it and the ``h5py.File``, which do not
    release each other: ``h5py.File.close()`` leaves a Python file object it was
    handed open.

    **Lifecycle.** :meth:`close` is idempotent, and a tileset reopens on next
    use -- `BaseTileset.__exit__` closes, and a tileset used again after its
    ``with`` block gets a second handle, owned on the same terms as the first.
    """

    _file: h5py.File | None
    _handle: IO[bytes] | None
    _lock: threading.Lock

    def __init__(self, source: SourceLike) -> None:
        self._src = Source.coerce(source)
        self._file = None
        self._handle = None
        self._lock = threading.Lock()

    @property
    def file(self) -> h5py.File:
        """The open ``h5py.File``, opening it on first access."""
        # Double-checked: the unsynchronized read is the whole-lifetime common
        # case, and the lock is what keeps a concurrent first access from
        # calling the factory N times and leaving N-1 handles that `close` can
        # no longer reach. A caller's handle has no finalizer to fall back on.
        # Bound once, and returned from the binding. Reading the attribute
        # again on the way out is a second, unsynchronized load, and a `close`
        # landing between the two hands the caller `None` through a signature
        # that promises an `h5py.File`. The read inside the lock makes the
        # result non-`None` by construction.
        #
        # What this does NOT make safe is closing a tileset while another
        # thread is reading through the reference it already has: the fast path
        # is deliberately unlocked, so the returned file can be closed under
        # its reader, which surfaces as `KeyError` from the first subscript.
        # Widening the lock to cover the caller's use would serialize every
        # reader to fix a caller error, so it is not done -- a tileset closed
        # concurrently with a request is the caller's race, not this class's.
        file = self._file
        if file is None:
            with self._lock:
                if self._file is None:
                    self._open()
                file = self._file
        return file

    def close(self) -> None:
        """Release the ``h5py.File`` and any handle this tileset opened."""
        with self._lock:
            try:
                if self._file is not None:
                    self._file.close()
            finally:
                # `finally`, because a failed HDF5 close must not strand the
                # handle: leaving `_file` set would make every retry raise at
                # the same point and the handle would never be reached again.
                self._file = None
                handle, self._handle = self._handle, None
                if handle is not None:
                    handle.close()

    def _open(self) -> None:
        """Open the source and publish it, or release what was opened."""
        # A path goes to `h5py` as the path string -- see the ownership note in
        # the class docstring for why that is not merely equivalent to handing
        # it a handle. Only a factory source yields something to own.
        handle = None if self._src.path is not None else self._src.open()
        target = self._src.for_reader() if handle is None else handle
        try:
            file = h5py.File(target, "r")
        except Exception:
            if handle is not None:
                handle.close()
            raise
        try:
            self._validate(file)
        except Exception:
            file.close()
            if handle is not None:
                handle.close()
            raise
        # One statement, so `close` can never observe one without the other.
        # Publishing last is also what keeps the refusal above repeatable: a
        # rejected file is not cached, so the next access raises the same
        # `TilesetError` rather than a bare `KeyError` off a bad cached file.
        self._handle, self._file = handle, file

    def _validate(self, file: h5py.File) -> None:
        """Refuse a file that opened but is not this tileset's format.

        Called with the file open and nothing published, so raising here
        releases both the file and any handle rather than caching a file whose
        every later use would fail differently from its first.
        """
