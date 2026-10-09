"""Where a tileset's bytes come from.

A tileset used to take a filesystem path and open it itself, which meant a
server fronting remote data had to download the file first and keep it on disk
for the tileset's lifetime. This module widens that to a second shape -- a
*factory*, a zero-argument callable returning a freshly opened binary handle --
so the caller owns the fetch and clodius owns only the tiling.

A factory rather than an open handle, because the readers disagree about what
they want and only one of the two shapes satisfies both. oxbow may reopen its
source between queries, so it takes a callable; h5py and pybigtools want a live
object. A factory yields a live object by being called once, so it serves
everyone. A handle cannot be un-consumed, so it serves only half of them -- and
the failure would not surface until the second read of an already-constructed
tileset. :meth:`Source.coerce` therefore refuses a handle outright, with a
message naming the fix, rather than accepting one and hoping.

The handle a factory returns must be **fresh, seekable, binary and
closeable**. All four are load-bearing and all four are checked, in
:meth:`Source.open`, rather than left to fail inside a reader: ``h5py``'s
file-like driver seeks, :meth:`Source.size` seeks and tells, a text handle
yields ``str`` where every consumer reads ``bytes``, and the ownership rule
below means clodius calls ``close`` on every handle it was handed. A reader's
own complaint about any of them names neither the tileset nor the fix --
``h5py`` reports a *missing seek* as "expected str, bytes or os.PathLike
object", which sends the caller looking for a path bug.

Those four names, and no more, are what `BinaryHandle` publishes.

Nothing here imports ``fsspec``. ``Callable[[], BinaryHandle]`` is a stdlib
contract -- four methods every binary file object has -- that an ``fsspec``
filesystem happens to satisfy
(``lambda: fs.open(url, "rb")``), and so do ``smart_open``, ``boto3`` and a
plain ``lambda: open(p, "rb")``. Staying agnostic is the point.

Ownership
---------
clodius closes what clodius opened, and nothing else. Every handle a tileset
holds came from calling the factory, so every handle is the tileset's to close.
The caller's callable is not a resource and needs no disposal. This is the
whole reason a handle is refused: once a caller's open file is erased into
``lambda: handle``, which of the two owns it is unrecoverable.

A *path* source is not opened here at all where a reader accepts a path --
:attr:`Source.path` hands back the path string it was given, so the reader
makes the same call it would have made without this module. That is what keeps
HDF5 on its native driver, and what keeps N tilesets over one file to one
descriptor rather than N.
"""

from __future__ import annotations

import inspect
import io
import os
from typing import IO, Callable, Protocol


class BinaryHandle(Protocol):
    """What this module requires of a handle, and all of it.

    Exactly the four names :meth:`Source._check` enforces. That is less than
    ``IO[bytes]`` declares, and the narrower type is the honest one: a handle
    reaching clodius is read, seeked, told and closed, and nothing in the
    package writes to one, flushes one, iterates one or enters one. Annotating
    :meth:`Source.open` as ``IO[bytes]`` promised a caller two dozen members
    that a conforming source is never asked for and a minimal one does not
    have, so `with source.open() as h:` type-checked and raised.

    Narrowed rather than widening `_check` to everything ``IO[bytes]``
    declares: both readers this package uses drive a minimal handle happily,
    so widening the check would refuse a source the library demonstrably
    serves. The published type follows the enforcement, not the other way
    round.

    Deliberately not ``runtime_checkable``. `isinstance` against a Protocol
    tests only that the names are present -- which `_check` already does, and
    `_check` goes on to probe the capabilities and name which one is missing,
    which `isinstance` cannot.
    """

    def read(self, size: int = -1, /) -> bytes: ...

    def seek(self, offset: int, whence: int = os.SEEK_SET, /) -> int: ...

    def tell(self) -> int: ...

    def close(self) -> None: ...


#: What a tileset will accept wherever it used to take a path. ``Source`` is
#: a member because `Source.coerce` passes an already-normalized one straight
#: back out, and `Source.sibling` and `Source.optional` both return one -- so
#: a caller who normalizes once and reuses the result was writing code the
#: annotation rejected and the runtime accepted.
#:
#: A PEP 695 ``type`` statement rather than a plain assignment, because
#: ``Source`` is defined below. An assignment is evaluated eagerly -- the
#: ``from __future__ import annotations`` above defers annotations, not
#: assignments -- so naming the class as a string there produced a
#: ``ForwardRef`` that only resolved inside this module. Every consumer
#: imports this alias without ``Source``, so `typing.get_type_hints` raised
#: on all four tileset constructors and both checkers rejected the alias as
#: not a type. A ``type`` statement is lazy and resolves in its defining
#: module's namespace, which is this one.
type SourceLike = (
    str | bytes | os.PathLike | Callable[[], BinaryHandle] | Source
)


class Source:
    """A filesystem path, or a factory for a binary handle.

    Two entry points, and the difference is what each is handed rather than
    what each enforces. :meth:`coerce` takes a caller's *untyped* argument --
    whatever arrived at a tileset constructor -- and decides which shape it is,
    refusing one that is neither. This constructor takes the shape already
    decided, and holds every invariant `coerce` would have: a path is decoded
    and must be non-empty, a factory must be callable, and exactly one of the
    two is given. So a ``Source`` is normalized no matter which door built it,
    which is what lets :meth:`coerce` pass one straight back out.

    :param path: A filesystem path, decoded with `os.fsdecode`. Mutually
        exclusive with ``factory``.
    :param factory: A zero-argument callable returning a freshly opened,
        seekable binary handle. Mutually exclusive with ``path``.
    :raises ValueError: If both or neither is given, or the path is empty.
    :raises TypeError: If ``factory`` is not callable.

    The two shapes are deliberately not unified into "always a factory". A
    path-backed source hands the reader back *the path string it was given*
    (see :attr:`path`), so the existing code path is not merely
    equivalent to what it was before this module existed -- it is the same
    call with the same argument. That is what makes adopting this safe.
    """

    __slots__ = ("_path", "_factory")

    _path: str | None
    _factory: Callable[[], BinaryHandle] | None

    def __init__(
        self,
        *,
        path: str | os.PathLike | bytes | None = None,
        factory: Callable[[], BinaryHandle] | None = None,
    ) -> None:
        # Keyword-only, and normalizing: this is a public constructor, so the
        # invariants `coerce` establishes cannot live only in `coerce`. A
        # factory passed positionally used to land in the `path` slot, and the
        # failure surfaced as a `__str__` that raises -- from inside the
        # error-reporting path that exists to name the file.
        if (path is None) == (factory is None):
            raise ValueError("a Source is exactly one of a path or a factory")
        if factory is not None:
            # Both guards live here rather than in `coerce`, because this
            # constructor is public: `coerce` passes an already-normalized
            # `Source` straight back out, and `Source` is a member of
            # `SourceLike`, so a factory reaching a tileset through this door
            # is never seen by `coerce` at all. Checked there and not here,
            # the arity guard was bypassable by the documented shape.
            if not callable(factory):
                raise TypeError(
                    "a Source factory must be callable; got "
                    f"{type(factory).__name__}. A zero-argument callable "
                    "opening a fresh handle (e.g. lambda: fs.open(url, "
                    "'rb')), not an already open file."
                )
            # Arity, not just callability. The contract is a *zero-argument*
            # callable, and `builtins.open` is the one that gets passed by
            # mistake -- it is callable, so it was accepted and failed
            # several frames later as "open() missing required argument
            # 'file'", a message naming neither the tileset nor the fix.
            #
            # After the callability test, never before it: `signature` raises
            # `TypeError` for a non-callable too, so checking arity first
            # reports an already-open handle as taking arguments.
            try:
                inspect.signature(factory).bind()
            except TypeError as exc:
                name = getattr(factory, "__name__", type(factory).__name__)
                raise TypeError(
                    "a tileset's source callable must take no arguments; "
                    f"{name} requires some. Bind them first, e.g. "
                    "functools.partial(fs.open, url, 'rb') or "
                    "lambda: fs.open(url, 'rb')."
                ) from exc
            except ValueError:
                # A C callable with no introspectable signature. Nothing can
                # be checked, so let it through rather than refuse a shape
                # that may well be correct.
                pass
        # `fsdecode`, not `fspath`: `__fspath__` may return bytes, and `fspath`
        # preserves them. A bytes `_path` satisfies none of this class's own
        # annotations, and `__str__` would raise `TypeError` from inside the
        # error-reporting path that names the file.
        decoded = None if path is None else os.fsdecode(path)
        if decoded is not None and not decoded:
            # An empty path is not a source anyone meant, and it reads as one
            # all the way down: a `Source` is unconditionally truthy, so a
            # consumer's empty-source guard does not fire, and it surfaces much
            # later as `FileNotFoundError` from `open("")`.
            raise ValueError(
                "a tileset needs a non-empty path; got an empty one"
            )
        self._path = decoded
        self._factory = factory

    def __str__(self) -> str:
        """The path, so error messages naming a file keep naming it."""
        if self._path is not None:
            return self._path
        return "<file-like source>"

    def __repr__(self) -> str:
        return f"<Source {self}>"

    @property
    def path(self) -> str | None:
        """The filesystem path, or ``None`` for a factory-backed source.

        ``None`` is the signal that anything derived from a *filename* --
        a sibling index, an extension sniff, a path-only library call -- has
        no answer here and must not be guessed at.
        """
        return self._path

    @classmethod
    def coerce(cls, obj: SourceLike) -> Source:
        """Normalize whatever a constructor was handed.

        Accepts a path, a factory, or an already-normalized ``Source`` -- the
        last so that a tileset delegating to a helper does not have to care
        whether normalization already happened.

        An open file is refused rather than accepted, because the obvious
        repair a caller reaches for, ``lambda: handle``, is a factory that
        lies: it returns the same exhausted handle every time. A reader that
        reopens would read from wherever the previous query left the cursor.
        Raising here turns that into a message at construction.
        """
        if isinstance(obj, Source):
            return obj
        # `bytes` alongside `str` and `PathLike`: it is one of the three path
        # representations `os` accepts, every other tileset normalizes it, and
        # h5py took it directly before this module existed. `fsdecode` already
        # handles all three, so admitting only two narrowed the parameter at
        # the moment this module's whole purpose was to widen it.
        if isinstance(obj, (str, bytes, os.PathLike)):
            # Decoding and the non-empty guard both live in `__init__`, which
            # is public and must hold them whichever door a caller uses.
            return cls(path=obj)
        # Callability is tested first because `read` is an attribute of a
        # handle *class* as much as of an instance, so probing for it first
        # tells someone who passed `io.BytesIO` -- a perfectly good factory --
        # that they passed an open file. Open files are not callable, so
        # nothing that should be refused reaches this branch.
        if callable(obj):
            # Callability and arity are both established by `__init__`, which
            # is public and must hold them whichever door a caller uses --
            # the same reason the path branch above delegates its decoding
            # and its non-empty test rather than repeating them here.
            return cls(factory=obj)
        if hasattr(obj, "read"):
            raise TypeError(
                "a tileset takes a path or a callable that opens a fresh "
                "handle (e.g. lambda: fs.open(url, 'rb')), not an already "
                f"open file. Got {type(obj).__name__}. Wrapping it as "
                "`lambda: handle` will not work: the reader may reopen its "
                "source, and would get the same exhausted handle back."
            )
        raise TypeError(
            "a tileset takes a path or a callable that opens a fresh binary "
            f"handle, not {type(obj).__name__}"
        )

    @classmethod
    def optional(cls, obj: SourceLike | None) -> Source | None:
        """:meth:`coerce`, but ``None`` passes through.

        For the companion inputs -- an index, a reference, a ``.fai`` -- which
        are genuinely optional and whose absence means "go and look for one".
        """
        return None if obj is None else cls.coerce(obj)

    def open(self) -> BinaryHandle:
        """A fresh binary handle, for a reader that wants a live object.

        The caller owns what this returns -- it was opened here, on the
        caller's behalf -- and must close it.

        A handle that is not seekable, or not binary, is refused here with the
        missing capability named. Both are requirements of the readers this
        serves rather than preferences of this class, and a reader that meets
        one of them for the first time several frames later reports it as
        something else entirely.

        Returns a `BinaryHandle` rather than an ``IO[bytes]``, because that is
        what :meth:`_check` enforces. The wider annotation promised members a
        conforming handle is never asked for and a minimal one does not have.
        """
        if self._path is not None:
            # Unbuffered because buffering measures as a no-op on the one
            # route that reaches this line, not because of anything about any
            # particular reader. `FileBacked._open` is this method's only
            # library caller and it is guarded by `_needs_handle()`, which is
            # `path is None` for both HDF5 backends -- so a path-backed cooler
            # or multivec never arrives here at all, and the only path-backed
            # source that does is a BBI file whose suffix `pybigtools` does
            # not dispatch on. Measured there: identical read counts and
            # identical bytes buffered or not, wall time inside noise.
            #
            # No mechanism is claimed on purpose. Three independent
            # measurements of h5py's read sizes through a Python handle gave
            # medians of 64 B, 512 B and 80,000 B, and all three were right --
            # read size tracks the dataset's chunk layout, not the library --
            # so any rationale of that shape is a property of one file.
            return open(self._path, "rb", buffering=0)
        handle = self._factory()
        try:
            self._check(handle)
        except Exception:
            # The factory opened it, which makes it this module's to release;
            # the ownership rule above does not get an exception for the path
            # that rejects the thing. Guarded rather than bare, and the guard
            # is not redundant with `close` now being a checked capability --
            # this arm is reached precisely when `_check` FAILED, which
            # includes failing *because* `close` is missing.
            try:
                handle.close()
            except Exception:
                pass
            raise
        return handle

    def open_text(self, encoding: str = "utf-8") -> IO[str]:
        """A fresh text handle, for the few sidecars that are text.

        A ``.fai`` is the only one today. Wrapping rather than reopening in
        text mode keeps the factory contract binary-only, so a caller writes
        one kind of callable regardless of what the file turns out to be.

        This is the one method that asks for more than `BinaryHandle`:
        `io.TextIOWrapper` calls ``readable()``, ``writable()`` and
        ``seekable()`` on what it wraps, none of which :meth:`_check`
        requires. So a minimal conforming handle is admitted by :meth:`open`
        and refused here, as an `AttributeError` from inside the wrapper. The
        divergence is named rather than closed: widening `_check` would refuse
        handles both readers this package uses accept, to serve a sidecar
        shape no caller in the tree reaches.
        """
        if self._path is not None:
            return open(self._path, encoding=encoding)
        return io.TextIOWrapper(self.open(), encoding=encoding)

    def size(self) -> int | None:
        """Bytes, or ``None`` when the source cannot be measured.

        Backs the ``max_scan_bytes`` ceiling. ``None`` means the question was
        unanswerable, not that the file is empty -- a seekable handle answers
        it (``fsspec``'s do), and only a genuinely streaming source does not.

        A caller enforcing a ceiling must refuse an unmeasurable source with
        ``TilesetUnavailable`` rather than read ``None`` as "no limit": the
        latter lets anyone past a stated limit by wrapping their path in a
        callable. Note the three unlike conditions this collapses together --
        a missing path, an unseekable stream, and a factory that raised -- so
        a failed remote fetch is not distinguishable here from a pipe.
        """
        if self._path is not None:
            # `isfile` before `getsize`: `getsize` succeeds on a directory and
            # hands back its inode size, a small plausible number the ceiling
            # would wave through -- and the source then fails much later, as
            # `IsADirectoryError` out of `open()`. Only a regular file has a
            # length worth comparing against a byte limit.
            if not os.path.isfile(self._path):
                return None
            try:
                return os.path.getsize(self._path)
            except OSError:
                return None
        try:
            handle = self.open()
        except Exception:
            return None
        try:
            # `seek` then `tell` rather than trusting `seek`'s return: the
            # io protocol promises the new position, but a file-like is only
            # required to be duck-compatible and some return None.
            handle.seek(0, os.SEEK_END)
            return handle.tell()
        except Exception:
            return None
        finally:
            try:
                handle.close()
            except Exception:
                pass

    def sibling(self, *suffixes: str) -> Source | None:
        """The first existing ``<path><suffix>``, or ``None``.

        Always ``None`` for a factory-backed source, and deliberately so: a
        factory is an opaque recipe with no surrounding namespace to probe, so
        there is no honest way to guess where its index lives. A tileset that
        needs one must say that it found none rather than invent a filename --
        which is what makes the caller pass it explicitly.
        """
        if self._path is None:
            return None
        for suffix in suffixes:
            candidate = self._path + suffix
            # `isfile`, not `exists`: a directory exists and would be returned
            # as an index that fails on first open, and `has_sibling` is what
            # decides whether a tileset range-queries or scans. `exists`
            # already follows symlinks, so a broken link is rejected -- this
            # only makes the two agree.
            if os.path.isfile(candidate):
                return type(self)(path=candidate)
        return None

    def has_sibling(self, *suffixes: str) -> bool:
        """Whether :meth:`sibling` would find one."""
        return self.sibling(*suffixes) is not None

    @staticmethod
    def _check(handle: object) -> None:
        """Refuse a handle a reader could not use, naming what is missing."""
        # `close` is required, not optional. The module's ownership rule is
        # that clodius closes what clodius opened, and `FileBacked.close`
        # calls `handle.close()` on the handle a factory produced -- so a
        # handle without one satisfied this check and then failed from inside
        # `close()`, as a bare `AttributeError`, with `_handle` already
        # cleared and the handle unreachable. Construction is where that
        # surfaces now that every backend releases its header read.
        #
        # `__enter__`/`__exit__` are deliberately NOT required, even though
        # `Source.open()` is used as a context manager in tests. Nothing in
        # the library enters a handle -- `FileBacked` only ever closes one --
        # so requiring them would refuse a conforming handle to serve a
        # convenience this package does not depend on.
        missing = [
            name
            for name in ("read", "seek", "tell", "close")
            if not hasattr(handle, name)
        ]
        if missing:
            raise TypeError(
                "the source callable must return an open, seekable binary "
                f"handle; {type(handle).__name__} is missing "
                f"{', '.join(missing)}"
            )
        # Both checks below test the capability *in addition to* its
        # declaration. Neither form subsumes the other: a `seekable()` may
        # simply be absent, and `isinstance(handle, io.TextIOBase)` misses a
        # handle that returns `str` without inheriting from `io` -- while a
        # declaration that contradicts the capability is itself a refusal,
        # because a reader asking `seekable()` first will believe it.
        #
        # The asymmetry is worth naming: for text, the nominal check catches
        # what the probe cannot (a `read(0)` that raises leaves `probe` as
        # `None`). For seek, the probe is the stronger test in every
        # realistic case, and the declaration arm adds one refusal class --
        # a handle that declares `False` and seeks anyway.
        seekable = getattr(handle, "seekable", None)
        declared = None
        if callable(seekable):
            try:
                declared = bool(seekable())
            except Exception:
                # A `seekable()` that raises has declared nothing; the probe
                # below is then the only evidence either way.
                declared = None
        try:
            handle.seek(0, os.SEEK_CUR)
        except Exception as exc:
            # A no-op seek: it moves nothing, so a handle that can seek pays
            # nothing for it and one that cannot says so here.
            raise TypeError(
                "the source callable must return a seekable handle; "
                f"{type(handle).__name__} refused a seek ({exc})"
            ) from exc
        if declared is False:
            raise TypeError(
                "the source callable must return a seekable handle; "
                f"{type(handle).__name__}.seekable() reports False. Its seek "
                "happens not to raise, but a reader that asks before seeking "
                "will believe the declaration."
            )
        # A zero-length read, so this consumes nothing of the stream and still
        # reveals which type the handle yields. Where it cannot run it proves
        # nothing and must not refuse: a handle may implement `seek`/`tell`
        # without being readable yet -- `size()` asks only those two of it --
        # and a `RawIOBase` subclass that leaves `readinto` abstract raises
        # here. So this strengthens the nominal check below rather than
        # replacing it.
        try:
            probe = handle.read(0)
        except Exception:
            probe = None
        if isinstance(probe, str) or isinstance(handle, io.TextIOBase):
            raise TypeError(
                "the source callable must return a binary handle, not a text "
                f"one; {type(handle).__name__} yields str. Open it with 'b'."
            )
