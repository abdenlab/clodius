"""Tests for clodius.core.source.

Every tileset's first argument passes through :meth:`Source.coerce`, so this
module is where the accepted shapes are pinned. Three of its promises are the
ones the rest of the package rests on.

The first is that a path-backed source is not merely *equivalent* to the old
behaviour -- ``for_reader`` hands the reader back the same path string object
it was given, so the existing call is the same call. A regression there would
be invisible in the new remote tests and would break every existing caller.

The second is that an already-open file is refused. The repair a caller
reaches for is ``lambda: handle``, a factory that lies: it returns the same
exhausted handle every time, so a reader that reopens its source reads from
wherever the last query left the cursor. That is a wrong-answers bug rather
than a crash, and it would surface on the second tile of an already-built
tileset, so it is caught at construction instead.

The third is that every question about a *filename* -- how big is it, is there
an index beside it -- is answered honestly or not at all. ``size`` and
``sibling`` back a scan ceiling and an indexed/scanning decision respectively,
and both once answered for things that are not files: ``getsize`` succeeds on
a directory and ``exists`` is true of one, so each handed back a confident
wrong answer that failed much later, somewhere else.
"""

import io
import os
import pathlib

import fsspec
import pytest

from clodius.core.source import Source

from ..source_helpers import Unseekable, handle_factory, recording_factory

#: Written by :func:`write` unless a test asks for something else. Asserted
#: literally rather than compared between two reads, so that a source
#: returning *nothing* cannot satisfy a test by matching another empty read.
CONTENT = "c1\t0\t5\n"


def write(path, text=CONTENT):
    """Write a small file and return its path as a string."""
    path.write_text(text)
    return str(path)


class SeekReturnsNothing(io.RawIOBase):
    """A handle whose ``seek`` reports nothing, as a duck-typed one may."""

    def __init__(self, length):
        self._length = length
        self._pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=os.SEEK_SET):
        self._pos = self._length if whence == os.SEEK_END else offset

    def tell(self):
        return self._pos


class BytesPath:
    """An ``os.PathLike`` whose ``__fspath__`` yields bytes, as it may."""

    def __init__(self, path):
        self._path = path

    def __fspath__(self):
        return os.fsencode(self._path)


# --- construction -----------------------------------------------------------


def test___init___should_refuse_both_a_path_and_a_factory():
    """Test the invariant that makes the shape safe to branch on.

    Given:
        Both a path and a factory.
    When:
        A source is constructed directly.
    Then:
        It should raise ``ValueError``. Every accessor discriminates on
        whether the path is set, so a source holding both would take the path
        branch while carrying a factory nothing would ever call.
    """
    # Act & assert
    with pytest.raises(ValueError, match="exactly one"):
        Source(path="/tmp/x.bed", factory=lambda: io.BytesIO())


def test___init___should_refuse_neither_a_path_nor_a_factory():
    """Test the other half of the same invariant.

    Given:
        Neither a path nor a factory.
    When:
        A source is constructed directly.
    Then:
        It should raise ``ValueError`` rather than produce a source that
        reports no path and calls ``None`` when opened.
    """
    # Act & assert
    with pytest.raises(ValueError, match="exactly one"):
        Source()


def test_coerce_should_return_a_path_source_when_given_a_string(tmp_path):
    """Test the shape every existing caller passes.

    Given:
        A path as a plain string.
    When:
        It is coerced.
    Then:
        It should produce a source reporting that path, so everything keyed
        on having a filename -- a sibling index, an extension sniff -- still
        has one to work from.
    """
    # Arrange
    path = write(tmp_path / "records.bed")

    # Act
    source = Source.coerce(path)

    # Assert
    assert source.path == path


def test_coerce_should_return_a_path_source_when_given_a_pathlike(tmp_path):
    """Test that a ``Path`` is not treated as some third shape.

    Given:
        A path as an ``os.PathLike``.
    When:
        It is coerced.
    Then:
        It should produce the same path source a string would, since
        converting one is what the tilesets used to do themselves.
    """
    # Arrange
    path = tmp_path / "records.bed"
    write(path)

    # Act
    source = Source.coerce(path)

    # Assert
    assert source.path == str(path)


def test_coerce_should_decode_a_pathlike_that_yields_bytes(tmp_path):
    """Test the path type that used to break error reporting.

    Given:
        An ``os.PathLike`` whose ``__fspath__`` returns bytes, which the
        protocol permits.
    When:
        It is coerced and then rendered into a message.
    Then:
        It should hold a decoded string. ``os.fspath`` preserves bytes, and a
        bytes path made ``__str__`` raise ``TypeError`` from inside the very
        error message meant to name the offending file -- turning a clear
        domain error into an unrelated crash.
    """
    # Arrange
    path = write(tmp_path / "records.bed")

    # Act
    source = Source.coerce(BytesPath(path))

    # Assert
    assert source.path == path
    assert f"{source}" == path


def test_coerce_should_return_a_factory_source_when_given_a_callable(tmp_path):
    """Test the shape this whole module exists to admit.

    Given:
        A zero-argument callable returning an open binary handle.
    When:
        It is coerced.
    Then:
        It should produce a source with no path. The absent path is the
        signal that nothing may be derived from a filename, which is what
        stops a sibling index from being guessed at.
    """
    # Arrange
    path = write(tmp_path / "records.bed")

    # Act
    source = Source.coerce(handle_factory(path))

    # Assert
    assert source.path is None


def test_coerce_should_return_a_factory_source_when_given_a_handle_class():
    """Test a factory that the open-file refusal used to swallow.

    Given:
        A handle type, which is a perfectly good zero-argument factory.
    When:
        It is coerced.
    Then:
        It should produce a factory source. ``read`` is an attribute of a
        handle *class* as much as of an instance, so probing for it before
        testing callability told this caller they had passed an open file,
        and advised them to wrap it in a lambda that would not have helped.
    """
    # Act
    source = Source.coerce(io.BytesIO)

    # Assert
    assert source.path is None
    assert source.open().read() == b""


def test_coerce_should_return_the_same_object_when_already_a_source(tmp_path):
    """Test that normalizing twice is free.

    Given:
        A source that has already been coerced.
    When:
        It is coerced again.
    Then:
        It should be the identical object, so a tileset delegating to a
        helper need not track whether normalization already happened.
    """
    # Arrange
    source = Source.coerce(write(tmp_path / "records.bed"))

    # Act & assert
    assert Source.coerce(source) is source


def test_coerce_should_refuse_an_open_handle(tmp_path):
    """Test the refusal that keeps a lying factory out of the codebase.

    Given:
        An already-open binary file.
    When:
        It is coerced.
    Then:
        It should raise ``TypeError`` naming the callable form. Accepting the
        handle would mean either carrying an ownership flag through every
        backend, or erasing it to ``lambda: handle`` -- which returns the same
        exhausted handle on a reader's second read, and so serves records
        from wherever the cursor was left.
    """
    # Arrange
    path = write(tmp_path / "records.bed")

    # Act & assert
    with open(path, "rb") as handle:
        with pytest.raises(TypeError, match="not an already open file"):
            Source.coerce(handle)


def test_coerce_should_refuse_a_value_that_is_neither_a_path_nor_a_callable():
    """Test that a wrong type fails here rather than at the first read.

    Given:
        A value that is neither a path nor a callable.
    When:
        It is coerced.
    Then:
        It should raise ``TypeError`` naming both accepted shapes. Deferring
        this produces an error from inside a reader, naming a library the
        caller never called -- so the guidance, not merely the raise, is the
        thing worth pinning.
    """
    # Act & assert
    with pytest.raises(TypeError, match=r"path or a callable.*not int"):
        Source.coerce(42)


def test_optional_should_pass_none_through():
    """Test the companion inputs, which are genuinely absent sometimes.

    Given:
        No source at all.
    When:
        It is coerced as optional.
    Then:
        It should be ``None``, which the tilesets read as "go and look for
        one" rather than as a source that fails on use.
    """
    # Act & assert
    assert Source.optional(None) is None


# --- what the readers want --------------------------------------------------


def test_for_reader_should_return_the_identical_string_when_path_backed(
    tmp_path,
):
    """Test the property that makes adopting this safe for existing callers.

    Given:
        A path-backed source.
    When:
        Its reader argument is taken.
    Then:
        It should be the identical string object, not merely an equal one.
        The module's claim is that every ``ox.from_*``, ``h5py.File`` and
        ``pybigtools.open`` call receives the argument it received before this
        module existed -- so the path branch is the same code, not a
        re-implementation. An ``abspath`` normalisation would satisfy equality
        and quietly break that claim.
    """
    # Arrange
    path = write(tmp_path / "records.bed")

    # Act
    argument = Source.coerce(path).for_reader()

    # Assert
    assert argument is path


def test_for_reader_should_return_the_factory_when_factory_backed(tmp_path):
    """Test that the callable reaches the reader unwrapped.

    Given:
        A factory-backed source.
    When:
        Its reader argument is taken.
    Then:
        It should be the factory itself. oxbow declares exactly this union,
        so passing it through means there is no adapter to keep correct.
    """
    # Arrange
    factory = handle_factory(write(tmp_path / "records.bed"))

    # Act & assert
    assert Source.coerce(factory).for_reader() is factory


def test_open_should_read_the_file_for_either_shape(tmp_path):
    """Test the accessor the h5py and pybigtools readers are given.

    Given:
        The same file as a path source and as a factory source.
    When:
        Each is opened and read.
    Then:
        Both should yield the file's actual bytes. Asserting only that the
        two agree is satisfied by both yielding nothing, which is exactly
        what a broken ``open`` would do.
    """
    # Arrange
    path = write(tmp_path / "records.bed")

    # Act
    with Source.coerce(path).open() as left:
        from_path = left.read()
    with Source.coerce(handle_factory(path)).open() as right:
        from_factory = right.read()

    # Assert
    assert from_path == from_factory == CONTENT.encode()


def test_open_should_read_the_same_bytes_through_an_fsspec_filesystem(
    tmp_path,
):
    """Test the contract against the library callers will actually reach for.

    Given:
        A factory built the way an fsspec user writes one,
        ``lambda: fs.open(path, "rb")``.
    When:
        The source is opened and read.
    Then:
        It should yield the file's actual bytes. This is the claim the whole
        design rests on -- that a remote filesystem needs no clodius-side
        adapter -- and it is worth pinning against a real implementation
        rather than only against ``open``, since fsspec's handles are the
        ones that will carry the remote case.
    """
    # Arrange
    path = write(tmp_path / "records.bed")
    fs = fsspec.filesystem("file")

    # Act
    with Source.coerce(lambda: fs.open(path, "rb")).open() as handle:
        served = handle.read()

    # Assert
    assert served == CONTENT.encode()


def test_open_should_refuse_a_factory_that_returns_something_else():
    """Test the one thing a caller can still get wrong after construction.

    Given:
        A callable that returns a value which is not a handle.
    When:
        The source is opened.
    Then:
        It should raise ``TypeError``. ``coerce`` cannot catch this, since
        calling the factory to check would open the source -- possibly a
        network round trip -- at construction.
    """
    # Arrange
    source = Source.coerce(lambda: "not a handle")

    # Act & assert
    with pytest.raises(TypeError, match="is missing read, seek, tell"):
        source.open()


def test_open_text_should_decode_for_either_shape(tmp_path):
    """Test the accessor the ``.fai`` read needs.

    Given:
        A text sidecar as a path source and as a factory source.
    When:
        Each is opened as text.
    Then:
        Both should decode to the file's actual contents. Wrapping rather
        than reopening in text mode keeps the factory contract binary-only,
        so a caller writes one kind of callable whatever the file turns out
        to be.
    """
    # Arrange
    text = "c1\t1000\t4\t60\t61\n"
    path = write(tmp_path / "records.fai", text)

    # Act
    with Source.coerce(path).open_text() as left:
        from_path = left.read()
    with Source.coerce(handle_factory(path)).open_text() as right:
        from_factory = right.read()

    # Assert
    assert from_path == from_factory == text


def test_open_text_should_close_the_binary_handle_when_the_wrapper_closes(
    tmp_path,
):
    """Test that the text wrapper does not leak the handle underneath it.

    Given:
        A factory-backed source opened as text.
    When:
        The text wrapper is closed.
    Then:
        The binary handle should be closed too. This holds today only because
        ``TextIOWrapper`` closes what it wraps; a refactor to ``detach`` would
        turn every ``.fai`` read on a long-lived server into a leaked
        descriptor, and nothing would fail.
    """
    # Arrange
    factory, opened = recording_factory(write(tmp_path / "records.fai"))

    # Act
    with Source.coerce(factory).open_text() as handle:
        handle.read()

    # Assert
    assert opened and all(handle.closed for handle in opened)


# --- what the guards need ---------------------------------------------------


def test_size_should_agree_between_a_path_and_a_factory(tmp_path):
    """Test the measurement the scan ceiling is enforced against.

    Given:
        The same file as a path source and as a factory source.
    When:
        Each is measured.
    Then:
        Both should report the byte count ``os.path.getsize`` reports, and
        the factory source should release the handle it opened to do so. A
        factory that measured differently would move the ``max_scan_bytes``
        boundary depending on how the caller spelled the source; one that
        held the handle would leak a descriptor per ceiling check.
    """
    # Arrange
    path = write(tmp_path / "records.bed", "c1\t0\t5\nc1\t10\t15\n")
    expected = os.path.getsize(path)
    factory, opened = recording_factory(path)

    # Act
    from_path = Source.coerce(path).size()
    from_factory = Source.coerce(factory).size()

    # Assert
    assert from_path == from_factory == expected
    assert opened and all(handle.closed for handle in opened)


def test_size_should_measure_a_handle_whose_seek_returns_nothing():
    """Test the duck-typed handle the implementation deliberately allows for.

    Given:
        A factory returning a handle whose ``seek`` reports nothing, which
        the io protocol promises against but a duck-typed handle may do.
    When:
        It is measured.
    Then:
        It should still report the length, because the position is read back
        with ``tell`` rather than taken from ``seek``. Trusting ``seek``
        instead reports an unmeasurable source for a handle that could answer
        perfectly well.
    """
    # Arrange
    source = Source.coerce(lambda: SeekReturnsNothing(42))

    # Act & assert
    assert source.size() == 42


def test_size_should_return_none_when_the_path_is_a_directory(tmp_path):
    """Test the answer that used to defeat the ceiling entirely.

    Given:
        A path source naming a directory.
    When:
        It is measured.
    Then:
        It should be ``None``. ``os.path.getsize`` succeeds on a directory
        and hands back its inode size -- a small, plausible number that a
        ``max_scan_bytes`` check waves straight through, leaving the source
        to fail much later as ``IsADirectoryError`` from somewhere else.
    """
    # Arrange
    directory = tmp_path / "records.bed"
    directory.mkdir()

    # Act & assert
    assert Source.coerce(str(directory)).size() is None


def test_size_should_return_none_when_the_path_does_not_exist(tmp_path):
    """Test the divergence from ``os.path.getsize`` this introduces.

    Given:
        A path source naming a file that is not there.
    When:
        It is measured.
    Then:
        It should be ``None`` rather than raising ``FileNotFoundError`` as
        ``getsize`` does. A caller enforcing a ceiling has to decide what an
        unmeasurable source is worth, and this is one of the three unlike
        conditions that reach it as the same answer.
    """
    # Act & assert
    assert Source.coerce(str(tmp_path / "missing.bed")).size() is None


def test_size_should_return_none_when_the_factory_raises():
    """Test the most surprising of the three ways to be unmeasurable.

    Given:
        A factory that raises, as a failed remote fetch would.
    When:
        It is measured.
    Then:
        It should be ``None`` rather than propagating. A failed fetch is
        therefore indistinguishable here from a stream that cannot seek, so a
        caller refusing an unmeasurable source will describe a network
        failure in terms of scan ceilings.
    """

    # Arrange
    def factory():
        raise RuntimeError("fetch failed")

    # Act & assert
    assert Source.coerce(factory).size() is None


def test_size_should_return_none_when_the_source_cannot_seek():
    """Test the answer a caller must not read as "no limit".

    Given:
        A factory returning a handle that cannot seek, as a pure stream
        cannot.
    When:
        It is measured.
    Then:
        It should be ``None``, meaning the question was unanswerable rather
        than the file being empty. A caller that treated this as unlimited
        would let anyone bypass ``max_scan_bytes`` by wrapping a path in a
        callable.
    """
    # Arrange
    source = Source.coerce(lambda: Unseekable())

    # Act & assert
    assert source.size() is None


def test_sibling_should_prefer_the_earlier_suffix_when_both_exist(tmp_path):
    """Test the ordering the callers will inherit.

    Given:
        A path source with two candidate indexes beside it.
    When:
        Its siblings are searched, most conventional suffix first.
    Then:
        The earlier suffix should win. Creating only one file proves the loop
        skips a missing suffix but says nothing about precedence, and
        precedence is the part that matters: adopting this moves the choice
        of index from the reader to clodius.
    """
    # Arrange
    path = write(tmp_path / "records.bed.gz")
    write(tmp_path / "records.bed.gz.tbi")
    write(tmp_path / "records.bed.gz.csi")

    # Act
    sibling = Source.coerce(path).sibling(".tbi", ".csi")

    # Assert
    assert sibling.path == path + ".tbi"


def test_sibling_should_skip_a_suffix_that_is_missing(tmp_path):
    """Test the ordinary case of one index under a less common suffix.

    Given:
        A path source whose only index uses the second suffix searched.
    When:
        Its siblings are searched.
    Then:
        It should return that one, so the caller never rebuilds a filename
        itself.
    """
    # Arrange
    path = write(tmp_path / "records.bed.gz")
    write(tmp_path / "records.bed.gz.csi")

    # Act
    sibling = Source.coerce(path).sibling(".tbi", ".csi")

    # Assert
    assert sibling.path == path + ".csi"


def test_sibling_should_return_none_when_no_suffix_exists(tmp_path):
    """Test the ordinary unindexed file.

    Given:
        A path source with nothing beside it.
    When:
        Its siblings are searched.
    Then:
        It should be ``None``, which is how the tilesets decide a file has to
        be scanned rather than range-queried.
    """
    # Arrange
    source = Source.coerce(write(tmp_path / "records.bed"))

    # Act & assert
    assert source.sibling(".tbi", ".csi") is None


def test_sibling_should_return_none_when_the_suffix_matches_a_directory(
    tmp_path,
):
    """Test the match that produced an index nothing could open.

    Given:
        A path source beside a *directory* bearing the index suffix.
    When:
        Its siblings are searched.
    Then:
        It should be ``None``. ``os.path.exists`` is true of a directory, so
        this used to route a tileset down the indexed path with an index that
        fails on first open -- while a broken symlink was already rejected,
        making the method disagree with itself.
    """
    # Arrange
    path = write(tmp_path / "records.bed.gz")
    (tmp_path / "records.bed.gz.tbi").mkdir()

    # Act & assert
    assert Source.coerce(path).sibling(".tbi") is None


def test_sibling_should_return_none_for_a_factory(tmp_path):
    """Test the case that forces an index to be passed explicitly.

    Given:
        A factory-backed source, whose index happens to exist on disk.
    When:
        Its siblings are searched.
    Then:
        It should be ``None`` regardless. A factory is an opaque recipe with
        no surrounding namespace, so there is no honest way to guess where
        its index lives -- and guessing would silently range-query the wrong
        file for a caller whose remote layout differs from their local one.
    """
    # Arrange
    path = write(tmp_path / "records.bed.gz")
    write(tmp_path / "records.bed.gz.tbi")

    # Act & assert
    assert Source.coerce(handle_factory(path)).sibling(".tbi") is None


def test_has_sibling_should_report_true_when_a_suffix_exists(tmp_path):
    """Test the predicate the indexed/scanning split is decided on.

    Given:
        A path source with an index beside it.
    When:
        It is asked whether it has a sibling.
    Then:
        It should be true. Answering this wrongly does not fail -- it
        silently hands the reader an index that is not there.
    """
    # Arrange
    path = write(tmp_path / "indexed.bed.gz")
    write(tmp_path / "indexed.bed.gz.tbi")

    # Act & assert
    assert Source.coerce(path).has_sibling(".tbi", ".csi")


def test_has_sibling_should_report_false_when_no_suffix_exists(tmp_path):
    """Test the other half of the indexed/scanning split.

    Given:
        A path source with nothing beside it.
    When:
        It is asked whether it has a sibling.
    Then:
        It should be false. Answering this wrongly does not fail either -- it
        quietly serves every tile by scanning the whole file.
    """
    # Arrange
    path = write(tmp_path / "plain.bed")

    # Act & assert
    assert not Source.coerce(path).has_sibling(".tbi", ".csi")


# --- identity ---------------------------------------------------------------


def test___str___should_name_the_shape_when_factory_backed(tmp_path):
    """Test that a source with no filename still reads sensibly.

    Given:
        A factory-backed source.
    When:
        It is rendered into a message.
    Then:
        It should say so rather than render as an object repr, since these
        strings reach a server operator through a refusal message.
    """
    # Arrange
    source = Source.coerce(handle_factory(write(tmp_path / "records.bed")))

    # Act & assert
    assert f"{source}" == "<file-like source>"


def test___repr___should_name_the_source_it_wraps(tmp_path):
    """Test what a source looks like in a traceback.

    Given:
        A path-backed source.
    When:
        It is rendered for a developer rather than for a message.
    Then:
        It should name both the type and the path, since this is what reaches
        a log when a tileset raises. This also covers ``__str__``'s path
        branch, which it delegates to.
    """
    # Arrange
    path = write(tmp_path / "records.bed")

    # Act & assert
    assert repr(Source.coerce(path)) == f"<Source {path}>"


def test_open_should_refuse_a_text_handle(tmp_path):
    """Test the mistake a caller makes by forgetting the mode string.

    Given:
        A factory returning a handle opened in text mode, which satisfies
        every attribute a binary one does.
    When:
        The source is opened.
    Then:
        It should raise ``TypeError`` naming the mode, rather than letting the
        handle reach a reader that decodes its bytes and reports a
        ``UnicodeDecodeError`` from several frames away.
    """
    # Arrange
    path = write(tmp_path / "records.bed")
    source = Source.coerce(lambda: open(path))

    # Act & assert
    with pytest.raises(TypeError, match="binary handle, not a text one"):
        source.open()


def test_open_should_refuse_a_handle_that_cannot_seek(tmp_path):
    """Test the capability every consumer of this handle relies on.

    Given:
        A factory returning a readable handle that reports it cannot seek.
    When:
        The source is opened.
    Then:
        It should raise ``TypeError`` naming seekability. ``h5py``'s file-like
        driver and :meth:`Source.size` both seek, and h5py reports the absence
        as a complaint about paths.
    """
    # Arrange
    source = Source.coerce(lambda: Unseekable())

    # Act & assert
    with pytest.raises(TypeError, match="seekable"):
        source.open()


def test_open_should_close_the_handle_it_refuses():
    """Test that the ownership rule holds on the path that rejects.

    Given:
        A factory returning a closeable object that is not a usable handle.
    When:
        The source is opened and the refusal raises.
    Then:
        It should have closed the object first. The factory call opened it, so
        by this module's own ownership rule it is this module's to release --
        and a remote handle has no finalizer to fall back on.
    """
    # Arrange
    opened = []

    class Closeable:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    def factory():
        handle = Closeable()
        opened.append(handle)
        return handle

    source = Source.coerce(factory)

    # Act
    with pytest.raises(TypeError):
        source.open()

    # Assert
    assert [handle.closed for handle in opened] == [True]


def test_coerce_should_return_a_path_source_when_given_bytes(tmp_path):
    """Test the third path representation ``os`` accepts.

    Given:
        A filesystem path as ``bytes``.
    When:
        It is coerced.
    Then:
        It should produce a path source decoded to ``str``, as a bytes-yielding
        ``PathLike`` already does. Admitting one and refusing the other
        narrowed the parameter at the moment this module exists to widen it.
    """
    # Arrange
    path = write(tmp_path / "records.bed")

    # Act
    source = Source.coerce(str(path).encode())

    # Assert
    assert source.path == str(path)
    assert source.open().read() == CONTENT.encode()


def test_coerce_should_refuse_an_empty_path():
    """Test the path that is truthy as a source and unusable as a file.

    Given:
        An empty string.
    When:
        It is coerced.
    Then:
        It should raise ``ValueError``. A ``Source`` has no ``__bool__``, so an
        empty path is truthy and a consumer's empty-source guard does not fire;
        left alone it surfaces as ``FileNotFoundError`` at first open instead.
    """
    # Act & assert
    with pytest.raises(ValueError, match="non-empty path"):
        Source.coerce("")


def test___init___should_refuse_a_positional_argument(tmp_path):
    """Test the slot a factory used to land in.

    Given:
        A path passed positionally to the constructor.
    When:
        A source is constructed.
    Then:
        It should raise ``TypeError``. The constructor is public, and its two
        parameters are a path and a factory in that order -- so a caller
        passing a factory positionally bound it to ``path``, and the mistake
        surfaced as a ``__str__`` that raises, from inside the error-reporting
        path that exists to name the file.
    """
    # Act & assert
    with pytest.raises(TypeError, match="positional"):
        Source(str(tmp_path / "records.bed"))


def test___init___should_normalize_a_pathlike(tmp_path):
    """Test that the public constructor holds the invariant ``coerce`` does.

    Given:
        A ``PathLike`` passed to the constructor rather than to ``coerce``.
    When:
        Its path is read and it is rendered.
    Then:
        It should have been decoded to ``str``, so the field matches its own
        annotation and ``__repr__`` cannot raise. Normalizing only in
        ``coerce`` left the second door open.
    """
    # Arrange
    path = write(tmp_path / "records.bed")

    # Act
    source = Source(path=pathlib.Path(path))

    # Assert
    assert source.path == str(path)
    assert repr(source) == f"<Source {path}>"


def test___init___should_refuse_a_factory_that_is_not_callable(tmp_path):
    """Test the refusal ``coerce`` makes, made by the constructor too.

    Given:
        An already-open file passed as the factory keyword.
    When:
        A source is constructed.
    Then:
        It should raise ``TypeError`` naming the callable requirement. This is
        the shape ``coerce`` refuses with a message naming the fix; reaching it
        through the constructor used to defer the failure to a bare "not
        callable" at the first read.
    """
    # Arrange
    path = write(tmp_path / "records.bed")

    # Act & assert
    with open(path, "rb") as handle:
        with pytest.raises(TypeError, match="must be callable"):
            Source(factory=handle)


def test___init___should_refuse_an_empty_path():
    """Test the one path invariant the constructor used to leave to ``coerce``.

    Given:
        An empty path passed to the constructor rather than to ``coerce``.
    When:
        A source is constructed.
    Then:
        It should raise ``ValueError``. A ``Source`` has no ``__bool__``, so an
        empty one is truthy and a consumer's empty-source guard does not fire;
        enforcing this only in ``coerce`` left the public constructor admitting
        a source whose first open raises ``FileNotFoundError`` instead.
    """
    # Act & assert
    with pytest.raises(ValueError, match="non-empty path"):
        Source(path="")
