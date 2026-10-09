"""Tests for clodius.tiles_v2._backed.

The open-and-release protocol itself, rather than one backend's use of it.
Every other module pins it incidentally -- `test_bbi.py`, `test_cooler.py` and
`test_multivec.py` each exercise it through a real reader -- which leaves the
arms that only fire when *cleanup* fails untested, because no real reader
fails that way on demand. Those arms are all `except Exception: pass`, so when
one is wrong it is wrong silently, and the error it swallows is the one the
caller needed.

A double rather than a real backend, for the same reason: the behaviours here
are "what happens when the reader's own `close` raises", and neither `h5py`
nor `pybigtools` can be asked to do that.
"""

import pytest

from clodius.tiles_v2._backed import FileBacked


class Reader:
    """A reader double whose ``close`` can be made to fail on demand."""

    def __init__(self, target, *, close_raises=False):
        self.target = target
        self.close_raises = close_raises
        self.closed = False

    def close(self):
        if self.close_raises:
            raise OSError("reader close failed")
        self.closed = True


class Backed(FileBacked[Reader]):
    """A minimal concrete `FileBacked`, with each failure arm switchable."""

    ndim = 1
    datatype = "vector"

    def __init__(
        self,
        source,
        *,
        close_raises=False,
        validate_raises=False,
        header_raises=False,
        needs_handle=None,
    ):
        self.close_raises = close_raises
        self._validate_raises = validate_raises
        self._needs_handle_override = needs_handle
        super().__init__(source)
        with self._configuring():
            # Open first, so a header failure below has something to release.
            self._opened()
            if header_raises:
                raise ValueError("header read failed")

    def _reader_open(self, target):
        return Reader(target, close_raises=self.close_raises)

    def _validate(self, reader):
        if self._validate_raises:
            raise ValueError("not this tileset's format")

    def _needs_handle(self):
        if self._needs_handle_override is None:
            return super()._needs_handle()
        return self._needs_handle_override


@pytest.fixture
def source_path(tmp_path):
    """A real file, so the path branch has something to open."""
    path = tmp_path / "backing.bin"
    path.write_bytes(b"payload")
    return str(path)


class TestFileBacked:
    """The open-and-release protocol `FileBacked` publishes."""

    def test___init___should_raise_the_construction_error_when_cleanup_also_fails(
        self, source_path
    ):
        """Test that a failing release cannot replace the error that caused it.

        Given:
            A tileset whose header read fails, over a reader whose ``close`` also
            fails.
        When:
            It is constructed.
        Then:
            It should raise the header error, not the teardown one. A server
            renders the two differently -- a refusal against a 500 -- so letting
            the teardown error escape changes what the operator is told, and
            demotes the real cause to ``__context__`` where nothing reads it.
        """
        # Act & assert
        with pytest.raises(ValueError, match="header read failed"):
            Backed(source_path, header_raises=True, close_raises=True)


    def test___init___should_succeed_when_only_the_release_fails(
        self, source_path
    ):
        """Test that the registration-is-free release cannot fail construction.

        Given:
            A tileset that constructs cleanly, over a reader whose ``close``
            fails.
        When:
            It is constructed.
        Then:
            It should succeed. Handing the descriptor back at the end of
            construction is an optimization rather than a correctness
            requirement, so a source that cannot be closed cleanly must still
            produce a usable tileset rather than failing at registration.
        """
        # Act
        tileset = Backed(source_path, close_raises=True)

        # Assert
        assert isinstance(tileset, Backed)


    def test___init___should_raise_the_refusal_when_the_reader_close_also_fails(
        self, source_path
    ):
        """Test the same precedence on the validation path.

        Given:
            A tileset whose `_validate` refuses the file, over a reader whose
            ``close`` also fails.
        When:
            It is constructed.
        Then:
            It should raise the refusal. `_validate` has already produced the
            message the caller needs; the reader's teardown failure is noise
            against it, and nothing has been published for a retry to reach.
        """
        # Act & assert
        with pytest.raises(ValueError, match="not this tileset's format"):
            Backed(source_path, validate_raises=True, close_raises=True)


    def test___init___should_refuse_an_override_that_narrows_needs_handle(
        self, source_path
    ):
        """Test the one contract a subclass is told it must not narrow.

        Given:
            A factory-backed source, and a subclass whose `_needs_handle` returns
            False for it.
        When:
            The tileset is constructed.
        Then:
            It should raise `TypeError` naming the override, because a pathless
            source has only a handle to offer and the open has no path to fall
            back on. Raised at this seam rather than left to the reader, where it
            arrives as a complaint about a path the caller never passed.
        """
        # Arrange
        def factory():
            return open(source_path, "rb")

        # Act & assert
        with pytest.raises(TypeError, match="_needs_handle"):
            Backed(factory, needs_handle=False)


    def test_reading_should_reopen_after_close(self, source_path):
        """Test that the base seam reopens rather than yielding a closed reader.

        Given:
            A tileset that has been closed.
        When:
            Its reader is taken again.
        Then:
            It should be a fresh reader. `close` is idempotent and construction
            releases what it opened, so every tileset is cold at its first serve
            and the seam has to be able to open as well as hand back.
        """
        # Arrange
        tileset = Backed(source_path)
        with tileset.reading() as first:
            pass
        tileset.close()

        # Act
        with tileset.reading() as second:
            pass

        # Assert
        assert first is not second
        assert first.closed
        tileset.close()

    def test_close_should_keep_a_reader_it_could_not_close(self, source_path):
        """Test that a failed release leaves the reader reclaimable.

        Given:
            An open tileset whose reader's ``close`` is made to fail.
        When:
            It is closed, reopened, and closed again once the readers can be
            closed.
        Then:
            The first close should raise, the reader should not be closed and
            not be lost, the tileset should hand back a *fresh* reader rather
            than the one it could not release, and the later close should
            reclaim both.

            Each clause is a way this went wrong. The old `close` detached in
            a `finally`, so the reader that failed to close became
            unreachable and its descriptor was left to the garbage collector
            -- which a CHANGELOG entry in this same change claims to have
            fixed. Putting it back in ``_reader`` instead would have
            republished it: `_opened` returns a non-`None` reader without
            taking the lock, so the next read would be served through a
            reader whose handle the same `close` had already closed.
        """
        # Arrange
        tileset = Backed(source_path)
        tileset.close_raises = True
        with tileset.reading() as reader:
            pass

        # Act
        with pytest.raises(OSError, match="reader close failed"):
            tileset.close()

        # Assert
        assert not reader.closed
        with tileset.reading() as reopened:
            pass
        assert reopened is not reader
        reader.close_raises = False
        reopened.close_raises = False
        tileset.close()
        assert reader.closed
        assert reopened.closed
