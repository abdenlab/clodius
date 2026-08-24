"""Tests for clodius.core.errors.

Every class in that module has a docstring-only body, so most of what is
asserted here is inherited ``BaseException`` behavior. That is deliberate: the
module docstring commits to rendering each error as ``{"error": str(exc)}`` at
the server boundary, and :class:`~clodius.core.payloads.ErrorTile` types that
field as ``str`` -- which makes the rendered text part of the wire contract
rather than an implementation detail. These are a regression net for a
hierarchy that is currently free of logic, not coverage of logic.
"""

import inspect

import pytest
from pydantic import TypeAdapter
from hypothesis import given, settings
from hypothesis import strategies as st

from clodius.core import errors
from clodius.core.payloads import ErrorTile
from clodius.core.errors import (
    MalformedTileId,
    TileError,
    TileOutOfBounds,
    TilesetUnavailable,
    UnsupportedModifier,
    UnsupportedOption,
)

# Listed explicitly rather than introspected from the module, so the identity
# and absence tests below assert something the parametrization did not assume.
SUBCLASSES = (
    TilesetUnavailable,
    MalformedTileId,
    UnsupportedModifier,
    UnsupportedOption,
    TileOutOfBounds,
)
ALL_ERRORS = (TileError, *SUBCLASSES)
ERROR_IDS = [cls.__name__ for cls in ALL_ERRORS]
SUBCLASS_IDS = [cls.__name__ for cls in SUBCLASSES]


@pytest.mark.parametrize("name", ["TileTooWide", "TileTooLarge"])
def test_errors_should_not_declare_the_deferred_classes(name):
    """Test that the unraised error classes stay deferred.

    Given:
        The errors module, from which two never-raised classes were dropped.
    When:
        Each name is looked up on it.
    Then:
        It should be absent, so a class returns only when something raises it.
    """
    # Act & assert
    assert not hasattr(errors, name)


class TestTileError:
    """The message payload, which the boundary renders onto the wire."""

    @pytest.mark.parametrize("cls", ALL_ERRORS, ids=ERROR_IDS)
    def test___str___should_render_the_message_verbatim(self, cls):
        """Test that the raiser's text survives to the wire.

        Given:
            Each error class and a message.
        When:
            The class is instantiated with it and rendered as a string.
        Then:
            It should return the message unchanged, since the boundary puts
            exactly this text in the error payload.
        """
        # Arrange
        message = "zoom 3 exceeds ladder of 3 resolutions"

        # Act & assert
        assert str(cls(message)) == message

    @pytest.mark.parametrize("cls", ALL_ERRORS, ids=ERROR_IDS)
    def test___init___should_accept_no_arguments(self, cls):
        """Test the bare raise.

        Given:
            Each error class.
        When:
            It is instantiated with no arguments.
        Then:
            It should carry an empty argument tuple and render as the empty
            string, so a bare raise yields an empty error rather than failing
            during rendering.
        """
        # Act
        exc = cls()

        # Assert
        assert exc.args == ()
        assert str(exc) == ""

    @pytest.mark.parametrize("cls", ALL_ERRORS, ids=ERROR_IDS)
    def test___init___should_retain_several_arguments(self, cls):
        """Test the multi-argument form.

        Given:
            Each error class and two positional arguments.
        When:
            It is instantiated with both.
        Then:
            It should expose them in order, which is the shape the boundary
            would render if a caller passes more than a message.
        """
        # Act
        exc = cls("context", 42)

        # Assert
        assert exc.args == ("context", 42)

    @given(message=st.text())
    @settings(max_examples=50)
    def test___str___should_render_any_message_unchanged(self, message):
        """Test message rendering over arbitrary text.

        Given:
            Any text, including empty strings, newlines, unicode, and the
            braces and quotes a JSON-rendering boundary must not mangle.
        When:
            A tile error is instantiated with it and rendered.
        Then:
            It should return the text exactly.
        """
        # Act & assert
        assert str(TileError(message)) == message

    def test_tile_error_should_subclass_exception(self):
        """Test that the hierarchy roots in the standard exception tree.

        Given:
            The base tile error.
        When:
            Its ancestry is checked.
        Then:
            It should subclass Exception, so a boundary catch-all still sees every
            tile error.
        """
        # Act & assert
        assert issubclass(TileError, Exception)

    @pytest.mark.parametrize("cls", SUBCLASSES, ids=SUBCLASS_IDS)
    def test_tile_error_should_be_the_base_of_every_declared_error(self, cls):
        """Test that every declared error descends from the base.

        Given:
            Each error class the module declares.
        When:
            Its ancestry is checked.
        Then:
            It should subclass the base, so the boundary needs exactly one except
            clause rather than one per error.
        """
        # Act & assert
        assert issubclass(cls, TileError)

    @pytest.mark.parametrize("cls", SUBCLASSES, ids=SUBCLASS_IDS)
    def test_tile_error_should_catch_every_subclass(self, cls):
        """Test that the base clause really catches each subclass.

        Given:
            An instance of each declared error.
        When:
            It is raised inside a try whose only handler names the base.
        Then:
            It should be caught, and the caught object should be the instance
            raised.
        """
        # Arrange
        raised = cls("boom")

        # Act
        try:
            raise raised
        except TileError as exc:
            caught = exc

        # Assert
        assert caught is raised

    def test_tile_error_should_declare_six_distinct_classes(self):
        """Test that the module binds exactly these six error classes.

        Given:
            Every exception class the errors module binds.
        When:
            They are compared with the six this module tests.
        Then:
            They should be the same set. That catches an alias -- which would
            still satisfy the ancestry and catchability checks above -- and a
            seventh class added to the module without a test, which counting
            ``ALL_ERRORS`` cannot: ``ALL_ERRORS`` is written here, so a class
            added over there leaves its length unchanged.
        """
        # Arrange
        bound = {
            obj
            for name in dir(errors)
            if not name.startswith("_")
            and inspect.isclass(obj := getattr(errors, name))
            and issubclass(obj, Exception)
        }

        # Act & assert
        assert bound == set(ALL_ERRORS)


class TestTilesetUnavailable:
    """The whole-info error, which replaces a tileset info rather than a tile."""

    def test___str___should_render_into_a_whole_info_error_payload(self):
        """Test the documented whole-info translation.

        Given:
            A tileset-unavailable error raised for an unservable tileset.
        When:
            It is caught by a handler naming the base and rendered the way the
            module docstring specifies.
        Then:
            It should produce a payload that validates as an ``ErrorTile`` and
            carries the raiser's message. Asserting only that the dict equals
            what the two lines above built is a tautology; validating it
            against the declared shape is the wire contract this module exists
            to pin.
        """
        # Arrange
        message = "file exceeds max_unindexed_filesize"

        # Act
        try:
            raise TilesetUnavailable(message)
        except TileError as exc:
            payload = {"error": str(exc)}

        # Assert
        assert TypeAdapter(ErrorTile).validate_python(payload) == {
            "error": message
        }
