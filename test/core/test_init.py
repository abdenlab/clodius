"""Tests for the ``clodius.core`` package facade.

The facade is the intended public surface -- though nothing imports it yet:
all twelve ``clodius.tiles_v2`` modules reach into ``clodius.core.coords``,
``.errors``, ``.tileid`` and the rest directly. Adopting it is separate work;
until then these tests keep the surface itself honest.

That matters because nothing else does. ``pyproject.toml`` switches pyflakes'
unused-import check off for ``__init__.py`` -- the one rule that would
otherwise notice a re-export gone stale -- so a dropped entry, a listed name
that no longer resolves, or a name imported but never listed all pass CI
silently. Exactly that happened: the facade listed ``Canvas``, which no
submodule binds, and never listed ``TileCanvas``, which it imports.

These are structural invariants over ``__all__`` rather than per-symbol tests,
so a name added to a submodule and forgotten here fails one assertion instead
of going unnoticed.
"""

import types

from clodius.core import errors
import clodius.core as core

#: Every error the tile layer declares, read off the module rather than
#: hand-listed -- a hand-list would be a second thing to keep in sync.
ERROR_NAMES = sorted(
    name
    for name, obj in vars(errors).items()
    if isinstance(obj, type)
    and issubclass(obj, Exception)
    and obj.__module__ == errors.__name__
)


def test___all___should_support_a_star_import():
    """Test the facade against the import form that reads ``__all__``.

    Given:
        The ``clodius.core`` package.
    When:
        Every public name is pulled in with a star import.
    Then:
        It should succeed. A listed name the package does not bind raises
        ``AttributeError`` here and nowhere else -- attribute access and
        ``dir`` both skip ``__all__`` entirely.
    """
    # Arrange
    namespace = {}

    # Act
    exec("from clodius.core import *", namespace)

    # Assert
    assert namespace["TileCanvas"] is core.TileCanvas


def test___all___should_be_exactly_the_star_import_surface():
    """Test that the export list covers every public name the facade binds.

    Given:
        The facade's public attributes, less the submodules that importing
        them binds as a side effect.
    When:
        They are compared with the export list.
    Then:
        They should be equal in both directions, so a name bound into the
        facade but omitted from the list is caught, as is the reverse.

        Reading the surface off ``dir`` rather than off a star import is what
        makes this falsifiable. ``from ... import *`` consults ``__all__``
        when it is defined, so comparing the bound names against ``__all__``
        compares the list with itself: a public name added to the facade and
        left out of the list is invisible to it.
    """
    # Arrange
    public = {name for name in dir(core) if not name.startswith("_")}
    submodules = {
        name
        for name in public
        if isinstance(getattr(core, name), types.ModuleType)
    }

    # Act
    bound = public - submodules

    # Assert
    assert bound == set(core.__all__)


def test___all___should_contain_no_duplicates():
    """Test that no name is exported twice.

    Given:
        The facade's export list.
    When:
        Its length is compared with the size of its set.
    Then:
        They should be equal, so a merge that re-adds an existing name is
        caught.
    """
    # Act & assert
    assert len(core.__all__) == len(set(core.__all__))


def test___all___should_be_ordered_by_isort_convention():
    """Test the export list's ordering.

    Given:
        The facade's export list, written in the grouped order ruff's RUF022
        enforces.
    When:
        It is partitioned into all-uppercase, then CamelCase, then lowercase
        names.
    Then:
        It should equal those three groups concatenated, each sorted. Note a
        plain sorted() does not hold and must not be asserted -- the uppercase
        constants precede the classes. Nothing else enforces this, since RUF is
        not in the project's ruff selection.
    """
    # Arrange
    names = list(core.__all__)
    upper = sorted(n for n in names if n.isupper())
    camel = sorted(n for n in names if not n.isupper() and n[0].isupper())
    lower = sorted(n for n in names if n[0].islower())

    # Act & assert
    assert names == upper + camel + lower


def test___all___should_export_every_declared_error():
    """Test that the facade keeps up with the error hierarchy.

    Given:
        Every exception class ``clodius.core.errors`` declares.
    When:
        Each is looked for in the facade's export list.
    Then:
        It should be there. A server catches these by name to decide whether a
        failure belongs to one tile or to the whole request, so an error added
        without an export is one a caller reaching through the facade cannot
        name -- and no other test here notices, since they all reason from
        ``__all__`` outwards rather than from the submodules in.
    """
    # Act
    missing = [name for name in ERROR_NAMES if name not in core.__all__]

    # Assert
    assert missing == []
