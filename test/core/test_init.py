"""Tests for the ``clodius.core`` package facade.

The facade is load-bearing public API: ``clodius.tiles_v2`` imports from
``clodius.core``, never from a submodule. Nothing enforced that until now.
``pyproject.toml`` switches pyflakes' unused-import check off for
``__init__.py`` -- the one rule that would otherwise notice a re-export gone
stale -- so a dropped entry, a listed name that no longer resolves, or a name
imported but never listed would all pass CI silently.

These are structural invariants over ``__all__`` rather than per-symbol tests,
so a name added to a submodule and forgotten here fails one assertion instead
of going unnoticed.
"""

import ast
import importlib
import pathlib
import subprocess
import sys
import types

import pytest

import clodius.core as core
from clodius.core import errors

CORE_DIR = pathlib.Path(core.__file__).parent
TILES_V2_DIR = CORE_DIR.parent / "tiles_v2"

SUBMODULES = sorted(
    p.stem for p in CORE_DIR.glob("*.py") if p.stem != "__init__"
)
TILESET_MODULES = sorted(
    p for p in TILES_V2_DIR.glob("*.py") if p.stem != "__init__"
)
TILESET_IDS = [p.stem for p in TILESET_MODULES]


def import_froms(path):
    """Every ``from X import a, b`` in ``path`` as ``(module, [names])``."""
    tree = ast.parse(pathlib.Path(path).read_text())
    return [
        (node.module, [alias.name for alias in node.names])
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ]


#: Which submodule each exported name is defined in, derived from the facade's
#: own import blocks rather than hand-listed -- a hand-list would be a second
#: thing to keep in sync.
EXPORT_SOURCES = {
    name: module.rsplit(".", 1)[-1]
    for module, names in import_froms(core.__file__)
    if module.startswith("clodius.core.")
    for name in names
}


@pytest.mark.parametrize("name", core.__all__)
def test_all_should_name_an_importable_symbol(name):
    """Test that every exported name resolves.

    Given:
        Each name the facade exports.
    When:
        It is looked up on the package.
    Then:
        It should resolve, so no entry names a symbol that was renamed or
        removed from its submodule.
    """
    # Act & assert
    assert hasattr(core, name)


@pytest.mark.parametrize("name", core.__all__)
def test_all_should_bind_a_real_object(name):
    """Test that no export is a placeholder.

    Given:
        Each name the facade exports.
    When:
        The bound object is inspected.
    Then:
        It should be the very object its defining submodule binds, so a
        shadowed import cannot masquerade as an export. Comparing against
        ``None`` would pass for any object at all, including a placeholder
        bound over the real one.
    """
    # Arrange
    source = importlib.import_module(f"clodius.core.{EXPORT_SOURCES[name]}")

    # Act & assert
    assert getattr(core, name) is getattr(source, name)


def test_all_should_contain_no_duplicates():
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


def test_all_should_be_ordered_by_isort_convention():
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


def test_all_should_be_exactly_the_star_import_surface():
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


@pytest.mark.parametrize("name", sorted(EXPORT_SOURCES))
def test_all_should_re_export_rather_than_redefine(name):
    """Test that the facade rebinds its submodules' objects.

    Given:
        Each exported name paired with the submodule that defines it.
    When:
        The object on the package is compared by identity with the attribute
        on that submodule.
    Then:
        It should be the same object, so catching an error imported from the
        facade catches what the submodule raises.
    """
    # Arrange
    submodule = getattr(core, EXPORT_SOURCES[name], None)
    if submodule is None:

        submodule = importlib.import_module(
            f"clodius.core.{EXPORT_SOURCES[name]}"
        )

    # Act & assert
    assert getattr(core, name) is getattr(submodule, name)


def test_all_should_export_every_declared_error():
    """Test that the facade is complete for the error hierarchy.

    Given:
        The error classes the errors submodule declares.
    When:
        Each name is checked against the export list.
    Then:
        It should be present, so adding an error without exporting it is a
        visible omission -- which is what the facade's docstring promises.
    """
    # Arrange
    declared = {
        name
        for name, obj in vars(errors).items()
        if isinstance(obj, type)
        and issubclass(obj, Exception)
        and obj.__module__ == errors.__name__
    }

    # Act & assert
    assert declared <= set(core.__all__)


@pytest.mark.parametrize("path", TILESET_MODULES, ids=TILESET_IDS)
def test_all_should_export_everything_the_tilesets_import(path):
    """Test that the facade is complete against its only consumer.

    Given:
        Each module under ``clodius.tiles_v2``, parsed for what it imports
        from the facade.
    When:
        Those names are checked against the export list.
    Then:
        Every one should be present, so the facade stays complete as the
        tilesets grow new imports.
    """
    # Arrange
    imported = {
        name
        for module, names in import_froms(path)
        if module == "clodius.core"
        for name in names
    }

    # Act & assert
    assert imported <= set(core.__all__)


@pytest.mark.parametrize("path", TILESET_MODULES, ids=TILESET_IDS)
def test_tiles_v2_should_reach_core_only_through_the_facade(path):
    """Test that no tileset reaches around the facade.

    Given:
        Each module under ``clodius.tiles_v2``.
    When:
        The module string of every from-import is examined.
    Then:
        None should name a ``clodius.core`` submodule, since a tileset
        importing one directly puts the facade back out of use.
    """
    # Act
    reaching = [
        module
        for module, _ in import_froms(path)
        if module.startswith("clodius.core.")
    ]

    # Assert
    assert reaching == []


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_should_not_import_the_facade(name):
    """Test that no submodule imports its own package.

    Given:
        Each module under ``clodius.core``.
    When:
        The module string of every from-import is examined.
    Then:
        None should be ``clodius.core`` itself. The package eagerly imports
        every submodule, so whether such an import resolves depends entirely
        on which block precedes it in ``__init__.py`` -- the same statement
        succeeds in the last submodule and raises in the first.
    """
    # Act
    reaching = [
        module
        for module, _ in import_froms(CORE_DIR / f"{name}.py")
        if module == "clodius.core"
    ]

    # Assert
    assert reaching == []


@pytest.mark.integration
@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_should_import_cleanly_on_its_own(name):
    """Test that no submodule has a circular import.

    Given:
        Each module under ``clodius.core``.
    When:
        It is imported as the very first clodius import in a fresh
        interpreter.
    Then:
        It should exit zero, catching a real cycle that the syntactic check
        above would miss.
    """
    # Act
    result = subprocess.run(
        [sys.executable, "-c", f"import clodius.core.{name}"],
        capture_output=True,
        text=True,
    )

    # Assert
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", ["TileTooWide", "TileTooLarge"])
def test_all_should_not_resurrect_the_deferred_errors(name):
    """Test that the dropped error classes stay out of the facade.

    Given:
        The facade, after two never-raised error classes were dropped.
    When:
        Each name is checked against the export list and the package
        namespace.
    Then:
        It should appear in neither.
    """
    # Act & assert
    assert name not in core.__all__
    assert not hasattr(core, name)
