"""Tests for clodius.tiles_v2.cooler.

A multi-resolution cooler is the one tileset with an explicit resolution
ladder, and the only one that has to tell the client how the matrix it is
serving is stored. A cooler written symmetric-upper holds half the matrix and
the client mirrors it; one written square holds the whole thing and mirroring
it would double every off-diagonal count. That is what ``mirror_tiles`` says,
and it is carried as an extra field on a model that is otherwise frozen --
so it has to be passed at construction, and a test that only checks the
square case passes just as well when the condition is dropped and every
cooler claims to be square.

The fixtures are built here rather than read from ``test/data/``: the
committed multires cooler is a legacy implicit-ladder file with no
``resolutions`` group, which this tileset refuses outright.
"""

import pytest

from clodius.tiles_v2.cooler import CoolerTileset

from ..core.mcool_fixture import build_mcool


@pytest.fixture(scope="module")
def symmetric_mcool(tmp_path_factory):
    """A multires cooler holding the upper triangle, the ordinary case."""
    path = str(tmp_path_factory.mktemp("cooler") / "symmetric.mcool")
    return build_mcool(path)


@pytest.fixture(scope="module")
def square_mcool(tmp_path_factory):
    """A multires cooler holding the whole matrix."""
    path = str(tmp_path_factory.mktemp("cooler") / "square.mcool")
    return build_mcool(path, symmetric_upper=False)


def test_info_should_tell_the_client_not_to_mirror_a_square_matrix(
    square_mcool,
):
    """Test the flag a whole-matrix cooler has to carry.

    Given:
        A multires cooler written with the whole matrix stored.
    When:
        Its info is requested.
    Then:
        It should carry ``mirror_tiles`` set to false. The client mirrors by
        default, which for a square matrix adds every off-diagonal count to
        itself -- a plausible-looking heatmap at twice the contact frequency.
    """
    # Arrange
    tileset = CoolerTileset(square_mcool)

    # Act
    served = tileset.info().to_dict()

    # Assert
    assert served["mirror_tiles"] == "false"


def test_info_should_omit_the_flag_for_a_half_matrix(symmetric_mcool):
    """Test that the flag is conditional rather than always emitted.

    Given:
        A multires cooler written symmetric-upper, the ordinary case.
    When:
        Its info is requested.
    Then:
        ``mirror_tiles`` should be absent, leaving the client to mirror the
        half-matrix it was given. Without this the previous test passes
        unchanged when the storage-mode condition is deleted and every cooler
        is declared square.
    """
    # Arrange
    tileset = CoolerTileset(symmetric_mcool)

    # Act
    served = tileset.info().to_dict()

    # Assert
    assert "mirror_tiles" not in served


