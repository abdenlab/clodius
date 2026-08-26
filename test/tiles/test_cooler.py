"""Regression tests for clodius.tiles.cooler's ladder guard.

``generate_tiles`` batches tile ids by zoom level and refuses any zoom the
ladder does not hold. On the ``resolutions`` branch that guard read
``zoom_level > len(sorted_resolutions)``, so the boundary case -- ``z`` equal to
the number of resolutions -- fell through and indexed the list out of range.
This branch changed it to ``>=``.

``test/tiles/test_conformance.py`` already asserts the fix, against a cooler in
``data/`` that is an un-smudged git-LFS pointer on a plain checkout. It has
therefore skipped for the whole life of the fix. Everything here builds its own
mcool.

The skip is also the behavior ``clodius/tiles_v2/cooler.py`` was aligned to on
this branch: an out-of-range id is omitted from the result while its siblings
in the same request still return. These tests establish the contract that one
copies.
"""

import pytest

from clodius.tiles import cooler as mod

from ..harness import builders
from ..harness.wire import square

#: The bin table a cooler tile is cut from, per side.
BINS_PER_TILE = 256


@pytest.fixture(scope="module")
def mcool(tmp_path_factory):
    """A three-resolution mcool, built once for the module.

    Module-scoped rather than session-scoped: this uses realistic kilobase
    binsizes, where the tiles_v2 fixture deliberately uses 1/2/4 to get more
    than one tile per zoom. Sharing one file across both would mean one of them
    testing the wrong geometry.
    """
    path = tmp_path_factory.mktemp("cooler") / "tiny.mcool"
    return str(builders.build_mcool(path, resolutions=(1000, 2000, 4000)))


@pytest.fixture
def past_the_ladder(mcool):
    """The first zoom level the ladder does not hold."""
    return len(mod.tileset_info(mcool)["resolutions"])


class TestTilesetInfo:
    """What the module advertises for a multi-resolution cooler."""

    def test_tileset_info_should_enumerate_the_stored_resolutions(self, mcool):
        """Test the explicit ladder.

        Given:
            An mcool with three resolution groups.
        When:
            The info is read.
        Then:
            It should enumerate them and declare no ``max_zoom``. Which of the
            two fields is present is what ``generate_tiles`` branches on, so a
            tileset carrying both would take the wrong guard.
        """
        # Act
        info = mod.tileset_info(mcool)

        # Assert
        assert sorted(info["resolutions"]) == [1000, 2000, 4000]
        assert "max_zoom" not in info


class TestLadderGuard:
    """The boundary this branch fixed."""

    def test_tiles_should_skip_the_tile_when_the_zoom_equals_the_ladder_length(
        self, mcool, past_the_ladder
    ):
        """Test the exact boundary case the ``>`` guard let through.

        Given:
            A zoom level equal to the number of stored resolutions, which is
            one past the deepest valid index.
        When:
            The tile is requested.
        Then:
            It should be skipped, leaving an empty result. Under ``>`` the
            equality case fell through to ``sorted_resolutions[zoom_level]``
            and raised IndexError out of the request handler.
        """
        # Act
        result = mod.tiles(mcool, [f"a.{past_the_ladder}.0.0"])

        # Assert
        assert result == []

    def test_tiles_should_skip_a_zoom_far_past_the_ladder(
        self, mcool, past_the_ladder
    ):
        """Test the case the old guard already handled.

        Given:
            A zoom several levels past the ladder.
        When:
            The tile is requested.
        Then:
            It should be skipped. ``>`` and ``>=`` agree here, which is why the
            boundary case survived as long as it did.
        """
        # Act
        result = mod.tiles(mcool, [f"a.{past_the_ladder + 3}.0.0"])

        # Assert
        assert result == []

    def test_tiles_should_serve_the_deepest_valid_zoom(
        self, mcool, past_the_ladder
    ):
        """Test that the guard did not move the boundary the other way.

        Given:
            The last zoom the ladder does hold.
        When:
            The tile is requested.
        Then:
            It should be served in full. Tightening a comparison is exactly the
            kind of fix that trades an IndexError for a silently missing zoom
            level.
        """
        # Act
        ((_, payload),) = mod.tiles(mcool, [f"a.{past_the_ladder - 1}.0.0"])

        # Assert
        assert square(payload).shape == (BINS_PER_TILE, BINS_PER_TILE)

    def test_tiles_should_keep_the_siblings_of_a_skipped_tile(
        self, mcool, past_the_ladder
    ):
        """Test batch integrity across the guard.

        Given:
            A batch pairing a valid tile id with one past the ladder.
        When:
            It is generated.
        Then:
            The valid tile should still be returned. The guard ``continue``s
            past a zoom group rather than returning, which is what keeps one
            bad id from discarding the rest of the request -- and is the
            behavior ``tiles_v2`` was aligned to on this branch.
        """
        # Act
        result = mod.tiles(
            mcool,
            [f"a.{past_the_ladder - 1}.0.0", f"a.{past_the_ladder}.0.0"],
        )

        # Assert
        assert [tid for tid, _ in result] == [f"a.{past_the_ladder - 1}.0.0"]

    def test_tiles_should_skip_a_position_past_the_genome_length_in_base_pairs(
        self, mcool
    ):
        """Test the position filter as it is actually written.

        Given:
            A tile index numerically larger than the genome's length in base
            pairs -- 9999 against a 3000 bp fixture.
        When:
            It is requested.
        Then:
            It should be skipped. Note what is being compared: the filter at
            ``cooler.py:681`` tests the *tile index* against ``max_pos[0]``,
            which is the genome length in *base pairs* (set at
            ``cooler.py:519``). The two are different units, so this passes
            only because 9999 happens to exceed 3000. On any real genome the
            filter never fires -- see the xfail below.
        """
        # Act
        result = mod.tiles(mcool, ["a.0.9999.0"])

        # Assert
        assert result == []

    @pytest.mark.xfail(
        strict=True,
        reason="tile index compared against a base-pair length at "
        "cooler.py:681, so the bounds filter never fires (#TBD)",
    )
    def test_tiles_should_skip_a_position_past_the_last_tile(self, mcool):
        """Test the bound the position filter is supposed to enforce.

        Given:
            A tile position wholly past the genome at its zoom. The fixture is
            3000 bp and a tile spans 1024 bp at zoom 0, so the genome holds
            three tiles and ``x=5`` covers [5120, 6144) -- entirely past it.
        When:
            It is requested.
        Then:
            It should be skipped, as the zoom guard skips a zoom past the
            ladder. It is not: the tile is served as a full 256x256 buffer,
            because 5 is below 3000 and the filter compares those two numbers
            directly.
        """
        # Act
        result = mod.tiles(mcool, ["a.0.5.0"])

        # Assert
        assert result == []


class TestTileShape:
    """That a served tile is still the shape the client expects."""

    @pytest.mark.parametrize("z", [0, 1, 2])
    def test_tiles_should_return_a_square_of_bins_per_tile(self, mcool, z):
        """Test the payload size across the ladder.

        Given:
            Each zoom the ladder holds.
        When:
            The origin tile is generated.
        Then:
            It should carry ``BINS_PER_TILE`` squared values. Cooler ships no
            shape field, so the client infers the side from the length and a
            wrong count is misread rather than rejected.
        """
        # Act
        ((_, payload),) = mod.tiles(mcool, [f"a.{z}.0.0"])

        # Assert
        assert square(payload).shape == (BINS_PER_TILE, BINS_PER_TILE)
