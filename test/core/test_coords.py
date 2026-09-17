"""Tests for the TileCanvas guards in clodius.core.coords.

``TileCanvas`` is the invertible map between a tile position and the genomic
intervals it covers. It is re-exported from ``clodius.core``, so anything
downstream can construct one directly rather than through
``TilesetInfo.canvas``, and the arguments it takes have no valid degenerate
case: a binsize or tile size of zero divides by zero one property later, and a
position off the end of the canvas inverts into an interval that looks like
any other.

The rest of the module -- ``Chromsizes``, ``GenomicRange``, ``natsorted`` --
is covered by ``coordinates_test.py``.
"""

import pytest

from clodius.core.coords import Chromsizes, GenomicRange, TileCanvas
from clodius.core.errors import TileOutOfBounds

CHROMSIZES = Chromsizes.from_pairs([("c1", 100), ("c2", 200)])

#: A ten-tile canvas: 400 bp of extent, 10 bp per bin, 4 bins per tile. The
#: genome is 300 bp, so the last two tiles reach past its end and stay in
#: range -- that padding is what fills the trailing NaN bins.
CANVAS = dict(z=0, binsize=10, tile_size=4, max_width=400)


class TestTileCanvas:
    """Construction and inversion at the edges of the lattice."""

    def test___init___should_construct_from_a_valid_lattice(self):
        """Test the ordinary case, so the guards cannot pass by rejecting all.

        Given:
            A positive binsize and tile size and a non-negative extent.
        When:
            The canvas is constructed.
        Then:
            It should report the tile count that extent implies.
        """
        # Act
        canvas = TileCanvas(**CANVAS, chromsizes=CHROMSIZES)

        # Assert
        assert canvas.n_tiles == 10

    @pytest.mark.parametrize(
        "field,value",
        [
            ("binsize", 0),
            ("binsize", -1),
            ("tile_size", 0),
            ("tile_size", -4),
            ("max_width", -5),
        ],
    )
    def test___init___should_raise_when_the_lattice_is_degenerate(
        self, field, value
    ):
        """Test the construction-time guards.

        Given:
            A canvas argument outside its valid range.
        When:
            The canvas is constructed.
        Then:
            It should raise ``ValueError`` at construction rather than
            constructing silently and failing later -- a zero binsize surfaces
            as ``ZeroDivisionError`` from ``n_bins``, one property away from
            the thing that was actually wrong.
        """
        # Arrange
        fields = {**CANVAS, field: value}

        # Act & assert
        with pytest.raises(ValueError, match=field):
            TileCanvas(**fields)

    def test_invert_should_return_the_intervals_the_tile_covers(self):
        """Test the ordinary inversion.

        Given:
            A tile inside the canvas and inside the genome.
        When:
            It is inverted.
        Then:
            It should yield the genomic intervals it covers.
        """
        # Arrange
        canvas = TileCanvas(**CANVAS, chromsizes=CHROMSIZES)

        # Act
        ranges = list(canvas.invert(0))

        # Assert
        assert [(r.name, r.start, r.end) for r in ranges] == [("c1", 0, 40)]

    def test_invert_should_return_an_out_of_bounds_tail_past_the_genome(self):
        """Test that padding past the genome is in range, not an error.

        Given:
            The last tile of a canvas whose extent exceeds the genome length,
            which is every quadtree canvas.
        When:
            It is inverted.
        Then:
            It should yield an interval flagged out of bounds rather than
            raising. That interval is what fills a low-zoom tile's trailing
            NaN bins, so rejecting it would empty the tail of every tile.
        """
        # Arrange
        canvas = TileCanvas(**CANVAS, chromsizes=CHROMSIZES)

        # Act
        ranges = list(canvas.invert(canvas.n_tiles - 1))

        # Assert
        assert ranges and ranges[-1].is_out_of_bounds

    @pytest.mark.parametrize("x", [10, 9999, -1])
    def test_invert_should_raise_when_the_position_is_off_the_canvas(self, x):
        """Test the bounds guard, on both sides of the lattice.

        Given:
            A tile position outside the canvas at this zoom.
        When:
            It is inverted.
        Then:
            It should raise ``TileOutOfBounds``. Without the guard the span
            arithmetic still produces a range, so the request does not fail --
            it succeeds with a plausible-looking wrong answer that reaches the
            client as a served tile.
        """
        # Arrange
        canvas = TileCanvas(**CANVAS, chromsizes=CHROMSIZES)

        # Act & assert
        with pytest.raises(TileOutOfBounds, match=f"position {x}"):
            list(canvas.invert(x))


class TestTileCanvasEdges:
    """The lattice at its boundaries: an empty extent and a partial last tile.

    Both are shapes ``TilesetInfo.canvas`` really produces -- an empty
    assembly, and any genome whose bin count is not a whole multiple of the
    tile size -- and both sit one character away from the guards above.
    """

    def test___init___should_construct_an_empty_canvas(self):
        """Test the one non-positive extent the guard admits.

        Given:
            A canvas of zero extent, which is what an empty coordinate system
            derives.
        When:
            It is constructed.
        Then:
            It should construct and hold no tiles. ``max_width`` is guarded
            against negatives, not against zero, and tightening that guard by
            one character would make an empty assembly unrepresentable rather
            than empty.
        """
        # Act
        canvas = TileCanvas(z=0, binsize=10, tile_size=4, max_width=0)

        # Assert
        assert canvas.n_tiles == 0

    def test_transform_should_yield_positions_invert_accepts(self):
        """Test that the two directions of the map agree at the last tile.

        Given:
            A lattice whose bin count is not a whole multiple of the tile size,
            so the last tile is partial, and a range at the end of the last
            chromosome.
        When:
            The range is transformed to tile positions and each is inverted.
        Then:
            Every position should be one ``invert`` accepts, and together they
            should cover the range. Rounding the tile count down instead of up
            drops the partial tile, and then the position ``transform`` reports
            for that range is one ``invert`` rejects -- the two halves of the
            map disagreeing about how many tiles exist.
        """
        # Arrange
        chromsizes = Chromsizes.from_pairs([("c1", 100), ("c2", 250)])
        canvas = TileCanvas(
            z=0, binsize=10, tile_size=4, max_width=350, chromsizes=chromsizes
        )
        query = GenomicRange(cid=1, name="c2", start=240, end=250)

        # Act
        positions = list(canvas.transform(query))
        covered = [gr for x in positions for gr in canvas.invert(x)]

        # Assert
        assert positions
        assert any(
            gr.name == "c2" and gr.start <= 240 and gr.end >= 250
            for gr in covered
        )

    def test_invert_should_raise_value_error_before_checking_bounds(self):
        """Test which of the two guards in ``invert`` runs first.

        Given:
            A canvas with no coordinate system, and a position off the end of
            it, so both guards apply.
        When:
            The position is inverted.
        Then:
            It should raise ``ValueError``, not ``TileOutOfBounds``. A tileset
            with no coordinate system is a configuration defect and must fail
            the request; ``TileOutOfBounds`` is a ``TileError``, which
            ``tiles()`` catches and returns as that tile's payload -- so
            checking bounds first would launder a broken tileset into fifteen
            cheerful per-tile errors.
        """
        # Arrange
        canvas = TileCanvas(z=0, binsize=10, tile_size=4, max_width=400)

        # Act & assert
        with pytest.raises(ValueError, match="chromsizes"):
            list(canvas.invert(9999))
