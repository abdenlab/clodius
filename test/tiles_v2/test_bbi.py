"""Tests for the tileset info in clodius.tiles_v2.bbi.

One module serves bigWig and bigBed through four concrete tilesets that share
a single ``_info_for``: three of them one-dimensional, and the bigInteract 2D
tileset two-dimensional. The shared builder has to describe whichever arity
its subclass declares, because ``min_pos`` and ``max_pos`` are per-axis on the
wire -- a 2D track reads the y extent from ``max_pos[1]``.

The arity is the only thing this info gets from the subclass, and it is the
only thing under test here. ``canvas()`` derives its extent from the scalar
``max_width``, so tile serving does not read these fields at all and a wrong
arity is invisible everywhere except the served info. Nothing looked at the
served info before.

The padded extent matters as much as the count. The range is padded out to the
full quadtree extent rather than the genome length, as legacy does, so the
fixtures below total 2500 bp against an extent of 4096 -- a build padding with
the genome total would otherwise be indistinguishable from a correct one.

Fixtures are synthesized with pybigtools in milliseconds, so the module runs
on a checkout with no git-LFS payload. Note the chromsizes go to ``write``,
not to ``open``: the wrapper's ``open`` takes only a path and a mode. Two
writer constraints shape the records below -- they must be coordinate-sorted,
and a contig carrying no records is dropped from the file's chrom list, so
every contig the tileset should see needs at least one.
"""

import pybigtools
import pytest

from clodius.tiles_v2.bbi import (
    BBIAnnotationTileset,
    BBIInteraction2DTileset,
    BBIInteractionLinksTileset,
    BBISignalTileset,
)

#: Totals 2500 bp, which a four-bin tile size covers with a quadtree extent of
#: 4096 -- so the padded range is distinguishable from the genome length.
CHROMSIZES = {"c1": 1000, "c2": 1500}

TILE_SIZE = 4
QUADTREE_EXTENT = 4096
GENOME_LENGTH = sum(CHROMSIZES.values())


def interact_record(name, source, target):
    """One bed5+13 interact row, as the tab-joined rest of a bigBed record.

    The hull is the record's own start and end; the anchors live in the custom
    fields, which is what makes this schema worth a fixture of its own.
    """
    source_chrom, source_start, source_end = source
    target_chrom, target_start, target_end = target
    return "\t".join(
        [
            name,
            "0",
            "+",
            "hg38",
            "0",
            source_chrom,
            str(source_start),
            str(source_end),
            "0,0,0",
            target_chrom,
            str(target_start),
            str(target_end),
            "0,0,0",
        ]
    )


@pytest.fixture(scope="module")
def bigwig(tmp_path_factory):
    """A bigWig covering every chromosome with flat intervals."""
    path = str(tmp_path_factory.mktemp("bbi") / "signal.bw")
    pybigtools.open(path, "w").write(
        CHROMSIZES,
        iter(
            [
                ("c1", 0, 500, 1.5),
                ("c1", 500, 1000, 3.0),
                ("c2", 0, 1500, 2.0),
            ]
        ),
    )
    return path


@pytest.fixture(scope="module")
def bigbed(tmp_path_factory):
    """A bed5+13 bigBed holding two interactions, one per contig."""
    path = str(tmp_path_factory.mktemp("bbi") / "interact.bb")
    pybigtools.open(path, "w").write(
        CHROMSIZES,
        iter(
            [
                (
                    "c1",
                    10,
                    900,
                    interact_record("a", ("c1", 10, 60), ("c1", 850, 900)),
                ),
                (
                    "c2",
                    20,
                    700,
                    interact_record("b", ("c2", 20, 70), ("c2", 650, 700)),
                ),
            ]
        ),
    )
    return path


def test_info_should_return_one_position_per_axis_for_a_2d_tileset(bigbed):
    """Test the arity of the info a two-dimensional tileset serves.

    Given:
        A bigInteract served as 2D rectangles, which declares ``ndim = 2``.
    When:
        Its info is requested.
    Then:
        ``min_pos`` and ``max_pos`` should each carry two entries. The client
        reads the y extent from ``max_pos[1]``, so a one-entry list leaves a
        2D track with no second axis to lay out -- and nothing else in the
        serving path reads these fields, so the served info is the only place
        the mistake is visible.
    """
    # Arrange
    tileset = BBIInteraction2DTileset(bigbed, tile_size=TILE_SIZE)

    # Act
    served = tileset.info().to_dict()

    # Assert
    assert served["min_pos"] == [0, 0]
    assert served["max_pos"] == [QUADTREE_EXTENT, QUADTREE_EXTENT]


@pytest.mark.parametrize(
    "cls,fixture",
    [
        (BBISignalTileset, "bigwig"),
        (BBIAnnotationTileset, "bigbed"),
        (BBIInteractionLinksTileset, "bigbed"),
    ],
    ids=["signal", "annotation", "links"],
)
def test_info_should_return_a_single_position_for_a_1d_tileset(
    cls, fixture, request
):
    """Test that the per-axis padding did not widen the one-dimensional types.

    Given:
        Each of the three tilesets that declare ``ndim = 1``.
    When:
        Info is requested.
    Then:
        ``min_pos`` and ``max_pos`` should each carry exactly one entry.
        Without this the 2D test above passes just as well against a build
        that pads every tileset to two axes, which would misdescribe three
        types to fix one.
    """
    # Arrange
    tileset = cls(request.getfixturevalue(fixture), tile_size=TILE_SIZE)

    # Act
    served = tileset.info().to_dict()

    # Assert
    assert served["min_pos"] == [0]
    assert served["max_pos"] == [QUADTREE_EXTENT]


@pytest.mark.parametrize(
    "cls,fixture",
    [
        (BBISignalTileset, "bigwig"),
        (BBIAnnotationTileset, "bigbed"),
        (BBIInteractionLinksTileset, "bigbed"),
        (BBIInteraction2DTileset, "bigbed"),
    ],
    ids=["signal", "annotation", "links", "interaction2d"],
)
def test_info_should_describe_as_many_axes_as_the_tileset_declares(
    cls, fixture, request
):
    """Test the invariant the four tilesets share, rather than four constants.

    Given:
        Each concrete tileset in the module, whatever arity it declares.
    When:
        Info is requested.
    Then:
        ``min_pos`` and ``max_pos`` should both be as long as ``ndim``. The
        two tests above pin the arities the module has today; this one pins
        the rule, so a fifth tileset added later cannot quietly inherit the
        wrong shape.
    """
    # Arrange
    tileset = cls(request.getfixturevalue(fixture), tile_size=TILE_SIZE)

    # Act
    served = tileset.info().to_dict()

    # Assert
    assert len(served["min_pos"]) == cls.ndim
    assert len(served["max_pos"]) == cls.ndim


def test_info_should_pad_every_axis_to_the_quadtree_extent(bigbed):
    """Test what each axis is padded *to*, not just how many there are.

    Given:
        A 2D tileset over a genome of 2500 bp, whose quadtree extent is 4096.
    When:
        Its info is requested.
    Then:
        Both ``max_pos`` entries should equal ``max_width`` rather than the
        genome length. The client's axis has to match the grid the tiles are
        cut from, and a fix that padded per axis with the genome total would
        satisfy the arity tests above while moving both axes to the wrong
        place.
    """
    # Arrange
    tileset = BBIInteraction2DTileset(bigbed, tile_size=TILE_SIZE)

    # Act
    served = tileset.info().to_dict()

    # Assert
    assert served["max_pos"] == [served["max_width"]] * 2
    assert served["max_width"] == QUADTREE_EXTENT > GENOME_LENGTH


def test_info_should_keep_the_type_specific_fields(bigwig):
    """Test that declaring the arity did not displace the extras beside it.

    Given:
        A bigWig signal tileset, the one type contributing extra info fields.
    When:
        Its info is requested.
    Then:
        It should still carry ``aggregation_modes`` and ``range_modes``. The
        arity is passed as a keyword alongside those extras, so a collision
        between the two is a ``TypeError`` at request time and a silent drop
        is an API the client can no longer see.
    """
    # Arrange
    tileset = BBISignalTileset(bigwig, tile_size=TILE_SIZE)

    # Act
    served = tileset.info().to_dict()

    # Assert
    assert served["aggregation_modes"]
    assert served["range_modes"]
