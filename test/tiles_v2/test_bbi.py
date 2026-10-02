"""Tests for clodius.tiles_v2.bbi.

One module serving bigWig and bigBed through four concrete tilesets that share
a single info builder: a signal tileset emitting dense vectors, an annotation
tileset emitting records, and the two bigInteract tilesets -- links, which is
one-dimensional, and rectangles, which is not. All four derive their info the
same way: build a quadtree over the chromsizes, then pad ``max_pos`` out to the
quadtree extent so the client's axis matches the grid the tiles are cut from.

That padding is the part under test here, in both of its dimensions. It is
applied by deriving a variant of the info, and the served info is the only
place a broken derivation becomes visible -- nothing looked at it before.

How many entries it pads matters as much as the value. ``min_pos`` and
``max_pos`` are per-axis on the wire, so a two-dimensional tileset that
publishes one of each leaves a 2D track with no second axis to lay out. The
arity is the only thing the shared builder takes from its subclass, and
``canvas()`` derives its extent from the scalar ``max_width``, so a wrong
arity is invisible everywhere except the served info.

The fixtures are synthesized with pybigtools in milliseconds. Note that the
chromsizes go to ``write``, not to ``open`` -- the wrapper's ``open`` takes
only a path and a mode.
"""

import base64
import contextlib
import gc
import hashlib
import inspect
import io
import os
import pathlib
import threading
import time

import numpy as np
import pybigtools
import pytest
from numpy.testing import assert_array_equal

from clodius.core.coords import Chromsizes
from clodius.core.policies import LinkPolicy, TilePolicy
from clodius.core.source import Source
from clodius.tiles_v2.bbi import (
    TILE_SIZE,
    BBIAnnotationTileset,
    BBIInteraction2DTileset,
    BBIInteractionLinksTileset,
    BBISignalTileset,
    BBITileset,
    to_bedlike,
    to_interaction,
)

from ..source_helpers import (
    JOIN_TIMEOUT,
    CountingHandle,
    Unseekable,
    handle_factory,
    ladder,
    recording_factory,
    released_together,
)

#: Totals 3000 bp, which a four-bin tile size covers with a quadtree extent of
#: 4096 -- so the padded `max_pos` is distinguishable from the genome length.
CHROMSIZES = {"c1": 1000, "c2": 1500, "c3": 500}

TEST_TILE_SIZE = 4
QUADTREE_EXTENT = 4096


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
                ("c3", 0, 500, 4.0),
            ]
        ),
    )
    return path


@pytest.fixture(scope="module")
def bigbed(tmp_path_factory):
    """A bigBed holding three features."""
    path = str(tmp_path_factory.mktemp("bbi") / "annot.bb")
    pybigtools.open(path, "w").write(
        CHROMSIZES,
        iter(
            [
                ("c1", 10, 20, "a\t100"),
                ("c1", 300, 400, "b\t200"),
                ("c2", 50, 60, "c\t300"),
            ]
        ),
    )
    return path


def interact_record(name, source, target, value="0"):
    """One bed5+13 interact row, as the tab-joined rest of a bigBed record.

    The hull is the record's own start and end; the anchors live in the custom
    fields, which is what makes this schema worth a fixture of its own.

    Fifteen fields, not thirteen. `sourceName` and `sourceStrand` sit between
    the source and target anchors, and omitting them shifts every target index
    by two -- `TARGET_CHROM` reads the target *start* -- so the record is two
    fields short of `INTERACT_FIELDS` and `to_interaction` refuses it outright.
    Nothing caught that while the fixture was only ever read by `info()` tests.

    ``value`` lands at the index `to_interaction` reads as `importance`; a
    non-numeric one exercises the digest fallback.
    """
    source_chrom, source_start, source_end = source
    target_chrom, target_start, target_end = target
    return "\t".join(
        [
            name,
            "0",
            str(value),
            "hg38",
            "0",
            source_chrom,
            str(source_start),
            str(source_end),
            f"{name}_src",
            "+",
            target_chrom,
            str(target_start),
            str(target_end),
            f"{name}_tgt",
            "-",
        ]
    )


@pytest.fixture(scope="module")
def bigInteract(tmp_path_factory):
    """A bed5+13 bigBed holding one interaction per contig."""
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
                (
                    "c3",
                    5,
                    400,
                    interact_record("c", ("c3", 5, 40), ("c3", 360, 400)),
                ),
            ]
        ),
    )
    return path


#: A single 1000 bp contig, so the quadtree extent is 1024 and a zoom-3 tile
#: spans 128 bp -- small enough to place anchors in known tiles by hand.
LINK_CHROMSIZES = {"c1": 1000}


@pytest.fixture(scope="module")
def bigInteractLinks(tmp_path_factory):
    """Three interactions that separate the three link policies at one tile.

    At zoom 3 tile 1 spans [128, 256): ``near`` has both anchors inside,
    ``straddle`` has one, and ``spanner`` has neither but a hull that crosses.
    """
    path = str(tmp_path_factory.mktemp("bbi") / "links.bb")
    pybigtools.open(path, "w").write(
        LINK_CHROMSIZES,
        iter(
            [
                (
                    "c1",
                    10,
                    190,
                    interact_record(
                        "straddle", ("c1", 10, 20), ("c1", 180, 190)
                    ),
                ),
                (
                    "c1",
                    10,
                    310,
                    interact_record(
                        "spanner", ("c1", 10, 20), ("c1", 300, 310)
                    ),
                ),
                (
                    "c1",
                    140,
                    210,
                    interact_record("near", ("c1", 140, 150), ("c1", 200, 210))
                ),
            ]
        ),
    )
    return path


def interaction_tuple(value="0", source=("c1", 10, 20), target=("c1", 30, 40)):
    """An 18-field interact record as ``fetch_records`` returns one."""
    src_chrom, src_start, src_end = source
    tgt_chrom, tgt_start, tgt_end = target
    return (
        src_chrom, 10, 40, "link", "0", str(value), "hg38", "0",
        src_chrom, str(src_start), str(src_end), "src", "+",
        tgt_chrom, str(tgt_start), str(tgt_end), "tgt", "-",
    )


def link_names(payload):
    """The name column of each served interaction."""
    return sorted(r["fields"][3] for r in payload)


def dense_values(payload):
    """The dense block of a payload, decoded to the array it encodes."""
    return np.frombuffer(
        base64.b64decode(payload["dense"]), dtype=payload["dtype"]
    )


def bin_count(payload):
    """Bins in a dense payload, discounting values-per-bin."""
    return len(dense_values(payload)) // payload.get("size", 1)


@pytest.mark.parametrize(
    "policy,expected",
    [
        (LinkPolicy.BOTH, ["near"]),
        (LinkPolicy.EITHER, ["near", "straddle"]),
        (LinkPolicy.HULL, ["near", "spanner", "straddle"]),
    ],
    ids=["both", "either", "hull"],
)
def test_tiles_should_select_the_links_the_policy_asks_for(
    bigInteractLinks, policy, expected
):
    """Test the three link policies against a tile that separates them.

    Given:
        A bigInteract holding one link with both anchors in the tile, one with
        a single anchor in it, and one with neither but a hull that crosses
        it, served under each policy.
    When:
        That tile is served.
    Then:
        Each policy should return a different set -- ``both`` the one link,
        ``either`` adding the single-anchor link, ``hull`` adding the link
        that merely passes over. Nothing else distinguishes ``BOTH`` from
        ``HULL``, so without this the two arms are interchangeable.
    """
    # Arrange
    tileset = BBIInteractionLinksTileset(
        bigInteractLinks, link_policy=policy, tile_size=TEST_TILE_SIZE
    )

    # Act
    (_, payload), = tileset.tiles([tileset.parse_tile_id("u.3.1")])

    # Assert
    assert link_names(payload) == expected


def test_tiles_should_default_to_the_either_link_policy(bigInteractLinks):
    """Test the policy a links tileset takes when none is named.

    Given:
        A links tileset constructed with no ``link_policy``.
    When:
        The separating tile is served.
    Then:
        It should return what ``either`` returns. The default governs the
        scan-versus-seek choice as well as the selection, so it is worth
        pinning rather than reading off the signature.
    """
    # Arrange
    tileset = BBIInteractionLinksTileset(
        bigInteractLinks, tile_size=TEST_TILE_SIZE
    )

    # Act
    (_, payload), = tileset.tiles([tileset.parse_tile_id("u.3.1")])

    # Assert
    assert link_names(payload) == ["near", "straddle"]


def test_link_policy_should_return_the_coerced_value(bigInteractLinks):
    """Test that the property hands back a `LinkPolicy`, not what was passed.

    Given:
        A links tileset constructed with the string ``"both"``.
    When:
        Its ``link_policy`` is read.
    Then:
        It should be `LinkPolicy.BOTH`. The coercion is what the tile loop's
        ``match`` dispatches on -- an uncoerced value falls past all three
        arms -- and this is the only assertion that reads the property at
        all, so without it the member could be reverted to a bare attribute
        with the suite still green.
    """
    # Arrange & act
    with BBIInteractionLinksTileset(
        bigInteractLinks, link_policy="both", tile_size=TEST_TILE_SIZE
    ) as tileset:
        # Assert
        assert tileset.link_policy is LinkPolicy.BOTH


def test_link_policy_should_refuse_assignment_after_construction(
    bigInteractLinks,
):
    """Test the refusal the read-only property was introduced for.

    Given:
        A constructed links tileset.
    When:
        A new policy is assigned to ``link_policy``.
    Then:
        It should raise `AttributeError`. As a bare public attribute, an
        assignment here skipped the coercion and the tile loop's catch-all
        arm then served every interaction whose hull crossed the tile under
        the id of whatever was asked for, with nothing raised -- so the
        refusal is the behaviour, not an incidental property of how it is
        implemented.
    """
    # Arrange
    with BBIInteractionLinksTileset(
        bigInteractLinks, tile_size=TEST_TILE_SIZE
    ) as tileset:
        # Act & assert
        with pytest.raises(AttributeError):
            tileset.link_policy = LinkPolicy.HULL


@pytest.mark.parametrize(
    "pos,expected",
    [("1.1", ["near"]), ("1.0", []), ("0.2", ["spanner"])],
)
def test_tiles_should_place_an_interaction_in_its_own_2d_cell(
    bigInteractLinks, pos, expected
):
    """Test that a 2D tile returns only the links both of its axes admit.

    Given:
        A bigInteract whose links put their two anchors in known tile columns.
    When:
        Cells on and off the diagonal are served.
    Then:
        Each should return only the link whose source falls in the x column
        and whose target falls in the y column. The index query matches on the
        hull, so it returns links the y filter must then reject -- the
        ``(1, 0)`` cell is that case.
    """
    # Arrange
    tileset = BBIInteraction2DTileset(
        bigInteractLinks, tile_size=TEST_TILE_SIZE
    )

    # Act
    (_, payload), = tileset.tiles([tileset.parse_tile_id(f"u.3.{pos}")])

    # Assert
    assert link_names(payload) == expected


def test_tiles_should_refuse_a_2d_tile_whose_y_is_off_the_canvas(
    bigInteractLinks,
):
    """Test the axis the restored bounds check is the only screen for.

    Given:
        A 2D tileset and a tile whose x exists but whose y is past the last
        tile at that zoom.
    When:
        It is served.
    Then:
        It should return a ``TileOutOfBounds`` payload rather than an empty
        tile. ``y`` reaches ``tile_span`` and nothing else, so this is the one
        position the guard restored in ``TileCanvas`` is the sole check on --
        and an unchecked one yields a well-formed range past the genome, zero
        rows, and a tile a client cannot tell from "no interactions here".
    """
    # Arrange
    tileset = BBIInteraction2DTileset(
        bigInteractLinks, tile_size=TEST_TILE_SIZE
    )

    # Act
    (_, payload), = tileset.tiles([tileset.parse_tile_id("u.3.1.99")])

    # Assert
    assert payload["error_type"] == "TileOutOfBounds"


def test_to_interaction_should_rank_by_the_records_value_column():
    """Test the importance an interact record carries in its own fields.

    Given:
        An 18-field record whose value column is numeric.
    When:
        It is converted.
    Then:
        Its importance should be that value, so a client ranks links the way
        the file says rather than by an arbitrary digest.
    """
    # Act
    record = to_interaction(interaction_tuple(value="7.5"), {"c1": 0})

    # Assert
    assert record["importance"] == 7.5


def test_to_interaction_should_fall_back_to_the_digest_for_a_bad_value():
    """Test a value column that is not a number, which interact permits.

    Given:
        An 18-field record whose value column is non-numeric.
    When:
        It is converted.
    Then:
        Its importance should be a stable digest in the unit interval rather
        than raising -- ranking has to stay total over real files.
    """
    # Act
    record = to_interaction(interaction_tuple(value="."), {"c1": 0})

    # Assert
    assert 0.0 <= record["importance"] < 1.0


def test_to_interaction_should_return_none_when_an_anchor_is_unplaceable():
    """Test a record naming a contig the coordinate system does not.

    Given:
        A record whose target anchor sits on an unknown contig.
    When:
        It is converted.
    Then:
        It should return ``None``. A ``KeyError`` here is not a ``TileError``,
        so it would fail the whole batch rather than drop one link.
    """
    # Act
    record = to_interaction(
        interaction_tuple(target=("cUNKNOWN", 30, 40)), {"c1": 0}
    )

    # Assert
    assert record is None


def test_to_interaction_should_raise_when_the_record_is_not_bed5_plus_13():
    """Test a bigBed that is not an interact file at all.

    Given:
        A record with fewer than eighteen fields.
    When:
        It is converted.
    Then:
        It should raise ``ValueError`` naming the schema, rather than reading
        a target coordinate out of whichever column happens to be there.
    """
    # Act & assert
    with pytest.raises(ValueError, match="bed5\\+13"):
        to_interaction(interaction_tuple()[:16], {"c1": 0})


def test_info_should_keep_the_type_specific_fields(bigwig):
    """Test that deriving the padded info does not drop the extras.

    Given:
        A signal tileset, which advertises the aggregations and range modes
        its modifier slot accepts.
    When:
        Its info is requested.
    Then:
        Those fields should survive alongside the padded range. Rebuilding the
        model field by field is the obvious way to derive a variant of a frozen
        model, and it silently drops everything the subclass contributed --
        leaving a client with no way to know which modifiers it may ask for.
    """
    # Arrange
    tileset = BBISignalTileset(bigwig, tile_size=TEST_TILE_SIZE)

    # Act
    served = tileset.info().to_dict()

    # Assert
    assert served["aggregation_modes"]
    assert served["range_modes"]


def test_tiles_should_hold_one_bin_per_tile_slot(bigwig):
    """Test that the grid the tiles are cut from matches the info.

    Given:
        A signal tileset with a four-bin tile size.
    When:
        The whole-genome tile is served.
    Then:
        It should carry exactly four bins. This is the property the
        cross-type conformance suite asserts for the legacy modules, reaching
        the served payload through the same info the previous tests check --
        an update that corrupts the ladder rather than the range shows up
        here rather than there.
    """
    # Arrange
    tileset = BBISignalTileset(bigwig, tile_size=TEST_TILE_SIZE)

    # Act
    (_, payload), = tileset.tiles([tileset.parse_tile_id("u.0.0")])

    # Assert
    assert bin_count(payload) == TEST_TILE_SIZE


def test_tiles_should_return_nothing_when_the_cap_is_zero(bigbed):
    """Test the record cap end to end, through a real tileset.

    Given:
        An annotation tileset over a bigBed holding three features, under a
        policy capping records at zero.
    When:
        The whole-genome tile is served.
    Then:
        It should return no records. A server configured to serve none must
        not emit an unbounded tile, and the slice that implements the cap
        silently inverts at zero to mean "everything".
    """
    # Arrange
    tileset = BBIAnnotationTileset(
        bigbed, tile_size=TEST_TILE_SIZE, policy=TilePolicy(max_records=0)
    )

    # Act
    (_, records), = tileset.tiles([tileset.parse_tile_id("u.0.0")])

    # Assert
    assert records == []


def test_tiles_should_refuse_a_zero_cap_without_reading_the_file(
    bigbed, tmp_path
):
    """Test that a cap of zero costs nothing, not merely that it serves nothing.

    Given:
        An annotation tileset over a copy of the file, capped at zero, whose
        file is removed after construction.
    When:
        The whole-genome tile is served.
    Then:
        It should return no records rather than raising. A missing file is the
        only way to observe the absence of I/O from outside: refusing after the
        read costs every record fetched and digested -- seconds on a zoom-0
        tile -- to produce an empty list.
    """
    # Arrange
    path = tmp_path / "annot.bb"
    path.write_bytes(pathlib.Path(bigbed).read_bytes())
    tileset = BBIAnnotationTileset(
        str(path), policy=TilePolicy(max_records=0), tile_size=TEST_TILE_SIZE
    )
    path.unlink()

    # Act
    (_, records), = tileset.tiles([tileset.parse_tile_id("u.0.0")])

    # Assert
    assert records == []


def test_tiles_should_read_the_file_when_the_cap_is_not_zero(bigbed, tmp_path):
    """Test the control for the refusal above, so it pins the short-circuit.

    Given:
        The same tileset over the same removed file, capped at one instead.
    When:
        The whole-genome tile is served.
    Then:
        It should raise, because it reaches the file. Without this, the test
        above would pass against a build that never reads anything. Either
        error says it reached: construction hands back whatever it opened to
        read the header, so the missing file is refused by the reopen --
        ``pybigtools`` calls that a bad path rather than a read failure -- and
        a tileset that held its reader open would instead fail on the read.
    """
    # Arrange
    path = tmp_path / "annot.bb"
    path.write_bytes(pathlib.Path(bigbed).read_bytes())
    tileset = BBIAnnotationTileset(
        str(path), policy=TilePolicy(max_records=1), tile_size=TEST_TILE_SIZE
    )
    path.unlink()

    # Act & assert
    with pytest.raises((OSError, ValueError)):
        tileset.tiles([tileset.parse_tile_id("u.0.0")])


def test_tiles_should_place_records_by_the_ordering_the_tile_selects(bigbed):
    """Test the ``,cos:`` path against a tileset built on that ordering.

    Given:
        An annotation tileset carrying an alternate chromosome ordering, and a
        second tileset built directly on that ordering.
    When:
        The whole-genome tile is served, selecting the alternate on the first.
    Then:
        Both should place the records identically. The alternate's info is now
        built once at construction rather than per tile, and a memo that
        handed back the wrong ordering's info would misplace every record
        while still looking like a well-formed tile.
    """
    # Arrange
    # Only the two contigs the fixture carries records for: pybigtools drops
    # a contig with no records from the file's chrom list, and asking for one
    # that is not there is a KeyError rather than an empty read.
    default = Chromsizes(("c1", "c2"), (1000, 1500))
    reordered = Chromsizes(("c2", "c1"), (1500, 1000))
    tileset = BBIAnnotationTileset(
        bigbed,
        chromsizes=default,
        chromsizes_alts={"alt": reordered},
        tile_size=TEST_TILE_SIZE,
    )
    reference = BBIAnnotationTileset(
        bigbed, chromsizes=reordered, tile_size=TEST_TILE_SIZE
    )

    # Act
    (_, selected), = tileset.tiles(
        [tileset.parse_tile_id("u.0.0,cos:alt")]
    )
    (_, expected), = reference.tiles([reference.parse_tile_id("u.0.0")])

    # Assert
    assert [r["xStart"] for r in selected] == [r["xStart"] for r in expected]
    assert selected != []


def test_to_bedlike_should_return_none_when_the_contig_is_unknown():
    """Test the converter's own contig check. A regression guard, not a pin.

    Given:
        A raw record on a contig the offsets do not name.
    When:
        It is converted.
    Then:
        It should return ``None``. No call site in the tree can reach this --
        records are named from the same chromsizes the offsets come from -- so
        this guards the converter's contract rather than reproducing a defect.
        The call site used to index the map directly, and a ``KeyError`` there
        is not a ``TileError``, so it would have failed the whole batch.
    """
    # Act
    record = to_bedlike(("cUNKNOWN", 10, 20), {"c1": 0})

    # Assert
    assert record is None


def test_to_bedlike_should_place_a_record_on_a_known_contig():
    """Test the converter's placing path, so the check above is not vacuous.

    Given:
        A raw record on a contig the offsets name.
    When:
        It is converted.
    Then:
        It should return a record positioned at that contig's offset.
    """
    # Act
    record = to_bedlike(("c2", 10, 20), {"c1": 0, "c2": 1000})

    # Assert
    assert record is not None
    assert (record["xStart"], record["xEnd"]) == (1010, 1020)


def test_tiles_should_return_an_error_payload_for_a_tile_off_the_canvas(
    bigwig,
):
    """Test that one bad position does not take the batch with it.

    Given:
        A batch of two tile ids, one inside the canvas and one past its last
        tile.
    When:
        The batch is served.
    Then:
        Both entries should come back, the second an error payload naming the
        refusal. A client batches sixteen tiles per request, and a single bad
        position raising through ``tiles()`` discards the fifteen that were
        servable.
    """
    # Arrange
    tileset = BBISignalTileset(bigwig, tile_size=TEST_TILE_SIZE)
    n_tiles = tileset.info().canvas(1).n_tiles
    ids = [
        tileset.parse_tile_id("u.1.0"),
        tileset.parse_tile_id(f"u.1.{n_tiles}"),
    ]

    # Act
    results = tileset.tiles(ids)

    # Assert
    assert bin_count(results[0][1]) == TEST_TILE_SIZE
    assert results[1][1]["error_type"] == "TileOutOfBounds"


def test_tiles_should_return_every_record_when_the_cap_is_none(bigbed):
    """Test the uncapped path, which now runs through the same thinning call.

    Given:
        An annotation tileset under a policy naming no record cap.
    When:
        The whole-genome tile is served.
    Then:
        It should return every feature. The caller used to short-circuit on
        ``None`` before reaching ``take_most_important``, which left two
        surfaces encoding what ``None`` means and the callee's own branch dead
        -- the one that would be missed when the rule changes.
    """
    # Arrange
    tileset = BBIAnnotationTileset(
        bigbed, tile_size=TEST_TILE_SIZE, policy=TilePolicy(max_records=None)
    )

    # Act
    (_, records), = tileset.tiles([tileset.parse_tile_id("u.0.0")])

    # Assert
    assert len(records) == 3


def test_tiles_should_digest_a_record_without_requiring_md5_for_security(
    bigbed, monkeypatch
):
    """Test that the per-record digest survives a FIPS-enforcing build.

    Given:
        An annotation tileset, and an ``md5`` that refuses any call not marked
        as non-security -- which is how a FIPS build behaves.
    When:
        A tile is served.
    Then:
        It should serve the records anyway. The flag was added to
        ``stable_importance``, but the uid is digested first and passed in, so
        this call raises before the marked one is ever reached and the
        hardening never takes effect.
    """
    # Arrange
    real_md5 = hashlib.md5

    def fips_md5(*args, **kwargs):
        if kwargs.get("usedforsecurity", True) is not False:
            raise ValueError("md5 is not available in FIPS mode")
        return real_md5(*args, **kwargs)

    monkeypatch.setattr(hashlib, "md5", fips_md5)
    tileset = BBIAnnotationTileset(bigbed, tile_size=TEST_TILE_SIZE)

    # Act
    (_, records), = tileset.tiles([tileset.parse_tile_id("u.0.0")])

    # Assert
    assert len(records) == 3


def test_info_should_return_one_position_per_axis_for_a_2d_tileset(
    bigInteract,
):
    """Test the arity of the info a two-dimensional tileset serves.

    Given:
        A bigInteract served as 2D rectangles, which declares ``ndim = 2``.
    When:
        Its info is requested.
    Then:
        ``min_pos`` and ``max_pos`` should each carry two entries. The client
        reads the y extent from ``max_pos[1]``, so a one-entry list leaves a
        2D track with no second axis to lay out -- and nothing in the serving
        path reads these fields, so the served info is the only place the
        mistake is visible.
    """
    # Arrange
    tileset = BBIInteraction2DTileset(bigInteract, tile_size=TEST_TILE_SIZE)

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
        (BBIInteractionLinksTileset, "bigInteract"),
    ],
    ids=["signal", "annotation", "links"],
)
def test_info_should_return_a_single_position_for_a_1d_tileset(
    cls, fixture, request
):
    """Test that the per-axis padding did not widen the 1D types.

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
    tileset = cls(request.getfixturevalue(fixture), tile_size=TEST_TILE_SIZE)

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
        (BBIInteractionLinksTileset, "bigInteract"),
        (BBIInteraction2DTileset, "bigInteract"),
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
    tileset = cls(request.getfixturevalue(fixture), tile_size=TEST_TILE_SIZE)

    # Act
    served = tileset.info().to_dict()

    # Assert
    assert len(served["min_pos"]) == cls.ndim
    assert len(served["max_pos"]) == cls.ndim


def test_info_should_pad_every_axis_to_the_quadtree_extent(bigInteract):
    """Test what each axis is padded *to*, not just how many there are.

    Given:
        A 2D tileset over a genome of 3000 bp, whose quadtree extent is 4096.
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
    tileset = BBIInteraction2DTileset(bigInteract, tile_size=TEST_TILE_SIZE)

    # Act
    served = tileset.info().to_dict()

    # Assert
    assert served["max_pos"] == [served["max_width"]] * 2
    assert served["max_width"] == QUADTREE_EXTENT > sum(CHROMSIZES.values())


#: Every concrete tileset in the module, each with a fixture it can serve and
#: the tile size its ladder is walked at. One table, because every
#: whole-module case parametrizes over the same four tilesets: the narrower
#: `BBI_TILESETS` below is derived from it rather than written again, so a
#: fifth subclass is added here once and inherits every case.
#:
#: A four-bin tile gives the 1D types an eleven-level ladder, 2047 tiles,
#: which walks in milliseconds. The same tile size makes the 2D quadtree
#: 1,398,101 tiles, so the 2D case walks its ladder at the module's own
#: production tile size, where the whole quadtree is 21 tiles. Both are the
#: *whole* ladder, which is the point: no case samples.
TILESET_CASES = [
    (BBISignalTileset, "bigwig", TEST_TILE_SIZE),
    (BBIAnnotationTileset, "bigbed", TEST_TILE_SIZE),
    (BBIInteractionLinksTileset, "bigInteract", TEST_TILE_SIZE),
    (BBIInteraction2DTileset, "bigInteract", TILE_SIZE),
]

TILESET_IDS = ["signal", "annotation", "links", "interaction2d"]


def comparable(payload):
    """A dense payload's non-array keys, with NaN made ``==``-comparable.

    ``min_value``/``max_value`` are already stringified to ``"NaN"`` by the
    encoder, so today this normalization is a no-op; it is here so that a
    future encoder emitting a bare ``float('nan')`` turns into a real
    comparison rather than a test that fails on every tile.
    """
    return {
        key: "NaN" if isinstance(value, float) and np.isnan(value) else value
        for key, value in payload.items()
        if key != "dense"
    }


def assert_same_payload(left, right, label):
    """Assert two served payloads are the same tile, and neither an error.

    Dense payloads go through `assert_array_equal`, which treats NaN in the
    same position as equal -- the signal tiles are full of NaN wherever the
    quadtree extent overshoots the genome, and ``nan != nan`` would fail every
    one of them. Record payloads are plain Python and compare directly.
    """
    assert not is_error(left), f"{label}: {left}"
    assert not is_error(right), f"{label}: {right}"
    if isinstance(left, list) or isinstance(right, list):
        assert left == right, label
        return
    assert_array_equal(
        dense_values(left), dense_values(right), err_msg=label
    )
    assert comparable(left) == comparable(right), label


def is_error(payload):
    """Whether a payload slot carries a refusal rather than a tile."""
    return isinstance(payload, dict) and "error" in payload


def is_populated(payload):
    """Whether a payload holds anything beyond empty or out-of-bounds space."""
    if isinstance(payload, list):
        return bool(payload)
    return bool(np.isfinite(dense_values(payload)).any())


def payload_for(tileset, tile_id):
    """The single payload a tileset serves for one raw tile id."""
    (_, payload), = tileset.tiles([tileset.parse_tile_id(tile_id)])
    return payload


@pytest.mark.parametrize(
    "cls,fixture,tile_size", TILESET_CASES, ids=TILESET_IDS
)
def test_tiles_should_agree_between_a_path_and_a_file_like_source(
    cls, fixture, tile_size, request
):
    """Test the two source shapes against each other across the whole ladder.

    Given:
        The same BBI file addressed by path and by a callable opening a fresh
        binary handle, for each of the four concrete tilesets.
    When:
        Every tile in the ladder is requested from each.
    Then:
        The payloads should match tile for tile, with real tiles among them.
        How a caller spelled the source is not part of the coordinate system,
        so a tileset serving even one tile differently would serve a different
        dataset under the same id. The path shape is also the one that reaches
        ``pybigtools.open`` with a *path*, where a factory reaches it with a
        handle, so this is the only test that pins those two readers to the
        same answer.
    """
    # Arrange
    path = request.getfixturevalue(fixture)
    compared = 0
    populated = 0

    # Act & assert
    with (
        cls(path, tile_size=tile_size) as by_path,
        cls(handle_factory(path), tile_size=tile_size) as by_handle,
    ):
        for tile_id in ladder(by_path):
            left = payload_for(by_path, tile_id)
            right = payload_for(by_handle, tile_id)
            assert_same_payload(left, right, tile_id)
            compared += 1
            populated += is_populated(left)
    assert compared
    assert populated


@pytest.mark.parametrize(
    "modifier", ["mean", "min", "max", "std", "sum", "minMax", "whisker"]
)
def test_tiles_should_agree_between_the_source_shapes_for_every_display_mode(
    bigwig, modifier
):
    """Test every display mode against both source shapes.

    Given:
        A bigWig addressed by path and by a handle factory, and one of the
        seven display modes a signal tile may select.
    When:
        Every tile in the ladder is requested from each in that mode.
    Then:
        The payloads should match tile for tile, with real tiles among them.
        A range mode returns several values per bin and fills the payload one
        column at a time, which is a different read against the reader than
        the single-stat modes take -- so agreeing on ``mean`` does not imply
        agreeing on ``whisker``.
    """
    # Arrange
    compared = 0
    populated = 0

    # Act & assert
    with (
        BBISignalTileset(bigwig, tile_size=TEST_TILE_SIZE) as by_path,
        BBISignalTileset(
            handle_factory(bigwig), tile_size=TEST_TILE_SIZE
        ) as by_handle,
    ):
        for tile_id in ladder(by_path):
            requested = f"{tile_id}.{modifier}"
            left = payload_for(by_path, requested)
            right = payload_for(by_handle, requested)
            assert_same_payload(left, right, requested)
            compared += 1
            populated += is_populated(left)
    assert compared
    assert populated


@pytest.mark.parametrize(
    "cls,fixture,tile_size", TILESET_CASES, ids=TILESET_IDS
)
def test_info_should_agree_between_a_path_and_a_file_like_source(
    cls, fixture, tile_size, request
):
    """Test the served description of the two source shapes.

    Given:
        The same BBI file addressed by path and by a handle factory, for each
        of the four concrete tilesets.
    When:
        Info and chromsizes are requested from each.
    Then:
        Both should be identical. Each is read from the file's own header at
        construction, which is the one place the two shapes open the file
        differently, and a client registers a tileset by its info long before
        it asks for a tile.
    """
    # Arrange
    path = request.getfixturevalue(fixture)

    # Act & assert
    with (
        cls(path, tile_size=tile_size) as by_path,
        cls(handle_factory(path), tile_size=tile_size) as by_handle,
    ):
        assert by_path.info() == by_handle.info()
        assert (
            by_path.chromsizes().to_pairs()
            == by_handle.chromsizes().to_pairs()
        )

#
# The five BBI tilesets take a `Source` where they used to take a path. The
# shapes are pinned at the `Source` level in test/core/test_source.py; what is
# pinned here is the *adoption site* -- that a BBI tileset actually admits each
# of them, refuses the two it must, and refuses them before pybigtools is
# reached. The h5py slice shipped with its shapes passing at the `Source`
# level and failing at the constructor, which is the gap this closes.

#: Each concrete tileset with a fixture it can serve, for the cases that do
#: not walk a ladder and so need no tile size. Derived from `TILESET_CASES`
#: rather than written again: the contract is `BBITileset`'s, and the two
#: lists drifting apart would silently stop covering a subclass.
BBI_TILESETS = [(cls, fixture) for cls, fixture, _ in TILESET_CASES]

#: The same four, for the refusals that never reach a file.
BBI_CLASSES = [cls for cls, _ in BBI_TILESETS]

#: The three path spellings `SourceLike` admits -- the forms `os` accepts and
#: that callers already pass. The fourth shape, a factory, is deliberately
#: absent: the two ladder-equality tests above compare a factory against a
#: path over every tile of every tileset, which subsumes a whole-genome
#: comparison on one of them.
SOURCE_SHAPES = {
    "str": lambda path: path,
    "bytes": os.fsencode,
    "pathlike": pathlib.Path,
    # `Source` itself, which `SourceLike` admits and `Source.coerce` passes
    # straight back out. It was admitted to the union without a test at any
    # adoption site, so this row is what pins that a tileset takes one.
    "source": lambda path: Source(path=path),
}


def whole_genome_tile_id(tileset):
    """The zoom-0 tile id, with one position per axis the tileset declares."""
    return "u.0." + ".".join(["0"] * tileset.ndim)


def exploding_open(*args, **kwargs):
    """A ``pybigtools.open`` that fails if a refusal ever reaches it."""
    raise AssertionError("pybigtools.open was reached")


@pytest.mark.parametrize("cls,fixture", BBI_TILESETS, ids=TILESET_IDS)
@pytest.mark.parametrize("shape", list(SOURCE_SHAPES), ids=list(SOURCE_SHAPES))
def test___init___should_serve_identically_for_every_accepted_source_shape(
    cls, fixture, shape, request
):
    """Test each path spelling ``SourceLike`` admits, at the constructor.

    Given:
        One BBI file addressed three ways -- a ``str`` path, a ``bytes`` path
        and an ``os.PathLike`` -- and a tileset built on each.
    When:
        Info and the whole-genome tile are requested from the shaped tileset
        and from a plain string-path tileset over the same file.
    Then:
        Both should answer identically. Accepting the shape is not enough:
        how a caller spelled the source is not part of the coordinate system,
        so a shape that opened a different reader would serve a different
        dataset under the same tile id. The fourth shape, a factory, is
        covered across the whole ladder by the two agreement tests above.
    """
    # Arrange
    path = request.getfixturevalue(fixture)
    source = SOURCE_SHAPES[shape](path)

    # Act
    with (
        cls(path, tile_size=TEST_TILE_SIZE) as baseline,
        cls(source, tile_size=TEST_TILE_SIZE) as shaped,
    ):
        tile_id = whole_genome_tile_id(shaped)
        expected_info = baseline.info()
        expected = payload_for(baseline, tile_id)
        served_info = shaped.info()
        served = payload_for(shaped, tile_id)

    # Assert
    assert served_info == expected_info
    assert served == expected


@pytest.mark.parametrize("cls,fixture", BBI_TILESETS, ids=TILESET_IDS)
def test___init___should_accept_the_source_as_a_keyword(
    cls, fixture, request
):
    """Test the name the renamed first parameter is now reachable under.

    Given:
        A BBI file and a tileset constructed with ``source=`` rather than
        positionally.
    When:
        Info is requested from it and from a positionally constructed one.
    Then:
        Both should agree. ``path`` was the published name of this parameter,
        so the rename moved a keyword every caller could have been using;
        ``source=`` has to be the name that works, and the positional form
        every call site in the tree uses has to keep working beside it.
    """
    # Arrange
    path = request.getfixturevalue(fixture)

    # Act
    with (
        cls(path, tile_size=TEST_TILE_SIZE) as positional,
        cls(source=path, tile_size=TEST_TILE_SIZE) as by_keyword,
    ):
        expected = positional.info()
        served = by_keyword.info()

    # Assert
    assert served == expected


@pytest.mark.parametrize("cls,fixture", BBI_TILESETS, ids=TILESET_IDS)
def test___init___should_refuse_an_already_open_file(cls, fixture, request):
    """Test the one file-like shape that is not accepted, at construction.

    Given:
        An open binary file rather than a callable that opens one.
    When:
        A tileset is constructed from it.
    Then:
        It should raise ``TypeError`` naming the callable the caller wants.
        pybigtools reads a handle happily, so nothing here would fail -- and
        the repair a caller reaches for, ``lambda: handle``, hands the same
        exhausted handle back on a reopen, which is wrong answers rather than
        a crash.
    """
    # Arrange
    path = request.getfixturevalue(fixture)

    # Act & assert
    with open(path, "rb") as handle:
        with pytest.raises(
            TypeError, match="callable that opens a fresh handle"
        ):
            cls(handle, tile_size=TEST_TILE_SIZE)


@pytest.mark.parametrize("cls", BBI_CLASSES, ids=TILESET_IDS)
def test___init___should_refuse_a_source_that_is_neither_path_nor_callable(
    cls,
):
    """Test a source outside the union entirely.

    Given:
        An ``int`` where a path or a factory belongs.
    When:
        A tileset is constructed from it.
    Then:
        It should raise ``TypeError`` naming the type it got. A widened
        parameter is where an unchecked argument stops being a ``TypeError``
        from ``open()`` and starts being whatever the reader makes of it.
    """
    # Act & assert
    with pytest.raises(TypeError, match="not int"):
        cls(42, tile_size=TEST_TILE_SIZE)


@pytest.mark.parametrize("cls,fixture", BBI_TILESETS, ids=TILESET_IDS)
def test___init___should_refuse_a_factory_that_returns_a_text_handle(
    cls, fixture, request, monkeypatch
):
    """Test the mode a caller forgets, refused before the reader sees it.

    Given:
        A factory returning a handle opened without ``'b'``, and a
        ``pybigtools.open`` that fails if it is reached at all.
    When:
        A tileset is constructed from it.
    Then:
        It should raise ``TypeError`` naming the mode. pybigtools does not
        refuse a text handle for being one -- it reports ``BBIReadError:
        File-like object is not a bigWig or bigBed``, which sends the caller
        looking for a corrupt file instead of a missing ``'b'``.
    """
    # Arrange
    path = request.getfixturevalue(fixture)
    monkeypatch.setattr(pybigtools, "open", exploding_open)

    # Act & assert
    with pytest.raises(TypeError, match="binary handle, not a text one"):
        cls(lambda: open(path), tile_size=TEST_TILE_SIZE)


@pytest.mark.parametrize("cls", BBI_CLASSES, ids=TILESET_IDS)
def test___init___should_refuse_a_factory_that_returns_an_unseekable_handle(
    cls, monkeypatch
):
    """Test the capability a BBI reader cannot do without, named as missing.

    Given:
        A factory returning a readable handle that reports it cannot seek,
        and a ``pybigtools.open`` that fails if it is reached at all.
    When:
        A tileset is constructed from it.
    Then:
        It should raise ``TypeError`` naming seekability. A BBI file is read
        by seeking its index, so a streaming source cannot back one -- and
        the reader discovers that as a read error somewhere in the middle of
        a tile rather than as a refusal of the source.
    """
    # Arrange
    monkeypatch.setattr(pybigtools, "open", exploding_open)

    # Act & assert
    with pytest.raises(TypeError, match="seekable"):
        cls(lambda: Unseekable(), tile_size=TEST_TILE_SIZE)


@pytest.mark.parametrize("cls", BBI_CLASSES, ids=TILESET_IDS)
def test___init___should_refuse_a_source_built_around_an_arity_mismatch(
    cls, monkeypatch
):
    """Test that a pre-normalized `Source` cannot smuggle a bad factory in.

    Given:
        A `Source` constructed directly around ``builtins.open``, which is
        callable but takes arguments, and a ``pybigtools.open`` that fails if
        it is reached at all.
    When:
        A tileset is constructed from it.
    Then:
        It should raise ``TypeError`` at construction naming the repair.
        `Source` is a member of ``SourceLike``, and `Source.coerce` passes an
        already-normalized one straight back out, so a guard living only in
        `coerce` is bypassed by the documented shape -- the failure then
        arrives frames later as "open() missing required argument 'file'".
    """
    # Arrange
    monkeypatch.setattr(pybigtools, "open", exploding_open)

    # Act & assert
    with pytest.raises(TypeError, match="must take no arguments"):
        cls(Source(factory=open), tile_size=TEST_TILE_SIZE)


@pytest.mark.parametrize("cls", BBI_CLASSES, ids=TILESET_IDS)
def test___init___should_refuse_a_factory_whose_handle_cannot_be_closed(
    cls, monkeypatch
):
    """Test the capability the tileset itself needs, as against the reader.

    Given:
        A factory returning a seekable binary handle with no ``close``, and a
        ``pybigtools.open`` that fails if it is reached at all.
    When:
        A tileset is constructed from it.
    Then:
        It should raise ``TypeError`` naming ``close``, rather than a bare
        ``AttributeError`` from inside `close` with the handle already
        detached and unreachable. The tileset opened the handle, so releasing
        it is the tileset's obligation, and construction releases what it
        read -- which is where a handle that cannot be closed now surfaces.
    """
    # Arrange
    monkeypatch.setattr(pybigtools, "open", exploding_open)

    # A plain duck-typed handle, not an `io` subclass: every `io` base
    # supplies a `close`, and the point is a handle that has none at all.
    class NoClose:
        def read(self, size=-1):
            return b""

        def seek(self, offset, whence=0):
            return 0

        def tell(self):
            return 0

        def seekable(self):
            return True

    # Act & assert
    with pytest.raises(TypeError, match="close"):
        cls(lambda: NoClose(), tile_size=TEST_TILE_SIZE)


def test___init___should_compose_a_link_policy_with_a_factory_source(
    bigInteractLinks,
):
    """Test the one subclass whose own keyword sits in front of the source.

    Given:
        A links tileset built from a factory and the strictest link policy,
        over a tile that separates the three policies.
    When:
        That tile is served.
    Then:
        It should return only the link with both anchors in view. This is the
        one subclass that adds a keyword of its own to the five its base
        publishes, and it coerces that keyword before the base constructor
        opens anything -- so a factory landing in the wrong slot, or a policy
        read after the source was consumed, would be indistinguishable here
        from a policy that stopped applying.
    """
    # Arrange
    factory = handle_factory(bigInteractLinks)

    # Act
    with BBIInteractionLinksTileset(
        factory, link_policy=LinkPolicy.BOTH, tile_size=TEST_TILE_SIZE
    ) as tileset:
        payload = payload_for(tileset, "u.3.1")

    # Assert
    assert link_names(payload) == ["near"]


def test_tiles_should_refuse_a_text_handle_when_chromsizes_defers_the_open(
    bigwig,
):
    """Test where the refusal lands when construction reads no header.

    Given:
        A text-handle factory and an explicit ``chromsizes``, which is the one
        argument that stops the constructor from opening the file to read the
        file's own header.
    When:
        The tileset is constructed and then asked for a tile.
    Then:
        Construction should succeed and ``tiles`` should raise ``TypeError``.
        The refusal is a property of opening, not of constructing, so passing
        ``chromsizes`` moves it to the first request -- and it raises through
        the batch rather than coming back as a per-tile error payload, so one
        bad source discards every other tile in the request.
    """
    # Arrange
    chromsizes = Chromsizes(tuple(CHROMSIZES), tuple(CHROMSIZES.values()))
    tileset = BBISignalTileset(
        lambda: open(bigwig), chromsizes=chromsizes, tile_size=TEST_TILE_SIZE
    )

    # Act & assert
    with pytest.raises(TypeError, match="binary handle, not a text one"):
        tileset.tiles([tileset.parse_tile_id("u.0.0")])


def test___repr___should_name_the_source_the_tileset_serves(bigwig):
    """Test what identifies a tileset in a traceback under each shape.

    Given:
        Tilesets built from a path and from a factory.
    When:
        Each is rendered.
    Then:
        Each should name its class and its source. A factory has no filename
        to report, so a server holding many datasets needs the shape named
        rather than a path silently rendered as something else.
    """
    # Act & assert
    with BBISignalTileset(bigwig, tile_size=TEST_TILE_SIZE) as by_path:
        assert repr(by_path) == f"<BBISignalTileset {bigwig}>"
    with BBISignalTileset(
        handle_factory(bigwig), tile_size=TEST_TILE_SIZE
    ) as by_factory:
        assert repr(by_factory) == "<BBISignalTileset <file-like source>>"

#
# `pybigtools.open` dispatches on the file extension, so a path it recognizes
# can be handed over as a *string* -- the native route, which is also what a
# reader that had never heard of `Source` would have been given. Anything else
# has to be opened here and handed over as a live Python handle. The two
# routes serve byte-identical tiles, so equality assertions cannot tell them
# apart; only the argument `pybigtools.open` receives can. The h5py slice
# shipped a handle for a path and nothing in the suite noticed.


def recording_open(monkeypatch):
    """Spy on `pybigtools.open`, recording both sides of every call.

    Returns ``(seen, readers)``: the type of each argument handed over, which
    is the only thing that separates the path route from the handle route,
    and each reader handed back, which is the only thing a leak check can
    look at. One spy rather than two, because a test wanting either wants the
    real open underneath and the same ``monkeypatch`` seam.
    """
    seen = []
    readers = []
    real_open = pybigtools.open

    def recorder(target, *args, **kwargs):
        seen.append(type(target))
        reader = real_open(target, *args, **kwargs)
        readers.append(reader)
        return reader

    monkeypatch.setattr(pybigtools, "open", recorder)
    return seen, readers


def routes_taken(seen):
    """Which `pybigtools.open` route each recorded argument type took.

    A set, because a tileset reaches ``pybigtools.open`` more than once over
    its life -- once for the header construction reads and hands back, and
    again for the reader opened afterwards -- and both go through the
    same branch. An argument that is neither a path nor a handle lands in the
    set as its own type, so a third route cannot pass as either of these two.
    """
    return {
        "path"
        if issubclass(arg_type, str)
        else "handle"
        if issubclass(arg_type, io.IOBase)
        else arg_type
        for arg_type in seen
    }


class PathLikeFactory:
    """A factory whose own string form resembles a recognized BBI path."""

    def __init__(self, path):
        self._path = path

    def __str__(self):
        return self._path

    def __call__(self):
        return open(self._path, "rb")


#: Each row is a filename suffix, a fixture to copy under that name, and the
#: route `pybigtools.open` must then be reached by. One table rather than four
#: tests: the Arrange and Act were identical five lines in each, and only the
#: expected route ever differed -- so a sixth spelling is a row.
#:
#: `BBISignalTileset` serves every row because the routing is `BBITileset`'s,
#: decided from the source alone, and no row serves a tile -- it only opens
#: the reader, which any subclass does the same way.
SUFFIX_ROUTES = [
    (".bw", "bigwig", "path"),
    (".bb", "bigbed", "path"),
    (".bigWig", "bigwig", "path"),
    (".BW", "bigwig", "handle"),
    (".BIGBED", "bigbed", "handle"),
    (".bigInteract", "bigInteract", "handle"),
]

SUFFIX_ROUTE_IDS = [
    "bw",
    "bb",
    "camelcase_bigwig",
    "shouted_bw",
    "shouted_bigbed",
    "unrecognized",
]


@pytest.mark.parametrize(
    "suffix,fixture,route", SUFFIX_ROUTES, ids=SUFFIX_ROUTE_IDS
)
def test_reading_should_route_a_path_by_the_exact_spellings_pybigtools_takes(
    suffix, fixture, route, request, tmp_path, monkeypatch
):
    """Test the suffix check against three spellings and three near misses.

    Given:
        A copy of a BBI file named with one suffix -- three that
        `pybigtools.open` dispatches on, including the canonical UCSC
        ``.bigWig``, and three it does not -- and a recording wrapper around
        that call.
    When:
        The reader is opened.
    Then:
        A dispatchable suffix should be handed over as a path string and
        every other one as a live handle. The comparison is case-sensitive by
        design: ``.bigWig`` takes the path route because it is one of the six
        exact spellings pybigtools accepts, not because a case fold rescued
        it, and ``.BW`` takes the handle route because it is not. Folding the
        comparison is what shipped before, and it sent ``signal.BW`` over as
        a path for pybigtools to refuse as an invalid file type -- turning an
        openable file into an unopenable tileset, where failing to recognize
        a suffix costs only a slower open. The two routes serve identical
        tiles, so no equality assertion separates them; only the argument
        ``pybigtools.open`` receives can.
    """
    # Arrange
    # The fixture resolves first: it writes its file through
    # `pybigtools.open`, and building it under the recorder would log that
    # write whenever this is the first test in the module to ask for it.
    source = request.getfixturevalue(fixture)
    path = tmp_path / f"copy{suffix}"
    path.write_bytes(pathlib.Path(source).read_bytes())
    seen, _ = recording_open(monkeypatch)

    # Act
    with BBISignalTileset(str(path), tile_size=TEST_TILE_SIZE) as tileset:
        with tileset.reading():
            pass

    # Assert
    assert routes_taken(seen) == {route}


def test_reading_should_hand_pybigtools_a_handle_for_a_factory_source(
    bigbed, monkeypatch
):
    """Test that the fast path keys off the source, not its string form.

    Given:
        A factory-backed source whose own ``__str__`` is a path ending in a
        recognized suffix.
    When:
        The reader is opened.
    Then:
        It should receive a live handle. A suffix check that sniffed the
        stringified source instead of asking the source for its path would
        take the fast path here, bypass the caller's callable entirely, and
        read a local file where a remote fetch was asked for -- while serving
        byte-identical tiles from this fixture.
    """
    # Arrange
    seen, _ = recording_open(monkeypatch)
    source = PathLikeFactory(bigbed)
    assert str(source).lower().endswith(".bb")

    # Act
    with BBIAnnotationTileset(source, tile_size=TEST_TILE_SIZE) as tileset:
        with tileset.reading():
            pass

    # Assert
    assert routes_taken(seen) == {"handle"}


def test_reading_should_open_the_reader_once_across_repeated_access(
    bigbed, monkeypatch
):
    """Test that the reader is a memo rather than an open per access.

    Given:
        A tileset given its chromsizes up front, so nothing opens at
        construction, and a recording wrapper around `pybigtools.open`.
    When:
        the reader is taken several times.
    Then:
        It should open once and hand back the same reader. Serving a tile
        takes the reader once per genomic range, so an open per access is a
        descriptor and a header parse per range on the hottest path in the
        module -- and the tiles are identical either way.
    """
    # Arrange
    chromsizes = Chromsizes(tuple(CHROMSIZES), tuple(CHROMSIZES.values()))
    seen, _ = recording_open(monkeypatch)

    # Act
    with BBIAnnotationTileset(
        bigbed, chromsizes=chromsizes, tile_size=TEST_TILE_SIZE
    ) as tileset:
        cold = seen[:]
        readers = []
        for _ in range(5):
            with tileset.reading() as reader:
                readers.append(reader)

    # Assert
    assert cold == []
    assert seen == [str]
    assert all(reader is readers[0] for reader in readers)


class RecordingReader:
    """A `pybigtools` reader proxy that logs the name of every call made."""

    def __init__(self, inner, log):
        self._inner = inner
        self._log = log

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def recording(*args, **kwargs):
            self._log.append(name)
            return attr(*args, **kwargs)

        return recording


def recording_signal_tileset(log):
    """A `BBISignalTileset` whose reader logs its calls into ``log``."""

    class Recorded(BBISignalTileset):
        def _reader_open(self, target):
            return RecordingReader(super()._reader_open(target), log)

    return Recorded


@pytest.mark.parametrize("shape", ["factory", "path"])
def test_tiles_should_read_the_reader_before_serving_through_it(bigwig, shape):
    """Test that a published reader has already been read from.

    Given:
        A signal tileset whose reader records every call, closed so that the
        next request opens a cold one.
    When:
        A tile is served.
    Then:
        The first call on that reader should be the header read, not the
        tile's own query. `BBITileset._validate` makes it, inside the open
        and under the lock that serializes opens, so exactly one thread
        issues it with nothing published -- which is also what gives BBI the
        format refusal the base's no-op `_validate` left unimplemented. Both
        shapes, because the open path is shared and only the lock differs.
    """
    # Arrange
    log = []
    source = handle_factory(bigwig) if shape == "factory" else bigwig
    tileset = recording_signal_tileset(log)(
        source, tile_size=TEST_TILE_SIZE
    )
    tileset.close()
    log.clear()

    # Act
    try:
        served = payload_for(tileset, "u.0.0")
    finally:
        tileset.close()

    # Assert
    assert not is_error(served)
    assert log[0] == "chroms"
    assert "values" in log


def test_tiles_should_refuse_a_zero_cap_without_reading_a_factory_source(
    bigbed,
):
    """Test that a cap of zero costs no I/O on a source with no path.

    Given:
        An annotation tileset over a factory handing out read-counting
        handles, capped at zero, opened and its read count recorded.
    When:
        The whole-genome tile is served.
    Then:
        The count should not move. The path-backed pair above observes the
        absence of I/O by deleting the file, which a factory-backed source
        does not admit -- the handle is already open, so the file can be gone
        and the reads still happen. Counting is the probe that works where
        deletion cannot; it does not replace the deletion pair, because a
        recognized-suffix path source opens no Python handle here at all and
        so has nothing to count through.
    """
    # Arrange
    factory, opened = recording_factory(bigbed, CountingHandle)
    with BBIAnnotationTileset(
        factory, policy=TilePolicy(max_records=0), tile_size=TEST_TILE_SIZE
    ) as tileset:
        # The last handle, not the first: construction reads the file's own
        # header and hands that handle back, so the read below opens a second.
        with tileset.reading():
            pass
        handle = opened[-1]
        before = handle.reads

        # Act
        records = payload_for(tileset, whole_genome_tile_id(tileset))

        # Assert
        assert records == []
        assert handle.reads == before


def test_tiles_should_read_a_factory_source_when_the_cap_is_not_zero(
    bigbed,
):
    """Test the control for the refusal above, so it pins the short-circuit.

    Given:
        The same counting factory and tileset, capped at one instead.
    When:
        The whole-genome tile is served.
    Then:
        The read count should move. Without this the test above passes
        against a build that never reads anything -- and against a counting
        handle pybigtools happens never to call.
    """
    # Arrange
    factory, opened = recording_factory(bigbed, CountingHandle)
    with BBIAnnotationTileset(
        factory, policy=TilePolicy(max_records=1), tile_size=TEST_TILE_SIZE
    ) as tileset:
        with tileset.reading():
            pass
        handle = opened[-1]
        before = handle.reads

        # Act
        records = payload_for(tileset, whole_genome_tile_id(tileset))

        # Assert
        assert len(records) == 1
        assert handle.reads > before

#: A coordinate system `TilesetInfo.quadtree` refuses: its total length is
#: zero, and `_quadtree_depth` demands a positive one. Handed in as an
#: alternate ordering it fails *after* the reader is open, which is the only
#: way to reach a post-open failure here -- `pybigtools.open` refuses a
#: non-BBI at open time, and an explicit ``chromsizes`` means the constructor
#: never touches the file at all.
UNSERVABLE_CHROMSIZES = Chromsizes(("c0",), (0,))


def is_closed(reader):
    """Whether a `pybigtools.BBIReader` has been closed."""
    try:
        reader.chroms()
    except pybigtools.BBIFileClosed:
        return True
    return False


def test___init___should_close_the_handle_it_opened_when_the_info_is_rejected(
    bigwig,
):
    """Test the ownership rule on the path where nothing is handed back.

    Given:
        A factory-backed tileset carrying an alternate ordering whose total
        length is zero, which the quadtree builder refuses after the reader
        has already been opened.
    When:
        Construction fails.
    Then:
        The handle it opened should already be closed. The caller never
        receives an object, so there is nothing for them to close -- and a
        pooled remote handle is not released by falling out of scope the way
        a local file is.
    """
    # Arrange
    factory, opened = recording_factory(bigwig)

    # Act
    with pytest.raises(ValueError, match="total_length must be positive"):
        BBISignalTileset(
            factory,
            chromsizes_alts={"bad": UNSERVABLE_CHROMSIZES},
            tile_size=TEST_TILE_SIZE,
        )

    # Assert
    assert opened and all(handle.closed for handle in opened)


def test___init___should_close_the_reader_it_opened_when_the_info_is_rejected(
    bigwig, monkeypatch
):
    """Test that the same failure does not strand a path-backed reader.

    Given:
        A path-backed tileset -- a suffix ``pybigtools`` takes directly, so no
        handle is opened -- carrying the same unservable alternate ordering.
    When:
        Construction fails.
    Then:
        Every reader opened should be closed. The handle test above says
        nothing about this case, because a path-backed source opens no handle:
        the reader is the only thing to release, and leaving it to the garbage
        collector holds the descriptor for an unbounded stretch.
    """
    # Arrange
    _, readers = recording_open(monkeypatch)

    # Act
    with pytest.raises(ValueError, match="total_length must be positive"):
        BBISignalTileset(
            bigwig,
            chromsizes_alts={"bad": UNSERVABLE_CHROMSIZES},
            tile_size=TEST_TILE_SIZE,
        )

    # Assert
    assert readers and all(is_closed(reader) for reader in readers)


def test_close_should_release_both_the_reader_and_the_handle(bigwig):
    """Test the ownership rule for a caller-supplied factory.

    Given:
        A factory-backed tileset, used once so that it opens.
    When:
        It is closed.
    Then:
        Both the reader and the handle underneath it should be released.
        ``BBIReader.close()`` does not close a Python file object it was
        handed, so closing only one of the two leaks a descriptor per
        tileset -- which on a server that caches tilesets is a leak
        proportional to the number of datasets served.
    """
    # Arrange
    factory, opened = recording_factory(bigwig)
    tileset = BBISignalTileset(factory, tile_size=TEST_TILE_SIZE)
    with tileset.reading() as reader:
        pass

    # Act
    tileset.close()

    # Assert
    assert is_closed(reader)
    assert opened and all(handle.closed for handle in opened)


def test_close_should_release_the_reader_when_the_source_is_a_path(bigwig):
    """Test that a path-backed tileset releases the one thing it owns.

    Given:
        A path-backed tileset, used once so that it opens.
    When:
        It is closed.
    Then:
        Its reader should be closed. ``close`` used to drop the reference and
        leave the reader to the garbage collector, which releases the
        descriptor whenever it next runs rather than when the caller asked --
        and a path-backed source opens no handle, so the reader is the only
        thing a leak can show up in.
    """
    # Arrange
    tileset = BBISignalTileset(bigwig, tile_size=TEST_TILE_SIZE)
    with tileset.reading() as reader:
        pass

    # Act
    tileset.close()

    # Assert
    assert is_closed(reader)


def test_close_should_succeed_when_called_twice(bigwig):
    """Test that releasing an already-released tileset is harmless.

    Given:
        A factory-backed tileset that has been opened and closed once.
    When:
        It is closed again.
    Then:
        It should not raise, and every handle should stay closed. A server
        closing a tileset it has already dropped is routine, and a second
        ``close`` that re-entered the release path would raise on a reader
        that is already shut.
    """
    # Arrange
    # Two handles, not one: construction opens to read the file's own header
    # and hands that one back, and the read below opens the second.
    factory, opened = recording_factory(bigwig)
    tileset = BBISignalTileset(factory, tile_size=TEST_TILE_SIZE)
    with tileset.reading():
        pass
    tileset.close()

    # Act
    tileset.close()

    # Assert
    assert len(opened) == 2
    assert all(handle.closed for handle in opened)


def test_tiles_should_reopen_the_source_when_the_tileset_was_closed(bigwig):
    """Test that a closed tileset is reusable rather than spent.

    Given:
        A factory-backed tileset that has served a tile and been closed.
    When:
        The same tile is requested again.
    Then:
        It should be served identically, from one more handle than before.
        ``close`` nulls the cached reader so the property re-opens, which is
        what makes ``with`` followed by reuse work -- and a ``close`` that
        stopped nulling would break this silently.
    """
    # Arrange
    factory, opened = recording_factory(bigwig)
    tileset = BBISignalTileset(factory, tile_size=TEST_TILE_SIZE)
    tile_id = whole_genome_tile_id(tileset)
    before = payload_for(tileset, tile_id)
    # Counted after the first serve rather than assumed: construction opens
    # once for the header and hands it back, so the serve above is already
    # the second open.
    first_round = len(opened)

    # Act
    tileset.close()
    after = payload_for(tileset, tile_id)
    tileset.close()

    # Assert
    assert before == after
    assert "error" not in before
    assert len(opened) == first_round + 1
    assert all(handle.closed for handle in opened)


def test___exit___should_release_the_reader_and_the_handle(bigwig):
    """Test that the block form releases what the tileset opened.

    Given:
        A factory-backed tileset used inside a ``with`` block.
    When:
        The block exits.
    Then:
        Both the reader and the handle should be released. A caller who never
        calls ``close`` by hand gets the ownership guarantee only through
        ``__exit__``, so a tileset that is not a context manager leaks every
        descriptor it opens in that style.
    """
    # Arrange
    factory, opened = recording_factory(bigwig)

    # Act
    with BBISignalTileset(factory, tile_size=TEST_TILE_SIZE) as tileset:
        with tileset.reading() as reader:
            assert not is_closed(reader)

    # Assert
    assert is_closed(reader)
    assert opened and all(handle.closed for handle in opened)


#
# A tile server answers requests on a thread pool, so first access to a
# tileset's reader is routinely concurrent: the lazy open behind
# `clodius.tiles_v2._backed.FileBacked.reading` is the one place in this
# module where N threads race over state that must be produced exactly once. These
# tests cover that race on both source shapes, the one reader a thread pool
# then shares on each of them, the descriptor cost of many tilesets over one
# path, and the one liveness promise `close` makes under load.
#
# What is deliberately NOT tested here: reading through a reader reference
# already in hand while another thread closes the tileset. The fast path of
# the lazy open is unlocked on purpose, and the implementation documents it as
# the caller's rather than something it prevents -- a test asserting it is safe
# would pin behaviour the class explicitly disclaims, and widening the lock to
# honour it would serialize every reader. The last test covers what IS
# promised at that boundary: that `close` returns rather than deadlocking, and
# whatever a concurrent request sees is recorded and not asserted on.
#
# Every wait below is bounded and every thread is joined with a timeout the
# test then asserts on, so a regression that deadlocks fails here instead of
# hanging the suite.

#: Threads released together onto one tileset. Enough that an unsynchronized
#: check-then-act loses reliably; small enough to cost milliseconds.
CONCURRENT_THREADS = 8

#: Tilesets opened over one path in the descriptor test.
CONCURRENT_TILESETS = 20

#: Tiles each thread serves in the three tests that serve under contention.
#:
#: 50 is the measured optimum rather than a round number. Raising it makes
#: the borrow race *less* likely to fire, not more -- at 150 a run of the
#: cold case produced no refusals at all against a build with no lock, where
#: 50 produced them every time. Widening the tile is worse again. The race is
#: what those tests observe, not what they rely on: the mechanism is pinned
#: deterministically by
#: `test_reading_should_admit_one_thread_at_a_time_on_a_shared_handle`.
ROUNDS_PER_THREAD = 50

#: How long the deterministic serialization test holds the reader seam open,
#: and how many times each thread does it. Long enough to swamp thread
#: start-up skew, so every thread is inside the seam at once unless something
#: serializes them; short enough that the whole test costs a few hundred
#: milliseconds.
HOLD_SECONDS = 0.02
HOLD_ROUNDS = 3

#: The fixture chromsizes as a tileset takes them. Passed explicitly so that
#: construction reads no header at all: `BBITileset` hands back whatever its
#: own header read opened, so a tileset is cold either way, but a constructor
#: that opened once makes every handle these tests count a handle plus one.
COLD_CHROMSIZES = Chromsizes(tuple(CHROMSIZES), tuple(CHROMSIZES.values()))

#: The contig names every reader of these fixtures reports.
FIXTURE_CONTIGS = tuple(sorted(CHROMSIZES))

#: The process descriptor table, as a directory. macOS and Linux only -- the
#: descriptor test skips elsewhere rather than asserting something weaker.
DEV_FD = "/dev/fd"


def reader_identity(reader):
    """One reader's identity and the contigs it can actually report.

    Identity alone would be satisfied by a closed or half-published reader,
    so the read is what makes "every thread got a working one" an assertion
    rather than an assumption.
    """
    return id(reader), tuple(sorted(reader.chroms()))


def test_reading_should_call_the_factory_once_under_concurrent_first_access(
    bigwig,
):
    """Test the lazy open against the threading a tile server actually does.

    Given:
        A cold factory-backed signal tileset and several threads released
        together.
    When:
        They all touch the file for the first time.
    Then:
        It should call the factory once, hand every thread the same working
        reader, and close that one handle on ``close``. Without a lock every
        thread opens its own, the last writer wins, and the losers are handles
        ``close`` can no longer reach -- a caller's handle has no finalizer to
        fall back on, so against a remote source those are leaked connections
        that accumulate for the life of the worker. `BBITileset` had no lock
        at all before the shared protocol gave it one.
    """
    # Arrange
    factory, opened = recording_factory(bigwig)
    tileset = BBISignalTileset(
        factory, chromsizes=COLD_CHROMSIZES, tile_size=TEST_TILE_SIZE
    )
    seen = []

    def touch():
        with tileset.reading() as reader:
            seen.append(reader_identity(reader))

    # Act
    alive, failures = released_together([touch] * CONCURRENT_THREADS)

    # Assert
    assert alive == [False] * CONCURRENT_THREADS
    assert failures == []
    assert len(seen) == CONCURRENT_THREADS
    assert len({reader for reader, _ in seen}) == 1
    assert {contigs for _, contigs in seen} == {FIXTURE_CONTIGS}
    assert len(opened) == 1
    tileset.close()
    assert [handle.closed for handle in opened] == [True]


def test_reading_should_open_one_reader_when_a_path_is_touched_concurrently(
    bigwig,
):
    """Test the same race on the source shape that owns no handle.

    Given:
        A cold path-backed signal tileset -- a ``.bw`` suffix, so the path
        goes to ``pybigtools.open`` unchanged and nothing is opened here to
        count calls through -- and several threads released together.
    When:
        They all touch the file for the first time.
    Then:
        It should expose one reader and every thread should get a usable one.
        Reader identity is the whole evidence here, and it is the shape that
        matters most: a second reader published over the first is one this
        tileset can never close, and the path case is the one a server runs
        for every caller who never asked for a file-like source.
    """
    # Arrange
    seen = []

    with BBISignalTileset(
        bigwig, chromsizes=COLD_CHROMSIZES, tile_size=TEST_TILE_SIZE
    ) as tileset:

        def touch():
            with tileset.reading() as reader:
                seen.append(reader_identity(reader))

        # Act
        alive, failures = released_together([touch] * CONCURRENT_THREADS)

        # Assert
        assert alive == [False] * CONCURRENT_THREADS
        assert failures == []
        assert len(seen) == CONCURRENT_THREADS
        assert len({reader for reader, _ in seen}) == 1
        assert {contigs for _, contigs in seen} == {FIXTURE_CONTIGS}


#: One contig long enough that a single `values()` call over it spends long
#: enough in `pybigtools` to hand the GIL to another thread mid-query. The
#: module's other fixtures total 3000 bp, which one call finishes inside a
#: single time slice -- so they can only ever pin the path-backed shape,
#: where nothing is shared and two queries cannot overlap. Measured at
#: 40 Mb: without the read lock, most of a 400-tile run comes back as the
#: borrow refusal; with it, none does.
BORROW_CHROMSIZES = {"big": 40_000_000}

#: Interval width the fixture is written at. 40,000 intervals, which
#: `pybigtools` synthesizes in about twenty milliseconds.
BORROW_INTERVAL = 1000


@pytest.fixture(scope="module")
def overlapping_bigwig(tmp_path_factory):
    """A bigWig big enough that two concurrent queries really do overlap."""
    path = str(tmp_path_factory.mktemp("bbi") / "overlapping.bw")
    pybigtools.open(path, "w").write(
        BORROW_CHROMSIZES,
        iter(
            [
                ("big", start, start + BORROW_INTERVAL, float(start % 7))
                for start in range(
                    0, BORROW_CHROMSIZES["big"], BORROW_INTERVAL
                )
            ]
        ),
    )
    return path


@pytest.mark.parametrize("shape", ["factory", "path"])
@pytest.mark.parametrize("opened_first", [True, False], ids=["warm", "cold"])
def test_tiles_should_serve_one_answer_when_threads_share_a_reader(
    overlapping_bigwig, opened_first, shape
):
    """Test the shape where two reads really can be in flight at once.

    Given:
        A signal tileset over a bigWig large enough that one query yields the
        GIL mid-read -- reached once through a factory and once through its
        path -- a second path-backed one to say what the tile is, and several
        threads released together, each serving the same tile repeatedly.
    When:
        Every payload is compared with the path-backed answer.
    Then:
        They should all be identical, with nothing raised. A factory-backed
        reader holds a borrow on the Python handle for the length of a query,
        so two overlapping reads make ``pybigtools`` raise ``RuntimeError:
        Already borrowed`` -- which arrives as that tile's refusal rather
        than as an exception, so the payload comparison is what catches it
        and the empty failure list is what catches a borrow error raised
        anywhere outside the per-tile boundary. This is the shape the read
        lock in ``BBITileset.reading`` exists for, and with it removed most of
        the factory row comes back refused.

        The path row is here because the lock's exemption for it is otherwise
        asserted and never tested: ``pybigtools`` gives a path-backed reader
        its own descriptor, so nothing is shared and there is no borrow to
        contend, and the sibling that covers that shape runs on a 3 kb fixture
        whose own comment concedes two queries cannot overlap inside it. The
        exemption also rests on an unpinned ``pybigtools`` internal -- in a
        process where numpy has not been imported the same path-backed reader
        does refuse overlapping queries -- which `bbi.py` makes true by
        importing numpy at module scope. This row is what would notice if an
        upgrade took the exemption away.

        Run both cold and already-opened, because the two are not the same
        test. A freshly constructed tileset holds nothing -- construction
        hands back what the chromsizes read opened -- so the cold case is
        what a server actually hits on a tileset's first batch, and a lock
        keyed on a published handle rather than on the source would take the
        unlocked branch for every thread of it. Cold was the case that
        failed, at 350 of 400 tiles refused.
    """
    # Arrange
    served = []
    source = (
        handle_factory(overlapping_bigwig)
        if shape == "factory"
        else overlapping_bigwig
    )

    with (
        BBISignalTileset(
            overlapping_bigwig, tile_size=TEST_TILE_SIZE
        ) as reference,
        BBISignalTileset(source, tile_size=TEST_TILE_SIZE) as tileset,
    ):
        tile_id = whole_genome_tile_id(tileset)
        # The expected tile comes from a second tileset, not from this one.
        # Serving one tile here first would populate this reader's own
        # caches, and a cached query returns in a thirtieth of the time an
        # uncached one takes -- fast enough that no two of them overlap, so
        # the test would pass against a build with no lock at all.
        expected = payload_for(reference, tile_id)
        if opened_first:
            with tileset.reading():
                pass

        def serve():
            for _ in range(ROUNDS_PER_THREAD):
                served.append(payload_for(tileset, tile_id))

        # Act
        alive, failures = released_together([serve] * CONCURRENT_THREADS)

        # Assert
        assert alive == [False] * CONCURRENT_THREADS
        assert failures == []
        assert len(served) == CONCURRENT_THREADS * ROUNDS_PER_THREAD
        assert all(payload == expected for payload in served)
        assert not is_error(expected)
        assert bin_count(expected) == TEST_TILE_SIZE


@pytest.mark.skipif(
    not os.path.isdir(DEV_FD), reason=f"no {DEV_FD} on this platform"
)
@pytest.mark.skipif(
    not os.path.isdir(DEV_FD), reason=f"no {DEV_FD} on this platform"
)
def test_close_should_release_a_handle_opened_from_an_unrecognized_path(
    tmp_path, bigwig
):
    """Test the third source shape, where a path still produces a handle.

    Given:
        A bigWig copied to a suffix ``pybigtools`` does not dispatch on, so
        `_needs_handle` is True and `_open` opens the path itself.
    When:
        The tileset serves a tile and is closed.
    Then:
        The descriptor count should return to its baseline, having risen
        while open. The other two shapes are covered -- a factory owns a
        handle and a dispatchable path owns none -- and this one owns a
        handle it opened *from a path*, which no assertion reached. It is
        also the only route that executes `Source.open`'s path branch at all,
        so it is the only arm where this PR's switch to an unbuffered open
        has any effect.
    """
    # Arrange
    misnamed = tmp_path / "signal.bigInteract"
    misnamed.write_bytes(pathlib.Path(bigwig).read_bytes())
    gc.collect()
    baseline = len(os.listdir(DEV_FD))
    tileset = BBISignalTileset(
        str(misnamed), chromsizes=COLD_CHROMSIZES, tile_size=TEST_TILE_SIZE
    )


    # Act
    try:
        served = payload_for(tileset, "u.0.0")
        while_open = len(os.listdir(DEV_FD))
    finally:
        tileset.close()

    # Assert
    assert not is_error(served)
    assert while_open > baseline
    assert len(os.listdir(DEV_FD)) == baseline


def test_close_should_release_every_descriptor_when_many_share_a_path(
    bigwig,
):
    """Test that cycling many tilesets over one path is flat, not cumulative.

    Given:
        Many path-backed signal tilesets over the same file, each with its
        reader opened.
    When:
        They are all closed.
    Then:
        The process descriptor count should return to where it started, having
        risen while they were open. Unlike HDF5, ``pybigtools`` shares nothing
        between readers of one path, so the count while open grows with the
        tileset count and a constant there would pin the reader's internals
        rather than this class's contract. What this class promises is that
        ``close`` reaches everything the lazy open produced, the invariant a
        long-lived server depends on as it registers and drops tilesets; the
        rise is the control that keeps the return to baseline from passing
        against a build that never opened anything.
    """
    # Arrange
    # A reader orphaned by an earlier test is released by refcount at that
    # test's teardown, but one caught in a reference cycle would be reclaimed
    # at an arbitrary later moment and could drop the count below the
    # baseline. Collect first, so the baseline is a floor.
    gc.collect()
    baseline = len(os.listdir(DEV_FD))
    tilesets = [
        BBISignalTileset(
            bigwig, chromsizes=COLD_CHROMSIZES, tile_size=TEST_TILE_SIZE
        )
        for _ in range(CONCURRENT_TILESETS)
    ]

    # Act
    try:
        for tileset in tilesets:
            with tileset.reading():
                pass
        while_open = len(os.listdir(DEV_FD))
    finally:
        for tileset in tilesets:
            tileset.close()

    # Assert
    assert while_open > baseline
    assert len(os.listdir(DEV_FD)) == baseline


@pytest.mark.parametrize("shape", ["factory", "path"])
def test_close_should_return_when_another_thread_is_serving_tiles(
    overlapping_bigwig, shape
):
    """Test that ``close`` neither raises nor strands a reader under load.

    Given:
        An open signal tileset over a bigWig large enough that one query
        really is in flight when the close lands -- reached once through a
        factory and once through its path -- and several threads serving
        tiles released together with one thread that closes it.
    When:
        The close lands while the others are mid-request.
    Then:
        ``close`` should return, every thread should finish, and the tileset
        should still be closeable and reopenable afterwards.

        This is the liveness half, over both source shapes and on a fixture
        where a query really is in flight -- the 3 kb one it used to run
        against finishes a query inside a single time slice. It does **not**
        pin the borrow failure `BBITileset.close` exists to prevent: against
        a page-cached local file a query is about a millisecond, so the close
        almost never lands inside one and this test passes with the override
        removed. The test below, which makes each read cost 10 ms, is the one
        that pins it.

        What a concurrent request *sees* is still the caller's race that
        `_opened` explicitly disclaims -- a request holding a reference from
        before the close may read a closed file -- so anything a serving
        thread raises is recorded rather than asserted on. Only the closing
        thread's failures are failures here.
    """
    # Arrange
    served = []
    raced = []
    source = (
        handle_factory(overlapping_bigwig)
        if shape == "factory"
        else overlapping_bigwig
    )

    tileset = BBISignalTileset(source, tile_size=TEST_TILE_SIZE)
    try:
        tile_id = whole_genome_tile_id(tileset)
        with tileset.reading():
            pass

        def serve():
            try:
                for _ in range(ROUNDS_PER_THREAD):
                    payload_for(tileset, tile_id)
                    served.append(1)
            except BaseException as exc:  # noqa: BLE001 -- see the docstring
                raced.append(type(exc).__name__)

        # Act
        alive, failures = released_together(
            [serve] * CONCURRENT_THREADS + [tileset.close]
        )

        # Assert
        assert alive == [False] * (CONCURRENT_THREADS + 1)
        assert failures == []
        assert served
        # Nothing stranded: a second close is clean, and the tileset reopens.
        tileset.close()
        with tileset.reading() as reader:
            assert sorted(reader.chroms()) == sorted(BORROW_CHROMSIZES)
    finally:
        tileset.close()


@pytest.mark.parametrize(
    "cls,fixture",
    [(BBISignalTileset, "bigwig"), (BBIAnnotationTileset, "bigbed")],
    ids=["signal_fetch", "annotation_fetch_records"],
)
def test_tiles_should_raise_an_unrelated_runtime_error_from_the_reader(
    cls, fixture, monkeypatch, request
):
    """Test that the translation is a needle, not a net.

    Given:
        A tileset whose reader raises a ``RuntimeError`` that is not the
        borrow error.
    When:
        A tile is served.
    Then:
        It should propagate rather than being converted into a refusal. One
        of the two call sites caught bare ``Exception`` before these were
        folded together, so anything whose text happened to contain the
        needle became a per-tile refusal and everything else in that arm was
        decided by a second substring match.
    """
    # Arrange
    path = request.getfixturevalue(fixture)
    tileset = cls(path, tile_size=TEST_TILE_SIZE)

    class BrokenReader:
        def __getattr__(self, name):
            def boom(*args, **kwargs):
                raise RuntimeError("the disk fell off")

            return boom

    @contextlib.contextmanager
    def broken(self):
        yield BrokenReader()

    monkeypatch.setattr(type(tileset), "reading", broken)

    # Act & assert
    with pytest.raises(RuntimeError, match="the disk fell off"):
        tileset.tiles([tileset.parse_tile_id(whole_genome_tile_id(tileset))])
    tileset.close()


@pytest.mark.parametrize(
    "cls,fixture",
    [(BBISignalTileset, "bigwig"), (BBIAnnotationTileset, "bigbed")],
    ids=["signal_fetch", "annotation_fetch_records"],
)
def test_tiles_should_refuse_one_tile_when_the_reader_reports_a_borrow(
    cls, fixture, monkeypatch, request
):
    """Test the translation that keeps a residual borrow inside the batch.

    Given:
        A tileset whose reader raises ``RuntimeError: Already borrowed`` --
        the error ``pybigtools`` raises when two queries overlap on one
        handle -- on the call this tileset reads through.
    When:
        A tile is served.
    Then:
        It should come back as that tile's refusal payload, not as a raised
        exception. ``BaseTileset.tiles`` catches only `TileError`, so an
        untranslated ``RuntimeError`` escapes the per-tile boundary and takes
        the whole batch down as a 500. The read lock makes this unreachable
        through `tiles`; this is the belt to that braces, and it shipped with
        no test at all -- both branches were uncovered, and the two copies of
        it had already drifted to different catch widths.
    """
    # Arrange
    path = request.getfixturevalue(fixture)
    tileset = cls(path, tile_size=TEST_TILE_SIZE)

    class BorrowedReader:
        """A reader that raises what `pybigtools` raises on an overlap.

        Substituted through `reading` rather than patched onto
        `pybigtools.BBIReader`, which cannot be done: that class reaches
        `records` through ``__getattr__`` to its Rust object and so carries
        no class-level attribute to replace. It is the same fact that keeps
        a `Closeable` bound off `FileBacked`'s type parameter.
        """

        def __getattr__(self, name):
            def borrow(*args, **kwargs):
                raise RuntimeError("Already borrowed")

            return borrow

    @contextlib.contextmanager
    def borrowed(self):
        yield BorrowedReader()

    monkeypatch.setattr(type(tileset), "reading", borrowed)

    # Act
    served = payload_for(tileset, whole_genome_tile_id(tileset))

    # Assert
    assert is_error(served)
    assert "concurrently" in served["error"]
    # Flavour-neutral: both classes serve a bigWig or a bigBed, so a message
    # naming one of the two is wrong for half the files that reach it.
    assert "bigWig" not in served["error"]
    assert "bigBed" not in served["error"]
    tileset.close()


def test___init___should_serve_a_misnamed_bbi_through_a_factory(tmp_path, bigwig):
    """Test the escape hatch the suffix comment documents and nothing pinned.

    Given:
        A bigWig named ``.bb`` -- a suffix ``pybigtools`` dispatches to the
        bigBed parser.
    When:
        It is opened by path, and then by factory.
    Then:
        The path route should refuse it and the factory route should serve
        it. ``pybigtools.open(<path>)`` picks the parser from the extension
        while ``pybigtools.open(<handle>)`` sniffs the bytes, so being
        unrecognized is strictly safer than being mis-recognized. The
        comment in `bbi.py` says so and this is what holds it to it: a
        future widening of the fast-path table would break here.
    """
    # Arrange
    misnamed = tmp_path / "signal.bb"
    misnamed.write_bytes(pathlib.Path(bigwig).read_bytes())

    # Act & assert
    with pytest.raises(Exception, match="(?i)bigbed"):
        BBISignalTileset(str(misnamed), tile_size=TEST_TILE_SIZE)

    with BBISignalTileset(
        handle_factory(str(misnamed)), tile_size=TEST_TILE_SIZE
    ) as tileset:
        served = payload_for(tileset, whole_genome_tile_id(tileset))
        assert not is_error(served)


@pytest.mark.parametrize("cls", BBI_CLASSES, ids=TILESET_IDS)
def test___init___should_publish_every_parameter_its_base_publishes(cls):
    """Test the drift the spelled-out subclass signature made possible.

    Given:
        Each BBI tileset class.
    When:
        Its constructor signature is compared with `BBITileset`'s.
    Then:
        It should accept every parameter the base does, with the same default
        and the same kind. `BBIInteractionLinksTileset` writes the base's five
        out by hand so ``inspect.signature`` publishes them, which means a
        sixth added to the base reaches four subclasses and not that one --
        and forwarding positionally would drop it with nothing raised.

        Compared as `inspect.Parameter` objects rather than as names, because
        the names are the half that does not drift: a hand-copied list keeps
        its spelling while a changed default or a parameter moving to
        keyword-only diverges silently, which is the failure the duplication
        actually creates.
    """
    # Act
    base = dict(inspect.signature(BBITileset.__init__).parameters)
    subclass = dict(inspect.signature(cls.__init__).parameters)

    # Assert
    assert set(base) <= set(subclass)
    assert {n: subclass[n] for n in base} == base


@pytest.mark.parametrize("shape", ["factory", "path"])
def test_reading_should_admit_the_same_thread_twice(
    overlapping_bigwig, shape
):
    """Test that the published seam can be nested, for either source shape.

    Given:
        A tileset reached through a factory or through its path.
    When:
        One thread enters `BBITileset.reading` inside a block it already
        holds open.
    Then:
        It should be admitted rather than blocking. `reading` is the
        published way to reach the reader, and nesting it is an ordinary
        thing to do with a context manager -- hold the reader across a
        helper, compare two ranges. A non-reentrant lock made that a
        permanent hang with no exception and no timeout, and only for a
        factory-backed source, so a path-backed test would never have seen
        it. Reentrancy is safe here because both read helpers return before
        the block ends, so a nested acquire on one thread cannot produce two
        overlapping queries.
    """
    # Arrange
    source = (
        handle_factory(overlapping_bigwig)
        if shape == "factory"
        else overlapping_bigwig
    )
    entered = []

    def nest(tileset):
        with tileset.reading() as outer:
            with tileset.reading() as inner:
                entered.append((outer, inner))

    # Act
    with BBISignalTileset(source, tile_size=TEST_TILE_SIZE) as tileset:
        worker = threading.Thread(target=nest, args=(tileset,), daemon=True)
        worker.start()
        worker.join(JOIN_TIMEOUT)

        # Assert
        assert not worker.is_alive(), "nested reading() never returned"
        assert len(entered) == 1
        assert entered[0][0] is entered[0][1]


@pytest.mark.parametrize("opened_first", [True, False], ids=["warm", "cold"])
def test_reading_should_admit_one_thread_at_a_time_on_a_shared_handle(
    overlapping_bigwig, opened_first, monkeypatch
):
    """Test the serialization itself, rather than the race it prevents.

    Given:
        A factory-backed tileset, whose reader is reached through one handle
        this tileset owns, and several threads serving tiles through it.
    When:
        Entries into `BBITileset.reading` are counted as they overlap.
    Then:
        At most one thread should ever be inside it. The sibling tests above
        observe the *consequence* -- ``pybigtools`` refusing an overlapping
        query -- which only shows up when two reads genuinely collide, and
        that is a race either parametrization can lose: measured against a
        build with no lock, each of warm and cold produced runs with no
        refusals at all, so neither is a reliable discriminator alone. This
        counts the mechanism instead, so it fails deterministically the
        moment the serialization goes.
    """
    # Arrange
    depth, peak = 0, []
    lock = threading.Lock()
    real = BBISignalTileset.reading

    @contextlib.contextmanager
    def counting(self):
        nonlocal depth
        with real(self) as reader:
            with lock:
                depth += 1
                peak.append(depth)
            try:
                # Hold the seam open far longer than thread start-up skew, so
                # overlap is forced rather than waited for. Without this the
                # test is back to hoping two reads collide inside one query,
                # which is the race it exists to stop depending on.
                time.sleep(HOLD_SECONDS)
                yield reader
            finally:
                with lock:
                    depth -= 1

    with BBISignalTileset(
        handle_factory(overlapping_bigwig), tile_size=TEST_TILE_SIZE
    ) as tileset:
        tile_id = whole_genome_tile_id(tileset)
        if opened_first:
            with tileset.reading():
                pass
        monkeypatch.setattr(type(tileset), "reading", counting)

        def serve():
            for _ in range(HOLD_ROUNDS):
                payload_for(tileset, tile_id)

        # Act
        alive, failures = released_together([serve] * CONCURRENT_THREADS)

    # Assert
    assert alive == [False] * CONCURRENT_THREADS
    assert failures == []
    assert peak, "no thread entered the seam"
    assert max(peak) == 1


#: Width of one record scan, in bp. A multiple of `BORROW_INTERVAL`, so the
#: record count it must return is exact rather than approximate.
RECORD_SCAN_SPAN = 900_000

#: Scans per thread, and the stride between them. The stride keeps the ranges
#: disjoint, so the reader is doing real work per call rather than answering
#: from its block cache -- a cached read never overlaps another.
RECORD_SCAN_ROUNDS = 20
RECORD_SCAN_STRIDE = 100_000


def record_scanner(tileset, tid, counts):
    """A callable scanning records through `reading`, draining in the block."""

    def scan():
        for k in range(RECORD_SCAN_ROUNDS):
            lo = (tid * RECORD_SCAN_ROUNDS + k) * RECORD_SCAN_STRIDE % 39_000_000
            with tileset.reading() as reader:
                # Drained *inside* the block. This is the invariant, and it is
                # what `fetch_records` does; see the test's docstring for why
                # the other shape is not exercised here.
                records = list(
                    reader.records("big", lo, lo + RECORD_SCAN_SPAN)
                )
            counts.append(len(records))

    return scan


def test_reading_should_serve_concurrent_record_scans_drained_in_the_block(
    overlapping_bigwig,
):
    """Test the invariant `BBITileset.reading`'s contract rests on.

    Given:
        A factory-backed tileset -- the shape that shares one reader -- and
        several threads released together, each scanning records over
        disjoint ranges and draining each scan inside its own `reading`
        block.
    When:
        Every scan is counted.
    Then:
        None should fail and every count should be exact. ``records()`` hands
        back a lazy iterator and *advancing* it is a read, so what makes this
        safe is that the drain finishes before the block does -- which is why
        the eagerness of `fetch` and `fetch_records` is load-bearing rather
        than incidental, and why a future `_tile` that streams records
        instead of accumulating them would break this and nothing else.

        The unsafe shape -- carrying the iterator out and draining it after
        the block -- is deliberately **not** exercised. Measured at 76 of 100
        scans dying with `pyo3_runtime.PanicException` and the remaining 24
        silently returning truncated lists, and one reported run aborted the
        interpreter outright, so a test performing it would take the suite
        down rather than report. The hazard is recorded in `reading`'s
        docstring; this pins the half that must keep working.

        `released_together` collects `BaseException`, which is what makes the
        panic reportable at all: its mro is `PanicException` ->
        `BaseException`, so a narrower catch loses it.
    """
    # Arrange
    counts = []
    expected = RECORD_SCAN_SPAN // BORROW_INTERVAL

    with BBISignalTileset(
        handle_factory(overlapping_bigwig), tile_size=TEST_TILE_SIZE
    ) as tileset:
        scanners = [
            record_scanner(tileset, tid, counts)
            for tid in range(CONCURRENT_THREADS)
        ]

        # Act
        alive, failures = released_together(scanners)

    # Assert
    assert alive == [False] * CONCURRENT_THREADS
    assert failures == []
    assert len(counts) == CONCURRENT_THREADS * RECORD_SCAN_ROUNDS
    assert set(counts) == {expected}


#: Milliseconds a `SlowHandle` read costs, and when the close lands. Sized so
#: the close is inside the query by a wide margin -- a whole-genome tile over
#: `overlapping_bigwig` costs four reads, so ~40 ms, and the close lands 50 ms
#: in from a 1 ms-granularity sleep.
SLOW_READ_SECONDS = 0.010
CLOSE_DELAY_SECONDS = 0.050

#: Bins per tile for that test only. The module's own four-bin size answers a
#: whole-genome tile off a coarse zoom level in one or two reads, so the
#: request was often over before the close landed and the test detected the
#: regression about one run in three. A thousand bins over 40 Mb costs enough
#: reads that the close is reliably inside one.
SLOW_TILE_SIZE = 1024


class SlowHandle:
    """A binary handle whose every read costs `SLOW_READ_SECONDS`.

    What a remote source is: the shape this module exists to serve, and the
    only way to hold a `pybigtools` borrow open long enough to observe what a
    concurrent `close` does to it. A page-cached local read is a thousand
    times faster and the window never opens.
    """

    def __init__(self, path):
        self._inner = open(path, "rb")
        self.closed = False

    def read(self, size=-1):
        time.sleep(SLOW_READ_SECONDS)
        return self._inner.read(size)

    def seek(self, offset, whence=0):
        return self._inner.seek(offset, whence)

    def tell(self):
        return self._inner.tell()

    def close(self):
        self.closed = True
        self._inner.close()


def test_close_should_serve_the_read_it_interrupts_on_a_slow_handle(
    overlapping_bigwig,
):
    """Test the borrow failure ``close`` used to produce, deterministically.

    Given:
        A factory-backed tileset whose handle costs 10 ms a read, one thread
        serving a tile, and a close landing 50 ms into that request.
    When:
        The close runs.
    Then:
        It should return, and the interrupted request should be *served*.
        Without `BBITileset.close` taking the read lock this failed two ways
        at once, 5 of 5: ``close`` called ``BBIReader.close()`` while
        `pybigtools` held its borrow and raised `RuntimeError: Already
        borrowed` -- which is not a `TilesetError`, and ``close`` is what
        `BaseTileset.__exit__` calls, so it escaped out of teardown -- and
        the request died with ``read of closed file``.

        The wait this buys is the query: 37-46 ms measured here, against a
        request of about the same, and a median of 0.15 ms against 0.11 ms
        under ordinary page-cached load.
    """
    # Arrange
    opened = []

    def factory():
        handle = SlowHandle(overlapping_bigwig)
        opened.append(handle)
        return handle

    tileset = BBISignalTileset(factory, tile_size=SLOW_TILE_SIZE)
    try:
        tile_id = whole_genome_tile_id(tileset)
        with tileset.reading():
            pass
        served = []

        def serve():
            served.append(payload_for(tileset, tile_id))

        worker = threading.Thread(target=serve, daemon=True)
        worker.start()
        time.sleep(CLOSE_DELAY_SECONDS)

        # Act
        tileset.close()
        worker.join(JOIN_TIMEOUT)

        # Assert
        assert not worker.is_alive()
        assert served and not is_error(served[0])
        assert opened and all(handle.closed for handle in opened)
    finally:
        tileset.close()
