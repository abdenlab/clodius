"""Tests for clodius.tiles_v2.multivec.

The only tileset combining an explicit ladder with the SEQUENTIAL grid policy,
and the only one with a *shaped* payload: each bin carries a vector of per-row
values, so a tile is ``(n_rows, tile_size)`` rather than a flat run.

Two divergences from the other dense types are asserted rather than normalized.
Padding past the genome end is zeros, not NaN -- multivec means "no signal"
where bigWig means "unmappable", and the client draws them differently. And the
payload omits ``size``/``min_value``/``max_value``, emitting only
``dense``/``dtype``/``shape``.

The fixture stores ``arange(n_bins * n_rows)`` per chromosome, so bin ``i`` of
row ``r`` holds ``n_rows * i + r``. Every value is therefore its own address,
which is what lets a tile assert where each bin came from instead of only how
many there are.
"""

import json

import h5py
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from clodius.core.policies import DEFAULT_POLICY
from clodius.core.coords import GenomicRange
from clodius.core.errors import TileOutOfBounds
from clodius.core.tileset import Ladder
from clodius.tiles_v2 import multivec as mod
from clodius.tiles_v2.multivec import MultivecTileset, bin_slice

from ..harness import genome
from ..harness.wire import decode

#: What the fixture builder writes: three resolutions, four rows, 256-bin tiles.
N_ROWS = 4
TILE_SIZE = 256


#: For the pure-arithmetic properties in this module, which build nothing and
#: take no fixture. They keep the wide budget; only the file-backed properties
#: below pay for a rebuild per example.
PROPERTY = settings(max_examples=200)

# Property tests here rebuild a real bigWig, bigBed, mcool or HDF5 file per
# example, so they cannot run on the ``pure`` profile's 200-example arithmetic
# budget. ``function_scoped_fixture`` is suppressed for the same reason it is
# in ``test/core/test_coords.py``: the fixtures these draw against are files
# built once per test, not state that leaks between examples.
PROPERTY_IO = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@pytest.fixture
def tileset(make_mv5):
    """A MultivecTileset over the canonical 3000 bp genome at 4/8/16 bp."""
    with MultivecTileset(str(make_mv5())) as ts:
        yield ts


def stack(payload):
    """A shaped dense payload as its ``(n_rows, tile_size)`` matrix."""
    return decode(payload).reshape(payload["shape"])


class TestBinSlice:
    """Chromosome-relative bin bounds, which are also the bin count."""

    @pytest.mark.parametrize(
        "start,end,binsize,expected",
        [
            (0, 10, 4, (0, 3)),
            (5, 10, 4, (1, 3)),
            (0, 8, 4, (0, 2)),
            (7, 7, 4, (1, 2)),
        ],
    )
    def test_bin_slice_should_round_outward(
        self, start, end, binsize, expected
    ):
        """Test the slice bounds against a stored bin grid.

        Given:
            An interval and a bin size.
        When:
            Its bin slice is computed.
        Then:
            It should run ``floor(start / b)`` to ``ceil(end / b)``, so the
            returned bins fully cover the interval. Unlike bigWig, the bins are
            materialized in the file, so these bounds *are* the bin count.
        """
        # Act & assert
        assert bin_slice(start, end, binsize) == expected

    @PROPERTY
    @given(
        start=st.integers(min_value=0, max_value=10_000),
        span=st.integers(min_value=0, max_value=10_000),
        binsize=st.integers(min_value=1, max_value=500),
    )
    def test_bin_slice_should_cover_the_interval(self, start, span, binsize):
        """Test that the slice never cuts the interval short.

        Given:
            Any interval and bin size.
        When:
            The slice is computed.
        Then:
            Its first bin should start at or before the interval and its last
            should end at or after it. A slice that fell short would make the
            reconciler raise on the bin count rather than pad.
        """
        # Act
        lo, hi = bin_slice(start, start + span, binsize)

        # Assert
        assert lo * binsize <= start
        assert hi * binsize >= start + span


class TestFetch:
    """Reading one interval's stored bins."""

    def test_fetch_should_return_the_stored_bins_for_the_interval(
        self, make_mv5
    ):
        """Test the ordinary in-bounds read.

        Given:
            An interval covering the first 40 bp of c1 at 4 bp per bin.
        When:
            It is fetched.
        Then:
            It should return the ten stored bins, each carrying its four row
            values -- the file's own numbering, unmodified.
        """
        # Arrange
        with h5py.File(make_mv5(), "r") as f:
            grp = f["resolutions/4/values"]

            # Act
            out = mod.fetch(grp, GenomicRange(0, "c1", 0, 40), 4, N_ROWS)

        # Assert
        assert out.shape == (10, N_ROWS)
        np.testing.assert_array_equal(
            out, np.arange(40, dtype=float).reshape(10, N_ROWS)
        )

    def test_fetch_should_return_zeros_when_out_of_bounds(self):
        """Test the padding contract, which differs from every other type.

        Given:
            An out-of-bounds range, which carries no chromosome name.
        When:
            It is fetched with a group that would raise if touched.
        Then:
            It should return zeros rather than NaN. bigWig fills the band past
            the last chromosome with NaN to mean unmappable; multivec fills it
            with zero to mean no signal, and the client draws the two
            differently.
        """
        # Act
        out = mod.fetch(None, GenomicRange(0, None, 0, 40), 4, N_ROWS)

        # Assert
        assert out.shape == (10, N_ROWS)
        assert np.all(out == 0)


class TestRowMetadataEncodings:
    """Row metadata, which the legacy writer stores three ways.

    Exercised through ``info()`` rather than through the private decoder it
    calls: what matters is that a file written by any of the three legacy
    spellings serves the same labels, and the decoder is reachable from the
    public surface by writing each spelling into a real file.
    """

    def test_info_should_decode_row_infos_stored_as_bytes(self, make_mv5):
        """Test the encoding h5py returns for a string dataset.

        Given:
            A multivec whose ``row_infos`` is JSON stored as bytes.
        When:
            The tileset info is read.
        Then:
            It should carry the parsed labels.
        """
        # Arrange
        path = make_mv5()
        with h5py.File(path, "r+") as f:
            f["info"].create_dataset(
                "row_infos", data=np.bytes_(json.dumps(["a", "b"]))
            )

        # Act
        with MultivecTileset(str(path)) as tileset:
            info = tileset.info().model_dump()

        # Assert
        assert info["row_infos"] == ["a", "b"]

    def test_info_should_decode_row_infos_stored_as_text(self, make_mv5):
        """Test the other encoding the same field arrives in.

        Given:
            A multivec whose ``row_infos`` is JSON stored as an h5py string.
        When:
            The tileset info is read.
        Then:
            It should carry the parsed object, so the two encodings are not
            distinguishable from the wire.
        """
        # Arrange
        path = make_mv5()
        with h5py.File(path, "r+") as f:
            f["info"].create_dataset(
                "row_infos",
                data=json.dumps({"k": 1}),
                dtype=h5py.string_dtype(),
            )

        # Act
        with MultivecTileset(str(path)) as tileset:
            info = tileset.info().model_dump()

        # Assert
        assert info["row_infos"] == {"k": 1}

    def test_info_should_pass_through_row_infos_that_are_not_json(
        self, make_mv5
    ):
        """Test row metadata that was never JSON.

        Given:
            A multivec whose ``row_infos`` is a bare label rather than JSON.
        When:
            The tileset info is read.
        Then:
            It should carry the string unparsed. Older writers store bare
            labels, so a decode failure is a format variant rather than
            corruption, and refusing the file would drop a track that renders
            correctly today.
        """
        # Arrange
        path = make_mv5()
        with h5py.File(path, "r+") as f:
            f["info"].create_dataset("row_infos", data=np.bytes_("not json"))

        # Act
        with MultivecTileset(str(path)) as tileset:
            info = tileset.info().model_dump()

        # Assert
        assert info["row_infos"] == "not json"

class TestMultivecTilesetDeclarations:
    """What the class states about itself."""

    def test_datatype_should_be_multivec(self):
        """Test the declared datatype.

        Given:
            The tileset class.
        When:
            Its ``datatype`` is read.
        Then:
            It should be "multivec", the string the wire uses for this format.
        """
        # Act & assert
        assert MultivecTileset.datatype == "multivec"

    def test_ndim_should_be_one(self):
        """Test the declared coordinate arity.

        Given:
            The tileset class.
        When:
            Its ``ndim`` is read.
        Then:
            It should be 1, the number of positional slots its tile ids
            carry.
        """
        # Act & assert
        assert MultivecTileset.ndim == 1

    def test_tile_kind_should_be_dense(self):
        """Test the declared payload kind.

        Given:
            The tileset class.
        When:
            Its ``tile_kind`` is read.
        Then:
            It should be "dense", which selects the payload shape a client
            decodes.
        """
        # Act & assert
        assert MultivecTileset.tile_kind == "dense"

    def test_grid_policy_should_be_sequential(self):
        """Test how this tileset bounds a single tile.

        Given:
            The tileset class.
        When:
            Its ``grid_policy`` is read.
        Then:
            It should be "sequential", describing a 1D dense multivec on a sequential grid. The row axis is carried in
            the payload's shape, not in ``ndim``, which counts tile-id positions.
        """
        # Act & assert
        assert MultivecTileset.grid_policy == "sequential"
    def test_grid_threshold_should_be_one_so_drift_is_floored(self):
        """Test the bin-drift threshold.

        Given:
            The declared grid threshold.
        When:
            It is read.
        Then:
            It should be 1.0, matching bigWig and what the legacy module
            intends. The legacy accumulator never decrements after a drop, so
            its drift only grows and it over-corrects from every later
            chromosome -- damage hidden by a hard length clamp.
        """
        # Act & assert
        assert MultivecTileset.grid_threshold == 1.0

    def test_modifiers_should_be_absent(self):
        """Test that the format declares no modifier slot.

        Given:
            The declared modifiers.
        When:
            They are read.
        Then:
            It should be ``None``, so any modifier is rejected at parse time
            rather than silently ignored. ``None`` is a stronger statement
            than an empty spec: ``TileId.parse`` refuses a modifier outright
            against ``None`` and validates one against a spec.
        """
        # Act & assert
        assert MultivecTileset.modifiers is None

    def test_options_should_be_empty(self):
        """Test that the format recognizes no tile-id options.

        Given:
            The declared option keys.
        When:
            They are read.
        Then:
            It should be an empty set, so an unrecognized ``,key:value`` is
            rejected at parse time.
        """
        # Act & assert
        assert MultivecTileset.options == frozenset()

class TestMultivecTilesetInfo:
    """The tileset info, read out of the stored file."""

    def test_resolutions_should_be_ascending(self, tileset):
        """Test the ladder order.

        Given:
            A multivec whose resolution groups are named in arbitrary order.
        When:
            The resolutions are read.
        Then:
            They should come back ascending, matching cooler and the explicit
            ladder contract. ``z`` still indexes coarsest-first.
        """
        # Act & assert
        assert tileset.resolutions == (4, 8, 16)

    def test_info_should_declare_an_explicit_ladder_over_the_genome(
        self, tileset
    ):
        """Test the ladder form and extent.

        Given:
            A multivec over the canonical genome.
        When:
            The info is read.
        Then:
            It should be explicit, run to the genome length rather than a
            quadtree extent, and carry no ``max_width``.
        """
        # Act
        info = tileset.info()

        # Assert
        assert info.ladder is Ladder.EXPLICIT
        assert info.resolutions == [4, 8, 16]
        assert info.min_pos == [0]
        assert info.max_pos == [genome.CANONICAL_TOTAL]
        assert info.max_width is None

    def test_info_should_read_the_tile_size_from_the_file(self, make_mv5):
        """Test that tile width is a property of the file, not a constant.

        Given:
            A multivec written with a non-default tile size.
        When:
            The info is read.
        Then:
            It should carry that size. Unlike bigWig and cooler, multivec
            stores its own, so hardcoding one would mis-tile any file that
            disagrees.
        """
        # Act
        with MultivecTileset(str(make_mv5(tile_size=64))) as tileset:
            info = tileset.info()

        # Assert
        assert info.tile_size == 64
        assert info.shape == [64, N_ROWS]

    def test_info_should_report_shape_as_tile_size_by_rows(self, tileset):
        """Test the transposition between the info and the payload.

        Given:
            The tileset info.
        When:
            ``shape`` is read.
        Then:
            It should be ``[tile_size, n_rows]`` even though the payload ships
            ``(n_rows, tile_size)``. That inversion is in the legacy module
            too; it is confusing and load-bearing, so it is preserved rather
            than tidied.
        """
        # Act & assert
        assert tileset.info().shape == [TILE_SIZE, N_ROWS]

    def test_canvas_should_size_the_extent_to_the_zoom(self, tileset):
        """Test that an explicit ladder derives its lattice per level.

        Given:
            A three-level ladder at 16, 8 and 4 bp per bin.
        When:
            Each level's canvas is built.
        Then:
            The tile counts should be 1, 2 and 3 -- however many 256-bin tiles
            cover the genome at that resolution.
        """
        # Act
        counts = [tileset.info().canvas(z).n_tiles for z in range(3)]

        # Assert
        assert counts == [1, 2, 3]

    def test_info_should_carry_row_infos_from_the_info_group(self, make_mv5):
        """Test the row labels in their primary location.

        Given:
            A multivec with ``row_infos`` stored under ``/info``.
        When:
            The tileset info is read.
        Then:
            It should carry the decoded labels, which are what the client names
            each track with.
        """
        # Act
        with MultivecTileset(str(make_mv5(row_infos=["a", "b"]))) as tileset:
            info = tileset.info().model_dump()

        # Assert
        assert info["row_infos"] == ["a", "b"]

    def test_info_should_fall_back_to_the_resolution_attributes(
        self, make_mv5
    ):
        """Test the older location the same labels are written to.

        Given:
            A multivec carrying ``row_infos`` only as an attribute on its
            coarsest resolution group.
        When:
            The tileset info is read.
        Then:
            It should find them there. The legacy module checks three locations
            with two encodings; dropping the fallback would blank the track
            labels on every file written by the older writer.
        """
        # Arrange
        path = make_mv5()
        with h5py.File(path, "r+") as f:
            f["resolutions/16"].attrs["row_infos"] = np.array(
                [b'"r0"', b'"r1"']
            )

        # Act
        with MultivecTileset(str(path)) as tileset:
            info = tileset.info().model_dump()

        # Assert
        assert info["row_infos"] == ["r0", "r1"]

    def test_info_should_carry_category_infos_when_present(self, make_mv5):
        """Test the second metadata field.

        Given:
            A multivec with ``category_infos`` under ``/info``.
        When:
            The tileset info is read.
        Then:
            It should be carried through decoded.
        """
        # Arrange
        path = make_mv5()
        with h5py.File(path, "r+") as f:
            f["info"].create_dataset(
                "category_infos", data=np.bytes_(json.dumps(["x", "y"]))
            )

        # Act
        with MultivecTileset(str(path)) as tileset:
            info = tileset.info().model_dump()

        # Assert
        assert info["category_infos"] == ["x", "y"]

    def test_info_should_omit_row_metadata_when_the_file_has_none(
        self, tileset
    ):
        """Test a file carrying no labels at all.

        Given:
            A multivec with neither metadata field anywhere.
        When:
            The info is read.
        Then:
            Neither field should be emitted, rather than appearing as null.

            Note that this cannot distinguish ``_row_metadata`` omitting the
            keys from its returning them as explicit ``None``: both reach
            ``TilesetInfo`` as the field default and both are stripped by
            ``exclude_none``. The wire payload is the contract and the two are
            the same payload, so the difference is not a behavior to pin. The
            emitting half is covered by
            ``test_info_should_carry_row_infos_from_the_info_group``.
        """
        # Act
        info = tileset.info().model_dump(exclude_none=True)

        # Assert
        assert "row_infos" not in info
        assert "category_infos" not in info

    def test_info_should_return_the_same_object_on_every_call(self, tileset):
        """Test that the info is built once.

        Given:
            A tileset whose info build reads several h5py groups.
        When:
            ``info`` is called twice.
        Then:
            It should return the identical object, preserving the cached
            coordinate system every tile depends on.
        """
        # Act & assert
        assert tileset.info() is tileset.info()

    def test_chromsizes_should_decode_the_stored_contig_names(self, tileset):
        """Test the coordinate system.

        Given:
            Contig names stored as fixed-width bytes, which is what h5py
            returns for an ``S32`` dataset.
        When:
            The chromsizes are read.
        Then:
            They should come back as text in the stored order.
        """
        # Act & assert
        assert tileset.chromsizes().to_pairs() == [
            list(p) for p in genome.CANONICAL_CHROMSIZES
        ]

    def test_policy_should_be_the_one_supplied_at_construction(self, make_mv5):
        """Test that the tileset carries its caller's limits.

        Given:
            A tileset constructed with a non-default policy.
        When:
            Its policy is read.
        Then:
            It should be that policy.
        """
        # Arrange

        policy = DEFAULT_POLICY.with_(max_tile_width=1_000_000)

        # Act
        with MultivecTileset(str(make_mv5()), policy=policy) as tileset:
            # Assert
            assert tileset.policy is policy


class TestMultivecTilesetLifetime:
    """Handle ownership."""

    def test_file_should_return_the_same_handle_on_every_read(self, make_mv5):
        """Test the handle cache.

        Given:
            A freshly constructed tileset, which opens nothing.
        When:
            The file property is read twice.
        Then:
            It should hand back the same handle. Note this does not test
            laziness despite the property being lazy -- only reuse; a handle
            opened eagerly in ``__init__`` is equally reused.
        """
        # Arrange
        tileset = MultivecTileset(str(make_mv5()))

        # Act & assert
        assert tileset.file is tileset.file
        tileset.close()

    def test_close_should_be_safe_to_call_twice(self, make_mv5):
        """Test idempotent release.

        Given:
            A tileset that has opened its file.
        When:
            It is closed twice.
        Then:
            The second call should be a no-op.
        """
        # Arrange
        tileset = MultivecTileset(str(make_mv5()))
        handle = tileset.file

        # Act
        tileset.close()
        tileset.close()

        # Assert
        assert tileset.file is not handle

    def test___exit___should_release_the_handle(self, make_mv5):
        """Test context-manager use.

        Given:
            A tileset used as a context manager.
        When:
            The block exits.
        Then:
            The handle should be released.
        """
        # Act
        with MultivecTileset(str(make_mv5())) as tileset:
            handle = tileset.file

        # Assert
        assert tileset.file is not handle


class TestMultivecTilesetTiles:
    """Tile generation."""

    def test_tiles_should_return_one_entry_per_requested_id(self, tileset):
        """Test batch shape.

        Given:
            Three tile ids.
        When:
            They are generated in one call.
        Then:
            The result should pair each id with its payload, in request order.
        """
        # Arrange
        ids = [tileset.parse_tile_id(t) for t in ("x.2.0", "x.1.1", "x.0.0")]

        # Act
        result = tileset.tiles(ids)

        # Assert
        assert [tid for tid, _ in result] == ids

    def test_tiles_should_emit_a_shaped_payload_without_statistics(
        self, tileset
    ):
        """Test the payload shape, which differs from the other dense types.

        Given:
            Any tile.
        When:
            It is generated.
        Then:
            It should carry only ``dense``, ``dtype`` and ``shape``. bigWig and
            cooler additionally ship ``size``, ``min_value`` and ``max_value``;
            whether that divergence is client-visible or just drift is still
            open, so it is reproduced rather than normalized.
        """
        # Act
        ((_, payload),) = tileset.tiles([tileset.parse_tile_id("x.0.0")])

        # Assert
        assert sorted(payload) == ["dense", "dtype", "shape"]
        assert payload["shape"] == [N_ROWS, TILE_SIZE]

    @pytest.mark.parametrize("z,x", [(0, 0), (1, 0), (1, 1), (2, 0), (2, 2)])
    def test_tiles_should_always_be_rows_by_tile_size(self, tileset, z, x):
        """Test the invariant the reconciler exists to hold.

        Given:
            Any in-range tile position.
        When:
            The tile is generated.
        Then:
            It should be ``(n_rows, tile_size)`` regardless of how many
            chromosome boundaries it straddles.
        """
        # Act
        ((_, payload),) = tileset.tiles([tileset.parse_tile_id(f"x.{z}.{x}")])

        # Assert
        assert stack(payload).shape == (N_ROWS, TILE_SIZE)

    def test_tiles_should_concatenate_the_chromosome_bands_in_order(
        self, tileset
    ):
        """Test where each bin in a straddling tile came from.

        Given:
            The first tile at the deepest zoom, spanning 1024 bp at 4 bp per
            bin over a genome whose first contig is 1000 bp.
        When:
            It is generated.
        Then:
            Its first 250 bins should be c1's whole stored band and its last 6
            should be c2's first bins, restarting from that contig's own
            numbering. The fixture stores each value as its own address, so a
            reconciler dropping from the wrong end shows up here and nowhere
            in the bin count.
        """
        # Act
        ((_, payload),) = tileset.tiles([tileset.parse_tile_id("x.2.0")])

        # Assert
        row0 = stack(payload)[0]
        np.testing.assert_array_equal(row0[:250], np.arange(250) * N_ROWS)
        np.testing.assert_array_equal(row0[250:], np.arange(6) * N_ROWS)

    def test_tiles_should_stack_the_rows_in_stored_order(self, tileset):
        """Test the transposition into the payload.

        Given:
            A tile over bins whose row values run consecutively.
        When:
            It is generated.
        Then:
            Row ``r`` should be the stored band offset by ``r``, so the client
            reads a stack of tracks rather than a run of vectors.
        """
        # Act
        ((_, payload),) = tileset.tiles([tileset.parse_tile_id("x.2.0")])

        # Assert
        matrix = stack(payload)
        for r in range(N_ROWS):
            np.testing.assert_array_equal(
                matrix[r, :250], np.arange(250) * N_ROWS + r
            )

    def test_tiles_should_pad_past_the_genome_with_zeros(self, tileset):
        """Test the overhanging tile.

        Given:
            The deepest-zoom tile that starts inside the genome and ends past
            it -- at 4 bp per bin, tile 2 begins at 2048 and the genome ends at
            3000, leaving 18 of its 256 bins unbacked.
        When:
            It is generated.
        Then:
            Exactly those 18 trailing bins should be zero across every row, no
            value should be NaN, and the bins before them should carry the
            file's data. Without that last assertion the test holds for a
            reader that zeroes every bin it touches, in bounds or not, since
            an all-zero tile satisfies the padding claim everywhere.
        """
        # Arrange
        z, x = genome.overhanging_tile(tileset.info())
        canvas = tileset.info().canvas(z)
        real = -(
            -(genome.CANONICAL_TOTAL - canvas.tile_span(x)[0])
            // int(canvas.binsize)
        )

        # Act
        ((_, payload),) = tileset.tiles([tileset.parse_tile_id(f"x.{z}.{x}")])

        # Assert
        matrix = stack(payload)
        assert np.all(matrix[:, real:] == 0)
        assert not np.isnan(matrix).any()
        assert matrix.shape[1] - real == 18
        assert matrix[:, :real].any()

    # --- properties ---

    @PROPERTY_IO
    @given(
        resolutions=st.lists(
            st.integers(min_value=1, max_value=64),
            min_size=1,
            max_size=4,
            unique=True,
        ).map(sorted),
        data=st.data(),
    )
    def test_tiles_should_hold_the_shape_at_every_position(
        self, make_mv5, resolutions, data
    ):
        """Test the shape across the full ladder, including indivisible bins.

        Given:
            Any position in a ladder whose bin sizes do not divide the contig
            lengths, so the concatenated bands over-represent the genome and
            the reconciler must drop trailing bins to compensate.
        When:
            The tile is generated.
        Then:
            It should still be ``(n_rows, tile_size)`` with no NaN. This module
            asserts the bin count rather than clamping it, so an accounting
            error raises here instead of silently shifting the data a bin left
            the way the legacy module does.
        """
        # Arrange
        path = make_mv5(name="drift.mv5", resolutions=resolutions)
        with MultivecTileset(str(path)) as tileset:
            info = tileset.info()
            z = data.draw(
                st.integers(min_value=0, max_value=info.num_zoom_levels - 1)
            )
            x = data.draw(
                st.integers(min_value=0, max_value=info.canvas(z).n_tiles - 1)
            )

            # Act
            ((_, payload),) = tileset.tiles(
                [tileset.parse_tile_id(f"x.{z}.{x}")]
            )

        # Assert
        matrix = stack(payload)
        assert matrix.shape == (N_ROWS, TILE_SIZE)
        assert not np.isnan(matrix).any()


class TestMultivecTilesetBounds:
    """What happens off the end of the ladder.

    Pinned, not asserted as desirable. ``clodius/tiles/multivec.py`` has no
    established skip contract of its own -- only cooler does -- so these record
    what the rewrite does today.
    """

    def test_tiles_should_raise_when_the_zoom_is_past_the_ladder(self, tileset):
        """Test a zoom level above the last resolution.

        Given:
            A zoom one past the three-entry ladder.
        When:
            The tile is requested.
        Then:
            It should raise TileOutOfBounds.
        """
        # Act & assert
        with pytest.raises(TileOutOfBounds, match="exceeds ladder"):
            tileset.tiles([tileset.parse_tile_id("x.3.0")])

    def test_tiles_should_raise_when_the_position_is_past_the_canvas(
        self, tileset
    ):
        """Test a position beyond the tile count at its zoom.

        Given:
            Position 9 at the deepest zoom, where the canvas holds three tiles.
        When:
            The tile is requested.
        Then:
            It should raise TileOutOfBounds.
        """
        # Act & assert
        with pytest.raises(TileOutOfBounds, match="outside the 3 tiles"):
            tileset.tiles([tileset.parse_tile_id("x.2.9")])
