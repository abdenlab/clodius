"""Tests for clodius.tiles_v2.hic.

The second 2D tileset, and a deliberate transposition of the first: a ``.hic``
and an ``.mcool`` are both multi-resolution contact matrices on an explicit
ladder, so ``TilesetInfo``, ``Canvas`` and ``reconcile_sequential_2d`` carry
over untouched and this file mirrors ``test_cooler.py`` where the behavior is
the same. What it does not mirror is where the two formats genuinely part:

* The genome carries an ``All`` pseudo-chromosome that has to be excluded, or
  every canvas is sized against a 3002 bp genome instead of a 3000 bp one.
* The modifier vocabulary is closed, so a misspelled normalization is rejected
  at parse time rather than at fetch time.
* A lower-triangle query is refused by hictkpy rather than mirrored, so the
  reflection happens in ``fetch_block``.

The fixture is a three-resolution .hic at 1/2/4 bp over the canonical 3000 bp
genome, giving 12/6/3 tiles per zoom -- the same geometry ``build_mcool``
produces, and for the same reason: at a realistic binsize the whole genome
collapses into one bin of one tile.

``hictkpy.hic.FileWriter`` writes no normalization vectors, so no synthesized
fixture can carry one. Everything about normalization selection is therefore
tested through ``resolve_normalization`` and ``shared_normalizations``, which
take listings of names rather than an open file precisely so that it can be.
"""

import math

import hictkpy
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from clodius.core.coords import GenomicRange, bin_count
from clodius.core.errors import TileError, UnsupportedModifier
from clodius.core.policies import DEFAULT_POLICY
from clodius.core.tileset import Ladder, ResamplesGrid, Tileset
from clodius.tiles_v2 import hic as mod
from clodius.tiles_v2.hic import (
    NO_NORMALIZATION,
    TILE_SIZE,
    HicTileset,
    resolve_normalization,
    shared_normalizations,
)

from ..harness import genome
from ..harness.wire import square

# Property tests here read a real .hic per example. ``function_scoped_fixture``
# is suppressed for the same reason it is in ``test_cooler.py``: the file these
# draw against is built once per session, not state that leaks between
# examples.
PROPERTY_IO = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@pytest.fixture
def tileset(shared_hic):
    """A HicTileset over the session-scoped .hic.

    Function-scoped over a session-scoped file: the tileset caches a handle and
    an info, so sharing the *tileset* would let one test's ``close()`` break the
    next. The file itself is immutable and safe to share.
    """
    with HicTileset(str(shared_hic)) as ts:
        yield ts


@pytest.fixture
def hfile(shared_hic):
    """The 4 bp resolution of the shared .hic, opened from the path."""
    return hictkpy.MultiResFile(str(shared_hic))[4]


def bins_past_the_genome(info, z, x):
    """How many trailing bins of tile ``x`` at zoom ``z`` lie past the genome."""
    canvas = info.canvas(z)
    lo, hi = canvas.tile_span(x)
    total = info.coordinate_system.total_length
    if hi <= total:
        return 0
    return int(math.ceil((hi - max(lo, total)) / canvas.binsize))


def normalized_handle(mocker, listings, chromsizes=genome.CANONICAL_CHROMSIZES):
    """Patch ``hictkpy.MultiResFile`` with a stub carrying ``listings``.

    ``listings`` is one ``avail_normalizations`` result per resolution, given
    finest first, and the stub's ladder is sized to match.

    This is the only patched boundary in the ``tiles_v2`` suite, and it exists
    because the alternative is no coverage at all: ``hictkpy.hic.FileWriter``
    writes no normalization vectors, so a synthesized ``.hic`` always reports
    an empty listing and the transform-advertising path in ``_build_info`` can
    never be reached through a real file. hictkpy is an external I/O library,
    which is the boundary §11 of the guide permits patching at.

    The stub answers only the three members ``_build_info`` uses. That is the
    cost of the approach: a refactor that reaches for a fourth breaks this
    without breaking behavior.
    """
    resolutions = [2**i for i in range(len(listings))]

    class _Resolution:
        def __init__(self, names):
            self._names = list(names)

        def avail_normalizations(self):
            return list(self._names)

    class _Handle:
        def __init__(self):
            self._by_resolution = dict(zip(resolutions, listings))

        def resolutions(self):
            return list(resolutions)

        def chromosomes(self, include_ALL=False):
            return {name: int(length) for name, length in chromsizes}

        def __getitem__(self, resolution):
            return _Resolution(self._by_resolution[resolution])

    mocker.patch.object(mod.hictkpy, "MultiResFile", return_value=_Handle())
    return HicTileset("stubbed.hic")


class TestResolveNormalization:
    """Translating the transform modifier into a hictkpy normalization."""

    @pytest.mark.parametrize("transform", [None, "default"])
    def test_resolve_normalization_should_not_normalize_by_default(
        self, transform
    ):
        """Test the no-modifier case against a file that carries vectors.

        Given:
            A file offering four normalizations, and either no modifier or the
            ``default`` sentinel.
        When:
            The normalization is resolved.
        Then:
            It should be ``NONE``. This is where the format parts company with
            cooler, whose default elects the ``weight`` column when it exists:
            ``weight`` is the only candidate a cooler ever has, while a .hic
            typically carries several, so electing one silently would be a
            rendering decision the client did not make.
        """
        # Arrange
        available = ["VC", "VC_SQRT", "KR", "SCALE"]

        # Act
        result = resolve_normalization(available, transform)

        # Assert
        assert result == NO_NORMALIZATION

    def test_resolve_normalization_should_accept_none_from_an_empty_file(self):
        """Test the literal ``NONE`` against a file with no vectors at all.

        Given:
            A file offering nothing -- what every synthesized .hic looks like.
        When:
            ``NONE`` is requested explicitly.
        Then:
            It should be accepted. ``NONE`` is the absence of a weight vector
            rather than one of them, so it is never listed and a membership
            test against the listing would reject the one request that always
            works.
        """
        # Act
        result = resolve_normalization([], NO_NORMALIZATION)

        # Assert
        assert result == NO_NORMALIZATION

    def test_resolve_normalization_should_pass_through_a_stored_vector(self):
        """Test a named normalization the file carries.

        Given:
            A file offering ``KR`` among others.
        When:
            ``KR`` is requested.
        Then:
            It should come back unchanged, ready to hand to ``File.fetch``.
        """
        # Arrange
        available = ["VC", "KR", "SCALE"]

        # Act
        result = resolve_normalization(available, "KR")

        # Assert
        assert result == "KR"

    def test_resolve_normalization_should_raise_and_list_what_is_stored(self):
        """Test a normalization the file does not carry.

        Given:
            A file offering only ``VC``.
        When:
            ``KR`` is requested.
        Then:
            It should raise, naming what is available. Falling back to ``NONE``
            would serve raw counts under a normalized tile id, which renders as
            a plausible map rather than as an error.
        """
        # Arrange
        available = ["VC"]

        # Act & assert
        with pytest.raises(TileError) as excinfo:
            resolve_normalization(available, "KR")
        assert "'KR'" in str(excinfo.value)
        assert "['VC']" in str(excinfo.value)


class TestSharedNormalizations:
    """Which normalizations the info advertises, and in what order."""

    def test_shared_normalizations_should_drop_one_missing_at_any_resolution(
        self,
    ):
        """Test the intersection rule.

        Given:
            Three resolutions, of which only two carry ``SCALE``.
        When:
            The shared set is taken.
        Then:
            ``SCALE`` should be absent. A client may request a transform at any
            zoom, so a dropdown entry that fails on some levels is worse than
            an absent one.
        """
        # Arrange
        listings = [
            ["KR", "VC", "SCALE"],
            ["KR", "VC", "SCALE"],
            ["KR", "VC"],
        ]

        # Act
        result = shared_normalizations(listings)

        # Assert
        assert result == ["KR", "VC"]

    def test_shared_normalizations_should_follow_the_finest_resolutions_order(
        self,
    ):
        """Test the ordering rule.

        Given:
            Two resolutions listing the same names in opposite orders, the
            first of them in an order that is neither alphabetical nor its
            reverse.
        When:
            The shared set is taken.
        Then:
            It should follow the first listing. The legacy cooler module builds
            this list by iterating a set, whose order varies with the process
            hash seed, so its dropdown reorders between server restarts.

            The names are chosen so that the expected answer is not a sorted
            one: ``sorted`` would yield ``KR, SCALE, VC``, so an implementation
            that sorts instead of preserving the listing is observable here.
        """
        # Arrange
        listings = [["VC", "KR", "SCALE"], ["SCALE", "KR", "VC"]]

        # Act
        result = shared_normalizations(listings)

        # Assert
        assert result == ["VC", "KR", "SCALE"]

    def test_shared_normalizations_should_pass_a_single_listing_through(self):
        """Test the one-resolution ladder.

        Given:
            A single resolution listing two names.
        When:
            The shared set is taken.
        Then:
            It should be that listing unchanged. With one entry there is
            nothing to intersect against, so ``intersection`` is called with
            no arguments -- a path neither the multi-listing nor the empty
            case reaches.
        """
        # Act
        result = shared_normalizations([["KR", "VC"]])

        # Assert
        assert result == ["KR", "VC"]

    def test_shared_normalizations_should_return_empty_for_no_resolutions(
        self,
    ):
        """Test the degenerate input.

        Given:
            No listings at all.
        When:
            The shared set is taken.
        Then:
            It should be empty rather than raising. ``set.intersection`` over
            an empty sequence is a TypeError, which is the failure this guard
            exists to prevent.
        """
        # Act
        result = shared_normalizations([])

        # Assert
        assert result == []


class TestFetchBlock:
    """Block fetching, which owns the padding and reflection contracts."""

    def test_fetch_block_should_pad_with_nan_when_a_range_is_out_of_bounds(
        self,
    ):
        """Test the padding case past the last chromosome.

        Given:
            A row range with no chromosome name, and both ranges starting
            mid-bin.
        When:
            The block is fetched with a file that would raise if touched.
        Then:
            It should return a correctly shaped all-NaN block without reading.
            The reconciler needs the shape; the values are padding, not data.

            Both starts land inside a bin rather than on a boundary, which is
            what makes the shape depend on ``bin_count`` rounding *outward*.
            Ranges starting at 0 give the same answer under either rounding.
        """
        # Arrange
        row = GenomicRange(0, None, 50, 350)
        col = GenomicRange(0, "c1", 50, 250)

        # Act
        block = mod.fetch_block(None, row, col, 100.0, NO_NORMALIZATION)

        # Assert
        assert block.shape == (4, 3)
        assert np.all(np.isnan(block))

    def test_fetch_block_should_return_float32(self, hfile):
        """Test the block dtype.

        Given:
            An in-bounds pair of ranges.
        When:
            The block is fetched.
        Then:
            It should be float32, so a NaN-padded strip and a fully-real one
            concatenate without a dtype promotion. hictkpy returns int32
            counts by default, which cannot hold the NaN the padding path
            produces.
        """
        # Arrange
        row = col = GenomicRange(0, "c1", 0, 400)

        # Act
        block = mod.fetch_block(hfile, row, col, 4.0, NO_NORMALIZATION)

        # Assert
        assert block.dtype == np.float32

    @pytest.mark.parametrize(
        "start,end",
        [(0, 400), (50, 150), (50, 54), (1, 2), (150, 999)],
        ids=[
            "aligned",
            "unaligned-start",
            "one-bin-wide-across-two",
            "sub-bin",
            "unaligned-both-ends",
        ],
    )
    def test_fetch_block_should_return_as_many_bins_as_bin_count_predicts(
        self, hfile, start, end
    ):
        """Test the premise the shared bin count rests on.

        Given:
            An in-bounds interval at the 4 bp resolution, unaligned at one or
            both ends.
        When:
            A block is fetched through the real reader.
        Then:
            Its row count should equal ``bin_count`` for that interval.

            ``bin_count``'s whole claim is that a fetcher returns every bin it
            *overlaps*, and every other test reaches it through the padding
            path, where no reader is involved at all. If hictkpy ever rounded
            differently, the reconciler would receive a block of the wrong
            shape and the failure would surface as a geometry error far from
            its cause. ``50-54`` is the discriminating case: exactly one bin
            wide, straddling a boundary, so it touches two.
        """
        # Arrange
        row = GenomicRange(0, "c1", start, end)
        col = GenomicRange(0, "c1", 0, 400)

        # Act
        block = mod.fetch_block(hfile, row, col, 4.0, NO_NORMALIZATION)

        # Assert
        assert block.shape[0] == bin_count(row, 4.0)

    @pytest.mark.pinned
    def test_fetch_block_should_rest_on_a_reader_that_refuses_lower_triangle(
        self, hfile
    ):
        """Test the upstream guarantee the reflection exists to work around.

        Given:
            A query pair whose row contig follows its column contig, issued
            directly to hictkpy rather than through ``fetch_block``.
        When:
            It is fetched.
        Then:
            It should raise. The reflection in ``fetch_block`` is only
            load-bearing while this holds: were hictkpy to start mirroring
            lower-triangle queries itself, the reflection would become dead
            code and every reflection test in this file would keep passing.
            Pinned because it is hictkpy's behavior, not ours.
        """
        # Act & assert
        with pytest.raises(RuntimeError, match="lower-triangle"):
            hfile.fetch(
                "c2\t0\t400", "c1\t0\t400", query_type="BED"
            ).to_numpy(query_span="full")

    def test_fetch_block_should_reflect_a_pair_below_the_diagonal(self, hfile):
        """Test the reflection that a .hic requires and a cooler does not.

        Given:
            A cross-chromosome pair whose row contig follows its column contig,
            which places the block wholly in the lower triangle.
        When:
            It is fetched, and so is its mirror.
        Then:
            Each should be the other's transpose. hictkpy raises "overlaps with
            the lower-triangle of the matrix" rather than mirroring, so without
            the reflection this call does not return a wrong block -- it does
            not return one at all.
        """
        # Arrange
        first = GenomicRange(0, "c1", 0, 400)
        second = GenomicRange(1, "c2", 0, 400)

        # Act
        lower = mod.fetch_block(hfile, second, first, 4.0, NO_NORMALIZATION)
        upper = mod.fetch_block(hfile, first, second, 4.0, NO_NORMALIZATION)

        # Assert
        assert np.array_equal(lower, upper.T)

    def test_fetch_block_should_reflect_within_one_chromosome_too(self, hfile):
        """Test that the ordering test looks past the contig index.

        Given:
            Two ranges on the *same* contig, the row one starting later than
            the column one.
        When:
            It is fetched, and so is its mirror.
        Then:
            Each should be the other's transpose. Comparing contig indices
            alone would call this pair equal and let the raw query through,
            which raises -- the position has to enter the comparison.
        """
        # Arrange
        early = GenomicRange(0, "c1", 0, 400)
        late = GenomicRange(0, "c1", 400, 800)

        # Act
        lower = mod.fetch_block(hfile, late, early, 4.0, NO_NORMALIZATION)
        upper = mod.fetch_block(hfile, early, late, 4.0, NO_NORMALIZATION)

        # Assert
        assert np.array_equal(lower, upper.T)


class TestHicTilesetDeclarations:
    """What the class states about itself."""

    def test_datatype_should_be_matrix(self):
        """Test the declared datatype.

        Given:
            The tileset class.
        When:
            Its ``datatype`` is read.
        Then:
            It should be "matrix", the same string cooler declares -- a client
            consuming contact matrices does not branch on the container.
        """
        # Act & assert
        assert HicTileset.datatype == "matrix"

    def test_ndim_should_be_two(self):
        """Test the declared coordinate arity.

        Given:
            The tileset class.
        When:
            Its ``ndim`` is read.
        Then:
            It should be 2, the number of positional slots its tile ids carry.
        """
        # Act & assert
        assert HicTileset.ndim == 2

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
        assert HicTileset.tile_kind == "dense"

    def test_grid_policy_should_be_sequential(self):
        """Test how this tileset bounds a single tile.

        Given:
            The tileset class.
        When:
            Its ``grid_policy`` is read.
        Then:
            It should be "sequential", describing a 2D dense matrix laid on a
            sequential grid.
        """
        # Act & assert
        assert HicTileset.grid_policy == "sequential"

    def test_modifiers_should_close_the_transform_vocabulary(self):
        """Test that the modifier spec enumerates the juicer names.

        Given:
            The declared transform modifier.
        When:
            Its unknown-value policy and members are read.
        Then:
            It should reject unknowns and hold exactly the juicer vocabulary
            plus the ``default`` sentinel. This is the opposite of cooler's,
            and the difference is real rather than stylistic: a cooler's
            modifier names a bin table column, which is per-file and unbounded,
            while these are fixed by the .hic format.

            Asserted as equality against a spelled-out literal, not as a subset
            of the declared tuple. A subset assertion cannot fail when the
            vocabulary *shrinks*: dropping the six genome-wide and inter-chrom
            variants leaves such an assertion green while a legitimate
            ``GW_KR`` request starts being rejected at parse time.
        """
        # Act & assert
        assert HicTileset.modifiers.allow_unknown is False
        assert HicTileset.modifiers.default == "default"
        assert HicTileset.modifiers.values == frozenset(
            {
                "default",
                "NONE",
                "VC",
                "VC_SQRT",
                "KR",
                "SCALE",
                "GW_VC",
                "GW_KR",
                "GW_SCALE",
                "INTER_VC",
                "INTER_KR",
                "INTER_SCALE",
            }
        )

    def test_parse_tile_id_should_reject_a_misspelled_transform(self, tileset):
        """Test what the closed vocabulary buys at parse time.

        Given:
            A tile id carrying a transform that is not a juicer name.
        When:
            It is parsed.
        Then:
            It should raise before any file is touched. Cooler cannot do this
            -- an arbitrary string may name a real weight column there -- so
            the same request reaches its fetch path and fails later.
        """
        # Act & assert
        with pytest.raises(UnsupportedModifier, match="not a valid transform"):
            tileset.parse_tile_id("x.0.0.0.ICE")

    def test_parse_tile_id_should_reject_coolers_none_spelling(self, tileset):
        """Test the one spelling that differs between the two formats.

        Given:
            A tile id carrying cooler's ``None`` literal.
        When:
            It is parsed.
        Then:
            It should raise. A .hic spells the absence of normalization
            ``NONE``, and silently accepting the other spelling would make the
            two formats disagree about what a client may send while appearing
            to agree.
        """
        # Act & assert
        with pytest.raises(UnsupportedModifier, match="not a valid transform"):
            tileset.parse_tile_id("x.0.0.0.None")

    def test_parse_tile_id_should_accept_a_genome_wide_normalization(
        self, tileset
    ):
        """Test the half of the vocabulary a subset assertion cannot reach.

        Given:
            A tile id carrying ``GW_KR``.
        When:
            It is parsed.
        Then:
            It should parse to that modifier. The genome-wide and
            inter-chromosomal variants are part of the juicer vocabulary, and
            a file carrying one has to be requestable; rejecting them at parse
            time would be indistinguishable from a typo to the client.
        """
        # Act
        result = tileset.parse_tile_id("x.0.0.0.GW_KR")

        # Assert
        assert result.modifier == "GW_KR"

    def test_hic_tileset_should_satisfy_the_dense_tileset_protocols(
        self, tileset
    ):
        """Test the conformance claim, which is otherwise only asserted by eye.

        Given:
            A constructed tileset.
        When:
            It is checked against the protocols.
        Then:
            It should satisfy ``Tileset`` and ``ResamplesGrid``. Both are
            runtime-checkable and structural, so a member renamed or dropped
            makes this fail where nothing else would -- ``ResamplesGrid`` in
            particular is dispatched on by nothing today and would otherwise
            rot silently.
        """
        # Act & assert
        assert isinstance(tileset, Tileset)
        assert isinstance(tileset, ResamplesGrid)

    def test_options_should_be_empty(self):
        """Test that no tile-id options are accepted.

        Given:
            The declared options.
        When:
            They are read.
        Then:
            They should be empty, so an unrecognized ``,key:value`` is rejected
            at parse time rather than silently ignored.
        """
        # Act & assert
        assert HicTileset.options == frozenset()


class TestHicTilesetInfo:
    """The tileset info, built from the stored resolutions."""

    @pytest.mark.pinned
    def test_resolutions_should_be_ascending(self, make_hic):
        """Test the ladder order on a ladder that is not a quadtree.

        Given:
            A .hic over the wide, non-power-of-two ladder 2/10/100.
        When:
            Its resolutions are read back.
        Then:
            They should come back ascending numerically, matching what legacy
            cooler serializes.

            Pinned rather than asserted as our contract, because nothing here
            can distinguish the ``sorted`` call in ``HicTileset.resolutions``
            from its absence, and no arrangement can. The format forbids the
            counter-example: ``FileWriter`` coarsens from each resolution to
            the next and rejects a descending list at ``finalize`` with
            ``coarsening factor should be > 1``, and ``build_hic`` sorts the
            argument before the writer would ever see it. A ``.hic`` whose
            ladder is stored descending cannot be written, so ``sorted`` is
            belt-and-braces. What this does catch is that guarantee changing
            upstream.
        """
        # Arrange
        path = make_hic(name="wide-ladder.hic", resolutions=(2, 10, 100))

        # Act
        with HicTileset(str(path)) as tileset:
            resolutions = tileset.resolutions

        # Assert
        assert resolutions == (2, 10, 100)

    def test_resolutions_should_be_computed_once(self, tileset):
        """Test the resolution cache.

        Given:
            A tileset whose info build walks the resolutions repeatedly.
        When:
            The property is read twice.
        Then:
            It should return the identical tuple rather than re-reading the
            file index each time.
        """
        # Act & assert
        assert tileset.resolutions is tileset.resolutions

    def test_chromsizes_should_exclude_the_all_pseudo_chromosome(
        self, tileset, shared_hic
    ):
        """Test the entry a .hic carries and a cooler does not.

        Given:
            A .hic written without ``skip_all_vs_all_matrix``, so the file
            holds a genome-wide ``All`` entry alongside the three real contigs.
        When:
            The chromsizes are read.
        Then:
            They should be exactly the canonical genome, and the total should
            stay 3000. The pseudo-chromosome is 2 bp, so including it shifts
            every canvas extent and every tile boundary by an amount too small
            to look wrong and large enough to be wrong -- which is why the
            total is asserted rather than only the names.
        """
        # Arrange
        stored = hictkpy.MultiResFile(str(shared_hic)).chromosomes(
            include_ALL=True
        )

        # Act
        result = tileset.chromsizes()

        # Assert
        assert "All" in stored, (
            "the fixture no longer carries the entry this test excludes; "
            "it now passes for a file with nothing to exclude"
        )

        # Assert
        assert result.to_pairs() == [
            list(pair) for pair in genome.CANONICAL_CHROMSIZES
        ]
        assert result.total_length == genome.CANONICAL_TOTAL

    def test_info_should_declare_an_explicit_ladder(self, tileset):
        """Test the ladder form.

        Given:
            A .hic, which enumerates its resolutions.
        When:
            The info is read.
        Then:
            It should be explicit, and carry no ``max_width``: for an explicit
            ladder the extent belongs to the zoom level, since only a
            power-of-two ladder makes it invariant.
        """
        # Act
        info = tileset.info()

        # Assert
        assert info.ladder is Ladder.EXPLICIT
        assert info.resolutions == [1, 2, 4]
        assert info.max_width is None
        assert info.tile_size == TILE_SIZE

    def test_info_should_span_the_genome_on_both_axes(self, tileset):
        """Test the declared extent.

        Given:
            The 3000 bp canonical genome.
        When:
            The info is read.
        Then:
            Both axes should run to the genome length, and ``min_pos`` should
            be one-based on both -- which is what legacy cooler emits.
        """
        # Act
        info = tileset.info()

        # Assert
        assert info.min_pos == [1, 1]
        assert info.max_pos == [genome.CANONICAL_TOTAL] * 2

    def test_info_should_size_the_extent_to_the_zoom(self, tileset):
        """Test that the extent is derived per level rather than inherited.

        Given:
            The three-entry ladder.
        When:
            Each zoom's canvas is taken.
        Then:
            Each should hold just enough tiles to cover the genome at its own
            resolution -- 3, 6 and 12. A single tileset-level ``max_width``
            taken from the coarsest resolution would over-report at every
            finer zoom.
        """
        # Arrange
        info = tileset.info()

        # Act
        counts = [
            info.canvas(z).n_tiles for z in range(info.num_zoom_levels)
        ]

        # Assert
        assert counts == [3, 6, 12]

    def test_chromsizes_should_preserve_the_stored_contig_order(
        self, make_hic
    ):
        """Test the ordering a .hic reports, against a genome that can tell.

        Given:
            A .hic over the ordering genome, whose file order ``c2, c1, c10``
            is a different permutation from natural ``c1, c2, c10`` and from
            lexicographic ``c1, c10, c2``.
        When:
            Its chromsizes are read.
        Then:
            They should come back in file order. This matches cooler, which
            takes ``chromnames`` as stored, and deliberately differs from
            bigwig, which natural-sorts -- the coordinate system a tile is
            inverted against has to be the one the file was written with, or
            every position is off by a contig.

            The canonical genome cannot test this: ``c1/c2/c3`` in declaration
            order is a fixed point of all three orderings, which is how
            ``natsorted`` once survived being replaced by both ``sorted`` and
            ``list``.
        """
        # Arrange
        path = make_hic(
            name="shuffled.hic", chromsizes=genome.SHUFFLED_CHROMSIZES
        )

        # Act
        with HicTileset(str(path)) as tileset:
            names = tileset.chromsizes().names

        # Assert
        assert names == genome.SHUFFLED_FILE_ORDER
        assert names != genome.SHUFFLED_NATURAL_ORDER
        assert names != genome.SHUFFLED_LEXICOGRAPHIC_ORDER

    def test_info_should_serve_a_single_entry_ladder(self, make_hic):
        """Test the degenerate ladder.

        Given:
            A .hic holding one resolution.
        When:
            Its info is read and its only zoom requested.
        Then:
            It should declare one zoom level and serve a tile there, and skip
            the zoom above it. A one-element ladder is the shape that reaches
            ``shared_normalizations`` with a single listing and ``canvas`` with
            no coarser level to inherit an extent from.
        """
        # Arrange
        path = make_hic(name="flat.hic", resolutions=(4,))

        # Act
        with HicTileset(str(path)) as tileset:
            info = tileset.info()
            served = tileset.tiles([tileset.parse_tile_id("x.0.0.0")])
            past = tileset.tiles([tileset.parse_tile_id("x.1.0.0")])

        # Assert
        assert info.num_zoom_levels == 1
        assert square(served[0][1]).shape == (TILE_SIZE, TILE_SIZE)
        assert past == []

    def test_chromsizes_should_exclude_all_even_when_it_was_never_written(
        self, make_hic
    ):
        """Test the other writer mode.

        Given:
            A .hic written with ``skip_all_vs_all_matrix``, so there is no
            ``All`` entry to exclude in the first place.
        When:
            Its chromsizes are read.
        Then:
            They should be the canonical genome, exactly as for a file that
            does carry the entry. Paired with the exclusion test above, this is
            what shows ``include_ALL=False`` is unconditional rather than
            compensating for one writer setting.
        """
        # Arrange
        path = make_hic(name="no-all.hic", skip_all_vs_all_matrix=True)

        # Act
        with HicTileset(str(path)) as tileset:
            result = tileset.chromsizes()

        # Assert
        assert result.to_pairs() == [
            list(pair) for pair in genome.CANONICAL_CHROMSIZES
        ]

    def test_info_should_advertise_no_transforms_without_stored_vectors(
        self, tileset, shared_hic
    ):
        """Test the transform list for a file carrying no normalizations.

        Given:
            A synthesized .hic. ``hictkpy.hic.FileWriter`` stores pixels and
            leaves computing KR and VC to juicer, so the file has none.
        When:
            The info is read.
        Then:
            It should advertise none. The intersection and ordering rules that
            apply when a file *does* carry vectors are exercised against
            ``shared_normalizations`` and the stubbed handle instead, since no
            synthesizable fixture can reach them.
        """
        # Arrange
        # What this pins is a limitation of the writer, not a contract of the
        # tileset. Guarded so that a hictkpy that gains vector support fails
        # here with a reason rather than somewhere downstream with none.
        stored = hictkpy.MultiResFile(str(shared_hic))[4]
        assert stored.avail_normalizations() == [], (
            "the fixture now carries normalization vectors; this test no "
            "longer pins what its docstring claims"
        )

        # Act
        info = tileset.info()

        # Assert
        assert info.transforms == []

    def test_info_should_advertise_only_normalizations_stored_at_every_zoom(
        self, mocker
    ):
        """Test the intersection rule where it is actually wired up.

        Given:
            A handle whose three resolutions carry ``VC`` and ``KR``
            throughout, but ``SCALE`` at only two of them.
        When:
            The info is read.
        Then:
            It should advertise ``VC`` and ``KR`` as ``name``/``value`` pairs
            and drop ``SCALE``. A client may request a transform at any zoom,
            so a dropdown entry that fails on some levels is worse than an
            absent one.

            ``shared_normalizations`` is tested directly elsewhere; what this
            covers is that ``_build_info`` calls it at all and shapes the
            result into the two-key dicts the wire expects. Without it,
            hardcoding ``transforms = []`` passes the entire suite.
        """
        # Arrange
        tileset = normalized_handle(
            mocker,
            [["VC", "KR", "SCALE"], ["VC", "KR", "SCALE"], ["VC", "KR"]],
        )

        # Act
        info = tileset.info()

        # Assert
        assert info.transforms == [
            {"name": "VC", "value": "VC"},
            {"name": "KR", "value": "KR"},
        ]

    def test_info_should_order_transforms_by_the_finest_resolution(
        self, mocker
    ):
        """Test which resolution supplies the advertised order.

        Given:
            A handle whose finest resolution lists ``SCALE, VC, KR`` and whose
            coarser ones list the reverse.
        When:
            The info is read.
        Then:
            It should follow the finest resolution's order. Feeding the
            listings coarsest-first is a one-character change in
            ``_build_info`` that produces a plausible dropdown in a different
            order every time the ladder differs; the names are chosen so the
            answer is also not the sorted one.
        """
        # Arrange
        tileset = normalized_handle(
            mocker,
            [["SCALE", "VC", "KR"], ["KR", "VC", "SCALE"], ["KR", "VC", "SCALE"]],
        )

        # Act
        info = tileset.info()

        # Assert
        assert [t["value"] for t in info.transforms] == ["SCALE", "VC", "KR"]

    def test_info_should_omit_mirror_tiles(self, tileset):
        """Test that the client is not asked to mirror.

        Given:
            A .hic, which stores the upper triangle only.
        When:
            The info is read.
        Then:
            It should not set ``mirror_tiles``. Unlike cooler, where the flag
            distinguishes triangular from square storage, ``fetch_block`` has
            already reflected the lower half -- hictkpy refuses a
            lower-triangle query outright, so the reflection happens here or
            not at all, and there is nothing left for the client to do.
        """
        # Act & assert
        assert "mirror_tiles" not in tileset.info().model_dump()

    def test_info_should_return_the_same_object_on_every_call(self, tileset):
        """Test the info cache.

        Given:
            A tileset whose info build opens every resolution in turn.
        When:
            ``info()`` is called twice.
        Then:
            It should hand back the identical object rather than rebuilding.
        """
        # Act & assert
        assert tileset.info() is tileset.info()

    def test_policy_should_be_the_one_supplied_at_construction(
        self, shared_hic
    ):
        """Test that the policy is carried rather than re-derived.

        Given:
            A tileset constructed with an explicit policy.
        When:
            Its policy is read.
        Then:
            It should be that object. A dense tileset does not consult the
            policy today, so a constructor that dropped it would go unnoticed
            until the first caller passed a non-default one.
        """
        # Arrange
        policy = DEFAULT_POLICY

        # Act
        with HicTileset(str(shared_hic), policy=policy) as tileset:
            result = tileset.policy

        # Assert
        assert result is policy


class TestHicTilesetLifetime:
    """Opening and releasing the underlying file."""

    @pytest.mark.parametrize(
        "kind", ["missing", "wrong-format", "not-binary"]
    )
    def test_file_should_reject_a_path_that_is_not_a_hic(
        self, tmp_path, shared_mcool, kind
    ):
        """Test the three ways a path can fail to be a .hic.

        Given:
            A path that does not exist, an .mcool, and a text file named .hic.
        When:
            The tileset opens it.
        Then:
            It should raise ``ValueError`` naming the path. hictkpy reports all
            three identically, as a bare ``RuntimeError`` out of the C++ layer,
            which a caller can neither dispatch on nor safely catch --
            ``RuntimeError`` is what anything at all raises.

            ``ValueError`` is the spelling ``CoolerTileset`` uses for a
            readable file of the wrong shape. The two matrix tilesets do not
            agree completely even after this: cooler still lets h5py's
            ``OSError`` through for a path it cannot open at all, and
            ``tiles_v2/bed.py`` raises ``TilesetUnavailable``. Which one
            belongs in ``core`` is open.
        """
        # Arrange
        paths = {
            "missing": tmp_path / "absent.hic",
            "wrong-format": shared_mcool,
            "not-binary": tmp_path / "prose.hic",
        }
        paths["not-binary"].write_text("this is not a contact matrix\n")

        # Act & assert
        with pytest.raises(ValueError, match="not a readable .hic"):
            HicTileset(str(paths[kind])).info()

    def test_file_should_open_nothing_at_construction(self, tmp_path):
        """Test that the handle is lazy, not merely reused.

        Given:
            A path that does not exist.
        When:
            A tileset is constructed over it.
        Then:
            Construction should succeed and only the first read should raise.
            A server builds a tileset per registered dataset at import time; an
            eager open would make one unreadable file take down startup rather
            than one request.
        """
        # Act
        tileset = HicTileset(str(tmp_path / "absent.hic"))

        # Assert
        with pytest.raises(ValueError):
            tileset.file

    def test_file_should_return_the_same_handle_on_every_read(
        self, shared_hic
    ):
        """Test the handle cache.

        Given:
            A freshly constructed tileset, which opens nothing.
        When:
            The file property is read twice.
        Then:
            It should hand back the same handle. Note this does not test
            laziness despite the property being lazy -- only reuse.
        """
        # Arrange
        tileset = HicTileset(str(shared_hic))

        # Act & assert
        assert tileset.file is tileset.file
        tileset.close()

    def test_close_should_be_safe_to_call_twice(self, shared_hic):
        """Test idempotent release.

        Given:
            A tileset that has opened its file.
        When:
            It is closed twice.
        Then:
            The second call should be a no-op, and the next read should issue a
            fresh handle rather than the released one.
        """
        # Arrange
        tileset = HicTileset(str(shared_hic))
        handle = tileset.file

        # Act
        tileset.close()
        tileset.close()

        # Assert
        assert tileset.file is not handle

    def test___exit___should_release_the_handle(self, shared_hic):
        """Test context-manager use.

        Given:
            A tileset used as a context manager.
        When:
            The block exits.
        Then:
            The handle should be released.
        """
        # Act
        with HicTileset(str(shared_hic)) as tileset:
            handle = tileset.file

        # Assert
        assert tileset.file is not handle


class TestHicTilesetTiles:
    """Tile generation."""

    def test_tiles_should_return_one_entry_per_requested_id(self, tileset):
        """Test batch shape.

        Given:
            Three in-range tile ids.
        When:
            They are generated in one call.
        Then:
            The result should pair each id with its payload, in request order.
        """
        # Arrange
        ids = [
            tileset.parse_tile_id(t) for t in ("x.0.0.0", "x.1.1.0", "x.2.3.3")
        ]

        # Act
        result = tileset.tiles(ids)

        # Assert
        assert [tid for tid, _ in result] == ids

    @pytest.mark.parametrize(
        "z,x,y", [(0, 0, 0), (1, 1, 0), (2, 3, 3), (2, 11, 11)]
    )
    def test_tiles_should_always_be_a_square_of_tile_size(
        self, tileset, z, x, y
    ):
        """Test the invariant the 2D reconciler exists to hold.

        Given:
            Any in-range tile position, interior or overhanging.
        When:
            The tile is generated.
        Then:
            It should carry exactly ``tile_size`` squared values. The payload
            ships no ``shape`` field, so a tile of any other size is silently
            misread rather than rejected.
        """
        # Act
        ((_, payload),) = tileset.tiles(
            [tileset.parse_tile_id(f"x.{z}.{x}.{y}")]
        )

        # Assert
        assert square(payload).shape == (TILE_SIZE, TILE_SIZE)

    def test_tiles_should_pad_only_past_the_genome_end(self, tileset):
        """Test the overhanging tile at the corner of the matrix.

        Given:
            The deepest-zoom diagonal tile that starts inside the genome and
            ends past it -- at 1 bp per bin, tile 11 begins at 2816 and the
            genome ends at 3000.
        When:
            It is generated.
        Then:
            It should hold 184 by 184 real values with the rest NaN, so the
            padding boundary lands on the genome end rather than a tile edge.
        """
        # Arrange
        z, x = genome.overhanging_tile(tileset.info())
        real = genome.CANONICAL_TOTAL - x * TILE_SIZE

        # Act
        ((_, payload),) = tileset.tiles(
            [tileset.parse_tile_id(f"x.{z}.{x}.{x}")]
        )

        # Assert
        matrix = square(payload)
        assert not np.isnan(matrix[:real, :real]).any()
        assert np.isnan(matrix[real:, :]).all()
        assert np.isnan(matrix[:, real:]).all()

    def test_tiles_should_pad_each_axis_independently(self, tileset):
        """Test a strip where one axis overhangs and the other does not.

        Given:
            A tile whose row axis runs past the genome end while its column
            axis sits entirely inside it.
        When:
            It is generated.
        Then:
            Only the rows past the end should be NaN, and every column should
            carry a real value somewhere above them. Padding one axis is what
            the 2D reconciler's block assembly has to keep separable; leaking
            it into the other axis would blank a whole strip of real contacts.

            The column claim is asserted over the in-genome rows rather than
            the whole matrix. A fully-NaN column cannot occur here anyway --
            the rows past the end are legitimately NaN in every column -- so
            ruling one out asserts nothing.
        """
        # Arrange
        z, x = genome.overhanging_tile(tileset.info())
        real = genome.CANONICAL_TOTAL - x * TILE_SIZE

        # Act
        ((_, payload),) = tileset.tiles(
            [tileset.parse_tile_id(f"x.{z}.0.{x}")]
        )

        # Assert
        matrix = square(payload)
        assert np.isnan(matrix).all(axis=1).sum() == TILE_SIZE - real
        assert not np.isnan(matrix[:real, :]).all(axis=0).any()

    def test_tiles_should_transpose_across_the_diagonal(self, tileset):
        """Test that the two positions index the axes consistently.

        Given:
            An off-diagonal tile and its mirror.
        When:
            Both are generated.
        Then:
            One should be the transpose of the other. Swapping the row and
            column axes anywhere in the block assembly produces a matrix that
            is still square, still the right size, and reflected.
        """
        # Act
        ((_, upper),) = tileset.tiles([tileset.parse_tile_id("x.2.4.7")])
        ((_, lower),) = tileset.tiles([tileset.parse_tile_id("x.2.7.4")])

        # Assert
        assert np.array_equal(square(upper), square(lower).T, equal_nan=True)

    def test_tiles_should_reassemble_the_stored_matrix_at_the_coarsest_zoom(
        self, tileset, shared_hic
    ):
        """Test the whole grid against the file read in one call.

        Given:
            Every tile at the coarsest zoom, where three tiles per side cover
            the genome.
        When:
            They are laid back out as a single matrix.
        Then:
            Its real values should sum to the file's own total. Every
            per-tile assertion above holds for an implementation that drops a
            block, double-counts one, or reflects the wrong half; the total
            over the assembled grid does not.
        """
        # Arrange
        n = tileset.info().canvas(0).n_tiles
        # An unbounded fetch -- no range arguments -- is the one call shape
        # that segfaults hictkpy 1.4.0 when a file holds no interactions. Safe
        # here because the shared fixture has pixels, and not a shape the
        # tileset itself ever issues: `fetch_block` always passes both ranges.
        stored = (
            hictkpy.MultiResFile(str(shared_hic))[4]
            .fetch(normalization=NO_NORMALIZATION)
            .to_numpy(query_span="full")
        )

        # Act
        assembled = np.block(
            [
                [
                    square(tileset.tiles([tileset.parse_tile_id(
                        f"x.0.{col}.{row}"
                    )])[0][1])
                    for col in range(n)
                ]
                for row in range(n)
            ]
        )

        # Assert
        assert np.nansum(assembled) == stored.sum()

    def test_tiles_should_serve_raw_counts_for_the_default_modifier(
        self, tileset
    ):
        """Test that ``default`` and no modifier agree.

        Given:
            The same position spelled three ways: bare, with the ``default``
            sentinel, and with an explicit ``NONE``.
        When:
            All three are generated.
        Then:
            They should be identical. All three resolve to ``NONE``, so they
            name one tile, and a cache keyed on the tile id must not split
            them into three entries.
        """
        # Act
        ((_, bare),) = tileset.tiles([tileset.parse_tile_id("x.1.2.2")])
        ((_, default),) = tileset.tiles(
            [tileset.parse_tile_id("x.1.2.2.default")]
        )
        ((_, none),) = tileset.tiles([tileset.parse_tile_id("x.1.2.2.NONE")])

        # Assert
        assert np.array_equal(square(bare), square(default), equal_nan=True)
        assert np.array_equal(square(bare), square(none), equal_nan=True)

    def test_tiles_should_propagate_an_unstored_normalization(self, tileset):
        """Test that a bad transform is not swallowed with out-of-range tiles.

        Given:
            A valid position carrying a normalization the file does not store.
        When:
            The tile is requested.
        Then:
            It should raise rather than being dropped from the result. A client
            asking for a vector that is not there has made a different kind of
            mistake than one asking for a tile off the end of the genome, and
            only the second is skipped.
        """
        # Act & assert
        with pytest.raises(TileError, match="no 'KR' normalization"):
            tileset.tiles([tileset.parse_tile_id("x.0.0.0.KR")])

    def test_tiles_should_serve_a_genome_whose_contigs_end_on_tile_edges(
        self, make_hic
    ):
        """Test the arrangement that produces zero-length intervals.

        Given:
            A genome of 512/1024/512 bp, where contig boundaries fall exactly
            on tile boundaries, so inverting a tile position yields intervals
            of zero length at the seam.
        When:
            Every diagonal tile at every zoom is generated, and a mirrored
            off-diagonal pair.
        Then:
            Each should be a ``TILE_SIZE`` square, and the pair should still
            transpose.

            ``_tile`` filters zero-length intervals out before fetching, and
            that filter is load-bearing rather than an optimization: hictkpy
            rejects a zero-width query with ``query end position should be
            greater than the start position``. The whole ladder is swept
            because only *some* zero-length intervals reach the reader --
            those past the genome end are flagged out of bounds and answered
            with padding without a fetch, so they hide the case. The one that
            bites is an interior contig boundary, which on this genome first
            appears at zoom 1. The canonical genome produces neither.
        """
        # Arrange
        aligned = [["c1", 512], ["c2", 1024], ["c3", 512]]
        path = make_hic(name="aligned.hic", chromsizes=aligned)

        # Act
        with HicTileset(str(path)) as tileset:
            info = tileset.info()
            tiles = [
                square(
                    tileset.tiles(
                        [tileset.parse_tile_id(f"x.{z}.{x}.{x}")]
                    )[0][1]
                )
                for z in range(info.num_zoom_levels)
                for x in range(info.canvas(z).n_tiles)
            ]
            ((_, upper),) = tileset.tiles([tileset.parse_tile_id("x.1.0.2")])
            ((_, lower),) = tileset.tiles([tileset.parse_tile_id("x.1.2.0")])

        # Assert
        assert all(m.shape == (TILE_SIZE, TILE_SIZE) for m in tiles)
        assert np.array_equal(square(upper), square(lower).T, equal_nan=True)

    def test_tiles_should_serve_zeros_for_a_hic_with_no_interactions(
        self, make_hic
    ):
        """Test a contact matrix that is empty rather than sparse.

        Given:
            A .hic finalized with no pixels at all -- what a fully filtered or
            aborted conversion leaves behind.
        When:
            An interior tile is requested.
        Then:
            It should be a ``TILE_SIZE`` square of zeros, with no NaN inside
            the genome. Zero and NaN are different answers here: NaN means
            unmappable and a client renders it as absent, while zero means
            observed-and-empty. The random pixel draw can never produce this
            file, so nothing else in the suite reaches the case.
        """
        # Arrange
        path = make_hic(name="empty.hic", n_pixels=0)

        # Act
        with HicTileset(str(path)) as tileset:
            ((_, payload),) = tileset.tiles(
                [tileset.parse_tile_id("x.0.0.0")]
            )

        # Assert
        matrix = square(payload)
        assert matrix.shape == (TILE_SIZE, TILE_SIZE)
        assert not np.isnan(matrix).any()
        assert not matrix.any()

    # --- properties ---

    @PROPERTY_IO
    @given(data=st.data())
    def test_tiles_should_hold_the_matrix_shape_at_every_position(
        self, shared_hic, data
    ):
        """Test the shape across the full ladder.

        Given:
            Any zoom in the ladder and any pair of positions at that zoom. The
            file itself is fixed rather than drawn: a .hic costs ~175 ms to
            write, so a rebuild per example would dominate the suite.
        When:
            The tile is generated.
        Then:
            It should be a ``tile_size`` square whose cells past the genome end
            are NaN and whose cells inside it are *not* -- finite, and
            non-negative, these being contact counts. A shape error anywhere in
            the block assembly is invisible to the client, which infers the
            side from the length.

            The in-genome block is asserted explicitly because the rest of this
            property is vacuous without it: ``np.all`` over an empty selection
            is ``True``, so a ``fetch_block`` that returned padding for every
            range would satisfy the finiteness and sign claims trivially, and
            the two NaN claims do not fire at all on an interior tile.
        """
        # Arrange
        with HicTileset(str(shared_hic)) as tileset:
            info = tileset.info()
            z = data.draw(
                st.integers(min_value=0, max_value=info.num_zoom_levels - 1)
            )
            n_tiles = info.canvas(z).n_tiles
            x = data.draw(st.integers(min_value=0, max_value=n_tiles - 1))
            y = data.draw(st.integers(min_value=0, max_value=n_tiles - 1))

            # Act
            ((_, payload),) = tileset.tiles(
                [tileset.parse_tile_id(f"x.{z}.{x}.{y}")]
            )

        # Assert
        matrix = square(payload)
        assert matrix.shape == (TILE_SIZE, TILE_SIZE)
        real = matrix[~np.isnan(matrix)]
        assert np.all(np.isfinite(real))
        assert np.all(real >= 0)
        # The tile id spells the column first: ``uid.z.<column>.<row>``.
        cols = bins_past_the_genome(info, z, x)
        rows = bins_past_the_genome(info, z, y)
        if rows:
            assert np.all(np.isnan(matrix[-rows:, :]))
        if cols:
            assert np.all(np.isnan(matrix[:, -cols:]))
        assert not np.isnan(
            matrix[: TILE_SIZE - rows, : TILE_SIZE - cols]
        ).any()

    @PROPERTY_IO
    @given(data=st.data())
    def test_tiles_should_be_symmetric_at_every_position(
        self, shared_hic, data
    ):
        """Test transpose symmetry across the whole matrix.

        Given:
            Any off-diagonal tile position -- ``assume`` discards the diagonal,
            where a tile is trivially its own transpose and the property holds
            for an implementation that never mirrors anything.
        When:
            The tile and its mirror are generated.
        Then:
            Each should be the other's transpose. The stored matrix is an upper
            triangle and hictkpy refuses to read the lower one, so every tile
            below the diagonal is assembled from reflected blocks -- an axis
            swap survives every shape check and shows up only here.
        """
        # Arrange
        with HicTileset(str(shared_hic)) as tileset:
            info = tileset.info()
            z = data.draw(
                st.integers(min_value=0, max_value=info.num_zoom_levels - 1)
            )
            n_tiles = info.canvas(z).n_tiles
            x = data.draw(st.integers(min_value=0, max_value=n_tiles - 1))
            y = data.draw(st.integers(min_value=0, max_value=n_tiles - 1))
            assume(x != y)

            # Act
            ((_, upper),) = tileset.tiles(
                [tileset.parse_tile_id(f"x.{z}.{x}.{y}")]
            )
            ((_, lower),) = tileset.tiles(
                [tileset.parse_tile_id(f"x.{z}.{y}.{x}")]
            )

        # Assert
        assert np.array_equal(square(upper), square(lower).T, equal_nan=True)


class TestHicTilesetBounds:
    """The skip contract, chosen here rather than inherited.

    ``clodius/tiles/cooler.py`` skips a tile past the ladder rather than
    raising, and ``CoolerTileset`` carries that forward because it is the
    format's established behavior. ``.hic`` has no predecessor, so matching it
    is a decision: both are explicit-ladder matrix tilesets and a client should
    not have to branch on which one it is talking to.
    """

    def test_tiles_should_skip_a_zoom_past_the_ladder(self, tileset):
        """Test a zoom above the deepest resolution.

        Given:
            A zoom one past the last resolution.
        When:
            The tile is requested.
        Then:
            It should be omitted, leaving an empty result rather than raising.
        """
        # Act
        result = tileset.tiles([tileset.parse_tile_id("x.3.0.0")])

        # Assert
        assert result == []

    def test_tiles_should_skip_a_position_past_the_canvas(self, tileset):
        """Test the position filter.

        Given:
            A position beyond the tile count at its zoom.
        When:
            The tile is requested.
        Then:
            It should be omitted.
        """
        # Act
        result = tileset.tiles([tileset.parse_tile_id("x.0.9.0")])

        # Assert
        assert result == []

    def test_tiles_should_keep_the_siblings_of_a_skipped_tile(self, tileset):
        """Test batch integrity, which is the point of skipping over raising.

        Given:
            A batch mixing two in-range ids with an out-of-range zoom and an
            out-of-range position.
        When:
            It is generated.
        Then:
            The two valid tiles should still come back. Letting the error
            propagate discards every sibling in the request, which is the
            failure this guard exists to prevent.
        """
        # Arrange
        ids = [
            tileset.parse_tile_id(t)
            for t in ("x.0.0.0", "x.3.0.0", "x.0.9.0", "x.1.1.1")
        ]

        # Act
        result = tileset.tiles(ids)

        # Assert
        assert [str(tid) for tid, _ in result] == ["x.0.0.0", "x.1.1.1"]
