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

import os
import threading

import h5py
import pytest

from clodius.core.coords import Chromsizes
from clodius.core.errors import (
    TilesetError,
    TilesetUnavailable,
    UnsupportedOption,
)
from clodius.core.tileid import TileId
from clodius.tiles_v2.cooler import CoolerTileset

from ..core.mcool_fixture import build_mcool
from ..source_helpers import ladder, recording_factory


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


def test_parse_tile_id_should_reject_an_option(symmetric_mcool):
    """Test the empty option declaration on a shipped tileset.

    Given:
        A cooler tileset, which declares that it recognizes no ``,key:value``
        options at all, and a tile id carrying one.
    When:
        The id is parsed.
    Then:
        It should raise ``UnsupportedOption``. An empty declaration collapsing
        into "accept anything" is the kind of defect that only shows up on a
        real tileset, because every tileset inherits the empty default and
        none of them would notice.
    """
    # Arrange
    tileset = CoolerTileset(symmetric_mcool)

    # Act & assert
    with pytest.raises(UnsupportedOption, match="bogus"):
        tileset.parse_tile_id("u.0.0.0,bogus:1")


def test_tiles_should_raise_when_a_batch_option_is_passed(symmetric_mcool):
    """Test the refusal an overriding tiles() does not inherit.

    Given:
        A cooler tileset, which declares it reads no tile options, and a batch
        carrying one.
    When:
        The batch is served.
    Then:
        It should raise a ``TilesetError`` rather than serve. The protocol
        makes a malformed option a whole-batch failure, since options arrive
        once for every tile at once -- and this class overrides ``tiles()``,
        so it does not inherit the base's check and silently served tiles that
        ignored what was asked for. Distinct from the ``,key:value`` slot of a
        tile id, which is a different channel and already covered.
    """
    # Arrange
    tileset = CoolerTileset(symmetric_mcool)
    ids = [tileset.parse_tile_id("u.0.0.0")]

    # Act & assert
    with pytest.raises(TilesetError):
        tileset.tiles(ids, {"aggFunc": "mean"})


def test_tiles_should_serve_the_batch_when_no_option_is_passed(symmetric_mcool):
    """Test the control, so the refusal above cannot pass by refusing all.

    Given:
        The same tileset and no options.
    When:
        A batch is served.
    Then:
        It should return a payload for the tile.
    """
    # Arrange
    tileset = CoolerTileset(symmetric_mcool)
    ids = [tileset.parse_tile_id("u.0.0.0")]

    # Act
    served = tileset.tiles(ids)

    # Assert
    assert len(served) == 1
    assert "error" not in served[0][1]


def test_tiles_should_raise_when_the_file_cannot_be_opened():
    """Test that a whole-request failure is not demoted to a per-tile one.

    Given:
        A tileset pointed at a path that does not exist.
    When:
        a tile is served.
    Then:
        It should raise rather than returning an error payload. The per-tile
        catch exists for refusals that are genuinely about one tile; widening
        it to every exception would answer a batch of sixteen with sixteen
        cheerful error payloads and a 200, when the honest answer is that the
        dataset cannot be served at all.
    """
    # Arrange. Chromsizes are passed so that construction stays lazy: the
    # constructor reads them off the file when they are not given, and the
    # failure under test belongs to tiles(), not to __init__.
    tileset = CoolerTileset(
        "/nonexistent/none.mcool", chromsizes=Chromsizes(("chr1",), (1000,))
    )
    tid = TileId.parse("u.0.0.0", ndim=2)

    # Act & assert
    with pytest.raises(Exception) as excinfo:
        tileset.tiles([tid])
    assert not isinstance(excinfo.value, dict)
    assert isinstance(excinfo.value, (OSError, ValueError))


def test_tiles_should_return_an_error_payload_for_a_zoom_past_the_ladder(
    symmetric_mcool,
):
    """Test that a bad zoom refuses one tile rather than the request.

    Given:
        A batch holding one well-formed tile id and one naming a zoom above
        the resolution ladder, as a client with a stale tileset info sends.
    When:
        The batch is served.
    Then:
        It should return a dense payload for the good id and an error payload
        for the bad one. Tiles are batched by ``(zoom, transform)``, so a raise
        here loses the answers already computed for the *other* zoom groups in
        the same request, not only the neighbours of the offending id.
    """
    # Arrange
    tileset = CoolerTileset(symmetric_mcool)
    past_the_end = tileset.info().num_zoom_levels + 5
    ids = [
        tileset.parse_tile_id("u.0.0.0"),
        tileset.parse_tile_id(f"u.{past_the_end}.0.0"),
    ]

    # Act
    served = tileset.tiles(ids)

    # Assert
    assert "error" not in served[0][1]
    assert served[1][1]["error"]


def test_tiles_should_return_an_error_payload_for_an_unknown_transform(
    symmetric_mcool,
):
    """Test the second per-batch key, which fails the same way.

    Given:
        A batch holding one well-formed tile id and one naming a weight column
        the file does not carry. The modifier spec sets ``allow_unknown``, so
        any transform string reaches the resolver.
    When:
        The batch is served.
    Then:
        It should return a dense payload for the good id and an error payload
        for the bad one. Screening positions alone leaves this open: the
        transform is resolved before any position is looked at.
    """
    # Arrange
    tileset = CoolerTileset(symmetric_mcool)
    ids = [
        tileset.parse_tile_id("u.0.0.0"),
        tileset.parse_tile_id("u.0.0.0.nosuchweight"),
    ]

    # Act
    served = tileset.tiles(ids)

    # Assert
    assert "error" not in served[0][1]
    assert served[1][1]["error"]


def test_tiles_should_not_open_a_reader_when_nothing_in_the_batch_is_servable(
    symmetric_mcool,
):
    """Test that an all-refused batch touches the pixel store not at all.

    Given:
        A batch in which every position lies off the lattice, so the position
        screen empties it.
    When:
        The batch is served.
    Then:
        It should answer with error payloads without constructing a reader.
        The constructor is not cheap setup -- it opens the store and
        materializes the whole bin1 offset index, tens of megabytes on a
        finely binned genome -- and a raise inside it would discard the error
        payloads already recorded for the batch.
    """
    # Arrange. `reader` is the instance attribute the batch loop constructs
    # through, so wrapping it counts openings without reaching into internals.
    tileset = CoolerTileset(symmetric_mcool)
    opened = []
    build = tileset.reader

    def counting_reader(*args, **kwargs):
        opened.append(args)
        return build(*args, **kwargs)

    tileset.reader = counting_reader
    ids = [tileset.parse_tile_id(f"u.0.{x}.0") for x in (9999, 8888)]

    # Act
    served = tileset.tiles(ids)

    # Assert
    assert [payload["error"] for _, payload in served]
    assert opened == []


@pytest.mark.parametrize("modifier", ["", ".weight"])
def test_tiles_should_agree_between_a_path_and_a_file_like_source(
    symmetric_mcool, modifier
):
    """Test the two source shapes against each other across the whole ladder.

    Given:
        The same multires cooler addressed by path and by a callable that
        opens a fresh binary handle, served both unbalanced and balanced.
    When:
        Every tile in the ladder is requested from each.
    Then:
        The payloads should match tile for tile, and none should be an error.
        How a caller spelled the source is not part of the coordinate system,
        so a remote tileset serving even one tile differently would serve a
        different dataset under the same id. The balanced case is the one
        that matters most: it is the only path that hands the tileset's own
        ``h5py.File`` back to cooler to re-open, which is exactly where a
        file-like source could diverge from a path.
    """
    # Arrange
    compared = 0

    # Act & assert
    with (
        CoolerTileset(symmetric_mcool) as by_path,
        CoolerTileset(lambda: open(symmetric_mcool, "rb")) as by_handle,
    ):
        assert by_path.info() == by_handle.info()
        for tile_id in ladder(by_path):
            requested = tile_id + modifier
            (_, left), = by_path.tiles([by_path.parse_tile_id(requested)])
            (_, right), = by_handle.tiles(
                [by_handle.parse_tile_id(requested)]
            )
            assert left == right, requested
            assert "error" not in left, requested
            compared += 1
    assert compared


def test_tiles_should_reopen_the_source_when_the_tileset_was_closed(
    symmetric_mcool,
):
    """Test that a closed tileset is reusable rather than spent.

    Given:
        A factory-backed tileset that has served a tile and been closed.
    When:
        The same tile is requested again.
    Then:
        It should be served identically, from a second handle. ``close`` nulls
        the cached file so the property re-opens, which is what makes
        ``with`` followed by reuse work -- and a ``close`` that stopped
        nulling would break this silently.
    """
    # Arrange
    factory, opened = recording_factory(symmetric_mcool)
    tileset = CoolerTileset(factory)
    tile_id = tileset.parse_tile_id("u.0.0.0")
    (_, before), = tileset.tiles([tile_id])

    # Act
    tileset.close()
    (_, after), = tileset.tiles([tile_id])
    tileset.close()

    # Assert
    assert before == after
    assert len(opened) == 2
    assert all(handle.closed for handle in opened)


def test___init___should_refuse_an_already_open_file(symmetric_mcool):
    """Test that the one wrong shape fails at construction.

    Given:
        An open file rather than a callable that opens one.
    When:
        A tileset is constructed from it.
    Then:
        It should raise ``TypeError``. h5py would accept the handle, so
        nothing would fail here -- but the same argument handed to an oxbow
        tileset is a source that cannot be reopened, and refusing it
        uniformly is what keeps one accepted shape across every backend.
    """
    # Act & assert
    with open(symmetric_mcool, "rb") as handle:
        with pytest.raises(TypeError, match="not an already open file"):
            CoolerTileset(handle)


def test___init___should_close_the_handle_it_opened_when_the_file_is_rejected(
    fixture_path,
):
    """Test the ownership rule on the path where nothing is handed back.

    Given:
        A factory-backed tileset over a cooler with no ``resolutions`` group,
        which the constructor refuses.
    When:
        Construction fails.
    Then:
        The handle it opened should already be closed. The caller never
        receives an object, so there is nothing for them to close -- and a
        pooled remote handle is not released by falling out of scope the way
        a local file is.
    """
    # Arrange
    legacy = fixture_path("Dixon2012-J1-NcoI-R1-filtered.100kb.multires.cool")
    factory, opened = recording_factory(legacy)

    # Act
    with pytest.raises(TilesetUnavailable, match="no 'resolutions' group"):
        CoolerTileset(factory)

    # Assert
    assert opened and all(handle.closed for handle in opened)


def test_file_should_close_a_handle_it_could_not_open_as_hdf5(tmp_path):
    """Test that a failed open does not strand the handle behind it.

    Given:
        A factory-backed tileset over a file that is not HDF5, accessed
        several times.
    When:
        Each access fails.
    Then:
        Every handle opened should be closed. Storing the handle before the
        open succeeded meant each retry overwrote the only reference to the
        previous one, so the tileset leaked a handle per attempt while its
        own contract promised to close what it opened.
    """
    # Arrange
    path = str(tmp_path / "notan.mcool")
    (tmp_path / "notan.mcool").write_bytes(b"this is not HDF5")
    factory, opened = recording_factory(path)
    tileset = CoolerTileset(factory, chromsizes=Chromsizes(("c1",), (1000,)))

    # Act
    for _ in range(3):
        with pytest.raises(OSError):
            tileset.file

    # Assert
    assert len(opened) == 3
    assert all(handle.closed for handle in opened)


def test_file_should_raise_the_same_error_every_time_there_is_no_resolutions(
    fixture_path,
):
    """Test that a refusal stays a refusal instead of decaying.

    Given:
        A tileset over a cooler with no ``resolutions`` group, built with
        explicit chromsizes so the constructor does not read the file.
    When:
        Its info is requested repeatedly.
    Then:
        It should raise the same ``TilesetUnavailable`` every time. Caching
        the file before checking it left the bad file in place, so the refusal
        fired once and every later access raised a bare ``KeyError`` instead.
        Neither that nor the ``ValueError`` this used to raise is a
        ``TilesetError``, so a server answered every request with a 500; the
        refusal only became readable once the type moved into the hierarchy.
    """
    # Arrange
    legacy = fixture_path("Dixon2012-J1-NcoI-R1-filtered.100kb.multires.cool")
    tileset = CoolerTileset(legacy, chromsizes=Chromsizes(("c1",), (1000,)))

    # Act & assert
    for _ in range(3):
        with pytest.raises(TilesetUnavailable, match="no 'resolutions' group"):
            tileset.info()


def test_close_should_release_both_the_file_and_the_handle(symmetric_mcool):
    """Test the ownership rule for a caller-supplied factory.

    Given:
        A factory-backed tileset, used once so that it opens.
    When:
        It is closed.
    Then:
        Both the HDF5 file and the handle underneath it should be released.
        ``h5py.File.close()`` releases the HDF5 objects but not a Python file
        object it was handed, so closing only one of the two leaks a
        descriptor per tileset -- which on a server that caches tilesets is a
        leak proportional to the number of datasets served.
    """
    # Arrange
    factory, opened = recording_factory(symmetric_mcool)
    tileset = CoolerTileset(factory)
    hdf5 = tileset.file

    # Act
    tileset.close()

    # Assert
    assert not hdf5
    assert opened and all(handle.closed for handle in opened)


def test___repr___should_name_the_source_the_tileset_serves(symmetric_mcool):
    """Test that a tileset is identifiable in a traceback.

    Given:
        Tilesets built from a path and from a callable.
    When:
        Each is rendered.
    Then:
        Each should name its class and its source. A tileset was previously
        anonymous in a stack trace, which on a server handling many datasets
        left no way to tell which file a failure came from.
    """
    # Act & assert
    assert repr(CoolerTileset(symmetric_mcool)) == (
        f"<CoolerTileset {symmetric_mcool}>"
    )
    assert repr(CoolerTileset(lambda: open(symmetric_mcool, "rb"))) == (
        "<CoolerTileset <file-like source>>"
    )


def test_file_should_open_a_path_through_the_native_hdf5_driver(
    symmetric_mcool,
):
    """Test the claim that makes adopting the source abstraction safe.

    Given:
        A tileset constructed from a path, and the same file from a factory.
    When:
        Each opens its HDF5 file.
    Then:
        The path-backed one should use HDF5's native ``sec2`` driver and the
        factory-backed one the Python ``fileobj`` driver. Handing h5py a handle
        for a path forfeits HDF5's per-file handle sharing, so N tilesets over
        one file cost N descriptors instead of one -- and it changes the call
        for every caller who never asked for a file-like source.
    """
    # Arrange & act
    with (
        CoolerTileset(symmetric_mcool) as by_path,
        CoolerTileset(lambda: open(symmetric_mcool, "rb")) as by_handle,
    ):
        # Assert
        assert by_path.file.driver == "sec2"
        assert by_handle.file.driver == "fileobj"


def test_file_should_share_one_descriptor_across_path_backed_tilesets(
    symmetric_mcool,
):
    """Test the consequence the driver choice actually has on a server.

    Given:
        Many tilesets over the same path.
    When:
        Each opens its HDF5 file.
    Then:
        The process should hold one descriptor for the file, not one per
        tileset. HDF5 recognizes a file already open read-only and shares its
        file struct; the Python driver cannot, which is what exhausted the
        descriptor limit at a few hundred tilesets.
    """
    # Arrange
    count = len(os.listdir("/dev/fd"))

    # Act
    tilesets = [CoolerTileset(symmetric_mcool) for _ in range(20)]
    try:
        for tileset in tilesets:
            tileset.file
        # Assert
        assert len(os.listdir("/dev/fd")) - count == 1
    finally:
        for tileset in tilesets:
            tileset.close()


def test_file_should_call_the_factory_once_under_concurrent_first_access(
    symmetric_mcool,
):
    """Test the lazy open against the threading a tile server actually does.

    Given:
        A cold factory-backed tileset and several threads released together.
    When:
        They all touch the file for the first time.
    Then:
        It should call the factory once and expose one file. Without a lock
        every thread opens its own, the last writer wins, and the losers are
        handles ``close`` can no longer reach -- a caller's handle has no
        finalizer to fall back on.
    """
    # Arrange
    factory, opened = recording_factory(symmetric_mcool)
    tileset = CoolerTileset(factory)
    barrier = threading.Barrier(8)
    seen = []

    def touch():
        barrier.wait()
        seen.append(id(tileset.file))

    # Act
    threads = [threading.Thread(target=touch) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Assert
    assert len(set(seen)) == 1
    assert len(opened) == 1
    tileset.close()
    assert [handle.closed for handle in opened] == [True]


def test_close_should_release_the_handle_when_the_hdf5_close_raises(
    symmetric_mcool, monkeypatch
):
    """Test that a failing HDF5 close cannot strand the caller's handle.

    Given:
        An open factory-backed tileset whose ``h5py.File.close`` raises.
    When:
        It is closed.
    Then:
        It should still close the handle and clear both fields. Releasing them
        in sequence left ``_file`` set, so every retry raised at the same point
        and the handle was never reached again.
    """
    # Arrange
    factory, opened = recording_factory(symmetric_mcool)
    tileset = CoolerTileset(factory)
    tileset.file

    def boom(self):
        raise OSError("flush failed")

    monkeypatch.setattr(h5py.File, "close", boom)

    # Act
    with pytest.raises(OSError, match="flush failed"):
        tileset.close()

    # Assert
    assert [handle.closed for handle in opened] == [True]
