"""Batched reads in the v2 cooler tileset."""

import pytest

from clodius.core.coords import Chromsizes
from clodius.tiles_v2 import cooler as ctiles
from test.core.mcool_fixture import build_mcool


@pytest.fixture(scope="module")
def mcool(tmp_path_factory):
    return build_mcool(str(tmp_path_factory.mktemp("cooler") / "test.mcool"))


class Reordered(ctiles.CoolerTileset):
    """Serves the cooler's chromosomes in reverse order."""

    def chromsizes(self) -> Chromsizes:
        cs = super().chromsizes()
        return Chromsizes(
            tuple(reversed(cs.names)), tuple(reversed(cs.lengths))
        )


@pytest.fixture
def recorder():
    """A batched reader that logs the bin rectangles it reads, in ``log``."""
    log = []

    class Recording(ctiles.BatchedBlockReader):
        def _read(self, rspan, cspan):
            log.append((rspan, cspan))
            return super()._read(rspan, cspan)

    Recording.log = log
    return Recording


def tileset(path, reader=None, cls=ctiles.CoolerTileset, **kwargs):
    ts = cls(path, **kwargs)
    if reader is not None:
        ts.reader = reader
    return ts


def grid(ts, z, upper_only=False):
    n = ts.info().canvas(z).n_tiles
    return [
        ts.parse_tile_id(f"uid.{z}.{x}.{y}")
        for y in range(n)
        for x in range(y if upper_only else 0, n)
    ]


def payloads(ts, ids):
    out = ts.tiles(ids)
    assert [tid for tid, _ in out] == list(ids)
    return [p["dense"] for _, p in out]


def area(rects):
    return sum((r1 - r0) * (c1 - c0) for (r0, r1), (c0, c1) in rects)


@pytest.mark.parametrize("z", range(4))
@pytest.mark.parametrize("upper_only", [False, True])
@pytest.mark.parametrize("cls", [ctiles.CoolerTileset, Reordered])
def test_batching_does_not_change_tiles(mcool, z, upper_only, cls):
    ids = grid(cls(mcool), z, upper_only)
    assert payloads(cls(mcool), ids) == payloads(cls(mcool, batched=False), ids)


def test_balanced_tiles_agree_between_readers(mcool):
    ts = ctiles.CoolerTileset(mcool)
    ids = [ts.parse_tile_id(f"uid.2.{x}.0.weight") for x in range(4)]
    assert payloads(ts, ids) == payloads(
        ctiles.CoolerTileset(mcool, batched=False), ids
    )


def test_full_grid_is_one_read(mcool, recorder):
    ts = tileset(mcool, recorder)
    payloads(ts, grid(ts, 3))
    assert len(recorder.log) == 1


def test_batches_split_by_zoom_and_transform(mcool, recorder):
    ts = tileset(mcool, recorder)
    ids = [ts.parse_tile_id(i) for i in ("uid.1.0.0", "uid.1.1.0", "uid.2.0.0")]
    payloads(ts, ids)
    assert len(recorder.log) == 2
    recorder.log.clear()
    payloads(ts, ids + [ts.parse_tile_id("uid.1.0.0.weight")])
    assert len(recorder.log) == 3


def test_reads_are_not_merged_across_a_reordered_boundary(mcool, recorder):
    """Tiles either side of a chromosome boundary are adjacent on the canvas but
    not in the file once the chromosomes are reordered, so they cannot share a
    read."""
    ts = tileset(mcool, recorder, cls=Reordered)
    payloads(ts, grid(ts, 3))
    assert len(recorder.log) > 1


def test_batching_does_not_widen_the_area_read(mcool, recorder):
    """Batching the staircase a symmetric cooler serves must not fill it out to
    a rectangle, which would double the dense area materialized."""

    class NoPrefetch(recorder):
        def prefetch(self, positions):
            pass

    ts = tileset(mcool, recorder)
    ids = grid(ts, 3, upper_only=True)
    payloads(ts, ids)
    batched = area(recorder.log)

    recorder.log.clear()
    payloads(tileset(mcool, NoPrefetch), ids)
    assert batched <= area(recorder.log)
