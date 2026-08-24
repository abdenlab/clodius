"""Synthetic fixture builders, one per tile format.

Every binary fixture under ``data/`` is an un-smudged git-LFS pointer on a
plain checkout, so nothing here reads one. Each builder writes a real, readable
file into a caller-supplied path and returns it.

These are plain functions rather than pytest fixtures on purpose: Hypothesis
evaluates ``@given`` at decoration time and cannot receive a fixture, so a test
that needs to build a file per example has to call the builder directly. The
per-package ``conftest.py`` files wrap them as fixtures for the ordinary case.

Three writer APIs are not what their documentation suggests, and each cost an
agent a debugging round:

* ``pybigtools.BigWigWriter(path)`` and ``BigBedWriter(path)`` raise
  ``TypeError: No constructor defined``. The entry point is
  ``pybigtools.open(path, "w")``, which dispatches on the file *extension* --
  so the path must end in ``.bw``/``.bigWig`` or ``.bb``/``.bigBed``. The
  returned object closes itself when ``write`` returns.
* bigBed entries must be pre-sorted by ``(chromosome order, start)`` or the
  write fails outright.
* ``cooler.create_cooler`` defaults to ``mode="w"``, which **truncates the
  whole file**. Writing several resolutions in a loop silently leaves only the
  last one, with no error at all.
"""

import numpy as np
import pytest

from .genome import CANONICAL_CHROMSIZES

DEFAULT_RECORDS = [
    ("c1", 10, 20, "a"),
    ("c1", 30, 40, "b"),
    ("c2", 0, 50, "c"),
    ("c2", 100, 150, "d"),
    ("c3", 5, 25, "e"),
]


def build_bed(path, records=DEFAULT_RECORDS):
    """A plain uncompressed BED holding ``records``."""
    path.write_text(
        "".join(f"{c}\t{s}\t{e}\t{name}\n" for c, s, e, name in records)
    )
    return path


def build_bed_bgzf(path, records=DEFAULT_RECORDS):
    """A BGZF-compressed BED with a sibling tabix index.

    ``path`` should end in ``.gz``; the ``.tbi`` lands beside it.
    """
    pysam = pytest.importorskip("pysam")
    plain = path.parent / f"{path.stem}.plain.bed"
    build_bed(plain, records)
    pysam.tabix_compress(str(plain), str(path), force=True)
    pysam.tabix_index(str(path), preset="bed", force=True)
    return path


def build_bigwig(path, chromsizes=CANONICAL_CHROMSIZES, values=None):
    """A bigWig covering every chromosome with one constant-value interval.

    ``values`` maps chromosome name to its constant signal; chromosomes absent
    from it get ``1.0``, ``2.0``, ... in declaration order. A distinct value per
    chromosome is what lets a one-bin shift at a boundary be read straight off
    the decoded array.

    A chromosome appears in the written file only if at least one interval
    names it, so every chromosome gets one whether or not the test cares.
    """
    pybigtools = pytest.importorskip("pybigtools")
    chroms = {name: int(length) for name, length in chromsizes}
    values = values or {}
    intervals = [
        (name, 0, int(length), float(values.get(name, i + 1)))
        for i, (name, length) in enumerate(chromsizes)
    ]
    writer = pybigtools.open(str(path), "w")
    writer.write(chroms, iter(intervals))
    return path


def build_bigbed(path, chromsizes=CANONICAL_CHROMSIZES, step=200, width=50):
    """A bigBed tiling every chromosome with fixed-width entries.

    Entries are emitted in ``(chromosome order, start)`` order, which the
    writer requires.
    """
    pybigtools = pytest.importorskip("pybigtools")
    chroms = {name: int(length) for name, length in chromsizes}
    entries = []
    for name, length in chromsizes:
        for start in range(0, int(length) - width, step):
            entries.append(
                (name, start, start + width, f"{name}_{start}\t500\t+")
            )
    writer = pybigtools.open(str(path), "w")
    writer.write(chroms, iter(entries))
    return path


def build_mcool(
    path,
    chromsizes=CANONICAL_CHROMSIZES,
    resolutions=(1, 2, 4),
    seed=0,
    weight=None,
    symmetric_upper=True,
):
    """A multi-resolution cooler with one group per entry in ``resolutions``.

    The first ``create_cooler`` call opens the file with ``mode="w"`` and every
    later one appends with ``mode="a"`` -- the default would truncate what the
    previous call just wrote.

    The default resolutions are deliberately tiny relative to the genome. A
    cooler tile is 256 bins wide, so a 1 kb binsize on a 3 kb genome puts the
    whole genome inside a single bin of a single tile -- one real value and
    65,535 padding cells, which exercises no geometry at all. At 1/2/4 the same
    genome spans 12/6/3 tiles, tiles straddle chromosome boundaries, and the
    last tile at every zoom overhangs the genome end.

    ``weight`` adds a constant ``weight`` bin column at every resolution, which
    is what makes the balancing modifiers reachable. Cooler *multiplies* by the
    two bins' weights, so a constant ``w`` scales every count by ``w**2`` --
    verified, not assumed: the ICE convention is a multiplier, not a divisor.

    ``symmetric_upper=False`` writes ``storage-mode: square`` instead of the
    usual upper triangle, which is the attribute the cooler tileset reads to
    decide whether the client must mirror across the diagonal. Cooler warns
    that it is disabling ``triucheck``; that is expected, not a misuse.
    """
    cooler = pytest.importorskip("cooler")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(seed)

    for i, binsize in enumerate(sorted(resolutions)):
        rows = []
        for name, length in chromsizes:
            starts = np.arange(0, int(length), binsize)
            ends = np.minimum(starts + binsize, int(length))
            rows.append(
                pd.DataFrame(
                    {"chrom": name, "start": starts, "end": ends}
                )
            )
        bins = pd.concat(rows, ignore_index=True)
        bins["chrom"] = pd.Categorical(
            bins["chrom"],
            categories=[name for name, _ in chromsizes],
            ordered=True,
        )
        if weight is not None:
            bins["weight"] = float(weight)

        n = len(bins)
        b1 = rng.integers(0, n, size=min(4 * n, 400))
        b2 = rng.integers(0, n, size=min(4 * n, 400))
        lo = np.minimum(b1, b2)
        hi = np.maximum(b1, b2)
        pixels = (
            pd.DataFrame({"bin1_id": lo, "bin2_id": hi, "count": 1})
            .groupby(["bin1_id", "bin2_id"], as_index=False)["count"]
            .sum()
            .sort_values(["bin1_id", "bin2_id"])
            .reset_index(drop=True)
        )

        cooler.create_cooler(
            f"{path}::/resolutions/{binsize}",
            bins,
            pixels,
            mode="w" if i == 0 else "a",
            ordered=True,
            symmetric_upper=symmetric_upper,
        )
    return path


def build_mv5(
    path,
    chromsizes=CANONICAL_CHROMSIZES,
    resolutions=(4, 8, 16),
    n_rows=4,
    tile_size=256,
    row_infos=None,
):
    """A multivec HDF5 file.

    Only three things are load-bearing for ``clodius.tiles_v2.multivec``: the
    ``tile-size`` attribute on ``/info``, the ``/chroms`` datasets, and
    ``/resolutions/<res>/values/<chrom>`` shaped ``(ceil(length / res),
    n_rows)``. The per-resolution ``chroms`` mirrors the legacy writer emits
    are not read, so they are omitted.

    The bin count must be exactly ``ceil(length / res)``; a short dataset makes
    the sequential reconciler raise rather than pad.
    """
    h5py = pytest.importorskip("h5py")
    import json

    names = [name for name, _ in chromsizes]
    lengths = [int(length) for _, length in chromsizes]

    with h5py.File(path, "w") as f:
        info = f.create_group("info")
        info.attrs["tile-size"] = tile_size
        if row_infos is not None:
            info.create_dataset(
                "row_infos", data=np.bytes_(json.dumps(row_infos))
            )

        chroms = f.create_group("chroms")
        chroms.create_dataset("name", data=np.array(names, dtype="S32"))
        chroms.create_dataset("length", data=np.array(lengths, dtype="i8"))

        for res in resolutions:
            values = f.create_group(f"resolutions/{res}/values")
            for name, length in zip(names, lengths):
                n_bins = -(-length // res)
                values.create_dataset(
                    name,
                    data=np.arange(n_bins * n_rows, dtype="f8").reshape(
                        n_bins, n_rows
                    ),
                )
    return path
