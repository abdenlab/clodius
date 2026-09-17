"""Build a small multi-resolution cooler (.mcool) on demand.

The repo has no usable mcool fixture -- ``data/hic-resolutions.cool`` is
truncated (7.2 MB on disk against a header claiming 43.3 MB) and will not open,
so the explicit-ladder cooler path has never been covered by a test.

Generating one rather than committing a binary keeps it out of git LFS and makes
the contents inspectable: the chromosome sizes are deliberately *not* multiples
of the bin size, so every chromosome ends on a short bin and the
chromosome-aligned grid genuinely disagrees with the genome-spanning one.
"""

from __future__ import annotations

import cooler
import numpy as np
import pandas as pd

# Deliberately not multiples of any resolution, so the last bin of each
# chromosome is short -- the condition that makes grid reconciliation non-trivial.
CHROMSIZES = {"chr1": 1_000_500, "chr2": 640_300, "chr3": 350_700}
BASE_RESOLUTION = 1_000

# Sized so the coarsest level puts the whole ~1.99 Mb genome in roughly one
# 256-bin tile and each finer level doubles the tile count: 1, 2, 4, 8 tiles
# across. A ladder coarser than this makes every tile mostly out-of-genome
# padding, which exercises nothing.
RESOLUTIONS = [1_000, 2_000, 4_000, 8_000]


def build_mcool(
    path: str, seed: int = 0, symmetric_upper: bool = True
) -> str:
    """Write a small .mcool to ``path`` and return it.

    ``symmetric_upper`` of False mirrors the pixel table and writes the whole
    matrix, which is what sets ``storage-mode`` to ``square`` -- the condition
    under which a tileset tells the client not to mirror the tile it is given.
    """
    bins = cooler.binnify(pd.Series(CHROMSIZES), BASE_RESOLUTION)
    n = len(bins)

    rng = np.random.default_rng(seed)
    # Upper-triangle pixels with a distance-decay flavour, so tiles near the
    # diagonal are dense and far ones are sparse -- closer to real Hi-C than
    # uniform noise, and it makes scatter collisions visible.
    rows, cols, counts = [], [], []
    for i in range(n):
        for j in range(i, min(i + 12, n)):
            if rng.random() < 0.6:
                rows.append(i)
                cols.append(j)
                counts.append(int(rng.integers(1, 200) / (1 + j - i)))
    pixels = pd.DataFrame({"bin1_id": rows, "bin2_id": cols, "count": counts})

    if not symmetric_upper:
        lower = pixels[pixels.bin1_id != pixels.bin2_id].rename(
            columns={"bin1_id": "bin2_id", "bin2_id": "bin1_id"}
        )
        pixels = pd.concat([pixels, lower], ignore_index=True).sort_values(
            ["bin1_id", "bin2_id"], ignore_index=True
        )

    # The single-resolution base must live in its own file: zoomify_cooler
    # truncates its output, which it cannot do while holding the base open.
    base_path = path + ".base.cool"
    cooler.create_cooler(
        base_path, bins, pixels, ordered=True, symmetric_upper=symmetric_upper
    )

    cooler.zoomify_cooler(base_path, path, RESOLUTIONS, chunksize=100_000)

    # Balancing weights, so the `weight` transform has something to divide by.
    for res in RESOLUTIONS:
        clr = cooler.Cooler(f"{path}::/resolutions/{res}")
        weights, _ = cooler.balance_cooler(clr, ignore_diags=1, min_nnz=0)
        with clr.open("r+") as grp:
            if "weight" in grp["bins"]:
                del grp["bins"]["weight"]
            grp["bins"].create_dataset("weight", data=weights)

    return path
