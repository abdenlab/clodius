"""The tile-serving interface layer.

This package is the import surface. Everything a tileset implementation needs
is re-exported here, so adding a name to a submodule and forgetting to export
it is a visible omission rather than a silent one -- ``clodius.tiles_v2``
imports from ``clodius.core``, never from ``clodius.core.<submodule>``.
"""

from clodius.core.coords import (
    Canvas,
    Chromsizes,
    GenomicRange,
    natsorted,
    reconcile_sequential,
    reconcile_sequential_2d,
)
from clodius.core.errors import (
    MalformedTileId,
    TileError,
    TileOutOfBounds,
    TilesetUnavailable,
    UnsupportedModifier,
    UnsupportedOption,
)
from clodius.core.payloads import (
    PAYLOAD_TYPES,
    Bedlike2DTile,
    BedlikeTile,
    ColumnarTile,
    DenseTile,
    DenseTileMinMaxPayload,
    DenseTilePayload,
    DenseTileShapedPayload,
    ErrorTile,
    GeneModelTile,
    ImageTile,
    RegionRow,
    SequenceTile,
    SubsTile,
    TileKind,
)
from clodius.core.policies import (
    DEFAULT_POLICY,
    DensityPolicy,
    GridPolicy,
    TilePolicy,
    stable_importance,
    take_most_important,
)
from clodius.core.tileid import ModifierSpec, TileId
from clodius.core.tileset import (
    BaseTileset,
    Dataset,
    DatasetInfo,
    Ladder,
    LimitsDensity,
    ProvidesChromsizes,
    ProvidesRegions,
    ResamplesGrid,
    Tileset,
    TilesetInfo,
    quadtree_depth,
)

__all__ = [
    "DEFAULT_POLICY",
    "PAYLOAD_TYPES",
    "BaseTileset",
    "Bedlike2DTile",
    "BedlikeTile",
    "Canvas",
    "Chromsizes",
    "ColumnarTile",
    "Dataset",
    "DatasetInfo",
    "DenseTile",
    "DenseTileMinMaxPayload",
    "DenseTilePayload",
    "DenseTileShapedPayload",
    "DensityPolicy",
    "ErrorTile",
    "GeneModelTile",
    "GenomicRange",
    "GridPolicy",
    "ImageTile",
    "Ladder",
    "LimitsDensity",
    "MalformedTileId",
    "ModifierSpec",
    "ProvidesChromsizes",
    "ProvidesRegions",
    "RegionRow",
    "ResamplesGrid",
    "SequenceTile",
    "SubsTile",
    "TileError",
    "TileId",
    "TileKind",
    "TileOutOfBounds",
    "TilePolicy",
    "Tileset",
    "TilesetInfo",
    "TilesetUnavailable",
    "UnsupportedModifier",
    "UnsupportedOption",
    "natsorted",
    "quadtree_depth",
    "reconcile_sequential",
    "reconcile_sequential_2d",
    "stable_importance",
    "take_most_important",
]
