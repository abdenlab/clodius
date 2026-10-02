"""The HDF5 half of the file-backed tileset protocol.

`CoolerTileset` and `MultivecTileset` differ in how they read an HDF5 file and
agree completely on how they open and close one. That agreement lives in
`clodius.tiles_v2._backed.FileBacked`, which the BBI tilesets share; all that
is left here is the reader call and the ``h5py`` type it returns.
"""

from __future__ import annotations

from typing import IO

import h5py

from clodius.tiles_v2._backed import FileBacked


class H5Backed(FileBacked[h5py.File]):
    """A tileset whose bytes are one HDF5 file.

    Opens, owns and releases its source exactly as `FileBacked` describes --
    including the reason a path is handed to ``h5py`` as the path string,
    which is what keeps N tilesets over one file to one descriptor rather than
    N. Subclass `_validate` to refuse a file that is valid HDF5 but not this
    tileset's format, and `_configure` to read anything the constructor needs.
    """

    def _reader_open(self, target: str | IO[bytes]) -> h5py.File:
        return h5py.File(target, "r")
