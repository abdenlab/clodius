"""Tests for clodius.core.payloads.

This module is pure -- no file I/O, no git-LFS exposure -- so every fixture
here is an in-memory array.

Two traps worth knowing before editing. ``DenseTile`` is a frozen dataclass
whose ``values`` field is an ndarray, so the generated ``__eq__`` returns an
array and comparing two tiles raises ``ValueError`` for anything but a
one-element array: assert on ``to_dict()`` output or the individual properties,
never on tile equality. And the wire shapes' ``__required_keys__`` is only
trustworthy because this module has no ``from __future__ import annotations``;
see the warning in its docstring.
"""

import json
import math
import typing
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from pydantic import TypeAdapter, ValidationError

from clodius.core import payloads
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
from ..harness.wire import decode

F16_MAX = float(np.finfo("float16").max)

# One minimal producer dict per wire shape, carrying exactly its required keys.
# Driving the parametrized shape tests from one table means a shape added to the
# module without a table entry is caught by TestWireShapes::test_table.
WIRE_SHAPES = {
    DenseTilePayload: {
        "dense": "",
        "dtype": "float32",
        "size": 1,
        "min_value": 0.0,
        "max_value": 1.0,
    },
    DenseTileShapedPayload: {"dense": "", "dtype": "float32", "shape": [2, 3]},
    DenseTileMinMaxPayload: {
        "dense": "",
        "mins": "",
        "maxs": "",
        "dtype": "float32",
    },
    BedlikeTile: {
        "uid": "u",
        "xStart": 0,
        "xEnd": 10,
        "chrOffset": 0,
        "importance": 0.5,
        "fields": ["c1", "0", "10"],
    },
    Bedlike2DTile: {
        "uid": "u",
        "xStart": 0,
        "xEnd": 10,
        "chrOffset": 0,
        "importance": 0.5,
        "fields": ["c1", "0", "10"],
        "yStart": 0,
        "yEnd": 10,
    },
    GeneModelTile: {"genes": [], "transcripts": {}},
    ColumnarTile: {
        "id": [],
        "readName": [],
        "chrName": [],
        "chrOffset": [],
        "md": [],
        "cigars": [],
        "variants": [],
        "mapq": [],
        "strand": [],
        "is_paired": [],
        "first_seq": [],
        "last_seq": [],
    },
    SubsTile: {"id": "s", "substitutions": [], "color": 1},
    SequenceTile: {"sequence": "ACGT"},
    ImageTile: {"image": b"\x89PNG"},
    RegionRow: {"uid": "u", "xStart": 0, "xEnd": 10, "fields": ["c1"]},
    ErrorTile: {"error": "boom"},
}

SHAPE_IDS = [shape.__name__ for shape in WIRE_SHAPES]

# One case per (shape, required key), so a key that stops being required is
# reported by name instead of as one red test covering a whole shape.
REQUIRED_KEYS = [
    (shape, key) for shape, producer in WIRE_SHAPES.items() for key in producer
]
REQUIRED_KEY_IDS = [
    f"{shape.__name__}-{key}" for shape, key in REQUIRED_KEYS
]


def shapes_of(entry):
    """The wire shapes a ``PAYLOAD_TYPES`` entry admits.

    Two kinds map to a union rather than a single ``TypedDict`` -- ``dense``
    to the three dense payload spellings and ``image`` to an image or a dense
    tile -- so an entry is a tuple of shapes, not one shape.
    """
    return typing.get_args(entry) or (entry,)


class TestTileKind:
    """The vocabulary a tileset declares so a server knows what it will get."""

    def test_tile_kind_should_carry_its_wire_string(self):
        """Test that each kind compares equal to the string it serializes as.

        Given:
            The declared tile kinds.
        When:
            Each member is compared with its wire string.
        Then:
            It should compare equal and be an instance of str.
        """
        # Act & assert
        assert TileKind.DENSE == "dense"
        assert isinstance(TileKind.DENSE, str)

    @pytest.mark.parametrize("member", list(TileKind), ids=lambda m: m.value)
    def test_tile_kind_should_round_trip_from_its_wire_string(self, member):
        """Test that a wire value resolves back to a member.

        Given:
            The wire string of a declared kind.
        When:
            The enum is called with that value.
        Then:
            It should return the matching member.
        """
        # Act & assert
        assert TileKind(member.value) is member

    def test_tile_kind_should_declare_the_documented_vocabulary(self):
        """Test the exact set of declared kinds.

        Given:
            The tile kind enum.
        When:
            The set of member values is taken.
        Then:
            It should be exactly the nine documented kinds.
        """
        # Act
        values = {member.value for member in TileKind}

        # Assert
        assert values == {
            "dense",
            "bedlike",
            "bedlike_2d",
            "gene_models",
            "columnar",
            "subs",
            "sequence",
            "image",
            "points",
        }


class TestDenseTile:
    """Construction, derived geometry, and the dense wire encoding."""

    # --- construction -------------------------------------------------------

    def test___init___should_accept_a_flat_array_with_no_declared_size(self):
        """Test the ordinary one-value-per-bin construction.

        Given:
            A 1D float array and no declared size.
        When:
            A dense tile is constructed.
        Then:
            It should expose the array unchanged and a size of None.
        """
        # Arrange
        values = np.array([1.0, 2.0, 3.0])

        # Act
        tile = DenseTile(values)

        # Assert
        assert tile.size is None
        assert np.array_equal(tile.values, values)

    def test___init___should_accept_a_size_matching_the_second_axis(self):
        """Test a declared size agreeing with the array.

        Given:
            A (4, 2) array and a declared size of 2.
        When:
            A dense tile is constructed.
        Then:
            It should construct without complaint.
        """
        # Act
        tile = DenseTile(np.zeros((4, 2)), size=2)

        # Assert
        assert tile.size == 2

    @pytest.mark.parametrize(
        "values,size,actual",
        [
            (np.zeros((4, 2)), 4, 2),  # 2D: compared against shape[1]
            (np.zeros(4), 2, 1),  # 1D: compared against 1
            (np.zeros((2, 3, 4)), 3, 1),  # >2D: also compared against 1
            (np.zeros((4, 2)), 0, 2),  # zero never matches a real axis
        ],
        ids=["2d", "1d", "3d", "zero"],
    )
    def test___init___should_raise_when_the_declared_size_disagrees(
        self, values, size, actual
    ):
        """Test rejection of a size the array cannot support.

        Given:
            An array and a declared size that does not match its per-bin width.
        When:
            A dense tile is constructed.
        Then:
            It should raise ValueError naming both the declared and the actual
            width, so a client is never handed a tile it will misread.
        """
        # Act & assert
        with pytest.raises(ValueError) as excinfo:
            DenseTile(values, size=size)
        assert f"declared size {size}" in str(excinfo.value)
        assert f"{actual} value" in str(excinfo.value)

    @pytest.mark.parametrize(
        "values",
        [
            np.array([]),  # empty
            np.array([1, 2, 3], dtype="int32"),  # integer dtype
            np.zeros((2, 3, 4)),  # three dimensions
            np.array(5.0),  # zero-dimensional
        ],
        ids=["empty", "int-dtype", "3d", "0d"],
    )
    def test___init___should_accept_any_array_when_no_size_is_declared(
        self, values
    ):
        """Test that construction validates nothing but a size mismatch.

        Given:
            An array of an unusual dtype, rank or emptiness, and no declared
            size.
        When:
            A dense tile is constructed.
        Then:
            It should construct, since the only check performed is the declared
            size against the array's per-bin width.
        """
        # Act
        tile = DenseTile(values)

        # Assert
        assert tile.values is values

    # --- derived geometry ---------------------------------------------------

    @pytest.mark.parametrize(
        "values,expected",
        [
            (np.zeros(3), 1),  # 1D carries one value per bin
            (np.zeros((4, 2)), 2),  # 2D infers from the second axis
            (np.zeros((2, 3, 4)), 1),  # past 2D falls back to one
            (np.zeros((3, 0)), 0),  # a second axis of zero width
            (np.array(5.0), 1),  # zero-dimensional
        ],
        ids=["1d", "2d", "3d", "empty-axis", "0d"],
    )
    def test_values_per_bin_should_be_inferred_from_the_array_shape(
        self, values, expected
    ):
        """Test the inferred per-bin width.

        Given:
            An array of a given rank and shape, with no declared size.
        When:
            The per-bin width is read.
        Then:
            It should come from the second axis for a 2D array and be one
            otherwise.
        """
        # Act & assert
        assert DenseTile(values).values_per_bin == expected

    def test_values_per_bin_should_prefer_the_declared_size(self):
        """Test that a declared size wins over inference.

        Given:
            A (4, 2) array with a declared size of 2.
        When:
            The per-bin width is read.
        Then:
            It should return the declared size.
        """
        # Act & assert
        assert DenseTile(np.zeros((4, 2)), size=2).values_per_bin == 2

    @pytest.mark.parametrize(
        "values,expected",
        [
            (np.zeros(3), 3),
            (np.zeros((4, 2)), 4),  # the first axis, not the element count
            (np.array([]), 0),
            (np.zeros((3, 0)), 3),  # rows exist even with no elements
            (np.array(5.0), 0),  # a scalar array has no first axis
        ],
        ids=["1d", "2d", "empty", "empty-axis", "0d"],
    )
    def test_n_bins_should_be_the_length_of_the_first_axis(
        self, values, expected
    ):
        """Test the bin count.

        Given:
            An array of a given shape.
        When:
            The bin count is read.
        Then:
            It should be the length of the first axis, or zero for a scalar.
        """
        # Act & assert
        assert DenseTile(values).n_bins == expected

    # --- encoding: key sets -------------------------------------------------

    def test_to_dict_should_emit_the_canonical_key_set(self):
        """Test the default wire shape.

        Given:
            A tile built from a small NaN-free array.
        When:
            It is encoded with the default stats flag.
        Then:
            It should carry exactly dense, dtype, min_value, max_value and
            size, and no shape.
        """
        # Arrange
        tile = DenseTile(np.array([1.0, 2.0, 3.0]))

        # Act
        payload = tile.to_dict()

        # Assert
        assert set(payload) == {
            "dense",
            "dtype",
            "min_value",
            "max_value",
            "size",
        }

    def test_to_dict_should_omit_the_statistics_when_stats_is_false(self):
        """Test the shaped wire form multivec emits.

        Given:
            The same tile.
        When:
            It is encoded with stats disabled.
        Then:
            It should carry only dense and dtype.
        """
        # Arrange
        tile = DenseTile(np.array([1.0, 2.0, 3.0]))

        # Act
        payload = tile.to_dict(stats=False)

        # Assert
        assert set(payload) == {"dense", "dtype"}

    def test_to_dict_should_emit_shape_when_the_tile_declares_one(self):
        """Test the shape field multivec and fasta send.

        Given:
            A (2, 3) array constructed with a declared shape.
        When:
            It is encoded with stats disabled.
        Then:
            It should carry shape as a list of ints alongside dense and dtype.
        """
        # Arrange
        tile = DenseTile(np.zeros((2, 3)), shape=(2, 3))

        # Act
        payload = tile.to_dict(stats=False)

        # Assert
        assert set(payload) == {"dense", "dtype", "shape"}
        assert payload["shape"] == [2, 3]

    def test_to_dict_should_omit_shape_when_the_tile_declares_none(self):
        """Test that the shaped form can lack the key its type requires.

        Given:
            A tile built with no declared shape.
        When:
            It is encoded with stats disabled.
        Then:
            It should omit shape entirely, even though the shaped wire type
            declares that key as required -- the live producers all pass a
            shape, so the type is stricter than the method.
        """
        # Act
        payload = DenseTile(np.zeros(4)).to_dict(stats=False)

        # Assert
        assert "shape" not in payload

    def test_to_dict_should_reject_a_positional_stats_flag(self):
        """Test that the stats flag is keyword-only.

        Given:
            Any dense tile.
        When:
            It is encoded with the flag passed positionally.
        Then:
            It should raise TypeError.
        """
        # Arrange
        tile = DenseTile(np.zeros(4))

        # Act & assert
        with pytest.raises(TypeError):
            tile.to_dict(False)

    # --- encoding: values ---------------------------------------------------

    def test_to_dict_should_round_trip_the_values(self):
        """Test that the encoded buffer decodes back to the input.

        Given:
            A 1D array of values inside float16's range.
        When:
            It is encoded and the dense field decoded at the declared dtype.
        Then:
            It should hold the same values in the same order.
        """
        # Arrange
        values = np.array([1.0, 2.5, -3.0, 4.0])

        # Act
        payload = DenseTile(values).to_dict()

        # Assert
        assert np.array_equal(decode(payload), values)

    def test_to_dict_should_flatten_in_row_major_order(self):
        """Test the flattening convention for a 2D tile.

        Given:
            A (2, 2) array whose elements are distinguishable.
        When:
            It is encoded and decoded.
        Then:
            It should equal the row-major flattening, not the transpose.
        """
        # Arrange
        values = np.array([[1.0, 2.0], [3.0, 4.0]])

        # Act
        payload = DenseTile(values).to_dict()

        # Assert
        assert np.array_equal(decode(payload), [1.0, 2.0, 3.0, 4.0])

    # --- encoding: float width ----------------------------------------------

    def test_to_dict_should_narrow_to_float16_when_the_data_fits(self):
        """Test the narrow float path.

        Given:
            A NaN-free array well inside float16's range.
        When:
            It is encoded.
        Then:
            It should report float16, halving the bytes on the wire.
        """
        # Act & assert
        assert DenseTile(np.array([1.0, 2.0])).to_dict()["dtype"] == "float16"

    def test_to_dict_should_widen_to_float32_when_the_data_holds_nan(self):
        """Test that a gap forces the wider float.

        Given:
            An array carrying one NaN alongside finite values.
        When:
            It is encoded.
        Then:
            It should report float32, since float16 cannot carry the NaN that
            marks an uncovered bin.
        """
        # Arrange
        values = np.array([1.0, np.nan, 2.0])

        # Act & assert
        assert DenseTile(values).to_dict()["dtype"] == "float32"

    def test_to_dict_should_widen_to_float32_when_a_value_is_out_of_range(
        self,
    ):
        """Test the range check at float16's ceiling.

        Given:
            An array whose maximum is exactly float16's largest finite value.
        When:
            It is encoded.
        Then:
            It should report float32, the range check being strict at both
            ends rather than inclusive.
        """
        # Arrange
        values = np.array([0.0, F16_MAX])

        # Act & assert
        assert DenseTile(values).to_dict()["dtype"] == "float32"

    # --- encoding: statistics ------------------------------------------------

    def test_to_dict_should_report_statistics_over_the_finite_values(self):
        """Test that NaN is excluded from the reported bounds.

        Given:
            An array of two finite values and one NaN.
        When:
            It is encoded.
        Then:
            Its bounds should come from the finite values alone, so a gap does
            not drag the client's colour scale to NaN.
        """
        # Arrange
        values = np.array([2.0, np.nan, 5.0])

        # Act
        payload = DenseTile(values).to_dict()

        # Assert
        assert payload["min_value"] == 2.0
        assert payload["max_value"] == 5.0

    def test_to_dict_should_report_nan_bounds_as_a_string_when_all_nan(self):
        """Test the all-NaN tile, which every past-genome position produces.

        Given:
            An array whose every value is NaN.
        When:
            It is encoded with warnings escalated to errors.
        Then:
            It should report both bounds as the string "NaN" and emit no
            all-NaN RuntimeWarning.
        """
        # Arrange
        values = np.full(4, np.nan)

        # Act
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            payload = DenseTile(values).to_dict()

        # Assert
        assert payload["min_value"] == "NaN"
        assert payload["max_value"] == "NaN"

    def test_to_dict_should_encode_an_empty_array_as_an_empty_buffer(self):
        """Test the degenerate empty tile.

        Given:
            An empty 1D array.
        When:
            It is encoded.
        Then:
            It should return an empty dense field, float32, and both bounds as
            "NaN".
        """
        # Act
        payload = DenseTile(np.array([])).to_dict()

        # Assert
        assert payload["dense"] == ""
        assert payload["dtype"] == "float32"
        assert payload["min_value"] == payload["max_value"] == "NaN"

    def test_to_dict_should_encode_an_array_with_a_zero_width_axis(self):
        """Test a non-empty first axis carrying no elements.

        Given:
            A (3, 0) array -- three bins, no values.
        When:
            It is encoded.
        Then:
            It should return an empty buffer without raising. A check on the
            first axis rather than the element count would take the non-empty
            branch here and raise from the statistics reduction.
        """
        # Act
        payload = DenseTile(np.zeros((3, 0))).to_dict()

        # Assert
        assert payload["dense"] == ""
        assert payload["min_value"] == "NaN"
        assert payload["size"] == 0

    def test_to_dict_should_encode_a_zero_dimensional_array(self):
        """Test a scalar array, whose geometry disagrees with itself.

        Given:
            A zero-dimensional array holding one value.
        When:
            It is encoded.
        Then:
            It should encode that single value and report it as both bounds --
            even though the bin count reads zero, since a scalar has no first
            axis.
        """
        # Arrange
        tile = DenseTile(np.array(5.0))

        # Act
        payload = tile.to_dict()

        # Assert
        assert tile.n_bins == 0
        assert payload["size"] == 1
        assert payload["min_value"] == payload["max_value"] == 5.0

    def test_to_dict_should_leak_infinity_as_a_bare_float(self):
        """Test the asymmetry between NaN and infinity in the bounds.

        Given:
            An array containing positive infinity.
        When:
            It is encoded and the payload serialized as strict JSON.
        Then:
            It should raise ValueError, since infinity passes through as a
            bare float while NaN is stringified -- so a client parsing strict
            JSON gets a serialization failure rather than a tile.
        """
        # Arrange
        payload = DenseTile(np.array([1.0, np.inf])).to_dict()

        # Act & assert
        with pytest.raises(ValueError):
            json.dumps(payload, allow_nan=False)

    # --- properties ---------------------------------------------------------

    @given(
        values=hnp.arrays(
            dtype="float64",
            shape=hnp.array_shapes(min_dims=1, max_dims=2, max_side=8),
            elements=st.floats(
                min_value=-1000, max_value=1000, allow_nan=False
            ),
        )
    )
    @settings(max_examples=200)
    def test_to_dict_should_round_trip_any_finite_array(self, values):
        """Test the encoding round-trip over arbitrary finite arrays.

        Given:
            Any 1D or 2D float array whose values lie inside float16's range.
        When:
            It is encoded and the dense field decoded at the declared dtype.
        Then:
            It should yield exactly the row-major flattening of the input.
        """
        # Act
        payload = DenseTile(values).to_dict()
        decoded = decode(payload)

        # Assert
        assert decoded.size == values.size
        np.testing.assert_array_equal(
            decoded, values.ravel().astype(payload["dtype"])
        )

    @given(
        values=hnp.arrays(
            dtype="float64",
            shape=hnp.array_shapes(min_dims=1, max_dims=2, max_side=8),
            elements=st.floats(allow_infinity=False, width=32),
        ),
        stats=st.booleans(),
    )
    @settings(max_examples=200)
    def test_to_dict_should_emit_a_stable_key_set(self, values, stats):
        """Test that the wire key set depends only on the stats flag.

        Given:
            Any 1D or 2D float array, NaN permitted, and either stats setting.
        When:
            It is encoded.
        Then:
            It should carry exactly the keys that flag declares, and the whole
            payload should survive strict JSON serialization.
        """
        # Act
        payload = DenseTile(values).to_dict(stats=stats)

        # Assert
        expected = {"dense", "dtype"}
        if stats:
            expected |= {"min_value", "max_value", "size"}
        assert set(payload) == expected
        json.dumps(payload, allow_nan=False)

    @given(
        values=hnp.arrays(
            dtype="float64",
            shape=hnp.array_shapes(min_dims=1, max_dims=2, max_side=8),
            elements=st.floats(allow_infinity=False, width=32),
        )
    )
    @settings(max_examples=200)
    def test_to_dict_should_choose_float16_exactly_when_the_data_fits(
        self, values
    ):
        """Test the float-width rule over arbitrary arrays.

        Given:
            Any 1D or 2D float array, NaN permitted.
        When:
            It is encoded.
        Then:
            It should report float16 exactly when the array is NaN-free and
            both its bounds lie strictly inside float16's range, and float32
            otherwise.
        """
        # Arrange
        if values.size:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        else:
            lo = hi = math.nan
        fits = (
            not bool(np.isnan(values).any())
            and -F16_MAX < lo < F16_MAX
            and -F16_MAX < hi < F16_MAX
        )

        # Act
        dtype = DenseTile(values).to_dict()["dtype"]

        # Assert
        assert dtype == ("float16" if fits else "float32")

    @given(
        values=hnp.arrays(
            dtype="float64",
            shape=hnp.array_shapes(min_dims=1, max_dims=2, max_side=8),
            elements=st.floats(
                min_value=-100, max_value=100, allow_nan=False
            ),
        )
    )
    @settings(max_examples=200)
    def test_n_bins_should_multiply_with_values_per_bin_to_the_element_count(
        self, values
    ):
        """Test that the declared geometry accounts for every value.

        Given:
            Any non-empty 1D or 2D float array.
        When:
            Its bin count, per-bin width and encoded size are read.
        Then:
            The bin count times the per-bin width should equal the element
            count -- the relationship a client uses to unpack the buffer.
        """
        # Arrange
        tile = DenseTile(values)

        # Act
        payload = tile.to_dict()

        # Assert
        assert tile.n_bins * tile.values_per_bin == values.size
        assert payload["size"] == tile.values_per_bin


class TestPayloadTypes:
    """The catalogue keyed on a tileset's declared kind."""

    @pytest.mark.parametrize("kind", list(TileKind), ids=lambda k: k.value)
    def test_payload_types_should_declare_a_shape_for_every_kind(self, kind):
        """Test that no declared kind is missing from the catalogue.

        Given:
            Each member of the tile kind enum.
        When:
            It is looked up in the payload catalogue.
        Then:
            Every shape it admits should be one the module declares, so a new
            kind cannot ship without one. The lookup is already an assertion --
            a missing kind raises ``KeyError`` here -- so a comparison against
            ``None`` adds nothing; what it does not check is that the entry
            names shapes that exist.
        """
        # Act
        shapes = shapes_of(PAYLOAD_TYPES[kind])

        # Assert
        assert shapes
        assert set(shapes) <= set(WIRE_SHAPES)

    def test_payload_types_should_carry_no_entry_without_a_kind(self):
        """Test that the catalogue and the enum agree in both directions.

        Given:
            The catalogue and the tile kind enum.
        When:
            Their key sets are compared.
        Then:
            They should be equal, so no entry outlives the kind it serves.
        """
        # Act & assert
        assert set(PAYLOAD_TYPES) == set(TileKind)

    @pytest.mark.parametrize("kind", list(TileKind), ids=lambda k: k.value)
    def test_payload_types_should_be_checkable_by_the_conformance_suite(
        self, kind
    ):
        """Test that every catalogued shape can be validated against.

        Given:
            Each entry in the payload catalogue.
        When:
            A type adapter is constructed from it.
        Then:
            It should construct and accept every admitted shape's minimal
            producer dict. Constructing alone is not an assertion --
            ``TypeAdapter`` never returns ``None`` -- so the adapter is
            exercised rather than merely built.
        """
        # Arrange
        adapter = TypeAdapter(PAYLOAD_TYPES[kind])

        # Act & assert
        for shape in shapes_of(PAYLOAD_TYPES[kind]):
            assert adapter.validate_python(dict(WIRE_SHAPES[shape]))

    @pytest.mark.parametrize(
        "producer",
        [
            WIRE_SHAPES[DenseTilePayload],
            WIRE_SHAPES[DenseTileShapedPayload],
            WIRE_SHAPES[DenseTileMinMaxPayload],
        ],
        ids=["canonical", "shaped", "minmax"],
    )
    def test_payload_types_should_accept_every_dense_variant(self, producer):
        """Test that the dense entry admits all three live encodings.

        Given:
            A minimal producer dict for each of the three dense wire forms.
        When:
            It is validated against the catalogue's dense entry.
        Then:
            It should validate, the entry being a union of the variants that
            exist rather than only the canonical one.
        """
        # Act & assert
        TypeAdapter(PAYLOAD_TYPES[TileKind.DENSE]).validate_python(producer)

    def test_payload_types_should_accept_a_rasterized_points_tile(self):
        """Test that points go over the wire as a dense tile.

        Given:
            The encoding of a dense tile.
        When:
            It is validated against the catalogue's points entry.
        Then:
            It should validate, since density rasterizes points into a grid
            and there is no distinct points wire shape.
        """
        # Arrange
        payload = DenseTile(np.array([1.0, 2.0])).to_dict()

        # Act & assert
        TypeAdapter(PAYLOAD_TYPES[TileKind.POINTS]).validate_python(payload)


class TestWireShapes:
    """The documented key sets each producer must emit."""

    def test_table_should_cover_every_declared_shape(self):
        """Test that the fixture table has not fallen behind the module.

        Given:
            The table of minimal producer dicts in this test module, and every
            ``TypedDict`` the payloads module actually binds.
        When:
            The two are compared.
        Then:
            They should match, so a shape added without a table entry fails
            here rather than going silently untested.

            The declared side is read off the module rather than written out
            again here. A second hand-maintained list falls behind in exactly
            the same way the table does, and comparing the two catches only
            the case where someone updates one of them -- a new ``TypedDict``
            appended to ``payloads.py`` leaves both untouched and both in
            agreement.
        """
        # Arrange
        declared = {
            getattr(payloads, name)
            for name in dir(payloads)
            if not name.startswith("_")
            and typing.is_typeddict(getattr(payloads, name))
        }

        # Act & assert
        assert set(WIRE_SHAPES) == declared

    @pytest.mark.parametrize("shape", list(WIRE_SHAPES), ids=SHAPE_IDS)
    def test_wire_shape_should_accept_its_required_keys_alone(self, shape):
        """Test that the documented minimum really is sufficient.

        Given:
            Each wire shape paired with a dict carrying only its required keys.
        When:
            The dict is validated against the shape.
        Then:
            It should validate, so a producer emitting the minimum is correct.
        """
        # Act & assert
        TypeAdapter(shape).validate_python(WIRE_SHAPES[shape])

    @pytest.mark.parametrize(
        "shape,key", REQUIRED_KEYS, ids=REQUIRED_KEY_IDS
    )
    def test_wire_shape_should_reject_a_missing_required_key(self, shape, key):
        """Test that every required key is genuinely enforced.

        Given:
            A wire shape and its minimal producer dict with one required key
            removed.
        When:
            The remainder is validated.
        Then:
            It should raise a validation error naming the absent key. Each
            key is its own case rather than a loop inside one, so a failure
            reports which field stopped being required.
        """
        # Arrange
        adapter = TypeAdapter(shape)
        short = {k: v for k, v in WIRE_SHAPES[shape].items() if k != key}

        # Act & assert
        with pytest.raises(ValidationError, match=key):
            adapter.validate_python(short)

    @pytest.mark.parametrize(
        "shape,optional",
        [
            (DenseTilePayload, {"shape"}),
            (BedlikeTile, {"name", "zoom"}),
            (SubsTile, {"extra"}),
            (RegionRow, {"chrOffset"}),
        ],
        ids=["dense", "bedlike", "subs", "region"],
    )
    def test_wire_shape_should_declare_its_optional_keys(self, shape, optional):
        """Test that NotRequired survives to runtime.

        Given:
            Each wire shape that declares optional keys.
        When:
            Its optional key set is read.
        Then:
            It should name exactly those keys. A ``from __future__ import
            annotations`` in the payloads module would hide them, leaving this
            set empty and the required set wrongly inflated.
        """
        # Act & assert
        assert set(shape.__optional_keys__) == optional

    @pytest.mark.parametrize(
        "shape,optional",
        [
            (DenseTilePayload, "shape"),
            (BedlikeTile, "name"),
            (SubsTile, "extra"),
            (RegionRow, "chrOffset"),
        ],
        ids=["dense", "bedlike", "subs", "region"],
    )
    def test_wire_shape_should_accept_an_optional_key_either_way(
        self, shape, optional
    ):
        """Test that an optional key may be present or absent.

        Given:
            A wire shape with an optional key, and its minimal producer dict.
        When:
            The dict is validated without the key and again with it.
        Then:
            It should validate both times.
        """
        # Arrange
        adapter = TypeAdapter(shape)
        samples = {
            "shape": [1, 2],
            "name": "n",
            "extra": {},
            "chrOffset": 0,
            "zoom": 1,
        }

        # Act & assert
        adapter.validate_python(WIRE_SHAPES[shape])
        adapter.validate_python(
            {**WIRE_SHAPES[shape], optional: samples[optional]}
        )

    def test_bedlike_2d_tile_should_require_both_extra_coordinates(self):
        """Test that the 2D annotation shape extends the 1D one.

        Given:
            A 1D annotation dict carrying every BedlikeTile key.
        When:
            It is validated against the 2D shape.
        Then:
            It should raise naming both missing y coordinates, while the 2D
            dict validates against both shapes. A bare ``pytest.raises`` here
            passes on any validation error at all, including one about a
            field the docstring never mentions.
        """
        # Arrange
        adapter = TypeAdapter(Bedlike2DTile)

        # Act & assert
        with pytest.raises(ValidationError, match="yStart") as excinfo:
            adapter.validate_python(WIRE_SHAPES[BedlikeTile])
        assert {e["loc"][0] for e in excinfo.value.errors()} == {
            "yStart",
            "yEnd",
        }
        adapter.validate_python(WIRE_SHAPES[Bedlike2DTile])
        TypeAdapter(BedlikeTile).validate_python(WIRE_SHAPES[Bedlike2DTile])
