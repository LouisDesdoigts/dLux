from collections import OrderedDict

import pytest
from jax import numpy as np, config

config.update("jax_debug_nans", True)

from dLux.utils import helpers as helpers_utils


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def tree():
    return {"a": {"b": 1, "c": 2}, "d": {"e": 3, "f": 4}}


@pytest.fixture
def fn():
    return lambda x: x + 1


@pytest.fixture
def leaf_fn():
    return lambda x: isinstance(x, int)


@pytest.fixture
def list_in():
    return [("layer", 1), ("layer", 2), 3]


@pytest.fixture
def ordered():
    return True


@pytest.fixture
def allowed_types():
    return (int, str)


@pytest.fixture
def layers():
    return OrderedDict([("layer1", 1), ("layer2", 2), ("layer3", 3)])


@pytest.fixture
def new_layer():
    return ("layer4", 4)


@pytest.fixture
def insert_index():
    return 1


@pytest.fixture
def key():
    return "layer2"


@pytest.fixture
def layer_type():
    return int


# ============================================================================
# Tests for map2array
# ============================================================================
class TestMap2Array:
    """Tests for pytree-to-array mapping."""

    def test_with_and_without_leaf_fn(self, tree, fn, leaf_fn):
        """Mapping returns a flat array both with and without a custom leaf function."""
        result = helpers_utils.map2array(fn, tree, leaf_fn)
        assert result.shape == (4,)

        result = helpers_utils.map2array(fn, tree)
        assert result.shape == (4,)


# ============================================================================
# Tests for list2dictionary
# ============================================================================
class TestList2Dictionary:
    """Tests for layer list normalization into dictionaries."""

    def test_basic(self, list_in, ordered, allowed_types):
        """Duplicate names are indexed and unnamed entries use their type name."""
        result = helpers_utils.list2dictionary(list_in, ordered, allowed_types)
        assert isinstance(result, OrderedDict)
        assert len(result) == 3
        assert list(result.keys()) == ["layer_0", "layer_1", "int"]
        assert list(result.values()) == [1, 2, 3]

    def test_invalid_type_raises(self, list_in, ordered):
        """Entries outside the allowed types raise TypeError."""
        with pytest.raises(TypeError):
            helpers_utils.list2dictionary(list_in, ordered, (float,))

    def test_space_in_name_raises(self, ordered):
        """Names containing spaces raise ValueError."""
        with pytest.raises(ValueError):
            helpers_utils.list2dictionary([("a space", 1)], ordered, (int,))


# ============================================================================
# Tests for insert_layer
# ============================================================================
class TestInsertLayer:
    """Tests for inserting layers into ordered dictionaries."""

    def test_basic(self, layers, new_layer, insert_index, layer_type):
        """
        Insertion preserves order and inserts the new layer at the requested index.
        """
        result = helpers_utils.insert_layer(layers, new_layer, insert_index, layer_type)
        assert isinstance(result, OrderedDict)
        assert len(result) == 4
        assert list(result.keys()) == ["layer1", "layer4", "layer2", "layer3"]
        assert list(result.values()) == [1, 4, 2, 3]


# ============================================================================
# Tests for remove_layer
# ============================================================================
class TestRemoveLayer:
    """Tests for removing layers from ordered dictionaries."""

    def test_basic(self, layers, key):
        """Removal deletes the requested key and preserves the remaining order."""
        result = helpers_utils.remove_layer(layers, key)
        assert isinstance(result, OrderedDict)
        assert len(result) == 2
        assert list(result.keys()) == ["layer1", "layer3"]
        assert list(result.values()) == [1, 3]


# ============================================================================
# Tests for inherit_docstrings
# ============================================================================
class TestInheritDocstrings:
    """Tests for docstring inheritance helpers."""

    def test_default_method_names(self):
        """When method_names is omitted, __call__ docstrings are inherited."""

        class Parent:
            def __call__(self):
                """Parent docstring."""

        class Child(Parent):
            def __call__(self):
                pass

        helpers_utils.inherit_docstrings(Child)
        assert Child.__call__.__doc__ == "Parent docstring."

    def test_missing_parent_docstring(self):
        """Methods remain undocumented when no parent provides documentation."""

        class Parent:
            def method(self):
                pass

        class Child(Parent):
            def method(self):
                pass

        helpers_utils.inherit_docstrings(Child, ["method"])
        assert Child.method.__doc__ is None


# ============================================================================
# Tests for imshow_extent
# ============================================================================
class TestImshowExtent:
    """Tests for image extent helper generation."""

    def test_output(self):
        """Extent is symmetric about zero with half-size endpoints."""
        result = helpers_utils.imshow_extent(2.0)
        assert result.shape == (4,)
        assert np.allclose(result, np.array([-1.0, 1.0, -1.0, 1.0]))


# ============================================================================
# Tests for from_complex
# ============================================================================
class TestFromComplex:
    """Tests for complex array decomposition."""

    @pytest.mark.parametrize("complex", [True, False])
    def test_round_trip(self, complex):
        """Cartesian and polar decompositions reconstruct the input."""
        array = np.array([[1 + 2j, 3 - 4j], [-5 + 6j, -7 - 8j]])
        vals, return_fn = helpers_utils.from_complex(array, complex)
        assert vals.shape == (2, 2, 2)
        assert np.allclose(return_fn(vals), array)

    def test_cartesian_values(self):
        """Cartesian decomposition returns real and imaginary components."""
        array = np.array([[1 + 2j, 3 - 4j]])
        vals, _ = helpers_utils.from_complex(array, complex=True)
        assert np.allclose(vals[0], array.real)
        assert np.allclose(vals[1], array.imag)

    def test_polar_values(self):
        """Polar decomposition returns amplitude and phase components."""
        array = np.array([[1 + 2j, 3 - 4j]])
        vals, _ = helpers_utils.from_complex(array, complex=False)
        assert np.allclose(vals[0], np.abs(array))
        assert np.allclose(vals[1], np.angle(array))


# ============================================================================
# Tests for missing_attribute_error
# ============================================================================
class TestMissingAttributeError:
    """Tests for standardized missing-attribute errors."""

    def test_basic(self):
        """The missing key appears in the error message."""

        class Foo:
            pass

        err = helpers_utils.missing_attribute_error(Foo(), "bar")
        assert "bar" in str(err)

    def test_valid_attrs(self):
        """Valid attribute lists are appended to the error message."""

        class Foo:
            pass

        err = helpers_utils.missing_attribute_error(
            Foo(), "bar", valid_attrs=["a", "b", "c"]
        )
        assert "a" in str(err)

    def test_hint(self):
        """Hints are appended to the error message."""

        class Foo:
            pass

        err = helpers_utils.missing_attribute_error(
            Foo(), "bar", hint="Try this instead."
        )
        assert "Try this instead." in str(err)


# ============================================================================
# Tests for private casting helpers
# ============================================================================
class TestCastTuple:
    """Tests for tuple casting helper."""

    def test_non_int_element_raises(self):
        """Tuple elements must be integers."""
        with pytest.raises(ValueError, match="must be integers"):
            helpers_utils._cast_tuple((1, 2.0), "npixels")

    def test_invalid_type_raises(self):
        """Non-int, non-tuple inputs raise ValueError."""
        with pytest.raises(ValueError, match="must be an int or a tuple"):
            helpers_utils._cast_tuple(1.0, "npixels")


class TestCastScalar:
    """Tests for scalar and scalar-tuple casting helper."""

    def test_array_valid(self):
        """A 1D array of the correct length is converted to a tuple."""
        arr = np.array([1.0, 2.0])
        result = helpers_utils._cast_scalar(arr, 2, "test")
        assert result == (1.0, 2.0)

    def test_array_wrong_shape_raises(self):
        """Higher-dimensional arrays raise ValueError."""
        arr = np.ones((3, 3))
        with pytest.raises(ValueError):
            helpers_utils._cast_scalar(arr, 2, "test")

    def test_tuple_wrong_length_raises(self):
        """Tuple lengths must match the requested dimensionality."""
        with pytest.raises(ValueError, match="Length"):
            helpers_utils._cast_scalar((1.0, 2.0), 3, "test")

    def test_non_scalar_element_raises(self):
        """Tuple elements must themselves be scalar-like."""
        with pytest.raises(ValueError, match="scalars"):
            helpers_utils._cast_scalar(([1.0, 2.0],), 1, "test")

    def test_invalid_type_raises(self):
        """Unsupported input types raise ValueError."""
        with pytest.raises(ValueError, match="must be a scalar"):
            helpers_utils._cast_scalar([1.0, 2.0], 2, "test")


class TestInputLen:
    """Tests for input dimensionality inspection."""

    def test_1d_array(self):
        """1D arrays report their leading length."""
        arr = np.array([1.0, 2.0, 3.0])
        assert helpers_utils._input_len(arr, "test") == 3

    def test_2d_array_raises(self):
        """Arrays with ndim > 1 raise ValueError."""
        arr = np.ones((3, 3))
        with pytest.raises(ValueError, match="ndim"):
            helpers_utils._input_len(arr, "test")
