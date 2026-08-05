"""Tests for dLux.utils.helpers."""

from collections import OrderedDict

import jax.numpy as np
import pytest

import dLux.utils as dlu


@pytest.mark.parametrize(
    ("value", "ndim", "expected"),
    [(4, None, (4,)), (4, 3, (4, 4, 4)), ((4, 6), 2, (4, 6))],
)
def test_as_size(value, ndim, expected):
    assert dlu.as_size(value, ndim) == expected


def test_as_axis():
    assert np.array_equal(dlu.as_axis(2.0, 2), [2.0, 2.0])
    assert np.array_equal(dlu.as_axis([1.0, 2.0]), [1.0, 2.0])

    mapped = dlu.as_axis(np.ones((3, 1)), 2)
    assert mapped.shape == (3, 2)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: dlu.as_size(2.5),
        lambda: dlu.as_size((2, 3), 3),
        lambda: dlu.as_size(0),
        lambda: dlu.as_axis((1.0, 2.0, 3.0), 2),
    ],
)
def test_axis_validation(operation):
    with pytest.raises((TypeError, ValueError)):
        operation()


def test_tree_and_dictionary_helpers():
    assert np.array_equal(dlu.map2array(lambda x: x + 1, {"a": 1, "b": 2}), [2, 3])

    values = dlu.list2dictionary([1, 2, ("named", 3)], ordered=True)
    assert isinstance(values, OrderedDict)
    assert list(values) == ["int_0", "int_1", "named"]


def test_tree_mapping_with_custom_leaves():
    tree = [(1, 2), (3, 4)]
    output = dlu.map2array(
        lambda value: sum(value), tree, leaf_fn=lambda value: isinstance(value, tuple)
    )

    assert np.array_equal(output, [3, 7])


def test_layer_dictionary_edits():
    layers = OrderedDict((("first", 1), ("last", 3)))
    inserted = dlu.insert_layer(layers, ("middle", 2), 1, (int,))
    assert list(inserted.items()) == [("first", 1), ("middle", 2), ("last", 3)]
    assert list(dlu.remove_layer(inserted, "middle")) == ["first", "last"]


@pytest.mark.parametrize("cartesian", [True, False])
def test_complex_roundtrip(cartesian):
    value = np.asarray([1 + 2j, -3 + 0.5j])
    values, reconstruct = dlu.from_complex(value, cartesian)
    assert np.allclose(reconstruct(values), value)


def test_display_and_error_helpers():
    assert np.array_equal(dlu.imshow_extent(2), [-1, 1, -1, 1])
    error = dlu.missing_attribute_error(object(), "bad", ["good"], "try this")
    assert isinstance(error, AttributeError)
    assert "bad" in str(error) and "good" in str(error)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: dlu.list2dictionary([("bad name", 1)], ordered=False),
        lambda: dlu.list2dictionary([1], ordered=False, allowed_types=(str,)),
    ],
)
def test_validation(operation):
    with pytest.raises((TypeError, ValueError)):
        operation()
