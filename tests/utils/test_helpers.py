"""Tests for dLux.utils.helpers."""

from collections import OrderedDict

import jax.numpy as np
import pytest

import dLux as dl
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
    error = dlu.missing_attribute_error(object(), "god", ["good"], "try this")
    assert isinstance(error, AttributeError)
    assert "god" in str(error) and "good" in str(error)


def test_raised_attribute_resolution():
    first = dl.Affine(translation=[0.1, 0.2])
    second = dl.Affine(translation=[0.3, 0.4])
    owner = dl.TransformChain()

    named = dlu.resolve_attr(owner, "first", {"first": first})
    raised = dlu.resolve_attr(owner, "translation", (first, second))

    assert named is first
    assert np.array_equal(raised, first.translation)

    with pytest.raises(AttributeError, match="Did you mean 'translation'"):
        dlu.resolve_attr(owner, "translaton", (first, second))


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


def test_update_strict_consumes_paths_once():
    first = dl.Polynomial(1, coeffs=[1.0, 2.0])
    second = dl.Polynomial(1, coeffs=[3.0, 4.0])

    first, second = dlu.update({"coeffs": np.zeros(2)}, first, second)

    assert np.allclose(first.coeffs, 0)
    assert np.allclose(second.coeffs, np.array([3.0, 4.0]))


def test_update_non_strict_applies_shared_paths():
    first = dl.Polynomial(1, coeffs=[1.0, 2.0])
    second = dl.Polynomial(1, coeffs=[3.0, 4.0])

    first, second = dlu.update({"coeffs": np.zeros(2)}, first, second, strict=False)

    assert np.allclose(first.coeffs, 0)
    assert np.allclose(second.coeffs, 0)


def test_update_rejects_unused_paths_without_mutating_mapping():
    params = {"missing": 1.0}
    model = dl.Polynomial(1)

    with pytest.raises(KeyError, match="'missing'"):
        dlu.update(params, model)

    assert params == {"missing": 1.0}


def test_update_nested_models_and_single_object_tuple():
    basis = dl.Basis(np.ones((1, 4, 4)), coeffs=[1.0])
    optics = dl.OpticalSystem(
        [("pupil", dl.Optic(opd=basis))],
        dl.GridSpec(4, 0.25, unit="m"),
    )
    source = dl.Source(1e-6, position=[0.0, 0.0])
    params = {"pupil.coeffs": np.zeros(1), "position": np.ones(2)}

    optics, source = dlu.update(params, optics, source)
    single = dlu.update({"position": np.zeros(2)}, source)

    assert np.allclose(optics.pupil.coeffs, 0)
    assert np.allclose(source.position, 1)
    assert isinstance(single, tuple) and len(single) == 1
