"""Generic batches are explicit and independent of physical interpretations."""

import jax.numpy as np
import pytest
import dLux.utils as dlu
from tests.helpers import assert_jittable


def test_scalar_vector_and_image_batches_preserve_value_axes():
    images = np.arange(24.0).reshape(2, 1, 3, 4)
    vectors = np.arange(15.0).reshape(5, 3)
    out, vec = assert_jittable(
        lambda a, b: dlu.broadcast_batches(a, b, ndims=(2, 1)), images, vectors
    )
    assert out.shape == (2, 5, 3, 4)
    assert vec.shape == (2, 5, 3)
    for i in range(2):
        for j in range(5):
            assert np.array_equal(out[i, j], images[i, 0])
            assert np.array_equal(vec[i, j], vectors[j])
    assert np.array_equal(dlu.flatten_batch(out, 2), out.reshape(10, 3, 4))
    assert dlu.flatten_batch(images[0, 0], 2).shape == (1, 3, 4)
    assert np.array_equal(dlu.broadcast_batch(2, (2, 3), 0), np.full((2, 3), 2))
    assert np.array_equal(
        dlu.flatten_batch(np.arange(6).reshape(2, 3), 0), np.arange(6)
    )
    a, b = dlu.broadcast_batches(np.ones((2, 3)), np.arange(3), ndims=1)
    assert a.shape == b.shape == (2, 3)


def test_as_axis_retains_component_dimension():
    assert dlu.as_axis(None) is None
    assert np.array_equal(dlu.as_axis(2), np.array([2.0]))
    assert np.array_equal(dlu.as_axis(2, 2), np.array([2.0, 2.0]))
    assert np.array_equal(dlu.as_axis([[1], [2]], 2), np.array([[1, 1], [2, 2]]))
    assert np.array_equal(dlu.as_axis([1, 2]), np.array([1.0, 2.0]))


@pytest.mark.parametrize(
    "operation",
    [
        lambda: dlu.broadcast_batch([1, 2], (), 2),
        lambda: dlu.flatten_batch([1, 2], -1),
        lambda: dlu.broadcast_batches([1, 2], ndims=(1, 1)),
        lambda: dlu.broadcast_batches([1, 2], ndims=2),
        lambda: dlu.as_axis([1, 2, 3], 2),
    ],
)
def test_batching_validation_tails(operation):
    """Coverage tails: malformed intrinsic dimensions and component counts."""
    with pytest.raises(ValueError):
        operation()
