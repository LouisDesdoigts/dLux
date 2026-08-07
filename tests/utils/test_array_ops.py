"""Tests for dLux.utils.array_ops."""

import jax.numpy as np
import pytest

import dLux.utils as dlu


@pytest.mark.parametrize(
    ("shape", "target", "output_shape"),
    [
        ((3, 8), (12,), (3, 12)),
        ((2, 6, 8), (12, 10), (2, 10, 12)),
        ((4, 6, 8), (12, 10, 8), (8, 10, 12)),
    ],
)
def test_pad_crop_roundtrip(shape, target, output_shape):
    array = np.arange(np.prod(np.asarray(shape))).reshape(shape)

    padded = dlu.pad_to(array, target)
    restored = dlu.crop_to(padded, shape[-len(target) :][::-1])

    assert padded.shape == output_shape
    assert np.array_equal(restored, array)


def test_resize_mixed_dimensions():
    array = np.ones((4, 6, 8))
    output = dlu.resize(array, (4, 10, 8))

    assert output.shape == (8, 10, 4)


@pytest.mark.parametrize("mean", [True, False])
def test_downsample_nd(mean):
    array = np.arange(4 * 6 * 8, dtype=float).reshape((4, 6, 8))
    output = dlu.downsample(array, (2, 3, 2), mean)

    assert output.shape == (2, 2, 4)
    if not mean:
        assert np.isclose(output.sum(), array.sum())


@pytest.mark.parametrize("method", ["scan", "scatter"])
def test_paste_compact_arrays(method):
    arrays = np.asarray(
        [
            [[[1.0, 1.0], [1.0, 1.0]]],
            [[[2.0, 2.0], [2.0, 2.0]]],
        ]
    )
    starts = np.asarray([[0, 1], [2, 1]])

    output = dlu.paste(arrays, starts, (4, 3), method)
    expected = np.asarray(
        [[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]]]
    )

    assert np.array_equal(output, expected)


def test_paste_validation():
    with pytest.raises(ValueError, match="method"):
        dlu.paste(np.ones((1, 2, 2)), np.zeros((1, 2), int), 4, "invalid")


@pytest.mark.parametrize(
    "operation",
    [
        lambda array: dlu.pad_to(array, (9, 10)),
        lambda array: dlu.crop_to(array, (7, 4)),
        lambda array: dlu.downsample(array, (3, 2)),
    ],
)
def test_invalid_spatial_sizes(operation):
    with pytest.raises(ValueError):
        operation(np.ones((6, 8)))
