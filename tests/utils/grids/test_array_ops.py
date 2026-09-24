"""Tests for dLux.utils.grids.array_ops."""

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


def test_fft_spec_tracks_working_and_output_sampling():
    source = dlu.Grid((4, 6), d=(0.5, 0.25), c=(0.2, -0.1))
    recipe = dlu.FFTSpec(pad=2, crop=2, c=(0.125, -0.25))
    working = recipe.working_grid(source)
    assert isinstance(recipe, dlu.Spec)
    assert recipe.ndim == 2
    assert working.n == (8, 12)
    assert np.allclose(working.d, source.d)
    assert np.allclose(working.c, source.c)
    native = working.frequency()
    output = recipe.output_grid(native, source.n)
    assert output.n == source.n
    assert np.allclose(output.c, np.array([0.125, -0.25]))
    assert np.allclose(output.d, native.d)
    assert dlu.FFTSpec().output_grid(source, source.n).n == source.n


@pytest.mark.parametrize(
    "operation",
    [
        lambda: dlu.ResizeSpec().output_size((3,)),
        lambda: dlu.as_size((1, 2.5)),
        lambda: dlu.pad_to(np.ones(3), 5),
        lambda: dlu.pad_to(np.ones((5, 5)), 3),
        lambda: dlu.crop_to(np.ones((3, 3)), 5),
        lambda: dlu.downsample(np.ones(3), (1, 1)),
        lambda: dlu.ResizeSpec(crop=8).counts((4, 4)),
        lambda: dlu.ResizeSpec().counts((4,)),
        lambda: dlu.ResizeSpec().crop_axes((np.arange(3),)),
        lambda: dlu.FFTSpec(n=2).working_grid(dlu.Grid(4, 1)),
    ],
)
def test_array_operation_validation_tails(operation):
    """Coverage tails: malformed size requests and crop/pad direction checks."""
    with pytest.raises((TypeError, ValueError)):
        operation()
