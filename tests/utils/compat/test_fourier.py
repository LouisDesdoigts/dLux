"""Tests for dLux.utils.fourier."""

import jax.numpy as np
import pytest
from jax import grad, vmap

import dLux.utils as dlu

from tests.helpers import assert_jittable


def test_fourier_shift_matches_periodic_mode():
    y, x = np.mgrid[:12, :16]
    image = np.cos(2 * np.pi * (2 * x / 16 + y / 12))
    shift = np.array([0.4, -0.25])
    expected = np.cos(2 * np.pi * (2 * (x - shift[0]) / 16 + (y - shift[1]) / 12))

    shifted = assert_jittable(dlu.fourier_shift, image, shift, pad=False)

    assert np.allclose(shifted, expected, atol=1e-5)


def test_fourier_shift_preserves_batches_and_complex_values():
    image = np.zeros((2, 7, 9), dtype=complex)
    image = image.at[:, 3, 4].set(np.array([1 + 2j, 3 - 1j]))
    expected = np.zeros_like(image).at[:, 2, 6].set(image[:, 3, 4])

    shifted = dlu.fourier_shift(image, np.array([2.0, -1.0]))

    assert np.issubdtype(shifted.dtype, np.complexfloating)
    assert np.allclose(shifted, expected, atol=1e-6)


def test_fourier_shift_avoids_edge_wrapping_and_differentiates():
    image = np.zeros((9, 9)).at[4, 7].set(1.0)
    shifted = dlu.fourier_shift(image, np.array([2.0, 0.0]))
    derivative = grad(lambda dx: dlu.fourier_shift(image, np.array([dx, 0.0]))[4, 7])(
        0.2
    )

    assert np.allclose(shifted, 0, atol=1e-6)
    assert np.isfinite(derivative)
    assert not np.isclose(derivative, 0)


def test_fourier_shift_vectorises_over_shifts():
    image = np.zeros((7, 7)).at[3, 3].set(1.0)
    shifts = np.array([[1.0, 0.0], [0.0, -1.0]])
    shifted = vmap(dlu.fourier_shift, in_axes=(None, 0))(image, shifts)

    assert np.allclose(shifted[0, 3, 4], 1)
    assert np.allclose(shifted[1, 2, 3], 1)


@pytest.mark.parametrize(
    "array, shift, pad, message",
    [
        (np.ones(3), np.zeros(2), True, "at least two dimensions"),
        (np.ones((3, 3)), np.zeros(1), True, "shape"),
        (np.ones((3, 3)), np.zeros(2), 1, "bool"),
    ],
)
def test_fourier_shift_validates_inputs(array, shift, pad, message):
    with pytest.raises((TypeError, ValueError), match=message):
        dlu.fourier_shift(array, shift, pad)


@pytest.mark.parametrize("n_modes", [4, 5])
def test_fourier_kernel_contract(n_modes):
    kernel = assert_jittable(dlu.fourier_kernel_1d, n_modes, 16)
    assert kernel.shape == (16, n_modes)
    assert np.issubdtype(kernel.dtype, np.floating)


def test_asymmetric_fourier_basis():
    kernel_x, kernel_y = dlu.fourier_kernels((3, 5), (16, 12))
    coeffs = np.arange(15.0).reshape((3, 5))
    output = assert_jittable(
        dlu.eval_fourier_basis,
        coeffs,
        kernel_x,
        kernel_y,
    )

    assert kernel_x.shape == (16, 3)
    assert kernel_y.shape == (12, 5)
    assert output.shape == (16, 12)


def test_fourier_scaling():
    reference = dlu.fourier_kernel_1d(5, 16)
    assert np.allclose(dlu.fourier_kernel_1d(5, 16, 2.0), 2 * reference)


@pytest.mark.parametrize("value", [0, (3,), (3, -1), "bad"])
def test_invalid_kernel_size(value):
    with pytest.raises((TypeError, ValueError)):
        dlu.fourier_kernels(value, 8)
