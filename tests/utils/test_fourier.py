"""Tests for dLux.utils.fourier."""

import jax.numpy as np
import pytest

import dLux.utils as dlu

from tests.helpers import assert_jittable


@pytest.mark.parametrize("n_modes", [4, 5])
def test_fourier_kernel_contract(n_modes):
    kernel = assert_jittable(dlu.fourier_kernel_1d, n_modes, 16)
    assert kernel.shape == (16, n_modes)
    assert np.issubdtype(kernel.dtype, np.floating)


def test_asymmetric_fourier_basis():
    kernel_x, kernel_y = dlu.fourier_kernels((3, 5), (16, 12))
    coefficients = np.arange(15.0).reshape((3, 5))
    output = assert_jittable(
        dlu.eval_fourier_basis,
        coefficients,
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
