"""Public Fourier kernels and validation of array-axis dimensions."""

import jax.numpy as np
import pytest
import dLux.utils as dlu


def test_fourier_modes_have_known_phase_and_scaling():
    kernel = dlu.fourier_kernel_1d(3, 4, scale=2)
    assert np.allclose(kernel[:, 0], 2)
    assert np.allclose(kernel[:, 1], np.array([-2.0, 0.0, 2.0, 0.0]), atol=2e-7)
    assert np.allclose(kernel[:, 2], np.array([0.0, -2.0, 0.0, 2.0]), atol=2e-7)


def test_fourier_size_validation_tail():
    """Coverage tail: the separable Fourier expansion is two-dimensional."""
    with pytest.raises(ValueError, match="scalar or a"):
        dlu.fourier_kernels((2, 3, 4), 5)
