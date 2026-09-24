"""Prepared MFT composition and scale fusion use ordinary batch broadcasting."""

import equinox as eqx
import jax
import jax.numpy as np
import pytest

from dLux.utils.propagation import MFT


def _axes(nx, ny):
    return np.linspace(-0.7, 0.9, nx), np.linspace(-0.5, 0.6, ny)


def _field(batches):
    values = np.arange(2 * batches * 4 * 3).reshape(2, batches, 4, 3)
    return (1 + values / values.size) * np.exp(0.13j * values)


@pytest.mark.parametrize("batches", [2, 3])
def test_composition_matches_sequential_rectangular_transforms(batches):
    first = MFT.from_axes(_axes(3, 4), _axes(5, 2), np.array([[-1.2], [0.7]]))
    coefficients = np.linspace(-0.9, 1.1, 2 * batches).reshape(2, batches)
    second = MFT.from_axes(_axes(5, 2), _axes(2, 6), coefficients)
    field = _field(batches)

    # An explicit singleton shares the first transform over the second batch axis.
    expected = second(first(field))
    combined = first.then(second)
    actual = eqx.filter_jit(lambda operation, value: operation(value))(combined, field)
    assert actual.shape == (2, batches, 6, 2)
    assert np.allclose(actual, expected, rtol=2e-5, atol=2e-5)

    for wavelength in range(2):
        for population in range(batches):
            local_first = MFT(tuple(kernel[wavelength, 0] for kernel in first.kernels))
            local_second = MFT(
                tuple(kernel[wavelength, population] for kernel in second.kernels)
            )
            local = local_second(local_first(field[wavelength, population]))
            assert np.allclose(actual[wavelength, population], local, rtol=2e-5)


def test_scale_fusion_matches_output_scaling_and_analytic_gradient():
    transform = MFT.from_axes(_axes(3, 4), _axes(5, 2), np.array([[-1.2], [0.7]]))
    field = _field(3)
    offsets = np.linspace(0.2, 0.7, 6).reshape(2, 3)
    unscaled = transform(field)

    def calculate(scale):
        return transform.fuse_scale(scale + offsets)(field)

    scale = np.asarray(0.4)
    actual = jax.jit(calculate)(scale)
    expected = unscaled * (scale + offsets)[..., None, None]
    assert np.allclose(actual, expected, rtol=2e-6, atol=2e-5)

    gradient = jax.jit(jax.grad(lambda value: np.sum(np.abs(calculate(value)) ** 2)))(
        scale
    )
    power = np.sum(np.abs(unscaled) ** 2, axis=(-2, -1))
    expected_gradient = np.sum(2 * (scale + offsets) * power)
    assert expected_gradient > 0
    assert np.allclose(gradient, expected_gradient, rtol=2e-6)
