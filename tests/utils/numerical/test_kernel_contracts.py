"""Canonical class conventions and preserved functional adapters."""

import equinox as eqx
import jax.numpy as np
import pytest

import dLux.utils as dlu


def test_gaussian_class_peak_and_legacy_sum_normalisation():
    coordinates = np.array(
        np.meshgrid(np.linspace(-2, 2, 9), np.linspace(-3, 3, 7), indexing="xy")
    )
    gaussian = dlu.Gaussian(mean=(0.2, -0.1), std=(0.7, 1.1))
    profile = eqx.filter_jit(lambda: gaussian.resolve(coordinates=coordinates))()
    sampled = dlu.gaussian(
        mean=(0.2, -0.1), std=(0.7, 1.1), npixels=(9, 7), extent=(2, 3)
    )

    assert profile.max() < 1
    assert np.isclose(sampled.sum(), 1)
    assert np.allclose(profile / profile.sum(), sampled, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("n_modes", [4, 5, 6])
def test_fourier_real_mode_order_and_amplitude(n_modes):
    coordinates = (np.arange(12) - 6) / 12
    kernel = dlu.fourier_kernel_1d(n_modes, 12, scale=2)
    expected = [np.ones_like(coordinates)]
    for column in range(1, n_modes):
        phase = 2 * np.pi * ((column + 1) // 2) * coordinates
        expected.append(np.cos(phase) if column % 2 else np.sin(phase))

    assert np.allclose(kernel, 2 * np.stack(expected, axis=-1), atol=1e-6)


def test_fourier_class_and_legacy_preserve_the_same_mapping():
    coeffs = np.arange(12.0).reshape(3, 4)
    fourier = dlu.FourierBasis(npix=(11, 7), n_modes=(3, 4), coeffs=coeffs)
    image = eqx.filter_jit(lambda: fourier.resolve())()
    legacy = dlu.eval_fourier_basis(coeffs, *fourier.kernels)
    materialised = dlu.Basis(fourier.calculate_basis(), coeffs, axes=2).resolve()

    assert image.shape == (11, 7)
    assert legacy.shape == (11, 7)
    assert np.allclose(image, legacy)
    assert np.allclose(image, materialised, atol=1e-5)


@pytest.mark.parametrize("nsides", [0, 5])
def test_dynamic_zernike_uses_the_functional_mode_convention(nsides):
    coordinates = dlu.pixel_coords(12, diameter=2)
    modes = (1, 2, 4, 7)
    basis = dlu.Basis(dlu.Zernike(nolls=modes, n_sides=nsides))
    sampled = basis.basis.evaluate(coordinates=coordinates)
    if nsides:
        expected = dlu.polike_basis(nsides, modes, coordinates)
    else:
        expected = dlu.zernike_basis(modes, coordinates)

    # Current Cartesian modes use JAX factorial; released polar modes preserve
    # their scalar-factorial arithmetic. Float32 cancellation differs by up to
    # 2.15e-6 here, particularly near polynomial zeros.
    assert np.allclose(sampled, expected, rtol=3e-6, atol=3e-6)
