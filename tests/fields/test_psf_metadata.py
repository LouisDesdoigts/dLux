"""Optical responses retain sampling without absorbing source brightness."""

import equinox as eqx
import jax.numpy as np
import pytest

import dLux as dl
from dLux.fields import PSF


def test_psf_keeps_wavelength_sampling_and_local_optical_loss():
    grid = dl.Grid(4, d=0.1, unit="m")
    wavelengths = np.array([5e-7, 7e-7])
    wavefront = dl.Wavefront(wavelengths, grid)
    transmitted = dl.Optic(transmission=0.5)(wavefront)
    result = eqx.filter_jit(PSF.from_wavefront)(transmitted)
    assert np.array_equal(result.wavelengths, wavelengths)
    assert result.positions is None
    assert np.allclose(result.data.sum((-2, -1)), 0.25)
    # Sampling changes retain the spectral labels and the photon response.
    coarse = result.downsample(2, mean=False)
    assert np.array_equal(coarse.wavelengths, wavelengths)
    assert np.allclose(coarse.data.sum((-2, -1)), 0.25)


def test_psf_metadata_accepts_monochromatic_and_component_responses():
    grid = dl.Grid(4, d=0.1, unit="m")
    monochromatic = PSF.from_wavefront(dl.Wavefront(5e-7, grid))
    assert monochromatic.wavelengths.shape == ()
    assert monochromatic.data.shape == (4, 4)
    positions = np.array([[0, 0], [1e-7, 0]])
    result = PSF(
        np.ones((2, 3, 4, 4)),
        grid,
        wavelengths=np.array([5e-7, 6e-7, 7e-7]),
        positions=positions,
    )
    assert np.array_equal(result.positions, positions)
    with pytest.raises(ValueError, match="wavelength axes"):
        PSF(result.data, grid, wavelengths=np.array([5e-7, 6e-7]))
    with pytest.raises(ValueError, match="positions"):
        PSF(result.data, grid, positions=np.zeros((2, 3)))
