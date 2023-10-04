import pytest
from jax import numpy as np, config

config.update("jax_debug_nans", True)
from dLux.utils.propagation import (
    FFT,
    MFT,
    fresnel_MFT,
    calc_nfringes,
)


@pytest.fixture
def phasor():
    return np.ones((32, 32)) * np.exp(1j * np.zeros((32, 32)))


@pytest.fixture
def wavelength():
    return 1.0


@pytest.fixture
def pixel_scale():
    return 0.1


@pytest.fixture
def focal_length():
    return 2.0


@pytest.fixture
def npixels_in():
    return 32


@pytest.fixture
def npixels_out():
    return 16


@pytest.fixture
def pixel_scale_out():
    return 0.05


@pytest.fixture
def shift():
    return np.array([0.1, 0.2])


@pytest.fixture
def focal_shift():
    return 0.1


# NOTE: 'Correctness' is non-trivial in this module, so most of these tests are
# high-level run checks for shapes and nans.


@pytest.mark.parametrize("inverse", [True, False])
def test_FFT(
    phasor, wavelength, pixel_scale, focal_length, npixels_in, inverse
):
    result, new_pixel_scale = FFT(
        phasor, wavelength, pixel_scale, focal_length, pad=2, inverse=inverse
    )
    assert result.shape == (2 * npixels_in, 2 * npixels_in)


def test_calc_nfringes(
    wavelength,
    npixels_in,
    pixel_scale,
    npixels_out,
    pixel_scale_out,
    focal_length,
):
    result = calc_nfringes(
        wavelength,
        npixels_in,
        pixel_scale,
        npixels_out,
        pixel_scale_out,
        focal_length,
    )
    assert isinstance(result, float)


@pytest.mark.parametrize("inverse, pixel", [[True, False], [True, False]])
def test_MFT(
    phasor,
    wavelength,
    pixel_scale,
    npixels_out,
    pixel_scale_out,
    focal_length,
    shift,
    pixel,
    inverse,
):
    result = MFT(
        phasor,
        wavelength,
        pixel_scale,
        npixels_out,
        pixel_scale_out,
        focal_length,
        shift,
        pixel=pixel,
        inverse=inverse,
    )
    assert not np.isnan(result).any()
    assert result.shape == (npixels_out, npixels_out)


@pytest.mark.parametrize("inverse, pixel", [[True, False], [True, False]])
def test_fresnel_MFT(
    phasor,
    wavelength,
    pixel_scale,
    npixels_out,
    pixel_scale_out,
    shift,
    focal_length,
    focal_shift,
    pixel,
    inverse,
):
    result = fresnel_MFT(
        phasor,
        wavelength,
        pixel_scale,
        npixels_out,
        pixel_scale_out,
        focal_length,
        focal_shift,
        shift,
        pixel=pixel,
        inverse=inverse,
    )
    assert not np.isnan(result).any()
    assert result.shape == (npixels_out, npixels_out)
