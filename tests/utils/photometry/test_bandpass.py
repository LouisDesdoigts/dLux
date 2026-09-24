"""BandPass resampling retains wavelength and integration contracts."""

import equinox as eqx
import jax.numpy as np
import pytest

import dLux.utils as dlu

from tests.helpers import assert_jittable


@pytest.mark.parametrize("explicit", [False, True])
def test_resample_preserves_bandpass_and_rebuilds_integration(explicit):
    options = {"unit": "nm"}
    if explicit:
        band = dlu.BandPass(wavelengths=[500, 550, 600, 650, 700], **options)
    else:
        band = dlu.BandPass(600, 200, n=5, **options)

    sampled = assert_jittable(lambda value: value.resample(3, 25.0), band)

    assert isinstance(sampled, dlu.BandPass)
    assert sampled.unit == "nm"
    assert sampled.n == 3
    assert sampled.alias == band.alias
    assert np.allclose(sampled.wavelengths / 1e-9, np.array([575, 600, 625]))
    assert np.allclose(sampled.widths / 1e-9, np.array([25, 25, 25]))
    expected = [500, 550, 600, 650, 700] if explicit else [520, 560, 600, 640, 680]
    assert np.allclose(band.wavelengths / 1e-9, np.array(expected))


def test_resample_to_one_sample_retains_a_finite_integration_bin():
    band = dlu.BandPass(600, 200, n=5, unit="nm")
    sampled = assert_jittable(lambda value: value.resample(1, 20.0), band)

    assert sampled.n == 1
    assert np.allclose(sampled.wavelengths / 1e-9, np.array([600]))
    assert np.allclose(sampled.widths / 1e-9, np.array([20]))


def test_field_scale_to_keeps_bandpass_sampling_and_complex_values():
    band = dlu.BandPass(600, 200, n=5, unit="nm")
    coordinate = (band.wavelengths / 1e-9 - 600) / 100
    field = dlu.Field(coordinate**3 + 1j * coordinate, band)

    # Fixed BandPass metadata is prepared outside tracing; array resampling is JIT-safe.
    output = band.resample(3, 25.0)
    sampled = eqx.filter_jit(lambda value: value.resample(output))(field)

    assert isinstance(sampled.grid, dlu.BandPass)
    assert sampled.grid.n == 3
    assert np.allclose(sampled.grid.wavelengths / 1e-9, np.array([575, 600, 625]))
    expected = np.array([-0.25, 0.0, 0.25])
    assert np.allclose(sampled.array, expected**3 + 1j * expected, atol=2e-6)
    assert np.allclose(sampled.grid.widths.sum() / 1e-9, 75)


def test_resampling_requires_fixed_python_spacing():
    band = dlu.BandPass(600, 200, n=5, unit="nm")
    with pytest.raises(TypeError, match="fixed Python"):
        band.resample(3, np.asarray(25.0))


def test_custom_integration_widths_are_not_part_of_equal_bin_sampling():
    with pytest.raises(TypeError, match="widths"):
        dlu.BandPass(wavelengths=[500, 600, 700], widths=[20, 30, 40], unit="nm")
