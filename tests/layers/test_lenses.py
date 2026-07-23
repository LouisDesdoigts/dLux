import equinox as eqx
import jax
import jax.numpy as np
import pytest

from dLux import PolarisedWavefront, Wavefront
from dLux.layers import Lens, Wedge
from dLux.parametric import (
    BaseParametric,
    CauchyIndex,
    InterpolatedIndex,
)

WAVEFRONT = Wavefront(1e-6, 8, diameter=1.0)


class WavefrontValue(BaseParametric):
    scale: float

    def __init__(self, scale):
        self.scale = np.asarray(scale, dtype=float)

    def evaluate(self, *, wavefront, **kwargs):
        return self.scale * np.ones_like(wavefront.amplitude)


def test_cauchy_index():
    assert np.allclose(
        CauchyIndex([1.5, 0.01, 0.001]).evaluate(wavefront=WAVEFRONT), 1.511
    )


def test_cauchy_index_is_vectorised_over_wavelength():
    wavefront = Wavefront(np.array([0.5e-6, 1e-6, 2e-6]), 8, diameter=1.0)
    output = CauchyIndex([1.5, 0.01]).evaluate(wavefront=wavefront)

    assert output.shape == (3,)
    assert np.allclose(output, np.array([1.54, 1.51, 1.5025]))


@pytest.mark.parametrize("coefficients", [[], [[1.5, 0.01]]])
def test_cauchy_index_validates_coefficients(coefficients):
    with pytest.raises(ValueError):
        CauchyIndex(coefficients)


def test_cauchy_index_validates_scale():
    with pytest.raises(ValueError):
        CauchyIndex([1.5], scale=0)


def test_interpolated_index():
    n = InterpolatedIndex([1e-6, 2e-6], [1.6, 1.5])
    wavefront = Wavefront(1.5e-6, 8, diameter=1.0)
    assert np.allclose(n.evaluate(wavefront=wavefront), 1.55)


@pytest.mark.parametrize(
    "wavelengths,indices",
    [
        ([[1e-6, 2e-6]], [1.6, 1.5]),
        ([1e-6, 2e-6], [[1.6, 1.5]]),
        ([1e-6, 2e-6], [1.5]),
        ([1e-6], [1.5]),
        ([2e-6, 1e-6], [1.5, 1.6]),
        ([1e-6, 1e-6], [1.5, 1.6]),
    ],
)
def test_interpolated_index_validation(wavelengths, indices):
    with pytest.raises(ValueError):
        InterpolatedIndex(wavelengths, indices)


def test_interpolated_index_extrapolation():
    wavefront = WAVEFRONT.set(wavelength=0.5e-6)
    samples = ([1e-6, 2e-6], [1.6, 1.5])
    assert InterpolatedIndex(*samples).extrapolate is False
    assert np.allclose(
        InterpolatedIndex(*samples, extrapolate=True).evaluate(wavefront=wavefront),
        1.65,
    )


def test_lens_applies_residual_material_opd():
    thickness = np.ones((8, 8)) * 1e-7
    output = Lens(thickness, n=1.5)(WAVEFRONT)
    assert np.allclose(output.phasor, WAVEFRONT.add_opd(0.5 * thickness).phasor)


def test_lens_resolves_parametric_thickness_and_index():
    lens = Lens(WavefrontValue(1e-7), n=CauchyIndex([1.5, 0.01]))
    output = lens(WAVEFRONT)
    expected = WAVEFRONT.add_opd(np.ones((8, 8)) * 0.51e-7)
    assert np.allclose(output.phasor, expected.phasor)


def test_lens_is_chromatic():
    lens = Lens(np.ones((8, 8)) * 1e-7, n=CauchyIndex([1.5, 0.01]))
    blue = WAVEFRONT.set(wavelength=0.5e-6)
    red = WAVEFRONT.set(wavelength=1e-6)
    assert np.allclose(lens(blue).phasor, blue.add_opd(0.54e-7).phasor)
    assert np.allclose(lens(red).phasor, red.add_opd(0.51e-7).phasor)


@pytest.mark.parametrize("wavefront_type", [Wavefront, PolarisedWavefront])
@pytest.mark.parametrize("mapped_sampling", [False, True])
def test_lens_supports_chromatic_wavefront_variants(wavefront_type, mapped_sampling):
    wavelengths = np.array([0.5e-6, 1e-6, 2e-6])
    wavefront = wavefront_type(wavelengths, 8, diameter=1.0)
    if mapped_sampling:
        wavefront = wavefront.set(pixel_scale=np.array([1 / 8, 1 / 10, 1 / 12]))
    thickness = np.ones((8, 8)) * 1e-7
    index = CauchyIndex([1.5, 0.01])

    output = Lens(thickness, n=index)(wavefront)
    n = index.evaluate(wavefront=wavefront)
    expected = wavefront.add_opd((n - 1)[..., None, None] * thickness)

    assert type(output) is wavefront_type
    assert output.phasor.shape == wavefront.phasor.shape
    assert np.allclose(output.phasor, expected.phasor)


@pytest.mark.parametrize("normalise", [False, True])
def test_lens_applies_transmission_and_normalisation(normalise):
    output = Lens(0, n=1.5, transmission=0.5, normalise=normalise)(WAVEFRONT)
    expected_power = 1.0 if normalise else 0.25 * WAVEFRONT.power
    assert np.allclose(output.power, expected_power)


def test_wedge_applies_linear_opd():
    angle = np.array([1e-3, -2e-3])
    output = Wedge(angle, n=1.5)(WAVEFRONT)
    x, y = WAVEFRONT.coordinates()
    thickness = x * np.tan(angle[0]) + y * np.tan(angle[1])
    assert np.allclose(output.phasor, WAVEFRONT.add_opd(0.5 * thickness).phasor)


@pytest.mark.parametrize("angle", [1e-3, [1e-3], [1e-3, 0, 0]])
def test_wedge_validates_angle(angle):
    with pytest.raises(ValueError):
        Wedge(angle, n=1.5)


def test_referenced_wedge_has_no_effect_at_reference_wavelength():
    n = InterpolatedIndex([1e-6, 2e-6], [1.6, 1.5])
    wavefront = Wavefront(1.5e-6, 8, diameter=1.0)
    output = Wedge([1e-3, 0], n, reference_wavelength=1.5e-6)(wavefront)
    assert np.allclose(output.phasor, wavefront.phasor)


def test_referenced_wedge_applies_differential_dispersion():
    n = InterpolatedIndex([1e-6, 2e-6], [1.6, 1.5])
    wavefront = WAVEFRONT.set(wavelength=1e-6)
    wedge = Wedge([1e-6, 0], n, reference_wavelength=2e-6)
    x, _ = wavefront.coordinates()
    expected = wavefront.add_opd(0.1 * x * np.tan(1e-6))
    assert np.allclose(wedge(wavefront).phasor, expected.phasor)


@pytest.mark.parametrize("wavefront_type", [Wavefront, PolarisedWavefront])
@pytest.mark.parametrize("mapped_sampling", [False, True])
def test_wedge_supports_chromatic_wavefront_variants(wavefront_type, mapped_sampling):
    wavelengths = np.array([1e-6, 1.5e-6, 2e-6])
    wavefront = wavefront_type(wavelengths, 8, diameter=1.0)
    if mapped_sampling:
        wavefront = wavefront.set(pixel_scale=np.array([1 / 8, 1 / 10, 1 / 12]))
    index = InterpolatedIndex([1e-6, 2e-6], [1.6, 1.5])
    angle = np.array([1e-3, -2e-3])

    output = Wedge(angle, n=index)(wavefront)
    n = index.evaluate(wavefront=wavefront)
    coordinates = wavefront.coordinates()
    x, y = coordinates[..., 0, :, :], coordinates[..., 1, :, :]
    thickness = x * np.tan(angle[0]) + y * np.tan(angle[1])
    expected = wavefront.add_opd((n - 1)[..., None, None] * thickness)

    assert type(output) is wavefront_type
    assert output.phasor.shape == wavefront.phasor.shape
    assert np.allclose(output.phasor, expected.phasor)


def test_lens_supports_jit_and_gradients():
    apply = eqx.filter_jit(lambda thickness: Lens(thickness, n=1.5)(WAVEFRONT))
    thickness = np.ones((8, 8)) * 1e-7
    assert np.allclose(apply(thickness).phasor, Lens(thickness, 1.5)(WAVEFRONT).phasor)

    loss = lambda value: np.sum(Lens(value, n=1.5)(WAVEFRONT).phasor.real)
    gradient = jax.grad(loss)(thickness)
    assert gradient.shape == thickness.shape
    assert np.all(np.isfinite(gradient))
