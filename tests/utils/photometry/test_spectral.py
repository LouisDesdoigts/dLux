"""One spectrum contract for numerical shapes, Planck spectra and photon samples."""

from math import prod

import jax
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux.utils as dlu
from tests.helpers import assert_jittable


def test_polynomial_spectrum_has_a_fixed_reference_and_differentiable_coefficients():
    wavelengths = np.linspace(600e-9, 700e-9, 5)
    shape = dlu.Polynomial(1, [1.0, 0.5], transform=zdx.Map(s=1e7, b=-6.5))
    spectrum = dlu.Spectrum(shape)
    values = assert_jittable(lambda model: model.sample(wavelengths), spectrum)
    assert np.allclose(values, np.linspace(0.75, 1.25, 5))
    assert np.allclose(spectrum.sample(wavelengths[1:3]), values[1:3])
    gradient = jax.grad(
        lambda c: spectrum.set("value.coeffs", c).sample(wavelengths)[-1]
    )(shape.coeffs)
    assert np.allclose(gradient, np.asarray([1.0, 0.5]))


def test_selected_polynomial_constructs_inside_jit_as_a_spectral_shape():
    coordinates = np.asarray([-1.0, -0.5, 0.5, 1.0])
    centre, scale = 2**-20, 2**-22
    wavelengths = centre + scale * coordinates
    coeffs = np.asarray([0.2, -0.1])

    def evaluate(coeffs):
        shape = dlu.Polynomial(
            degrees=(3, 1, 3),
            coeffs=coeffs,
            transform=zdx.Map(s=1 / scale, b=-centre / scale),
        )
        return dlu.Spectrum(zdx.Map(x=shape, b=1)).sample(wavelengths)

    expected = 1 + 0.2 * coordinates - 0.1 * coordinates**3
    expected_gradient = 2 * np.asarray(
        [np.sum(expected * coordinates), np.sum(expected * coordinates**3)]
    )
    assert np.allclose(jax.jit(evaluate)(coeffs), expected)
    gradient = jax.jit(jax.grad(lambda c: np.sum(evaluate(c) ** 2)))(coeffs)
    assert np.allclose(gradient, expected_gradient)


@pytest.mark.parametrize("kind", ["polynomial", "basis", "blackbody"])
def test_spectral_batches_match_independent_models(kind):
    wavelengths = np.linspace(0.8e-6, 1.2e-6, 7)
    if kind == "polynomial":
        make = lambda x: dlu.Spectrum(
            dlu.Polynomial(1, x, transform=zdx.Map(s=1e6, b=-1))
        )
        parameters = np.asarray([[1.0, 0.2], [1.0, -0.2]])
    elif kind == "basis":
        vectors = np.stack([np.ones(7), np.linspace(-0.2, 0.2, 7)])
        make = lambda x: dlu.Spectrum(dlu.Basis(vectors, x, axes=1))
        parameters = np.asarray([[1.0, 0.5], [1.0, -0.5]])
    else:
        make = dlu.Blackbody
        parameters = np.asarray([4000.0, 8000.0])
    model = make(parameters)
    values = assert_jittable(lambda value: value.sample(wavelengths), model)
    expected = np.stack([make(p).sample(wavelengths) for p in parameters])
    assert values.shape == (2, 7)
    assert np.allclose(values, expected)


@pytest.mark.parametrize("shape", [(), (1, 1), (2, 1), (2, 3)])
def test_blackbody_numerical_function_retains_general_query_batches(shape):
    wavelengths = np.linspace(0.6e-6, 0.8e-6, prod(shape)).reshape(shape)
    query = np.atleast_1d(wavelengths)
    temperature = np.asarray([4000.0, 8000.0])
    h, c, k = 6.62607015e-34, 299792458.0, 1.380649e-23
    exponent = h * c / (query * k * temperature.reshape((2,) + (1,) * query.ndim))
    expected = 2 * c / (query**4 * np.expm1(exponent))
    actual = jax.jit(dlu.blackbody)(wavelengths, temperature)
    assert actual.shape == (2,) + query.shape
    assert np.allclose(actual / expected, 1, rtol=1e-5)


def test_blackbody_density_and_temperature_derivatives():
    wavelengths = np.linspace(0.5e-6, 1e-6, 8)
    model = dlu.Blackbody(5800)
    h, c, k = 6.62607015e-34, 299792458.0, 1.380649e-23
    expected = 2 * c / (wavelengths**4 * np.expm1(h * c / (wavelengths * k * 5800)))
    values = model.sample(wavelengths)
    assert np.allclose(values / expected, 1, rtol=1e-5)
    evaluate = lambda t: model.set(temperature=t).sample(wavelengths) / values
    assert np.all(jax.jacrev(evaluate)(model.temperature) > 0)
    assert np.all(np.isfinite(jax.jacfwd(jax.jacrev(evaluate))(model.temperature)))


@pytest.mark.parametrize("kind", ["polynomial", "blackbody"])
def test_shared_sampling_flux_and_coordinate_namespace(kind):
    axis = dlu.Axis([500, 600, 700], unit="nm")
    model = (
        dlu.Blackbody(5800, bandpass=axis, flux=7)
        if kind == "blackbody"
        else dlu.Spectrum(
            dlu.Polynomial(1, [1, 0.2], transform=zdx.Map(s=1e7, b=-6)),
            bandpass=axis,
            flux=7,
        )
    )
    assert isinstance(model, dlu.BaseSpectrum)
    assert isinstance(model.resolve(), type(model))
    values = assert_jittable(lambda value: value.sample(), model)
    assert np.allclose(values, model.sample(axis.axes[0]))
    spatial = dlu.Grid(n=3, d=1.0, unit="m")
    query = np.asarray([550e-9, 650e-9])
    assert np.allclose(
        model.sample(query, grid=spatial, coordinates=spatial.coordinates),
        model.sample(query),
    )
    # Axis inputs still locate curve samples. Integration needs explicit bin edges.
    assert np.allclose(model(query), model.sample(query))
    band = dlu.BandPass(wavelengths=[500, 600, 700], unit="nm")
    model = model.set(bandpass=band)
    result = assert_jittable(lambda value: value(), model)
    wl, weights = result.wavelengths, result.value
    assert np.allclose(wl, axis.axes[0])
    assert np.allclose(weights.sum(), 7)
    with pytest.raises(TypeError, match="BandPass"):
        model.integrate(query)


def test_integrated_samples_use_weights_and_preserve_counts():
    model = dlu.Spectrum.from_weights([500e-9, 700e-9], [2.0, 3.0])
    assert type(model) is dlu.Weights
    assert np.allclose(
        assert_jittable(lambda x: x.weights(), model)[1], np.asarray([2, 3])
    )
    assert np.allclose((model * 2).integrate(), 10)
    with pytest.raises(ValueError, match="sampling"):
        dlu.Photometry(model)(dlu.BandPass(600e-9, 200e-9, n=5))


def test_generic_basis_owns_its_existing_coefficient_axes():
    vectors = np.arange(24.0).reshape(2, 3, 4)
    coefficients = np.ones((2, 3))
    model = dlu.Spectrum(dlu.Basis(vectors, coefficients, axes=2))
    assert np.allclose(model.sample(np.arange(4.0)), vectors.sum((0, 1)))


def test_single_calibration_sample_can_be_used_on_a_monochromatic_bandpass():
    band = dlu.BandPass(600e-9)
    model = dlu.Spectrum(2.0, bandpass=band, flux=7.0)
    assert np.allclose(assert_jittable(lambda x: x(band).value, model), 7.0)
    query = np.asarray([500e-9, 600e-9, 700e-9])
    assert np.allclose(model.sample(query), np.asarray([0.0, 2.0, 0.0]))
    response = dlu.Filter(0.5, knots=band)
    assert np.allclose(response.sample(query), np.asarray([0.0, 0.5, 0.0]))


@pytest.mark.parametrize("kind", [dlu.Spectrum, dlu.Filter])
def test_shared_curve_sampling_supports_native_defaults_and_fixed_bases(kind):
    band = dlu.BandPass(range=(1.0, 4.0), n=3)
    basis = dlu.Basis(np.eye(3), coeffs=[1.0, 3.0, 1.0])
    curve = kind(basis, bandpass=band)
    assert np.allclose(curve.sample(), np.array([1.0, 3.0, 1.0]))
    assert np.allclose(curve.sample(band), curve.sample())
    query = np.array([1.5, 2.0, 2.5])
    expected = np.array([1.0, 2.5, 3.0])
    actual = assert_jittable(lambda c: c.sample(query), curve)
    assert np.allclose(actual, expected)
    derivative = jax.grad(lambda c: curve.set("value.coeffs", c).sample(query).sum())(
        basis.coeffs
    )
    assert np.allclose(derivative, np.array([1.375, 1.75, -0.125]))


def test_filter_integral_is_effective_bandwidth_in_metres():
    band = dlu.BandPass(range=(500, 700), n=3, unit="nm")
    response = dlu.Filter(0.5, bandpass=band)
    integral = assert_jittable(lambda curve: curve.integrate(), response)
    assert np.allclose(integral / 1e-9, 100.0)
    assert np.allclose(response.integrate(band) / 1e-9, 100.0)
    assert np.allclose(response.sample(band), 0.5)
    with pytest.raises(TypeError, match="BandPass"):
        response.integrate(band.wavelengths)


def test_filter_integral_uses_native_curve_and_requested_boundaries():
    native = dlu.BandPass(wavelengths=[500, 550, 600, 650, 700], unit="nm")
    response = dlu.Filter([0.0, 0.5, 1.0, 0.5, 0.0], bandpass=native, method="linear")
    full = dlu.BandPass(range=(500, 700), n=1, unit="nm")
    middle = dlu.BandPass(range=(550, 650), n=1, unit="nm")
    assert np.allclose(response.integrate(full) / 1e-9, 100.0)
    assert np.allclose(response.integrate(middle) / 1e-9, 75.0)
    density = dlu.Spectrum(response.value, native, flux=None, method="linear")
    assert np.allclose(response.integrate(full) / 1e-9, density.integrate(full) / 1e-9)


def test_filter_integral_preserves_response_batches_and_gradients():
    band = dlu.BandPass(range=(500, 700), n=3, unit="nm")
    response = dlu.Filter(np.array([[0.5] * 3, [0.2] * 3]), bandpass=band)
    actual = assert_jittable(lambda curve: curve.integrate(), response)
    assert np.allclose(actual / 1e-9, np.array([100.0, 40.0]))
    scalar = dlu.Filter(0.5, bandpass=band)
    derivative = jax.jit(jax.grad(lambda x: scalar.set(value=x).integrate()))(
        scalar.value
    )
    assert np.allclose(derivative / 1e-9, 200.0)
