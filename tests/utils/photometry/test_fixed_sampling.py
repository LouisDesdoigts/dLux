"""Fixed sampling, polynomial convenience and fine-to-optical integration."""

import equinox as eqx
import jax
import jax.numpy as np
import pytest

import dLux.utils as dlu


def test_bandpass_is_static_metadata_and_changes_recompile():
    band = dlu.BandPass(min=500.0, max=700.0, n=3, unit="nm")
    assert band.__dataclass_fields__["range"].metadata["static"]
    calls = []

    @eqx.filter_jit
    def evaluate(sampling):
        calls.append(sampling.max)
        return sampling.wavelengths

    assert evaluate(band).shape == (3,)
    evaluate(dlu.BandPass(600.0, 200.0, n=3, unit="nm"))
    assert len(calls) == 1
    evaluate(dlu.BandPass(min=500.0, max=800.0, n=3, unit="nm"))
    assert len(calls) == 2
    assert jax.tree.leaves(band) == []


def test_polyspectrum_fixes_intercept_and_reference_coordinates():
    reference = dlu.BandPass(min=500.0, max=700.0, n=9, unit="nm")
    spectrum = dlu.PolySpectrum(reference, coeffs=[0.2, 0.1], flux=12)
    u = np.linspace(-8 / 9, 8 / 9, 9)
    expected = 1 + 0.2 * u + 0.1 * u**2
    sample = eqx.filter_jit(lambda value: value.sample(reference))(spectrum)
    assert np.allclose(sample, expected)
    assert np.allclose(spectrum.sample(reference.wavelengths[2:6]), expected[2:6])
    assert np.allclose(dlu.PolySpectrum(reference, degree=3).sample(reference), 1)
    grad = jax.grad(lambda c: spectrum.set("value.coeffs", c).sample(reference)[-1])(
        spectrum.value.coeffs
    )
    assert np.allclose(grad, np.array([8 / 9, (8 / 9) ** 2]))
    for n in (3, 7, 15):
        query = dlu.BandPass(min=500.0, max=700.0, n=n, unit="nm")
        assert np.allclose(spectrum(query).integrate(), 12)


@pytest.mark.parametrize("method", ["linear", "cubic", "cubic2"])
def test_curve_integral_uses_the_selected_interpolant(method):
    knots = np.linspace(1e-6, 3e-6, 9)
    u = (knots - 1e-6) / 2e-6
    values = u**2
    edges = np.array([1e-6, 2e-6, 3e-6])
    actual = eqx.filter_jit(lambda y: dlu.integrate_curve(y, knots, edges, method))(
        values
    )
    if method == "linear":
        expected = np.array(
            [np.trapezoid(values[:5], knots[:5]), np.trapezoid(values[4:], knots[4:])]
        )
        assert np.allclose(actual / 1e-6, expected / 1e-6)
    else:
        # Cubic2 reproduces this quadratic; local cubic has endpoint approximation.
        expected = np.array([1 / 12, 7 / 12]) * 1e-6
        error = np.max(np.abs(actual - expected))
        trap = np.trapezoid(values, knots)
        assert error < abs(trap - 2e-6 / 3)
        if method == "cubic2":
            assert np.allclose(actual / 1e-6, expected / 1e-6, atol=2e-6)


def test_narrow_response_is_not_lost_between_optical_nodes():
    native = dlu.BandPass(wavelengths=np.linspace(500, 700, 9), unit="nm")
    # Triangle at 550 nm, between the optical nodes at 500 and 700 nm.
    response = dlu.Filter([0, 1, 2, 1, 0, 0, 0, 0, 0], native)
    model = dlu.Photometry(dlu.Spectrum(flux=10), filter=response, qe=0.5)
    for n in (1, 2, 3, 7):
        optical = dlu.BandPass(min=500.0, max=700.0, n=n, unit="nm")
        weights = eqx.filter_jit(lambda p: p(optical))(model)
        assert weights.value.shape == (n,)
        assert np.allclose(weights.integrate(), 2.5)


def test_spectral_coefficients_remain_differentiable_through_binning():
    native = dlu.BandPass(min=500.0, max=700.0, n=7, unit="nm")
    shape = dlu.Basis(np.stack((np.ones(7), np.linspace(-6 / 7, 6 / 7, 7))), [1.0, 0.2])
    spectrum = dlu.Spectrum(shape, native, flux=10, method="cubic2")
    model = dlu.Photometry(
        spectrum, filter=dlu.Filter(np.linspace(1 / 14, 13 / 14, 7), native)
    )
    optical = dlu.BandPass(min=500.0, max=700.0, n=2, unit="nm")
    fn = lambda c: model.set("spectrum.value.coeffs", c)(optical).integrate()
    derivative = jax.jit(jax.grad(fn))(shape.coeffs)
    assert np.allclose(derivative, np.array([-1 / 3, 5 / 3]), rtol=2e-5, atol=2e-5)


def test_magnitude_helpers_use_a_common_reference_flux():
    flux = np.array([1.0, 10.0, 100.0])
    magnitude = jax.jit(dlu.flux2mag)(flux, 10.0)
    assert np.allclose(magnitude, np.array([2.5, 0.0, -2.5]))
    assert np.allclose(dlu.mag2flux(magnitude, 10.0), flux)


def test_single_sample_native_curve_retains_its_finite_support():
    native = dlu.BandPass(min=500.0, max=700.0, unit="nm")
    optical = dlu.BandPass(min=500.0, max=700.0, n=3, unit="nm")
    for spectrum in (
        dlu.Spectrum([1.0], native, flux=10),
        dlu.PolySpectrum(native, coeffs=[0.2], flux=10),
    ):
        model = dlu.Photometry(spectrum, filter=dlu.Filter([0.5], native))
        weights = eqx.filter_jit(lambda m: m(optical))(model)
        assert np.allclose(weights.integrate(), 5)


@pytest.mark.parametrize("flux", [None, 12.0])
@pytest.mark.parametrize("method", ["linear", "cubic2"])
def test_stored_explicit_and_spectrum_only_calls_share_bin_integrals(flux, method):
    band = dlu.BandPass(range=(1.0, 4.0), n=3)
    spectrum = dlu.Spectrum([1.0, 3.0, 1.0], band, flux=flux, method=method)
    model = dlu.Photometry(spectrum)

    # Integrate straight segments or the exact quadratic 3 - 2 * (x - 2.5)**2.
    if method == "linear":
        expected = np.array([1.0, 2.5, 1.0])
    else:
        expected = np.array([5 / 6, 17 / 6, 5 / 6])
    if flux is not None:
        expected = expected * flux / 4.5
    for operation in (spectrum, model):
        assert np.allclose(operation().value, expected)
        assert np.allclose(operation(band).value, expected)
        assert np.allclose(operation.integrate(), expected.sum())
        evaluate = eqx.filter_jit(lambda op: (op().value, op(band).value))
        for values in evaluate(operation):
            assert np.allclose(values, expected)

    # Filter and QE losses remain absolute after incident flux is assigned.
    detected = dlu.Photometry(spectrum, filter=0.5, qe=0.2)
    assert np.allclose(detected().value, expected * 0.1)
    assert np.allclose(detected(band).value, expected * 0.1)


def test_wavelength_calls_and_sample_return_unscaled_point_values():
    band = dlu.BandPass(range=(1.0, 4.0), n=3)
    spectrum = dlu.Spectrum([1.0, 3.0, 1.0], band, flux=100)
    model = dlu.Photometry(spectrum, filter=0.5, qe=0.2)
    wavelengths = np.array([1.5, 2.0, 2.5])

    for operation, expected in (
        (spectrum, np.array([1.0, 2.5, 3.0])),
        (model, np.array([0.1, 0.25, 0.3])),
    ):
        assert np.allclose(operation(wavelengths), expected)
        assert np.allclose(operation.sample(wavelengths), expected)
        assert np.allclose(
            eqx.filter_jit(lambda op: op(wavelengths))(operation), expected
        )
        assert np.allclose(operation.sample(band), operation(band.wavelengths))
        with pytest.raises(TypeError, match="BandPass"):
            operation.integrate(wavelengths)
    with pytest.raises(TypeError, match="BandPass"):
        model.integrate(wavelengths)


def test_integrated_inputs_are_not_resampled_or_integrated_again():
    band = dlu.BandPass(range=(1.0, 4.0), n=3)
    model = dlu.Photometry(dlu.Weights([1.0, 3.0, 1.0], band), filter=0.5)
    assert np.allclose(model().value, np.array([0.5, 1.5, 0.5]))
    assert np.allclose(model.integrate(), 2.5)
    with pytest.raises(TypeError, match="density"):
        model.sample(band.wavelengths)


@pytest.mark.parametrize("flux", [0.0, 12.0])
def test_flux_gradients_and_empty_shapes_survive_normalisation(flux):
    band = dlu.BandPass(range=(500.0, 700.0), n=3, unit="nm")
    spectrum = dlu.Spectrum([1.0, 2.0, 1.0], band, flux=flux)
    for response, throughput in ((None, 1.0), (0.25, 0.25)):
        model = dlu.Photometry(spectrum, filter=response)
        total = lambda value: model.set("spectrum.flux", value).integrate()
        assert np.allclose(jax.jit(total)(np.asarray(flux)), flux * throughput)
        assert np.allclose(jax.jit(jax.grad(total))(np.asarray(flux)), throughput)

        empty = model.set("spectrum.value", np.zeros(3))
        values = eqx.filter_jit(lambda m: m().value)(empty)
        assert np.array_equal(values, np.zeros(3))


def test_default_cubic2_sampling_and_integration_reproduce_a_quadratic():
    knots = np.linspace(1.0, 3.0, 5)
    values = knots**2
    query = np.array([1.25, 1.75, 2.25, 2.75])
    band = dlu.BandPass(range=(1.0, 3.0), n=2)
    spectrum = dlu.Spectrum(values, knots, flux=None)
    response = dlu.Filter(values, knots)

    # The default must reproduce x² between samples, unlike straight segments.
    for curve in (spectrum, response):
        assert curve.method == "cubic2"
        assert np.allclose(curve.sample(query), query**2)
        assert np.allclose(curve.integrate(band), 26 / 3)
    assert np.allclose(dlu.sample_curve(values, knots, query), query**2)
    expected = np.array([7 / 3, 19 / 3])
    assert np.allclose(dlu.integrate_curve(values, knots, band.edges), expected)

    # The default quadrature must also integrate the product x² * x² exactly.
    result = dlu.bin_spectrum(spectrum, band.edges, knots, filter=response)
    assert np.allclose(result.value, np.array([31 / 5, 211 / 5]))
