"""Density integration, absolute throughput and wavelength sampling."""

import jax
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux.utils as dlu

from tests.helpers import assert_jittable


@pytest.mark.parametrize("slot", ["spectrum", "filter", "qe"])
@pytest.mark.parametrize("kind", ["map", "basis"])
def test_fixed_expressions_bind_to_the_calculation_bandpass(slot, kind):
    band = dlu.BandPass(range=(500, 700), n=3, unit="nm")
    values = np.asarray([0.2, 0.5, 0.8])
    if kind == "map":
        definition = zdx.Map(x=values)
    else:
        definition = dlu.Basis(np.eye(3), coeffs=values)

    def model(value):
        inputs = dict(spectrum=dlu.Spectrum(), filter=None, qe=None)
        inputs[slot] = dlu.Spectrum(value) if slot == "spectrum" else dlu.Filter(value)
        return dlu.Photometry(**inputs)

    expected = model(values)(band).value
    actual = assert_jittable(lambda curve: curve(band).value, model(definition))
    assert np.allclose(actual, expected)
    if slot == "spectrum":
        standalone = dlu.Spectrum(definition)(band).value
        assert np.allclose(standalone, expected)


def test_bandpass_equal_bins_and_optical_sampling():
    band = dlu.BandPass(600, 200, n=5, unit="nm")
    wl, widths = assert_jittable(lambda x: x.sample(), band)
    assert np.allclose(wl / 1e-9, np.asarray([520, 560, 600, 640, 680]))
    assert np.allclose(widths / 1e-9, np.asarray([40, 40, 40, 40, 40]))
    assert np.allclose(band.wavelengths / 1e-9, np.asarray([520, 560, 600, 640, 680]))
    with pytest.raises(ValueError, match="regularly spaced"):
        dlu.BandPass(wavelengths=[500, 520, 700], unit="nm")


def test_bandpass_and_axis_units_agree():
    axis = dlu.Axis(n=3, d=100, c=600, unit="nm")
    band = dlu.BandPass(wavelengths=axis)
    assert np.allclose(band.wavelengths / 1e-9, np.asarray([500, 600, 700]))
    assert np.allclose(band.widths, np.full(3, 100e-9))
    assert np.allclose(band.d_si, axis.d_si)
    assert np.allclose(band.c_si, axis.c_si)
    mono = dlu.BandPass(600, unit="nm")
    assert mono.widths is None
    assert np.allclose(dlu.BandPass(600, 20, unit="nm").widths / 1e-9, np.asarray([20]))


@pytest.mark.parametrize("field", ["min", "max", "wl", "delta_wl"])
def test_bandpass_requires_fixed_numbers(field):
    kwargs = (
        {"min": 500.0, "max": 700.0}
        if field in ("min", "max")
        else {"wl": 600.0, "delta_wl": 200.0}
    )
    for value in (np.asarray(kwargs[field]), zdx.StateRef("sampling_value")):
        with pytest.raises(TypeError, match="fixed Python"):
            dlu.BandPass(**dict(kwargs, **{field: value}), n=3)


def test_bandpass_centres_and_widths_are_derived_from_static_range():
    band = dlu.BandPass(range=(500.0, 700.0), n=4, unit="nm")
    wavelengths, widths = assert_jittable(lambda value: value.sample(), band)
    assert np.allclose(wavelengths / 1e-9, np.array([525, 575, 625, 675]))
    assert np.array_equal(widths, np.full(4, 50e-9))
    assert np.allclose(band.edges / 1e-9, np.array([500, 550, 600, 650, 700]))


def test_irregular_raw_calibration_and_quadrature_remain_numerical_inputs():
    wavelengths = np.asarray([500, 520, 700]) * 1e-9
    spectrum = dlu.Spectrum(
        [1.0, 2.0, 4.0], bandpass=wavelengths, flux=None, method="linear"
    )
    query = np.asarray([510, 610]) * 1e-9
    assert np.allclose(spectrum.sample(query), np.asarray([1.5, 3.0]))
    integrated = assert_jittable(
        lambda samples: dlu.bin_spectrum(
            spectrum, samples[np.array([0, -1])], samples
        ).integrate(),
        wavelengths,
    )
    assert np.allclose(integrated / 1e-9, 570)


def test_normalisation_precedes_absolute_filter_and_qe_losses():
    band = dlu.BandPass(600e-9, 200e-9, n=5)
    photometry = dlu.Photometry(3.0, filter=0.5, qe=0.2)
    absolute = photometry.set("spectrum.flux", None)
    assert np.allclose(absolute.integrate(band) / 1e-9, 60)
    assert np.allclose(photometry.integrate(band), 0.1)
    weights = assert_jittable(lambda model: model(band).value, photometry)
    assert np.allclose(weights, np.full(5, 0.02))


def test_response_is_independent_of_the_spectrum_and_retains_numerical_batches():
    class UnavailableSpectrum(zdx.Expression):
        def evaluate(self, **context):
            raise AssertionError("Response calculation must not evaluate a spectrum.")

    wavelengths = np.asarray([500e-9, 600e-9, 700e-9])
    curves = np.asarray([[0.2, 0.4, 0.8], [0.1, 0.3, 0.5]])
    model = dlu.Photometry(UnavailableSpectrum(), filter=curves, qe=0.5)
    actual = assert_jittable(lambda value: value.response(wavelengths), model)
    assert np.allclose(actual, curves * 0.5)
    weights = np.asarray([1.0, 2.0, 3.0])
    assert np.allclose(model.apply(wavelengths, weights), curves * 0.5 * weights)


def test_missing_responses_have_unit_transmission_and_preserve_weight_scale():
    wavelengths = np.asarray([500e-9, 600e-9])
    weights = np.asarray([2.0, 5.0])
    model = dlu.Photometry()
    assert np.array_equal(model.response(wavelengths), np.ones(2))
    transmitted = model.apply(wavelengths, weights)
    assert np.array_equal(transmitted, weights)
    assert transmitted.dtype == weights.dtype


def test_curves_keep_calibration_and_query_sampling_separate():
    values = dlu.Basis(np.eye(3), [1.0, 2.0, 3.0])
    spectrum = dlu.Spectrum(
        values, bandpass=np.asarray([500, 600, 700]) * 1e-9, method="linear"
    )
    query = np.asarray([450, 550, 650, 750]) * 1e-9
    assert np.allclose(spectrum.sample(query), np.asarray([0, 1.5, 2.5, 0]))
    same = dlu.Spectrum(
        values, bandpass=dlu.Axis([500, 600, 700], unit="nm"), method="linear"
    )
    assert np.allclose(same.sample(query), spectrum.sample(query))


def test_spectral_weights_are_masses_and_never_reintegrated():
    masses = dlu.Spectrum.from_weights([500e-9, 700e-9], [2.0, 3.0])
    model = dlu.Photometry(masses, filter=0.5)
    assert np.allclose(model().value, np.asarray([1.0, 1.5]))
    with pytest.raises(ValueError, match="sampling"):
        model(dlu.BandPass(600e-9, 200e-9, n=3))
    with pytest.raises(ValueError, match="bandwidth"):
        dlu.Photometry(dlu.Spectrum(flux=None))(dlu.BandPass(600e-9))


def test_bin_preserves_piecewise_linear_integral():
    band = dlu.BandPass(wavelengths=np.asarray([1.0, 2.0, 3.0]))
    model = dlu.Photometry(dlu.Spectrum([1.0, 3.0, 1.0], flux=None, method="linear"))
    result = assert_jittable(
        lambda m: dlu.bin_spectrum(m.spectrum, [1.0, 2.0, 3.0], band), model
    )
    assert np.allclose(result.value, np.asarray([2.0, 2.0]))
    # Explicit integration nodes cover [1, 3]; the BandPass extends to [0.5, 3.5].
    assert np.allclose(result.value.sum(), 4.0)
    assert np.allclose(model.integrate(band), 4.5)
    assert np.allclose(
        dlu.bin_spectrum(model.spectrum, [0.0, 1.0, 3.0, 4.0], band).value,
        np.asarray([0, 4, 0]),
    )


def test_expression_context_and_spectral_gradients():
    band = dlu.BandPass(600e-9, 100e-9, n=5)
    model = dlu.Photometry(dlu.Blackbody(zdx.StateRef("temperature")), filter=0.4)
    state = zdx.State(temperature=5500.0)
    weights = model(band, state=state).value
    assert np.all(np.isfinite(weights))
    derivative = jax.jit(
        jax.grad(lambda t: model(band, state=state.set(temperature=t)).value[0])
    )(np.asarray(5500.0))
    assert np.isfinite(derivative)
    assert derivative != 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"wl": 600e-9, "n": 3},
        {"wl": 600e-9, "delta_wl": -100e-9},
        {"wavelengths": [700e-9, 500e-9]},
        {"wl": 600e-9, "n": 0},
    ],
)
def test_invalid_sampling_has_concrete_feedback(kwargs):
    with pytest.raises(ValueError):
        dlu.BandPass(**kwargs).sample()
