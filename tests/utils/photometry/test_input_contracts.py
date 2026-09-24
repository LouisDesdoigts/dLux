"""Validation tails for spectral sampling, units and integrated photon counts.

Numerical integration is tested separately. These cases exercise user-facing
errors at the boundaries between densities, sampling and already-integrated weights.
"""

import pytest

import dLux.utils as dlu


@pytest.mark.parametrize(
    "inputs,error,message",
    [
        ({"range": [500]}, TypeError, "two fixed Python endpoints"),
        ({"range": (500, 700), "wl": 600}, ValueError, "separately"),
        ({"min": 500, "max": 700, "wl": 600}, ValueError, "separately"),
        ({}, ValueError, "Supply min/max"),
        ({"wavelengths": [500, 600], "n": 2}, ValueError, "Supply wavelengths or"),
        ({"wavelengths": []}, ValueError, "non-empty vector"),
        ({"range": (500, 500), "n": 2}, ValueError, "positive wavelength interval"),
        (
            {"wavelengths": dlu.Axis(n=3, d=100, c=600, unit="nm"), "unit": "nm"},
            ValueError,
            "owns its own unit",
        ),
    ],
)
def test_bandpass_constructor_validation_tails(inputs, error, message):
    with pytest.raises(error, match=message):
        dlu.BandPass(**inputs)


@pytest.mark.parametrize("curve", [dlu.Spectrum, dlu.Filter])
def test_curve_sampling_has_one_owner(curve):
    band = dlu.BandPass(range=(500, 700), n=3, unit="nm")
    with pytest.raises(ValueError, match="not both"):
        curve([1, 2, 1], bandpass=band, wavelengths=[500, 600, 700])
    with pytest.raises(ValueError, match="bandpass owns its unit"):
        curve([1, 2, 1], bandpass=band, unit="nm")
    with pytest.raises(ValueError, match="one.dimensi"):
        curve([1, 2, 1], bandpass=dlu.Grid(3, d=100, unit="nm"))


def test_filter_legacy_knots_spelling_cannot_add_a_second_sampling():
    axis = dlu.Axis(n=3, d=100, c=600, unit="nm")
    with pytest.raises(ValueError, match="separately"):
        dlu.Filter([1, 2, 1], bandpass=axis, knots=axis)
    with pytest.raises(ValueError, match="own their wavelength unit"):
        dlu.Filter([1, 2, 1], knots=axis, unit="nm")


@pytest.mark.parametrize("sampling", [[], [[5e-7, 6e-7]], dlu.Grid(3, d=1, unit="nm")])
def test_curve_queries_require_one_wavelength_axis(sampling):
    with pytest.raises(ValueError, match="one dimension|non-empty vector"):
        dlu.Filter(0.5).sample(sampling)


@pytest.mark.parametrize("curve", [dlu.Spectrum, dlu.Filter])
def test_curve_values_need_sampling_and_a_matching_sample_count(curve):
    with pytest.raises(ValueError, match="Supply wavelengths"):
        curve([1, 2, 3]).sample()
    with pytest.raises(ValueError, match="wavelength sample axis"):
        curve([1, 2, 3]).sample([5e-7, 6e-7])


def test_integrated_weights_keep_their_wavelength_axis_and_quantity():
    band = dlu.BandPass(range=(500, 700), n=3, unit="nm")
    weights = dlu.Weights([1, 2, 3], band)
    for values in (1, [1, 2]):
        with pytest.raises(ValueError, match="wavelength axis"):
            dlu.Weights(values, band)
        with pytest.raises(ValueError, match="wavelength sample axis"):
            dlu.Photometry().apply(band, values)
    with pytest.raises(ValueError, match="matching wavelength samples"):
        weights + dlu.Weights([1], dlu.BandPass(600, unit="nm"))
    with pytest.raises(TypeError):
        weights + 1


def test_polynomial_and_blackbody_setup_errors():
    band = dlu.BandPass(range=(500, 700), n=3, unit="nm")
    with pytest.raises(ValueError, match="finite reference BandPass"):
        dlu.PolySpectrum(dlu.BandPass(600, unit="nm"))
    with pytest.raises(ValueError, match="coefficient axis"):
        dlu.PolySpectrum(band, coeffs=1)
    with pytest.raises(ValueError, match="positive integer"):
        dlu.PolySpectrum(band, degree=0)
    with pytest.raises(ValueError, match="Supply wavelengths"):
        dlu.Blackbody(5000).sample()


def test_companion_fluxes_require_one_primary_and_a_companion_vector():
    with pytest.raises(ValueError, match="primary must be a scalar"):
        dlu.ContrastFlux(primary=[1, 2])
    with pytest.raises(ValueError, match="companion vector"):
        dlu.ContrastFlux(contrast=[[1, 2]])
