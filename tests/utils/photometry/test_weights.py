"""Native spectral curves feed one independently chosen calculation BandPass."""

import equinox as eqx
import jax
import jax.numpy as np
import pytest

import dLux as dl
import dLux.utils as dlu


@pytest.mark.parametrize("n", [1, 3, 7])
def test_interval_representations_and_integration(n):
    bounds = dlu.BandPass(min=500, max=700, n=n, unit="nm")
    centre = dlu.BandPass(600, 200, n=n, unit="nm")
    assert np.allclose(bounds.wavelengths, centre.wavelengths, rtol=1e-6, atol=0)
    assert np.allclose(bounds.widths, centre.widths, rtol=1e-6, atol=0)
    if n > 1:
        explicit = dlu.BandPass(wavelengths=bounds.wavelengths / 1e-9, unit="nm")
        assert np.allclose(bounds.widths, explicit.widths, rtol=1e-6, atol=0)
    values = 2 + (bounds.wavelengths / 1e-9 - 500) / 200
    spectrum = dlu.Spectrum(values, bandpass=bounds, flux=None)
    assert np.allclose(spectrum.integrate() / 1e-9, 500, rtol=2e-6)


def test_native_samples_basis_and_responses_share_calculation_grid():
    native = dlu.BandPass(min=500, max=700, n=3, unit="nm")
    target = dlu.BandPass(min=500, max=700, n=5, unit="nm")
    basis = dlu.Basis(np.array([[1, 1, 1], [0, 1, 0]]), coeffs=[1.0, 1.0])
    spectrum = dlu.Spectrum(basis, native, flux=12, method="linear")
    response = dlu.Filter(
        [0, 1, 0], wavelengths=[500, 600, 700], unit="nm", method="linear"
    )
    qe = dlu.Filter(0.5)

    incident = spectrum(target)
    detected = eqx.filter_jit(lambda s: qe(response(s(target))))(spectrum)
    assert isinstance(detected, dlu.Weights)
    assert detected.bandpass.n == 5
    assert spectrum.bandpass.n == 3
    assert isinstance(spectrum.value, dlu.Basis)
    # The triangular basis integrates to [-8, 16, 34, 16, -8] nm;
    # the flat basis contributes 40 nm per bin. Normalise their sum to 12 photons.
    expected = 12 * np.array([32, 56, 74, 56, 32]) / 250
    assert np.allclose(incident.value, expected, rtol=2e-6, atol=1e-6)
    assert np.allclose(
        detected.value, expected * np.array([0.2, 0.6, 1, 0.6, 0.2]) * 0.5, atol=1e-6
    )

    # Basis coefficients remain live through interpolation and photon allocation.
    derivative = jax.grad(
        lambda c: qe(response(spectrum.set("value.coeffs", c)(target))).integrate()
    )(basis.coeffs)
    assert np.allclose(derivative, np.array([-288 / 625, 288 / 625]), atol=2e-6)


def test_different_sources_can_be_added_after_shared_sampling():
    query = dlu.BandPass(min=500, max=700, n=7, unit="nm")
    first = dlu.Spectrum([1, 2, 1], wavelengths=[500, 600, 700], unit="nm", flux=3)
    second = dlu.Blackbody(5800, flux=7)
    combined = eqx.filter_jit(lambda a, b: a(query) + b(query))(first, second)
    assert combined.value.shape == (7,)
    assert np.allclose(combined.integrate(), 10)
    assert np.allclose(dlu.Filter(0.4)(combined).integrate(), 4)


def test_prepared_response_is_an_explicit_snapshot():
    native = dlu.BandPass(min=500, max=700, n=3, unit="nm")
    target = dlu.BandPass(min=500, max=700, n=9, unit="nm")
    curve = dlu.Filter([0.2, 0.8, 0.4], native)
    prepared = curve.prepare(target)
    incident = dlu.Blackbody(6000)(target)
    assert prepared.value.shape == (9,)
    assert np.allclose(prepared(incident).value, curve(incident).value)
    changed = curve.set(value=np.ones(3))
    assert np.allclose(changed(incident).integrate(), 1)
    assert prepared(incident).integrate() < 1
    assert np.allclose(changed.prepare(target)(incident).value, incident.value)


def test_fixed_bandpass_comparison_skips_interpolation(monkeypatch):
    from dLux.utils.photometry import spectral

    target = dlu.BandPass(min=500, max=700, n=5, unit="nm")
    same = dlu.BandPass(wl=600, delta_wl=200, n=5, unit="nm")
    spectrum = dlu.Spectrum(np.ones(5), target, flux=3)
    response = dlu.Filter(np.ones(5) * 0.4, target)

    def unexpected(*args, **kwargs):
        raise AssertionError("Equivalent sampling must bypass interpolation.")

    monkeypatch.setattr(spectral, "Interp", unexpected)
    assert np.allclose(eqx.filter_jit(lambda s: s.sample(same))(spectrum), 1)
    assert np.allclose(eqx.filter_jit(lambda r: r.sample(same))(response), 0.4)

    # Photometry must preserve the same shortcut through its public entry points.
    model = dlu.Photometry(spectrum, filter=response, qe=0.5)
    assert np.allclose(eqx.filter_jit(lambda m: m.response(same))(model), 0.2)
    assert np.allclose(eqx.filter_jit(lambda m: m.sample(same))(model), 0.2)
    assert np.allclose(model.apply(same, np.ones(5)), 0.2)
    integrated = model.set(spectrum=dlu.Weights(np.ones(5), same))
    assert np.allclose(eqx.filter_jit(lambda m: m().value)(integrated), 0.2)


def test_weights_do_not_multiply_counts_or_add_different_samples():
    first = dlu.Weights([1, 2], dlu.BandPass(min=500, max=700, n=2, unit="nm"))
    second = dlu.Weights([1, 2], dlu.BandPass(min=550, max=750, n=2, unit="nm"))
    with pytest.raises(ValueError, match="matching wavelength"):
        first + second
    with pytest.raises(TypeError, match="response"):
        first * second
    with pytest.raises(TypeError, match="Weights"):
        dlu.Filter(0.5)(first.bandpass)
    with pytest.raises(ValueError, match="both min and max"):
        dlu.BandPass(min=500, n=2)
    with pytest.raises(ValueError, match="separately"):
        dlu.BandPass(600, 200, min=500, max=700)


def test_imager_photometry_returns_weights_on_its_calculation_bandpass():
    native = dlu.BandPass(min=500, max=700, n=3, unit="nm")
    calculation = dlu.BandPass(min=500, max=700, n=5, unit="nm")
    spectrum = dlu.Spectrum([1, 2, 1], native, flux=10)
    response = dlu.Filter([0, 0.8, 0], native)
    imager = dl.Imager(
        dl.Grid(6, diam=1, unit="m"),
        dl.Grid(8, d=0.1, unit="arcsec"),
        source=dl.Source(spectrum),
        bandpass=calculation,
        filter=response,
        detector=dl.Detector(qe=0.5),
    )
    result = eqx.filter_jit(lambda model: model.photometry())(imager)
    expected = dlu.Photometry(spectrum, filter=response, qe=0.5)(calculation)
    assert isinstance(result, dlu.Weights)
    assert result.bandpass.n == 5
    assert np.allclose(result.value[0], expected.value)
    assert np.all(np.isfinite(imager.model().data))


def test_static_bin_matching_is_checked_inside_jit():
    band = dlu.BandPass(range=(500, 700), n=2, unit="nm")
    same = dlu.BandPass(wl=600, delta_wl=200, n=2, unit="nm")
    add = eqx.filter_jit(lambda first, second: first + second)
    first = dlu.Weights([1.0, 2.0], band)
    assert np.allclose(
        add(first, dlu.Weights([3.0, 4.0], same)).value, np.array([4, 6])
    )
    metres = dlu.BandPass(range=(500e-9, 700e-9), n=2)
    assert np.allclose(
        add(first, dlu.Weights([3.0, 4.0], metres)).value, np.array([4, 6])
    )
    different = dlu.BandPass(range=(550, 750), n=2, unit="nm")
    with pytest.raises(ValueError, match="matching wavelength"):
        add(first, dlu.Weights([3.0, 4.0], different))

    # Equal monochromatic centres do not make different integration bins equal.
    narrow = dlu.Weights([1.0], dlu.BandPass(range=(550, 650), unit="nm"))
    wide = dlu.Weights([1.0], dlu.BandPass(range=(500, 700), unit="nm"))
    with pytest.raises(ValueError, match="matching wavelength"):
        add(narrow, wide)


def test_raw_vector_addition_uses_the_callers_matching_sampling():
    wavelengths = np.array([500e-9, 550e-9, 700e-9])
    first = dlu.Weights([1.0, 2.0, 3.0], wavelengths)
    second = dlu.Weights([3.0, 2.0, 1.0], wavelengths)
    result = eqx.filter_jit(lambda a, b: a + b)(first, second)
    assert np.allclose(result.value, 4)
