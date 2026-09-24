"""Spectral integration preserves sharp support edges and native interpolants."""

import equinox as eqx
import jax
import jax.numpy as np
import pytest

import dLux.utils as dlu
from tests.helpers import assert_jittable


@pytest.mark.parametrize("native_count", [1, 2, 5])
@pytest.mark.parametrize("optical_count", [1, 3, 7])
@pytest.mark.parametrize("method", ["cubic", "cubic2"])
def test_constant_cubic_response_integrates_only_its_support(
    native_count, optical_count, method
):
    optical = dlu.BandPass(range=(400, 800), n=optical_count, unit="nm")
    native = dlu.BandPass(range=(550, 650), n=native_count, unit="nm")
    response = dlu.Filter(np.ones(native_count), native, method=method)
    model = dlu.Photometry(dlu.Spectrum(flux=1), filter=response)

    actual = assert_jittable(lambda value: value(optical).value, model)
    edges = optical.edges / 1e-9
    overlap = np.maximum(np.minimum(edges[1:], 650) - np.maximum(edges[:-1], 550), 0)
    assert np.allclose(actual, overlap / 400, rtol=3e-6, atol=2e-7)
    assert np.allclose(actual.sum(), 0.25, rtol=3e-6)


@pytest.mark.parametrize("flux", [None, 12.0])
@pytest.mark.parametrize("response_range", [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3, 4)])
def test_incident_flux_uses_spectrum_support_before_response_losses(
    flux, response_range
):
    optical = dlu.BandPass(range=(0.25, 4.25), n=4)
    native = dlu.BandPass(range=(1, 3), n=3)
    response = dlu.BandPass(range=response_range, n=2)
    model = dlu.Photometry(
        dlu.Spectrum(np.full(3, 2.0), native, flux=flux, method="cubic2"),
        filter=dlu.Filter([0.5, 0.5], response, method="cubic2"),
        qe=0.4,
    )

    actual = assert_jittable(lambda value: value(optical).value, model)
    low, high = max(1, response_range[0]), min(3, response_range[1])
    overlap = np.maximum(
        np.minimum(optical.edges[1:], high) - np.maximum(optical.edges[:-1], low), 0
    )
    density = 2.0 if flux is None else flux / 2
    assert np.allclose(actual, overlap * density * 0.5 * 0.4, atol=1e-6)


@pytest.mark.parametrize("method, degree", [("linear", 1), ("cubic2", 3)])
def test_product_integral_preserves_each_native_polynomial(method, degree):
    native = dlu.BandPass(range=(1, 2), n=5)
    optical = dlu.BandPass(range=(1, 2), n=2)
    values = (native.wavelengths - 1) ** degree
    model = dlu.Photometry(
        dlu.Spectrum(values, native, flux=None, method=method),
        filter=dlu.Filter(values, native, method=method),
        qe=dlu.Filter(values, native, method=method),
    )

    power = 3 * degree + 1
    expected = np.diff((optical.edges - 1) ** power) / power
    actual = assert_jittable(lambda value: value(optical).value, model)
    assert np.allclose(actual, expected, rtol=2e-5, atol=1e-7)


def test_response_amplitude_gradient_retains_fractional_bandwidth():
    optical = dlu.BandPass(range=(400, 800), n=3, unit="nm")
    native = dlu.BandPass(range=(550, 650), n=2, unit="nm")
    model = dlu.Photometry(
        dlu.Spectrum(flux=12), filter=dlu.Filter([0.4, 0.4], native, method="cubic2")
    )
    total = lambda amplitude: model.set(
        "filter.value", np.full(2, amplitude)
    ).integrate(optical)
    assert np.allclose(eqx.filter_jit(total)(0.4), 1.2, rtol=3e-6)
    assert np.allclose(jax.jit(jax.grad(total))(0.4), 3.0, rtol=3e-6)


def test_raw_native_knots_and_traced_mesh_keep_support_boundaries():
    spectrum = dlu.Spectrum(1.0, flux=None)
    response = dlu.Filter([0.5, 0.5], knots=np.array([1.5, 2.5]), method="cubic2")
    integrate = lambda knots: dlu.bin_spectrum(
        spectrum, [1.0, 2.0, 3.0], knots, filter=response, method="cubic2"
    ).value
    actual = assert_jittable(integrate, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(actual, np.array([0.25, 0.25]))


def test_monochromatic_counts_retain_response_losses_without_a_bin_width():
    model = dlu.Photometry(dlu.Spectrum(flux=3), filter=0.5, qe=0.4)
    band = dlu.BandPass(600, unit="nm")
    result = assert_jittable(lambda value: value(band), model)
    assert np.allclose(result.value, np.array([0.6]))
    assert np.allclose((2 * result).integrate(), 1.2)
    assert (2 * result).bandpass.same_sampling(result.bandpass)


@pytest.mark.parametrize(
    "operation, error, message",
    [
        (lambda: dlu.integrate_curve([1], [1], [1, 2]), ValueError, "matching"),
        (
            lambda: dlu.integrate_spectrum(dlu.Weights([1], [1]), dlu.BandPass(1)),
            TypeError,
            "density",
        ),
        (
            lambda: dlu.bin_spectrum(dlu.Weights([1], [1]), [1, 2], [1, 2]),
            TypeError,
            "density",
        ),
        (
            lambda: dlu.bin_spectrum(dlu.Spectrum(), [1, 2], [1]),
            ValueError,
            "at least two",
        ),
        (
            lambda: dlu.bin_spectrum(dlu.Spectrum(), [1, 2], [1, 2], method="invalid"),
            ValueError,
            "method",
        ),
    ],
)
def test_integration_input_errors(operation, error, message):
    """Coverage tails: reject counts-as-density and incomplete integration inputs."""
    with pytest.raises(error, match=message):
        operation()
