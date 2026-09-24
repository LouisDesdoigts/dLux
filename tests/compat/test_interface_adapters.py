"""Released names remain views of the current numerical objects."""

from dataclasses import fields

import equinox as eqx
import jax
import jax.numpy as np
import pytest

import dLux as dl
import dLux.utils as dlu
from dLux import compat as old
from dLux.compat.modelling import result

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def released_jitter_kernel(sigma, size, oversample):
    """Independent released Gaussian sampling and block integration."""
    coordinates = np.linspace(-5, 5, size * oversample)
    x, y = np.meshgrid(coordinates, coordinates)
    density = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    density /= density.sum()
    return density.reshape(size, oversample, size, oversample).sum((1, 3))


@pytest.mark.parametrize("size", [3, 9])
@pytest.mark.parametrize("oversample", [1, 2, 3])
@pytest.mark.parametrize("sigma", [0.3, 1.3])
def test_jitter_preserves_released_kernel(sigma, size, oversample):
    layer = old.ApplyJitter(sigma, size, oversample)
    expected = released_jitter_kernel(sigma, size, oversample)
    actual = eqx.filter_jit(lambda op: op.kernel.resolve())(layer)
    assert np.allclose(actual, expected, rtol=2e-6, atol=2e-7)
    assert np.allclose(layer.sigma, sigma)
    assert np.allclose(layer.kernel.profile.std, sigma)


def test_jitter_conversion_tracks_parameter_and_sampling_updates():
    layer = old.ApplyJitter(1.3)
    changed = layer.set(
        ["kernel.profile.std", "kernel.kernel_size", "kernel.oversample"],
        [0.8, (5, 5), (2, 2)],
    )
    values = eqx.filter_jit(lambda op: op.kernel.resolve())(changed)
    assert np.allclose(values, released_jitter_kernel(0.8, 5, 2), atol=2e-7)
    assert np.allclose(changed.sigma, 0.8)
    assert np.allclose(layer.sigma, 1.3)

    # A second spatial moment responds to fitted width; compare both derivatives.
    x, y = np.meshgrid(np.arange(9) - 4, np.arange(9) - 4)
    radius_squared = x**2 + y**2
    response = lambda sigma: (
        layer.set("kernel.profile.std", sigma).kernel.resolve() * radius_squared
    ).sum()
    reference = lambda sigma: (
        released_jitter_kernel(sigma, 9, 3) * radius_squared
    ).sum()
    derivative = jax.jit(jax.grad(response))(1.3)
    assert derivative > 0
    assert np.allclose(derivative, jax.grad(reference)(1.3), rtol=2e-6)


@pytest.mark.parametrize("n", [8, 9])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_fft_layer_matches_released_phase_convention(n, center, inverse):
    grid = dlu.Grid(n, diam=1, unit="rad" if inverse else "m")
    x, y = grid.coordinates
    phasor = (1 + x) * np.exp(0.3j * y)
    wavefront = dl.Wavefront(600e-9, grid, phasor)
    pad = 1 if n % 2 else 2
    operation = old.FFT(pad=pad, crop=pad, center=center, inverse=inverse)

    # Independent discrete Fourier sum, including the released half-pixel ramps.
    size = n * pad
    sign = -1 if inverse else 1
    if center and size % 2 == 0:
        phasor *= np.exp(-sign * 1j * np.pi * (x + y) / (size / n))
    padding = (size - n) // 2
    phasor = np.pad(phasor, padding)
    samples = np.arange(size) - size // 2
    matrix = np.exp(-sign * 2j * np.pi * np.outer(samples, samples) / size)
    expected = matrix @ phasor @ matrix.T / size
    if center and size % 2 == 0:
        coordinates = np.arange(size) - (size - 1) / 2
        xy = coordinates[:, None] + coordinates[None, :]
        expected *= np.exp(-sign * 1j * np.pi * xy / size)
    expected = expected[padding : padding + n, padding : padding + n]

    actual = eqx.filter_jit(lambda layer, field: layer(field))(operation, wavefront)
    assert np.allclose(actual.phasor, expected, rtol=2e-5, atol=2e-6)


def test_aperture_views_follow_current_parameter_updates():
    aperture = old.RegPolyAperture(6, 0.4, softening=1.5)
    changed = aperture.set("transmission.diameter", 1.2)
    assert np.allclose(aperture.rmax, 0.4)
    assert np.allclose(changed.rmax, 0.6)
    assert aperture.nsides == 6
    assert np.allclose(aperture.softness, 1.5)
    assert not aperture.occulting
    assert aperture.transformation is None
    assert {field.name for field in fields(aperture)} == {
        field.name for field in fields(dl.Optic)
    }
    circular = old.CircularAperture(0.4)
    assert circular.nsides == 0
    assert np.allclose(circular.radius, 0.4)
    assert np.allclose(circular.set("transmission.diameter", 1.2).radius, 0.6)
    assert old.RectangularAperture(0.4, 0.5).nsides == 4
    assert old.CircularAperture(0.4, softening=None).softness is None
    spider = old.Spider(0.05, [0, 90])
    for name in ("extent", "nsides"):
        with pytest.raises(TypeError, match="Spiders"):
            getattr(spider, name)


def test_detector_and_rotation_views_do_not_store_duplicate_parameters():
    jitter = old.ApplyJitter(0.3)
    assert np.allclose(jitter.sigma, 0.3)
    assert np.allclose(jitter.set("kernel.profile.std", 0.6).sigma, 0.6)
    saturation = old.ApplySaturation(5)
    assert saturation.set(limit=3).threshold == 3
    bias = old.AddConstant(2)
    assert bias.set(bias=4).value == 4
    rotation = old.Rotate(0.2)
    assert np.allclose(rotation.angle, 0.2)
    assert np.allclose(rotation.set("transformation.rotation", 0.4).angle, 0.4)
    for obj, base in [
        (jitter, dl.ConvolveLayer),
        (saturation, dl.SaturationLayer),
        (bias, dl.BiasLayer),
        (rotation, dl.InterpolateLayer),
    ]:
        assert {field.name for field in fields(obj)} == {
            field.name for field in fields(base)
        }


@pytest.mark.parametrize("value", [0.3, [1, 2], np.ones((2, 2, 2))])
def test_released_pixel_response_requires_an_image(value):
    with pytest.raises(ValueError, match="two-dimensional"):
        old.ApplyPixelResponse(value)


@pytest.mark.parametrize("distorted", [False, True])
def test_calculate_generates_released_coordinates(distorted):
    if distorted:
        operation = old.DistortedCoords(2, np.ones((2, 5)) * 0.02)
    else:
        operation = old.CoordTransform([0.1, -0.2], 0.3, [0.8, 1.1])
    coordinates = old.utils.pixel_coords(8, 2.0)
    expected = operation(coordinates)
    assert np.allclose(operation.calculate(8, 2.0), expected)
    actual = eqx.filter_jit(lambda op: op.calculate(8, 2.0))(operation)
    assert np.allclose(actual, expected, atol=2e-7)


@pytest.mark.parametrize("return_wf", [False, True])
def test_propagation_weights_require_the_wavelength_shape(return_wf):
    field = dl.Wavefront(np.array([500e-9, 600e-9]), dlu.Grid(4, diam=1, unit="m"))
    with pytest.raises(ValueError, match="same shape as wavelengths"):
        result(field, np.ones(3), return_wf=return_wf)
    output = result(field, np.array([1, 3]), return_wf=return_wf)
    data = output.intensity if return_wf else output
    assert np.allclose(data.sum(), 4)


@pytest.mark.parametrize("name", ["Source", "PointSource", "ResolvedSource"])
def test_explicit_compatibility_sources_warn_even_with_current_spectra(name):
    kwargs = {"spectrum": dlu.Spectrum()}
    if name == "ResolvedSource":
        kwargs.update(distribution=np.ones((3, 3)), profile_grid=dlu.Grid(3, 0.1))
    with pytest.warns(DeprecationWarning, match=f"compat.{name}") as caught:
        getattr(old, name)(**kwargs)
    assert caught[0].filename == __file__


def test_released_resolved_keywords_warn_at_the_canonical_boundary():
    with pytest.warns(DeprecationWarning, match="distribution/position"):
        dl.ResolvedSource(distribution=np.ones((3, 3)), profile_grid=dlu.Grid(3, 0.1))


def test_released_modulation_can_expand_a_singleton_field():
    field = dl.Wavefront(600e-9, dlu.Grid(1, d=0.1, unit="m"))
    transmission = np.arange(16, dtype=float).reshape(4, 4) / 16
    optic = old.Optic(transmission=transmission)
    output = eqx.filter_jit(lambda layer, wf: layer(wf))(optic, field)
    assert np.allclose(output.phasor, field.phasor * transmission)
    assert output.grid.n == (4, 4)
    assert np.array_equal(output.grid.d, field.grid.d)
    with pytest.raises(ValueError, match="broadcast"):
        dl.Optic(transmission=transmission)(field)


def test_released_polynomial_keeps_metre_coefficients_and_live_updates():
    wavelengths = np.array([500e-9, 600e-9, 700e-9])
    spectrum = old.PolySpectrum(wavelengths, [1, 1e6])
    expected = 1 + 1e6 * wavelengths
    assert np.allclose(spectrum.weights, expected / expected.sum())
    changed = spectrum.set(coefficients=np.array([2.0, 0.0]))
    assert np.allclose(changed.weights, np.ones(3) / 3)
    assert np.allclose(
        eqx.filter_jit(lambda value: value.sample())(spectrum), spectrum.weights
    )
    stored = old.Spectrum(wavelengths).set(weights=np.array([1.0, 2.0, 3.0]))
    assert np.allclose(stored.normalise().weights, np.array([1, 2, 3]) / 6)


@pytest.mark.parametrize("kind", ["fft", "asm"])
def test_released_propagator_accepts_size_only_sampling(kind):
    field = dl.Wavefront(600e-9, dlu.Grid(8, diam=1, unit="m"))
    if kind == "fft":
        operation = old.FFTPropagator([old.ABCDConjugatePlane(1)], old.CoordSpec(n=12))
        native = dlu.FFTPropagator(dlu.FFTSpec(n=12, c=0), abcd=dlu.ABCD(fraunhofer=1))
    else:
        operation = old.ASMPropagator(1, old.CoordSpec(n=12, c=None))
        native = dlu.FreeSpace(1, dlu.ResizeSpec(n=12))
    output = operation(field)
    expected = field.propagate(native)
    assert output.phasor.shape == (12, 12)
    assert np.allclose(output.phasor, expected.phasor)
