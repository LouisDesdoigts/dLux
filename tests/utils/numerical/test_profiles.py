"""Spatial density sampling, pixel integration and independent normalisation."""

import equinox as eqx
import jax
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux as dl
import dLux.utils as dlu
from dLux.utils.numerical.profiles import Profile


class ChromaticDensity(zdx.Expression):
    """A fixed physical profile whose width depends on wavelength."""

    def evaluate(self, *, coordinates, wavelengths, **context):
        widths = wavelengths[..., None, None]
        radius2 = np.square(coordinates).sum(-3)
        return np.exp(-radius2 / (2 * widths**2))


def test_sampled_profiles_keep_independent_spatial_normalisation():
    grid = dlu.Grid(n=(5, 3), d=(0.2, 0.4))
    arrays = np.arange(1.0, 31.0).reshape(2, 3, 5)
    profile = Profile(arrays, grid)

    actual = eqx.filter_jit(lambda p: p.sample())(profile)

    assert actual.shape == arrays.shape
    assert np.allclose(actual, arrays / arrays.sum((-2, -1), keepdims=True))
    assert np.allclose(actual.sum((-2, -1)), 1)
    assert np.allclose(Profile(dlu.Field(arrays, grid)).sample(), actual)


def test_source_resolution_retains_sampled_profile_and_grid():
    grid = dlu.Grid(n=(5, 3), d=(0.02, 0.03), unit="arcsec")
    samples = np.arange(1.0, 16.0).reshape(3, 5)
    source = dl.ResolvedSource(profile=Profile(samples, grid))

    prepared = eqx.filter_jit(lambda s: s.resolve())(source)

    assert isinstance(prepared.profile, Profile)
    assert eqx.tree_equal(prepared.profile.grid, grid)
    assert np.array_equal(prepared.profile.profile, samples)
    assert np.allclose(prepared.profile.sample(), source.profile.sample())


def test_profile_resolution_prepares_parameters_without_sampling():
    width = zdx.Map(x=0.3, s=2)
    source = dl.ResolvedSource(profile=Profile(dlu.Gaussian(width)))

    prepared = source.resolve()

    assert isinstance(prepared.profile, Profile)
    assert isinstance(prepared.profile.profile, dlu.Gaussian)
    assert np.allclose(prepared.profile.profile.std, 0.6)
    grid = dlu.Grid(n=5, d=0.4)
    assert np.allclose(prepared.profile.sample(grid), source.profile.sample(grid))


def test_unnormalised_density_integrates_si_pixel_area():
    grid = dlu.Grid(n=3, d=20.0, unit="mm")
    arrays = np.ones((3, 3)) * 7

    actual = Profile(arrays, grid, normalise=False).sample()

    assert np.allclose(actual, arrays * 0.02**2)


def test_expression_waits_for_sampling_and_uses_runtime_grid():
    profile = Profile(dlu.Gaussian(0.7))
    target = dlu.Grid(n=(5, 3), d=(0.2, 0.4), c=(0.3, -0.1))
    coordinates = target.coordinates
    expected = np.exp(-np.square(coordinates).sum(0) / (2 * 0.7**2))
    expected /= expected.sum()

    assert isinstance(profile.resolve(), Profile)
    with pytest.raises(zdx.Unresolved, match="requires a grid"):
        profile.sample()
    assert np.allclose(profile.sample(target), expected)


def test_stored_grid_samples_expressions_before_resampling():
    source = dlu.Grid(n=(7, 5), d=(0.4, 0.6))
    x, y = source.coordinates
    sampled = dlu.Basis(np.stack([np.ones_like(x), x, y]), [3, 0.2, 0.3])
    profile = Profile(sampled, source, oversample=2)
    target = dlu.Grid(n=(3, 3), d=(0.2, 0.3), c=(0.1, -0.2))
    x, y = target.coordinates
    expected = 3 + 0.2 * x + 0.3 * y
    expected /= expected.sum()

    actual = eqx.filter_jit(lambda p, g: p.sample(g))(profile, target)

    assert np.allclose(actual, expected, rtol=2e-6, atol=1e-7)


def test_subpixel_integration_matches_direct_midpoint_quadrature():
    grid = dlu.Grid(n=(5, 3), d=(0.4, 0.7), c=(0.1, -0.2))
    profile = Profile(dlu.Gaussian([0.5, 0.9], mean=[0.2, -0.1]), oversample=(2, 3))
    x, y = grid.coordinates
    expected = np.zeros(grid.shape)
    for i in range(2):
        for j in range(3):
            dx = (i - 0.5) * 0.4 / 2
            dy = (j - 1) * 0.7 / 3
            exponent = ((x + dx - 0.2) / 0.5) ** 2 + ((y + dy + 0.1) / 0.9) ** 2
            expected += np.exp(-exponent / 2)
    expected /= expected.sum()

    actual = eqx.filter_jit(lambda p, g: p.sample(g))(profile, grid)

    assert np.allclose(actual, expected, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("width", [0.0, 1e-3])
def test_narrow_gaussians_retain_finite_pixel_masses(width):
    grid = dlu.Grid(n=5, d=1)
    profile = Profile(dlu.Gaussian(width), oversample=2)

    actual = eqx.filter_jit(lambda p: p.sample(grid))(profile)

    expected = np.zeros((5, 5)).at[2, 2].set(1)
    assert np.all(np.isfinite(actual))
    assert np.allclose(actual, expected)


def test_sampled_density_resampling_and_integration():
    source = dlu.Grid(n=(7, 5), d=(0.4, 0.6))
    x, y = source.coordinates
    profile = Profile(3 + 0.2 * x + 0.3 * y, source, oversample=3, normalise=False)
    target = dlu.Grid(n=(3, 3), d=(0.2, 0.3), c=(0.1, -0.2))
    x, y = target.coordinates

    actual = eqx.filter_jit(lambda p, g: p.sample(g))(profile, target)

    # Midpoint quadrature is exact for a linear density in every output pixel.
    expected = (3 + 0.2 * x + 0.3 * y) * 0.2 * 0.3
    assert np.allclose(actual, expected, rtol=2e-6, atol=1e-7)


def test_chromatic_context_keeps_one_normalisation_per_wavelength():
    grid = dlu.Grid(n=(7, 5), d=(0.3, 0.4))
    wavelengths = np.array([0.3, 0.6, 0.9])
    profile = Profile(ChromaticDensity(), oversample=2)

    actual = eqx.filter_jit(lambda p, w: p.sample(grid, wavelengths=w))(
        profile, wavelengths
    )
    expected = np.stack([profile.sample(grid, wavelengths=w) for w in wavelengths])

    assert actual.shape == (3, 5, 7)
    assert np.allclose(actual, expected, rtol=2e-6, atol=1e-7)
    assert np.allclose(actual.sum((-2, -1)), 1)
    assert actual[0, 2, 3] > actual[-1, 2, 3]


def test_profile_parameter_gradient_matches_normalised_gaussian_moment():
    grid = dlu.Grid(n=(7, 5), d=(0.3, 0.4))
    x = grid.coordinates[0]
    width, mean = 0.6, 0.2

    def centroid(value):
        centre = zdx.Map(x=np.array([value, -0.1]))
        profile = Profile(dlu.Gaussian(width, mean=centre))
        return (profile.sample(grid) * x).sum()

    gradient = jax.jit(jax.grad(centroid))(mean)
    weights = Profile(dlu.Gaussian(width, mean=[mean, -0.1])).sample(grid)
    expected = ((weights * x**2).sum() - (weights * x).sum() ** 2) / width**2

    assert gradient > 0
    assert np.allclose(gradient, expected, rtol=2e-6, atol=1e-7)


def test_jitter_delegates_the_same_local_profile_sampling():
    grid = dlu.Grid(n=(13, 11), d=(0.08, 0.12), c=(3, -4), unit="arcsec")
    gaussian = dlu.Gaussian([0.13, 0.21], mean=[0.02, -0.01])
    jitter = dlu.Jitter(gaussian, kernel_size=(5, 3), oversample=(2, 3), unit="arcsec")
    assert isinstance(jitter, dlu.Unitful)
    local = dlu.Grid(n=(11, 7), d=(0.08, 0.12))
    expected = Profile(gaussian, oversample=(2, 3)).sample(local)

    actual = eqx.filter_jit(lambda p: p.resolve(grid=grid, oversample=2))(jitter)

    assert np.allclose(actual, expected, rtol=2e-6, atol=1e-7)


def test_pixel_jitter_matches_physical_width_without_scaling_kernel_values():
    grid = dlu.Grid(n=9, d=0.1, unit="arcsec")
    pixel = dlu.Jitter(1.0, kernel_size=5)
    physical = dlu.Jitter(0.1, kernel_size=5, unit="arcsec")
    evaluate = eqx.filter_jit(lambda value: value.resolve(grid=grid))

    assert isinstance(pixel, dlu.Unitful)
    assert np.allclose(evaluate(pixel), evaluate(physical), atol=1e-6)
    assert np.allclose(evaluate(pixel).sum(), 1.0)


def test_sampled_profiles_require_declared_sampling():
    with pytest.raises(ValueError, match="requires a grid"):
        Profile(np.ones((3, 3)))
    field = dlu.Field(np.ones((3, 3)), dlu.Grid(n=(3, 3), d=1))
    with pytest.raises(ValueError, match="already supplies"):
        Profile(field, grid=field.grid)


@pytest.mark.parametrize(
    "operation,error",
    [
        (lambda: dlu.Profile(dlu.Gaussian(1), grid=dlu.Axis(n=3)), TypeError),
        (lambda: dlu.Profile(np.ones(3), grid=dlu.Grid(3, 1)), ValueError),
        (lambda: dlu.Profile(dlu.Gaussian(1)).sample(dlu.Axis(n=3)), TypeError),
        (lambda: dlu.Profile(dlu.Gaussian(1)).sample(dlu.Grid(n=3)), zdx.Unresolved),
        (
            lambda: dlu.Profile(zdx.Map(), normalise=False).sample(dlu.Grid(3, 1)),
            zdx.Unresolved,
        ),
        (
            lambda: dlu.Profile(zdx.Map(x=np.ones((2, 2)))).sample(dlu.Grid(3, 1)),
            ValueError,
        ),
        (lambda: dlu.Jitter([1.0, 2.0, 3.0]), ValueError),
        (lambda: dlu.Jitter(1.0, unit="1/m"), ValueError),
        (lambda: dlu.Jitter(1.0, kernel_size=4), ValueError),
        (lambda: dlu.Jitter(1.0, unit="m").evaluate(), zdx.Unresolved),
        (
            lambda: dlu.Jitter(1.0, unit="m").evaluate(grid=dlu.Grid(3, 1, unit="rad")),
            ValueError,
        ),
    ],
)
def test_profile_validation_tails(operation, error):
    """Coverage tails: required sampling, density dimensions and jitter units."""
    with pytest.raises(error):
        operation()
