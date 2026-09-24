"""Declarative source inputs and their astrometric/brightness parametrisations."""

import jax
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux as dl
import dLux.utils as dlu
from tests.helpers import assert_jittable


def test_source_defaults_and_position_promotion():
    source = dl.Source()
    assert np.array_equal(source.positions, np.zeros((1, 2)))
    assert np.asarray(source.flux) == 1
    assert isinstance(source.spectrum, dlu.Spectrum)
    assert np.array_equal(dl.Source(positions=[1, 2]).positions, [[1, 2]])
    assert dl.Source(dlu.Spectrum(flux=None)).flux is None


@pytest.mark.parametrize("kwargs, factor", [({}, np.pi / 648000), ({"unit": "rad"}, 1)])
def test_point_source_retains_a_single_position_definition(kwargs, factor):
    source = dl.PointSource(position=[1.0, 2.0], flux=3.0, **kwargs)
    assert isinstance(source.positions, dlu.Point)
    resolved = assert_jittable(lambda value: value.resolve(), source)
    assert np.allclose(resolved.positions / factor, np.array([[1.0, 2.0]]))
    assert resolved.flux == 3.0
    assert dl.PointSource(dlu.Spectrum(flux=None)).flux is None


def test_binary_brightness_and_parameter_gradients():
    source = dl.BinarySource(
        position=[1.0, 2.0],
        separation=3.0,
        unit="rad",
        primary_flux=10.0,
        contrast=0.2,
        log=False,
    )
    resolved = assert_jittable(lambda value: value.resolve(), source)
    assert np.allclose(resolved.positions, np.array([[1.0, 2.0], [1.0, 5.0]]))
    assert np.allclose(resolved.flux, np.array([10.0, 2.0]))
    separation = jax.grad(
        lambda v: source.set("positions.separation", v).resolve().positions[1, 1]
    )(np.asarray(3.0))
    ratio = jax.grad(lambda v: source.set("flux.contrast", v).resolve().flux.sum())(
        np.asarray(0.2)
    )
    assert np.allclose(separation, 1.0)
    assert np.allclose(ratio, 10.0)
    assert np.allclose(
        dlu.ContrastFlux(10, 0.2, log=False).resolve(), np.array([10, 2])
    )


def test_source_keeps_expressions_until_their_context_is_available():
    class Positions(zdx.Expression):
        def evaluate(self, *, positions, **context):
            return positions

    source = dl.Source(positions=Positions())
    assert isinstance(source.resolve().positions, Positions)
    result = source.resolve(positions=np.ones((2, 2)))
    assert np.array_equal(result.positions, np.ones((2, 2)))


def test_resolved_source_owns_explicit_angular_sampling():
    grid = dlu.Grid(n=3, d=1.0, unit="arcsec")
    array = np.ones((3, 3))
    sampled = dl.ResolvedSource(profile=array, profile_grid=grid)
    field = dl.ResolvedSource(profile=dlu.Field(array, grid))
    parametric = dl.ResolvedSource(profile=dlu.Gaussian(1e-5))
    assert sampled.profile.grid.unit == field.profile.grid.unit == "arcsec"
    assert isinstance(parametric.profile, dlu.Profile)
    assert isinstance(parametric.profile.profile, dlu.Gaussian)


def test_scene_keeps_named_immutable_paths():
    scene = dl.Scene(
        {
            "star": dl.Source(),
            "pair": dl.Scene(
                {"binary": dl.BinarySource(primary_flux=2.0)}, position=[0, 1]
            ),
        }
    )
    resolved = assert_jittable(lambda value: value.resolve(), scene)
    assert np.allclose(
        resolved.sources["pair"].sources["binary"].flux, np.array([2.0, 2.0])
    )
    updated = scene.set("sources.star.flux", np.asarray(3.0))
    assert updated.sources["star"].flux == 3
    assert scene.sources["star"].flux == 1


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: dl.Source(flux=dlu.Flux(1, unit="unsupported")),
        lambda: dl.Source(positions=np.zeros((2, 3))),
        lambda: dl.Source(positions=np.empty((0, 2))),
        lambda: dl.Source(positions=np.zeros((2, 2)), flux=np.ones(3)),
        lambda: dl.ResolvedSource(profile=np.ones((3, 3))),
        lambda: dl.ResolvedSource(
            profile=np.ones((3, 3)), profile_grid=dlu.Grid(n=3, d=1, unit="m")
        ),
        lambda: dl.Scene({}),
        lambda: dl.Scene({"star": object()}),
    ],
)
def test_definition_shape_and_sampling_validation(constructor):
    with pytest.raises((TypeError, ValueError)):
        constructor()


def test_binary_retains_the_primary_flux_unit():
    source = dl.BinarySource(primary_flux=dlu.Flux(10.0), contrast=0.2, log=False)
    assert source.spectrum.flux.unit == "photons"
    values = assert_jittable(lambda s: s.spectrum.flux.value.resolve(), source)
    assert np.allclose(values, np.array([10.0, 2.0]))


def test_source_definition_validation_tails():
    """Malformed declarations fail at construction, without runtime guards."""
    profile = dlu.Profile(dlu.Gaussian(1e-6))
    assert dl.ResolvedSource(profile=profile).profile is profile
    for call in (
        lambda: dl.Source(flux=np.ones((2, 2))),
        lambda: dl.ResolvedSource(profile=profile, profile_grid=dlu.Grid(3, 1e-6)),
        lambda: dl.Scene({"star": dl.Source()}, position=np.ones(3)),
    ):
        with pytest.raises(ValueError):
            call()
    with pytest.raises(TypeError):
        dl.Scene({1: dl.Source()})


@pytest.mark.parametrize("brightness", [20.0, dlu.Flux(20.0)])
def test_binary_preserves_supplied_spectrum_brightness(brightness):
    spectrum = dlu.Spectrum(flux=brightness)
    source = dl.BinarySource(spectrum, contrast=0.25, log=False)
    flux = source.spectrum.flux
    values = flux.value.resolve() if isinstance(flux, dlu.Flux) else flux.resolve()
    assert np.allclose(values, np.array([20.0, 5.0]))
    replaced = dl.BinarySource(spectrum, contrast=0.25, log=False, primary_flux=8.0)
    assert np.allclose(replaced.spectrum.flux.resolve(), np.array([8.0, 2.0]))
    derivative = jax.jit(
        jax.grad(
            lambda primary: dl.BinarySource(
                spectrum, primary_flux=primary, contrast=0.25, log=False
            )
            .spectrum.flux.resolve()
            .sum()
        )
    )(np.asarray(8.0))
    assert np.allclose(derivative, 1.25)


def test_sources_require_densities_and_binary_brightness_is_explicit():
    band = dlu.BandPass(range=(500.0, 700.0), n=3, unit="nm")
    with pytest.raises(TypeError, match="spectral density"):
        dl.Source(dlu.Weights(np.ones(3), band))
    with pytest.raises(ValueError, match="primary_flux"):
        dl.BinarySource(dlu.Spectrum(flux=None))
    source = dl.BinarySource(dlu.Spectrum(flux=None), primary_flux=3.0)
    assert np.allclose(source.spectrum.flux.resolve(), np.array([3.0, 3.0]))


def test_sources_live_with_systems_and_binary_defaults_are_angular_and_logarithmic():
    from dLux.systems.sources import Source

    assert dl.Source is Source
    source = dl.BinarySource(separation=1, pa=90, contrast=-2)
    resolved = assert_jittable(lambda source: source.resolve(), source)
    assert np.allclose(
        resolved.positions[1], np.array([dlu.unit_factor("arcsec"), 0]), atol=1e-12
    )
    assert np.allclose(resolved.flux, np.array([1, 0.01]))
    cartesian = dl.Source(positions=dlu.Positions([[0, 0], [1, 0]]), flux=[1, 0.01])
    assert np.allclose(cartesian.resolve_positions(), resolved.positions, atol=1e-12)


def test_unitful_point_and_linear_binary_defaults():
    source = dl.PointSource(position=[1000, 0], unit="mas")
    assert np.allclose(
        source.resolve_positions(),
        np.array([[dlu.unit_factor("arcsec"), 0]]),
        atol=1e-12,
    )
    assert np.array_equal(
        dl.BinarySource(log=False).spectrum.flux.resolve(), np.ones(2)
    )
