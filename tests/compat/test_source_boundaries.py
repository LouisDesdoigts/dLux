"""Released point weights remain separate from canonical Imager photometry."""

import equinox as eqx
import jax.numpy as np
import pytest

import dLux as dl
import dLux.utils as dlu
from dLux import compat as old


@pytest.mark.parametrize("angular", [False, True])
def test_released_sources_use_integrated_point_weights_at_the_compatibility_boundary(
    angular,
):
    if angular:
        system = old.AngularOpticalSystem(6, 1.0, [dl.Optic(normalise=True)], 6, 0.1)
    else:
        relay = dlu.Propagator(dlu.Grid(6, d=0.1, unit="arcsec"))
        system = old.LayeredOpticalSystem(6, 1.0, [dl.Optic(normalise=True), relay])
    source = old.PointSource([5e-7, 7e-7], weights=[1.0, 3.0], flux=8.0)
    expected = system.propagate(np.array([5e-7, 7e-7]), weights=np.array([2.0, 6.0]))
    actual = eqx.filter_jit(lambda optics, star: optics.model(star))(system, source)
    assert np.allclose(actual, expected, rtol=2e-5, atol=1e-7)


def test_telescope_preserves_irregular_point_weights_and_applies_losses_once():
    system = old.AngularOpticalSystem(6, 1.0, [dl.Optic(normalise=True)], 6, 0.1)
    wavelengths = np.array([500e-9, 570e-9, 800e-9])
    source = old.PointSource(wavelengths, weights=[1.0, 2.0, 6.0], flux=9.0)
    detector = dl.Detector(qe=dlu.Filter(0.25))
    telescope = old.Telescope(system.set(filter=dlu.Filter(0.5)), source, detector)

    # These are point masses on irregular wavelengths, with no inferred bin widths.
    expected_weights = np.array([1.0, 2.0, 6.0]) * 0.5 * 0.25
    actual = eqx.filter_jit(lambda model: model.photometry().value)(telescope)
    assert np.allclose(actual, expected_weights[None, :])
    expected = system.propagate(wavelengths, weights=expected_weights)
    image = eqx.filter_jit(lambda model: model.model().data)(telescope)
    assert np.allclose(image, expected, rtol=2e-5, atol=1e-7)


def test_canonical_imager_still_requires_density_sampling_for_photometry():
    system = dl.Imager(dlu.Grid(6, diam=1.0, unit="m"), dlu.Grid(6, 0.1, unit="arcsec"))
    source = old.PointSource([500e-9, 700e-9], weights=[1.0, 3.0])
    assert system.psf(source).data.shape == (1, 2, 6, 6)
    with pytest.raises(TypeError, match="BandPass"):
        system.model(source)


def test_released_point_positions_remain_radians():
    position = np.array([1e-7, -2e-7])
    source = old.PointSource([600e-9], position=position)
    assert np.allclose(source.resolve_positions(), position[None, :], rtol=1e-6, atol=0)
    current = dl.PointSource(position=position)
    expected = dlu.convert(position, "arcsec", "rad")
    assert np.allclose(
        current.resolve_positions(), expected[None, :], rtol=1e-6, atol=0
    )
