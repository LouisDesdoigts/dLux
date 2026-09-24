"""Released Zernike methods preserve their arguments, shapes and mode values."""

import equinox as eqx
import jax
import jax.numpy as np
import pytest

import dLux.utils as dlu
from dLux import compat as old


@pytest.mark.parametrize("nsides", [0, 3, 6])
@pytest.mark.parametrize("j", [1, 4, 8, 12])
def test_single_mode_matches_released_numerics(j, nsides):
    coordinates = old.utils.pixel_coords(16, 2.0)
    mode = old.Zernike(j)
    expected = old.utils.zernike(j, coordinates)
    if nsides:
        expected = old.utils.polike(nsides, j, coordinates)

    # The frozen release's polar calculation is independent of the new generator.
    with jax.debug_nans(True):
        actual = eqx.filter_jit(lambda model: model.calculate(coordinates, nsides))(
            mode
        )
    assert isinstance(mode, dlu.Zernike)
    assert actual.shape == (16, 16)
    assert np.allclose(actual, expected, rtol=1e-5, atol=2e-6)
    assert mode.resolve(coordinates=coordinates).shape == (1, 16, 16)


@pytest.mark.parametrize("nsides", [0, 3, 6])
def test_basis_preserves_requested_order_and_mode_axis(nsides):
    coordinates = old.utils.pixel_coords(16, 2.0)
    indices = [4, 2, 1, 4]
    basis = old.ZernikeBasis(indices)
    expected = old.utils.zernike_basis(indices, coordinates)
    if nsides:
        expected = old.utils.polike_basis(nsides, indices, coordinates)
    actual = eqx.filter_jit(
        lambda model: model.calculate_basis(coordinates, nsides=nsides)
    )(basis)
    assert isinstance(basis, dlu.Zernike)
    assert actual.shape == (4, 16, 16)
    assert np.allclose(actual, expected, rtol=1e-5, atol=2e-6)


def test_inherited_diameter_scales_the_mode_and_remains_differentiable():
    coordinates = old.utils.pixel_coords(8, 1.0)
    mode = old.Zernike(2)
    resized = mode.set(diameter=4.0)
    assert np.allclose(resized.calculate(coordinates), mode.calculate(coordinates / 2))
    assert np.allclose(mode.diameter, 2.0)

    # Noll 2 is 2*x/radius: its diameter derivative is -4*x/diameter**2.
    derivative = jax.jacfwd(
        lambda diameter: mode.set(diameter=diameter).calculate(coordinates)
    )(np.asarray(2.0))
    assert np.allclose(derivative, -coordinates[0])
