"""Optic evaluates array definitions in its local spatial context."""

import jax
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux as dl
import dLux.utils as dlu
from tests.helpers import assert_jittable


class CoordinateValue(zdx.Expression):
    def evaluate(self, *, coordinates, **context):
        return coordinates[0]


@pytest.mark.parametrize("stored_grid", [False, True])
@pytest.mark.parametrize("transformed", [False, True])
def test_optic_coordinates_follow_grid_and_transform(
    stored_grid, transformed, make_wavefront
):
    wavefront = make_wavefront()
    grid = (
        wavefront.grid.set(c=np.array([0.1, -0.1])) if stored_grid else wavefront.grid
    )
    transform = dlu.Affine(translation=[0.1, 0.0]) if transformed else None
    optic = dl.Optic(
        phase=CoordinateValue(), grid=grid if stored_grid else None, transform=transform
    )
    coordinates = grid.coordinates if transform is None else transform(grid.coordinates)
    result = assert_jittable(optic, wavefront)
    assert np.allclose(result.phasor, wavefront.phasor * np.exp(1j * coordinates[0]))
    assert result.grid == wavefront.grid
    assert isinstance(optic.phase, CoordinateValue)


def test_transformed_expression_has_a_differentiable_local_frame(make_wavefront):
    wavefront = make_wavefront()
    optic = dl.Optic(
        phase=CoordinateValue(), transform=dlu.Affine(translation=[0.1, 0.0])
    )
    function = lambda offset: np.real(
        optic.set("transform.translation", offset)(wavefront).phasor
    ).sum()
    gradient = jax.jit(jax.grad(function))(np.array([0.1, 0.0]))
    assert np.isfinite(gradient).all()
    assert np.abs(gradient[0]) > 0.01
    assert np.allclose(gradient[1], 0.0)


@pytest.mark.parametrize(
    "kwargs", [{"grid": np.ones((2, 4, 4))}, {"transform": object()}]
)
def test_optic_context_definition_validation(kwargs):
    with pytest.raises(TypeError):
        dl.Optic(**kwargs)
