"""Pure positional models, independent of spectra and image propagation."""

import jax
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux.utils as dlu
from tests.helpers import assert_jittable


def test_point_retains_the_component_axis_and_defaults_on_axis():
    assert np.array_equal(dlu.Point().resolve(), np.zeros((1, 2)))
    model = dlu.Point([1.0, 2.0])
    values = assert_jittable(lambda value: value.resolve(), model)
    assert np.array_equal(values, np.array([[1.0, 2.0]]))


@pytest.mark.parametrize("pa, displacement", [(0.0, [0.0, 3.0]), (90.0, [3.0, 0.0])])
def test_binary_is_primary_anchored(pa, displacement):
    model = dlu.Binary(position=[1.0, 2.0], separation=3.0, pa=pa, unit="rad")
    output = assert_jittable(lambda value: value.resolve(), model)
    assert np.allclose(output[0], np.array([1.0, 2.0]))
    assert np.allclose(output[1], model.position + np.array(displacement), atol=2e-7)


def test_companions_share_one_primary_and_broadcast_scalar_parameters():
    model = dlu.Companions([1.0, 2.0], separation=[3.0, 4.0], pa=0.0, unit="rad")
    output = assert_jittable(lambda value: value.resolve(), model)
    assert np.array_equal(output, np.array([[1.0, 2.0], [1.0, 5.0], [1.0, 6.0]]))
    rotated = model.set(pa=np.array([0.0, 90.0])).resolve()
    assert np.allclose(rotated[-1], np.array([5.0, 2.0]), atol=2e-7)
    assert np.array_equal(dlu.Companions().resolve(), np.zeros((1, 2)))


def test_companion_angles_are_differentiable_and_allow_expression_inputs():
    model = dlu.Companions(separation=[2.0, 3.0], pa=zdx.StateRef("angles"), unit="rad")
    assert isinstance(model.resolve(), dlu.Companions)
    angles = np.array([0.2, 0.4])
    result = assert_jittable(
        lambda value: model.resolve(state=zdx.State(angles=value)), angles
    )
    assert result.shape == (3, 2)
    gradient = jax.jit(
        jax.grad(
            lambda value: model.resolve(state=zdx.State(angles=value))[1:, 0].sum()
        )
    )(angles)
    assert np.allclose(
        gradient, np.array([2.0, 3.0]) * np.cos(np.deg2rad(angles)) * np.pi / 180
    )


def test_geometry_and_brightness_can_model_visibility_without_a_source():
    positions = dlu.Companions(
        separation=[0.1, 0.2], pa=[0.0, 90.0], unit="rad"
    ).resolve()
    brightness = dlu.ContrastFlux(primary=2.0, contrast=[0.1, 0.2], log=False).resolve()
    baselines = np.array([[0.0, 0.0], [1.0, 2.0]])
    phasors = np.exp(-2j * np.pi * (baselines @ positions.T))
    visibility = phasors @ brightness / brightness.sum()
    expected = (
        1
        + 0.1 * np.exp(-2j * np.pi * baselines[:, 1] * 0.1)
        + 0.2 * np.exp(-2j * np.pi * baselines[:, 0] * 0.2)
    ) / 1.3
    assert np.allclose(visibility, expected)


@pytest.mark.parametrize(
    "make",
    [
        lambda: dlu.Point(np.zeros(3)),
        lambda: dlu.Binary(separation=np.ones(2)),
        lambda: dlu.Companions(pa=np.ones((2, 2))),
    ],
)
def test_geometry_reports_wrong_component_shapes(make):
    with pytest.raises(ValueError):
        make()


@pytest.mark.parametrize(
    "unit, value", [("arcsec", 1.0), ("mas", 1000.0), ("deg", 1 / 3600)]
)
def test_angular_units_share_radian_output(unit, value):
    positions = dlu.Positions([[value, 0], [0, value]], unit=unit)
    expected = np.eye(2) * np.pi / (180 * 3600)
    result = assert_jittable(lambda model: model.resolve(), positions)
    assert np.allclose(result, expected, atol=1e-12)
    binary = dlu.Binary(separation=value, pa=90, unit=unit)
    assert np.allclose(binary.resolve()[1], expected[0], atol=1e-12)


def test_log_contrast_defaults_and_derivative():
    model = dlu.ContrastFlux(primary=10, contrast=np.array([-2.0, -1.0]))
    values = assert_jittable(lambda model: model.resolve(), model)
    assert np.allclose(values, np.array([10, 0.1, 1]))
    gradient = jax.jit(jax.grad(lambda c: model.set(contrast=c).resolve().sum()))(
        model.contrast
    )
    assert np.allclose(gradient, np.log(10.0) * np.array([0.1, 1.0]))
    assert np.array_equal(dlu.ContrastFlux().resolve(), np.ones(2))
    assert np.array_equal(dlu.ContrastFlux(log=False).resolve(), np.ones(2))


def test_positions_preserve_deferred_expressions_and_validate_units():
    model = dlu.Positions(zdx.StateRef("position"), unit="mas")
    assert isinstance(model.resolve(), dlu.Positions)
    value = model.resolve(state=zdx.State(position=np.array([10.0, 20.0])))
    assert np.allclose(
        value, np.array([[10.0, 20.0]]) * dlu.unit_factor("mas"), atol=1e-12
    )
    with pytest.raises(ValueError):
        dlu.Positions([1, 2], unit="m")
