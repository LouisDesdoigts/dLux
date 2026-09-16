"""Paraxial matrix construction, composition and compiled numerical contracts."""

import jax
import jax.numpy as np
import pytest

import dLux.utils as dlu


@pytest.mark.parametrize(
    "matrix",
    [
        dlu.abcd_lens(2.0),
        dlu.abcd_mirror(2.0),
        dlu.abcd_free_space(2.0),
        dlu.abcd_fraunhofer(2.0),
    ],
)
def test_matrix_contract(matrix):
    assert matrix.shape == (2, 2)
    assert np.all(np.isfinite(matrix))
    assert np.allclose(dlu.abcd_unimodularity(matrix), 1)


def test_composition_and_classification():
    free_space = dlu.abcd_free_space(2.0)
    lens = dlu.abcd_lens(3.0)
    system = dlu.compose_abcd([free_space, lens])
    assert system.shape == (2, 2)
    assert dlu.is_free_space(free_space)
    assert dlu.is_surface(lens)


@pytest.mark.parametrize(
    "construct", [dlu.ABCD.lens, dlu.ABCD.mirror, dlu.ABCD.fraunhofer]
)
@pytest.mark.parametrize("invalid", [0.0, float("nan"), float("inf")])
def test_physical_constructors_reject_invalid_concrete_scales(construct, invalid):
    with pytest.raises(ValueError, match="finite|non-zero"):
        construct(invalid)


def test_abcd_batch_composition_and_inverse_restore_incident_rays():
    distance = np.array([0.1, -0.2])
    focal_length = np.array([0.7, 0.9])
    translation = dlu.ABCD.free_space(distance)
    lens = dlu.ABCD.lens(focal_length)
    system = translation.then(lens)
    ray = np.array([0.003, 0.02])
    propagated = np.einsum("...ij,j->...i", system.matrix, ray)
    expected = np.stack(
        [
            ray[0] + distance * ray[1],
            ray[1] - (ray[0] + distance * ray[1]) / focal_length,
        ],
        axis=-1,
    )
    recovered = np.einsum("...ij,...j->...i", system.inverse().matrix, propagated)
    assert np.allclose(propagated, expected)
    assert np.allclose(recovered, np.broadcast_to(ray, recovered.shape))
    assert np.allclose(system.determinant, 1.0)


def test_abcd_jit_has_no_runtime_checks_and_keeps_physical_derivatives():
    def traced_system(focal_length):
        matrix = dlu.ABCD.free_space(0.1).then(dlu.ABCD.lens(focal_length))
        return matrix.inverse().matrix

    focal_length = np.array(0.7)
    jaxpr = jax.make_jaxpr(traced_system)(focal_length)
    assert not jaxpr.effects
    assert "callback" not in str(jaxpr).lower()
    actual = jax.jit(traced_system)(focal_length)
    assert np.allclose(actual, traced_system(focal_length))
    derivative = jax.jacfwd(traced_system)(focal_length)
    expected = np.array([[0.1 / focal_length**2, 0.0], [-1 / focal_length**2, 0.0]])
    assert np.allclose(derivative, expected, rtol=2e-6, atol=2e-6)


def test_abcd_rejects_bad_shapes_and_nonfinite_concrete_matrices():
    for value in (np.zeros((2, 3)), np.array([[1, np.inf], [0, 1]])):
        with pytest.raises(ValueError):
            dlu.ABCD(value)
    with pytest.raises(ValueError, match="non-zero"):
        dlu.ABCD(np.zeros((2, 2))).inverse()
