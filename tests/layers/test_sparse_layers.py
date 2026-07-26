"""Tests for dLux.layers.sparse_layers."""

import jax.numpy as np
import pytest

import dLux as dl

from tests.helpers import assert_differentiable, assert_jittable


@pytest.fixture
def positions():
    return np.array([[-0.2, 0.0], [0.2, 0.0]])


@pytest.fixture
def wavefront(make_wavefront, make_spec):
    return make_wavefront(spec=make_spec(n=16, d=0.05))


@pytest.mark.parametrize("dynamic", [False, True])
def test_sparse_optic_contract(dynamic, positions, wavefront):
    optic_type = dl.SparseDynamicOptic if dynamic else dl.SparseOptic
    optic = optic_type(
        positions,
        transmission=dl.Circle(0.2, softening=0.01),
    )

    output = assert_jittable(optic, wavefront)
    local = assert_jittable(optic.wavefronts, wavefront)
    assert optic.n_apertures == len(positions)
    assert output.phasor.shape == wavefront.phasor.shape
    assert local.phasor.shape[0] == optic.n_apertures


def test_shared_and_local_coefficients(positions, wavefront):
    shared = dl.DynamicZernikeBasis(
        js=[4],
        coefficients=[1e-7],
        diameter=0.2,
    )
    local = shared.set(coefficients=np.array([[1e-7], [2e-7]]))
    common = {
        "positions": positions,
        "transmission": dl.Circle(0.2, softening=0.01),
    }
    shared_optic = dl.SparseDynamicOptic(opd=shared, **common)
    local_optic = dl.SparseDynamicOptic(opd=local, **common)

    assert_jittable(shared_optic, wavefront)
    assert_jittable(local_optic, wavefront)
    assert not np.allclose(
        shared_optic(wavefront).phasor,
        local_optic(wavefront).phasor,
    )
    assert_differentiable(
        lambda coefficients: np.real(
            local_optic.set("opd.coefficients", coefficients)(wavefront).phasor
        ),
        local.coefficients,
    )


def test_shared_and_local_distortions(positions, wavefront):
    shared = dl.DistortCoords(order=2, shift_invariant=True)
    local = shared.set(
        distortion=np.stack(
            [
                shared.distortion,
                shared.distortion.at[0, 0].set(0.01),
            ]
        )
    )
    common = {
        "positions": positions,
        "transmission": dl.Circle(0.2, softening=0.01),
    }

    assert_jittable(
        dl.SparseDynamicOptic(transformation=shared, **common),
        wavefront,
    )
    assert_jittable(
        dl.SparseDynamicOptic(transformation=local, **common),
        wavefront,
    )


def test_position_gradients(positions, wavefront):
    optic = dl.SparseDynamicOptic(
        positions,
        transmission=dl.Circle(0.2, softening=0.01),
    )

    assert_differentiable(
        lambda value: optic.set(positions=value)(wavefront),
        positions,
    )


def test_position_validation():
    with pytest.raises(ValueError, match="positions"):
        dl.SparseOptic([0.0, 1.0])
