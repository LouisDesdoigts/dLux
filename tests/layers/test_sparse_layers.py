import jax
import jax.numpy as np
import pytest

from dLux import (
    Circle,
    DistortedCoords,
    DynamicZernikeBasis,
    SparseDynamicOptic,
    SparseOptic,
    Wavefront,
)


@pytest.fixture
def wavefront():
    return Wavefront(1e-6, 32, diameter=4.0)


@pytest.fixture
def positions():
    return np.array([[-1.0, 0.0], [1.0, 0.0]])


def test_sparse_optic_positions(positions):
    optic = SparseOptic(positions)
    assert optic.n_apertures == 2
    with pytest.raises(ValueError, match="positions"):
        SparseOptic([0.0, 1.0])


def test_sparse_dynamic_optic_places_shape(wavefront, positions):
    optic = SparseDynamicOptic(positions, transmission=Circle(diameter=1.0))
    output = optic(wavefront)

    assert output.phasor.shape == wavefront.phasor.shape
    assert np.count_nonzero(np.abs(output.phasor)) > 0


def test_sparse_coefficients_are_shared_or_local(wavefront, positions):
    shared = DynamicZernikeBasis(js=[1], coefficients=np.array([1e-8]), diameter=1.0)
    local = DynamicZernikeBasis(
        js=[1], coefficients=np.array([[1e-8], [2e-8]]), diameter=1.0
    )
    common = {"positions": positions, "transmission": Circle(diameter=1.0)}

    shared_output = SparseDynamicOptic(opd=shared, **common)(wavefront)
    local_optic = SparseDynamicOptic(opd=local, **common)
    local_output = local_optic(wavefront)
    gradient = jax.grad(
        lambda coefficients: SparseDynamicOptic(
            opd=local.set(coefficients=coefficients), **common
        )(wavefront).power
    )(local.coefficients)

    assert not np.allclose(shared_output.phasor, local_output.phasor)
    assert np.isfinite(gradient).all()


def test_sparse_distortions_are_shared_or_local(wavefront, positions):
    shared = DistortedCoords(order=2, shift_invariant=True)
    local = DistortedCoords(
        powers=shared.powers,
        distortion=np.stack((shared.distortion, shared.distortion.at[0, 0].set(0.1))),
    )
    shape = Circle(diameter=1.0)

    shared_output = SparseDynamicOptic(
        positions, transmission=shape, transformation=shared
    )(wavefront)
    local_output = SparseDynamicOptic(
        positions, transmission=shape, transformation=local
    )(wavefront)

    assert not np.allclose(shared_output.phasor, local_output.phasor)
