"""Numerical Jones objects compose with the ordinary Optic layer."""

import jax
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux as dl
import dLux.utils as dlu
from tests.helpers import assert_jittable


@pytest.mark.parametrize(
    "jones",
    [
        dlu.Jones.Identity(),
        dlu.Jones.LinearPolariser(0.2),
        dlu.Jones.HalfWavePlate(0.1),
        dlu.Jones.Retarder(np.pi / 2, 0.1),
    ],
)
def test_uniform_jones_retains_factored_wavefront(jones, make_wavefront):
    wavefront = make_wavefront()
    optic = dl.Optic(jones=jones)
    output = assert_jittable(optic, wavefront)
    assert output.phasor.shape == wavefront.phasor.shape
    assert output.jones.shape == (2, 2)
    materialised = output.materialise()
    assert np.allclose(output.intensity, materialised.intensity)
    assert np.allclose(output.stokes(), materialised.stokes())


def test_jones_chain_equals_separate_optics(make_wavefront):
    first = dlu.Jones.LinearPolariser(0.2)
    second = dlu.Jones.HalfWavePlate(0.1)
    wavefront = make_wavefront()
    sequence = dl.Optic(jones=second)(dl.Optic(jones=first)(wavefront))
    combined = dl.Optic(jones=dlu.JonesChain([first, second]))(wavefront)
    assert np.allclose(combined.materialise().phasor, sequence.materialise().phasor)


def test_spatial_jones_materialises_only_when_required(make_wavefront):
    wavefront = make_wavefront()
    angles = np.linspace(0, np.pi, 64).reshape(8, 8)
    output = assert_jittable(
        dl.Optic(jones=dlu.Jones.LinearPolariser(angles)), wavefront
    )
    assert output.jones is True
    assert output.phasor.shape == (2, 2, 8, 8)
    assert np.allclose(output.intensity, 0.5 * wavefront.intensity, rtol=2e-6)


def test_polarisation_angle_gradients_through_optic(make_wavefront):
    wavefront = make_wavefront().apply_jones(dlu.Jones.horizontal_polariser())
    optic = dl.Optic(jones=dlu.Jones.LinearPolariser(0.3))
    loss = lambda angle: optic.set("jones.angle", angle)(wavefront).power
    derivative = jax.jit(jax.grad(loss))(np.asarray(0.3))
    assert np.allclose(derivative, -0.5 * np.sin(0.6), rtol=3e-6)


def test_jones_operand_resolves_from_state(make_wavefront):
    wavefront = make_wavefront(wavelength=np.array([5e-7, 7e-7]))
    matrix = np.array([[1.0, 0.0], [0.0, 0.3]])
    optic = dl.Optic(jones=zdx.StateRef("jones"))
    output = assert_jittable(
        lambda model: model(wavefront, state=zdx.State(jones=matrix)), optic
    )
    assert np.allclose(
        output.materialise().phasor, wavefront.apply_jones(matrix).materialise().phasor
    )
