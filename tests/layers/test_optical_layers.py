import jax.numpy as np
import pytest

from dLux import Affine, Wavefront
from dLux.layers import (
    DynamicOptic,
    Interpolate,
    LinearPolariser,
    Optic,
    PolarisingOptic,
    Tilt,
    UniformPolarisingOptic,
)
from dLux.layers.unified_layers import AberratedLayer, Normalise, TransmissiveLayer
from dLux.parametric import Circle


@pytest.fixture
def wavefront():
    return Wavefront(1e-6, 16, diameter=1.0)


def test_optic_phasor_and_normalisation(wavefront):
    optic = Optic(transmission=0.5, opd=1e-7, phase=0.2, normalise=True)
    assert isinstance(optic, TransmissiveLayer)
    assert isinstance(optic, AberratedLayer)
    assert isinstance(optic, Normalise)
    params = optic.params(wavefront)
    assert optic.context(wavefront) == {"wavefront": wavefront}
    assert optic.phasor(wavefront).shape == (1, 1)
    assert optic.phasor(wavefront, params).shape == (1, 1)
    assert np.allclose(optic(wavefront).power, 1)


def test_optic_validation_and_propagator(wavefront):
    with pytest.raises(TypeError, match="propagator"):
        Optic(propagator=object())
    assert (
        Optic(propagator=Tilt([0, 0]))(wavefront).phasor.shape == wavefront.phasor.shape
    )


def test_optic_polarisation_forms(wavefront):
    direct = PolarisingOptic(np.eye(2))
    oriented = UniformPolarisingOptic(np.eye(2), orientation=0.1)
    parametric = LinearPolariser(0.1)
    for value in [direct, [direct], (oriented, direct), parametric]:
        output = Optic(polarisation=value)(wavefront)
        assert output.phasor.shape == (2, 2, 16, 16)


def test_dynamic_optic(wavefront):
    aperture = Circle(0.3)
    optic = DynamicOptic(aperture, transformation=Affine.translate([0.1, 0]))
    context = optic.context(wavefront)
    assert context["diameter"] == 2 * aperture.extent
    assert optic.params(wavefront)["transmission"].shape == (16, 16)
    assert optic(wavefront).phasor.shape == wavefront.phasor.shape
    assert DynamicOptic(aperture).context(wavefront)["coordinates"].shape == (
        2,
        16,
        16,
    )
    with pytest.raises(TypeError, match="aperture"):
        DynamicOptic(object())
    with pytest.raises(TypeError, match="transformation"):
        DynamicOptic(aperture, object())


def test_interpolate(wavefront):
    layer = Interpolate(Affine.translate([0.01, 0]), fill=1)
    assert layer(wavefront).phasor.shape == wavefront.phasor.shape
    with pytest.raises(TypeError, match="transformation"):
        Interpolate(object())


def test_tilt(wavefront):
    layer = Tilt([1, 2], unit="arcsec")
    assert layer(wavefront).phasor.shape == wavefront.phasor.shape
    with pytest.raises(ValueError, match="shape"):
        Tilt([1])
