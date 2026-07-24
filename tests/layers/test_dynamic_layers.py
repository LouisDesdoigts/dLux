import jax.numpy as np
import pytest

import dLux.utils as dlu
from dLux import Affine, CoordSpec, Wavefront
from dLux.layers import (
    DynamicAberratedLayer,
    DynamicOptic,
    DynamicTransmissiveLayer,
)
from dLux.parametric import BaseParametric, Circle


class CoordinateValue(BaseParametric):
    def evaluate(self, *, coordinates, **kwargs):
        return coordinates[0]


@pytest.fixture
def wavefront():
    return Wavefront(1e-6, 16, diameter=1.0)


def test_dynamic_context_uses_wavefront_coordinates(wavefront):
    layer = DynamicTransmissiveLayer(Circle(0.3))
    context = layer.context(wavefront)
    assert context["coordinates"].shape == (2, 16, 16)
    assert context["pixel_scale"] == wavefront.pixel_scale


def test_dynamic_context_accepts_coord_spec(wavefront):
    spec = CoordSpec(16, 1 / 16, 0.1)
    layer = DynamicTransmissiveLayer(Circle(0.3), coordinates=spec)
    context = layer.context(wavefront)
    assert context["coordinates"].shape == (2, 16, 16)
    assert context["pixel_scale"] == spec.d


def test_dynamic_context_accepts_transform_coordinate_source(wavefront):
    spec = CoordSpec(16, 1 / 16, 0.1)
    transformation = Affine(translation=[0.1, 0], coordinates=spec)
    layer = DynamicTransmissiveLayer(Circle(0.3), transformation=transformation)
    context = layer.context(wavefront)

    assert context["coordinates"].shape == (2, 16, 16)
    assert context["pixel_scale"] == spec.d
    assert np.allclose(context["coordinates"], transformation())


def test_dynamic_context_accepts_arrays_and_transformations(wavefront):
    coordinates = dlu.pixel_coords(16, 1.0)
    layer = DynamicTransmissiveLayer(
        CoordinateValue(),
        coordinates=coordinates,
        transformation=Affine(translation=[0.1, 0]),
    )
    expected = Affine(translation=[0.1, 0])(coordinates)[0]
    assert np.allclose(layer.context(wavefront)["coordinates"][0], expected)
    assert layer(wavefront).phasor.shape == wavefront.phasor.shape


def test_dynamic_context_validation():
    with pytest.raises(ValueError, match="coordinates"):
        DynamicTransmissiveLayer(coordinates=np.ones((3, 4, 4)))
    with pytest.raises(TypeError, match="transformation"):
        DynamicTransmissiveLayer(transformation=object())


def test_dynamic_transmission_supports_static_values(wavefront):
    layer = DynamicTransmissiveLayer(0.5, normalise=True)
    assert np.allclose(layer(wavefront).power, 1)
    assert DynamicTransmissiveLayer()(wavefront).phasor.shape == wavefront.phasor.shape


def test_dynamic_aberrated_layer(wavefront):
    layer = DynamicAberratedLayer(opd=CoordinateValue(), phase=0.1)
    output = layer(wavefront)
    expected = wavefront.add_opd(wavefront.coordinates()[0]).add_phase(0.1)
    assert np.allclose(output.phasor, expected.phasor)


def test_dynamic_optic_mixes_static_and_dynamic_leaves(wavefront):
    optic = DynamicOptic(
        transmission=Circle(0.3),
        opd=CoordinateValue(),
        phase=0.1,
    )
    params = optic.params(wavefront)
    assert params["transmission"].shape == (16, 16)
    assert params["opd"].shape == (16, 16)
    assert optic(wavefront).phasor.shape == wavefront.phasor.shape
