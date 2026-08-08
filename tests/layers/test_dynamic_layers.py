"""Tests for dLux.layers.dynamic_layers."""

import jax.numpy as np
import pytest

import dLux as dl

from tests.helpers import assert_differentiable, assert_jittable


class CoordinateValue(dl.Parametric):
    """Return one component from the dynamic coordinate context."""

    def evaluate(self, *, coordinates, **kwargs):
        return 1e-7 * coordinates[0]


@pytest.fixture
def wavefront(make_wavefront):
    return make_wavefront()


@pytest.mark.parametrize(
    "layer",
    [
        dl.DynamicTransmissiveLayer(dl.Circle(0.5, edge=0.02)),
        dl.DynamicTransmissiveLayer(0.5, normalise=True),
        dl.DynamicAberratedLayer(opd=CoordinateValue(), phase=0.1),
        dl.DynamicOptic(
            transmission=dl.Circle(0.5, edge=0.02),
            opd=CoordinateValue(),
            phase=0.1,
        ),
    ],
)
def test_dynamic_layer_contract(layer, wavefront):
    output = assert_jittable(layer, wavefront)
    assert output.phasor.shape == wavefront.phasor.shape


def test_coordinate_sources(wavefront, make_grid):
    grid = make_grid(c=[0.1, -0.1])
    coordinates = grid.coordinates
    transformations = [
        dl.Affine(translation=[0.1, 0.0]),
        dl.Affine(translation=[0.1, 0.0]),
    ]
    layers = [
        dl.DynamicTransmissiveLayer(CoordinateValue()),
        dl.DynamicTransmissiveLayer(CoordinateValue(), coordinates=coordinates),
        dl.DynamicTransmissiveLayer(CoordinateValue(), coordinates=grid),
        dl.DynamicTransmissiveLayer(
            CoordinateValue(),
            transformation=transformations[0],
        ),
        dl.DynamicTransmissiveLayer(
            CoordinateValue(),
            transformation=transformations[1],
        ),
    ]

    for layer in layers:
        context = layer.context(wavefront)
        assert context["coordinates"].shape == wavefront.coordinates.shape
        assert_jittable(layer, wavefront)

        assert np.allclose(layers[2].context(wavefront)["pixel_scale"], grid.d)
        assert np.allclose(
            layers[-1].context(wavefront)["coordinates"],
            transformations[-1](wavefront.coordinates),
        )


@pytest.mark.parametrize(
    ("layer", "path"),
    [
        (
            dl.DynamicTransmissiveLayer(
                dl.Circle(0.5, edge=0.02),
            ),
            "transmission.diameter",
        ),
        (
            dl.DynamicAberratedLayer(opd=CoordinateValue(), phase=0.1),
            "phase",
        ),
        (
            dl.DynamicOptic(
                transmission=dl.Circle(0.5, edge=0.02),
            ),
            "transmission.diameter",
        ),
        (
            dl.DynamicOptic(
                transmission=CoordinateValue(),
                transformation=dl.Affine(translation=[0.1, 0.0]),
            ),
            "transformation.translation",
        ),
    ],
)
def test_dynamic_layer_gradients(layer, path, wavefront):
    value = layer.get(path)

    assert_differentiable(
        lambda replacement: np.real(layer.set(path, replacement)(wavefront).phasor),
        value,
    )


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: dl.DynamicTransmissiveLayer(coordinates=np.ones((3, 4, 4))),
        lambda: dl.DynamicTransmissiveLayer(transformation=object()),
    ],
)
def test_validation(constructor):
    with pytest.raises((TypeError, ValueError)):
        constructor()
