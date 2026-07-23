import jax.numpy as np
import pytest

import dLux.utils as dlu
from dLux.coordinates import (
    BaseCoordTransform,
    Distort,
    DistortedCoords,
    TransformChain,
)


def test_distorted_coordinates_and_aliases():
    coords = dlu.pixel_coords(8, 1.0)
    transform = DistortedCoords()
    assert transform.calculate(8, 1.0).shape == coords.shape
    assert np.allclose(transform.apply(coords), transform(coords))


def test_distorted_coordinate_validation():
    with pytest.raises(ValueError, match="distortion shape"):
        DistortedCoords(2, np.zeros(5))


def test_distort_wraps_distorted_coordinates():
    coords = dlu.pixel_coords(8, 1.0)
    transform = Distort()
    assert isinstance(transform, BaseCoordTransform)
    assert np.array_equal(transform.coefficients, transform.distortion.distortion)
    assert np.allclose(transform(coords), transform.distortion(coords))


def test_transform_chain_order_and_inputs():
    coords = dlu.pixel_coords(4, 1.0)
    first = Distort()
    chain = TransformChain({"first": first})
    assert chain.transformations["first"] is first
    assert np.allclose(chain(coords), first(coords))
    assert np.allclose(TransformChain()(coords), coords)
