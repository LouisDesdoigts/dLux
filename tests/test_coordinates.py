import jax
import jax.numpy as np
import pytest

import dLux.utils as dlu
from dLux.coordinates import (
    Affine,
    BaseCoordTransform,
    Distort,
    DistortedCoords,
    TransformChain,
)


def test_affine_defaults_and_validation():
    affine = Affine()
    matrix, offset = affine.coefficients()
    assert np.array_equal(matrix, np.eye(2))
    assert np.array_equal(offset, np.zeros(2))
    for keyword in ("translation", "shear", "offset"):
        with pytest.raises(ValueError, match=keyword):
            Affine(**{keyword: [1]})
    with pytest.raises(ValueError, match="rotation"):
        Affine(rotation=[1])
    with pytest.raises(ValueError, match="matrix"):
        Affine(matrix=np.ones((3, 3)))
    with pytest.raises(ValueError, match="non-zero"):
        Affine(scale=0)
    with pytest.raises(ValueError, match="order"):
        Affine(order=("rotation", "rotation"))
    with pytest.raises(ValueError, match="order"):
        Affine(order=("unknown",))


def test_affine_semantic_parameters_and_order():
    affine = Affine(
        translation=[0.1, -0.2],
        rotation=0.2,
        scale=[0.9, 1.1],
        shear=[0.1, -0.05],
    )
    coords = dlu.pixel_coords(4, 1.0)
    assert affine(coords).shape == coords.shape
    reordered = Affine(
        translation=[0.1, -0.2],
        rotation=0.2,
        order=("rotation", "translation"),
    )
    assert not np.allclose(affine(coords), reordered(coords))


def test_affine_direct_matrix_and_offset():
    coords = dlu.pixel_coords(4, 1.0)
    assert np.allclose(Affine(matrix=2 * np.eye(2))(coords), 2 * coords)
    assert np.allclose(
        Affine(offset=[0.1, 0.2])(coords),
        coords + np.array([0.1, 0.2])[:, None, None],
    )


def test_affine_gradients():
    coords = dlu.pixel_coords(4, 1.0)
    gradient = jax.grad(lambda angle: Affine(rotation=angle)(coords).sum())(0.1)
    assert np.isfinite(gradient)


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
