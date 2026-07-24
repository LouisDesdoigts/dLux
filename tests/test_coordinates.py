import jax
import jax.numpy as np
import pytest

import dLux.utils as dlu
from dLux import CoordSpec
from dLux.coordinates import (
    Affine,
    AffineMap,
    DistortedCoords,
    TransformChain,
)


def test_affine_defaults_and_validation():
    affine = Affine()
    matrix, offset = affine.coefficients()
    assert np.array_equal(matrix, np.eye(2))
    assert np.array_equal(offset, np.zeros(2))
    for keyword in ("translation", "shear"):
        with pytest.raises(ValueError, match=keyword):
            Affine(**{keyword: [1]})
    with pytest.raises(ValueError, match="rotation"):
        Affine(rotation=[1])
    with pytest.raises(ValueError, match="non-zero"):
        Affine(scale=0)
    with pytest.raises(ValueError, match="order"):
        Affine(order=("rotation", "rotation"))
    with pytest.raises(ValueError, match="order"):
        Affine(order=("unknown",))
    with pytest.raises(ValueError, match="coordinates"):
        Affine(coordinates=np.ones((3, 4, 4)))


def test_coordinate_sources_and_precedence():
    coordinates = dlu.pixel_coords(4, 1.0)
    other = dlu.pixel_coords(4, 2.0)
    stored = Affine(translation=[0.1, 0.0], coordinates=coordinates)
    specified = Affine(coordinates=CoordSpec(4, 0.25))

    assert np.allclose(stored(), stored(coordinates))
    assert np.allclose(stored(other), other - np.array([0.1, 0.0])[:, None, None])
    assert np.allclose(specified(), coordinates)
    with pytest.raises(ValueError, match="Provide coordinates"):
        Affine()()


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


def test_affine_map():
    coords = dlu.pixel_coords(4, 1.0)
    assert np.allclose(AffineMap(matrix=2 * np.eye(2))(coords), 2 * coords)
    assert np.allclose(
        AffineMap(offset=[0.1, 0.2])(coords),
        coords + np.array([0.1, 0.2])[:, None, None],
    )
    assert np.allclose(AffineMap()(coords), coords)
    with pytest.raises(ValueError, match="matrix"):
        AffineMap(matrix=np.ones((3, 3)))
    with pytest.raises(ValueError, match="offset"):
        AffineMap(offset=np.ones(3))


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
    with pytest.raises(ValueError, match="trailing dimensions"):
        DistortedCoords(2, np.zeros(5))
    with pytest.raises(ValueError, match="powers"):
        DistortedCoords(powers=np.ones((3, 2)))


def test_distorted_coordinates_accept_explicit_powers():
    powers = np.array([[2.0, 1.0], [0.0, 1.0]])
    distortion = np.array([[0.1, 0.0], [0.0, 0.1]])
    coordinates = dlu.pixel_coords(8, 1.0)
    transform = DistortedCoords(distortion=distortion, powers=powers)

    assert np.array_equal(transform.powers, powers)
    assert transform(coordinates).shape == coordinates.shape


def test_distorted_coordinates_select_orders_and_shift_invariance():
    selected = DistortedCoords(orders=[2])
    invariant = DistortedCoords(order=2, shift_invariant=True)

    assert np.all(selected.powers.sum(0) == 2)
    assert np.array_equal(selected.powers, invariant.powers)
    with pytest.raises(ValueError, match="only one"):
        DistortedCoords(order=2, orders=[2])
    with pytest.raises(ValueError, match="positive"):
        DistortedCoords(orders=[0])


def test_transform_chain_order_and_inputs():
    coords = dlu.pixel_coords(4, 1.0)
    first = DistortedCoords()
    chain = TransformChain({"first": first})
    assert chain.transformations["first"] is first
    assert np.allclose(chain(coords), first(coords))
    assert np.allclose(TransformChain()(coords), coords)
    assert np.allclose(TransformChain(coordinates=coords)(), coords)
