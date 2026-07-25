"""Tests for dLux.coordinates."""

import jax.numpy as np
import pytest

import dLux as dl
import dLux.utils as dlu

from .helpers import assert_differentiable, assert_jittable


class TestSpecifications:
    def test_coordinate_interface(self):
        spec = dl.CoordSpec(n=(6, 4), d=(0.2, 0.3), c=(0.1, -0.2), unit="m")

        output = assert_jittable(
            lambda value: (
                value.coordinates,
                value.axes,
                value.xs_for((4, 4)),
                value.fov,
                value.extent,
            ),
            spec,
        )

        assert output[0].shape == (2, 4, 6)
        assert spec.shape == (4, 6)
        assert spec.ndim == 2

    def test_broadcast_and_mapped_centres(self):
        base = dl.CoordSpec(n=4, d=0.1, c=0.0, unit="m").broadcast(2)
        mapped = base.set(c=np.asarray([[0.0, 0.0], [0.1, -0.1]]))

        assert base.coordinates.shape == (2, 4, 4)
        assert_jittable(lambda value: value.coordinates, mapped)
        assert mapped.coordinates.shape == (2, 2, 4, 4)

    def test_sampling_spec_contracts(self):
        pad = dl.PadSpec(pad=2, crop=3, c=(0.1, -0.1))
        resize = dl.ResizeSpec((8, 6), c=0.0).broadcast(2)

        assert pad.pad == 2 and pad.crop == 3
        assert resize.n == (8, 6)
        assert resize.c.shape == ()

    def test_units_and_differentiation(self):
        spec = dl.CoordSpec(n=(4, 6), d=(2.0, 3.0), unit="mm")

        assert np.max(np.abs(spec.coordinates)) < 0.01
        assert_differentiable(lambda value: value.coordinates, spec)

    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"n": 0}, ValueError),
            ({"n": 2.5}, TypeError),
            ({"d": 0.0}, ValueError),
            ({"unit": ""}, ValueError),
            ({"n": (2, 3), "d": (1, 2, 3)}, ValueError),
        ],
    )
    def test_validation(self, kwargs, error):
        with pytest.raises(error):
            dl.CoordSpec(**kwargs)

        with pytest.raises(ValueError):
            dl.PadSpec(pad=0)


class TestTransforms:
    @pytest.fixture
    def coordinates(self):
        return dlu.pixel_coords(6, 1.0)

    @pytest.mark.parametrize(
        "transform",
        [
            dl.AffineMap(matrix=[[1.1, 0.1], [0.0, 0.9]], offset=[0.1, -0.2]),
            dl.Affine(
                translation=[0.1, -0.2],
                rotation=0.2,
                scale=[0.9, 1.1],
                shear=[0.1, -0.05],
            ),
            dl.DistortedCoords(order=2),
            dl.TransformChain(
                [
                    dl.Affine(translation=[0.1, 0.0]),
                    dl.DistortedCoords(order=2),
                ]
            ),
        ],
    )
    def test_transform_contract(self, transform, coordinates):
        assert_jittable(lambda value: value(coordinates), transform)

    def test_transform_gradients(self, coordinates):
        affine_map = dl.AffineMap()
        assert_differentiable(
            lambda matrix: affine_map.set(matrix=matrix)(coordinates),
            affine_map.matrix,
        )

        affine = dl.Affine(rotation=0.2)
        assert_differentiable(
            lambda rotation: affine.set(rotation=rotation)(coordinates),
            affine.rotation,
        )

        distorted = dl.DistortedCoords(order=2)
        assert_differentiable(
            lambda distortion: distorted.set(distortion=distortion)(coordinates),
            distorted.distortion,
        )

    def test_coordinate_sources_and_aliases(self, coordinates):
        transform = dl.Affine(translation=[0.1, 0.0], coordinates=coordinates)

        assert np.allclose(transform(), transform.apply(coordinates))
        assert np.allclose(transform.calculate(6, 1.0), transform(coordinates))
        with pytest.raises(ValueError, match="Provide coordinates"):
            dl.Affine()()

    def test_vectorised_distortion(self, coordinates):
        transform = dl.DistortedCoords(order=2, distortion=np.zeros((3, 2, 5)))

        output = assert_jittable(lambda value: value(coordinates), transform)
        assert output.shape == (3,) + coordinates.shape

    @pytest.mark.parametrize(
        "constructor",
        [
            lambda: dl.Affine(rotation=[1.0]),
            lambda: dl.Affine(scale=0.0),
            lambda: dl.Affine(order=("rotation", "rotation")),
            lambda: dl.AffineMap(matrix=np.ones((3, 3))),
            lambda: dl.DistortedCoords(powers=np.ones((3, 2))),
            lambda: dl.DistortedCoords(order=2, orders=[2]),
        ],
    )
    def test_validation(self, constructor):
        with pytest.raises(ValueError):
            constructor()
