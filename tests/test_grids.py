"""Tests for dLux.grids."""

import jax.numpy as np
import pytest

import dLux as dl
import dLux.utils as dlu

from .helpers import assert_differentiable, assert_jittable


class TestSpecifications:
    def test_coordinate_interface(self):
        spec = dl.GridSpec(n=(6, 4), d=(0.2, 0.3), c=(0.1, -0.2), unit="m")

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
        base = dl.GridSpec(n=4, d=0.1, c=0.0, unit="m").broadcast(2)
        mapped = base.set(c=np.asarray([[0.0, 0.0], [0.1, -0.1]]))

        assert base.coordinates.shape == (2, 4, 4)
        assert_jittable(lambda value: value.coordinates, mapped)
        assert mapped.coordinates.shape == (2, 2, 4, 4)

    def test_sampling_spec_contracts(self):
        pad = dl.ResizeSpec(pad=2, crop=3, c=(0.1, -0.1))
        resize = dl.ResizeSpec((8, 6), c=0.0).broadcast(2)

        assert pad.pad_factor == (2,) and pad.crop_factor == (3,)
        assert resize.n == (8, 6)
        assert resize.c.shape == ()
        array = np.ones((6, 6))
        assert pad.pad(array).shape == (12, 12)
        assert pad.crop(array).shape == (2, 2)
        assert pad.resize(array).shape == (4, 4)
        assert pad.crop_size((12, 12)) == (4, 4)
        assert resize.resize(array).shape == (6, 8)

    def test_explicit_resize_contract(self):
        spec = dl.ResizeSpec((8, 6)).broadcast(2)
        array = np.ones((4, 4))

        assert spec.explicit
        assert spec.padding == {"pad_to": (8, 6)}
        assert spec.output_size(array.shape) == (8, 6)
        assert spec.crop_size(array.shape) == (8, 6)
        assert spec.pad(array).shape == (6, 8)
        assert spec.crop(np.ones((10, 10))).shape == (6, 8)

    def test_grid_sampling_transforms(self):
        spec = dl.GridSpec(n=(8, 6), d=(0.1, 0.2), unit="m")

        resized = spec.resize((10, 8))
        downsampled = spec.downsample((2, 3))
        resampled = spec.resample((12, 10), (0.05, 0.08))

        assert resized.n == (10, 8)
        assert np.allclose(resized.d, spec.d)
        assert downsampled.n == (4, 2)
        assert np.allclose(downsampled.d, np.asarray((0.2, 0.6)))
        assert resampled.n == (12, 10)
        assert np.allclose(resampled.d, np.asarray((0.05, 0.08)))

    def test_units_and_differentiation(self):
        spec = dl.GridSpec(n=(4, 6), d=(2.0, 3.0), unit="mm")

        assert np.max(np.abs(spec.coordinates)) < 0.01
        assert_differentiable(lambda value: value.coordinates, spec)

    def test_diameter_sampling(self):
        spec = dl.GridSpec(n=(4, 6), diam=(2.0, 3.0), unit="m")

        assert np.allclose(spec.d, np.asarray((0.5, 0.5)))
        assert all(np.isclose(axis.mean(), 0.0) for axis in spec.axes)
        assert spec.coordinates.shape == (2, 6, 4)

    def test_rectangular_axes_contract(self):
        spec = dl.GridSpec(n=(6, 4), d=(0.2, 0.3), unit="m")

        assert tuple(axis.shape for axis in spec.axes) == ((6,), (4,))
        assert tuple(x.shape for x in spec.xs) == ((6,), (4,))
        assert np.allclose(spec.extent, np.asarray((-0.6, 0.6, -0.6, 0.6)))

    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"n": 0}, ValueError),
            ({"n": 2.5}, TypeError),
            ({"d": 0.0}, ValueError),
            ({"unit": ""}, ValueError),
            ({"n": (2, 3), "d": (1, 2, 3)}, ValueError),
            ({"n": 4, "d": 0.1, "diam": 1.0}, ValueError),
            ({"diam": 1.0}, ValueError),
        ],
    )
    def test_validation(self, kwargs, error):
        with pytest.raises(error):
            dl.GridSpec(**kwargs)

        with pytest.raises(ValueError):
            dl.ResizeSpec(pad=0)


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
            dl.DistortCoords(order=2),
            dl.TransformChain(
                [dl.Affine(translation=[0.1, 0.0]), dl.DistortCoords(order=2)]
            ),
        ],
    )
    def test_transform_contract(self, transform, coordinates):
        assert_jittable(lambda value: value(coordinates), transform)

    def test_transform_gradients(self, coordinates):
        affine_map = dl.AffineMap()
        assert_differentiable(
            lambda matrix: affine_map.set(matrix=matrix)(coordinates), affine_map.matrix
        )

        affine = dl.Affine(rotation=0.2)
        assert_differentiable(
            lambda rotation: affine.set(rotation=rotation)(coordinates), affine.rotation
        )

        distorted = dl.DistortCoords(order=2)
        assert_differentiable(
            lambda distortion: distorted.set(distortion=distortion)(coordinates),
            distorted.distortion,
        )

    def test_transform_requires_external_grid(self, coordinates):
        transform = dl.Affine(translation=[0.1, 0.0])

        assert np.allclose(transform(coordinates), transform.apply(coordinates))
        spec = dl.GridSpec(n=6, d=1 / 6, unit="m").broadcast(2)
        with pytest.raises((TypeError, ValueError)):
            transform(spec)
        with pytest.raises((TypeError, ValueError)):
            dl.Affine()()

    def test_vectorised_distortion(self, coordinates):
        transform = dl.DistortCoords(order=2, distortion=np.zeros((3, 2, 5)))

        output = assert_jittable(lambda value: value(coordinates), transform)
        assert output.shape == (3,) + coordinates.shape

    def test_paired_vectorised_distortion(self, coordinates):
        base = dl.DistortCoords(orders=(1, 2))
        distortion = np.stack((base.distortion, base.distortion.at[0, 0].set(0.1)))
        transform = base.set(distortion=distortion)
        mapped_coordinates = np.stack((coordinates, coordinates + 0.1))

        output = assert_jittable(lambda value: transform(value), mapped_coordinates)
        assert output.shape == mapped_coordinates.shape

    def test_batched_affine_map(self, coordinates):
        coordinates = np.stack((coordinates, coordinates + 0.1))
        matrix = np.stack((np.eye(2), np.asarray(((1.0, 0.1), (0.0, 1.0)))))
        offset = np.asarray(((0.0, 0.0), (0.1, -0.1)))
        transform = dl.AffineMap(matrix, offset)

        output = assert_jittable(lambda value: transform(value), coordinates)

        assert output.shape == coordinates.shape
        assert np.allclose(output[0], coordinates[0])

    @pytest.mark.parametrize(
        "constructor",
        [
            lambda: dl.Affine(rotation=[[1.0]]),
            lambda: dl.Affine(scale=0.0),
            lambda: dl.Affine(order=("rotation", "rotation")),
            lambda: dl.AffineMap(matrix=np.ones((3, 3))),
            lambda: dl.AffineMap(offset=np.ones(3)),
            lambda: dl.Affine(translation=np.ones(3)),
            lambda: dl.DistortCoords(powers=np.ones((3, 2))),
            lambda: dl.DistortCoords(order=2, orders=[2]),
        ],
    )
    def test_validation(self, constructor):
        with pytest.raises(ValueError):
            constructor()
