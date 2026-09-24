"""Regular grids, resize specifications and coordinate transforms."""

import jax
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux as dl
import dLux.utils as dlu

from tests.helpers import assert_differentiable, assert_jittable, assert_tree_allclose


class TestSpecifications:
    def test_sampling_equivalence_uses_physical_units_and_broadcast_metadata(self):
        grid = dl.Grid(4, d=(1, 2), c=(3, 4), unit="mm")
        equivalent = dl.Grid(
            4,
            d=np.array([[0.001, 0.002], [0.001, 0.002]]),
            c=(0.003, 0.004),
            unit="m",
        )
        assert grid.check_sampling(equivalent) is grid
        assert equivalent.check_sampling(grid) is equivalent

        # Unknown values stay deferred until the owner supplies their context.
        deferred = grid.set(d=zdx.StateRef("spacing"))
        assert deferred.check_sampling(grid) is deferred
        resolved = deferred.resolve(state=zdx.State(spacing=np.array([1.0, 2.0])))
        assert np.array_equal(resolved.d, grid.d)
        assert resolved.check_sampling(grid) is resolved

    @pytest.mark.parametrize(
        "other",
        [
            dl.Grid(6, d=(1, 2), unit="mm"),
            dl.Grid(4, unit="mm"),
            dl.Grid(4, d=(1, 2), unit="rad"),
            dl.Grid(4, d=(2, 2), unit="mm"),
            dl.Grid(4, d=(1, 2), c=(1, 0), unit="mm"),
            dl.Grid(4, d=np.ones((3, 2)), unit="mm"),
        ],
        ids=["counts", "incomplete", "dimension", "spacing", "centre", "batches"],
    )
    def test_sampling_mismatch_validation_tails(self, other):
        grid = dl.Grid(4, d=np.array([[1.0, 2.0], [1.0, 2.0]]), unit="mm")
        with pytest.raises(ValueError, match="Wrong pupil sampling"):
            grid.check_sampling(other, "Wrong pupil sampling")

    @pytest.mark.parametrize("n", [(1, 1), (3, 1), (1, 4), (3, 2)])
    def test_two_dimensional_coordinates(self, n):
        d = np.arange(1, len(n) + 1, dtype=float)
        c = d / 3
        grid = dl.Grid(n, d, c, unit="mm")
        axes = tuple(
            (np.linspace(-(size - 1) / 2, (size - 1) / 2, size) * step + center) * 1e-3
            for size, step, center in zip(n, d, c)
        )
        expected = np.stack(np.meshgrid(*axes, indexing="xy"))

        output = assert_jittable(lambda value: value.coordinates, grid)

        assert grid.shape == n[::-1]
        assert output.shape == (len(n),) + grid.shape
        assert np.allclose(output, expected)
        assert all(np.allclose(actual, axis) for actual, axis in zip(grid.axes, axes))

    def test_broadcast_metadata_matches_independent_grids(self):
        d = np.asarray([[[1.0, 2.0]], [[2.0, 3.0]]])
        c = np.asarray([[[0.0, 1.0], [2.0, -1.0], [1.0, 0.0]]])
        grid = dl.Grid((4, 3), d, c, unit="mm")
        coordinates, extent = assert_jittable(
            lambda value: (value.coordinates, value.extent()), grid
        )
        expected = np.stack(
            [
                np.stack(
                    [
                        dl.Grid(grid.n, spacing, center, "mm").coordinates
                        for center in c[0]
                    ]
                )
                for spacing in d[:, 0]
            ]
        )

        assert grid.batch_shape == (2, 3)
        assert np.allclose(coordinates, expected)
        assert extent.shape == (2, 3, 4)

        # A shared centre must also broadcast over independently batched spacings
        shared = grid.set(c=None)
        bounds = np.stack((-shared.fov / 2, shared.fov / 2), -1).reshape((2, 1, 4))
        assert np.allclose(shared.extent(), bounds)

    def test_coordinate_gradients_and_vectorisation(self):
        grid = dl.Grid((3, 1), (2.0, 3.0), (1.0, 2.0), "mm")
        gradient = assert_differentiable(lambda value: value.coordinates, grid)

        assert np.allclose(gradient.d, np.asarray((8e-6, 0.0)), atol=1e-10)
        assert np.allclose(gradient.c, np.asarray((6e-6, 12e-6)), atol=1e-10)

        centres = np.asarray(((0.0, 0.0), (1.0, -2.0)))
        mapped = jax.vmap(lambda center: grid.set(c=center).coordinates)(centres)
        assert np.allclose(mapped, grid.set(c=centres).coordinates)

    def test_scalar_sampling_and_field_promotion(self):
        grid = dl.Grid(4, 2.0, unit="mm")
        image = dl.Intensity(np.ones((4, 4)), grid)
        partial = dl.Grid(d=2.0, unit="mm").match_shape((3, 4))

        assert grid.n == (4, 4)
        assert grid.coordinates.shape == (2, 4, 4)
        assert image.grid.coordinates.shape == (2, 4, 4)
        assert np.allclose(image.coordinates, grid.coordinates)
        assert partial.coordinates.shape == (2, 3, 4)
        assert np.allclose(partial.d, np.asarray((2.0, 2.0)))

    @pytest.mark.parametrize(
        "width, spacing",
        [(12.0, (3.0, 4.0)), ((8.0, 9.0), (2.0, 3.0))],
        ids=["scalar-diameter", "axis-diameters"],
    )
    def test_contextual_sampling_keeps_physical_units(self, width, spacing):
        grid = dl.Grid(
            n=(4, 3),
            diam=zdx.StateRef("width"),
            c=zdx.StateRef("centre"),
            unit="mm",
        )
        state = zdx.State(width=np.asarray(width), centre=1.0)
        sampled = assert_jittable(lambda value: grid.resolve(state=value), state)
        expected = dl.Grid((4, 3), spacing, 1.0, "mm")

        assert isinstance(grid.d, zdx.Expression)
        assert sampled.unit == "mm"
        assert np.allclose(sampled.d, expected.d)
        assert np.allclose(sampled.coordinates, expected.coordinates)
        assert np.allclose(
            zdx.resolve(grid, state=state).coordinates, expected.coordinates
        )

    @pytest.mark.parametrize("old, new", [(4, 6), (6, 4), (5, 7), (7, 5)])
    def test_resize_samples_preserves_retained_coordinates(self, old, new):
        grid = dl.Grid(old, 0.2, 0.7, "mm")
        resized = grid.resize_samples(new)
        coordinates = dlu.resize(grid.axes[0], (new,), fill=np.nan)
        retained = np.isfinite(coordinates)

        assert np.allclose(resized.axes[0][retained], coordinates[retained])
        assert np.allclose(resized.d, grid.d)
        assert np.allclose(grid.resize(new).c, grid.c)

    def test_frequency_grid_matches_fft_bins(self):
        grid = dl.Grid((4, 5), (2.0, 3.0), (1.0, -2.0), "mm")
        frequency = assert_jittable(lambda value: value.frequency(), grid)

        assert frequency.unit == "1/m"
        for size, spacing, axis in zip(grid.n, grid.d_si, frequency.axes):
            expected = np.fft.fftshift(np.fft.fftfreq(size, spacing))
            assert np.allclose(axis, expected)

    def test_coordinate_interface(self):
        spec = dl.Grid(n=(6, 4), d=(0.2, 0.3), c=(0.1, -0.2), unit="m")

        output = assert_jittable(
            lambda value: (
                value.coordinates,
                value.axes,
                value.resize((4, 4)).axes,
                value.fov,
                value.extent(),
            ),
            spec,
        )

        assert output[0].shape == (2, 4, 6)
        assert spec.shape == (4, 6)
        assert spec.ndim == 2

    def test_broadcast_and_mapped_centres(self):
        base = dl.Grid(n=4, d=0.1, c=0.0, unit="m")
        mapped = base.set(c=np.asarray([[0.0, 0.0], [0.1, -0.1]]))

        assert base.coordinates.shape == (2, 4, 4)
        assert_jittable(lambda value: value.coordinates, mapped)
        assert mapped.coordinates.shape == (2, 2, 4, 4)

    def test_match_shape(self):
        grid = dl.Grid(d=0.1, unit="m")
        matched = grid.match_shape((6, 8))

        assert matched.n == (8, 6)
        assert matched.d.shape == (2,)
        assert dl.Grid((8, 6), 0.1).match_shape((6, 8)).n == (8, 6)

        with pytest.raises(ValueError, match="shape must match its sampling UnitSpec"):
            dl.Grid((8, 6), 0.1).match_shape((8, 6))

    def test_sampling_spec_contracts(self):
        pad = dl.ResizeSpec(pad=2, crop=3)
        resize = dl.ResizeSpec((8, 6))

        assert pad.pad == (2, 2) and pad.crop == (3, 3)
        assert resize.n == (8, 6)
        array = np.ones((6, 6))
        assert pad.pad_array(array).shape == (12, 12)
        assert pad.crop_array(array).shape == (2, 2)
        assert pad.resize(array).shape == (4, 4)
        assert pad.crop_size((12, 12)) == (4, 4)
        assert resize.resize(array).shape == (6, 8)

    def test_paste_spec_contract(self):
        grid = dl.Grid(n=(8, 6), d=0.5, unit="m")
        centers = np.asarray(((-0.5, 0.0), (0.5, 0.0)))
        spec = dl.PasteSpec.from_grid(grid, centers, extent=0.4)
        arrays = np.ones((2, 5, 5))

        scan = spec.paste(arrays, "scan")
        scatter = spec.paste(arrays, "scatter")

        assert spec.n == (8, 6)
        assert spec.stamp_shape == (5, 5)
        assert spec.shape == (6, 8)
        assert spec.coordinates.shape == (2, 2, 5, 5)
        assert np.array_equal(scan, scatter)
        assert_jittable(lambda value, array: value.paste(array), spec, arrays)

        with pytest.raises(ValueError, match="within the output"):
            dl.PasteSpec((4, 4), (3, 3), [[3, 0]], [[0.0, 0.0]], [1.0, 1.0])

    def test_explicit_resize_contract(self):
        spec = dl.ResizeSpec((8, 6))
        array = np.ones((4, 4))

        assert spec.explicit
        assert spec.output_size(array.shape) == (8, 6)
        assert spec.crop_size(array.shape) == (8, 6)
        assert spec.pad_array(array).shape == (6, 8)
        assert spec.crop_array(np.ones((10, 10))).shape == (6, 8)

    def test_crop_axes(self):
        spec = dl.ResizeSpec(crop=(2, 3))
        x = np.linspace(-1.0, 1.0, 8)
        y = np.linspace(-2.0, 2.0, 12)

        x_out, y_out = spec.crop_axes((x, y))

        assert x_out.shape == (4,)
        assert y_out.shape == (4,)
        assert np.allclose(x_out, dlu.crop_to(x, (4,)))
        assert np.allclose(y_out, dlu.crop_to(y, (4,)))

    def test_grid_from_axes(self):
        x = np.linspace(-3.5e-3, 3.5e-3, 8)
        y = np.linspace(-2.5e-3, 2.5e-3, 6)

        spec = dl.Grid.from_axes((x, y), unit="mm")
        batched = dl.Grid.from_axes(
            (np.stack((x, x + 1e-3)), np.stack((y, y - 2e-3))),
            unit="mm",
        )

        assert spec.n == (8, 6)
        assert np.allclose(spec.d, np.asarray((1.0, 1.0)))
        assert np.allclose(spec.c, np.zeros(2))
        assert spec.unit == "mm"
        assert batched.d.shape == batched.c.shape == (2, 2)
        assert np.allclose(batched.c, np.asarray(((0.0, 0.0), (1.0, -2.0))))

    def test_grid_from_axes_broadcasts_in_xy_order(self):
        x = np.asarray((-1.0, 0.0, 1.0)) * 1e-3
        y = np.asarray(((0.0, 2.0), (1.0, 3.0))) * 1e-3
        grid = dl.Grid.from_axes((x, y), unit="mm")

        assert grid.coordinates.shape == (2, 2, 2, 3)
        assert np.allclose(grid.axes[0], np.broadcast_to(x, (2, 3)))
        assert np.allclose(grid.axes[1], y)

    @pytest.mark.parametrize("axes", [(), (1.0,), ([1.0],), ([],)])
    def test_grid_from_axes_requires_sampled_vectors(self, axes):
        with pytest.raises(ValueError, match="coordinate|samples"):
            dl.Grid.from_axes(axes)

    def test_grid_sampling_transforms(self):
        spec = dl.Grid(n=(8, 6), d=(0.1, 0.2), unit="m")

        resized = spec.resize((10, 8))
        downsampled = spec.downsample((2, 3))
        oversampled = spec.oversample((2, 3))
        resampled = spec.resample((12, 10), (0.05, 0.08))

        assert resized.n == (10, 8)
        assert np.allclose(resized.d, spec.d)
        assert downsampled.n == (4, 2)
        assert np.allclose(downsampled.d, np.asarray((0.2, 0.6)))
        assert oversampled.n == (16, 18)
        assert np.allclose(oversampled.d, np.asarray((0.05, 0.2 / 3)))
        assert np.allclose(oversampled.fov, spec.fov)
        assert resampled.n == (12, 10)
        assert np.allclose(resampled.d, np.asarray((0.05, 0.08)))

    def test_batched_metadata_sampling_transforms(self):
        d = np.asarray(((0.1, 0.2), (0.3, 0.4)))
        c = np.asarray(((0.0, 0.1), (-0.1, 0.2)))
        spec = dl.Grid(n=(8, 6), d=d, c=c, unit="m")

        resized = spec.resize((10, 8))
        downsampled = spec.downsample((2, 3))
        oversampled = spec.oversample((2, 3))
        resampled = spec.resample((12, 10), d / 2)

        assert resized.n == (10, 8)
        assert resized.d.shape == resized.c.shape == (2, 2)
        assert downsampled.n == (4, 2)
        assert np.allclose(downsampled.d, d * np.asarray((2, 3)))
        assert np.allclose(downsampled.c, c)
        assert oversampled.n == (16, 18)
        assert np.allclose(oversampled.d, d / np.asarray((2, 3)))
        assert np.allclose(oversampled.c, c)
        assert np.allclose(oversampled.fov, spec.fov)
        assert resampled.n == (12, 10)
        assert np.allclose(resampled.d, d / 2)
        assert np.allclose(resampled.c, c)

    def test_incomplete_spec_contracts(self):
        empty = dl.Grid()
        size_only = dl.Grid(n=4)
        spacing_only = dl.Grid(d=0.1)

        assert empty.ndim == size_only.ndim == spacing_only.ndim == 2
        with pytest.raises(ValueError, match="n and d"):
            size_only.downsample(2)
        with pytest.raises(ValueError, match="n and d"):
            size_only.oversample(2)
        with pytest.raises(ValueError, match="n must"):
            _ = spacing_only.shape
        with pytest.raises(ValueError, match="n must"):
            _ = spacing_only.axes
        with pytest.raises(ValueError, match="n must"):
            _ = spacing_only.coordinates
        with pytest.raises(ValueError, match="n and d"):
            _ = size_only.fov
        with pytest.raises(ValueError, match="d must"):
            size_only.resize((4,)).axes

    def test_units_and_differentiation(self):
        spec = dl.Grid(n=(4, 6), d=(2.0, 3.0), unit="mm")

        assert np.max(np.abs(spec.coordinates)) < 0.01
        assert np.allclose(spec.fov, np.asarray((8.0, 18.0)))
        assert_differentiable(lambda value: value.coordinates, spec)

    def test_diameter_sampling(self):
        spec = dl.Grid(n=(4, 6), diam=(2.0, 3.0), unit="m")

        assert np.allclose(spec.d, np.asarray((0.5, 0.5)))
        assert all(np.isclose(axis.mean(), 0.0) for axis in spec.axes)
        assert spec.coordinates.shape == (2, 6, 4)

    def test_rectangular_axes_contract(self):
        spec = dl.Grid(n=(6, 4), d=(0.2, 0.3), unit="m")

        assert tuple(axis.shape for axis in spec.axes) == ((6,), (4,))
        assert tuple(x.shape for x in spec.axes) == ((6,), (4,))
        assert np.allclose(spec.extent(), np.asarray((-0.6, 0.6, -0.6, 0.6)))

    def test_extent_units(self):
        spec = dl.Grid(n=4, d=2.0, c=1.0, unit="mm")

        native = assert_jittable(lambda value: value.extent(), spec)
        assert np.allclose(native, np.asarray((-3.0, 5.0, -3.0, 5.0)))
        assert np.allclose(spec.extent(unit="m"), 1e-3 * native)

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
            ({"unit": 1}, TypeError),
        ],
    )
    def test_validation(self, kwargs, error):
        with pytest.raises(error):
            dl.Grid(**kwargs)

        with pytest.raises(ValueError):
            dl.ResizeSpec(pad=0)

        with pytest.raises(ValueError):
            dl.ResizeSpec(n=4, pad=2)


class TestSamplingOperations:
    @pytest.mark.parametrize("width", [None, 20.0])
    def test_singleton_bandpass_resize_keeps_its_interval(self, width):
        bandpass = dlu.BandPass(wl=500.0, delta_wl=width, n=1, unit="nm")
        resized = assert_jittable(
            lambda value: (value.resize(1), value.resize_samples(1)), bandpass
        )

        for result in resized:
            assert_tree_allclose(result, bandpass)
            assert result.d == width

    @pytest.mark.parametrize("old, new", [(4, 6), (6, 4), (5, 7), (7, 5)])
    def test_resize_samples_preserves_coordinates(self, old, new):
        axis = dlu.Axis(n=old, d=0.2, c=0.7, unit="mm")
        resized = assert_jittable(lambda value: value.resize_samples(new), axis)
        coordinates = dlu.resize(axis.axes[0], (new,), fill=np.nan)
        retained = np.isfinite(coordinates)

        assert isinstance(resized, dlu.Axis)
        assert resized.n == (new,)
        assert resized.unit == axis.unit
        assert np.allclose(resized.axes[0][retained], coordinates[retained])
        assert axis.n == (old,)

    def test_downsampling_matches_block_centres_and_preserves_extent(self):
        axis = dlu.Axis(n=6, d=2.0, c=3.0, unit="mm")
        coarse, fine = assert_jittable(
            lambda value: (value.downsample(2), value.oversample(2)), axis
        )

        expected = axis.axes[0].reshape(3, 2).mean(-1)
        assert coarse.n == (3,)
        assert fine.n == (12,)
        assert np.allclose(coarse.axes[0], expected)
        assert np.allclose(coarse.fov, axis.fov)
        assert np.allclose(fine.fov, axis.fov)
        assert np.allclose(fine.downsample(2).axes[0], axis.axes[0])

        gradient = assert_differentiable(
            lambda value: value.downsample(2).coordinates, axis
        )
        assert np.allclose(gradient.d, 32e-6, atol=1e-10)
        assert np.allclose(gradient.c, 18e-6, atol=1e-10)

    def test_sampling_preserves_independent_axis_batches(self):
        axis = dlu.Axis(
            n=6,
            d=np.asarray([[0.2], [0.4]]),
            c=np.asarray([[1.0], [2.0]]),
            unit="mm",
        )
        sampled = assert_jittable(lambda value: value.downsample(2), axis)
        expected = axis.axes[0].reshape(2, 3, 2).mean(-1)

        assert sampled.coordinates.shape == (2, 1, 3)
        assert np.allclose(sampled.axes[0], expected)
        assert np.allclose(sampled.c, axis.c)

    @pytest.mark.parametrize("sampling", [dlu.Axis, dlu.Grid])
    def test_resizing_keeps_partial_sampling(self, sampling):
        shape_only = sampling(n=4)
        resized = assert_jittable(lambda value: value.resize_samples(6), shape_only)
        assert resized.d is None
        assert resized.n == (6,) * shape_only.ndim

        defined_later = sampling(n=4, d=zdx.StateRef("spacing"))
        resized = defined_later.resize_samples(6)
        assert isinstance(resized.d, zdx.Expression)
        resolved = assert_jittable(
            lambda state: resized.resolve(state=state), zdx.State(spacing=0.5)
        )
        assert np.allclose(resolved.d_si, 0.5)

    def test_geometric_resize_and_sample_resize_are_distinct(self):
        axis = dlu.Axis(n=4, d=1.0, c=2.0, unit=None)
        assert np.allclose(axis.resize(5).axes[0], np.arange(5.0))
        with pytest.raises(ValueError, match="parity"):
            axis.resize_samples(5)
        with pytest.raises(ValueError, match="divide"):
            axis.downsample(3)


class TestTransforms:
    @pytest.mark.parametrize("angle", [-0.3, 0.3, np.pi / 2])
    def test_rotation_turns_coordinates_anticlockwise(self, angle):
        coordinates = dl.Grid((7, 5), 0.2).coordinates
        transform = dl.Affine(rotation=angle)
        values = assert_jittable(lambda op: op(coordinates), transform)

        x, y = coordinates
        expected = np.stack(
            [
                np.cos(angle) * x - np.sin(angle) * y,
                np.sin(angle) * x + np.cos(angle) * y,
            ]
        )
        assert np.allclose(values, expected, atol=1e-7)

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
            dl.Distortion(order=2),
            dl.TransformChain(
                [dl.Affine(translation=[0.1, 0.0]), dl.Distortion(order=2)]
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

        distorted = dl.Distortion(order=2)
        assert_differentiable(
            lambda distortion: distorted.set(distortion=distortion)(coordinates),
            distorted.distortion,
        )

    def test_transform_requires_external_grid(self, coordinates):
        transform = dl.Affine(translation=[0.1, 0.0])

        with pytest.warns(DeprecationWarning, match="dLux 0.17.0"):
            applied = transform.apply(coordinates)
        assert np.allclose(transform(coordinates), applied)
        spec = dl.Grid(n=6, d=1 / 6, unit="m")
        assert np.allclose(transform(spec), transform(spec.coordinates))
        with pytest.raises((TypeError, ValueError)):
            dl.Affine()()
        with pytest.raises(ValueError, match="Provide coordinates"):
            dl.Affine()(None)
        with pytest.raises(ValueError, match="shape"):
            dl.Affine()(np.ones((2, 6)))

        with pytest.raises(ValueError, match="2 dimensions"):
            dl.Grid(n=(4, 5, 6), d=0.1)

    def test_transform_dictionary(self, coordinates):
        transforms = {
            "translate": dl.Affine(translation=[0.1, 0.0]),
            "rotate": dl.Affine(rotation=0.1),
        }
        chain = dl.TransformChain(transforms)

        assert list(chain.layers) == list(transforms)
        assert_jittable(chain, coordinates)

    def test_vectorised_distortion(self, coordinates):
        transform = dl.Distortion(order=2, distortion=np.zeros((3, 2, 5)))

        output = assert_jittable(lambda value: value(coordinates), transform)
        assert output.shape == (3,) + coordinates.shape

    def test_batched_affine_coordinates(self, coordinates):
        mapped_coordinates = np.stack((coordinates, coordinates + 0.1))
        translation = np.asarray(((0.1, 0.0), (-0.1, 0.2)))
        rotation = np.asarray((0.2, -0.1))
        transform = dl.Affine(translation=translation, rotation=rotation)

        output = assert_jittable(lambda value: transform(value), mapped_coordinates)
        matrix, offset = transform.coeffs

        assert output.shape == mapped_coordinates.shape
        assert matrix.shape == (2, 2, 2)
        assert offset.shape == (2, 2)

    def test_paired_vectorised_distortion(self, coordinates):
        base = dl.Distortion(orders=(1, 2))
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
            lambda: dl.Affine(scale=0.0),
            lambda: dl.Affine(order=("rotation", "rotation")),
            lambda: dl.AffineMap(matrix=np.ones((3, 3))),
            lambda: dl.AffineMap(offset=np.ones(3)),
            lambda: dl.Affine(translation=np.ones(3)),
            lambda: dl.Distortion(powers=np.ones((3, 2))),
            lambda: dl.Distortion(order=2, orders=[2]),
            lambda: dl.Distortion(orders=[]),
            lambda: dl.Distortion(order=2, distortion=np.ones((2, 2))),
        ],
    )
    def test_validation(self, constructor):
        with pytest.raises(ValueError):
            constructor()


def test_transformed_grid_identity():
    grid = dl.Grid((3, 5), 1)
    assert np.array_equal(grid.transformed(), grid.coordinates)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: dl.Grid(d=1).frequency(),
        lambda: dl.Grid(3, 1, unit="rad").frequency(),
        lambda: dl.Grid.from_axes(([1.0], [1.0, 2.0])),
        lambda: dl.Grid(3, 1).transformed(np.eye(2)),
        lambda: dl.Grid(3).d_si,
        lambda: dl.Grid(d=1).resize_samples(3),
        lambda: dl.Axis([1.0, 2.0], n=2),
        lambda: dl.Axis([]),
    ],
)
def test_grid_validation_tails(operation):
    """Coverage tails: partial definitions and invalid sampling requests."""
    with pytest.raises((TypeError, ValueError)):
        operation()
