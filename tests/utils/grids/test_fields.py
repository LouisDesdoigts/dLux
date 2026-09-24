"""Sampling and basic operations of numerical real and complex fields."""

import jax
import equinox as eqx
import jax.numpy as np
import pytest
import zodiax as zdx
import dLux.utils as dlu

from tests.helpers import assert_jittable


def test_field_components_and_regular_grid_resampling():
    grid = dlu.Grid((5, 5), 1)
    x, y = grid.coordinates
    value = x + 1j * y
    field = dlu.Field(value, grid)
    assert np.allclose(field.real, x)
    assert np.allclose(field.imag, y)
    assert np.allclose(field.intensity, x * x + y * y)
    target = dlu.Grid((3, 3), 0.5)
    resampled = assert_jittable(lambda model: model.resample(target), field)
    a, b = target.coordinates
    assert np.allclose(resampled.array, a + 1j * b, atol=1e-6)
    assert resampled.grid.n == target.n


def test_field_axis_sampling_and_leading_batch_convolution():
    field = dlu.Field(np.asarray([1.0, 3.0, 5.0]), dlu.Axis([0.0, 1.0, 2.0], unit=None))
    assert np.allclose(field.sample(np.asarray([0.5, 1.5])), np.asarray([2, 4]))
    grid = dlu.Grid((5, 5), 1)
    array = np.arange(25.0).reshape(5, 5)
    fields = dlu.Field(np.stack([array, 2 * array]), grid)
    identity = np.zeros((3, 3)).at[1, 1].set(1)
    assert np.allclose(fields.convolve(identity).array, fields.array, atol=1e-5)


def test_field_dimension_follows_grid_or_axis_type():
    values = np.arange(50.0).reshape(2, 5, 5)
    image = dlu.Field(values, dlu.Grid(n=5, d=1))
    assert image.grid.ndim == 2
    assert image.spatial_shape == (5, 5)
    assert image.batch_ndim == 1
    assert image.coordinates.shape == (2, 5, 5)

    curves = dlu.Field(values, dlu.Axis(n=5, d=1, unit=None))
    assert curves.grid.ndim == 1
    assert curves.spatial_shape == (5,)
    assert curves.batch_ndim == 2


def test_axis_field_resampling_infers_counts_and_retains_centre():
    axis = dlu.Axis(n=5, d=1.0, c=2.0, unit=None)
    field = dlu.Field(axis.values**3, axis)
    target = dlu.Axis(d=0.5, c=2.0, unit=None)

    sampled = assert_jittable(lambda value: value.resample(target), field)
    assert isinstance(sampled.grid, dlu.Axis)
    assert sampled.grid.n == (5,)
    assert np.allclose(sampled.array, sampled.grid.values**3, atol=1e-6)
    smaller = assert_jittable(lambda value: value.scale_to(3, 0.5), field)
    assert smaller.grid.n == (3,)
    assert np.allclose(smaller.grid.c, axis.c)
    assert np.allclose(smaller.array, smaller.grid.values**3, atol=1e-6)


def test_field_resampling_forwards_context_to_its_coordinate_map():
    source = dlu.Grid(5, d=1.0)
    target = dlu.Grid(3, d=0.5)
    field = dlu.Field(source.coordinates[0], source)
    operation = dlu.Interp(
        grid_out=target, shift=zdx.StateRef("shift"), method="linear"
    )
    state = zdx.State(shift=[0.2, 0.0])

    sampled = assert_jittable(
        lambda value: field.resample(operation, state=value), state
    )
    assert np.allclose(sampled.array, target.coordinates[0] - 0.2, atol=1e-6)


def test_axis_field_padding_and_cropping_retain_sample_coordinates():
    axis = dlu.Axis(n=5, d=0.5, c=2.0, unit=None)
    values = np.stack((axis.values, 1j * axis.values))
    field = dlu.Field(values, axis)

    padded = assert_jittable(lambda value: value.resize(9), field)
    assert isinstance(padded.grid, dlu.Axis)
    assert padded.array.shape == (2, 9)
    assert np.array_equal(padded.array[:, 2:-2], values)
    assert np.all(padded.array[:, :2] == 0)
    assert np.all(padded.array[:, -2:] == 0)
    assert np.allclose(padded.grid.values[2:-2], axis.values)

    cropped = assert_jittable(lambda value: value.resize(3), padded)
    assert np.array_equal(cropped.array, values[:, 1:-1])
    assert np.allclose(cropped.grid.values, axis.values[1:-1])
    with pytest.raises(ValueError, match="parity"):
        field.resize(8)


@pytest.mark.parametrize("ndim", [1, 2])
def test_field_padding_can_retain_unspecified_spacing(ndim):
    grid = dlu.Axis(n=3) if ndim == 1 else dlu.Grid(n=3)
    field = dlu.Field(np.ones(grid.shape), grid)
    padded = assert_jittable(lambda value: value.resize(5), field)
    assert padded.grid.d is None
    assert padded.array.shape == (5,) * ndim
    assert np.allclose(padded.array.sum(), field.array.sum())


@pytest.mark.parametrize("complex_values", [False, True])
def test_axis_field_downsampling_averages_samples_independently(complex_values):
    axis = dlu.Axis(n=6, d=0.5, c=2.0, unit=None)
    values = np.stack((np.arange(6.0), np.arange(6.0) ** 2))
    if complex_values:
        values = values * (1 + 2j)
    field = dlu.Field(values, axis)
    output = assert_jittable(lambda value: value.downsample(2), field)
    expected = np.stack([values[:, i : i + 2].mean(-1) for i in range(0, 6, 2)], -1)
    assert np.allclose(output.array, expected)
    assert np.allclose(output.grid.values, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(output.grid.fov, axis.fov)
    assert np.allclose(field.downsample(2, mean=False).array, 2 * expected)

    if not complex_values:
        gradient = jax.jit(
            jax.grad(lambda array: dlu.Field(array, axis).downsample(2).array.sum())
        )(values)
        assert np.allclose(gradient, 0.5)


@pytest.mark.parametrize("mode", ["same", "full", "valid"])
def test_axis_convolution_matches_reference_and_tracks_coordinates(mode):
    axis = dlu.Axis(n=6, d=0.5, c=2.0, unit=None)
    values = np.stack((np.arange(6.0), np.arange(6.0) ** 2))
    field = dlu.Field(values * (1 + 1j), axis)
    kernel = np.array([0.25, 0.5, 0.25])
    output = assert_jittable(
        lambda value: value.convolve(kernel, mode=mode, method="direct"), field
    )
    expected = np.stack([np.convolve(value, kernel, mode) for value in field.array])
    assert np.allclose(output.array, expected)
    assert isinstance(output.grid, dlu.Axis)
    assert np.allclose(output.grid.c, axis.c)
    assert np.allclose(output.grid.d, axis.d)
    if mode == "full":
        assert np.allclose(output.grid.values[1:-1], axis.values)
    elif mode == "valid":
        assert np.allclose(output.grid.values, axis.values[1:-1])
    else:
        assert np.allclose(output.grid.values, axis.values)


@pytest.mark.parametrize("mode", ["same", "full", "valid"])
@pytest.mark.parametrize("ndim", [1, 2])
def test_convolution_rejects_half_sample_kernel_centres(mode, ndim):
    grid = dlu.Axis(n=6, d=1) if ndim == 1 else dlu.Grid(n=6, d=1)
    field = dlu.Field(np.ones(grid.shape), grid)
    kernel = np.ones((2,) if ndim == 1 else (3, 2))
    with pytest.raises(ValueError, match="odd kernel"):
        field.convolve(kernel, mode=mode)


def test_field_views_and_flip_preserve_sampling():
    values = np.array([[1 + 2j, 2 - 3j], [4j, -1 + 0j]])
    field = dlu.Field(values, dlu.Grid(2, 0.5))
    assert np.allclose(field.amplitude, np.abs(values))
    assert np.allclose(field.phase, np.angle(values))
    assert np.array_equal(field.complex, np.stack((values.real, values.imag)))
    assert np.allclose(field.polar, np.stack((np.abs(values), np.angle(values))))
    assert all(np.array_equal(a, b) for a, b in zip(field.xs, field.axes))
    assert field.npixels == 2
    assert np.array_equal(field.flip(-1).array, values[:, ::-1])
    assert eqx.tree_equal(field.flip(-1).grid, field.grid)


@pytest.mark.parametrize(
    "method",
    [
        "__add__",
        "__sub__",
        "__mul__",
        "__truediv__",
        "__iadd__",
        "__isub__",
        "__imul__",
        "__itruediv__",
    ],
)
@pytest.mark.parametrize("operand", ["scalar", "field", "none"])
def test_field_arithmetic_is_immutable(method, operand):
    import operator

    grid = dlu.Grid(2, 1)
    values = np.arange(1.0, 5.0).reshape(2, 2)
    field = dlu.Field(values, grid)
    other = {
        "scalar": 2.0,
        "field": dlu.Field(np.full((2, 2), 2.0), grid),
        "none": None,
    }[operand]
    output = getattr(field, method)(other)
    name = method.replace("__", "").removeprefix("i")
    expected = values if operand == "none" else getattr(operator, name)(values, 2.0)
    assert np.array_equal(output.array, expected)
    assert np.array_equal(field.array, values)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: dlu.Field(np.ones((3, 3)), dlu.ResizeSpec(3)),
        lambda: dlu.Field(np.ones(3), dlu.Grid(3, 1)),
        lambda: dlu.Field(np.ones((3, 3)), dlu.Grid(3, 1)).convolve(np.ones(3)),
        lambda: dlu.Field(np.ones((3, 3)), dlu.Grid(3, 1)) + "invalid",
    ],
)
def test_field_validation_tails(operation):
    """Coverage tails: incompatible sampling, kernels and operands."""
    with pytest.raises((TypeError, ValueError)):
        operation()


@pytest.mark.parametrize("method", ["sample", "scale_to", "interpolate", "rotate"])
def test_field_sampling_entry_points(method):
    grid = dlu.Grid(5, 1)
    values = grid.coordinates[0] + 2 * grid.coordinates[1]
    field = dlu.Field(values, grid)
    if method == "sample":
        result = field.sample(grid.coordinates)
    elif method == "scale_to":
        result = field.scale_to(5, 1).array
    elif method == "interpolate":
        result = field.interpolate(dlu.Affine()).array
    else:
        result = field.rotate(0).array
    assert np.allclose(result, values, atol=1e-6)
