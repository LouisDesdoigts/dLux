"""Interpolation on trailing sample axes, context and prepared snapshots."""

import equinox as eqx
import jax
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux.utils as dlu

from tests.helpers import assert_jittable


def test_array_function_uses_shared_numerics_without_constructing_an_expression():
    from dLux.utils.numerical.interpolation import interp

    assert dlu.interp is interp
    grid = dlu.Grid((5, 4), (0.4, 0.6))
    queries = dlu.Grid((3, 3), 0.2, c=(0.1, 0.1)).coordinates
    x, y = grid.coordinates
    image = x + 2 * y + 1j * (3 * x - y)
    image = np.stack((image, 2 * image))
    qx, qy = queries
    expected = qx + 2 * qy + 1j * (3 * qx - qy)
    expected = np.stack((expected, 2 * expected))

    sampled = assert_jittable(interp, image, grid.coordinates, queries)
    assert np.allclose(sampled, expected, atol=1e-6)
    assert np.allclose(
        dlu.Interp(image, grid, method="linear").sample(queries), expected, atol=1e-6
    )
    gradient = jax.grad(
        lambda gain: interp(gain * image, grid.coordinates, queries).real.sum()
    )(1.0)
    assert np.allclose(gradient, expected.real.sum(), atol=1e-6)


def test_array_function_retains_complex_fill_and_linear_default():
    grid = dlu.Grid((3, 3), 1)
    x, y = grid.coordinates
    image = x**2 + 1j * y**2
    query = np.asarray([[[0.5, 2.0]], [[0.5, 2.0]]])
    sampled = dlu.interp(image, grid.coordinates, query, fill=2 + 3j)
    assert np.allclose(sampled, np.asarray([[0.5 + 0.5j, 2 + 3j]]))


def test_interpolation_preserves_batches_before_arbitrary_query_shape():
    samples = np.asarray([[0.0, 2.0, 4.0], [10.0, 12.0, 14.0]])
    operation = dlu.Interp(samples, [0.0, 1.0, 2.0], ndim=1, method="linear")
    query = np.asarray([[0.0, 0.5], [1.0, 1.5]])
    result = assert_jittable(lambda q: operation.sample((q,)), query)
    assert result.shape == (2, 2, 2)
    assert np.allclose(result, np.stack((2 * query, 10 + 2 * query)))
    assert isinstance(operation.resolve(), dlu.Interp)
    assert np.allclose(operation.resolve(coordinates=(query,)), result)
    assert np.allclose(operation.apply(), samples)


def test_interpolation_contextual_query_and_values_have_gradients():
    operation = dlu.Interp(
        zdx.StateRef("samples"), [0.0, 1.0, 2.0], ndim=1, method="linear"
    )
    state = zdx.State(time=0.25, samples=[0.0, 4.0, 0.0])
    assert np.allclose(
        operation.resolve(coordinates=zdx.StateRef("time"), state=state), 1.0
    )
    gradient = jax.grad(lambda q: operation.sample(q, state=state))(0.25)
    assert np.allclose(gradient, 4.0)
    sample_gradient = jax.jit(
        jax.grad(lambda x: operation.sample(state.time, state=state.set(samples=x)))
    )(state.samples)
    assert np.allclose(sample_gradient, np.asarray([0.75, 0.25, 0.0]))


def test_interpolation_extrapolation_and_snapshot():
    operation = dlu.Interp(
        [0.0, 1.0],
        [0.0, 1.0],
        ndim=1,
        method="linear",
        fill=-2.0,
    )
    assert np.allclose(
        operation.sample([-1.0, 0.5, 2.0]), np.asarray([-2.0, 0.5, -2.0])
    )
    prepared = operation.prepare().set(array=np.asarray([0.0, 4.0]))
    assert np.allclose(prepared.sample(0.5), 0.5)
    assert np.allclose(prepared.prepare().sample(0.5), 2.0)
    assert np.allclose(eqx.filter_jit(lambda x: x.sample(0.5))(prepared), 0.5)


def test_expression_source_keeps_its_sampling_when_destination_changes():
    source = dlu.Grid(7, d=0.5)
    target = dlu.Grid(5, d=0.3)
    profile = dlu.Gaussian(std=0.6, grid=source)
    operation = dlu.Interp(profile, source, target)
    reference = dlu.Interp(profile.resolve(), source, target)

    sampled = assert_jittable(lambda value: value.sample(target.coordinates), operation)
    assert np.allclose(sampled, reference.sample(target.coordinates), atol=1e-6)
    assert np.allclose(operation.apply(), reference.apply(), atol=1e-6)
    assert np.allclose(operation.resolve(grid=target), reference.apply(), atol=1e-6)


def test_prepared_expression_can_query_points_without_resolving_source_again():
    grid = dlu.Grid(7, d=0.5)
    profile = dlu.Gaussian(std=zdx.StateRef("width"), grid=grid)
    state = zdx.State(width=0.6)
    operation = dlu.Interp(profile, grid).prepare(state=state)
    points = np.asarray([[0.25, 0.5], [0.0, 0.25]])
    reference = dlu.Interp(profile.resolve(state=state), grid)

    # The snapshot no longer needs the source's State or full coordinate grid.
    sampled = assert_jittable(lambda value: value.sample(tuple(points)), operation)
    assert np.allclose(sampled, reference.sample(tuple(points)), atol=1e-6)
    assert np.allclose(operation.sample(tuple(points[:, 0])), sampled[0], atol=1e-6)

    changed = operation.set(array=dlu.Gaussian(std=1.2, grid=grid))
    assert np.allclose(changed.sample(tuple(points)), sampled, atol=1e-6)
    assert not np.allclose(changed.prepare().sample(tuple(points)), sampled, atol=1e-3)


def test_preparing_expression_inside_jit_retains_source_gradients():
    grid = dlu.Grid(5, d=0.5)

    def sample(width):
        profile = dlu.Gaussian(std=width)
        return dlu.Interp(profile, grid).prepare().sample((0.5, 0.0))

    # Query an original knot, where the Gaussian value and derivative are exact.
    width = 0.7
    derivative = jax.jit(jax.grad(sample))(width)
    expected = np.exp(-0.5 * (0.5 / width) ** 2) * 0.5**2 / width**3
    assert np.allclose(derivative, expected, rtol=1e-5)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_expression_source_is_sampled_on_raw_rectilinear_axes(ndim):
    knots = np.asarray([-2.0, -0.5, 0.0, 1.0, 2.0])
    axes = tuple(knots + i for i in range(ndim))
    query = tuple(np.asarray([0.25, 0.75]) + i for i in range(ndim))
    coeffs = np.arange(1.0, ndim + 1)
    source = dlu.Polynomial(powers=2 * np.eye(ndim, dtype=int), coeffs=coeffs)
    operation = dlu.Interp(source, axes[0] if ndim == 1 else axes, ndim=ndim)

    sampled = assert_jittable(lambda value: value.sample(query), operation)
    expected = sum((i + 1) * values**2 for i, values in enumerate(query))
    assert np.allclose(sampled, expected, rtol=1e-5, atol=1e-6)
    assert np.allclose(operation.prepare().sample(query), expected, rtol=1e-5)


def test_expression_source_supports_broadcast_batches_of_raw_axes():
    x = np.linspace(-1.0, 1.0, 5)
    y = np.linspace(-1.0, 1.0, 4)
    x = np.stack((x, 1.2 * x + 0.1))
    source = dlu.Polynomial(powers=[[1, 0], [0, 1]], coeffs=[2.0, 3.0])
    operation = dlu.Interp(source, (x, y))
    query = (np.asarray([0.2, 0.4]), np.asarray([-0.3, 0.2]))

    sampled = assert_jittable(lambda value: value.sample(query), operation)
    expected = np.broadcast_to(2 * query[0] + 3 * query[1], (2, 2))
    assert np.allclose(sampled, expected, atol=1e-6)


@pytest.mark.parametrize("source_sampling", ["axis", "array", "tuple"])
def test_expression_source_pairs_1d_sampling_and_coefficient_batches(source_sampling):
    axis = dlu.Axis(
        n=5,
        d=np.asarray([[0.5], [0.4]]),
        c=np.asarray([[-0.2], [0.3]]),
        unit=None,
    )
    knots = axis.values
    if source_sampling == "axis":
        sampling = axis
    else:
        # Raw vectors can be irregular and still retain the same batch contract.
        knots = knots + np.asarray([0.0, 0.05, -0.05, 0.05, 0.0])
        sampling = (knots,) if source_sampling == "tuple" else knots
    coefficients = np.asarray([[1.0, 2.0, 0.5], [-2.0, -1.0, 1.5]])
    target = dlu.Axis(n=3, d=0.2, c=0.1, unit=None)
    query = target.values
    expected = np.stack(
        [
            offset + slope * query + curve * query**2
            for offset, slope, curve in coefficients
        ]
    )
    source = dlu.Polynomial(degree=2, coeffs=coefficients)
    operation = dlu.Interp(source, sampling, grid_out=target, ndim=1)

    sampled = assert_jittable(lambda value: value.apply(), operation)
    assert sampled.shape == expected.shape == (2, 3)
    assert np.allclose(sampled, expected, atol=1e-6)
    assert np.allclose(operation.sample(query), expected, atol=1e-6)
    assert np.allclose(operation.prepare().sample(query), expected, atol=1e-6)

    gradient = jax.jit(
        jax.grad(lambda values: operation.set("array.coeffs", values).apply().sum())
    )(coefficients)
    expected_gradient = np.asarray([query.size, query.sum(), (query**2).sum()])
    assert np.allclose(gradient, np.broadcast_to(expected_gradient, (2, 3)), atol=1e-6)


def test_batched_1d_source_expressions_receive_explicit_coordinate_components():
    class LinearSamples(zdx.Expression):
        gain: jax.Array

        def evaluate(self, *, coordinates):
            return self.gain[..., None] * coordinates[..., 0, :]

    knots = np.asarray([[-1.0, -0.3, 0.2, 1.0], [-0.5, -0.1, 0.4, 0.7]])
    source = LinearSamples(np.asarray([2.0, 3.0]))
    operation = dlu.Interp(source, knots, ndim=1)
    query = np.asarray([-0.2, 0.3])

    sampled = assert_jittable(lambda value: value.sample(query), operation)
    expected = source.gain[:, None] * query
    assert sampled.shape == (2, 2)
    assert np.allclose(sampled, expected, atol=1e-6)


def test_direct_sampling_populates_links_before_separating_source_and_query():
    grid = dlu.Grid(5, d=0.5)
    width = zdx.Linked(0.5, key="width")
    operation = dlu.Interp(
        dlu.Gaussian(std=width), grid, fill=width.defer(), grid_out=dlu.Grid(3, d=2.0)
    )
    points = np.asarray([[0.0, 3.0], [0.0, 3.0]])

    sampled = assert_jittable(lambda value: value.sample(tuple(points)), operation)
    assert np.allclose(sampled, np.asarray([1.0, 0.5]), atol=1e-6)
    prepared = operation.prepare()
    assert np.allclose(prepared.sample(tuple(points)), sampled, atol=1e-6)
    assert np.allclose(operation.apply(), operation.resolve(), atol=1e-6)

    # Update the linked definition, then explicitly prepare a fresh snapshot.
    updated = operation.set("array.std.value", 0.8).prepare()
    assert np.allclose(updated.sample(tuple(points)), np.asarray([1.0, 0.8]), atol=1e-6)
    assert np.allclose(prepared.sample(tuple(points)), sampled, atol=1e-6)


def test_default_spline_and_complex_samples():
    knots = np.linspace(-1, 1, 5)
    operation = dlu.Interp(knots**3 + 1j * knots**2, knots, ndim=1)
    query = np.asarray([-0.7, 0.1, 0.6])
    assert np.allclose(operation.sample(query), query**3 + 1j * query**2, atol=1e-6)


def test_two_dimensional_rotation_preserves_released_direction():
    grid = dlu.Grid((7, 7), 1)
    image = np.zeros((7, 7)).at[3, 5].set(1.0)
    operation = dlu.Interp(image, grid, rotation=np.pi / 2, method="linear")
    values = assert_jittable(lambda op: op.resolve(), operation)
    assert np.allclose(values, np.rot90(image, 1), atol=1e-6)
    field = dlu.Field(image, grid)
    rotated = assert_jittable(lambda obj: obj.rotate(np.pi / 2, method="linear"), field)
    assert np.allclose(rotated.array, values, atol=1e-6)


def test_resampling_preserves_values_without_area_scaling():
    grid = dlu.Grid((4, 4), 1)
    operation = dlu.Interp(np.ones((4, 4)), grid, grid.oversample(2))
    values = assert_jittable(lambda op: op.apply(), operation)
    assert values.shape == (8, 8)
    assert np.allclose(values, 1.0)
    assert np.allclose(values.sum(), 64.0)


@pytest.mark.parametrize("knots,values", [([0.0], [1.0]), ([0.0, 1.0], [1.0])])
def test_interpolation_rejects_invalid_sample_data_when_prepared(knots, values):
    with pytest.raises(ValueError):
        dlu.Interp(values, knots, ndim=1).prepare()


@pytest.mark.parametrize("ndim", [1, 3])
def test_default_sampling_retains_dimensions_batches_and_gradients(ndim):
    counts = (5,) if ndim == 1 else (5, 4, 3)
    array = np.zeros(counts[::-1])
    for i, n in enumerate(counts):
        shape = [1] * ndim
        shape[-1 - i] = n
        axis = (np.arange(n) - (n - 1) / 2).reshape(shape)
        array = array + (i + 1) * axis
    values = np.stack((array, 2 * array))
    operation = dlu.Interp(values, ndim=ndim, method="linear")
    queries = tuple(np.asarray([-0.5, 0.25]) / (i + 1) for i in range(ndim))
    expected = sum((i + 1) * query for i, query in enumerate(queries))
    expected = np.stack((expected, 2 * expected))

    assert np.allclose(operation.apply(), values, atol=1e-6)
    result = assert_jittable(lambda model: model.sample(queries), operation)
    assert np.allclose(result, expected, atol=1e-6)
    assert np.allclose(operation.prepare().sample(queries), expected, atol=1e-6)
    derivative = jax.jit(
        jax.grad(lambda gain: operation.set(array=gain * values).sample(queries).sum())
    )(1.0)
    assert np.allclose(derivative, expected.sum(), atol=1e-6)


@pytest.mark.parametrize("ndim", [2, 3])
def test_tuple_points_and_batched_grid_queries_have_explicit_shapes(ndim):
    axes = tuple(np.linspace(-1, 1, 5) for _ in range(ndim))
    components = np.meshgrid(*axes[::-1], indexing="ij")[::-1]
    values = sum((i + 1) * x for i, x in enumerate(components))
    operation = dlu.Interp(np.stack((values, 2 * values)), axes, ndim=ndim)
    points = tuple(np.asarray([0.1, -0.2, 0.3]) for _ in range(ndim))
    sampled = assert_jittable(lambda model: model.sample(points), operation)
    expected = sum((i + 1) * x for i, x in enumerate(points))
    assert sampled.shape == (2, 3)
    assert np.allclose(sampled, np.stack((expected, 2 * expected)), atol=2e-6)
    assert operation.sample(tuple(0.1 for _ in range(ndim))).shape == (2,)
    # Explicit leading query batches pair with the two source batches.
    query = np.stack([np.full((ndim,) + (2,) * ndim, q) for q in (0.1, 0.2)])
    result = assert_jittable(lambda model: model.sample(query), operation)
    assert result.shape == (2,) + (2,) * ndim
    assert np.allclose(result[0], expected[0], atol=2e-6)
    assert np.allclose(result[1], 4 * expected[0], atol=2e-6)
    for ambiguous in (np.zeros(ndim), np.zeros((ndim, 3))):
        with pytest.raises(ValueError, match="tuple"):
            operation.sample(ambiguous)


@pytest.mark.parametrize(
    "operation,error",
    [
        (lambda: dlu.Interp(ndim=4), ValueError),
        (lambda: dlu.Interp(grid_out=np.ones(2)), TypeError),
        (lambda: dlu.Interp(ndim=1, rotation=1), ValueError),
        (lambda: dlu.Interp(transform=dlu.Affine(), rotation=1), ValueError),
        (lambda: dlu.Interp().axes, ValueError),
        (lambda: dlu.Interp().evaluate(), zdx.Unresolved),
        (lambda: dlu.Interp(np.ones((3, 3))).evaluate(), zdx.Unresolved),
        (lambda: dlu.Interp(grid_out=dlu.Axis(n=3)).output_grid((3, 3)), ValueError),
        (lambda: dlu.Interp().prepare(), zdx.Unresolved),
        (lambda: dlu.Interp(np.ones((3, 3)), dlu.Axis(n=3, d=1)).prepare(), ValueError),
        (lambda: dlu.Interp(np.ones(3), [0j, 1j, 2j], ndim=1).prepare(), TypeError),
        (lambda: dlu.Interp(np.ones((3, 3)), (np.arange(3.0),)).prepare(), ValueError),
        (lambda: dlu.Interp(np.ones(3), [0.0, 1.0], ndim=1).prepare(), ValueError),
        (lambda: dlu.Interp(np.ones((3, 3))).sample((np.ones(3),)), ValueError),
        (lambda: dlu.Interp(np.ones((3, 3))).sample(np.ones((2, 4))), ValueError),
        (lambda: dlu.Interp(np.ones((3, 3))).sample((1j, 0j)), TypeError),
    ],
)
def test_interpolation_validation_tails(operation, error):
    """Coverage tails: rejected definitions and point/grid query layouts."""
    with pytest.raises(error):
        operation()


def test_unbound_call_waits_for_sampling():
    operation = dlu.Interp(np.ones((3, 3)))
    assert isinstance(operation(), dlu.Interp)


def test_default_fill_extrapolates_exact_requested_points():
    operation = dlu.Interp([0.0, 2.0, 4.0], [0.0, 1.0, 2.0], ndim=1, method="linear")
    query = np.array([-0.5, -1e-7, 2.0 + 1e-6, 2.5])
    assert np.allclose(operation.sample(query), 2 * query, atol=1e-8)


@pytest.mark.parametrize("cartesian", [True, False])
def test_complex_interpolation_and_fill(cartesian):
    # Constant phase makes either representation an exact affine signal.
    operation = dlu.Interp(
        np.array([1.0, 2.0, 3.0]) * (1 + 1j),
        [0.0, 1.0, 2.0],
        ndim=1,
        method="linear",
        complex=cartesian,
        fill=2 + 3j,
    )
    expected = np.array([2 + 3j, 1.5 + 1.5j, 2 + 3j])
    assert np.allclose(operation.sample([-1.0, 0.5, 3.0]), expected, atol=1e-6)
