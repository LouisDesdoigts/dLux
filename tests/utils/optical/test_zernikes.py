"""Zernike generators and Basis expansions and their numerical contract."""

import jax
import jax.numpy as np
import pytest
import zodiax as zdx
from jax import Array

import dLux.utils as dlu

from tests.helpers import assert_jittable


def test_negative_azimuthal_coma_has_finite_radial_polynomial():
    coordinates = dlu.Grid((7, 7), 0.2).coordinates
    x, y = coordinates
    rho2 = x * x + y * y
    expected = np.sqrt(8.0) * (3 * rho2 - 2) * y
    expected = np.where(rho2 <= 1, expected, 0)
    mode = dlu.Basis(dlu.Zernike(7), coeffs=[1.0])
    result = assert_jittable(lambda model: model.resolve(coordinates=coordinates), mode)
    assert np.allclose(result, expected, atol=5e-6)


def test_missing_coefficient_context_defers_only_the_expansion():
    grid = dlu.Grid(4, d=0.1)
    dynamic = dlu.Basis(
        dlu.Zernike([1, 2], grid=grid), coeffs=zdx.StateRef("coeffs"), unit="nm"
    )
    pending = dynamic.resolve()
    assert isinstance(pending, dlu.Basis)
    assert pending.basis.shape == (2,) + grid.shape

    state = zdx.State(coeffs=[2.0, 3.0])
    result = assert_jittable(lambda value: pending.resolve(state=value), state)
    reference = dlu.Basis(dlu.Zernike([1, 2], grid=grid), coeffs=[2.0, 3.0], unit="nm")
    assert np.allclose(result / 1e-9, reference.resolve() / 1e-9, atol=2e-6)


@pytest.mark.parametrize("constructor", [dlu.Basis])
def test_units_do_not_accept_a_quantity_with_its_own_amplitude(constructor):
    with pytest.raises(TypeError, match="unit must be a string or None"):
        constructor(dlu.Zernike([1]), unit=zdx.Unit(x=10, unit="nm"))


class ContextCoefficients(zdx.Expression):
    scale: Array | zdx.Expression

    def __init__(self, scale):
        self.scale = zdx.as_array(scale)

    def evaluate(self, *, coordinates, diameter, n_sides, **context):
        self = self.resolve_children(
            coordinates=coordinates, diameter=diameter, n_sides=n_sides
        )
        position = coordinates.mean((-2, -1))
        return self.scale * (position + np.array([diameter, n_sides]))


def test_coefficients_receive_parent_context_independently_of_generator_transform():
    grid = dlu.Grid(4, d=0.1, c=(0.2, 0.1))
    transform = dlu.Affine(translation=(0.1, 0.05))
    modes = dlu.Zernike([1, 2], diameter=1.0, n_sides=6, transform=transform)
    model = dlu.Basis(modes, coeffs=ContextCoefficients(2.0), unit="nm")
    context = dict(coordinates=grid.coordinates, diameter=1.0, n_sides=6)

    # Coefficients see the supplied frame; only the generator transforms its grid.
    coefficients = 2.0 * (grid.coordinates.mean((-2, -1)) + np.array([1.0, 6.0]))
    sampled = modes.resolve(**context)
    expected = np.einsum("k,kij->ij", coefficients, sampled)
    result = assert_jittable(lambda value: value.resolve(**context), model)
    assert np.allclose(result / 1e-9, expected, atol=2e-5)
    assert isinstance(model.coeffs, ContextCoefficients)


def test_noll_indices_keep_signed_mode_order():
    expected = [
        (0, 0),
        (1, 1),
        (1, -1),
        (2, 0),
        (2, -2),
        (2, 2),
        (3, -1),
        (3, 1),
        (3, -3),
        (3, 3),
    ]
    assert [dlu.Zernike.noll_indices(j) for j in range(1, 11)] == expected


def test_supported_normalised_basis_matches_dynamic_with_units():
    import dLux as dl

    grid = dlu.Grid((8, 6), d=0.2)
    x, y = grid.coordinates
    support = ((x * x + y * y < 0.7**2) & (x > -0.5)).astype(float)
    definition = dl.ZernikeDef(
        [1, 2, 4], diameter=1.6, support=support, norm="rms", unit="nm"
    )
    fixed = definition(grid, coeffs=[2.0, -0.3, 0.7])
    dynamic = definition(grid, coeffs=[2.0, -0.3, 0.7], dynamic=True)
    modes = dynamic.basis.evaluate()
    rms = np.sqrt(np.sum(modes**2 * support, axis=(-2, -1)) / support.sum())
    assert np.allclose(rms, 1.0, atol=2e-6)
    assert np.all(modes[:, support == 0] == 0)
    expected = np.tensordot(fixed.coeffs, modes, axes=1) * 1e-9
    actual = assert_jittable(lambda model: model.resolve(), dynamic)
    assert np.allclose(actual / 1e-9, expected / 1e-9, atol=2e-6)
    assert np.allclose(fixed.resolve() / 1e-9, expected / 1e-9, atol=2e-6)
    assert np.allclose(dynamic.solve(actual), fixed.coeffs, atol=2e-5)


@pytest.mark.parametrize(
    "selection, expected",
    [
        ({"order": 2}, [1, 2, 3, 4, 5, 6]),
        ({"orders": 2}, [4, 5, 6]),
        ({"orders": [2, 0]}, [4, 5, 6, 1]),
        ({"nolls": np.array([3, 1, 4])}, [3, 1, 4]),
    ],
)
def test_mode_selection_allocates_editable_zero_coefficients(selection, expected):
    model = dlu.Basis(dlu.Zernike(**selection))
    assert np.array_equal(model.basis.nolls, np.array(expected))
    assert np.array_equal(model.coeffs, np.zeros(len(expected)))
    assert isinstance(model.resolve(), dlu.Basis)

    grid = dlu.Grid(4, d=0.2)
    assert np.array_equal(model.resolve(grid=grid), np.zeros(grid.shape))
    updated = model.set(coeffs=np.ones(len(expected)))
    assert np.any(updated.resolve(grid=grid) != 0)
    assert np.array_equal(model.coeffs, np.zeros(len(expected)))


def test_noll_names_orders_and_radial_factors():
    assert dlu.Zernike.name(1) == "Piston"
    assert dlu.Zernike.name(4) == "Defocus"
    assert dlu.Zernike.name(37) == "Zernike 37"
    assert dlu.Zernike.radial_orders_to_indices([0, 2]) == [1, 4, 5, 6]
    c, k = dlu.Zernike.factors(11)
    assert np.allclose(c, np.array([6, -6, 1]), atol=1e-5)
    assert np.array_equal(k, np.array([0, 1, 2]))

    radius = np.array([[0.0, 0.2, 0.5, 1.0]])
    radial = assert_jittable(dlu.Zernike.radial, radius, 4, c, k)
    assert np.allclose(radial, 6 * radius**4 - 6 * radius**2 + 1, atol=2e-6)
    angle = np.array([[0.0, np.pi / 4, np.pi / 2]])
    assert np.allclose(
        dlu.Zernike.azimuthal(angle, 2, -2), np.sqrt(6) * np.sin(2 * angle)
    )
    assert np.allclose(
        dlu.Zernike.azimuthal(angle, 2, 2), np.sqrt(6) * np.cos(2 * angle)
    )


@pytest.mark.parametrize(
    "indices, orders, family",
    [
        ((2, 3), (1, 1), "Tilt"),
        ((5, 6), (2, 2), "Astig"),
        ((7, 8), (3, 1), "Coma"),
        ((9, 10), (3, 3), "Trefoil"),
        ((12, 13), (4, 2), "2nd Astig"),
        ((14, 15), (4, 4), "Quadrafoil"),
        ((16, 17), (5, 1), "2nd Coma"),
        ((18, 19), (5, 3), "2nd Trefoil"),
        ((20, 21), (5, 5), "Pentafoil"),
        ((23, 24), (6, 2), "3rd Astig"),
        ((25, 26), (6, 4), "2nd Quadrafoil"),
        ((27, 28), (6, 6), "Hexafoil"),
        ((29, 30), (7, 1), "3rd Coma"),
        ((31, 32), (7, 3), "3rd Trefoil"),
        ((33, 34), (7, 5), "2nd Pentafoil"),
        ((35, 36), (7, 7), "Heptafoil"),
    ],
)
def test_zernike_names_match_noll_families_and_angular_convention(
    indices, orders, family
):
    """Names follow the existing Noll polynomials; renaming must not reorder them."""
    for j in indices:
        n, m = dlu.Zernike.noll_indices(j)
        assert (n, abs(m)) == orders
        direction = "X" if m > 0 else "Y"
        assert dlu.Zernike.name(j) == f"{family} {direction}"


@pytest.mark.parametrize("method", ["padded", "mapped"])
def test_low_order_modes_match_cartesian_polynomials_in_requested_order(method):
    coordinates = dlu.Grid((7, 5), d=0.4).coordinates
    x, y = coordinates
    r2 = x * x + y * y
    # Explicit Cartesian polynomials give a reference independent of radial grouping.
    expected = np.stack(
        (
            np.sqrt(5) * (6 * r2**2 - 6 * r2 + 1),
            2 * y,
            np.ones_like(x),
            2 * np.sqrt(6) * x * y,
            np.sqrt(3) * (2 * r2 - 1),
            np.sqrt(8) * (3 * r2 - 2) * y,
            2 * y,
        )
    )
    expected = np.where(r2 <= 1, expected, 0)
    model = dlu.Basis(dlu.Zernike([11, 3, 1, 5, 4, 7, 3], method=method))
    result = assert_jittable(
        lambda value: value.basis.evaluate(coordinates=coordinates), model
    )
    # float32 factorial uses gamma evaluation: the spherical mode differs by
    # 1.1e-5 here (its -6 radial factor differs by 5.2e-6). Keep this explicit.
    assert np.allclose(result, expected, rtol=1e-6, atol=2e-5)


@pytest.mark.parametrize("method", ["padded", "mapped"])
def test_square_polygon_piston_matches_its_radial_boundary(method):
    coordinates = dlu.Grid((6, 4), d=0.35).coordinates
    x, y = coordinates
    radius = np.hypot(x, y)
    # Four-sided modes map the unit circle onto |x| + |y| <= 1.
    boundary_scale = radius / (np.abs(x) + np.abs(y))
    expected = np.where(np.abs(x) + np.abs(y) <= 1, 1 / boundary_scale, 0)
    model = dlu.Basis(dlu.Zernike(1, n_sides=4, method=method), coeffs=[1])
    result = assert_jittable(
        lambda value: value.resolve(coordinates=coordinates), model
    )
    assert np.allclose(result, expected, atol=2e-6)


def test_diameter_axes_and_oversize_control_support():
    grid = dlu.Grid((6, 4), d=0.3)
    x, y = grid.coordinates
    model = dlu.Basis(
        dlu.Zernike(1, diameter=np.array([1.0, 2.0]), grid=grid), coeffs=[1]
    )
    expected = ((x / 0.5) ** 2 + y**2 <= 1).astype(float)
    result = model.resolve()
    assert np.array_equal(result != 0, expected.astype(bool))
    # The float32 factorial ratio gives piston amplitude 0.99999905, not exact 1.
    assert np.allclose(result, expected, rtol=0, atol=2e-6)

    enlarged = model.set("basis.oversize", np.array(0.5))
    reference = model.set("basis.diameter", np.array([1.5, 3.0]))
    assert np.allclose(enlarged.resolve(), reference.resolve())
    assert enlarged.resolve().sum() > model.resolve().sum()


def test_coefficient_batches_and_solve_preserve_units():
    coefficients = np.array([[1.2, -0.3, 0.7], [-0.4, 0.5, 0.2]])
    grid = dlu.Grid((8, 6), d=0.15)
    model = dlu.Basis(dlu.Zernike([1, 2, 4], grid=grid), coeffs=coefficients, unit="nm")
    modes = model.basis.evaluate()
    output = assert_jittable(lambda value: value.resolve(), model)
    expected = np.stack(
        [np.sum(row[:, None, None] * modes, axis=0) for row in coefficients]
    )
    assert output.shape == (2,) + grid.shape
    assert np.allclose(output / 1e-9, expected, atol=3e-6)
    recovered = assert_jittable(lambda value: value.solve(output), model)
    assert np.allclose(recovered, coefficients, atol=2e-5)


def test_stored_apertures_and_population_batches_do_not_pair():
    grid = dlu.Grid((8, 4), d=0.2)
    centers = np.array([[-0.4, 0.0], [0.4, 0.0]])
    coefficients = np.array([[[1.0], [2.0]], [[3.0], [-1.0]]])
    x, y = grid.coordinates
    masks = np.stack([(x - cx) ** 2 + (y - cy) ** 2 <= 0.3**2 for cx, cy in centers])
    # Both population and aperture dimensions have size two; only apertures sum.
    model = dlu.Basis(
        dlu.Zernike(1, centers=centers, diameter=0.6, grid=grid), coeffs=coefficients
    )
    assert model.coeffs_ndim == 2
    assert model.basis.coeffs_shape == (2, 1)
    modes = model.basis.evaluate()
    assert np.allclose(modes[:, 0], masks)
    result = assert_jittable(lambda value: value.resolve(), model)
    expected = np.stack(
        [np.sum(row[:, :, None] * masks, axis=0) for row in coefficients]
    )
    assert result.shape == (2,) + grid.shape
    assert np.allclose(result, expected)
    assert np.allclose(model.solve(result), coefficients, atol=2e-6)

    support = np.stack([x < 0, x > 0])
    assert np.allclose(model.basis.evaluate(support=support)[:, 0], masks * support)


def test_coefficient_and_geometry_gradients_away_from_polar_origin():
    grid = dlu.Grid((4, 2), d=0.1, c=(0.2, 0.15))
    x, y = grid.coordinates
    model = dlu.Basis(dlu.Zernike(4, diameter=2.0, grid=grid), coeffs=[1.0])
    loss = lambda diameter: model.set("basis.diameter", diameter).resolve().sum()
    derivative = assert_jittable(jax.grad(loss), np.array(2.0))
    expected = (-16 * np.sqrt(3) * (x * x + y * y) / 2.0**3).sum()
    assert np.allclose(derivative, expected, atol=2e-6)

    coefficient_loss = (
        lambda coefficients: model.set(coeffs=coefficients).resolve().sum()
    )
    derivative = jax.grad(coefficient_loss)(model.coeffs)
    assert np.allclose(derivative, model.basis.evaluate().sum((-2, -1)), atol=2e-6)


def test_expression_geometry_resolves_at_evaluation_and_can_be_updated():
    model = dlu.Basis(dlu.Zernike(4, diameter=zdx.StateRef("diameter")), coeffs=[1])
    grid = dlu.Grid(4, d=0.1)
    assert isinstance(model.resolve(grid=grid), dlu.Basis)
    state = zdx.State(diameter=1.0)
    actual = assert_jittable(lambda value: model.resolve(grid=grid, state=value), state)
    expected = dlu.Basis(dlu.Zernike(4, diameter=1.0, grid=grid), coeffs=[1]).resolve()
    assert np.allclose(actual, expected)
    assert isinstance(model.basis.diameter, zdx.StateRef)


@pytest.mark.parametrize(
    "options, message",
    [
        ({}, "exactly one"),
        ({"nolls": [1], "order": 2}, "exactly one"),
        ({"nolls": np.array([[1]])}, "one-dimensional"),
        ({"orders": np.array([[1]])}, "one-dimensional"),
        ({"nolls": []}, "At least one"),
        ({"nolls": [0]}, "greater than zero"),
        ({"orders": [-1]}, "non-negative"),
        ({"nolls": [1], "method": "unknown"}, "method"),
        ({"nolls": [1], "diameter": 0}, "diameter"),
        ({"nolls": [1], "n_sides": 2}, "n_sides"),
        ({"nolls": [1], "norm": "unknown"}, "norm"),
        ({"nolls": [1], "centers": [0, 0]}, "centers"),
    ],
)
def test_zernike_constructor_validation_tails(options, message):
    """Coverage tails: reject unsupported topology and malformed stored inputs."""
    with pytest.raises(ValueError, match=message):
        dlu.Basis(dlu.Zernike(**options))


@pytest.mark.parametrize("function", [dlu.Zernike.noll_indices, dlu.Zernike.factors])
def test_noll_index_validation_tails(function):
    """Coverage tails: public metadata helpers reject non-positive Noll indices."""
    with pytest.raises(ValueError, match="greater than"):
        function(0)


def test_runtime_geometry_validation_tails():
    """Coverage tails: runtime overrides cannot change coefficient topology."""
    grid = dlu.Grid(4, d=0.1)
    model = dlu.Basis(dlu.Zernike(1, grid=grid))
    with pytest.raises(ValueError, match="n_sides"):
        model.basis.evaluate(n_sides=2)
    with pytest.raises(ValueError, match="coefficient shape"):
        model.basis.evaluate(centers=np.array([[0.0, 0.0], [0.1, 0.1]]))


@pytest.mark.parametrize(
    "j, expected",
    [(1, [1, 0, 0]), (2, [0, 1, 0]), (4, [-1, 0, 4]), (11, [1, 0, -12])],
)
def test_radial_value_gradient_and_hessian_at_zero(j, expected):
    """A constant radial term must not contaminate derivatives of the polynomial."""
    n, _ = dlu.Zernike.noll_indices(j)
    c, k = dlu.Zernike.factors(j)
    function = lambda rho: dlu.Zernike.radial(rho.reshape(1, 1), n, c, k).sum()

    def evaluate(rho):
        return np.stack(
            (function(rho), jax.grad(function)(rho), jax.grad(jax.grad(function))(rho))
        )

    result = assert_jittable(evaluate, np.array(0.0))
    assert np.allclose(result, np.array(expected), rtol=2e-6, atol=2e-5)


def cartesian_modes(xy):
    """Independent low-order polynomials, with no polar coordinates or mode metadata."""
    x, y = xy
    r2 = x * x + y * y
    return np.stack(
        (
            np.ones_like(x),
            2 * x,
            2 * y,
            np.sqrt(3) * (2 * r2 - 1),
            2 * np.sqrt(6) * x * y,
            np.sqrt(6) * (x * x - y * y),
            np.sqrt(8) * (3 * r2 - 2) * y,
            np.sqrt(8) * (3 * r2 - 2) * x,
            np.sqrt(8) * (3 * x * x * y - y**3),
            np.sqrt(8) * (x**3 - 3 * x * y * y),
            np.sqrt(5) * (6 * r2**2 - 6 * r2 + 1),
        )
    )


@pytest.mark.parametrize("method", ["padded", "mapped"])
@pytest.mark.parametrize("position", [(0.0, 0.0), (1e-6, -2e-6), (0.1, -0.2)])
def test_circular_modes_preserve_cartesian_derivatives_at_and_near_origin(
    method, position
):
    """Compare slopes and curvature, not just finiteness: tilt slopes must survive."""
    model = dlu.Basis(dlu.Zernike(nolls=list(range(1, 12)), method=method))
    function = lambda xy: model.basis.evaluate(coordinates=xy[:, None, None])[:, 0, 0]
    position = np.array(position)

    def evaluate(xy):
        return (
            function(xy),
            jax.jacrev(function)(xy),
            jax.jacfwd(jax.jacrev(function))(xy),
        )

    values, slopes, curvature = assert_jittable(evaluate, position, atol=3e-5)
    # The factorial-derived coefficients retain the measured float32 roundoff.
    assert np.allclose(values, cartesian_modes(position), rtol=2e-6, atol=2e-5)
    assert np.allclose(
        slopes, jax.jacrev(cartesian_modes)(position), rtol=2e-6, atol=3e-5
    )
    assert np.allclose(
        curvature, jax.hessian(cartesian_modes)(position), rtol=2e-6, atol=3e-5
    )


def test_diameter_and_translation_derivatives_on_grid_containing_origin():
    grid = dlu.Grid(3, d=0.1)
    model = dlu.Basis(dlu.Zernike(4, grid=grid), coeffs=[1.0])

    def prediction(parameters):
        diameter, center = parameters[0], parameters[1:]
        changed = model.set(
            {
                "basis.diameter": diameter,
                "basis.transform": dlu.Affine(translation=center),
            }
        )
        return changed.resolve().sum()

    def reference(parameters):
        diameter, center = parameters[0], parameters[1:]
        coordinates = grid.coordinates - center[:, None, None]
        radius_squared = np.sum(coordinates**2, axis=0)
        return (np.sqrt(3) * (8 * radius_squared / diameter**2 - 1)).sum()

    def evaluate(parameters):
        return (
            prediction(parameters),
            jax.grad(prediction)(parameters),
            jax.hessian(prediction)(parameters),
        )

    parameters = np.array([2.0, 0.0, 0.0])
    values, slopes, curvature = assert_jittable(evaluate, parameters, atol=2e-5)
    assert np.allclose(values, reference(parameters), atol=2e-5)
    assert np.allclose(slopes, jax.grad(reference)(parameters), atol=2e-5)
    assert np.allclose(curvature, jax.hessian(reference)(parameters), atol=2e-5)
