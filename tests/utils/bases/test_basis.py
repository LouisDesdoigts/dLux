"""Tests for dLux.utils.bases.basis."""

import jax.numpy as np
import jax
import equinox as eqx
import pytest
import zodiax as zdx

import dLux as dl
import dLux.utils as dlu

from tests.helpers import assert_differentiable, assert_jittable


class LinearModes(zdx.Expression):
    def evaluate(self, *, coordinates):
        return np.stack([np.ones_like(coordinates), coordinates])


@pytest.fixture
def bases():
    return [
        dl.Basis(
            np.arange(120.0).reshape(2, 3, 4, 5),
            np.linspace(-0.2, 0.2, 6).reshape(2, 3),
            axes=2,
        ),
        dl.CLIMBBasis(
            np.arange(4 * 6 * 6.0).reshape(4, 6, 6),
            np.linspace(-0.2, 0.2, 4),
            oversample=3,
        ),
        dl.FourierBasis((8, 10), (3, 4), np.linspace(-0.2, 0.2, 12).reshape(3, 4)),
    ]


@pytest.mark.parametrize("index", range(3))
def test_basis_evaluation_contract(index, bases):
    basis = bases[index]
    assert_jittable(lambda value: value.evaluate(), basis, rtol=1e-5, atol=1e-5)
    assert_differentiable(
        lambda coeffs: basis.set(coeffs=coeffs).evaluate(),
        basis.coeffs,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("index", [0, 2])
def test_basis_solution_contract(index, bases):
    basis = bases[index]
    value = basis.evaluate()

    output = assert_jittable(lambda item, data: item.solve(data), basis, value)
    assert output.shape == basis.coeffs.shape


@pytest.mark.parametrize("index", [2])
def test_implicit_basis_contract(index, bases):
    basis = bases[index]
    calculated = assert_jittable(lambda value: value.calculate_basis(), basis)

    assert calculated.shape[: len(basis.shape)] == basis.shape


def test_basis_specific_operations(bases):
    climb = bases[1]
    latent = assert_jittable(lambda value: value.expansion.resolve(), climb)
    output = climb.evaluate()

    assert latent.shape == (6, 6)
    assert np.all((output >= climb.values[0]) & (output <= climb.values[1]))

    resized = assert_jittable(lambda value: value.resize((5, 7)), bases[2])
    assert resized.evaluate().shape == (5, 7)


@pytest.mark.parametrize(
    "basis,ndim",
    [
        (dl.Basis(np.ones((2, 3, 4, 5)), np.zeros((7, 2, 3)), axes=2), 2),
        (dl.FourierBasis((5, 4), (2, 3), np.zeros((7, 2, 3))), 2),
        (dl.Polynomial(1, np.zeros((7, 2))), 1),
        (dl.Basis(dl.utils.Zernike(nolls=[1, 4]), coeffs=np.zeros((7, 2))), 1),
    ],
)
def test_coefficient_domain_count_excludes_leading_batches(basis, ndim):
    assert basis.coeffs_ndim == ndim
    assert basis.coeffs.ndim == ndim + 1


def test_omitted_explicit_coeffs_start_at_zero():
    basis = dl.Basis(np.ones((2, 3, 4)))
    assert np.array_equal(basis.coeffs, np.zeros(2))
    assert np.array_equal(basis.evaluate(), np.zeros((3, 4)))
    assert np.allclose(basis.set(coeffs=np.array([1.0, 2.0])).evaluate(), 3.0)


def test_generated_basis_has_fittable_zeros_before_sampling():
    # No coordinates are available when the parameters are constructed.
    basis = dl.Basis(LinearModes(), shape=(2,))
    assert isinstance(basis.basis, LinearModes)
    assert np.array_equal(basis.coeffs, np.zeros(2))

    coordinates = np.array([0.0, 1.0, 2.0])
    observations = np.array([2.0, 5.0, 8.0])

    def loss(model):
        prediction = model.resolve(coordinates=coordinates)
        return np.sum((prediction - observations) ** 2)

    gradient = eqx.filter_jit(eqx.filter_grad(loss))(basis)
    assert np.allclose(gradient.coeffs, np.array([-30.0, -42.0]))

    coefficients = basis.solve(observations, coordinates=coordinates)
    assert np.allclose(coefficients, np.array([2.0, 3.0]), atol=1e-5)
    fitted = basis.set(coeffs=coefficients)
    assert np.allclose(fitted.resolve(coordinates=coordinates), observations)


def test_basis_solve_populates_linked_mode_definitions():
    owner = zdx.Linked(np.eye(2), key="modes")
    modes = zdx.Map(x=owner, s=owner.defer())
    basis = dl.Basis(modes, coeffs=np.asarray([1.0, 2.0]))
    samples = basis.resolve()

    coefficients = assert_jittable(lambda model: model.solve(samples), basis)
    assert np.allclose(coefficients, np.asarray([1.0, 2.0]), atol=1e-6)

    # Both uses of the shared matrix contribute to the scale-squared modes.
    def solve(scale):
        model = basis.set("basis.x.value", scale * np.eye(2))
        return model.solve(samples).sum()

    scale = 2.0
    derivative = jax.jit(jax.grad(solve))(scale)
    assert np.allclose(derivative, -2 * samples.sum() / scale**3, atol=1e-6)


def test_generated_basis_needs_shape_only_when_coefficients_are_omitted():
    with pytest.raises(ValueError, match="Provide coeffs or shape"):
        dl.Basis(LinearModes())

    basis = dl.Basis(LinearModes(), coeffs=np.array([2.0, 3.0]))
    assert np.allclose(
        basis.resolve(coordinates=np.array([0.0, 1.0])), np.array([2.0, 5.0])
    )


def test_generated_basis_shape_can_be_supplied_inside_jit():
    @eqx.filter_jit
    def make(vectors):
        return dl.Basis(zdx.Map(x=vectors), shape=(2, 3), axes=2)

    basis = make(np.ones((2, 3, 4)))
    assert np.array_equal(basis.coeffs, np.zeros((2, 3)))
    assert np.array_equal(basis.resolve(), np.zeros(4))


def test_vectorised_basis_coeffs():
    shared_basis = np.arange(12.0).reshape(1, 3, 4)
    shared = dl.Basis(shared_basis, [[1.0], [2.0], [3.0]], axes=1)
    local_basis = np.arange(48.0).reshape(2, 2, 3, 4)
    local_coeffs = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    local = dl.Basis(local_basis, local_coeffs, axes=1)

    shared_output = assert_jittable(lambda value: value.evaluate(), shared)
    local_output = assert_jittable(lambda value: value.evaluate(), local)

    assert shared_output.shape == (3, 3, 4)
    assert np.allclose(shared_output, shared.coeffs[:, 0, None, None] * shared_basis[0])
    assert local_output.shape == (2, 2, 3, 4)
    assert np.allclose(local_output[0], local_basis[0])
    assert np.allclose(local_output[1], local_basis[1])


@pytest.mark.parametrize("method", ["scan", "scatter"])
def test_pasted_basis_contract(method):
    spec = dl.PasteSpec(
        n=(6, 4),
        stamp_shape=(2, 2),
        starts=[[0, 0], [4, 2]],
        offsets=np.zeros((2, 2)),
        d=np.ones(2),
    )
    modes = np.asarray(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ]
    )
    basis = np.broadcast_to(modes, (2, 2, 2, 2))
    coeffs = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    pasted = dl.PastedBasis(basis, spec, coeffs, method)

    output = assert_jittable(lambda value: value.evaluate(), pasted)
    recovered = assert_jittable(lambda value: pasted.solve(value), output)
    assert_differentiable(lambda values: pasted.set(coeffs=values).evaluate(), coeffs)

    assert output.shape == (4, 6)
    assert np.allclose(recovered, coeffs)


@pytest.mark.parametrize("method", ["scan", "scatter"])
def test_pasted_basis_preserves_leading_batches_and_intrinsic_coefficient_axes(method):
    spec = dl.PasteSpec(
        n=(6, 2),
        stamp_shape=(3, 2),
        starts=[[0, 0], [3, 0]],
        offsets=np.zeros((2, 2)),
        d=np.ones(2),
    )
    # Each coefficient controls one pixel; the second stamp doubles its values.
    modes = np.eye(6).reshape(2, 3, 2, 3)
    modes = np.stack((modes, 2 * modes))
    coeffs = np.arange(72.0).reshape(2, 3, 2, 2, 3) / 10
    pasted = dl.PastedBasis(modes, spec, coeffs, method)
    expected = np.concatenate((coeffs[..., 0, :, :], 2 * coeffs[..., 1, :, :]), -1)

    output = assert_jittable(lambda model: model.resolve(), pasted)
    assert output.shape == (2, 3, 2, 6)
    assert np.allclose(output, expected, atol=1e-6)

    recovered = assert_jittable(lambda values: pasted.solve(values), output)
    assert recovered.shape == coeffs.shape
    assert np.allclose(recovered, coeffs, atol=1e-5)
    assert np.allclose(pasted.set(coeffs=recovered).resolve(), output, atol=1e-5)

    gradient = jax.jit(
        jax.grad(lambda values: pasted.set(coeffs=values).resolve().sum())
    )(coeffs)
    expected_gradient = np.broadcast_to(
        np.asarray([1.0, 2.0])[:, None, None], coeffs.shape
    )
    assert np.allclose(gradient, expected_gradient, atol=1e-6)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: dl.PastedBasis(np.ones((2, 1, 2, 2)), "invalid"),
        lambda: dl.PastedBasis(
            np.ones((2, 1, 2, 2)),
            dl.PasteSpec((4, 4), (2, 2), [[0, 0]], [[0.0, 0.0]], [1.0, 1.0]),
        ),
    ],
)
def test_pasted_basis_validation(constructor):
    with pytest.raises((TypeError, ValueError)):
        constructor()


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: dl.Basis(np.ones((2, 3, 4)), axes=0),
        lambda: dl.Basis(np.ones((2, 3, 4)), np.ones((2, 3)), axes=1),
        lambda: dl.Basis(np.ones((2, 3, 4)), shape=(3,)),
        lambda: dl.Basis(LinearModes(), shape=(2, 3), axes=1),
        lambda: dl.Basis(LinearModes(), shape=(0,)),
        lambda: dl.CLIMBBasis(np.ones((4, 6, 6)), np.ones(4), values=(0.0, 0.5, 1.0)),
        lambda: dl.CLIMBBasis(np.ones((4, 5, 5)), np.ones(4), oversample=3),
        lambda: dl.FourierBasis(8, 3, np.ones((2, 3))),
    ],
)
def test_validation(constructor):
    with pytest.raises((TypeError, ValueError)):
        constructor()


def test_basis_rank_deficient_solve_returns_minimum_norm_representative():
    basis = dl.Basis(np.asarray([[1.0, 0.0], [1.0, 0.0]]))
    value = np.asarray([4.0, 3.0])
    coeffs = assert_jittable(lambda model: model.solve(value), basis)
    assert np.allclose(coeffs, np.asarray([2.0, 2.0]), atol=1e-6)
    assert np.allclose(
        basis.set(coeffs=coeffs).evaluate(), np.asarray([4.0, 0.0]), atol=1e-6
    )


def test_fourier_rectangular_image_axes_and_compact_evaluation():
    coeffs = np.arange(6.0).reshape(2, 3)
    fourier = dl.FourierBasis((7, 5), (2, 3), coeffs)
    explicit = dl.Basis(fourier.calculate_basis(), coeffs, axes=2)
    assert fourier.resolve().shape == (7, 5)
    assert np.allclose(fourier.resolve(), explicit.resolve(), atol=1e-5)

    # Released evaluation contracts coefficients directly between the two kernels.
    kernels = dlu.fourier_kernels((2, 3), (7, 5))
    expected = kernels[0] @ coeffs @ kernels[1].T
    assert np.allclose(fourier.resolve(), expected, atol=1e-5)
    recovered = assert_jittable(lambda values: fourier.solve(values), expected)
    assert np.allclose(recovered, coeffs, atol=1e-5)


@pytest.mark.parametrize("unit", [None, "nm"])
def test_fourier_resize_preserves_output_unit(unit):
    coeffs = np.arange(6.0).reshape(2, 3)
    fourier = dl.FourierBasis((7, 5), (2, 3), coeffs, unit=unit)
    assert np.allclose(
        fourier.resize((7, 5)).resolve() / dlu.unit_factor(fourier.unit),
        fourier.resolve() / dlu.unit_factor(fourier.unit),
        atol=1e-6,
    )

    resized = assert_jittable(lambda model: model.resize((8, 6)), fourier)
    expected = dl.FourierBasis((8, 6), (2, 3), coeffs, unit=unit)
    assert resized.unit == unit
    assert np.allclose(
        resized.resolve() / dlu.unit_factor(resized.unit),
        expected.resolve() / dlu.unit_factor(expected.unit),
        atol=1e-6,
    )


def test_fourier_resize_preserves_independent_fitted_kernel_scales():
    fourier = dl.FourierBasis(4, 1, coeffs=np.ones((1, 1)))

    def resize(scales):
        kernels = tuple(
            scale * kernel for scale, kernel in zip(scales, fourier.kernels)
        )
        return fourier.set(kernels=kernels).resize((6, 5)).resolve()

    scales = np.asarray([2.0, 3.0])
    output = assert_jittable(resize, scales)
    assert output.shape == (6, 5)
    assert np.allclose(output, 6.0, atol=1e-6)

    gradient = jax.jit(jax.grad(lambda values: resize(values).sum()))(scales)
    assert np.allclose(gradient, 30 * np.asarray([3.0, 2.0]), atol=1e-6)


def test_basis_coefficients_batch_like_independent_models():
    vectors = np.arange(24.0).reshape(2, 3, 4)
    coeffs = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    basis = dl.Basis(vectors, coeffs, axes=1)
    expected = jax.vmap(lambda values: dl.Basis(vectors, values).resolve())(coeffs)
    assert np.allclose(basis.resolve(), expected)


@pytest.mark.parametrize(
    "unit,factor", [(None, 1.0), ("nm", 1e-9), ("arcsec", np.pi / 648000)]
)
@pytest.mark.parametrize("family", ["explicit", "fourier", "polynomial"])
def test_basis_output_units_evaluation_solve_and_gradients(family, unit, factor):
    if family == "explicit":
        basis = dl.Basis(np.array([[1.0, 0.0], [0.0, 2.0]]), unit=unit)
        coeffs = np.array([2.0, 3.0])
        expected = np.array([2.0, 6.0])
        expected_gradient = np.array([1.0, 2.0])
    elif family == "fourier":
        basis = dl.FourierBasis((3, 2), (1, 1), unit=unit)
        coeffs = np.array([[2.0]])
        expected = np.full((3, 2), 2.0)
        expected_gradient = np.array([[6.0]])
    else:
        basis = dl.Polynomial(1, grid=dl.Axis(n=3, d=1.0), unit=unit)
        coeffs = np.array([2.0, 3.0])
        expected = np.array([-1.0, 2.0, 5.0])
        expected_gradient = np.array([3.0, 0.0])

    # Unit conversion does not change the stored zero initialization or modes.
    assert np.array_equal(basis.coeffs, np.zeros_like(coeffs))
    assert np.array_equal(basis.resolve(), np.zeros_like(expected))
    model = basis.set(coeffs=coeffs)
    output = assert_jittable(lambda item: item.resolve(), model, atol=0)
    assert np.allclose(output / factor, expected, atol=1e-6)

    recovered = assert_jittable(lambda values: model.solve(values), output)
    assert np.allclose(recovered, coeffs, atol=1e-5)
    derivative = jax.jit(jax.jacrev(model.solve))(output)
    unitless_derivative = jax.jacrev(model.set(unit=None).solve)(expected)
    assert np.allclose(derivative * factor, unitless_derivative, atol=1e-6)

    gradient = jax.jit(
        jax.grad(lambda values: model.set(coeffs=values).resolve().sum())
    )(coeffs)
    assert np.allclose(gradient / factor, expected_gradient, atol=1e-6)

    if family != "explicit":
        modes = model.calculate_basis()
        assert np.array_equal(modes, model.set(unit=None).calculate_basis())
        expanded = dlu.eval_basis(modes, coeffs, axes=model.coeffs_ndim)
        assert np.allclose(output / factor, expanded, atol=1e-6)


@pytest.mark.parametrize("method", ["scan", "scatter"])
def test_pasted_basis_output_unit_is_applied_once(method):
    spec = dl.PasteSpec(
        n=(4, 2),
        stamp_shape=(2, 2),
        starts=[[0, 0], [2, 0]],
        offsets=np.zeros((2, 2)),
        d=np.ones(2),
    )
    modes = np.ones((2, 1, 2, 2))
    model = dl.PastedBasis(modes, spec, unit="nm", method=method)
    assert np.array_equal(model.resolve(), np.zeros((2, 4)))
    coeffs = np.array([[[2.0], [3.0]], [[4.0], [5.0]]])
    model = model.set(coeffs=coeffs)
    expected = np.repeat(coeffs[..., 0], 2, axis=-1)[:, None, :]
    output = assert_jittable(lambda item: item.resolve(), model, atol=0)
    assert np.allclose(output / 1e-9, expected, atol=1e-6)
    recovered = assert_jittable(lambda values: model.solve(values), output)
    assert np.allclose(recovered, coeffs, atol=1e-5)
    gradient = jax.jit(
        jax.grad(lambda values: model.set(coeffs=values).resolve().sum())
    )(coeffs)
    assert np.allclose(gradient / 1e-9, 4.0, atol=1e-6)


@pytest.mark.parametrize("unit", [2.0, zdx.Unit(x=10, unit="nm")])
@pytest.mark.parametrize("family", ["explicit", "fourier", "polynomial", "pasted"])
def test_basis_unit_requires_a_name(family, unit):
    with pytest.raises(TypeError, match="unit must be a string or None"):
        if family == "explicit":
            dl.Basis(np.eye(2), unit=unit)
        elif family == "fourier":
            dl.FourierBasis(3, 1, unit=unit)
        elif family == "polynomial":
            dl.Polynomial(1, unit=unit)
        else:
            spec = dl.PasteSpec((2, 2), (2, 2), [[0, 0]], [[0.0, 0.0]], [1.0, 1.0])
            dl.PastedBasis(np.ones((1, 1, 2, 2)), spec, unit=unit)


def test_basis_units_are_independent_of_zodiax_global_settings():
    previous = zdx.get_units()
    try:
        zdx.set_units({"cartesian": "km", "angular": "deg"})
        basis = dl.Basis(np.eye(2), [2.0, 3.0], unit="nm")
        assert np.allclose(basis.resolve() / 1e-9, np.array([2.0, 3.0]), atol=1e-6)
    finally:
        zdx.set_units(previous)


@pytest.mark.parametrize("family", ["explicit", "fourier", "polynomial", "pasted"])
def test_basis_waits_for_coefficient_context(family):
    coefficients = zdx.StateRef("coeffs")
    if family == "explicit":
        # Independent ready modes still prepare while the coefficients wait.
        model = dl.Basis(zdx.Map(x=np.eye(2), s=2), coefficients, axes=1, unit="nm")
        values = np.array([2.0, 3.0])
        expected = np.array([4.0, 6.0])
    elif family == "fourier":
        model = dl.FourierBasis((3, 2), 1, coefficients, unit="nm")
        values = np.array([[2.0]])
        expected = np.full((3, 2), 2.0)
    elif family == "polynomial":
        model = dl.Polynomial(1, coefficients, grid=dl.Axis(n=3, d=1), unit="nm")
        values = np.array([2.0, 3.0])
        expected = np.array([-1.0, 2.0, 5.0])
    else:
        spec = dl.PasteSpec((2, 2), (2, 2), [[0, 0]], [[0.0, 0.0]], [1.0, 1.0])
        model = dl.PastedBasis(np.ones((1, 1, 2, 2)), spec, coefficients, unit="nm")
        values = np.array([[2.0]])
        expected = np.full((2, 2), 2.0)

    partial = assert_jittable(lambda item: item.resolve(), model)
    assert isinstance(partial, type(model))
    assert isinstance(partial.coeffs, zdx.StateRef)
    if family == "explicit":
        assert np.array_equal(partial.basis, 2 * np.eye(2))

    output = assert_jittable(
        lambda state: partial.resolve(state=state), zdx.State(coeffs=values), atol=0
    )
    assert np.allclose(output / 1e-9, expected, atol=1e-6)


def test_basis_waits_for_modes_and_solve_requires_them():
    basis = dl.Basis(zdx.StateRef("modes"), coeffs=np.array([2.0, 3.0]), unit="nm")
    partial = assert_jittable(lambda item: item.resolve(), basis)
    assert isinstance(partial.basis, zdx.StateRef)
    samples = np.array([2.0, 3.0]) * 1e-9
    with pytest.raises(zdx.Unresolved, match="Basis modes and samples"):
        partial.solve(samples)

    state = zdx.State(modes=np.eye(2))
    coefficients = assert_jittable(
        lambda values: partial.solve(values, state=state), samples
    )
    assert np.allclose(coefficients, np.array([2.0, 3.0]), atol=1e-6)


def test_basis_infers_all_coefficient_axes_and_recovers_coefficients():
    modes = np.eye(6).reshape(2, 3, 6)
    coeffs = np.arange(6.0).reshape(2, 3)
    model = dl.Basis(modes, coeffs)
    assert model.axes == model.coeffs_ndim == 2
    output = assert_jittable(lambda value: value.resolve(), model)
    assert np.array_equal(output, coeffs.reshape(-1))
    assert np.allclose(model.solve(output), coeffs)
    gradient = jax.jit(jax.grad(lambda c: dl.Basis(modes, c).resolve().sum()))(coeffs)
    assert np.array_equal(gradient, np.ones_like(coeffs))


def test_generated_basis_infers_zero_coefficient_shape_without_evaluation():
    model = dl.Basis(zdx.StateRef("modes"), shape=(2, 3))
    assert model.axes == 2
    assert np.array_equal(model.coeffs, np.zeros((2, 3)))
    assert isinstance(model.resolve(), dl.Basis)
    state = zdx.State(modes=np.eye(6).reshape(2, 3, 6))
    assert np.array_equal(model.resolve(state=state), np.zeros(6))


def test_expression_coefficients_require_explicit_topology():
    coefficients = zdx.StateRef("coeffs")
    with pytest.raises(ValueError, match="axes or shape"):
        dl.Basis(np.eye(6).reshape(2, 3, 6), coefficients)
    model = dl.Basis(np.eye(6).reshape(2, 3, 6), coefficients, shape=(2, 3))
    assert model.axes == 2
    state = zdx.State(coeffs=np.arange(6.0).reshape(2, 3))
    assert np.array_equal(model.resolve(state=state), np.arange(6.0))


@pytest.mark.parametrize(
    "operation,error",
    [
        (lambda: dlu.eval_basis(np.ones((2, 3)), [1.0, 2.0], axes=0), ValueError),
        (lambda: dlu.solve_basis(np.ones(3), np.ones((2, 3)), axes=0), ValueError),
        (lambda: dl.Basis(np.ones((2, 3)), axes=True), TypeError),
        (lambda: dl.Basis(np.ones((0, 3))), ValueError),
    ],
)
def test_basis_definition_validation_tails(operation, error):
    """Coverage tails: invalid contraction dimensions and empty mode axes."""
    with pytest.raises(error):
        operation()


@pytest.mark.parametrize("method", ["scan", "scatter"])
def test_overlapping_pasted_basis_adds_values_and_solves_local_projections(method):
    spec = dlu.PasteSpec((3, 2), (2, 2), [[0, 0], [1, 0]], np.zeros((2, 2)), [1, 1])
    model = dlu.PastedBasis(np.ones((2, 1, 2, 2)), spec, [[3.0], [5.0]], method)
    output = assert_jittable(lambda value: value.resolve(), model)
    assert np.allclose(output, np.array([[3.0, 8.0, 5.0], [3.0, 8.0, 5.0]]))
    # Extraction sees the summed overlap, so solving is a local projection.
    assert np.allclose(model.solve(output), np.array([[5.5], [6.5]]))


@pytest.mark.parametrize(
    "basis,method,message",
    [
        (np.ones((1, 2, 2)), "scan", "basis must have shape"),
        (np.ones((1, 1, 3, 3)), "scan", "spatial shape"),
        (np.ones((1, 1, 2, 2)), "invalid", "method must be"),
    ],
)
def test_pasted_basis_constructor_validation_tails(basis, method, message):
    """Coverage tails: intrinsic coefficient axes, stamp sampling and method."""
    spec = dlu.PasteSpec((2, 2), (2, 2), [[0, 0]], [[0.0, 0.0]], [1.0, 1.0])
    with pytest.raises(ValueError, match=message):
        dlu.PastedBasis(basis, spec, method=method)


def test_pasted_basis_evaluation_validation_tails():
    """Coverage tails: explicit evaluation reports missing or malformed coefficients."""
    spec = dlu.PasteSpec((2, 2), (2, 2), [[0, 0]], [[0.0, 0.0]], [1.0, 1.0])
    model = dlu.PastedBasis(np.ones((1, 1, 2, 2)), spec, zdx.StateRef("coeffs"))
    with pytest.raises(zdx.Unresolved, match="coefficients"):
        model.evaluate()
    with pytest.raises(ValueError, match="coeffs must end"):
        model.set(coeffs=np.zeros((2, 2))).evaluate()
