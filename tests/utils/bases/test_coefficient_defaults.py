"""Fit-ready coefficient defaults and explicitly empty expansions."""

import equinox as eqx
import jax
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux as dl
import dLux.utils as dlu


class GeneratedModes(zdx.Expression):
    """A third-party generator needs only static shape metadata and evaluate."""

    @property
    def coeffs_shape(self):
        return (2, 3)

    def evaluate(self, *, samples, **context):
        return np.eye(6).reshape(2, 3, 6) * samples


def test_generated_basis_initialises_without_evaluating_and_preserves_batches():
    model = dlu.Basis(GeneratedModes())
    assert model.axes == 2
    assert np.array_equal(model.coeffs, np.zeros((2, 3)))
    assert isinstance(model.resolve(), dlu.Basis)

    coefficients = np.arange(24.0).reshape(4, 2, 3)
    model = dlu.Basis(GeneratedModes(), coefficients)
    result = eqx.filter_jit(lambda basis: basis.resolve(samples=2.0))(model)
    assert np.array_equal(result, 2 * coefficients.reshape(4, 6))
    assert np.allclose(model.solve(result, samples=2.0), coefficients)
    gradient = jax.jit(jax.grad(lambda c: model.set(coeffs=c).resolve(samples=2).sum()))
    assert np.array_equal(gradient(coefficients), np.full(coefficients.shape, 2.0))


def test_generator_shape_contract_checks_only_intrinsic_axes():
    with pytest.raises(ValueError, match="coeffs must end"):
        dlu.Basis(GeneratedModes(), np.zeros((2, 4)))
    with pytest.raises(ValueError, match="generator's coeffs_shape"):
        dlu.Basis(GeneratedModes(), shape=(2, 4))
    with pytest.raises(ValueError, match="axes positive"):
        dlu.Basis(GeneratedModes(), axes=1)

    # Missing expression inputs do not prevent zero allocation or mode metadata use.
    basis = dlu.Basis(GeneratedModes(), zdx.StateRef("coeffs"))
    assert basis.axes == 2
    assert isinstance(basis.resolve(samples=1).coeffs, zdx.StateRef)


def test_unknown_generator_can_be_explicitly_empty():
    basis = dlu.Basis(zdx.StateRef("modes"), empty=True)
    pending = basis.resolve(state=zdx.State(modes=np.eye(2)))
    assert isinstance(pending, dlu.Basis)
    assert pending.coeffs is None
    assert np.array_equal(pending.basis, np.eye(2))
    assert np.array_equal(pending.set(coeffs=np.array([2.0, 3.0])).resolve(), [2, 3])


def make_expansion(family, empty=False, coeffs=None):
    if family == "basis":
        return dlu.Basis(np.ones((2, 3, 3)), coeffs, empty=empty)
    if family == "fourier":
        return dlu.FourierBasis(3, 2, coeffs, empty=empty)
    if family == "polynomial":
        return dlu.Polynomial(1, coeffs, grid=dlu.Axis(n=3, d=1), empty=empty)
    if family == "pasted":
        spec = dlu.PasteSpec((3, 3), (3, 3), [[0, 0]], [[0.0, 0.0]], [1.0, 1.0])
        return dlu.PastedBasis(np.ones((1, 2, 3, 3)), spec, coeffs, empty=empty)
    if family == "zernike":
        return dlu.Basis(
            dlu.Zernike([1, 4], grid=dlu.Grid(3, d=0.1)), coeffs, empty=empty
        )
    return dlu.CLIMBBasis(np.ones((2, 6, 6)), coeffs, empty=empty)


@pytest.mark.parametrize(
    "family", ["basis", "fourier", "polynomial", "pasted", "zernike", "climb"]
)
def test_zero_defaults_and_empty_expansions_can_be_filled_under_jit(family):
    complete = make_expansion(family)
    empty = make_expansion(family, empty=True)
    path = "expansion.coeffs" if family == "climb" else "coeffs"
    coefficients = complete.get(path)
    assert np.array_equal(coefficients, np.zeros_like(coefficients))
    assert empty.get(path) is None
    assert not hasattr(empty, "empty")
    assert isinstance(eqx.filter_jit(lambda m: m.resolve())(empty), type(empty))

    supplied = np.ones_like(coefficients)
    expected = make_expansion(family, empty=True, coeffs=supplied).resolve()
    result = eqx.filter_jit(lambda c: empty.set(path, c).resolve())(supplied)
    assert np.allclose(result, expected)


def test_zernike_definition_initialises_coefficients_and_keeps_empty_choice():
    grid = dlu.Grid(4, d=0.2)
    centers = np.array([[-0.1, 0], [0.1, 0]])
    definition = dl.ZernikeDef([1, 4])
    assert np.array_equal(definition.coeffs, np.zeros(2))
    for dynamic in (False, True):
        basis = definition(grid, centers=centers, dynamic=dynamic)
        assert type(basis) is dlu.Basis
        assert np.array_equal(basis.coeffs, np.zeros((2, 2)))
        assert basis.axes == 2
        assert np.array_equal(basis.resolve(), np.zeros(grid.shape))

        empty = dl.ZernikeDef([1, 4], empty=True)(grid, dynamic=dynamic)
        assert empty.coeffs is None
        assert isinstance(empty.resolve(), dlu.Basis)
        assert np.any(empty.set(coeffs=np.ones(2)).resolve() != 0)


def test_zernike_static_tools_share_the_numerical_api():
    assert dlu.Zernike.name(4) == "Defocus"
    assert dlu.Zernike.noll_indices(4) == (2, 0)
    assert dlu.Zernike.indices(orders=[0, 2]) == [1, 4, 5, 6]
    assert dlu.Zernike.radial_orders_to_indices([0, 2]) == [1, 4, 5, 6]
    c, k = dlu.Zernike.factors(4)
    rho = np.array([[0.0, 0.5]])
    assert np.allclose(dlu.Zernike.radial(rho, 2, c, k), 2 * rho**2 - 1, atol=2e-6)
    assert np.allclose(dlu.Zernike.azimuthal(rho, 0, 0), np.ones_like(rho))
