import jax
import jax.numpy as np
import pytest

import dLux.utils as dlu
from dLux.parametric import CLIMBBasis, ExplicitBasis, FourierBasis, SplineBasis


class TestExplicitBasis:
    def test_coefficient_aliases(self):
        coefficients = np.arange(3.0)
        parametric = ExplicitBasis(np.eye(3), coefficients)

        assert np.array_equal(parametric.coeffs, coefficients)
        assert np.array_equal(parametric.c, coefficients)
        assert np.array_equal(parametric.alpha, coefficients)

    def test_arbitrary_coefficient_and_output_shapes(self):
        coefficients = np.arange(6.0).reshape(2, 3)
        basis = np.arange(120.0).reshape(2, 3, 4, 5)
        parametric = ExplicitBasis(basis, coefficients)

        assert parametric.evaluate().shape == (4, 5)
        assert np.allclose(parametric.evaluate(), dlu.eval_basis(basis, coefficients))

    def test_solve_basis(self):
        basis = np.eye(6).reshape(2, 3, 2, 3)
        coefficients = np.arange(6.0).reshape(2, 3)
        parametric = ExplicitBasis(basis, coefficients)

        assert np.allclose(parametric.solve_basis(parametric.evaluate()), coefficients)

    def test_default_coefficients_require_shape(self):
        basis = np.ones((2, 3, 4))
        with pytest.raises(ValueError, match="coefficients or coefficient_shape"):
            ExplicitBasis(basis)

        parametric = ExplicitBasis(basis, coefficient_shape=(2, 3))
        assert np.allclose(parametric.coefficients, 0)

    def test_leading_dimensions_must_match(self):
        with pytest.raises(ValueError, match="leading basis dimensions"):
            ExplicitBasis(np.ones((2, 3, 4)), np.ones((3, 2)))

    def test_jit_and_gradient(self):
        basis = np.arange(24.0).reshape(3, 2, 4)

        evaluate = jax.jit(
            lambda coefficients: ExplicitBasis(basis, coefficients).evaluate()
        )
        gradient = jax.grad(lambda coefficients: np.sum(evaluate(coefficients) ** 2))

        coefficients = np.arange(6.0).reshape(3, 2)
        assert evaluate(coefficients).shape == (4,)
        assert np.isfinite(gradient(coefficients)).all()


class TestCLIMBBasis:
    @pytest.fixture
    def basis(self):
        return np.arange(4 * 6 * 6.0).reshape(4, 6, 6)

    @pytest.fixture
    def coefficients(self):
        return np.linspace(-1.0, 1.0, 4)

    def test_latent_and_binarised_evaluation(self, basis, coefficients):
        parametric = CLIMBBasis(basis, coefficients, values=(-1.0, 1.0), oversample=3)
        latent = dlu.eval_basis(basis, coefficients)
        expected = 2 * dlu.soft_binarise(latent, 3) - 1

        assert np.allclose(parametric.evaluate_latent(), latent)
        assert np.allclose(parametric.evaluate(), expected)
        assert parametric.evaluate().shape == (2, 2)

    def test_default_output_values(self, basis, coefficients):
        output = CLIMBBasis(basis, coefficients, oversample=2).evaluate()

        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_solve_basis_targets_latent_field(self, basis, coefficients):
        parametric = CLIMBBasis(basis, coefficients)

        assert np.allclose(
            parametric.solve_basis(parametric.evaluate_latent()),
            coefficients,
            rtol=1e-4,
            atol=1e-4,
        )

    def test_values_validation(self, basis, coefficients):
        with pytest.raises(ValueError, match="exactly two"):
            CLIMBBasis(basis, coefficients, values=(0.0, 0.5, 1.0))

    @pytest.mark.parametrize("oversample", [0, -1])
    def test_oversample_must_be_positive(self, basis, coefficients, oversample):
        with pytest.raises(ValueError, match="positive"):
            CLIMBBasis(basis, coefficients, oversample=oversample)

    def test_output_must_be_square_and_divisible(self, coefficients):
        with pytest.raises(ValueError, match="square 2D"):
            CLIMBBasis(np.ones((4, 6, 5)), coefficients)
        with pytest.raises(ValueError, match="divisible"):
            CLIMBBasis(np.ones((4, 5, 5)), coefficients, oversample=3)

    def test_jit_and_gradient(self, basis, coefficients):
        evaluate = jax.jit(
            lambda values: CLIMBBasis(
                basis, values, values=(-1.0, 1.0), oversample=3
            ).evaluate()
        )
        gradient = jax.grad(lambda values: np.sum(evaluate(values) ** 2))

        assert evaluate(coefficients).shape == (2, 2)
        assert np.isfinite(gradient(coefficients)).all()


class TestFourierBasis:
    def test_optimized_and_explicit_evaluation_match(self):
        coefficients = np.arange(12.0).reshape(3, 4)
        basis = FourierBasis((8, 10), (3, 4), coefficients)

        assert basis.evaluate().shape == (8, 10)
        assert np.allclose(
            basis.evaluate(), dlu.eval_basis(basis.calculate_basis(), coefficients)
        )

    def test_resize(self):
        basis = FourierBasis(8, 3, scale=2.0)
        resized = basis.resize((5, 7), scale=3.0)

        assert resized.coefficient_shape == (3, 3)
        assert resized.evaluate().shape == (5, 7)
        assert not hasattr(resized, "scale")

    def test_invalid_coefficients(self):
        with pytest.raises(ValueError, match="Coefficient shape"):
            FourierBasis(8, 3, np.ones((2, 3)))

    @pytest.mark.parametrize(
        ("npix", "n_modes", "error"),
        [
            ((4, 5, 6), 3, TypeError),
            (8, (2, 3, 4), TypeError),
            (0, 3, ValueError),
            (8, -1, ValueError),
        ],
    )
    def test_invalid_grid_inputs(self, npix, n_modes, error):
        with pytest.raises(error):
            FourierBasis(npix, n_modes)

    def test_scalar_and_rectangular_forms(self):
        square = FourierBasis(8, 3)
        rectangular = FourierBasis((8, 10), (2, 3))

        assert square.coefficient_shape == (3, 3)
        assert square.evaluate().shape == (8, 8)
        assert rectangular.coefficient_shape == (2, 3)
        assert rectangular.evaluate().shape == (8, 10)


class TestSplineBasis:
    @pytest.mark.parametrize(
        ("n_knots", "coefficient_shape"), [(2, (2, 2)), (3, (3, 3))]
    )
    def test_knot_count(self, n_knots, coefficient_shape):
        basis = SplineBasis(8, n_knots, method="linear")

        assert basis.coefficient_shape == coefficient_shape
        assert basis.evaluate().shape == (8, 8)
        assert basis.calculate_basis().shape == coefficient_shape + (8, 8)

    def test_constant_interpolation(self):
        basis = SplineBasis(9, 3, np.ones((3, 3)), method="linear")
        assert np.allclose(basis.evaluate(), 1.0)

    def test_explicit_and_optimized_evaluation_match(self):
        coefficients = np.arange(9.0).reshape(3, 3)
        basis = SplineBasis(8, 3, coefficients, method="linear")

        assert np.allclose(
            basis.evaluate(),
            dlu.eval_basis(basis.calculate_basis(), coefficients),
        )

    def test_rectangular_grids(self):
        basis = SplineBasis((8, 10), (2, 3), method="linear")
        assert basis.coefficient_shape == (3, 2)
        assert basis.evaluate().shape == (10, 8)

    @pytest.mark.parametrize(
        ("npix", "n_knots"),
        [((4, 5, 6), 2), (8, (2, 3, 4)), (8, 1)],
    )
    def test_invalid_grid_shapes(self, npix, n_knots):
        with pytest.raises(ValueError):
            SplineBasis(npix, n_knots)

    @pytest.mark.parametrize("npix", [0, -1, (8, 0)])
    def test_npixels_must_be_positive(self, npix):
        with pytest.raises(ValueError, match="positive"):
            SplineBasis(npix, 2)

    def test_coefficient_shape_must_match_knots(self):
        with pytest.raises(ValueError, match="Coefficient shape"):
            SplineBasis((8, 10), (2, 3), coefficients=np.ones((2, 3)))

    def test_jit_and_gradient(self):
        coefficients = np.arange(9.0).reshape(3, 3)
        evaluate = jax.jit(
            lambda values: SplineBasis(8, 3, values, method="linear").evaluate()
        )
        gradient = jax.grad(lambda values: np.sum(evaluate(values) ** 2))

        assert evaluate(coefficients).shape == (8, 8)
        assert np.isfinite(gradient(coefficients)).all()
