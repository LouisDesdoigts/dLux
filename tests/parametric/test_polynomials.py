import jax
import jax.numpy as np
import pytest

import dLux.utils as dlu
from dLux.parametric.polynomials import (
    DynamicZernike,
    DynamicZernikeBasis,
    PolynomialBasis,
    ZernikeBasis,
)
from dLux.wavefronts import Wavefront

coords = dlu.pixel_coords(8, diameter=2.0)


class TestDynamicZernike:
    def test_properties_and_calculation(self):
        zernike = DynamicZernike(4)

        assert zernike.j == 4
        assert zernike.name == dlu.zernike_name(4)
        assert zernike.calculate(coords).shape == (8, 8)

    def test_invalid_index(self):
        with pytest.raises(ValueError, match="greater than 0"):
            DynamicZernike(0)

    def test_diameter_contract(self):
        coordinates = dlu.pixel_coords(8, diameter=1.1)
        zernike = DynamicZernike(7)

        assert np.allclose(
            zernike.calculate(coordinates, diameter=1.1),
            dlu.zernike(7, coordinates, diameter=1.1),
        )

    def test_polygonal_calculation(self):
        zernike = DynamicZernike(4)

        assert np.allclose(
            zernike.calculate(coords, nsides=6),
            dlu.polike(6, 4, coords),
        )


class TestZernikeBasis:
    def test_radial_orders_match_noll_indices(self):
        coefficients = np.arange(3.0)
        by_index = ZernikeBasis(coords, js=[4, 5, 6], coefficients=coefficients)
        by_order = ZernikeBasis(coords, radial_orders=[2], coefficients=coefficients)

        assert np.allclose(by_index.basis, by_order.basis)
        assert np.allclose(by_index.evaluate(), by_order.evaluate())

    def test_requires_one_selector(self):
        with pytest.raises(ValueError, match="exactly one"):
            ZernikeBasis(coords)
        with pytest.raises(ValueError, match="exactly one"):
            ZernikeBasis(coords, js=[1], radial_orders=[0])

    @pytest.mark.parametrize(
        ("selector", "match"),
        [({"js": []}, "At least one"), ({"js": [0]}, "greater than zero")],
    )
    def test_invalid_mode_selection(self, selector, match):
        with pytest.raises(ValueError, match=match):
            ZernikeBasis(coords, **selector)

    def test_negative_radial_order(self):
        with pytest.raises(ValueError, match="non-negative"):
            ZernikeBasis(coords, radial_orders=[-1])

    def test_coefficient_shape_must_match_modes(self):
        with pytest.raises(ValueError, match="Coefficient shape"):
            ZernikeBasis(coords, js=[4, 5], coefficients=np.ones(3))

    def test_nondefault_diameter(self):
        coordinates = dlu.pixel_coords(8, diameter=1.1)
        basis = ZernikeBasis(coordinates, js=[4, 5], diameter=1.1)

        assert np.allclose(
            basis.basis,
            dlu.zernike_basis([4, 5], coordinates, diameter=1.1),
        )


class TestDynamicZernikeBasis:
    def test_explicit_coordinates(self):
        basis = DynamicZernikeBasis(radial_orders=[2])

        assert basis.coefficient_shape == (3,)
        assert basis.calculate_basis(coordinates=coords).shape == (3, 8, 8)
        assert basis.evaluate(coordinates=coords).shape == (8, 8)

    def test_requires_coordinate_context(self):
        with pytest.raises(ValueError, match="wavefront or coordinates"):
            DynamicZernikeBasis(js=[1]).evaluate()

    @pytest.mark.parametrize("nsides", [-1, 1, 2])
    def test_invalid_polygon_sides(self, nsides):
        with pytest.raises(ValueError, match="zero or greater"):
            DynamicZernikeBasis(js=[1], nsides=nsides)

    def test_coefficient_shape_must_match_modes(self):
        with pytest.raises(ValueError, match="Coefficient shape"):
            DynamicZernikeBasis(js=[4, 5], coefficients=np.ones(3))

    def test_wavefront_infers_diameter(self):
        diameter = 1.1
        wavefront = Wavefront(700e-9, 8, diameter=diameter)
        coefficients = np.arange(3.0)
        dynamic = DynamicZernikeBasis(radial_orders=[2], coefficients=coefficients)
        explicit = ZernikeBasis(
            wavefront.coordinates(),
            radial_orders=[2],
            coefficients=coefficients,
            diameter=diameter,
        )

        assert np.allclose(dynamic.evaluate(wavefront=wavefront), explicit.evaluate())

    def test_explicit_coordinates_accept_diameter(self):
        diameter = 0.8
        coordinates = dlu.pixel_coords(8, diameter=diameter)
        dynamic = DynamicZernikeBasis(js=[4, 5], coefficients=np.ones(2))
        explicit = ZernikeBasis(
            coordinates, js=[4, 5], coefficients=np.ones(2), diameter=diameter
        )

        assert np.allclose(
            dynamic.evaluate(coordinates=coordinates, diameter=diameter),
            explicit.evaluate(),
        )

    def test_owns_optional_diameter(self):
        diameter = 0.8
        coordinates = dlu.pixel_coords(8, diameter=diameter)
        dynamic = DynamicZernikeBasis(
            js=[4, 5], coefficients=np.ones(2), diameter=diameter
        )
        explicit = ZernikeBasis(
            coordinates, js=[4, 5], coefficients=np.ones(2), diameter=diameter
        )
        assert np.allclose(
            dynamic.evaluate(coordinates=coordinates), explicit.evaluate()
        )
        with pytest.raises(ValueError, match="diameter"):
            DynamicZernikeBasis(js=[1], diameter=0)

    def test_jit_and_gradient(self):
        wavefront = Wavefront(700e-9, 8, diameter=1.1)
        basis = DynamicZernikeBasis(radial_orders=[2])
        evaluate = jax.jit(
            lambda coefficients: basis.set(coefficients=coefficients).evaluate(
                wavefront=wavefront
            )
        )
        gradient = jax.grad(lambda coefficients: np.sum(evaluate(coefficients) ** 2))
        coefficients = np.ones(3)

        assert evaluate(coefficients).shape == (8, 8)
        assert np.isfinite(gradient(coefficients)).all()


class TestPolynomialBasis:
    def test_degree_and_evaluation(self):
        basis = PolynomialBasis(2)
        explicit = dlu.polynomial_basis(coords, basis.powers)

        assert basis.coefficient_shape == (6,)
        assert explicit.shape == (6, 8, 8)
        assert np.allclose(
            basis.evaluate(coordinates=coords),
            dlu.eval_basis(explicit, basis.coefficients),
        )

    def test_solve_basis(self):
        coefficients = np.arange(6.0)
        basis = PolynomialBasis(2, coefficients)
        value = basis.evaluate(coordinates=coords)

        assert np.allclose(
            basis.solve_basis(value, coordinates=coords), coefficients, atol=1e-5
        )

    def test_arbitrary_coordinate_dimensions(self):
        coordinates = np.stack(
            np.meshgrid(
                np.linspace(-1.0, 1.0, 3),
                np.linspace(-1.0, 1.0, 4),
                np.linspace(-1.0, 1.0, 5),
                indexing="ij",
            )[:2]
        )
        basis = PolynomialBasis(2)

        assert basis.calculate_basis(coordinates=coordinates).shape == (6, 3, 4, 5)
        assert basis.evaluate(coordinates=coordinates).shape == (3, 4, 5)

    def test_negative_degree(self):
        with pytest.raises(ValueError, match="non-negative"):
            PolynomialBasis(-1)

    def test_coefficient_shape_must_match_degree(self):
        with pytest.raises(ValueError, match="Coefficient shape"):
            PolynomialBasis(2, coefficients=np.ones(5))

    def test_requires_coordinate_context(self):
        with pytest.raises(ValueError, match="wavefront or coordinates"):
            PolynomialBasis(1).evaluate()
