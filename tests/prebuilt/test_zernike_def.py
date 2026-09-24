"""Constructor-only Zernike definitions and their sampled/dynamic outputs."""

import equinox as eqx
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux as dl
import dLux.utils as dlu

from tests.helpers import assert_jittable


@pytest.mark.parametrize(
    "selection", [{"nolls": [1, 2, 4]}, {"order": 2}, {"orders": [0, 2]}]
)
@pytest.mark.parametrize("method", ["padded", "mapped"])
def test_construct_modes_inside_jit(selection, method):
    grid = dlu.Grid(7, d=0.1)

    def calculate(diameter):
        return dl.ZernikeDef(**selection, diameter=diameter, method=method)(grid).basis

    expected = calculate(np.array(1.0))
    assert np.allclose(eqx.filter_jit(calculate)(np.array(1.0)), expected, atol=2e-6)


def test_constructed_modes_retain_diameter_gradient():
    # Polar-coordinate derivatives at the origin remain a separate kernel issue.
    coordinates = np.array([[[0.1]], [[0.0]]])

    def defocus(diameter):
        return dl.ZernikeDef([4], diameter=diameter)(coordinates=coordinates).basis[
            0, 0, 0
        ]

    derivative = eqx.filter_jit(eqx.filter_grad(defocus))(np.array(1.0))
    expected_derivative = -16 * np.sqrt(3.0) * 0.1**2
    assert np.allclose(derivative, expected_derivative, rtol=2e-5)


def test_factory_returns_ordinary_or_dynamic_expansions():
    definition = dl.ZernikeDef([1, 2, 4], unit="nm")
    grid = dlu.Grid(8, d=0.1)
    sampled = definition(grid)
    dynamic = definition()
    stored = definition(grid, dynamic=True)

    assert isinstance(definition, dlu.Unitful)
    assert isinstance(definition.resolve(), dl.ZernikeDef)
    assert type(sampled) is dlu.Basis
    assert isinstance(dynamic, dlu.Basis)
    assert isinstance(dynamic.resolve(), dlu.Basis)
    assert dynamic.basis.grid is None
    assert sampled.coeffs.shape == dynamic.coeffs.shape == (3,)
    assert np.array_equal(sampled.coeffs, np.zeros(3))
    assert sampled.unit == dynamic.unit == "nm"
    assert np.allclose(stored.basis.evaluate(), sampled.basis)
    assert np.allclose(stored.resolve(), np.zeros(grid.shape))


@pytest.mark.parametrize("support", [True, 0.5])
def test_scalar_support_broadcasts_over_modes(support):
    grid = dlu.Grid(4, d=0.1)
    definition = dl.ZernikeDef([1, 2], coeffs=[2.0, 3.0], support=support)
    sampled = definition(grid)
    dynamic = definition(grid, dynamic=True)
    reference = dl.ZernikeDef([1, 2], coeffs=[2.0, 3.0])(grid)

    result = assert_jittable(lambda model: model.resolve(), dynamic)
    assert np.allclose(result, reference.resolve() * support)
    assert np.allclose(sampled.resolve(), result)


@pytest.mark.parametrize("constructor", [dl.ZernikeDef])
def test_units_do_not_accept_a_quantity_with_its_own_amplitude(constructor):
    with pytest.raises(TypeError, match="unit must be a string or None"):
        constructor([1], unit=zdx.Unit(x=10, unit="nm"))


@pytest.mark.parametrize("method", ["padded", "mapped"])
@pytest.mark.parametrize("n_sides", [0, 6])
def test_sampled_dynamic_unit_scaling_and_coefficient_solve(method, n_sides):
    grid = dlu.Grid(12, d=0.12)
    x, y = grid.coordinates
    support = x * x + y * y < 0.45**2
    coefficients = np.array([13.0, -5.0, 7.0])
    definition = dl.ZernikeDef(
        [1, 2, 4],
        coeffs=coefficients,
        diameter=1.0,
        n_sides=n_sides,
        norm="rms",
        method=method,
        unit="nm",
    )
    sampled = definition(grid, support=support)
    dynamic = definition(grid, support=support, dynamic=True)

    values = assert_jittable(lambda model: model.resolve(), dynamic)
    # Compare in coefficient units so SI absolute tolerances cannot hide a scale bug.
    assert np.allclose(values / 1e-9, sampled.resolve() / 1e-9, atol=2e-5)
    assert np.allclose(dynamic.basis.evaluate(), sampled.basis, atol=2e-6)
    rms = np.sqrt((sampled.basis**2).sum((-2, -1)) / support.sum())
    assert np.allclose(rms, 1.0, atol=2e-6)
    assert np.allclose(
        assert_jittable(lambda model: model.solve(values), dynamic),
        coefficients,
        atol=3e-5,
    )
    assert np.allclose(sampled.solve(values), coefficients, atol=3e-5)


def test_precalculated_basis_is_a_snapshot_with_live_coefficients():
    grid = dlu.Grid(8, d=0.1)
    definition = dl.ZernikeDef([2], coeffs=[2.0], diameter=1.0)
    sampled = definition(grid)
    dynamic = definition(grid, dynamic=True)
    enlarged = dynamic.set("basis.oversize", 0.2)

    assert np.allclose(enlarged.resolve(), sampled.resolve() / 1.2, atol=2e-6)
    assert np.allclose(
        sampled.set(coeffs=np.array([3.0])).resolve(),
        1.5 * sampled.resolve(),
        atol=2e-6,
    )
    assert np.allclose(dynamic.resolve(), sampled.resolve(), atol=2e-6)


def test_aperture_centres_define_two_coefficient_axes():
    grid = dlu.Grid((14, 8), d=0.15)
    centers = np.array([[-0.5, 0.0], [0.5, 0.0]])
    coefficients = np.array([[2.0, 3.0], [-4.0, 5.0]])
    definition = dl.ZernikeDef([1, 2], coeffs=coefficients, diameter=0.7, unit="nm")
    sampled = definition(grid, centers=centers)
    dynamic = definition(grid, centers=centers, dynamic=True)

    expected = []
    for center, coeffs in zip(centers, coefficients):
        coordinates = grid.coordinates - center[:, None, None]
        local = dl.ZernikeDef([1, 2], coeffs=coeffs, diameter=0.7, unit="nm")
        expected.append(local(coordinates=coordinates).resolve())
    expected = np.stack(expected).sum(0)
    result = assert_jittable(lambda model: model.resolve(), dynamic)

    assert sampled.axes == dynamic.coeffs_ndim == 2
    assert sampled.coeffs.shape == dynamic.basis.coeffs_shape == (2, 2)
    assert sampled.basis.shape == (2, 2) + grid.shape
    assert np.allclose(result / 1e-9, expected / 1e-9, atol=2e-5)
    assert np.allclose(sampled.resolve() / 1e-9, expected / 1e-9, atol=2e-5)


def test_definitions_belong_to_core_construction():
    assert issubclass(dl.ZernikeDef, dl.Def)
    assert dl.Def is dl.prebuilt.Def
    assert dl.ZernikeDef is dl.prebuilt.ZernikeDef
    assert dl.ApertureBuilder is dl.prebuilt.ApertureBuilder
    assert dl.Def.__module__ == "dLux.prebuilt.definitions"
    assert dl.ZernikeDef.__module__ == "dLux.prebuilt.prebuilt"
    assert not hasattr(dl.systems, "ZernikeDef")
    assert not hasattr(dl.systems, "ApertureBuilder")
    assert not hasattr(dlu, "Def")
    assert not hasattr(dlu, "ZernikeDef")
