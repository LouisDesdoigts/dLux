"""Tests for dLux.utils.optical.refractive."""

import equinox as eqx
import jax
import jax.numpy as np
import pytest

import dLux as dl
import dLux.utils as dlu
import zodiax as zdx

from tests.helpers import assert_differentiable, assert_jittable


@pytest.mark.parametrize("kind, power", [(dl.PolynomialIndex, 1), (dl.CauchyIndex, -2)])
def test_index_construction_accepts_a_traced_scale(kind, power):
    wavelengths = np.asarray([0.8e-6, 1.0e-6, 1.2e-6])
    scale = np.asarray(1e-6)

    def evaluate(value):
        return kind([1.5, 0.01], scale=value).resolve(wavelengths=wavelengths)

    actual = assert_jittable(evaluate, scale)
    expected = 1.5 + 0.01 * (wavelengths / scale) ** power
    assert np.allclose(actual, expected)
    gradient = jax.jit(jax.grad(lambda value: evaluate(value).sum()))(scale)
    expected_gradient = (-power * 0.01 * (wavelengths / scale) ** power / scale).sum()
    assert np.allclose(gradient, expected_gradient)


@pytest.mark.parametrize(
    ("index", "parameter"),
    [
        (dl.CauchyIndex([1.5, 0.01, 0.001]), "coeffs"),
        (dl.PolynomialIndex([1.5, 0.01, -0.001]), "coeffs"),
        (
            dl.InterpolatedIndex(
                [0.8e-6, 1.0e-6, 1.2e-6],
                [1.52, 1.50, 1.49],
            ),
            "array",
        ),
    ],
)
def test_refractive_index_contract(index, parameter, make_wavefront):
    wavefront = make_wavefront(wavelength=np.asarray([0.9e-6, 1.0e-6, 1.1e-6]))

    output = assert_jittable(
        lambda value: value.resolve(wavelengths=wavefront.wavelength),
        index,
    )
    assert output.shape == wavefront.wavelength.shape
    assert_differentiable(
        lambda value: index.set(parameter, value).resolve(
            wavelengths=wavefront.wavelength
        ),
        getattr(index, parameter),
    )


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: dl.CauchyIndex([]),
        lambda: dl.CauchyIndex([[1.5]]),
        lambda: dl.PolynomialIndex(),
        lambda: dl.PolynomialIndex([1.5], scale=[1e-6]),
        lambda: dl.PolynomialIndex([1.5], scale=0.0),
        lambda: dl.InterpolatedIndex([1e-6], [1.5]),
        lambda: dl.InterpolatedIndex([1e-6, 2e-6], [1.5]),
        lambda: dl.InterpolatedIndex([1e-6, 2e-6], [[1.5, 1.4]]),
    ],
)
def test_validation(constructor):
    # Constructor-error coverage; numerical behaviour is checked separately above.
    with pytest.raises(ValueError):
        constructor()


def test_interpolated_index_rejects_complex_wavelengths():
    with pytest.raises(TypeError, match="real"):
        dlu.InterpolatedIndex([1e-6 + 1j, 2e-6], [1.5, 1.4])


@pytest.mark.parametrize(
    "index",
    [
        dl.PolynomialIndex([1.5, 0.01]),
        dl.CauchyIndex([1.5, 0.01]),
        dl.InterpolatedIndex([500e-9, 700e-9], [1.51, 1.50]),
    ],
)
def test_refractive_models_do_not_infer_wavelengths_from_wavefront(
    index, make_wavefront
):
    wavefront = make_wavefront(wavelength=600e-9)
    assert isinstance(index.resolve(wavefront=wavefront), type(index))
    if isinstance(index, dl.InterpolatedIndex):
        with pytest.raises(TypeError, match="wavelengths"):
            index.evaluate(wavefront=wavefront)
        with pytest.raises(ValueError, match="explicitly"):
            index.evaluate(wavelengths=None, wavefront=wavefront)
    else:
        with pytest.raises(zdx.Unresolved, match="grid or coordinates"):
            index.evaluate(wavefront=wavefront)
    expected = index.evaluate(wavelengths=600e-9)
    result = assert_jittable(
        lambda model: model.resolve(wavelengths=600e-9, wavefront=wavefront), index
    )
    assert np.allclose(result, expected)


@pytest.mark.parametrize("kind", [dl.PolynomialIndex, dl.CauchyIndex])
def test_refractive_stored_sampling_and_separate_spatial_context(kind):
    axis = dlu.Axis([500, 600, 700], unit="nm")
    model = kind([1.5, 0.01], grid=axis)
    expected = kind(model.coeffs).resolve(wavelengths=axis.axes[0])
    values = assert_jittable(lambda value: value.resolve(), model)
    assert np.allclose(values, expected)
    query = np.asarray([[550e-9, 650e-9], [600e-9, 700e-9]])
    spatial_grid = dlu.Grid(n=(3, 3), d=1.0, unit="m")
    assert np.allclose(model.resolve(grid=spatial_grid), expected)
    optic = dl.Optic(thickness=1e-6, n=kind(model.coeffs))
    partial = optic.resolve(grid=spatial_grid, coordinates=spatial_grid.coordinates)
    assert isinstance(partial.n, kind)
    assert np.allclose(partial.resolve(wavelengths=axis.axes[0]).n, expected)
    values = model.resolve(
        wavelengths=query, grid=spatial_grid, coordinates=spatial_grid.coordinates
    )
    assert values.shape == query.shape
    assert np.allclose(values, kind(model.coeffs).resolve(wavelengths=query))
    assert_differentiable(lambda value: model.set(coeffs=value).resolve(), model.coeffs)
    with pytest.raises(ValueError, match="strict"):
        kind([1.5, 0.01], grid=axis, strict=True).resolve(wavelengths=query)


@pytest.mark.parametrize("kind", [dlu.PolynomialIndex, dlu.CauchyIndex])
def test_degree_initialises_editable_zero_coefficients(kind):
    index = kind(degree=2)
    assert np.array_equal(index.coeffs, np.zeros(3))
    assert np.allclose(index.resolve(wavelengths=600e-9), 0)
    updated = index.set(coeffs=np.array([1.5, 0.0, 0.0]))
    assert np.allclose(updated.resolve(wavelengths=600e-9), 1.5)
    assert np.array_equal(index.coeffs, np.zeros(3))


def test_interpolated_index_prepares_expressions_at_source_knots():
    class IndexSamples(zdx.Expression):
        def evaluate(self, *, coordinates, offset, **context):
            return offset + (coordinates / 1e-6) ** 2

    index = dlu.InterpolatedIndex([500e-9, 600e-9, 700e-9], IndexSamples())
    query = np.array([[550e-9, 650e-9]])
    expected = 1.5 + (query / 1e-6) ** 2
    # Deliberately different source/query shapes expose premature table resolution.
    values = assert_jittable(
        lambda model: model.resolve(
            wavelengths=query, coordinates=np.zeros((2, 4, 4)), offset=1.5
        ),
        index,
    )
    assert values.shape == query.shape
    assert np.allclose(values, expected)
    prepared = index.prepare(offset=1.5)
    assert np.allclose(prepared.resolve(wavelengths=query), expected)
    assert isinstance(index.resolve(wavelengths=query), dlu.InterpolatedIndex)


def test_interpolation_method_can_change_through_an_immutable_update():
    index = dlu.InterpolatedIndex([1e-6, 2e-6, 3e-6], [1.0, 4.0, 9.0])
    linear = index.set(method="linear")
    # Changing a Python method string must select a new filtered JIT calculation.
    sample = eqx.filter_jit(lambda model: model.resolve(wavelengths=1.5e-6))
    assert np.allclose(sample(index), 2.25)
    assert np.allclose(sample(linear), 2.5)
    assert index.method == "cubic2"
