"""Numerical operators broadcast leading axes without assigning physical roles."""

from itertools import product
from math import prod

import equinox as eqx
import jax
import jax.numpy as np
import pytest

import dLux.utils as dlu
from dLux.utils import propagation as prop


def _values(shape, offset=0.0):
    values = np.arange(prod(shape), dtype=float).reshape(shape) + offset
    return (1 + values / values.size) * np.exp(0.17j * values)


@pytest.mark.parametrize(
    "operator_batch, field_batch",
    [
        ((), (2, 3)),
        ((2,), (2,)),
        ((2,), (3, 2)),
        ((2, 1), (1, 2)),
        ((2, 1), (3,)),
        ((2,), ()),
    ],
    ids=["shared", "paired", "right-aligned", "equal-outer", "outer", "new-axis"],
)
def test_mft_broadcasting_matches_independent_matrix_products(
    operator_batch, field_batch
):
    kernel_x = _values(operator_batch + (2, 4), 0.3)
    kernel_y = _values(operator_batch + (5, 3), 1.1)
    field = _values(field_batch + (3, 4), 2.5)
    operation = prop.MFT((kernel_x, kernel_y))
    batch = np.broadcast_shapes(operator_batch, field_batch)
    kernels_x = np.broadcast_to(kernel_x, batch + (2, 4))
    kernels_y = np.broadcast_to(kernel_y, batch + (5, 3))
    fields = np.broadcast_to(field, batch + (3, 4))
    expected = np.stack(
        [
            kernels_y[index] @ fields[index] @ kernels_x[index].T
            for index in product(*(range(size) for size in batch))
        ]
    ).reshape(batch + (5, 2))

    actual = eqx.filter_jit(lambda op, values: op(values))(operation, field)

    assert actual.shape == batch + (5, 2)
    assert np.allclose(actual, expected, rtol=2e-6, atol=2e-5)


def test_mft_axis_kernels_broadcast_independently():
    kernel_x = _values((2, 1, 2, 4), 0.3)
    kernel_y = _values((2, 5, 3), 1.1)
    field = _values((3, 4), 2.5)
    operation = prop.MFT((kernel_x, kernel_y))
    expected = np.stack(
        [np.stack([y @ field @ x[0].T for y in kernel_y]) for x in kernel_x]
    )

    actual = eqx.filter_jit(lambda op, values: op(values))(operation, field)

    assert operation.batch_shape == (2, 2)
    assert actual.shape == (2, 2, 5, 2)
    assert np.allclose(actual, expected, rtol=2e-6, atol=2e-5)


def test_mft_outer_batches_retain_independent_phase_derivatives():
    coordinates_in = np.linspace(-0.4, 0.5, 4)
    coordinates_out = np.linspace(-0.3, 0.6, 2)
    products = coordinates_out[:, None] * coordinates_in[None, :]
    kernel_y = _values((5, 3), 0.7)
    fields = _values((2, 3, 4), 1.3)
    phases = np.asarray([-1.1, 0.4])

    def calculate(phases):
        kernel_x = np.exp(1j * phases[:, None, None, None] * products)
        return prop.MFT((kernel_x, kernel_y))(fields)

    expected_gradient = []
    for phase in phases:
        derivative = 1j * products * np.exp(1j * phase * products)
        expected_gradient.append(
            sum(np.real(kernel_y @ field @ derivative.T).sum() for field in fields)
        )

    actual = jax.jit(calculate)(phases)
    gradient = jax.jit(jax.grad(lambda value: calculate(value).real.sum()))(phases)

    assert actual.shape == (2, 2, 5, 2)
    assert np.all(np.abs(np.asarray(expected_gradient)) > 0.1)
    assert np.allclose(gradient, np.asarray(expected_gradient), rtol=2e-6, atol=2e-5)


def test_mft_fused_factors_broadcast_independent_batches():
    operation = prop.MFT((_values((2, 4)), _values((5, 3), 0.4)))
    before = prop.Chirp((_values((2, 1, 4)), _values((2, 1, 3), 0.1)))
    after = prop.Chirp((_values((2, 2), 0.2), _values((2, 5), 0.3)))
    field = _values((3, 4), 0.7)
    expected = after(operation(before(field)))
    fused = operation.fuse_input(before).fuse_output(after)

    actual = eqx.filter_jit(lambda op, values: op(values))(fused, field)

    assert actual.shape == (2, 2, 5, 2)
    assert np.allclose(actual, expected, rtol=2e-6, atol=2e-4)


def test_chirp_outer_batches_match_explicit_products_and_composition():
    first = prop.Chirp((_values((2, 1, 4)), _values((2, 1, 3), 0.1)))
    second = prop.Chirp((_values((2, 4), 0.2), _values((2, 3), 0.3)))
    field = _values((3, 4), 0.7)
    expected = []
    for index in range(2):
        rows = []
        for other in range(2):
            factor_x = first.factors[0][index, 0] * second.factors[0][other]
            factor_y = first.factors[1][index, 0] * second.factors[1][other]
            rows.append(field * factor_y[:, None] * factor_x[None, :])
        expected.append(np.stack(rows))
    expected = np.stack(expected)

    sequential = eqx.filter_jit(lambda a, b, values: b(a(values)))(first, second, field)
    fused = eqx.filter_jit(lambda a, b, values: a.then(b)(values))(first, second, field)

    assert sequential.shape == (2, 2, 3, 4)
    assert np.allclose(sequential, expected, rtol=2e-6, atol=2e-5)
    assert np.allclose(fused, expected, rtol=2e-6, atol=2e-5)


def test_fft_equal_sized_batch_axes_use_right_alignment():
    field = _values((2, 2, 4, 5))
    operation = prop.FFT((5, 4), inverse=np.asarray([False, True]), centered=False)
    expected = np.stack(
        [
            np.stack(
                (
                    np.fft.fft2(values[0], norm="ortho"),
                    np.fft.ifft2(values[1], norm="ortho"),
                )
            )
            for values in field
        ]
    )

    actual = eqx.filter_jit(lambda op, values: op(values))(operation, field)

    assert np.allclose(actual, expected, rtol=2e-6, atol=2e-5)


def test_fft_signed_matrix_outer_batches_match_independent_lct():
    grid = dlu.Grid((4, 6), (0.02, 0.03), unit="m")
    fields = _values((2,) + grid.shape)
    distances = np.asarray([[-0.1], [0.2]])

    def calculate(distance):
        definition = prop.FFTPropagator(abcd=prop.ABCD(free_space=distance))
        return definition(fields, grid, 5e-4, return_grid=True)

    actual, output_grid = eqx.filter_jit(calculate)(distances)
    expected = []
    for index, distance in enumerate(distances[:, 0]):
        local_grid = output_grid.set(
            d=output_grid.d[index, 0], c=output_grid.c[index, 0]
        )
        definition = prop.LCT(prop.ABCD(free_space=distance), local_grid)
        expected.append(np.stack([definition(field, grid, 5e-4) for field in fields]))

    assert actual.shape == (2, 2) + grid.shape
    assert np.allclose(actual, np.stack(expected), rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("kind", ["direct", "fft", "space", "spectral"])
def test_physical_preparation_broadcasts_with_fields_and_matches_scalar_calls(kind):
    grid = dlu.Grid((4, 6), (0.02, 0.03), (0.01, -0.02), unit="m")
    target = dlu.Grid((2, 4), (0.01, 0.02), (0.005, -0.001), unit="m")
    definitions = {
        "direct": prop.Propagator(target, focal_length=0.7, defocus=0.01),
        "fft": prop.FFTPropagator(dlu.FFTSpec(c=(0.005, -0.001)), focal_length=0.7),
        "space": prop.FreeSpace(0.02),
        "spectral": prop.SpectralPropagator(
            prop.ABCD(free_space=0.02), dlu.FFTSpec(), grid
        ),
    }
    definition = definitions[kind]
    wavelengths = np.asarray([[4e-4], [7e-4]])
    fields = _values((1, 2) + grid.shape)
    prepared, output_grid = definition.prepare(grid, wavelengths, return_grid=True)

    actual = eqx.filter_jit(lambda op, value: op(value))(prepared, fields)
    direct, direct_grid = eqx.filter_jit(
        lambda model, wavelength, field: model(
            field, grid, wavelength, return_grid=True
        )
    )(definition, wavelengths, fields)
    expected = np.stack(
        [
            np.stack([definition(field, grid, wavelength) for field in fields[0]])
            for wavelength in wavelengths[:, 0]
        ]
    )

    assert actual.shape == (2, 2) + output_grid.shape
    assert np.allclose(actual, expected, rtol=3e-5, atol=3e-5)
    assert np.allclose(direct, expected, rtol=3e-5, atol=3e-5)
    assert np.allclose(direct_grid.d_si, output_grid.d_si)
    assert np.allclose(direct_grid.c_si, output_grid.c_si)


@pytest.mark.parametrize("kind", ["direct", "fft", "space", "spectral"])
def test_physical_propagation_promotes_scalar_grids_to_squares(kind):
    grid = dlu.Grid(4, 0.02, unit="m")
    target = dlu.Grid(5, 0.01, unit="m")
    definitions = {
        "direct": prop.Propagator(target, abcd=prop.ABCD(free_space=0.1)),
        "fft": prop.FFTPropagator(abcd=prop.ABCD(free_space=0.1)),
        "space": prop.FreeSpace(0.1),
        "spectral": prop.SpectralPropagator(
            prop.ABCD(free_space=0.1), dlu.FFTSpec(), target
        ),
    }
    definition = definitions[kind]
    field = _values((4, 4))
    actual, output_grid = eqx.filter_jit(
        lambda model, grid_in, value: model(value, grid_in, 5e-4, return_grid=True)
    )(definition, grid, field)
    expected = definition(field, grid, 5e-4)

    assert output_grid.ndim == 2
    assert actual.shape == output_grid.shape
    assert np.allclose(actual, expected, rtol=3e-5, atol=3e-5)
