"""Prepared array operations keep two spatial axes and ordinary immutable updates."""

import equinox as eqx
import jax.numpy as np
import pytest

import dLux.utils as dlu


def test_fft_topology_updates_and_compiled_specialisation():
    original = dlu.propagation.FFT((4, 6))
    updated = original.set(n=(6, 8), centered=False)
    transform = eqx.filter_jit(lambda op, field: op(field))
    for operation in (original, updated):
        field = np.arange(operation.n[0] * operation.n[1]).reshape(operation.n[::-1])
        expected = np.fft.fft2(field, norm="ortho")
        if operation.centered:
            expected = np.fft.fftshift(
                np.fft.fft2(np.fft.ifftshift(field), norm="ortho")
            )
        assert np.allclose(transform(operation, field), expected, atol=2e-5)
    assert original.n == (4, 6) and original.centered


def test_fft_direction_arrays_add_batch_axes_instead_of_selecting_spatial_rows():
    operation = dlu.propagation.FFT((4, 3), inverse=np.array([False, True, False]))
    field = np.arange(12.0).reshape((3, 4)) * (1 + 0.2j)
    shifted = np.fft.ifftshift(field)
    forward = np.fft.fftshift(np.fft.fft2(shifted, norm="ortho"))
    inverse = np.fft.fftshift(np.fft.ifft2(shifted, norm="ortho"))
    expected = np.stack((forward, inverse, forward))
    for apply in (operation, eqx.filter_jit(operation)):
        output = apply(field)
        assert output.shape == (3, 3, 4)
        assert np.allclose(output, expected, rtol=2e-6, atol=2e-5)


@pytest.mark.parametrize("centered", [False, True])
def test_fft_batched_directions_match_separate_transforms_under_jit(centered):
    field = np.arange(120.0).reshape((2, 3, 4, 5))
    field = field + 1j * np.sin(field)
    directions = np.array([[False], [True]])
    operation = dlu.propagation.FFT((5, 4), inverse=directions, centered=centered)

    expected = []
    for values, transform in zip(field, (np.fft.fft2, np.fft.ifft2)):
        if centered:
            values = np.fft.ifftshift(values, axes=(-2, -1))
        values = transform(values, axes=(-2, -1), norm="ortho")
        if centered:
            values = np.fft.fftshift(values, axes=(-2, -1))
        expected.append(values)

    output = eqx.filter_jit(lambda op, values: op(values))(operation, field)

    assert output.shape == field.shape
    assert np.allclose(output, np.stack(expected), rtol=2e-6, atol=2e-5)


def test_resize_updates_shape_without_changing_centre_or_leading_axes():
    original = dlu.propagation.Resize((4, 6))
    updated = original.set(n=(8, 10))
    field = np.arange(48.0).reshape(2, 6, 4)
    result = eqx.filter_jit(lambda op, value: op(value))(updated, field)
    expected = np.pad(field, ((0, 0), (2, 2), (2, 2)))
    assert np.array_equal(result, expected)
    assert np.array_equal(original(result), field)
    with pytest.raises(ValueError, match="parity"):
        updated.set(n=(9, 10))(field)


def test_mft_matches_direct_separable_contraction_with_complex_batches():
    grid_in = dlu.Grid((5, 4), (0.3, 0.2))
    grid_out = dlu.Grid((3, 6), (0.2, 0.1))
    field = np.arange(40.0).reshape(2, 4, 5) * (1 + 0.2j)
    x_in, y_in = grid_in.axes
    x_out, y_out = grid_out.axes
    coefficient = -2.3
    x_kernel = np.exp(1j * coefficient * x_out[:, None] * x_in[None, :])
    y_kernel = np.exp(1j * coefficient * y_out[:, None] * y_in[None, :])
    expected = np.stack([y_kernel @ item @ x_kernel.T for item in field])
    operation = dlu.propagation.MFT.from_axes(grid_in.axes, grid_out.axes, coefficient)
    assert np.allclose(eqx.filter_jit(operation)(field), expected, rtol=2e-6, atol=2e-5)
