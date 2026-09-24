"""Released functional FFT/MFT contracts; numerical classes have their own tests."""

import jax
import jax.numpy as np
import pytest

from dLux.utils.compat import propagation as legacy
from tests.helpers import assert_jittable


def _field(n):
    """A complex square field with neither reflection nor translation symmetry."""
    y, x = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return (1 + x / n + 0.2 * y / n) * np.exp(0.17j * x - 0.31j * y)


@pytest.mark.parametrize("n,pad", [(6, 1), (6, 2), (7, 1), (7, 3)])
@pytest.mark.parametrize("focal_length", [None, 0.7])
def test_fft_padding_sampling_power_and_inverse_roundtrip(n, pad, focal_length):
    field = _field(n)
    wavelength, spacing = 5e-4, 0.02
    focal, output_spacing = legacy.FFT(
        field, wavelength, spacing, focal_length=focal_length, pad=pad
    )
    expected_spacing = wavelength / (spacing * n * pad)
    if focal_length is not None:
        expected_spacing *= focal_length
    assert focal.shape == (n * pad, n * pad)
    assert np.allclose(output_spacing, expected_spacing)
    assert np.allclose(np.sum(np.abs(focal) ** 2), np.sum(np.abs(field) ** 2))

    recovered, recovered_spacing = legacy.FFT(
        focal,
        wavelength,
        output_spacing,
        focal_length=focal_length,
        pad=1,
        inverse=True,
    )
    padded = np.pad(field, n * (pad - 1) // 2)
    assert np.allclose(recovered, padded, rtol=2e-6, atol=2e-6)
    assert np.allclose(recovered_spacing, spacing)


@pytest.mark.parametrize("inverse", [False, True])
def test_fft_centred_point_has_flat_complex_output(inverse):
    n = 8
    field = np.zeros((n, n), dtype=complex).at[n // 2, n // 2].set(1 + 2j)
    result, _ = legacy.FFT(field, 5e-4, 0.02, pad=1, inverse=inverse)
    assert np.allclose(result, (1 + 2j) / n)


@pytest.mark.parametrize("n", [6, 7])
@pytest.mark.parametrize("focal_length", [None, 0.7])
def test_mft_native_sampling_preserves_power_and_inverts(n, focal_length):
    field = _field(n)
    wavelength, spacing = 5e-4, 0.02
    output_spacing = wavelength / (n * spacing)
    if focal_length is not None:
        output_spacing *= focal_length
    focal = legacy.MFT(
        field, wavelength, spacing, n, output_spacing, focal_length=focal_length
    )
    recovered = legacy.MFT(
        focal,
        wavelength,
        output_spacing,
        n,
        spacing,
        focal_length=focal_length,
        inverse=True,
    )
    assert np.allclose(np.sum(np.abs(focal) ** 2), np.sum(np.abs(field) ** 2))
    assert np.allclose(recovered, field, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("inverse", [False, True])
def test_mft_centred_point_has_flat_complex_output(inverse):
    n, n_out = 7, 9
    wavelength, spacing, output_spacing = 5e-4, 0.02, 0.003
    field = np.zeros((n, n), dtype=complex).at[n // 2, n // 2].set(1 + 2j)
    result = legacy.MFT(
        field, wavelength, spacing, n_out, output_spacing, inverse=inverse
    )
    # The MFT samples the Fourier integral with the input/output pixel-area factor.
    expected = (1 + 2j) * spacing * output_spacing / wavelength
    assert result.shape == (n_out, n_out)
    assert np.allclose(result, expected)


def test_mft_shift_uses_the_declared_units():
    field = _field(7)
    output_spacing = 0.003
    shift = np.array([1.3, -0.7])
    pixels = legacy.MFT(field, 5e-4, 0.02, 9, output_spacing, shift=shift)
    physical = legacy.MFT(
        field,
        5e-4,
        0.02,
        9,
        output_spacing,
        shift=shift * output_spacing,
        pixel=False,
    )
    centred = legacy.MFT(field, 5e-4, 0.02, 9, output_spacing)
    assert np.allclose(pixels, physical, rtol=2e-6, atol=2e-6)
    assert not np.allclose(pixels, centred)


@pytest.mark.parametrize("kind", ["fft", "mft"])
def test_released_transforms_jit_vmap_and_field_gradient(kind):
    field = _field(7)
    if kind == "fft":
        propagate = lambda x: legacy.FFT(x, 5e-4, 0.02, pad=1)[0]
    else:
        propagate = lambda x: legacy.MFT(x, 5e-4, 0.02, 7, 5e-4 / (7 * 0.02))
    assert_jittable(propagate, field, rtol=2e-6, atol=2e-6)
    fields = np.stack([field, field * np.exp(0.3j)])
    mapped = jax.jit(jax.vmap(propagate))(fields)
    expected = np.stack([propagate(incident) for incident in fields])
    assert np.allclose(mapped, expected, rtol=2e-6, atol=2e-6)

    def power(amplitude):
        return np.sum(np.abs(propagate(amplitude * field)) ** 2)

    derivative = jax.jit(jax.grad(power))(1.2)
    assert np.allclose(derivative, 2 * 1.2 * np.sum(np.abs(field) ** 2), rtol=2e-6)
