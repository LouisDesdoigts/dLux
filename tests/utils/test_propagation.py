"""Tests for dLux.utils.propagation."""

import jax
import jax.numpy as np
import pytest

import dLux.utils as dlu
from tests.helpers import assert_differentiable, assert_jittable


def _field(spec):
    x, y = spec
    X, Y = np.meshgrid(x, y, indexing="xy")
    return np.exp(-20 * (X**2 + Y**2)) * np.exp(0.2j * X - 0.1j * Y)


def _assert_field_close(actual, expected):
    """Assert agreement within the floating-point accumulation budget."""
    error = np.max(np.abs(actual - expected)) / np.max(np.abs(expected))
    tolerance = actual.size * np.finfo(actual.real.dtype).eps
    assert error < tolerance, f"Relative field error {error} exceeds {tolerance}."


@pytest.mark.parametrize(
    ("shape", "center"),
    [((8, 6), (0.0, 0.0)), ((8, 6), (0.07, -0.04)), ((7, 9), (0.07, -0.04))],
)
@pytest.mark.parametrize("inverse", [False, True])
def test_fft_matches_native_mft(shape, center, inverse):
    spec_in = dlu.nd_axes(shape, (0.1, 0.13), offsets=tuple(-c for c in center))
    phasor = _field(spec_in)
    ABCD = dlu.abcd_fraunhofer(2.0)
    spec_out = dlu.FFT_spec(spec_in, 0.5, ABCD)

    actual, resolved = dlu.FFT(phasor, 0.5, spec_in, focal_length=2.0, inverse=inverse)
    expected = dlu.MFT(
        phasor, 0.5, spec_in, spec_out, focal_length=2.0, inverse=inverse
    )

    _assert_field_close(actual, expected)
    assert all(np.allclose(a, b) for a, b in zip(resolved, spec_out))


def test_fft_inverse_roundtrip():
    spec_in = dlu.nd_axes((8, 6), (0.1, 0.13))
    phasor = _field(spec_in)
    focal, spec_focal = dlu.FFT(
        phasor, 0.5, spec_in, pad=(2, 2), output_center=np.zeros(2)
    )
    recovered, spec_recovered = dlu.FFT(
        focal, 0.5, spec_focal, inverse=True, output_center=np.zeros(2)
    )

    expected = dlu.pad_to(phasor, (16, 12))
    _assert_field_close(recovered, expected)
    centers = np.stack([axis.mean() for axis in spec_recovered])
    assert np.max(np.abs(centers)) < np.finfo(centers.dtype).eps


def test_mft_inverse_roundtrip():
    spec_in = dlu.nd_axes((8, 6), (0.1, 0.13))
    phasor = _field(spec_in)
    spec_out = dlu.FFT_spec(spec_in, 0.5, dlu.abcd_fraunhofer(2.0))

    focal = dlu.MFT(phasor, 0.5, spec_in, spec_out, focal_length=2.0)
    recovered = dlu.MFT(focal, 0.5, spec_out, spec_in, focal_length=2.0, inverse=True)

    _assert_field_close(recovered, phasor)


@pytest.mark.parametrize(
    ("shape", "padding"),
    [((8, 8), (1, 1)), ((7, 7), (1, 1)), ((8, 6), (3, 5)), ((7, 5), (3, 5))],
)
def test_fourier_transform_identities(shape, padding):
    spec = dlu.nd_axes(shape, (0.1, 0.13))
    phasor = _field(spec)
    pad_to = tuple(n * p for n, p in zip(shape, padding))
    expected = dlu.pad_to(phasor, pad_to)
    padded_spec = dlu.FFT_pad(phasor, spec, pad=padding)[1]

    focal, focal_spec = dlu.FFT(
        phasor, 0.5, spec, pad=padding, output_center=np.zeros(2)
    )
    fft_inverse = dlu.FFT(
        focal, 0.5, focal_spec, inverse=True, output_center=np.zeros(2)
    )[0]
    mft_inverse = dlu.MFT(focal, 0.5, focal_spec, padded_spec, inverse=True)
    mft_forward = dlu.MFT(expected, 0.5, padded_spec, focal_spec)
    mixed_inverse = dlu.FFT(
        mft_forward, 0.5, focal_spec, inverse=True, output_center=np.zeros(2)
    )[0]
    twice_forward = dlu.FFT(focal, 0.5, focal_spec, output_center=np.zeros(2))[0]

    _assert_field_close(fft_inverse, expected)
    _assert_field_close(mft_inverse, expected)
    _assert_field_close(mixed_inverse, expected)
    _assert_field_close(twice_forward, -np.flip(expected))
    assert np.allclose(np.sum(np.abs(focal) ** 2), np.sum(np.abs(expected) ** 2))


@pytest.mark.parametrize("center", [(0.0, 0.0), (0.17, -0.11)])
def test_shifted_fft_matches_mft(center):
    spec = dlu.nd_axes((8, 6), (0.1, 0.13))
    phasor = _field(spec)
    actual, spec_out = dlu.FFT(
        phasor, 0.5, spec, pad=(3, 5), output_center=np.asarray(center)
    )
    padded_spec = dlu.FFT_pad(phasor, spec, pad=(3, 5))[1]
    expected = dlu.MFT(
        phasor=dlu.pad_to(phasor, (24, 30)),
        wavelength=0.5,
        spec_in=padded_spec,
        spec_out=spec_out,
    )

    _assert_field_close(actual, expected)


def test_shifted_abcd_fft_matches_mft():
    spec_in = dlu.nd_axes((8, 6), (0.1, 0.13), offsets=(-0.07, 0.04))
    phasor = _field(spec_in)
    ABCD = dlu.compose_abcd([dlu.abcd_fraunhofer(2.0), dlu.abcd_free_space(0.3)])
    actual, spec_out = dlu.ABCD_FFT(
        phasor, 0.5, spec_in, ABCD, output_center=np.asarray((0.2, -0.1))
    )
    expected = dlu.ABCD_MFT(phasor, 0.5, spec_in, spec_out, ABCD)

    _assert_field_close(actual, expected)


def test_fraunhofer_matches_abcd():
    spec = dlu.nd_axes((8, 6), (0.1, 0.13))
    phasor = _field(spec)
    ABCD = dlu.abcd_fraunhofer(2.0)
    spec_out = dlu.FFT_spec(spec, 0.5, ABCD)

    direct = dlu.MFT(phasor, 0.5, spec, spec_out, focal_length=2.0)
    abcd = dlu.ABCD_MFT(phasor, 0.5, spec, spec_out, ABCD)
    direct_fft, fft_spec = dlu.FFT(phasor, 0.5, spec, focal_length=2.0)
    abcd_fft, abcd_spec = dlu.ABCD_FFT(phasor, 0.5, spec, ABCD)

    _assert_field_close(direct, abcd)
    _assert_field_close(direct_fft, abcd_fft)
    assert all(np.allclose(a, b) for a, b in zip(fft_spec, abcd_spec))


@pytest.mark.parametrize("propagate", [dlu.FFT, dlu.MFT])
def test_fourier_transforms(propagate):
    spec = dlu.nd_axes((7, 5), (0.1, 0.13))
    phasor = _field(spec)
    spec_out = dlu.FFT_spec(spec, 0.5, dlu.abcd_fraunhofer(2.0))

    def function(field):
        if propagate is dlu.FFT:
            return propagate(field, 0.5, spec, focal_length=2.0)[0]
        return propagate(field, 0.5, spec, spec_out, focal_length=2.0)

    assert_jittable(function, phasor)
    assert_differentiable(function, phasor)


def test_fft_vectorises_over_fields():
    spec = dlu.nd_axes((7, 5), (0.1, 0.13))
    phasor = _field(spec)
    fields = np.stack((phasor, phasor * np.exp(0.3j)))
    propagate = lambda field: dlu.FFT(field, 0.5, spec, focal_length=2.0)[0]

    mapped = jax.vmap(propagate)(fields)
    expected = np.stack([propagate(field) for field in fields])

    assert np.allclose(mapped, expected)


@pytest.mark.parametrize("propagate", [dlu.FFT, dlu.MFT])
def test_inverse_defocus_requires_reverse_system(propagate):
    spec = dlu.nd_axes((8, 6), (0.1, 0.13))
    kwargs = {"spec_out": spec} if propagate is dlu.MFT else {}

    with pytest.raises(ValueError, match="no inverse flag"):
        propagate(_field(spec), 0.5, spec, defocus=0.1, inverse=True, **kwargs)
