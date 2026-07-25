"""Tests for dLux.utils.propagation."""

import jax.numpy as np
import pytest

import dLux.utils as dlu


def _field(spec):
    x, y = spec
    X, Y = np.meshgrid(x, y, indexing="xy")
    return np.exp(-20 * (X**2 + Y**2)) * np.exp(0.2j * X - 0.1j * Y)


@pytest.mark.parametrize(
    ("shape", "center"),
    [
        ((8, 6), (0.0, 0.0)),
        ((8, 6), (0.07, -0.04)),
        ((7, 9), (0.07, -0.04)),
    ],
)
@pytest.mark.parametrize("inverse", [False, True])
def test_fft_matches_native_mft(shape, center, inverse):
    spec_in = dlu.nd_axes(shape, (0.1, 0.13), offsets=tuple(-c for c in center))
    phasor = _field(spec_in)
    ABCD = dlu.abcd_fraunhofer(2.0)
    spec_out = dlu.FFT_spec(spec_in, 0.5, ABCD)

    actual, resolved = dlu.FFT(
        phasor,
        0.5,
        spec_in,
        focal_length=2.0,
        inverse=inverse,
    )
    expected = dlu.MFT(
        phasor,
        0.5,
        spec_in,
        spec_out,
        focal_length=2.0,
        inverse=inverse,
    )

    assert np.allclose(actual, expected, rtol=2e-5, atol=2e-6)
    assert all(np.allclose(a, b) for a, b in zip(resolved, spec_out))


def test_fft_inverse_roundtrip():
    spec_in = dlu.nd_axes((8, 6), (0.1, 0.13))
    phasor = _field(spec_in)
    focal, spec_focal = dlu.FFT(
        phasor,
        0.5,
        spec_in,
        pad=(2, 2),
        output_center=np.zeros(2),
    )
    recovered, spec_recovered = dlu.FFT(
        focal,
        0.5,
        spec_focal,
        inverse=True,
        output_center=np.zeros(2),
    )

    expected = dlu.pad_to(phasor, (16, 12))
    assert np.allclose(recovered, expected, rtol=2e-5, atol=2e-6)
    assert np.allclose(
        np.stack([axis.mean() for axis in spec_recovered]),
        0.0,
        atol=1e-7,
    )


def test_mft_inverse_roundtrip():
    spec_in = dlu.nd_axes((8, 6), (0.1, 0.13))
    phasor = _field(spec_in)
    spec_out = dlu.FFT_spec(spec_in, 0.5, dlu.abcd_fraunhofer(2.0))

    focal = dlu.MFT(phasor, 0.5, spec_in, spec_out, focal_length=2.0)
    recovered = dlu.MFT(
        focal,
        0.5,
        spec_out,
        spec_in,
        focal_length=2.0,
        inverse=True,
    )

    assert np.allclose(recovered, phasor, rtol=2e-5, atol=2e-6)


def test_shifted_abcd_fft_matches_mft():
    spec_in = dlu.nd_axes((8, 6), (0.1, 0.13), offsets=(-0.07, 0.04))
    phasor = _field(spec_in)
    ABCD = dlu.compose_abcd(
        [
            dlu.abcd_fraunhofer(2.0),
            dlu.abcd_free_space(0.3),
        ]
    )
    actual, spec_out = dlu.ABCD_FFT(
        phasor,
        0.5,
        spec_in,
        ABCD,
        output_center=np.asarray((0.2, -0.1)),
    )
    expected = dlu.ABCD_MFT(phasor, 0.5, spec_in, spec_out, ABCD)

    assert np.allclose(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("propagate", [dlu.FFT, dlu.MFT])
def test_inverse_defocus_requires_reverse_system(propagate):
    spec = dlu.nd_axes((8, 6), (0.1, 0.13))
    kwargs = {"spec_out": spec} if propagate is dlu.MFT else {}

    with pytest.raises(ValueError, match="no inverse flag"):
        propagate(
            _field(spec),
            0.5,
            spec,
            defocus=0.1,
            inverse=True,
            **kwargs,
        )
