import pytest
from jax import numpy as np, config

config.update("jax_debug_nans", True)
from dLux.utils.propagation import (
    FFT,
    MFT,
    fresnel_MFT,
    calc_nfringes,
    fresnel_AS_transfer_function,
    fresnel_AS,
)


@pytest.fixture
def phasor():
    return np.ones((32, 32)) * np.exp(1j * np.zeros((32, 32)))


@pytest.fixture
def wavelength():
    return 1.0


@pytest.fixture
def pixel_scale():
    return 0.1


@pytest.fixture
def focal_length():
    return 2.0


@pytest.fixture
def npixels_in():
    return 32


@pytest.fixture
def npixels_out():
    return 16


@pytest.fixture
def pixel_scale_out():
    return 0.05


@pytest.fixture
def shift():
    return np.array([0.1, 0.2])


@pytest.fixture
def focal_shift():
    return 0.1


# NOTE: 'Correctness' is non-trivial in this module, so most of these tests are
# high-level run checks for shapes and nans.


@pytest.mark.parametrize("inverse", [True, False])
def test_FFT(
    phasor, wavelength, pixel_scale, focal_length, npixels_in, inverse
):
    result, new_pixel_scale = FFT(
        phasor, wavelength, pixel_scale, focal_length, pad=2, inverse=inverse
    )
    assert result.shape == (2 * npixels_in, 2 * npixels_in)


def test_calc_nfringes(
    wavelength,
    npixels_in,
    pixel_scale,
    npixels_out,
    pixel_scale_out,
    focal_length,
):
    result = calc_nfringes(
        wavelength,
        npixels_in,
        pixel_scale,
        npixels_out,
        pixel_scale_out,
        focal_length,
    )
    assert isinstance(result, float)


# NOTE: The below parameterization produces two identical cases,
# the intention was likely to test two different cases
@pytest.mark.parametrize("inverse, pixel", [[True, False], [True, False]])
def test_MFT(
    phasor,
    wavelength,
    pixel_scale,
    npixels_out,
    pixel_scale_out,
    focal_length,
    shift,
    pixel,
    inverse,
):
    result = MFT(
        phasor,
        wavelength,
        pixel_scale,
        npixels_out,
        pixel_scale_out,
        focal_length,
        shift,
        pixel=pixel,
        inverse=inverse,
    )
    assert not np.isnan(result).any()
    assert result.shape == (npixels_out, npixels_out)


@pytest.mark.parametrize("inverse, pixel", [[True, False], [True, False]])
def test_fresnel_MFT(
    phasor,
    wavelength,
    pixel_scale,
    npixels_out,
    pixel_scale_out,
    shift,
    focal_length,
    focal_shift,
    pixel,
    inverse,
):
    result = fresnel_MFT(
        phasor,
        wavelength,
        pixel_scale,
        npixels_out,
        pixel_scale_out,
        focal_length,
        focal_shift,
        shift,
        pixel=pixel,
        inverse=inverse,
    )
    assert not np.isnan(result).any()
    assert result.shape == (npixels_out, npixels_out)


# --- Fresnel Angular Spectrum (AS) tests ---


def test_fresnel_AS_transfer_function_basic_properties(npixels_in, wavelength):
    """
    Validate basic properties of the Fresnel Angular Spectrum (AS) transfer function.

    This test checks that the computed transfer function:
      1. Has the correct padded shape and complex dtype.
      2. Contains no NaN values.
      3. Has unit magnitude across all spatial frequencies (paraxial, lossless form).
      4. Is symmetric about its center, as expected from its separable 1D construction.

    These checks verify that the kernel was constructed correctly as the outer product
    of identical 1D phase factors, which should yield a unitary, symmetric propagation
    operator in the paraxial regime.
    """
    N = npixels_in
    pad = 2
    diameter = 1.0
    prop_dist = 2.0

    H = fresnel_AS_transfer_function(
        N, wavelength, diameter, prop_dist, pad=pad
    )

    # --- Basic structure and dtype ---
    assert H.shape == (N * pad, N * pad)
    assert np.iscomplexobj(H)
    assert not np.isnan(H).any()

    # --- Physical properties ---
    mag = np.abs(H)
    assert np.allclose(
        mag, np.ones_like(mag), atol=1e-12
    ), "Magnitude deviates from unity"

    # For a separable outer product of identical 1D terms, H is symmetric (H.T == H)
    assert np.allclose(H.T, H, atol=1e-12), "Kernel symmetry violated"


def test_fresnel_AS_transfer_function_z0_is_ones(npixels_in, wavelength):
    """
    Verify that the Fresnel Angular Spectrum transfer function reduces to unity at z = 0

    When the propagation distance is zero, no phase change should occur and the
    transfer function must be identically 1 across the entire frequency domain.
    This ensures that the propagation kernel correctly reproduces the identity
    operation in the zero-distance limit.
    """
    N = npixels_in
    pad = 2
    diameter = 1.0
    prop_dist = 0.0

    H = fresnel_AS_transfer_function(
        N, wavelength, diameter, prop_dist, pad=pad
    )

    assert np.allclose(
        H, np.ones_like(H), atol=1e-12
    ), "Transfer function at z=0 should be exactly unity."


@pytest.mark.parametrize("pad", [1, 2, 3])
def test_fresnel_AS_output_determinism_consistency_and_energy(
    phasor, npixels_in, wavelength, pad
):
    """
    Validate deterministic, consistent, and energy-conserving behavior of `fresnel_AS`.

    This test checks that:
      1. The propagated field has the expected padded shape and complex dtype.
      2. The output contains no NaN values.
      3. Repeated calls with the same precomputed transfer function are deterministic.
      4. Providing a precomputed transfer function yields results consistent
         with those computed internally at runtime.
      5. The total optical energy (sum of |U|²) is conserved during propagation.

    These checks ensure that the Fresnel propagation routine is numerically stable,
    free of nondeterministic artifacts, energy-preserving, and that both
    computation paths (explicit vs. internal transfer-function construction)
    are equivalent.
    """
    N = npixels_in
    diameter = 1.0
    prop_dist = 1.0

    # --- Precompute transfer function ---
    H = fresnel_AS_transfer_function(
        N, wavelength, diameter, prop_dist, pad=pad
    )

    # --- Compute propagated outputs ---
    out1 = fresnel_AS(
        phasor, wavelength, diameter, prop_dist, pad=pad, transfer_function=H
    )
    out2 = fresnel_AS(
        phasor, wavelength, diameter, prop_dist, pad=pad, transfer_function=H
    )
    out3 = fresnel_AS(phasor, wavelength, diameter, prop_dist, pad=pad)

    # --- Structural checks ---
    assert out1.shape == (N * pad, N * pad), "Incorrect output array shape"
    assert np.iscomplexobj(out1), "Output array must be complex"
    assert not np.isnan(out1).any(), "Output contains NaN values"

    # --- Determinism check ---
    assert np.allclose(
        out1, out2, atol=1e-12
    ), "Non-deterministic behavior detected between repeated identical runs"

    # --- Consistency check ---
    assert np.allclose(
        out1, out3, atol=1e-12
    ), "Mismatch between explicit and internal transfer-function paths"

    # --- Energy conservation check ---
    energy_in = np.sum(np.abs(phasor) ** 2)
    energy_out = np.sum(np.abs(out1) ** 2)
    assert np.isclose(
        energy_in, energy_out, rtol=0.01
    ), f"Energy not conserved: input={energy_in:.6f}, output={energy_out:.6f}"
