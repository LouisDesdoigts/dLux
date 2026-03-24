from jax import Array
import jax.numpy as np
from abcdLux.mft import mft_kernels

__all__ = ["fourier_kernels", "eval_fourier_basis", "fft_spec", "fft_phase_ramp"]


def fft_spec(npixels_in, pixel_scale_in, wavelength, focal_length=None):
    """ """
    # Output pixel scale - fl * lambda / D
    d_out = wavelength / (npixels_in * pixel_scale_in)
    if focal_length is not None:
        d_out = d_out * focal_length

    # Return the centered case
    if npixels_in % 2 != 0:
        return d_out, 0.0
    return d_out, -0.5 * d_out


def fft_phase_ramp(xs, wavelength, shift, focal_length=None, inverse=False):
    """ """
    # Sign convention: positive for forward transform, negative for inverse transform
    sign = -1 if inverse else 1

    # Calculate the phase ramp to shift the FFT from native_out to spec_out
    alpha = wavelength if focal_length is None else wavelength * focal_length
    ramp = np.exp(sign * 2j * np.pi * xs * shift / alpha)
    return ramp[None, :] * ramp[:, None]


def _realpack_to_exp_transform(N: int) -> Array:
    """
    Build T (N,N) such that:
      c_exp = T @ a_real
    with real-packed a_real = [DC, cos1, sin1, cos2, sin2, ...]
    and exp-packed c_exp = [0, +1, -1, +2, -2, ...].
    """
    T = np.zeros((N, N), dtype=complex)

    # DC maps directly
    T = T.at[0, 0].set(1.0)

    # Number of cos/sin harmonic pairs present (excluding Nyquist if even)
    if N % 2 == 0:
        n_harm = (N // 2) - 1
        has_nyq = True
    else:
        n_harm = (N - 1) // 2
        has_nyq = False

    if n_harm > 0:
        k = np.arange(1, n_harm + 1)

        # exp indices for +k and -k in [0, +1, -1, +2, -2, ...]
        ip = 2 * k - 1
        im = 2 * k

        # real indices for cosk and sink in [DC, cos1, sin1, cos2, sin2, ...]
        ic = 2 * k - 1
        is_ = 2 * k

        half = np.array(0.5, dtype=complex)
        jhalf = np.array(0.5j, dtype=complex)

        # c(+k) = 0.5*cosk  - 0.5j*sink
        T = T.at[ip, ic].set(half)
        T = T.at[ip, is_].set(-jhalf)

        # c(-k) = 0.5*cosk  + 0.5j*sink
        T = T.at[im, ic].set(half)
        T = T.at[im, is_].set(+jhalf)

    # Nyquist (even N): only cosine exists; sin term is identically zero on-grid
    if has_nyq:
        T = T.at[N - 1, N - 1].set(1.0)

    return T


def fourier_kernels(N: int, M: int):
    """
    Returns Kx_real, Ky_real such that:
      out ≈ (Ky_real @ C_real @ Kx_real.T).real
    with C_real real-packed coefficients (N,N) and output shape (M,M).
    """

    # exp basis frequency indices: [0, +1, -1, +2, -2, ...]
    # (same along x and y)
    def k_order_monotone(N):
        if N % 2 == 1:
            kmax = (N - 1) // 2
            ks = [0]
            for m in range(1, kmax + 1):
                ks.extend([m, -m])
        else:
            kmax = N // 2
            ks = [0]
            for m in range(1, kmax):
                ks.extend([m, -m])
            ks.append(kmax)  # Nyquist
        return np.asarray(ks)

    # Input and output coordinates
    kx = k_order_monotone(N)
    x = (np.arange(M) - (M // 2)) / M

    # spec_in is kx/ky, spec_out is x/y.
    Kx, Ky = mft_kernels(spec_in=kx, spec_out=x, alpha=2 * np.pi, weight=1.0)

    # Fold transforms into kernels: Ky_real = Ky @ Ty, Kx_real = Kx @ Tx
    Tx = _realpack_to_exp_transform(N)
    return Kx @ Tx, Ky @ Tx


def eval_fourier_basis(C: Array, Kx: Array, Ky: Array) -> Array:
    """
    Evaluate with cached real-output kernels.
    C: (N,N) real-packed coeffs (float)
    Kx: (M,N) complex
    Ky: (M,N) complex
    Returns: (M,M) real (float)
    """
    return (Ky @ (C @ Kx.T)).real
