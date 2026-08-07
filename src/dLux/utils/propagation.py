"""High-level optical propagation wrappers around abcdLux."""

import jax.numpy as np
from abcdLux import asm, fraunhofer, lct
from abcdLux.coords import unpack_coord_spec
from abcdLux.mft import mft as _mft
from jax import Array

import dLux.utils as dlu

__all__ = [
    "FFT_pad",
    "FFT_spec",
    "FFT_shift",
    "FFT_ramp",
    "FFT",
    "MFT",
    "ABCD_MFT",
    "ABCD_FFT",
    "ASM",
]


def _resolve_pad(spec_in, pad=None, pad_to=None):
    """Resolve exclusive factor-based or absolute FFT padding."""
    if pad is not None and pad_to is not None:
        raise ValueError("Provide only one of pad or pad_to.")
    x_in, y_in = unpack_coord_spec(spec_in)
    if pad is not None:
        pad = dlu.as_size(pad, 2, "pad")
        pad_to = (x_in.size * pad[0], y_in.size * pad[1])
    return (x_in, y_in), pad_to


def _spec_parameters(spec):
    """Return axis sizes, spacings, and centres."""
    x, y = unpack_coord_spec(spec)
    sizes = (x.size, y.size)
    spacings = (x[1] - x[0], y[1] - y[0])
    centers = ((x[-1] + x[0]) / 2, (y[-1] + y[0]) / 2)
    return sizes, spacings, centers


def FFT_pad(
    phasor: Array,
    spec_in: Array | tuple,
    pad: int | tuple[int, int] | None = None,
    pad_to: int | tuple[int, int] | None = None,
) -> tuple[Array, tuple[Array, Array]]:
    """Pad a field and its coordinate axes exactly once."""
    spec_in, pad_to = _resolve_pad(spec_in, pad, pad_to)
    if pad_to is None:
        return phasor, spec_in
    _, spacings, centers = _spec_parameters(spec_in)
    spec_in = dlu.nd_axes(
        pad_to, spacings, offsets=tuple(-center for center in centers)
    )
    return dlu.pad_to(phasor, pad_to), spec_in


def FFT_spec(
    spec_in: Array | tuple, wavelength: float, ABCD: Array
) -> tuple[Array, Array]:
    """Return the native FFT output coordinate axes."""
    return lct.lct_fft_output_spec(
        spec_in=spec_in, lam=wavelength, ABCD=ABCD, npad=None
    )


def FFT_shift(
    spec_out: Array | tuple, output_center: Array | None = None
) -> tuple[tuple[Array, Array], Array | None]:
    """Shift FFT output axes to a requested physical centre."""
    x_out, y_out = unpack_coord_spec(spec_out)
    if output_center is None:
        return (x_out, y_out), None
    native_center = np.asarray(((x_out[-1] + x_out[0]) / 2, (y_out[-1] + y_out[0]) / 2))
    output_center = dlu.as_axis(output_center, 2, "output_center")
    shift = output_center - native_center
    return (x_out + shift[0], y_out + shift[1]), shift


def FFT_ramp(
    wavelength: float,
    spec: Array | tuple,
    ABCD: Array,
    shift: Array | None,
    plane: str = "input",
    inverse: bool = False,
) -> Array | float:
    """Return the phase ramp associated with a shifted FFT output grid.

    The output-grid piston is the constant term in
    ``|coordinate + shift|² - |coordinate|²``. It does not affect intensity, but it
    must be retained when propagated fields are coherently compared or combined.
    """
    if shift is None:
        return 1.0
    if plane not in ("input", "output"):
        raise ValueError("plane must be 'input' or 'output'.")
    sizes, spacings, centers = _spec_parameters(spec)
    coordinates = dlu.nd_coords(sizes, spacings, offsets=tuple(-c for c in centers))
    _, b, _, d = np.asarray(ABCD).flatten()
    field = np.ones(coordinates.shape[1:], dtype=complex)
    if plane == "input":
        sign = 1 if inverse else -1
        return dlu.tilt(field, coordinates, sign * shift / b, wavelength)
    ramp = dlu.tilt(field, coordinates, d * shift / b, wavelength)
    piston = np.exp(1j * np.pi * d * np.sum(shift**2) / (wavelength * b))
    return piston * ramp


def _collins_phase(inverse):
    """Return the Collins global phase for a forward or inverse Fourier step."""
    return 1j if inverse else -1j


def ABCD_MFT(
    phasor: Array,
    wavelength: float,
    spec_in: Array | tuple,
    spec_out: Array | tuple,
    ABCD: Array,
    apply_out_curv: bool = True,
) -> Array:
    """Propagate through an arbitrary ABCD system onto explicit axes."""
    return lct.lct_prop(
        u_in=phasor,
        spec_in=spec_in,
        spec_out=spec_out,
        lam=wavelength,
        ABCD=np.asarray(ABCD),
        apply_out_curv=apply_out_curv,
    )


def ABCD_FFT(
    phasor: Array,
    wavelength: float,
    spec_in: Array | tuple,
    ABCD: Array,
    pad: int | tuple[int, int] | None = None,
    pad_to: int | tuple[int, int] | None = None,
    output_center: Array | None = None,
    apply_out_curv: bool = True,
) -> tuple[Array, tuple[Array, Array]]:
    """Propagate an ABCD system onto native or shifted FFT axes."""
    ABCD = np.asarray(ABCD)
    phasor, spec_in = dlu.FFT_pad(phasor, spec_in, pad, pad_to)
    spec_native = dlu.FFT_spec(spec_in, wavelength, ABCD)
    spec_out, shift = dlu.FFT_shift(spec_native, output_center)
    phasor = phasor * dlu.FFT_ramp(wavelength, spec_in, ABCD, shift)
    field, _ = lct.lct_prop_fft(
        u_in=phasor,
        spec_in=spec_in,
        lam=wavelength,
        ABCD=ABCD,
        npad=None,
        apply_out_curv=apply_out_curv,
    )
    if apply_out_curv:
        field *= dlu.FFT_ramp(wavelength, spec_native, ABCD, shift, plane="output")
    return field, spec_out


def _fraunhofer_abcd(focal_length, defocus, inverse=False):
    """Build the directed ABCD matrix for defocused focal propagation."""
    if inverse:
        return dlu.compose_abcd(
            [dlu.abcd_free_space(-defocus), dlu.abcd_fraunhofer(-focal_length)]
        )
    return dlu.compose_abcd(
        [dlu.abcd_fraunhofer(focal_length), dlu.abcd_free_space(defocus)]
    )


def _fraunhofer_fft(
    phasor: Array,
    wavelength: float,
    spec_in: Array | tuple,
    focal_length: float,
    inverse: bool,
) -> tuple[Array, tuple[Array, Array]]:
    """Apply a pure optical FFT without calculating LCT chirps."""
    ABCD = dlu.abcd_fraunhofer(focal_length)
    spec_out = dlu.FFT_spec(spec_in, wavelength, ABCD)
    _, spacings, centers = _spec_parameters(spec_out)
    coordinates = dlu.nd_coords(
        phasor.shape[-2:][::-1], spacings, offsets=tuple(-center for center in centers)
    )
    sizes_in, spacings_in, centers_in = _spec_parameters(spec_in)
    input_origin = np.asarray(
        tuple(
            center + (0.5 * spacing if size % 2 == 0 else 0.0)
            for size, spacing, center in zip(sizes_in, spacings_in, centers_in)
        )
    )
    norm = np.sqrt(phasor.shape[-2] * phasor.shape[-1])
    if inverse:
        field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(phasor)))
        field *= norm
    else:
        field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(phasor)))
        field /= norm
    sign = 1 if inverse else -1
    field = dlu.tilt(field, coordinates, sign * input_origin / focal_length, wavelength)
    return field * _collins_phase(inverse), spec_out


def MFT(
    phasor: Array,
    wavelength: float,
    spec_in: Array | tuple,
    spec_out: Array | tuple,
    focal_length: float | None = None,
    defocus: float | None = None,
    inverse: bool = False,
    apply_out_curv: bool = True,
) -> Array:
    """Propagate to an explicit grid using a pure MFT or defocused LCT."""
    focal_length = 1.0 if focal_length is None else focal_length
    field = phasor
    if defocus is None:
        if inverse:
            scale, kernel_x, kernel_y = fraunhofer.fraunhofer_kernels(
                spec_in=spec_in, spec_out=spec_out, lam=wavelength, f=focal_length
            )
            return (
                _collins_phase(inverse)
                * scale
                * _mft(phasor, kernel_x, kernel_y, left_conj=True, right_conj=True)
            )
        return _collins_phase(inverse) * fraunhofer.fraunhofer_prop(
            u_pupil=phasor,
            spec_in=spec_in,
            spec_out=spec_out,
            lam=wavelength,
            f=focal_length,
        )
    else:
        field = dlu.ABCD_MFT(
            phasor=field,
            wavelength=wavelength,
            spec_in=spec_in,
            spec_out=spec_out,
            ABCD=_fraunhofer_abcd(focal_length, defocus, inverse),
            apply_out_curv=apply_out_curv,
        )
    return field


def FFT(
    phasor: Array,
    wavelength: float,
    spec_in: Array | tuple,
    pad: int | tuple[int, int] | None = None,
    pad_to: int | tuple[int, int] | None = None,
    focal_length: float | None = None,
    defocus: float | None = None,
    inverse: bool = False,
    output_center: Array | None = None,
    apply_out_curv: bool = True,
) -> tuple[Array, tuple[Array, Array]]:
    """Propagate using a pure FFT or a defocused FFT-based LCT."""
    focal_length = 1.0 if focal_length is None else focal_length
    if inverse and defocus is not None:
        raise ValueError("Inverse Fresnel propagation is not supported by FFT.")
    phasor, spec_in = dlu.FFT_pad(phasor, spec_in, pad, pad_to)
    if defocus is None:
        ABCD = dlu.abcd_fraunhofer(focal_length)
        spec_native = dlu.FFT_spec(spec_in, wavelength, ABCD)
        spec_out, shift = dlu.FFT_shift(spec_native, output_center)
        phasor *= dlu.FFT_ramp(wavelength, spec_in, ABCD, shift, inverse=inverse)
        field, spec_native = _fraunhofer_fft(
            phasor, wavelength, spec_in, focal_length, inverse
        )
        return field, spec_out
    field, spec_out = dlu.ABCD_FFT(
        phasor=phasor,
        wavelength=wavelength,
        spec_in=spec_in,
        ABCD=_fraunhofer_abcd(focal_length, defocus),
        output_center=output_center,
        apply_out_curv=apply_out_curv,
    )
    return field, spec_out


def ASM(
    phasor: Array,
    wavelength: float,
    spec_in: Array | tuple,
    distance: float,
    pad: int | tuple[int, int] | None = None,
    pad_to: int | tuple[int, int] | None = None,
    crop: bool = True,
) -> Array:
    """Propagate through free space using the angular-spectrum method."""
    shape = phasor.shape[-2:]
    phasor, spec_in = dlu.FFT_pad(phasor, spec_in, pad, pad_to)
    kernel, nx, ny = asm.asm_kernels(
        spec_in=spec_in, lam=wavelength, z=distance, npad=None
    )
    field = asm.asm_kernel_prop(u_pad=phasor, H=kernel, Nx_in=nx, Ny_in=ny, crop=False)
    return dlu.crop_to(field, shape[::-1]) if crop else field
