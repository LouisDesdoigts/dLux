"""Fourier transforms, phase-ramp shifts, and prepared FFT operations."""

from __future__ import annotations

from jax import Array
import jax.numpy as np

from ..grids.specs import _check_parity
from ..grids.array_ops import as_size
from ..sequential import BaseLayer
from .operations import (
    Chirp,
    Resize,
    _series,
)

__all__ = ["FFT", "fourier_shift"]


def fourier_shift(array: Array, shift: Array, pad: bool = True) -> Array:
    """Shift an array by a subpixel offset using Fourier interpolation.

    The final two axes are interpreted as ``(y, x)``, while ``shift`` is
    supplied in physical ``(x, y)`` order. Leading dimensions are preserved and
    receive the same shift. By default, the spatial axes are zero-padded before
    interpolation and cropped afterwards to suppress periodic wrapping.

    Parameters
    ----------
    array : Array
        Real or complex array with shape ``(..., ny, nx)``.
    shift : Array
        Length-two ``(x, y)`` shift measured in pixels. Positive values move
        samples towards increasing indices.
    pad : bool
        Whether to zero-pad each spatial axis by half its length at both ends.
        Disable this only when periodic boundary conditions are intended.

    Returns
    -------
    shifted : Array
        Fourier-interpolated array with the same shape and dtype kind as ``array``.

    Examples
    --------
    Shift an image half a pixel along the x-axis:

    >>> image = np.eye(5)
    >>> shifted = fourier_shift(image, np.array([0.5, 0.0]))

    Notes
    -----
    The operation is differentiable with respect to both the array and shift.
    Vectorise over ``shift`` with ``jax.vmap`` when different shifts are
    required for a batch.
    """
    # Convert and validate the array contract
    array = np.asarray(array)
    shift = np.asarray(shift)
    if array.ndim < 2:
        raise ValueError("array must have at least two dimensions.")
    if shift.shape != (2,):
        raise ValueError("shift must have shape (2,).")
    if not isinstance(pad, bool):
        raise TypeError("pad must be a bool.")

    # Pad the spatial axes to suppress periodic wrapping
    ny, nx = array.shape[-2:]
    py, px = (ny // 2, nx // 2) if pad else (0, 0)
    padding = [(0, 0)] * (array.ndim - 2) + [(py, py), (px, px)]
    padded = np.pad(array, padding)

    # Apply the Fourier-shift phase ramp over the final two axes
    ny, nx = padded.shape[-2:]
    fy, fx = np.fft.fftfreq(ny), np.fft.fftfreq(nx)
    phase = np.exp(-2j * np.pi * (fy[:, None] * shift[1] + fx[None, :] * shift[0]))
    shifted = np.fft.ifft2(np.fft.fft2(padded, axes=(-2, -1)) * phase, axes=(-2, -1))

    # Restore real inputs and crop back to the original spatial extent
    shifted = shifted if np.iscomplexobj(array) else shifted.real
    return shifted[..., py : py + array.shape[-2], px : px + array.shape[-1]]


class FFT(BaseLayer):
    """A unitary FFT over the final two spatial array axes.

    Leading axes broadcast with any array-valued direction. The input origin shifts
    to index zero before the transform and zero frequency shifts back to the centre.
    Set ``centered=False`` for raw transforms used inside convolution series.

    Parameters
    ----------
    n : int or tuple[int, int]
        Transform counts in (x, y) order; a scalar defines a square. The final
        input shape must match (ny, nx).
    inverse : bool or Array[bool]
        Use the inverse Fourier sign. Array directions broadcast with the leading
        field batch, allowing different directions and explicit outer batches.
    centered : bool
        Apply centred input and output conventions.
    """

    n: tuple[int, int]
    inverse: bool | Array = False
    centered: bool = True

    def __init__(self, n, *, inverse=False, centered=True, alias=None):
        """Initialise a unitary FFT with fixed transform topology.

        Parameters
        ----------
        n : int or tuple[int, int]
            Positive transform counts in (x, y) order; a scalar defines a square.
        inverse : bool or Array[bool]
            If true, apply the inverse Fourier transform. Boolean arrays select
            directions per leading field batch, including under JIT.
        centered : bool
            Apply centred input and output conventions when true.
        alias : mapping or sequence of pairs, optional
            Zodiax parameter aliases.
        """
        if not isinstance(inverse, bool):
            inverse = np.asarray(inverse, dtype=bool)
        self.n = as_size(n, 2, "n")
        self.inverse = inverse
        self.centered = bool(centered)

        self.alias = alias

    @classmethod
    def prepare(
        cls,
        n_in,
        n_fft=None,
        *,
        n_out=None,
        input=None,
        output=None,
        inverse=False,
        centered=True,
    ) -> BaseLayer:
        """Build one FFT or a series with optional resizing and phase factors.

        Counts use (x, y) order. Input factors apply before padding; output
        factors apply after cropping. Both are pairs of separable vectors.
        Every resize preserves the centre, so counts must retain their parity.
        """
        # Resolve working counts and check factors, bounds and centre-preserving parity.
        n_in = as_size(n_in, 2, "n_in")
        n_fft = n_in if n_fft is None else as_size(n_fft, 2, "n_fft")
        before = None if input is None else Chirp(input)
        after = None if output is None else Chirp(output)
        if n_out is None:
            n_out = n_fft if after is None else after.n
        n_out = as_size(n_out, 2, "n_out")
        for counts in (n_in, n_out):
            if any(size > limit for size, limit in zip(counts, n_fft)):
                raise ValueError("Input and output counts may not exceed FFT counts.")
        if before is not None and before.n != n_in:
            raise ValueError("Input factors must match input counts.")
        if after is not None and after.n != n_out:
            raise ValueError("Output factors must match output counts.")
        _check_parity(n_in, n_fft, "padding")
        _check_parity(n_fft, n_out, "cropping")

        # Assemble phase, resize and Fourier steps in their execution order.
        steps = []
        if before is not None:
            steps.append(before)
        if n_in != n_fft:
            steps.append(Resize(n_fft))
        steps.append(cls(n_fft, inverse=inverse, centered=centered))
        if n_out != n_fft:
            steps.append(Resize(n_out))
        if after is not None:
            steps.append(after)
        return _series(*steps)

    @property
    def n_in(self) -> tuple[int, int]:
        """Return input counts in physical-axis order."""
        return self.n

    @property
    def n_out(self) -> tuple[int, int]:
        """Return transformed output counts in physical-axis order."""
        return self.n

    @staticmethod
    def origin(grid) -> Array:
        """Return the physical coordinate assigned to a centred FFT's index zero.

        Grid centres lie between pixels for even counts; the FFT origin lies on
        the upper of those two pixels. This offset sets the spectral phase ramp.
        """
        even = np.asarray(grid.n) % 2 == 0
        return grid.c_si + np.where(even, grid.d_si / 2, 0.0)

    def apply(self, field, **context) -> Array:
        """Transform final axes using resolved physical-order counts."""
        n, inverse, centered = self.n, self.inverse, self.centered
        field = np.asarray(field)
        if field.shape[-2:] != n[::-1]:
            raise ValueError("field spatial shape must match n.")

        axes = (-2, -1)
        if centered:
            field = np.fft.ifftshift(field, axes=axes)

        if isinstance(inverse, bool):
            transform = np.fft.ifft2 if inverse else np.fft.fft2
            field = transform(field, axes=axes, norm="ortho")
        else:
            # For unitary transforms, IFFT(x) = conj(FFT(conj(x))). This keeps
            # one FFT when matrix values select different directions under JIT.
            inverse = inverse[..., None, None]
            field = np.where(inverse, np.conj(field), field)
            field = np.fft.fft2(field, axes=axes, norm="ortho")
            field = np.where(inverse, np.conj(field), field)
        return np.fft.fftshift(field, axes=axes) if centered else field
