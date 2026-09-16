"""Prepared unitary Fourier-transform evaluations."""

from __future__ import annotations


import equinox as eqx
from jax import Array
import jax.numpy as np

from ..helpers import as_size
from .operations import (
    BaseOperation,
    Chirp,
    Resize,
    _factor_sizes,
    _prepare_factors,
    _series,
)

__all__ = ["FFT"]


def _require_within(n, limit, name) -> None:
    """Require every physical-order count to fit within its FFT count."""
    if any(size > bound for size, bound in zip(n, limit)):
        raise ValueError(f"{name} may not exceed the FFT counts.")


def _resize_step(n_in, n_out):
    """Return a resize operation only when the sample counts differ."""
    return None if n_in == n_out else Resize(n_out)


def _resized_fft(n_in, n_fft, *, inverse=False, centered=True) -> BaseOperation:
    """Build an optional central resize followed by a fixed-size FFT."""
    n_in = as_size(n_in, name="n_in")
    n_fft = as_size(n_fft, len(n_in), "n_fft")
    _require_within(n_in, n_fft, "n_in")

    resize = _resize_step(n_in, n_fft)
    if resize is None:
        return FFT(n_fft, inverse=inverse, centered=centered)
    return _series(resize, FFT(n_fft, inverse=inverse, centered=centered))


def _output_fft(n_in, n_fft, factors, *, inverse=False) -> BaseOperation:
    """Build a centred FFT, crop, and output-factor series."""
    n_in = as_size(n_in, name="n_in")
    n_fft = as_size(n_fft, len(n_in), "n_fft")
    factors = _prepare_factors(factors)
    n_out = _factor_sizes(factors)

    if len(factors) != len(n_in):
        raise ValueError("factors must contain one factor per transform axis.")
    _require_within(n_in, n_fft, "n_in")
    _require_within(n_out, n_fft, "output factor lengths")

    steps = (
        _resize_step(n_in, n_fft),
        FFT(n_fft, inverse=inverse),
        _resize_step(n_fft, n_out),
        Chirp(factors),
    )
    return _series(*(step for step in steps if step is not None))


def _shifted_fft(n_fft, input, output, *, inverse=False) -> BaseOperation:
    """Build a centred FFT between separable input and output factors."""
    input = _prepare_factors(input, "input")
    output = _prepare_factors(output, "output")
    n_fft = as_size(n_fft, len(input), "n_fft")
    n_in = _factor_sizes(input)
    n_out = _factor_sizes(output)

    if len(output) != len(input):
        raise ValueError("input and output must contain equal factor counts.")
    _require_within(n_in, n_fft, "input factor lengths")
    _require_within(n_out, n_fft, "output factor lengths")

    steps = (
        Chirp(input),
        _resize_step(n_in, n_fft),
        FFT(n_fft, inverse=inverse),
        _resize_step(n_fft, n_out),
        Chirp(output),
    )
    return _series(*(step for step in steps if step is not None))


def _fft_convolution(n_in, n_out, transfer) -> BaseOperation:
    """Build a raw FFT, transfer-factor, inverse-FFT convolution series."""
    transfer = _prepare_factors(transfer, "transfer")
    n_fft = _factor_sizes(transfer)
    ndim = len(transfer)
    n_in = as_size(n_in, ndim, "n_in")
    n_out = as_size(n_out, ndim, "n_out")

    _require_within(n_in, n_fft, "n_in")
    _require_within(n_out, n_fft, "n_out")

    steps = (
        _resize_step(n_in, n_fft),
        FFT(n_fft, centered=False),
        Chirp(transfer),
        FFT(n_fft, inverse=True, centered=False),
        _resize_step(n_fft, n_out),
    )
    return _series(*(step for step in steps if step is not None))


class FFT(BaseOperation):
    """A unitary FFT over fixed final array axes.

    Leading axes are preserved. The input origin is shifted to index zero before the
    transform and the zero-frequency output is shifted back to the centre by default.
    Set ``centered=False`` for raw transforms used inside convolution series.

    Parameters
    ----------
    n : int or tuple[int, ...]
        Transform counts in physical-axis order. The input must already match them.
    inverse : bool or Array[bool]
        Use the inverse Fourier sign. Array directions align with the leading
        field prefix, allowing different directions within a batch.
    centered : bool
        Apply centred input and output conventions.
    """

    n: tuple[int, ...] = eqx.field(static=True)
    inverse: bool | Array
    centered: bool = eqx.field(static=True)

    def __init__(self, n, *, inverse=False, centered=True, alias=None):
        """Initialise a unitary FFT with fixed transform topology.

        Parameters
        ----------
        n : int or tuple[int, ...]
            Positive transform counts in physical-axis order.
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
        self.n = as_size(n, name="n")
        self.inverse = inverse
        self.centered = bool(centered)

        self.alias = alias

    @property
    def n_in(self) -> tuple[int, ...]:
        """Return input counts in physical-axis order."""
        return self.n

    @property
    def n_out(self) -> tuple[int, ...]:
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
        ndim = len(n)
        if field.shape[-ndim:] != n[::-1]:
            raise ValueError("field spatial shape must match n.")

        axes = tuple(range(field.ndim - ndim, field.ndim))
        if centered:
            field = np.fft.ifftshift(field, axes=axes)

        if isinstance(inverse, bool):
            transform = np.fft.ifftn if inverse else np.fft.fftn
            field = transform(field, axes=axes, norm="ortho")
        else:
            # For unitary transforms, IFFT(x) = conj(FFT(conj(x))). This keeps
            # one FFT when matrix values select different directions under JIT.
            shape = inverse.shape + (1,) * (field.ndim - inverse.ndim)
            inverse = inverse.reshape(shape)
            field = np.where(inverse, np.conj(field), field)
            field = np.fft.fftn(field, axes=axes, norm="ortho")
            field = np.where(inverse, np.conj(field), field)
        return np.fft.fftshift(field, axes=axes) if centered else field
