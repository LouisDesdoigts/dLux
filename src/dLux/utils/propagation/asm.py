"""Prepared angular-spectrum propagation as a Fourier-space series."""

from collections import OrderedDict

from ..grids.array_ops import as_size
from ..sequential import Series
from .fft import FFT
from .operations import Chirp

__all__ = ["ASM"]


class ASM(Series):
    """FFT, spectral transfer, inverse FFT, with optional padding and cropping.

    ``transfer`` is a pair of separable complex factors in raw FFT frequency
    order. Its lengths determine the working counts. FreeSpace prepares these
    factors from wavelength, distance and a physical input grid using the
    paraxial approximation; this class applies the prepared array operation.
    """

    # Inherited fields
    layers: OrderedDict

    def __init__(self, transfer, n_in=None, n_out=None, *, alias=None):
        transfer = Chirp(transfer)
        n_fft = transfer.n
        n_in = n_fft if n_in is None else as_size(n_in, 2, "n_in")
        n_out = n_in if n_out is None else as_size(n_out, 2, "n_out")
        forward = FFT.prepare(n_in, n_fft, centered=False)
        backward = FFT.prepare(n_fft, n_out=n_out, inverse=True, centered=False)
        super().__init__(
            (("forward", forward), ("transfer", transfer), ("backward", backward)),
            alias=alias,
        )
