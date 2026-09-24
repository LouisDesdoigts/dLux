"""Numerical propagators and callable array operations.

Propagators accept (field, grid_in, wavelength), returning an array by default or
(array, Grid) with return_grid=True. Preparation returns one array operation or an
unbound Series. Layers adapt the numerical classes to Wavefront inputs.
"""

from .. import reexport
from ..sequential import Sequential, Series
from . import abcd
from .fft import FFT, fourier_shift
from .asm import ASM
from .mft import (
    MFT,
    mft_kernel,
    mft_kernels,
    mft_phase_parameter,
    mft_sampling_parameter,
)
from .operations import Chirp, Resize
from .propagators import (
    BasePropagator,
    LCT,
    Propagator,
    FFTPropagator,
    FreeSpace,
    SpectralPropagator,
    ThinPower,
)

__all__ = [
    "abcd",
    "Series",
    "Resize",
    "Chirp",
    "MFT",
    "FFT",
    "ASM",
    "fourier_shift",
    "Sequential",
    "mft_kernel",
    "mft_kernels",
    "mft_phase_parameter",
    "mft_sampling_parameter",
    "BasePropagator",
    "LCT",
    "Propagator",
    "FFTPropagator",
    "SpectralPropagator",
    "FreeSpace",
    "ThinPower",
]

__all__ += reexport((abcd,), globals())
