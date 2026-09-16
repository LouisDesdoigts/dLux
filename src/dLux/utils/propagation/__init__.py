"""Numerical propagators and callable array operations.

Propagators accept (field, grid_in, wavelength), returning an array by default or
(array, GridSpec) with grid_out=True. Array operations and Sequential are ordinary
Zodiax Modules. Layers inherit the numerical classes and adapt Wavefront inputs.
"""

from ..helpers import reexport
from ..sequential import Sequential
from . import abcd
from .fft import FFT
from .mft import (
    MFT,
    mft_kernel,
    mft_kernels,
    mft_phase_parameter,
    mft_sampling_parameter,
)
from .operations import BaseOperation, Chirp, Resize
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
    "BaseOperation",
    "Resize",
    "Chirp",
    "MFT",
    "FFT",
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
