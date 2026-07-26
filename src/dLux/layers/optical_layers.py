"""User-facing optical elements and direct wavefront operations."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import jax.numpy as np
from jax import Array

import dLux.utils as dlu
from ..parametric import Interpolation, Parametric, ParametricHolder
from ..fields import Wavefront

__all__ = [
    "BaseLayer",
    "BaseOpticalLayer",
    "OpticalLayer",
    "TransmissiveLayer",
    "AberratedLayer",
    "Optic",
    "Filter",
    "Tilt",
]


class BaseLayer(ParametricHolder):
    """Base class for callable transformations of dLux objects."""

    def __init_subclass__(cls, **kwargs):
        """Inherit callable documentation for concrete layer implementations."""
        super().__init_subclass__(**kwargs)
        dlu.helpers.inherit_docstrings(cls, ["__call__"])

    @abstractmethod
    def __call__(self, target: Any) -> Any:  # pragma: no cover
        """Apply this layer to its target."""

    def apply(self, target: Any) -> Any:
        """Backwards-compatible alias for calling the layer."""
        return self(target)


class BaseOpticalLayer(BaseLayer):
    """Base class for layers that transform wavefronts."""

    @abstractmethod
    def __call__(self, wavefront: Wavefront) -> Wavefront:  # pragma: no cover
        """Transform a wavefront."""


class OpticalLayer(BaseOpticalLayer):
    """Public contract for layers that transform wavefronts."""


class TransmissiveLayer(OpticalLayer):
    """Apply a transmission, with optional output normalisation."""

    transmission: Array | Parametric | None
    normalise: bool

    def __init__(self, transmission=None, normalise=False):
        self.transmission = self.as_parametric(transmission)
        self.normalise = bool(normalise)

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        self = self.resolve(wavefront=wavefront)
        if self.transmission is not None:
            transmission = wavefront._to_phasor_shape(self.transmission)
            wavefront = wavefront.set(phasor=wavefront.phasor * transmission)
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront


class AberratedLayer(OpticalLayer):
    """Apply optical-path and phase aberrations to a wavefront."""

    opd: Array | Parametric | None
    phase: Array | Parametric | None

    def __init__(self, opd=None, phase=None):
        self.opd = self.as_parametric(opd)
        self.phase = self.as_parametric(phase)

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        self = self.resolve(wavefront=wavefront)
        wavefront = wavefront.add_opd(self.opd)
        return wavefront.add_phase(self.phase)


class Optic(TransmissiveLayer, AberratedLayer):
    """A scalar physical optic evaluated at one plane."""

    transmission: Array | Parametric | None
    opd: Array | Parametric | None
    phase: Array | Parametric | None
    normalise: bool

    def __init__(
        self,
        transmission=None,
        opd=None,
        phase=None,
        normalise=False,
    ):
        TransmissiveLayer.__init__(self, transmission, normalise)
        AberratedLayer.__init__(self, opd, phase)

    def context(self, wavefront: Wavefront) -> dict[str, Any]:
        """Return the parameter context shared by this optic's properties."""
        return {"wavefront": wavefront}

    def phasor(self, wavefront: Wavefront) -> Array:
        """Return the cumulative complex scalar field for this optical plane."""
        self = self.resolve(**self.context(wavefront))
        transmission = self.transmission
        opd = self.opd
        phase = self.phase

        transmission = 1.0 if transmission is None else transmission
        opd = 0.0 if opd is None else opd
        phase = 0.0 if phase is None else phase

        wavenumber = wavefront._to_phasor_shape(wavefront.wavenumber)
        opd = wavefront._to_phasor_shape(opd)
        phase = wavefront._to_phasor_shape(phase)
        transmission = wavefront._to_phasor_shape(transmission)
        return transmission * np.exp(1j * (wavenumber * opd + phase))

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        phasor = wavefront.phasor * self.phasor(wavefront)
        wavefront = wavefront.set(phasor=phasor)
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront


class Filter(OpticalLayer):
    """Apply a parametric spectral throughput curve to a wavefront.

    The throughput is an intensity response, so its square root is applied
    to the complex field amplitude. Array inputs are promoted to an
    :class:`Interpolation` evaluated at the actual wavefront wavelengths.

    Parameters
    ----------
    throughput : Array or Parametric
        Parametric intensity throughput, or sampled values to interpolate.
    wavelengths : Array or None
        Wavelength knots for an array throughput. Must be omitted when
        throughput is already parametric.
    unit : str
        Unit of ``wavelengths`` and ``bin_width``. Values are converted
        to metres internally.
    bin_width : Array or None
        Width of each modelled wavelength bin. When supplied, the filter
        throughput is integrated over each bin and divided by its width.
        When omitted, throughput is evaluated at the central wavelength.
    """

    throughput: Parametric
    bin_width: Array | None

    def __init__(
        self,
        throughput,
        wavelengths=None,
        unit="m",
        method="linear",
        bin_width=None,
    ):
        scale = dlu.unit_factor(unit)
        self.bin_width = (
            None if bin_width is None else np.asarray(bin_width, dtype=float) * scale
        )
        if self.bin_width is not None and not bool(np.all(self.bin_width > 0)):
            raise ValueError("bin_width must contain positive values.")
        if isinstance(throughput, Parametric):
            if wavelengths is not None:
                raise ValueError(
                    "wavelengths must be omitted for parametric throughput."
                )
            self.throughput = throughput
            return
        if wavelengths is None:
            raise ValueError("wavelengths are required for array throughput.")
        if self.bin_width is not None and method != "linear":
            raise ValueError(
                "Integrated array throughput currently requires method='linear'."
            )
        wavelengths = np.asarray(wavelengths, dtype=float)
        wavelengths = wavelengths * scale
        self.throughput = Interpolation(
            wavelengths, throughput, method=method, extrapolate=0.0
        )

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        wavelength = wavefront.wavelength
        context = {
            "variables": wavelength,
            "wavelengths": wavelength,
            "wavefront": wavefront,
        }
        if self.bin_width is None:
            throughput = self.resolve_parametric(self.throughput, **context)
        else:
            lower = wavelength - self.bin_width / 2
            upper = wavelength + self.bin_width / 2
            throughput = self.throughput.integrate(lower, upper, **context)
            throughput = throughput / self.bin_width
        amplitude = np.sqrt(throughput)
        amplitude = wavefront._to_phasor_shape(amplitude)
        return wavefront.set(phasor=wavefront.phasor * amplitude)


class Tilt(OpticalLayer):
    """Tilt a wavefront by two angular coordinates."""

    angles: Array
    unit: str

    def __init__(self, angles, unit="rad"):
        self.angles = np.asarray(angles, dtype=float)
        if self.angles.shape != (2,):
            raise ValueError("angles must have shape (2,).")
        self.unit = str(unit)

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        return wavefront.tilt(self.angles, self.unit)
