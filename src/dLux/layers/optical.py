"""User-facing optical elements and direct wavefront operations."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import jax.numpy as np
import equinox as eqx
from jax import Array

import dLux.utils as dlu

from ..parametric import Parametric, ParametricHolder
from ..fields import Wavefront

__all__ = [
    "BaseLayer",
    "BaseOpticalLayer",
    "OpticalLayer",
    "TransmissiveLayer",
    "AberratedLayer",
    "Optic",
    "Tilt",
]


def _optic_phasor(optic, wavefront):
    """Combine a resolved optic into one scalar complex field."""
    # Resolve absent optical terms to their identities
    transmission = 1.0 if optic.transmission is None else optic.transmission
    opd = 0.0 if optic.opd is None else optic.opd
    phase = 0.0 if optic.phase is None else optic.phase

    # Promote every term to the wavefront field shape
    wavenumber = wavefront._to_phasor_shape(wavefront.wavenumber)
    transmission = wavefront._to_phasor_shape(transmission)
    opd = wavefront._to_phasor_shape(opd)
    phase = wavefront._to_phasor_shape(phase)

    # Construct the combined scalar phasor
    return transmission * np.exp(1j * (wavenumber * opd + phase))


class BaseLayer(ParametricHolder):
    """Base class for callable transformations of dLux objects."""

    @abstractmethod
    def __call__(self, target: Any) -> Any:
        """Apply this layer to its target."""


class BaseOpticalLayer(BaseLayer):
    """Base class for layers that transform wavefronts."""

    @abstractmethod
    def apply_mono(self, wavefront: Wavefront) -> Wavefront:
        """Transform one monochromatic wavefront."""

    def apply(self, wavefront: Wavefront) -> Wavefront:
        """Apply this layer over every leading wavefront axis."""
        # Apply directly to non-wavefront targets and scalar wavefronts
        if not isinstance(wavefront, Wavefront):
            return self.apply_mono(wavefront)
        axes = wavefront._mapped_axis
        if axes is None:
            return self.apply_mono(wavefront)

        # Define application to one monochromatic field
        def apply_one(phasor, wavelength, d, c):
            spec = wavefront.spec.set(d=d, c=c)
            wavefront_i = wavefront.set(phasor=phasor, wavelength=wavelength, spec=spec)
            return self.apply(wavefront_i)

        # Vectorise the layer over leading wavefront dimensions
        apply_one = eqx.filter_vmap(apply_one, in_axes=axes)
        output = apply_one(
            wavefront.phasor, wavefront.wavelength, wavefront.spec.d, wavefront.spec.c
        )

        # Remove axes introduced for grid values that were not vectorised
        c = output.spec.c
        c = c[0] if c is not None and axes[3] is None else c
        d = output.spec.d[0] if axes[2] is None else output.spec.d

        # Restore the realised wavefront grid
        return output.set(spec=output.spec.set(d=d, c=c))

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        """Call :meth:`apply` using concise layer syntax."""
        return self.apply(wavefront)


class OpticalLayer(BaseOpticalLayer):
    """Public contract for layers that transform wavefronts."""

    @staticmethod
    def context(wavefront: Wavefront) -> dict[str, Any]:
        """Return the context used to resolve parametric attributes."""
        return {"wavefront": wavefront}


class TransmissiveLayer(OpticalLayer):
    """Apply a transmission with optional output normalisation.

    Parameters
    ----------
    transmission : Array, Parametric, or None
        Scalar or sampled amplitude transmission.
    normalise : bool
        Normalise the resulting wavefront to unit power.
    """

    transmission: Array | Parametric | None
    normalise: bool

    def __init__(self, transmission=None, normalise=False):
        self.transmission = dlu.to_value(transmission, optional=True, types=Parametric)
        self.normalise = bool(normalise)

    def apply_mono(self, wavefront: Wavefront) -> Wavefront:
        """Apply the resolved transmission and optional normalisation."""
        self = self.resolve(**self.context(wavefront))
        if self.transmission is not None:
            transmission = wavefront._to_phasor_shape(self.transmission)
            wavefront = wavefront.set(phasor=wavefront.phasor * transmission)
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront


class AberratedLayer(OpticalLayer):
    """Apply optical-path and phase aberrations to a wavefront.

    Parameters
    ----------
    opd : Array, Parametric, or None
        Optical path difference in meters.
    phase : Array, Parametric, or None
        Additional phase in radians.
    """

    opd: Array | Parametric | None
    phase: Array | Parametric | None

    def __init__(self, opd=None, phase=None):
        self.opd = dlu.to_value(opd, optional=True, types=Parametric)
        self.phase = dlu.to_value(phase, optional=True, types=Parametric)

    def apply_mono(self, wavefront: Wavefront) -> Wavefront:
        """Apply the resolved optical-path and phase aberrations."""
        self = self.resolve(**self.context(wavefront))
        wavefront = wavefront.add_opd(self.opd)
        return wavefront.add_phase(self.phase)


class Optic(TransmissiveLayer, AberratedLayer):
    """Represent a scalar physical optic evaluated at one plane.

    Parameters
    ----------
    transmission : Array, Parametric, or None
        Scalar or sampled amplitude transmission.
    opd : Array, Parametric, or None
        Optical path difference in meters.
    phase : Array, Parametric, or None
        Additional phase in radians.
    normalise : bool
        Normalise the resulting wavefront to unit power.
    """

    transmission: Array | Parametric | None
    opd: Array | Parametric | None
    phase: Array | Parametric | None
    normalise: bool

    def __init__(self, transmission=None, opd=None, phase=None, normalise=False):
        TransmissiveLayer.__init__(self, transmission, normalise)
        AberratedLayer.__init__(self, opd, phase)

    def phasor(self, wavefront: Wavefront) -> Array:
        """Return the cumulative complex scalar field for this optical plane."""
        self = self.resolve(**self.context(wavefront))
        return _optic_phasor(self, wavefront)

    def apply_mono(self, wavefront: Wavefront) -> Wavefront:
        """Apply the cumulative complex optic phasor to a wavefront."""
        phasor = wavefront.phasor * self.phasor(wavefront)
        wavefront = wavefront.set(phasor=phasor)
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront


class Tilt(OpticalLayer):
    """Tilt a wavefront by two angular coordinates.

    Parameters
    ----------
    angles : ArrayLike
        Two angular offsets in ``(x, y)`` order.
    unit : str
        Angular unit associated with ``angles``.
    """

    angles: Array
    unit: str

    def __init__(self, angles, unit="rad"):
        self.angles = dlu.to_value(angles)
        if self.angles.shape != (2,):
            raise ValueError("angles must have shape (2,).")
        self.unit = str(unit)

    def apply_mono(self, wavefront: Wavefront) -> Wavefront:
        """Apply the configured angular tilt to a wavefront."""
        return wavefront.tilt(self.angles, self.unit)
