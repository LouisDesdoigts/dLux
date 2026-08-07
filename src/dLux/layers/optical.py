"""User-facing optical elements and direct wavefront operations."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import jax.numpy as np
import equinox as eqx
from jax import Array

import dLux.utils as dlu

from ..parametric import Parametric, ParametricHolder
from ..fields import Wavefront

if TYPE_CHECKING:
    from .propagation import Fraunhofer

__all__ = [
    "BaseLayer",
    "BaseOpticalLayer",
    "OpticalLayer",
    "TransmissiveLayer",
    "AberratedLayer",
    "Optic",
    "Tilt",
    "SoummerFPM",
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

    def apply(self, target: Any) -> Any:
        """Backwards-compatible alias for calling the layer."""
        return self(target)


class BaseOpticalLayer(BaseLayer):
    """Base class for layers that transform wavefronts."""

    @abstractmethod
    def __call__(self, wavefront: Wavefront) -> Wavefront:
        """Transform a wavefront."""

    def apply(self, wavefront: Wavefront) -> Wavefront:
        """Apply a monochromatic layer over every leading wavefront axis."""
        # Apply directly to non-wavefront targets and scalar wavefronts
        if not isinstance(wavefront, Wavefront):
            return self(wavefront)
        axes = wavefront._mapped_axis
        if axes is None:
            return self(wavefront)

        # Define application to one monochromatic field
        def apply(phasor, wavelength, d, c):
            spec = wavefront.spec.set(d=d, c=c)
            wavefront_i = wavefront.set(phasor=phasor, wavelength=wavelength, spec=spec)
            return self.apply(wavefront_i)

        # Vectorise the layer over leading wavefront dimensions
        apply = eqx.filter_vmap(apply, in_axes=axes)
        output = apply(
            wavefront.phasor, wavefront.wavelength, wavefront.spec.d, wavefront.spec.c
        )

        # Remove axes introduced for grid values that were not vectorised
        c = output.spec.c
        c = c[0] if c is not None and axes[3] is None else c
        d = output.spec.d[0] if axes[2] is None else output.spec.d

        # Restore the realised wavefront grid
        return output.set(spec=output.spec.set(d=d, c=c))


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

    def __call__(self, wavefront: Wavefront) -> Wavefront:
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

    def __call__(self, wavefront: Wavefront) -> Wavefront:
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

    def __call__(self, wavefront: Wavefront) -> Wavefront:
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

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        """Apply the configured angular tilt to a wavefront."""
        return wavefront.tilt(self.angles, self.unit)


class SoummerFPM(OpticalLayer):
    """Apply a focal-plane optical layer with the Soummer MFT algorithm.

    The focal-plane field is evaluated only on the propagator's compact output grid.
    The difference introduced by ``optic`` is inverse transformed and subtracted
    from the original pupil field. This supports scalar amplitude and phase optics,
    parametric optics, and Jones optics that promote the wavefront to a polarised
    representation.

    This implementation requires a forward MFT ``Fraunhofer`` propagator. The same
    propagator is configured for inverse propagation when returning the modified
    field to the input pupil grid.

    Parameters
    ----------
    optic : BaseOpticalLayer
        Optical layer applied on the sampled focal-plane grid. It should act as the
        identity outside the compact region represented by ``focal_spec``.
    propagator : Fraunhofer
        Forward ``Fraunhofer`` propagator configured with ``method="mft"``.

    References
    ----------
    Soummer, R., Pueyo, L., Sivaramakrishnan, A., & Vanderbei, R. J. (2007),
    "Fast computation of Lyot-style coronagraph propagation", Optics Express,
    15(24), 15935--15951. https://doi.org/10.1364/OE.15.015935
    """

    optic: BaseOpticalLayer
    propagator: Fraunhofer

    def __init__(self, optic, propagator):
        from .propagation import Fraunhofer

        if not isinstance(optic, BaseOpticalLayer):
            raise TypeError("optic must be a BaseOpticalLayer.")
        if not isinstance(propagator, Fraunhofer):
            raise TypeError("propagator must be a Fraunhofer layer.")
        if propagator.method != "mft":
            raise ValueError("SoummerFPM requires an MFT Fraunhofer propagator.")
        if propagator.inverse:
            raise ValueError("SoummerFPM requires a forward propagator.")
        self.optic = optic
        self.propagator = propagator

    def context(self, wavefront):
        """Return focal-plane context used to resolve the wrapped optic."""
        return {
            "wavefront": wavefront,
            "coordinates": wavefront.coordinates,
            "pixel_scale": wavefront.spec.d * wavefront.spec.scale,
            "spec": wavefront.spec,
        }

    def __call__(self, wavefront):
        """Apply the compact focal-plane optic and return to the input pupil."""
        # Propagate to and apply the compact focal-plane optic
        focal = self.propagator(wavefront)
        optic = self.optic.resolve(**self.context(focal))
        difference = focal - optic.apply(focal)

        # Inverse propagate only the field introduced by the optic
        inverse = self.propagator.set(spec=wavefront.spec, inverse=True)
        pupil_difference = inverse(difference)

        # Subtract the focal-plane modification from the original pupil
        return wavefront - pupil_difference
