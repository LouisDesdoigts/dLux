"""User-facing optical elements and direct wavefront operations."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import jax.numpy as np
import equinox as eqx
from jax import Array

import dLux.utils as dlu

from ..grids import GridSpec
from ..parametric import Parametric, ParametricHolder, to_param
from ..fields import Wavefront

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
    transmission = 1.0 if optic.transmission is None else optic.transmission
    opd = 0.0 if optic.opd is None else optic.opd
    phase = 0.0 if optic.phase is None else optic.phase
    wavenumber = wavefront._to_phasor_shape(wavefront.wavenumber)
    transmission = wavefront._to_phasor_shape(transmission)
    opd = wavefront._to_phasor_shape(opd)
    phase = wavefront._to_phasor_shape(phase)
    return transmission * np.exp(1j * (wavenumber * opd + phase))


class BaseLayer(ParametricHolder):
    """Base class for callable transformations of dLux objects."""

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

    def apply(self, wavefront: Wavefront) -> Wavefront:
        """Apply a monochromatic layer over every leading wavefront axis."""
        if not isinstance(wavefront, Wavefront):
            return self(wavefront)
        axes = wavefront._mapped_axis
        if axes is None:
            return self(wavefront)

        def apply(phasor, wavelength, d, c):
            spec = wavefront.spec.set(d=d, c=c)
            wavefront_i = wavefront.set(phasor=phasor, wavelength=wavelength, spec=spec)
            return self.apply(wavefront_i)

        output = eqx.filter_vmap(apply, in_axes=axes)(
            wavefront.phasor, wavefront.wavelength, wavefront.spec.d, wavefront.spec.c
        )
        d = output.spec.d[0] if axes[2] is None else output.spec.d
        c = output.spec.c
        c = c[0] if c is not None and axes[3] is None else c
        return output.set(spec=output.spec.set(d=d, c=c))


class OpticalLayer(BaseOpticalLayer):
    """Public contract for layers that transform wavefronts."""

    @staticmethod
    def context(wavefront: Wavefront) -> dict[str, Any]:
        """Return the context used to resolve parametric attributes."""
        return {"wavefront": wavefront}


class TransmissiveLayer(OpticalLayer):
    """Apply a transmission, with optional output normalisation."""

    transmission: Array | Parametric | None = eqx.field(converter=to_param)
    normalise: bool

    def __init__(self, transmission=None, normalise=False):
        self.transmission = transmission
        self.normalise = bool(normalise)

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        self = self.resolve(**self.context(wavefront))
        if self.transmission is not None:
            transmission = wavefront._to_phasor_shape(self.transmission)
            wavefront = wavefront.set(phasor=wavefront.phasor * transmission)
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront


class AberratedLayer(OpticalLayer):
    """Apply optical-path and phase aberrations to a wavefront."""

    opd: Array | Parametric | None = eqx.field(converter=to_param)
    phase: Array | Parametric | None = eqx.field(converter=to_param)

    def __init__(self, opd=None, phase=None):
        self.opd = opd
        self.phase = phase

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        self = self.resolve(**self.context(wavefront))
        wavefront = wavefront.add_opd(self.opd)
        return wavefront.add_phase(self.phase)


class Optic(TransmissiveLayer, AberratedLayer):
    """A scalar physical optic evaluated at one plane."""

    transmission: Array | Parametric | None = eqx.field(converter=to_param)
    opd: Array | Parametric | None = eqx.field(converter=to_param)
    phase: Array | Parametric | None = eqx.field(converter=to_param)
    normalise: bool

    def __init__(self, transmission=None, opd=None, phase=None, normalise=False):
        TransmissiveLayer.__init__(self, transmission, normalise)
        AberratedLayer.__init__(self, opd, phase)

    def phasor(self, wavefront: Wavefront) -> Array:
        """Return the cumulative complex scalar field for this optical plane."""
        self = self.resolve(**self.context(wavefront))
        return _optic_phasor(self, wavefront)

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        phasor = wavefront.phasor * self.phasor(wavefront)
        wavefront = wavefront.set(phasor=phasor)
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront


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


class SoummerFPM(OpticalLayer):
    """Apply a focal-plane optical layer with the Soummer MFT algorithm.

    The focal-plane field is evaluated only on the compact ``focal_spec`` grid.
    The difference introduced by ``optic`` is inverse transformed and subtracted
    from the original pupil field. This supports scalar amplitude and phase optics,
    parametric optics, and Jones optics that promote the wavefront to a polarised
    representation.

    This implementation deliberately uses a conjugate-plane MFT pair. Defocused or
    otherwise non-conjugate propagation is not the Soummer algorithm represented by
    this layer.

    Parameters
    ----------
    optic : BaseOpticalLayer
        Optical layer applied on the sampled focal-plane grid. It should act as the
        identity outside the compact region represented by ``focal_spec``.
    focal_spec : GridSpec
        Explicit sampling of the focal-plane optic. Angular units are required when
        ``focal_length`` is omitted; physical units are required otherwise.
    focal_length : float, optional
        Physical focal length in metres. Omit for angular focal-plane coordinates.

    References
    ----------
    Soummer, R., Pueyo, L., Sivaramakrishnan, A., & Vanderbei, R. J. (2007),
    "Fast computation of Lyot-style coronagraph propagation", Optics Express,
    15(24), 15935--15951. https://doi.org/10.1364/OE.15.015935
    """

    optic: BaseOpticalLayer
    focal_spec: GridSpec
    focal_length: Array | None

    def __init__(self, optic, focal_spec, focal_length=None):
        if not isinstance(optic, BaseOpticalLayer):
            raise TypeError("optic must be a BaseOpticalLayer.")
        if not isinstance(focal_spec, GridSpec):
            raise TypeError("focal_spec must be a GridSpec.")
        self.optic = optic
        self.focal_spec = focal_spec.broadcast(2)
        self.focal_length = (
            None if focal_length is None else np.asarray(focal_length, dtype=float)
        )

    def validate(self, wavefront):
        """Validate the pupil and focal coordinate systems."""
        from .propagation import _validate_grid

        _validate_grid(wavefront.spec, "input", angular=False)
        angular = _validate_grid(self.focal_spec, "focal", ndim=2)
        if self.focal_length is None and not angular:
            raise ValueError(
                "SoummerFPM without a focal length requires angular focal units."
            )
        if self.focal_length is not None and angular:
            raise ValueError(
                "SoummerFPM with a focal length requires physical focal units."
            )

    def context(self, wavefront):
        """Return focal-plane context used to resolve the wrapped optic."""
        return {
            "wavefront": wavefront,
            "coordinates": wavefront.coordinates,
            "pixel_scale": wavefront.spec.d * wavefront.spec.scale,
            "spec": wavefront.spec,
        }

    def __call__(self, wavefront):
        self.validate(wavefront)
        focal_phasor = dlu.MFT(
            wavefront.phasor,
            wavefront.wavelength,
            wavefront.axes,
            self.focal_spec.axes,
            focal_length=self.focal_length,
        )
        focal = wavefront.set(phasor=focal_phasor, spec=self.focal_spec)
        optic = self.optic.resolve(**self.context(focal))
        modified = optic.apply(focal)
        difference = focal.phasor - modified.phasor
        pupil_difference = dlu.MFT(
            difference,
            wavefront.wavelength,
            self.focal_spec.axes,
            wavefront.axes,
            focal_length=self.focal_length,
            inverse=True,
        )
        phasor = wavefront.phasor - pupil_difference
        return modified.set(phasor=phasor, spec=wavefront.spec)
