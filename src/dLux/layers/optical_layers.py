"""User-facing optical elements and direct wavefront operations."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import jax.numpy as np
import zodiax as zdx
from jax import Array

import dLux.utils as dlu
from ..coordinates import BaseCoordTransform
from ..parametric import BaseParametric, Shape
from ..wavefronts import Wavefront

__all__ = [
    "BaseLayer",
    "BaseOpticalLayer",
    "OpticalLayer",
    "TransmissiveLayer",
    "AberratedLayer",
    "Optic",
    "DynamicOptic",
    "Tilt",
]


class BaseLayer(zdx.Base):
    """Base class for callable transformations of dLux objects."""

    @abstractmethod
    def __call__(self, target: Any) -> Any:  # pragma: no cover
        """Apply this layer to its target."""

    def apply(self, target: Any) -> Any:
        """Backwards-compatible alias for calling the layer."""
        return self(target)

    @staticmethod
    def resolve(value: Any, **kwargs: Any) -> Any:
        """Evaluate a parametric value, or return an ordinary value unchanged."""
        if isinstance(value, BaseParametric):
            return value.evaluate(**kwargs)
        return value

    @staticmethod
    def as_parametric(value: Any, dtype: Any = float) -> Any:
        """Preserve parametric values and convert ordinary values to arrays."""
        if value is None or isinstance(value, BaseParametric):
            return value
        return np.asarray(value, dtype=dtype)

    def __init_subclass__(cls, **kwargs):
        """Inherit callable documentation for concrete layer implementations."""
        super().__init_subclass__(**kwargs)
        dlu.helpers.inherit_docstrings(cls, ["__call__"])


class BaseOpticalLayer(BaseLayer):
    """Base class for layers that transform wavefronts."""

    @abstractmethod
    def __call__(self, wavefront: Wavefront) -> Wavefront:  # pragma: no cover
        """Transform a wavefront."""


class OpticalLayer(BaseOpticalLayer):
    """Public contract for layers that transform wavefronts."""


class TransmissiveLayer(OpticalLayer):
    """Apply a transmission, with optional output normalisation."""

    transmission: Array | BaseParametric | None
    normalise: bool

    def __init__(self, transmission=None, normalise=False):
        self.transmission = self.as_parametric(transmission)
        self.normalise = bool(normalise)

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        transmission = self.resolve(self.transmission, wavefront=wavefront)
        if transmission is not None:
            transmission = wavefront._to_phasor_shape(transmission)
            wavefront = wavefront.set(phasor=wavefront.phasor * transmission)
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront


class AberratedLayer(OpticalLayer):
    """Apply optical-path and phase aberrations to a wavefront."""

    opd: Array | BaseParametric | None
    phase: Array | BaseParametric | None

    def __init__(self, opd=None, phase=None):
        self.opd = self.as_parametric(opd)
        self.phase = self.as_parametric(phase)

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        opd = self.resolve(self.opd, wavefront=wavefront)
        phase = self.resolve(self.phase, wavefront=wavefront)
        wavefront = wavefront.add_opd(opd)
        return wavefront.add_phase(phase)


class Optic(TransmissiveLayer, AberratedLayer):
    """A scalar physical optic evaluated at one plane."""

    transmission: Array | BaseParametric | None
    opd: Array | BaseParametric | None
    phase: Array | BaseParametric | None
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

    def params(self, wavefront: Wavefront) -> dict[str, Any]:
        """Resolve all physical parameters exactly once for one application."""
        context = self.context(wavefront)
        return {
            "transmission": self.resolve(self.transmission, **context),
            "opd": self.resolve(self.opd, **context),
            "phase": self.resolve(self.phase, **context),
        }

    def phasor(self, wavefront: Wavefront, params: dict = None) -> Array:
        """Return the cumulative complex scalar field for this optical plane."""
        params = self.params(wavefront) if params is None else params
        transmission = params["transmission"]
        opd = params["opd"]
        phase = params["phase"]

        transmission = 1.0 if transmission is None else transmission
        opd = 0.0 if opd is None else opd
        phase = 0.0 if phase is None else phase

        wavenumber = wavefront._to_phasor_shape(wavefront.wavenumber)
        opd = wavefront._to_phasor_shape(opd)
        phase = wavefront._to_phasor_shape(phase)
        transmission = wavefront._to_phasor_shape(transmission)
        return transmission * np.exp(1j * (wavenumber * opd + phase))

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        params = self.params(wavefront)
        phasor = wavefront.phasor * self.phasor(wavefront, params)
        wavefront = wavefront.set(phasor=phasor)
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront


class DynamicOptic(Optic):
    """An optic evaluated in one shared transformed coordinate frame."""

    aperture: Shape
    transformation: BaseCoordTransform | None

    def __init__(
        self,
        aperture,
        transformation=None,
        opd=None,
        phase=None,
        normalise=False,
    ):
        if not isinstance(aperture, Shape):
            raise TypeError("aperture must be a Shape.")
        if transformation is not None and not isinstance(
            transformation, BaseCoordTransform
        ):
            raise TypeError("transformation must be a BaseCoordTransform or None.")
        self.aperture = aperture
        self.transformation = transformation
        super().__init__(
            opd=opd,
            phase=phase,
            normalise=normalise,
        )

    def context(self, wavefront: Wavefront) -> dict[str, Any]:
        """Return a shared coordinate context for every dynamic property."""
        coordinates = wavefront.coordinates()
        if self.transformation is not None:
            coordinates = self.transformation(coordinates)
        extent = self.aperture.extent
        return {
            "wavefront": wavefront,
            "coordinates": coordinates,
            "pixel_scale": wavefront.pixel_scale,
            "diameter": wavefront.diameter if extent is None else 2 * extent,
            "aperture": self.aperture,
        }

    def params(self, wavefront: Wavefront) -> dict[str, Any]:
        """Resolve the aperture and remaining properties in one shared context."""
        context = self.context(wavefront)
        return {
            "transmission": self.aperture.evaluate(**context),
            "opd": self.resolve(self.opd, **context),
            "phase": self.resolve(self.phase, **context),
        }


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
