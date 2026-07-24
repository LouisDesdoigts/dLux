"""Coordinate-aware optical layers."""

from __future__ import annotations

from typing import Any

import jax.numpy as np
from jax import Array

from ..coordinates import CoordSpec, CoordTransform
from ..parametric import BaseParametric
from ..wavefronts import Wavefront
from .optical_layers import AberratedLayer, BaseOpticalLayer, Optic, TransmissiveLayer

__all__ = [
    "BaseDynamicLayer",
    "DynamicTransmissiveLayer",
    "DynamicAberratedLayer",
    "DynamicOptic",
]


class BaseDynamicLayer(BaseOpticalLayer):
    """Base class for optical layers evaluated in a coordinate context."""

    coordinates: Array | CoordSpec | None
    transformation: CoordTransform | None

    def __init__(self, coordinates=None, transformation=None):
        if coordinates is not None and not isinstance(coordinates, CoordSpec):
            coordinates = np.asarray(coordinates, dtype=float)
            if coordinates.shape[-3] != 2:
                raise ValueError("coordinates must have shape (..., 2, n, n).")
        if transformation is not None and not isinstance(
            transformation, CoordTransform
        ):
            raise TypeError("transformation must be a CoordTransform or None.")
        self.coordinates = coordinates
        self.transformation = transformation

    @staticmethod
    def _from_spec(spec: CoordSpec) -> Array:
        return spec.coordinates

    def context(self, wavefront: Wavefront) -> dict[str, Any]:
        """Return the coordinate context used to resolve parametric leaves."""
        transform_coordinates = (
            None if self.transformation is None else self.transformation.coordinates
        )
        coordinate_source = (
            self.coordinates if self.coordinates is not None else transform_coordinates
        )

        if coordinate_source is None:
            coordinates = wavefront.coordinates
            pixel_scale = wavefront.pixel_scale
        elif isinstance(coordinate_source, CoordSpec):
            coordinates = self._from_spec(coordinate_source)
            pixel_scale = coordinate_source.d
        else:
            coordinates = coordinate_source
            pixel_scale = wavefront.pixel_scale

        if self.transformation is not None:
            coordinates = self.transformation(coordinates)
        return {
            "wavefront": wavefront,
            "coordinates": coordinates,
            "pixel_scale": pixel_scale,
        }


class DynamicTransmissiveLayer(BaseDynamicLayer, TransmissiveLayer):
    """Apply a static or coordinate-dependent transmission."""

    def __init__(
        self,
        transmission=None,
        coordinates=None,
        transformation=None,
        normalise=False,
    ):
        BaseDynamicLayer.__init__(self, coordinates, transformation)
        TransmissiveLayer.__init__(self, transmission, normalise)

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        transmission = self.resolve(self.transmission, **self.context(wavefront))
        if transmission is not None:
            transmission = wavefront._to_phasor_shape(transmission)
            wavefront = wavefront.set(phasor=wavefront.phasor * transmission)
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront


class DynamicAberratedLayer(BaseDynamicLayer, AberratedLayer):
    """Apply static or coordinate-dependent OPD and phase aberrations."""

    def __init__(
        self,
        opd=None,
        phase=None,
        coordinates=None,
        transformation=None,
    ):
        BaseDynamicLayer.__init__(self, coordinates, transformation)
        AberratedLayer.__init__(self, opd, phase)

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        context = self.context(wavefront)
        opd = self.resolve(self.opd, **context)
        phase = self.resolve(self.phase, **context)
        return wavefront.add_opd(opd).add_phase(phase)


class DynamicOptic(BaseDynamicLayer, Optic):
    """A scalar optic with independently static or coordinate-dependent leaves."""

    transmission: Array | BaseParametric | None
    opd: Array | BaseParametric | None
    phase: Array | BaseParametric | None
    normalise: bool

    def __init__(
        self,
        transmission=None,
        opd=None,
        phase=None,
        coordinates=None,
        transformation=None,
        normalise=False,
    ):
        BaseDynamicLayer.__init__(self, coordinates, transformation)
        Optic.__init__(self, transmission, opd, phase, normalise)

    def params(self, wavefront: Wavefront) -> dict[str, Any]:
        """Resolve every optic leaf once in one shared coordinate context."""
        context = self.context(wavefront)
        return {
            "transmission": self.resolve(self.transmission, **context),
            "opd": self.resolve(self.opd, **context),
            "phase": self.resolve(self.phase, **context),
        }
