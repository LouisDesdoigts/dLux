"""Coordinate-aware optical layers."""

from __future__ import annotations

from typing import Any

import jax.numpy as np
from jax import Array

from ..grids import CoordTransform, GridSpec
from ..parametric import Parametric
from ..fields import Wavefront
from .optical_layers import AberratedLayer, BaseOpticalLayer, Optic, TransmissiveLayer

__all__ = [
    "BaseDynamicLayer",
    "DynamicTransmissiveLayer",
    "DynamicAberratedLayer",
    "DynamicOptic",
]


class BaseDynamicLayer(BaseOpticalLayer):
    """Base class for optical layers evaluated in a coordinate context."""

    coordinates: Array | GridSpec | None
    transformation: CoordTransform | None

    def __init__(self, coordinates=None, transformation=None):
        if coordinates is not None and not isinstance(coordinates, GridSpec):
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
    def _from_spec(spec: GridSpec) -> Array:
        return spec.coordinates

    def context(self, wavefront: Wavefront) -> dict[str, Any]:
        """Return the coordinate context used to resolve parametric leaves."""
        if self.coordinates is None:
            coordinates = wavefront.coordinates
            pixel_scale = wavefront.pixel_scale
        elif isinstance(self.coordinates, GridSpec):
            coordinates = self._from_spec(self.coordinates)
            pixel_scale = self.coordinates.d
        else:
            coordinates = self.coordinates
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

    coordinates: Array | GridSpec | None
    transformation: CoordTransform | None
    transmission: Array | Parametric | None
    normalise: bool

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
        self = self.resolve(**self.context(wavefront))
        if self.transmission is not None:
            transmission = wavefront._to_phasor_shape(self.transmission)
            wavefront = wavefront.set(phasor=wavefront.phasor * transmission)
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront


class DynamicAberratedLayer(BaseDynamicLayer, AberratedLayer):
    """Apply static or coordinate-dependent OPD and phase aberrations."""

    coordinates: Array | GridSpec | None
    transformation: CoordTransform | None
    opd: Array | Parametric | None
    phase: Array | Parametric | None

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
        self = self.resolve(**self.context(wavefront))
        return wavefront.add_opd(self.opd).add_phase(self.phase)


class DynamicOptic(BaseDynamicLayer, Optic):
    """A scalar optic with independently static or coordinate-dependent leaves."""

    coordinates: Array | GridSpec | None
    transformation: CoordTransform | None
    transmission: Array | Parametric | None
    opd: Array | Parametric | None
    phase: Array | Parametric | None
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

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        self = self.resolve(**self.context(wavefront))
        if self.transmission is not None:
            transmission = wavefront._to_phasor_shape(self.transmission)
            wavefront = wavefront.set(phasor=wavefront.phasor * transmission)
        wavefront = wavefront.add_opd(self.opd).add_phase(self.phase)
        return wavefront.normalise() if self.normalise else wavefront
