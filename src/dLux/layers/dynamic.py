"""Coordinate-aware optical layers."""

from __future__ import annotations

from typing import Any

from jax import Array

import dLux.utils as dlu

from ..grids import CoordTransform, GridSpec
from ..parametric import Parametric
from ..fields import Wavefront
from .optical import AberratedLayer, BaseOpticalLayer, Optic, TransmissiveLayer

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
        if transformation is not None and not isinstance(
            transformation, CoordTransform
        ):
            raise TypeError("transformation must be a CoordTransform or None.")

        coordinates = dlu.to_value(coordinates, optional=True, types=GridSpec)
        if (
            coordinates is not None
            and not isinstance(coordinates, GridSpec)
            and (coordinates.ndim < 3 or coordinates.shape[-3] != 2)
        ):
            raise ValueError("coordinates must have shape (..., 2, ny, nx).")

        self.coordinates = coordinates
        self.transformation = transformation

    def context(self, wavefront: Wavefront) -> dict[str, Any]:
        """Return the coordinate context used to resolve parametric leaves."""
        if self.coordinates is None:
            coords, d = wavefront.coordinates, wavefront.pixel_scale
        elif isinstance(self.coordinates, GridSpec):
            coords, d = self.coordinates.coordinates, self.coordinates.d
        else:
            coords, d = self.coordinates, wavefront.pixel_scale

        if self.transformation is not None:
            coords = self.transformation(coords)
        return {"wavefront": wavefront, "coordinates": coords, "pixel_scale": d}


class DynamicTransmissiveLayer(BaseDynamicLayer, TransmissiveLayer):
    """Apply a static or coordinate-dependent transmission."""

    coordinates: Array | GridSpec | None
    transformation: CoordTransform | None
    transmission: Array | Parametric | None
    normalise: bool

    def __init__(
        self, transmission=None, coordinates=None, transformation=None, normalise=False
    ):
        BaseDynamicLayer.__init__(self, coordinates, transformation)
        TransmissiveLayer.__init__(self, transmission, normalise)


class DynamicAberratedLayer(BaseDynamicLayer, AberratedLayer):
    """Apply static or coordinate-dependent OPD and phase aberrations."""

    coordinates: Array | GridSpec | None
    transformation: CoordTransform | None
    opd: Array | Parametric | None
    phase: Array | Parametric | None

    def __init__(self, opd=None, phase=None, coordinates=None, transformation=None):
        BaseDynamicLayer.__init__(self, coordinates, transformation)
        AberratedLayer.__init__(self, opd, phase)


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
