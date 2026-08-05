"""Coordinate-aware optical layers."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as np
from jax import Array

from ..grids import CoordTransform, GridSpec
from ..parametric import Parametric, to_param
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
    transmission: Array | Parametric | None = eqx.field(converter=to_param)
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
    opd: Array | Parametric | None = eqx.field(converter=to_param)
    phase: Array | Parametric | None = eqx.field(converter=to_param)

    def __init__(self, opd=None, phase=None, coordinates=None, transformation=None):
        BaseDynamicLayer.__init__(self, coordinates, transformation)
        AberratedLayer.__init__(self, opd, phase)


class DynamicOptic(BaseDynamicLayer, Optic):
    """A scalar optic with independently static or coordinate-dependent leaves."""

    coordinates: Array | GridSpec | None
    transformation: CoordTransform | None
    transmission: Array | Parametric | None = eqx.field(converter=to_param)
    opd: Array | Parametric | None = eqx.field(converter=to_param)
    phase: Array | Parametric | None = eqx.field(converter=to_param)
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
