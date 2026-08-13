"""Coordinate-aware optical layers."""

from __future__ import annotations

from typing import Any

from jax import Array

import dLux.utils as dlu

from ..grids import BaseCoordTransform, GridSpec
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
    transformation: BaseCoordTransform | None

    def __init__(self, coordinates=None, transformation=None):
        """Initialise a coordinate context for dynamic optical evaluation.

        Parameters
        ----------
        coordinates : Array, GridSpec, or None
            Explicit ``(..., 2, ny, nx)`` coordinates, a generating grid, or ``None``
            to use the incident wavefront coordinates.
        transformation : BaseCoordTransform or None
            Optional transformation applied before resolving parametric leaves.
        """
        if transformation is not None and not isinstance(
            transformation, BaseCoordTransform
        ):
            raise TypeError("transformation must be a BaseCoordTransform or None.")

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
        """Return the coordinate context used to resolve parametric leaves.

        The mapping contains ``wavefront``, SI-valued ``coordinates`` with shape
        ``(..., 2, ny, nx)``, and per-axis ``pixel_scale``. Explicit coordinates or a
        generating grid replace the wavefront coordinates before the optional
        transformation is applied.
        """
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
    transformation: BaseCoordTransform | None
    transmission: Array | Parametric | None
    normalise: bool

    def __init__(
        self, transmission=None, coordinates=None, transformation=None, normalise=False
    ):
        """Initialise a coordinate-dependent transmissive layer.

        Parameters
        ----------
        transmission : Array, Parametric, or None
            Static transmission or parametric resolved in the coordinate context.
        coordinates : Array, GridSpec, or None
            Explicit coordinate source, or incident coordinates when omitted.
        transformation : BaseCoordTransform or None
            Optional map into the transmission's local frame.
        normalise : bool
            Renormalise wavefront power after application.
        """
        BaseDynamicLayer.__init__(self, coordinates, transformation)
        TransmissiveLayer.__init__(self, transmission, normalise)


class DynamicAberratedLayer(BaseDynamicLayer, AberratedLayer):
    """Apply static or coordinate-dependent OPD and phase aberrations."""

    coordinates: Array | GridSpec | None
    transformation: BaseCoordTransform | None
    opd: Array | Parametric | None
    phase: Array | Parametric | None

    def __init__(self, opd=None, phase=None, coordinates=None, transformation=None):
        """Initialise coordinate-dependent aberrations.

        Parameters
        ----------
        opd, phase : Array, Parametric, or None
            Static or parametric OPD in metres and phase in radians.
        coordinates : Array, GridSpec, or None
            Explicit coordinate source, or incident coordinates when omitted.
        transformation : BaseCoordTransform or None
            Optional map into the aberration's local frame.
        """
        BaseDynamicLayer.__init__(self, coordinates, transformation)
        AberratedLayer.__init__(self, opd, phase)


class DynamicOptic(BaseDynamicLayer, Optic):
    """A scalar optic with independently static or coordinate-dependent leaves.

    Examples
    --------
    Evaluate a Zernike OPD on polynomially distorted wavefront coordinates:

    ```python
    import dLux as dl

    # Construct a coordinate-dependent optic
    optic = dl.DynamicOptic(
        transmission=dl.Circle(diameter=1.0, edge=1.0),
        opd=dl.DynamicZernikeBasis(orders=[2, 3], diameter=1.0),
        transformation=dl.Distortion(order=3),
        normalise=True,
    )

    # Evaluate and apply the optic on the incident wavefront grid
    grid = dl.GridSpec(n=128, diam=1.2, unit="m")
    wavefront = dl.Wavefront(wavelength=650e-9, grid=grid)
    wavefront = optic(wavefront)
    ```
    """

    coordinates: Array | GridSpec | None
    transformation: BaseCoordTransform | None
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
        """Initialise a general coordinate-dependent scalar optic.

        Parameters
        ----------
        transmission : Array, Parametric, or None
            Static or parametric amplitude transmission.
        opd, phase : Array, Parametric, or None
            Static or parametric OPD in metres and phase in radians.
        coordinates : Array, GridSpec, or None
            Explicit coordinate source, or incident coordinates when omitted.
        transformation : BaseCoordTransform or None
            Optional map into the optic's local frame.
        normalise : bool
            Renormalise wavefront power after application.
        """
        BaseDynamicLayer.__init__(self, coordinates, transformation)
        Optic.__init__(self, transmission, opd, phase, normalise)
