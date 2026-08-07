"""Layers that operate on both wavefronts and PSFs."""

from __future__ import annotations

from jax import Array

import dLux.utils as dlu

from ..grids import CoordTransform
from ..fields import PSF, Wavefront
from .detector import DetectorLayer
from .optical import OpticalLayer

__all__ = [
    "UnifiedLayer",
    "Resize",
    "Downsample",
    "Flip",
    "Interpolate",
    "Normalise",
    "Lambda",
]


class UnifiedLayer(OpticalLayer, DetectorLayer):
    """Public contract for operations shared by wavefronts and PSFs."""


class Resize(UnifiedLayer):
    """Resize a wavefront or PSF by padding or cropping."""

    npixels: tuple[int, ...]

    def __init__(self, npixels: int | tuple[int, ...]):
        self.npixels = dlu.as_size(npixels, name="npixels")

    def __call__(self, target: Wavefront | PSF) -> Wavefront | PSF:
        """Resize the target to ``npixels`` along its spatial axes."""
        return target.resize(self.npixels)


class Downsample(UnifiedLayer):
    """Downsample a wavefront or PSF by an integer factor."""

    n: tuple[int, ...]

    def __init__(self, n: int | tuple[int, ...]):
        self.n = dlu.as_size(n, name="n")

    def __call__(self, target: Wavefront | PSF) -> Wavefront | PSF:
        """Downsample the target by the configured integer factors."""
        return target.downsample(self.n)


class Flip(UnifiedLayer):
    """Flip a wavefront or PSF about one or more array axes."""

    axes: tuple[int, ...] | int

    def __init__(self, axes: tuple[int, ...] | int):
        self.axes = axes
        axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
        if not all(isinstance(axis, int) for axis in axes):
            raise ValueError("axes must be an int or tuple of ints.")

    def __call__(self, target: Wavefront | PSF) -> Wavefront | PSF:
        """Flip the target about the configured array axes."""
        return target.flip(self.axes)


class Interpolate(UnifiedLayer):
    """Interpolate a wavefront or PSF through a coordinate transformation."""

    transformation: CoordTransform
    method: str
    complex: bool
    fill: Array

    def __init__(self, transformation, method="linear", complex=True, fill=0.0):
        if not isinstance(transformation, CoordTransform):
            raise TypeError("transformation must be a CoordTransform.")
        self.transformation = transformation
        self.method = str(method)
        self.complex = bool(complex)
        self.fill = dlu.to_value(fill)

    def __call__(self, target: Wavefront | PSF) -> Wavefront | PSF:
        """Interpolate the target through the coordinate transformation."""
        return target.interpolate(
            self.transformation,
            method=self.method,
            complex=self.complex,
            fill=self.fill,
        )


class Normalise(UnifiedLayer):
    """Normalise a wavefront or PSF to unit total power."""

    mode: str
    value: Array

    def __init__(self, mode="power", value=1.0):
        self.mode = str(mode)
        self.value = dlu.to_value(value)

    def __call__(self, target: Wavefront | PSF) -> Wavefront | PSF:
        """Normalise the target to the configured value."""
        return target.normalise(self.mode, self.value)


class Lambda(UnifiedLayer):
    """Return a wavefront or PSF unchanged."""

    def __call__(self, target: Wavefront | PSF) -> Wavefront | PSF:
        """Return the target unchanged."""
        return target
