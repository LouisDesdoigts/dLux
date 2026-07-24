"""Layers that operate on both wavefronts and PSFs."""

from __future__ import annotations

import jax.numpy as np
from jax import Array

from ..coordinates import CoordTransform
from ..psfs import PSF
from ..wavefronts import Wavefront
from .detector_layers import DetectorLayer
from .optical_layers import OpticalLayer

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

    npixels: int

    def __init__(self, npixels: int):
        self.npixels = int(npixels)

    def __call__(self, target: Wavefront | PSF) -> Wavefront | PSF:
        return target.resize(self.npixels)


class Downsample(UnifiedLayer):
    """Downsample a wavefront or PSF by an integer factor."""

    n: int

    def __init__(self, n: int):
        self.n = int(n)
        if self.n <= 0:
            raise ValueError("n must be greater than 0.")

    def __call__(self, target: Wavefront | PSF) -> Wavefront | PSF:
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
        self.fill = np.asarray(fill, dtype=float)

    def __call__(self, target: Wavefront | PSF) -> Wavefront | PSF:
        if isinstance(target, Wavefront):
            return target.interpolate(
                self.transformation,
                method=self.method,
                complex=self.complex,
                fill=self.fill,
            )
        return target.interpolate(
            self.transformation, method=self.method, fill=self.fill
        )


class Normalise(UnifiedLayer):
    """Normalise a wavefront or PSF to unit total power."""

    mode: str
    value: Array

    def __init__(self, mode="power", value=1.0):
        self.mode = str(mode)
        self.value = np.asarray(value, dtype=float)

    def __call__(self, target: Wavefront | PSF) -> Wavefront | PSF:
        return target.normalise(self.mode, self.value)


class Lambda(UnifiedLayer):
    """Return a wavefront or PSF unchanged."""

    def __call__(self, target: Wavefront | PSF) -> Wavefront | PSF:
        return target
