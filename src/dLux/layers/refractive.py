"""Residual refractive optical layers."""

from __future__ import annotations

import jax.numpy as np
from jax import Array

import dLux.utils as dlu

from ..parametric import Parametric
from ..fields import Wavefront
from .optical import OpticalLayer

__all__ = ["RefractiveOptic", "Wedge"]


class RefractiveOptic(OpticalLayer):
    """Apply a refractive thickness profile as optical path difference."""

    thickness: Array | Parametric
    n: Array | Parametric

    def __init__(self, thickness, n):
        self.thickness = dlu.to_value(thickness, types=Parametric)
        self.n = dlu.to_value(n, types=Parametric)

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        self = self.resolve(wavefront=wavefront)
        n = np.asarray(self.n - 1)
        if n.ndim:
            n = n[..., None, None]
        return wavefront.add_opd(n * self.thickness)


class Wedge(OpticalLayer):
    """Apply the optical path of a thin refractive wedge."""

    angle: Array
    n: Array | Parametric

    def __init__(self, angle, n):
        self.angle = dlu.to_value(angle)
        if self.angle.shape != (2,):
            raise ValueError("angle must have shape (2,).")
        self.n = dlu.to_value(n, types=Parametric)

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        self = self.resolve(wavefront=wavefront)
        coordinates = wavefront.coordinates
        x, y = coordinates[..., 0, :, :], coordinates[..., 1, :, :]
        thickness = x * np.tan(self.angle[0]) + y * np.tan(self.angle[1])
        n = np.asarray(self.n - 1)
        if n.ndim:
            n = n[..., None, None]
        return wavefront.add_opd(n * thickness)
