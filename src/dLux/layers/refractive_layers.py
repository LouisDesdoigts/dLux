"""Residual refractive optical layers."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as np
from jax import Array

from ..parametric import Parametric, to_param
from ..fields import Wavefront
from .optical_layers import OpticalLayer

__all__ = ["RefractiveOptic", "Wedge"]


class RefractiveOptic(OpticalLayer):
    """Apply a refractive thickness profile as optical path difference."""

    thickness: Array | Parametric = eqx.field(converter=to_param)
    n: Array | Parametric = eqx.field(converter=to_param)

    def __init__(self, thickness, n):
        self.thickness = thickness
        self.n = n

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        self = self.resolve(wavefront=wavefront)
        n = np.asarray(self.n - 1)
        if n.ndim:
            n = n[..., None, None]
        return wavefront.add_opd(n * self.thickness)


class Wedge(OpticalLayer):
    """Apply the optical path of a thin refractive wedge."""

    angle: Array
    n: Array | Parametric = eqx.field(converter=to_param)

    def __init__(self, angle, n):
        self.angle = np.asarray(angle, dtype=float)
        if self.angle.shape != (2,):
            raise ValueError("angle must have shape (2,).")
        self.n = n

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        self = self.resolve(wavefront=wavefront)
        coordinates = wavefront.coordinates
        x, y = coordinates[..., 0, :, :], coordinates[..., 1, :, :]
        thickness = x * np.tan(self.angle[0]) + y * np.tan(self.angle[1])
        n = np.asarray(self.n - 1)
        if n.ndim:
            n = n[..., None, None]
        return wavefront.add_opd(n * thickness)
