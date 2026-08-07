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
    """Apply a refractive thickness profile as optical path difference.

    Parameters
    ----------
    thickness : Array or Parametric
        Scalar or sampled material thickness in meters.
    n : Array or Parametric
        Wavelength-dependent or fixed refractive index.
    """

    thickness: Array | Parametric
    n: Array | Parametric

    def __init__(self, thickness, n):
        self.thickness = dlu.to_value(thickness, types=Parametric)
        self.n = dlu.to_value(n, types=Parametric)

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        """Apply the resolved refractive optical path to a wavefront."""
        self = self.resolve(wavefront=wavefront)
        n = np.asarray(self.n - 1)
        if n.ndim:
            n = n[..., None, None]
        return wavefront.add_opd(n * self.thickness)


class Wedge(OpticalLayer):
    """Apply the optical path of a thin refractive wedge.

    Parameters
    ----------
    angle : ArrayLike
        Two wedge angles in radians along the physical ``(x, y)`` axes.
    n : Array or Parametric
        Wavelength-dependent or fixed refractive index.
    """

    angle: Array
    n: Array | Parametric

    def __init__(self, angle, n):
        self.angle = dlu.to_value(angle)
        if self.angle.shape != (2,):
            raise ValueError("angle must have shape (2,).")
        self.n = dlu.to_value(n, types=Parametric)

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        """Apply the wavelength-dependent optical path of the wedge."""
        # Resolve the sampled wedge thickness
        self = self.resolve(wavefront=wavefront)
        coordinates = wavefront.coordinates
        x, y = coordinates[..., 0, :, :], coordinates[..., 1, :, :]
        thickness = x * np.tan(self.angle[0]) + y * np.tan(self.angle[1])

        # Convert thickness into refractive optical path
        n = np.asarray(self.n - 1)
        if n.ndim:
            n = n[..., None, None]
        return wavefront.add_opd(n * thickness)
