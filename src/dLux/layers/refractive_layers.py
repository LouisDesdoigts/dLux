"""Residual refractive optical layers."""

from __future__ import annotations

import jax.numpy as np
from jax import Array

from ..parametric import BaseParametric
from ..wavefronts import Wavefront
from .optical_layers import OpticalLayer

__all__ = ["Lens", "Wedge"]


class Lens(OpticalLayer):
    """Apply residual refractive thickness as optical path difference."""

    thickness: Array | BaseParametric
    n: Array | BaseParametric

    def __init__(self, thickness, n):
        self.thickness = self.as_parametric(thickness)
        self.n = self.as_parametric(n)

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        thickness = self.resolve(self.thickness, wavefront=wavefront)
        n = np.asarray(self.resolve(self.n, wavefront=wavefront) - 1)
        if n.ndim:
            n = n[..., None, None]
        return wavefront.add_opd(n * thickness)


class Wedge(OpticalLayer):
    """Apply the residual optical path of a thin refractive wedge."""

    angle: Array
    n: Array | BaseParametric
    reference_wavelength: Array | None

    def __init__(self, angle, n, reference_wavelength=None):
        self.angle = np.asarray(angle, dtype=float)
        if self.angle.shape != (2,):
            raise ValueError("angle must have shape (2,).")
        self.n = self.as_parametric(n)
        self.reference_wavelength = (
            None
            if reference_wavelength is None
            else np.asarray(reference_wavelength, dtype=float)
        )

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        n = self.resolve(self.n, wavefront=wavefront)
        index_difference = n - 1
        if self.reference_wavelength is not None:
            reference = wavefront.set(wavelength=self.reference_wavelength)
            index_difference = n - self.resolve(self.n, wavefront=reference)

        coordinates = wavefront.coordinates()
        x, y = coordinates[..., 0, :, :], coordinates[..., 1, :, :]
        thickness = x * np.tan(self.angle[0]) + y * np.tan(self.angle[1])
        index_difference = np.asarray(index_difference)
        if index_difference.ndim:
            index_difference = index_difference[..., None, None]
        return wavefront.add_opd(index_difference * thickness)
