"""Parametric refractive-index and residual optical-path models."""

from __future__ import annotations

import equinox as eqx
import interpax as ipx
import jax.numpy as np
from jax import Array

from ..wavefronts import Wavefront
from .bases import BaseParametric

__all__ = ["CauchyIndex", "InterpolatedIndex"]


class CauchyIndex(BaseParametric):
    """A refractive index represented by a Cauchy dispersion relation.

    ??? abstract "UML"
        ![UML](../assets/uml/CauchyIndex.png)
    """

    coefficients: Array
    scale: Array

    def __init__(self, coefficients: Array, scale: float = 1e-6):
        """Initialise Cauchy coefficients and their wavelength scale."""
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.ndim != 1 or coefficients.size == 0:
            raise ValueError("coefficients must be a non-empty 1d array.")
        if scale <= 0:
            raise ValueError("scale must be positive.")
        self.coefficients = coefficients
        self.scale = np.asarray(scale, dtype=float)

    def evaluate(self, *, wavefront: Wavefront, **kwargs) -> Array:
        """Evaluate ``A + B/x² + C/x⁴ + ...`` at the wavefront wavelength."""
        x = wavefront.wavelength / self.scale
        powers = 2 * np.arange(self.coefficients.size)
        return np.sum(self.coefficients / x[..., None] ** powers, axis=-1)


class InterpolatedIndex(BaseParametric):
    """A refractive index interpolated from wavelength-index samples.

    ??? abstract "UML"
        ![UML](../assets/uml/InterpolatedIndex.png)
    """

    wavelengths: Array
    indices: Array
    method: str = eqx.field(static=True)
    extrapolate: bool = eqx.field(static=True)

    def __init__(
        self,
        wavelengths: Array,
        indices: Array,
        method: str = "linear",
        extrapolate: bool = False,
    ):
        self.wavelengths = np.asarray(wavelengths, dtype=float)
        self.indices = np.asarray(indices, dtype=float)
        if self.wavelengths.ndim != 1 or self.indices.ndim != 1:
            raise ValueError("wavelengths and indices must be 1d arrays.")
        if self.wavelengths.shape != self.indices.shape:
            raise ValueError("wavelengths and indices must have the same shape.")
        if self.wavelengths.size < 2:
            raise ValueError("At least two wavelength-index samples are required.")
        if not bool(np.all(np.diff(self.wavelengths) > 0)):
            raise ValueError("wavelengths must be strictly increasing.")
        self.method = str(method)
        self.extrapolate = bool(extrapolate)

    def evaluate(self, *, wavefront: Wavefront, **kwargs) -> Array:
        return ipx.interp1d(
            wavefront.wavelength,
            self.wavelengths,
            self.indices,
            method=self.method,
            extrap=self.extrapolate,
        )
