"""Residual refractive optical layers and refractive-index models."""

from __future__ import annotations

import equinox as eqx
import interpax as ipx
import jax.numpy as np
from jax import Array

from ..parametric import Parametric
from ..wavefronts import Wavefront
from .optical_layers import TransmissiveLayer

__all__ = ["CauchyIndex", "InterpolatedIndex", "Lens", "Wedge"]


class CauchyIndex(Parametric):
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


class InterpolatedIndex(Parametric):
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


class Lens(TransmissiveLayer):
    """Apply residual refractive OPD from a material-thickness profile.

    ??? abstract "UML"
        ![UML](../assets/uml/Lens.png)
    """

    thickness: Array | Parametric
    n: Array | Parametric

    def __init__(
        self,
        thickness: Array | Parametric,
        n: Array | Parametric,
        transmission: Array | Parametric = None,
        normalise: bool = False,
    ):
        """
        Parameters
        ----------
        thickness : Array | Parametric, metres
            Material thickness relative to the ideal optic. It describes residual
            figure or fabrication errors; ideal focusing remains in the propagator.
        n : Array | Parametric
            Refractive index, optionally dependent on wavefront wavelength.
        """
        super().__init__(transmission=transmission, normalise=normalise)
        self.thickness = self.as_parametric(thickness)
        self.n = self.as_parametric(n)

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        thickness = self.resolve(self.thickness, wavefront=wavefront)
        n = self.resolve(self.n, wavefront=wavefront)
        wavefront *= self.resolve(self.transmission, wavefront=wavefront)
        index_difference = np.asarray(n - 1)
        if index_difference.ndim:
            index_difference = index_difference[..., None, None]
        wavefront = wavefront.add_opd(index_difference * thickness)
        return wavefront.normalise() if self.normalise else wavefront


class Wedge(TransmissiveLayer):
    """Apply the chromatic linear OPD of a thin refractive wedge.

    ??? abstract "UML"
        ![UML](../assets/uml/Wedge.png)
    """

    angle: Array
    n: Array | Parametric
    reference_wavelength: Array | None

    def __init__(
        self,
        angle: Array,
        n: Array | Parametric,
        transmission: Array | Parametric = None,
        reference_wavelength: float = None,
        normalise: bool = False,
    ):
        super().__init__(transmission=transmission, normalise=normalise)
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
            n_reference = self.resolve(self.n, wavefront=reference)
            index_difference = n - n_reference

        coordinates = wavefront.coordinates()
        x, y = coordinates[..., 0, :, :], coordinates[..., 1, :, :]
        thickness = x * np.tan(self.angle[0]) + y * np.tan(self.angle[1])
        wavefront *= self.resolve(self.transmission, wavefront=wavefront)
        index_difference = np.asarray(index_difference)
        if index_difference.ndim:
            index_difference = index_difference[..., None, None]
        wavefront = wavefront.add_opd(index_difference * thickness)
        return wavefront.normalise() if self.normalise else wavefront
