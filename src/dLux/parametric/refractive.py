"""Parametric refractive-index and residual optical-path models."""

from __future__ import annotations

import equinox as eqx
import interpax as ipx
import jax.numpy as np
from jax import Array

import dLux.utils as dlu

from ..fields import Wavefront
from .bases import _resolve_coeffs
from .parametrics import Parametric

__all__ = ["CauchyIndex", "PolynomialIndex", "InterpolatedIndex"]


class CauchyIndex(Parametric):
    """Represent refractive index with a Cauchy dispersion relation.

    Coefficients define ``A + B/x² + C/x⁴ + ...`` for
    ``x = wavelength / scale``. Wavelength and scale are in metres and the output is
    dimensionless. The model is evaluated from wavefront context and can be supplied
    directly to `RefractiveOptic` or `Wedge`.

    Examples
    --------
    Construct a simple dispersive refractive optic:

    ```python
    import dLux as dl

    index = dl.CauchyIndex(coeffs=[1.5, 0.004])
    optic = dl.RefractiveOptic(thickness=1e-3, n=index)
    ```
    """

    coeffs: Array
    scale: Array

    def __init__(self, coeffs: Array = None, scale: float = 1e-6, *, coefficients=None):
        """Initialise a Cauchy refractive-index model.

        Parameters
        ----------
        coeffs : Array or None
            Non-empty one-dimensional Cauchy coefficients.
        scale : float
            Wavelength scale in metres used by the polynomial terms.
        coefficients : Array or None
            Deprecated alias for ``coeffs``; the two cannot both be supplied.
        """
        coeffs = _resolve_coeffs(coeffs, coefficients)
        self.coeffs = dlu.to_value(coeffs)
        self.scale = dlu.to_value(scale)

        if self.coeffs.ndim != 1 or self.coeffs.size == 0:
            raise ValueError("coeffs must be a non-empty 1d array.")
        if self.scale <= 0:
            raise ValueError("scale must be positive.")

    @property
    def coefficients(self) -> Array:
        """Return ``coeffs`` through the deprecated attribute alias.

        Access emits a migration warning and the alias will be removed in dLux 0.17.
        """
        # Keep compatibility lazy to avoid the core/legacy import cycle.
        from ..compatibility import warn_deprecated

        warn_deprecated(
            ".coefficients attribute",
            ".coeffs",
            "`model.coefficients` -> `model.coeffs`",
            stacklevel=3,
        )
        return self.coeffs

    def evaluate(self, *, wavefront: Wavefront, **kwargs) -> Array:
        """Evaluate ``A + B/x² + C/x⁴ + ...`` at wavefront wavelengths.

        Here ``x = wavelength / scale`` with both quantities in metres. The returned
        refractive index has the wavelength shape and no spatial axes.
        """
        x = wavefront.wavelength / self.scale
        powers = 2 * np.arange(self.coeffs.size)
        return np.sum(self.coeffs / x[..., None] ** powers, axis=-1)


class PolynomialIndex(Parametric):
    """Represent refractive index as a polynomial in normalised wavelength.

    Coefficients are ordered from constant to increasing degree and evaluated at
    ``x = wavelength / scale``. Wavelength and scale are in metres and the output is
    dimensionless. This flexible model does not enforce a physically causal material
    dispersion relation.

    Examples
    --------
    Apply a polynomial index through a refractive layer:

    ```python
    import dLux as dl

    index = dl.PolynomialIndex(coeffs=[1.5, 0.01, -0.002])
    optic = dl.RefractiveOptic(thickness=1e-3, n=index)
    ```
    """

    coeffs: Array
    scale: Array

    def __init__(self, coeffs: Array = None, scale: float = 1e-6, *, coefficients=None):
        """Initialise a polynomial refractive-index model.

        Parameters
        ----------
        coeffs : Array or None
            Non-empty one-dimensional coefficients in ascending degree order.
        scale : float
            Wavelength scale in metres used before polynomial evaluation.
        coefficients : Array or None
            Deprecated alias for ``coeffs``; the two cannot both be supplied.
        """
        coeffs = _resolve_coeffs(coeffs, coefficients)
        self.coeffs = dlu.to_value(coeffs)
        self.scale = dlu.to_value(scale)

        if self.coeffs.ndim != 1 or self.coeffs.size == 0:
            raise ValueError("coeffs must be a non-empty 1d array.")
        if self.scale <= 0:
            raise ValueError("scale must be positive.")

    @property
    def coefficients(self) -> Array:
        """Return ``coeffs`` through the deprecated attribute alias.

        Access emits a migration warning and the alias will be removed in dLux 0.17.
        """
        # Keep compatibility lazy to avoid the core/legacy import cycle.
        from ..compatibility import warn_deprecated

        warn_deprecated(
            ".coefficients attribute",
            ".coeffs",
            "`model.coefficients` -> `model.coeffs`",
            stacklevel=3,
        )
        return self.coeffs

    def evaluate(self, *, wavefront: Wavefront, **kwargs) -> Array:
        """Evaluate the index polynomial for ``x = wavelength / scale``.

        Wavelength and scale are measured in metres. The returned refractive index has
        the wavelength shape and no spatial axes.
        """
        x = wavefront.wavelength / self.scale
        powers = np.arange(self.coeffs.size)
        return np.sum(self.coeffs * x[..., None] ** powers, axis=-1)


class InterpolatedIndex(Parametric):
    """Interpolate refractive index from tabulated wavelength samples.

    Sample wavelengths are strictly increasing and measured in metres; indices are
    dimensionless. Evaluation uses the incident wavefront wavelength. Extrapolation
    is disabled by default because behaviour outside measured material data is
    generally model-dependent.

    Examples
    --------
    Construct an index model from sampled material data:

    ```python
    import dLux as dl

    wavelengths = [500e-9, 600e-9, 700e-9]
    indices = [1.52, 1.51, 1.50]
    index = dl.InterpolatedIndex(wavelengths, indices)
    optic = dl.RefractiveOptic(thickness=1e-3, n=index)
    ```
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
        """Initialise an interpolated refractive-index model.

        Parameters
        ----------
        wavelengths : Array, metres
            One-dimensional sample wavelengths.
        indices : Array
            Refractive indices with shape matching ``wavelengths``.
        method : str
            Interpolation method accepted by Interpax.
        extrapolate : bool
            Permit evaluation outside the sampled wavelength interval.
        """
        self.wavelengths = dlu.to_value(wavelengths)
        self.indices = dlu.to_value(indices)
        self.method = str(method)
        self.extrapolate = bool(extrapolate)

        if self.wavelengths.ndim != 1 or self.indices.ndim != 1:
            raise ValueError("wavelengths and indices must be 1d arrays.")
        if self.wavelengths.shape != self.indices.shape:
            raise ValueError("wavelengths and indices must have the same shape.")
        if self.wavelengths.size < 2:
            raise ValueError("At least two wavelength-index samples are required.")
        if not bool(np.all(np.diff(self.wavelengths) > 0)):
            raise ValueError("wavelengths must be strictly increasing.")

    def evaluate(self, *, wavefront: Wavefront, **kwargs) -> Array:
        """Interpolate refractive index at wavefront wavelengths in metres.

        The returned array has the wavelength shape. Out-of-range behaviour follows
        the configured ``extrapolate`` value.
        """
        return ipx.interp1d(
            wavefront.wavelength,
            self.wavelengths,
            self.indices,
            method=self.method,
            extrap=self.extrapolate,
        )
