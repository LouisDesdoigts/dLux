"""Physical refractive-index conventions over sampled dLux numerical models.

Wavelengths and dispersion scales are in metres; refractive indices are
unitless. Models require explicit ``wavelengths``. Numerical coefficients and
samples have one owner and may resolve from external state. Wavelengths and scales
must be positive.
"""

from __future__ import annotations

import jax.numpy as np
import zodiax as zdx
from jax import Array

import dLux.utils as dlu

from .bases import _coefficients
from .parametrics import Interpolation, Parametric

__all__ = ["CauchyIndex", "PolynomialIndex", "InterpolatedIndex"]


class PolynomialIndex(Parametric):
    """Represent refractive index as a polynomial in normalised wavelength.

    Coefficients are ordered from constant to increasing degree and evaluated at
    ``wavelength / scale``. Both quantities are in metres and the index is
    dimensionless. This flexible model does not enforce a physically causal
    dispersion relation.

    Polynomial powers are fixed at construction and coefficients are stored in
    ``coeffs``. Evaluation retains the wavelength shape, without adding spatial
    axes. Custom aliases may be supplied explicitly.

    Examples
    --------
    ```python
    import dLux as dl

    index = dl.PolynomialIndex([1.5, 0.01, -0.002])
    values = index.evaluate(wavelengths=[500e-9, 600e-9])
    ```
    """

    coeffs: Array | zdx.Expression
    powers: Array
    scale: Array | zdx.Expression

    def __init__(
        self,
        coeffs=None,
        scale=1e-6,
        *,
        degree=None,
        alias=None,
    ):
        """Initialise a polynomial index with fixed term topology.

        Parameters
        ----------
        coeffs : Array or zodiax.Expression
            Non-empty one-dimensional coefficients in ascending degree order.
        scale : float or zodiax.Expression, metres
            Positive scalar wavelength scale; defaults to one micrometre.
        degree : int or None
            Maximum non-negative degree, inferred from concrete coefficients.
            Required for expressions, whose resolved shape must be ``(degree+1,)``.
        alias : mapping or sequence of pairs, optional
            Optional user-defined parameter aliases.
        """
        # Fix the term topology without resolving contextual coefficients.
        coeffs = zdx.as_array(coeffs, dtype=float)
        if isinstance(coeffs, zdx.Expression):
            if degree is None:
                raise ValueError("degree is required for coefficient expressions.")
        else:
            if coeffs is None or coeffs.ndim != 1 or coeffs.size == 0:
                raise ValueError("coeffs must be a non-empty 1d array.")
            degree = coeffs.size - 1 if degree is None else degree
        self.powers = dlu.polynomial_powers(degree)
        self.coeffs = _coefficients(coeffs, (self.powers.shape[1],))

        # Retain the physical wavelength scale as one differentiable input.
        self.scale = zdx.as_array(scale, dtype=float)
        if not isinstance(self.scale, zdx.Expression):
            if self.scale.ndim != 0 or self.scale <= 0:
                raise ValueError("scale must be a positive scalar.")
        self.alias = alias

    def evaluate(self, *, wavelengths, **context) -> Array:
        """Evaluate dimensionless index at positive metre wavelengths.

        Required ``wavelengths`` are in metres. Scalars and
        arbitrary wavelength arrays retain their shape. Named context, including
        ``state``, resolves coefficients and the physical wavelength scale.
        """
        if wavelengths is None:
            raise ValueError("wavelengths must be supplied explicitly.")
        wavelengths = np.asarray(zdx.resolve(wavelengths, **context), dtype=float)
        context = {**context, "wavelengths": wavelengths}
        # Flatten samples to avoid interpreting wavelength axes as coordinate axes.
        coordinates = self._coordinates(wavelengths, **context)
        coeffs = zdx.resolve(self.coeffs, **context)
        values = dlu.eval_polynomial(coordinates.reshape(-1), coeffs, self.powers)
        return values.reshape(wavelengths.shape)

    def _coordinates(self, wavelength, **context):
        """Return dimensionless wavelengths for the stored polynomial."""
        scale = np.asarray(zdx.resolve(self.scale, **context), dtype=float)
        return wavelength / scale


class CauchyIndex(PolynomialIndex):
    """Represent Cauchy dispersion using inverse-square wavelength coordinates.

    Coefficients define ``A + B/x² + C/x⁴ + ...`` for
    ``x = wavelength / scale``. Wavelength and scale are in metres and the output
    is dimensionless. Coefficient storage and polynomial evaluation are
    shared with ``PolynomialIndex``; only the physical coordinates differ.

    Examples
    --------
    ```python
    import dLux as dl

    index = dl.CauchyIndex([1.5, 0.004])
    values = index.evaluate(wavelengths=[500e-9, 600e-9])
    ```
    """

    def __init__(
        self,
        coeffs=None,
        scale=1e-6,
        *,
        degree=None,
        alias=None,
    ):
        """Initialise a Cauchy polynomial with fixed term topology.

        Parameters
        ----------
        coeffs : Array or zodiax.Expression
            Non-empty one-dimensional coefficients ``[A, B, C, ...]``.
        scale : float or zodiax.Expression, metres
            Positive scalar wavelength scale; defaults to one micrometre.
        degree : int or None
            Highest power of inverse-square wavelength, inferred from concrete
            coefficients. Required for expressions of shape ``(degree+1,)``.
        alias : mapping or sequence of pairs, optional
            Optional user-defined parameter aliases.
        """
        super().__init__(coeffs, scale, degree=degree, alias=alias)

    def _coordinates(self, wavelength, **context):
        """Return inverse-square dimensionless wavelengths for Cauchy dispersion."""
        return super()._coordinates(wavelength, **context) ** -2


class InterpolatedIndex(Interpolation):
    """Interpolate dimensionless index from tabulated metre wavelengths.

    This is a dLux interpolation with refractive-index defaults and wavelength
    context. Constructor samples are stored in ``knots`` and ``values``; use those
    names for immutable updates. No sample-name aliases are added automatically.
    Out-of-range values are NaN by default;
    extrapolation beyond measured material data is an explicit modelling choice.

    Examples
    --------
    ```python
    import dLux as dl

    index = dl.InterpolatedIndex([500e-9, 600e-9, 700e-9], [1.52, 1.51, 1.50])
    values = index.evaluate(wavelengths=[500e-9, 600e-9])
    ```
    """

    def __init__(
        self,
        wavelengths,
        indices,
        method="linear",
        extrapolate=False,
        *,
        alias=None,
    ):
        """Initialise a tabulated refractive-index model.

        Parameters
        ----------
        wavelengths : Array or zodiax.Expression, metres
            Strictly increasing one-dimensional samples, with at least two entries.
        indices : Array or zodiax.Expression
            Dimensionless indices with shape matching ``wavelengths``. Expressions
            resolve from context; their data validation follows ``Interpolation``.
        method : str
            Interpax interpolation method, defaulting to linear.
        extrapolate : bool
            Extrapolate beyond the table when true; otherwise return NaN there.
        alias : mapping or sequence of pairs, optional
            Optional user-defined parameter aliases.
        """
        indices = zdx.as_array(indices, dtype=float)
        if not isinstance(indices, zdx.Expression) and indices.ndim != 1:
            raise ValueError("indices must be a one-dimensional array.")
        super().__init__(
            knots=wavelengths,
            values=indices,
            method=method,
            extrapolate=bool(extrapolate),
            alias=alias,
        )

    def evaluate(self, *, wavelengths, **context) -> Array:
        """Interpolate at metre wavelengths, retaining their scalar or array shape.

        Named context, including external state, resolves the tabulated samples.
        """
        if wavelengths is None:
            raise ValueError("wavelengths must be supplied explicitly.")
        wavelengths = np.asarray(zdx.resolve(wavelengths, **context), dtype=float)
        # The interpolation query is wavelength, even inside a spatial optic.
        context = {
            **context,
            "coordinates": wavelengths,
            "wavelengths": wavelengths,
        }
        return super().evaluate(**context)
