"""Physical refractive-index conventions over sampled dLux numerical models.

Wavelengths and dispersion scales are in metres; refractive indices are
unitless. Models use explicit ``wavelengths`` or stored one-dimensional sampling.
Numerical coefficients and samples have one owner and may resolve from external
state. Wavelengths and scales must be positive.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import interpax as ipx
import jax.numpy as np
import zodiax as zdx
from jax import Array, core

from .. import grids
from ..grids.specs import UnitSpec
from ..grids.transforms import BaseCoordTransform
from ..bases.basis import initialise_coeffs
from ..numerical.interpolation import Interp
from ..numerical.polynomials import eval_polynomial, polynomial_powers
from ..sampled import Sampled

__all__ = ["CauchyIndex", "PolynomialIndex", "InterpolatedIndex"]


class PolynomialIndex(Sampled):
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

    # Inherited fields
    grid: UnitSpec | None = None
    transform: Callable | None = None
    strict: bool = eqx.field(static=True, default=False)

    coeffs: Array | zdx.Expression
    powers: Array
    scale: Array | zdx.Expression

    def __init__(
        self,
        coeffs=None,
        scale=1e-6,
        *,
        degree=None,
        grid=None,
        transform=None,
        strict=False,
        alias=None,
    ):
        """Initialise a polynomial index with fixed term topology.

        Parameters
        ----------
        coeffs : Array, zodiax.Expression, or None
            One-dimensional coefficients in ascending degree order. None creates
            zero coefficients when degree is supplied.
        scale : float or zodiax.Expression, metres
            Positive scalar wavelength scale; defaults to one micrometre.
        degree : int or None
            Maximum non-negative degree, inferred from concrete coefficients.
            Required for expressions, whose resolved shape must be ``(degree+1,)``.
        grid : UnitSpec or None
            Optional one-dimensional wavelength sampling, expressed in SI by UnitSpec.
        transform : callable or None
            Optional map applied to wavelength samples before evaluation.
        strict : bool
            Reject replacing stored sampling or transformation at evaluation.
        alias : mapping or sequence of pairs, optional
            Optional user-defined parameter aliases.
        """
        super().__init__(grid, transform, strict, alias=alias)
        # Fix the term topology without resolving contextual coefficients.
        coeffs = zdx.as_array(coeffs, dtype=float)
        if coeffs is None or isinstance(coeffs, zdx.Expression):
            if degree is None:
                raise ValueError("degree is required without concrete coefficients.")
        else:
            if coeffs.ndim != 1 or coeffs.size == 0:
                raise ValueError("coeffs must be a non-empty 1d array.")
            degree = coeffs.size - 1 if degree is None else degree
        self.powers = polynomial_powers(degree)
        self.coeffs = initialise_coeffs(coeffs, (self.powers.shape[1],))

        # Retain the physical wavelength scale as one differentiable input.
        self.scale = zdx.as_array(scale, dtype=float)
        if not isinstance(self.scale, zdx.Expression):
            if self.scale.ndim != 0:
                raise ValueError("scale must be a positive scalar.")
            if not isinstance(self.scale, core.Tracer) and self.scale <= 0:
                raise ValueError("scale must be a positive scalar.")

    def evaluate(
        self, *, wavelengths=None, grid=None, transform=None, **context
    ) -> Array:
        """Evaluate index using explicit wavelengths or stored spectral sampling.

        Scalars and arbitrary wavelength arrays retain their shape. Ambient
        spatial coordinates are ignored; explicit wavelengths also take priority
        over a runtime spatial grid.
        """
        # Replace spatial context with wavelength sampling before resolving dispersion.
        context.pop("coordinates", None)
        if isinstance(grid, grids.UnitSpec) and grid.ndim != 1:
            grid = None
        context = self.sampling_context(
            grid if wavelengths is None else None,
            coordinates=wavelengths,
            transform=transform,
            ndim=1,
            **context,
        )
        wavelengths = np.asarray(context.pop("coordinates"), dtype=float)
        context["wavelengths"] = wavelengths
        coeffs, scale = zdx.resolve((self.coeffs, self.scale), **context)
        # Flatten samples to avoid interpreting wavelength axes as coordinate axes.
        coordinates = self.set(scale=scale)._coordinates(wavelengths)
        values = eval_polynomial(coordinates.reshape(-1), coeffs, self.powers)
        return values.reshape(wavelengths.shape)

    def _coordinates(self, wavelength):
        """Return dimensionless wavelengths for the resolved polynomial."""
        return wavelength / self.scale


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

    # Inherited fields
    grid: UnitSpec | None = None
    transform: Callable | None = None
    strict: bool = eqx.field(static=True, default=False)

    coeffs: Array | zdx.Expression
    powers: Array
    scale: Array | zdx.Expression

    def _coordinates(self, wavelength):
        """Return inverse-square dimensionless wavelengths for Cauchy dispersion."""
        return super()._coordinates(wavelength) ** -2


class InterpolatedIndex(Interp):
    """Interpolate dimensionless index from tabulated metre wavelengths.

    This is a dLux interpolation with refractive-index defaults and wavelength
    context. Samples follow Interp: metre wavelengths are stored in ``grid_in``
    and dimensionless values in ``array``. Use those names for immutable updates.
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

    # Inherited fields
    array: Array | zdx.Expression | None = None
    grid_in: UnitSpec | Array | tuple[Array, ...] | zdx.Expression | None = None
    grid_out: UnitSpec | None = None
    transform: BaseCoordTransform | None = None
    complex: bool = eqx.field(static=True, default=True)
    interpolator: ipx.AbstractInterpolator | None = None

    ndim: int = eqx.field(static=True, default=1)
    method: str
    fill: Array | zdx.Expression | None = None

    def __init__(
        self,
        wavelengths,
        indices,
        method="cubic2",
        fill=np.nan,
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
            may resolve from context before Interp samples them.
        method : str
            Interpax interpolation method, defaulting to cubic2.
        fill : float, Array, zodiax.Expression, or None
            Outside value, defaulting to NaN. None extrapolates beyond the table.
        alias : mapping or sequence of pairs, optional
            Optional user-defined parameter aliases.
        """
        # Check concrete wavelength/index tables while retaining deferred expressions.
        indices = zdx.as_array(indices, dtype=float)
        wavelengths = zdx.as_array(wavelengths)
        if isinstance(wavelengths, Array):
            if np.iscomplexobj(wavelengths):
                raise TypeError("wavelengths must be real.")
            if wavelengths.ndim != 1 or wavelengths.size < 2:
                raise ValueError(
                    "wavelengths must be a vector with at least two samples."
                )
        if isinstance(indices, Array):
            if indices.ndim != 1:
                raise ValueError("indices must be a one-dimensional array.")
            if isinstance(wavelengths, Array) and indices.shape != wavelengths.shape:
                raise ValueError("indices must match the wavelength sample shape.")
        super().__init__(
            array=indices,
            grid_in=wavelengths,
            ndim=1,
            method=method,
            fill=fill,
            alias=alias,
        )

    def evaluate(self, *, wavelengths, **context) -> Array:
        """Interpolate at metre wavelengths, retaining their scalar or array shape.

        Named context, including external state, resolves the tabulated samples.
        """
        if wavelengths is None:
            raise ValueError("wavelengths must be supplied explicitly.")
        wavelengths = np.asarray(zdx.resolve(wavelengths, **context), dtype=float)
        # Interp resolves table expressions on their native knots. Query wavelengths
        # must not replace those coordinates before source preparation.
        # A tuple preserves arbitrary query shapes without introducing batch axes.
        context.pop("coordinates", None)
        return self.sample((wavelengths,), **context)
