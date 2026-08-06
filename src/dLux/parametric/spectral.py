"""Parametric spectral-weight models.

All models use the trailing axis as wavelength. With ``normalise=True``, realised
weights are divided by their sum along that axis, independently for every leading
batch or source element. This is equal-sample normalization, not wavelength
quadrature. Realised weights must be positive with a finite, non-zero sum; these
conditions are documented rather than enforced inside compiled evaluation.
"""

from __future__ import annotations

import equinox as eqx
import jax.nn as jnn
import jax.numpy as np
from jax import Array

import dLux.utils as dlu
from .bases import Basis
from .parametrics import Parametric
from .polynomials import Polynomial

__all__ = ["SpectralPolynomial", "SpectralBasis", "Blackbody"]


def _normalise(weights, normalise):
    """Optionally apply the shared unit-sum trailing-axis spectral contract."""
    return weights / weights.sum(-1, keepdims=True) if normalise else weights


class SpectralPolynomial(Polynomial):
    """Polynomial perturbations around a flat spectral baseline of one.

    Wavelengths are mapped onto ``[-0.5, 0.5]`` before evaluation. ``degree``
    includes every non-constant degree through the requested value, while
    ``degrees`` selects non-constant degrees explicitly. When ``normalise=True``,
    the realised weights are divided by their sum.

    Coefficients may have leading batch axes; each resulting spectrum occupies the
    trailing wavelength axis. Realised weights must remain positive with a finite,
    non-zero sum. Equal normalization weights every wavelength sample equally, so it
    represents equal-width bins and is not quadrature for nonuniform sampling.
    """

    normalise: bool

    def __init__(
        self,
        degree=None,
        coefficients=None,
        degrees=None,
        normalise=True,
    ):
        if degree is not None:
            if degrees is not None:
                raise ValueError("Provide only one of degree or degrees.")
            degree = int(degree)
            if degree < 1:
                raise ValueError("degree must be positive for SpectralPolynomial.")
            degrees = np.arange(1, degree + 1)
        if degrees is not None:
            degrees = np.atleast_1d(np.asarray(degrees, dtype=int))
            if np.any(degrees < 1):
                raise ValueError("SpectralPolynomial degrees must be positive.")
        super().__init__(coefficients=coefficients, degrees=degrees)
        self.normalise = bool(normalise)

    def evaluate(self, *, wavelengths, **context):
        """Evaluate weights on centred, dimensionless wavelengths."""
        context.pop("variables", None)
        wavelengths = np.asarray(wavelengths, dtype=float)
        lower = wavelengths.min()
        upper = wavelengths.max()
        span = np.where(upper == lower, 1.0, upper - lower)
        variables = (wavelengths - (lower + upper) / 2) / span
        basis = self.calculate_basis(variables=variables, **context)
        weights = 1 + np.tensordot(self.coefficients, basis, axes=((-1,), (0,)))
        return _normalise(weights, self.normalise)


class SpectralBasis(Basis):
    """Explicit spectral basis with optional unit-sum normalization.

    Basis vectors are combined exactly as supplied. Coefficients may have leading
    batch axes; each resulting spectrum occupies the trailing wavelength axis.
    Realised weights must remain positive with a finite, non-zero sum. Equal
    normalization weights every wavelength sample equally and is not quadrature for
    nonuniform sampling.
    """

    normalise: bool

    def __init__(
        self,
        basis,
        coefficients=None,
        coefficient_shape=None,
        normalise=True,
    ):
        super().__init__(basis, coefficients, coefficient_shape)
        self.normalise = bool(normalise)

    def evaluate(self, **context):
        """Evaluate and optionally normalize the sampled spectral weights."""
        ndim = len(self.shape)
        coefficient_axes = tuple(
            range(self.coefficients.ndim - ndim, self.coefficients.ndim)
        )
        basis_axes = tuple(range(ndim))
        weights = np.tensordot(
            self.coefficients,
            self.basis,
            axes=(coefficient_axes, basis_axes),
        )
        return _normalise(weights, self.normalise)


class Blackbody(Parametric):
    """Blackbody photon spectrum parameterized by effective temperature.

    This evaluates the photon-number form of Planck's law, proportional to
    ``1 / (wavelength**4 * expm1(h*c / (wavelength*k*T)))``. Wavelengths must be
    supplied in metres and temperature in kelvin. When ``normalise=True``, the
    realised weights are divided by their sum.

    Temperature may have leading batch or source axes; each resulting spectrum
    occupies the trailing wavelength axis. Equal normalization weights every
    wavelength sample equally, so it represents equal-width bins and is not
    quadrature for nonuniform sampling.
    """

    temperature: Array = eqx.field(converter=dlu.as_float)
    normalise: bool

    def __init__(self, temperature, normalise=True):
        temperature = dlu.as_float(temperature)
        if np.any(temperature <= 0):
            raise ValueError("temperature values must be positive.")
        self.temperature = temperature
        self.normalise = bool(normalise)

    def evaluate(self, *, wavelengths, **context):
        """Evaluate the blackbody photon spectrum at supplied wavelengths."""
        wavelengths = np.asarray(wavelengths, dtype=float)
        second_radiation_constant = 1.438776877e-2
        exponent = second_radiation_constant / (
            wavelengths * self.temperature[..., None]
        )
        log_expm1 = exponent + np.log(-np.expm1(-exponent))
        log_weights = -4 * np.log(wavelengths) - log_expm1
        if self.normalise:
            return jnn.softmax(log_weights)
        return np.exp(log_weights)
