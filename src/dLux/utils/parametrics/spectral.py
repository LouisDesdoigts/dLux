"""Parametric spectral-weight models.

All models use the trailing axis as wavelength. With ``normalise=True``, realised
weights are divided by their sum along that axis, independently for every leading
batch or source element. This is equal-sample normalisation, not wavelength
quadrature. Realised weights must be positive with a finite, non-zero sum.
Positivity remains a caller contract. Sum-based normalisation rejects invalid
measures eagerly; compiled evaluation requires valid inputs. Blackbody uses a
logarithmic softmax for numerical stability.
"""

from __future__ import annotations

import jax.nn as jnn
import jax.numpy as np
import zodiax as zdx
from jax import Array

import dLux.utils as dlu

from .parametrics import Parametric
from .bases import Basis, _coefficients
from .polynomials import _powers

__all__ = ["SpectralPolynomial", "SpectralBasis", "Blackbody"]


def _normalise(weights, normalise):
    """Optionally apply the shared unit-sum trailing-axis spectral contract."""
    return dlu.normalise(weights, mode="sum", axis=-1) if normalise else weights


def _wavelength_samples(wavelengths):
    """Promote one wavelength to a non-empty trailing sample axis."""
    wavelengths = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    if wavelengths.ndim != 1 or wavelengths.size == 0:
        raise ValueError("wavelengths must be a non-empty one-dimensional array.")
    return wavelengths


class SpectralPolynomial(Parametric):
    """Polynomial perturbations around a flat spectral baseline of one.

    Wavelengths are mapped onto ``[-0.5, 0.5]`` before evaluation. ``degree``
    includes every non-constant degree through the requested value, while
    ``degrees`` selects non-constant degrees explicitly. When ``normalise=True``,
    the realised weights are divided by their sum.

    Coefficients may have leading batch axes; each resulting spectrum occupies the
    trailing wavelength axis. Realised weights must remain positive with a finite,
    non-zero sum. Equal normalisation weights every wavelength sample equally, so it
    represents equal-width bins and is not quadrature for nonuniform sampling.

    Examples
    --------
    ```python
    import jax.numpy as np

    import dLux as dl

    # Construct polynomial spectral weights around a flat baseline
    spectral_weights = dl.SpectralPolynomial(degree=2, coeffs=[0.5, -0.2])

    wavelengths = np.linspace(600e-9, 700e-9, 10)
    # Evaluate the unit-sum spectral weights
    weights = spectral_weights.evaluate(wavelengths=wavelengths)
    ```
    """

    coeffs: Array | zdx.Expression
    powers: Array
    normalise: bool

    def __init__(
        self,
        degree=None,
        coeffs=None,
        degrees=None,
        normalise=True,
        *,
        alias=None,
    ):
        """Initialise polynomial spectral weights on normalised coordinates.

        Parameters
        ----------
        degree : int or None
            Maximum degree, mutually exclusive with ``degrees``.
        coeffs : Array, zodiax.Expression, or None
            Coefficients for the selected non-constant terms, defaulting to zeros.
        degrees : int, sequence[int], or None
            Explicit polynomial degrees; the fixed unit baseline is separate.
        normalise : bool
            Normalise resolved weights to unit sum over wavelength.
        alias : mapping or sequence of pairs, optional
            Optional user-defined parameter aliases.
        """
        if degree is not None:
            if degrees is not None:
                raise ValueError("Provide only one of degree or degrees.")
            powers = dlu.polynomial_powers(degree)[:, 1:]
        else:
            powers = _powers(degrees=degrees)
        if powers.shape[1] == 0 or np.any(powers < 1):
            raise ValueError("SpectralPolynomial degrees must be positive.")

        coeffs = np.zeros(powers.shape[1]) if coeffs is None else coeffs
        self.powers = powers
        self.coeffs = _coefficients(coeffs, (powers.shape[1],))
        self.normalise = bool(normalise)

        self.alias = alias

    def evaluate(self, *, wavelengths, **context):
        """Evaluate weights on centred, dimensionless wavelengths.

        ``wavelengths`` is a scalar or non-empty one-dimensional array in any
        consistent unit; only its relative span is used. The trailing axis matches
        wavelength samples; leading coefficient axes are preserved. Unit-sum
        normalisation is applied along the trailing axis when configured.
        """
        # Map wavelengths onto centred dimensionless coordinates
        wavelengths = _wavelength_samples(wavelengths)
        lower = wavelengths.min()
        upper = wavelengths.max()
        span = np.where(upper == lower, 1.0, upper - lower)
        variables = (wavelengths - (lower + upper) / 2) / span

        # Evaluate perturbations around a fixed flat baseline
        context = {**context, "wavelengths": wavelengths}
        coeffs = zdx.resolve(self.coeffs, **context)
        perturbation = dlu.eval_polynomial(variables, coeffs, self.powers)
        weights = 1 + perturbation

        # Apply the shared spectral normalisation contract
        return _normalise(weights, self.normalise)


class SpectralBasis(Parametric):
    """Explicit spectral basis with optional unit-sum normalisation.

    Basis vectors are combined exactly as supplied. Coefficients may have leading
    batch axes; each resulting spectrum occupies the trailing wavelength axis.
    Realised weights must remain positive with a finite, non-zero sum. Equal
    normalisation weights every wavelength sample equally and is not quadrature for
    nonuniform sampling.
    """

    expansion: Basis
    normalise: bool

    def __init__(
        self,
        basis,
        coeffs=None,
        shape=None,
        normalise=True,
        *,
        alias=None,
    ):
        """Initialise spectral weights from an explicit basis.

        Parameters
        ----------
        basis : Array
            Spectral basis with trailing wavelength axis.
        coeffs : Array, zodiax.Expression, or None
            Coefficients ending in ``shape``; leading axes are independent batches.
            Defaults to zeros, which must be updated before unit-sum evaluation.
        shape : tuple[int, ...] or None
            Coefficient dimensions represented by the leading basis axes. Defaults
            to every basis axis except the final wavelength axis, independently of
            coefficient batches or whether coefficients are expressions.
        normalise : bool
            Normalise resolved weights to unit sum over wavelength.
        alias : mapping or sequence of pairs, optional
            Optional user-defined parameter aliases.
        """
        coeffs = zdx.as_array(coeffs, dtype=float)
        basis = zdx.as_array(basis, dtype=float)
        if basis.ndim < 2 or basis.shape[-1] == 0:
            raise ValueError("basis must have coefficient axes and wavelength samples.")
        if shape is None:
            shape = basis.shape[:-1]
        shape = tuple(shape)
        if not shape or len(shape) >= basis.ndim:
            raise ValueError("shape must leave a trailing wavelength axis.")
        if basis.shape[: len(shape)] != shape:
            raise ValueError("The leading basis dimensions must match shape.")
        coeffs = np.zeros(shape) if coeffs is None else coeffs
        self.expansion = Basis(basis, coeffs, axes=len(shape))
        self.normalise = bool(normalise)
        self.alias = alias

    def evaluate(self, **context):
        """Evaluate and optionally normalise sampled spectral weights.

        Coefficient axes are contracted against the leading basis axes. The returned
        trailing axis is the sampled wavelength axis; any leading coefficient batch
        axes are preserved.
        """
        # Contract trailing coefficient axes while preserving every batch axis
        weights = zdx.resolve(self.expansion, **context)

        # Apply the shared spectral normalisation contract
        return _normalise(weights, self.normalise)


class Blackbody(Parametric):
    """Blackbody photon spectrum parameterised by effective temperature.

    This evaluates the photon-number form of Planck's law, proportional to
    ``1 / (wavelength**4 * expm1(h*c / (wavelength*k*T)))``. Wavelengths must be
    supplied in metres and temperature in kelvin. When ``normalise=True``, the
    realised weights are divided by their sum.

    Temperature may have leading batch or source axes; each resulting spectrum
    occupies the trailing wavelength axis. Equal normalisation weights every
    wavelength sample equally, so it represents equal-width bins and is not
    quadrature for nonuniform sampling.

    Examples
    --------
    ```python
    import jax.numpy as np

    import dLux as dl

    # Construct a blackbody photon spectrum
    spectral_weights = dl.Blackbody(temperature=6000)

    wavelengths = np.linspace(400e-9, 900e-9, 20)
    # Evaluate the unit-sum spectral weights
    weights = spectral_weights.evaluate(wavelengths=wavelengths)
    ```
    """

    temperature: Array | zdx.Expression
    normalise: bool

    def __init__(self, temperature, normalise=True, *, alias=None):
        """Initialise blackbody spectral weights.

        Parameters
        ----------
        temperature : float, Array, or zodiax.Expression, kelvin
            Positive effective temperature; leading axes vectorise spectra.
            Expressions resolve from context, including external ``state``.
        normalise : bool
            Normalise resolved weights to unit sum over wavelength.
        alias : mapping or sequence of pairs, optional
            Parameter paths into the temperature model.
        """
        temperature = zdx.as_array(temperature, dtype=float)
        if not isinstance(temperature, zdx.Expression) and np.any(temperature <= 0):
            raise ValueError("temperature values must be positive.")
        self.temperature = temperature
        self.normalise = bool(normalise)

        self.alias = alias

    def evaluate(self, *, wavelengths, **context):
        """Evaluate the blackbody photon spectrum at wavelengths in metres.

        ``wavelengths`` is a positive scalar or non-empty one-dimensional array
        in metres. The output has temperature leading axes followed by wavelength.
        Context, including ``state``, resolves the stored temperature. Unit-sum
        normalisation is applied along the final axis when configured.
        """
        # Evaluate the dimensionless Planck exponent
        wavelengths = _wavelength_samples(wavelengths)
        context = {**context, "wavelengths": wavelengths}
        temperature = zdx.resolve(self.temperature, **context)
        temperature = np.asarray(temperature, dtype=float)
        c2 = 1.438776877e-2
        exponent = c2 / (wavelengths * temperature[..., None])

        # Evaluate stable logarithmic photon-number weights
        log_expm1 = exponent + np.log(-np.expm1(-exponent))
        log_weights = -4 * np.log(wavelengths) - log_expm1

        # Return normalised or absolute relative spectral weights
        if self.normalise:
            return jnn.softmax(log_weights, axis=-1)
        return np.exp(log_weights)
