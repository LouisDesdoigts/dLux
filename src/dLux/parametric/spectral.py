"""Parametric spectral-weight models.

All models use the trailing axis as wavelength. With ``normalise=True``, realised
weights are divided by their sum along that axis, independently for every leading
batch or source element. This is equal-sample normalisation, not wavelength
quadrature. Realised weights must be positive with a finite, non-zero sum; these
conditions are documented rather than enforced inside compiled evaluation.
"""

from __future__ import annotations

import jax.nn as jnn
import jax.numpy as np
from jax import Array

import dLux.utils as dlu

from .bases import Basis, _resolve_coeffs
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
    non-zero sum. Equal normalisation weights every wavelength sample equally, so it
    represents equal-width bins and is not quadrature for nonuniform sampling.

    Examples
    --------
    Use polynomial spectral weights directly in a source:

    ```python
    import jax.numpy as np

    import dLux as dl

    # Construct polynomial spectral weights around a flat baseline
    spectrum = dl.SpectralPolynomial(degree=2, coeffs=[0.5, -0.2])

    # Use the parameterisation directly in a source
    wavelengths = np.linspace(600e-9, 700e-9, 10)
    source = dl.Source(wavelengths, weights=spectrum)

    # Evaluate the unit-sum spectral weights
    weights = spectrum.evaluate(wavelengths=wavelengths)
    ```
    """

    normalise: bool

    def __init__(
        self,
        degree=None,
        coeffs=None,
        degrees=None,
        normalise=True,
        *,
        coefficients=None,
    ):
        """Initialise polynomial spectral weights on normalised coordinates.

        Parameters
        ----------
        degree : int or None
            Maximum degree, mutually exclusive with ``degrees``.
        coeffs : Array or None
            Coefficients for the selected non-constant terms.
        degrees : int, sequence[int], or None
            Explicit polynomial degrees; the fixed unit baseline is separate.
        normalise : bool
            Normalise resolved weights to unit sum over wavelength.
        coefficients : Array or None
            Deprecated alias for ``coeffs``.
        """
        coeffs = _resolve_coeffs(coeffs, coefficients)
        if degree is not None:
            if degrees is not None:
                raise ValueError("Provide only one of degree or degrees.")
            degree = int(degree)
            if degree < 1:
                raise ValueError("degree must be positive for SpectralPolynomial.")
            degrees = np.arange(1, degree + 1)
        if degrees is not None:
            degrees = np.atleast_1d(dlu.to_value(degrees, int))
            if np.any(degrees < 1):
                raise ValueError("SpectralPolynomial degrees must be positive.")
        super().__init__(coeffs=coeffs, degrees=degrees)
        self.normalise = bool(normalise)

    def evaluate(self, *, wavelengths, **context):
        """Evaluate weights on centred, dimensionless wavelengths.

        ``wavelengths`` is a one-dimensional array in any consistent unit; only its
        relative span is used. The returned trailing axis matches the wavelength
        samples and leading coefficient axes are preserved. Unit-sum normalisation
        is applied along the trailing axis when configured.
        """
        # Map wavelengths onto centred dimensionless coordinates
        context.pop("variables", None)
        wavelengths = np.asarray(wavelengths, dtype=float)
        lower = wavelengths.min()
        upper = wavelengths.max()
        span = np.where(upper == lower, 1.0, upper - lower)
        variables = (wavelengths - (lower + upper) / 2) / span

        # Evaluate perturbations around a fixed flat baseline
        basis = self.calculate_basis(variables=variables, **context)
        weights = 1 + np.tensordot(self.coeffs, basis, axes=((-1,), (0,)))

        # Apply the shared spectral normalisation contract
        return _normalise(weights, self.normalise)


class SpectralBasis(Basis):
    """Explicit spectral basis with optional unit-sum normalisation.

    Basis vectors are combined exactly as supplied. Coefficients may have leading
    batch axes; each resulting spectrum occupies the trailing wavelength axis.
    Realised weights must remain positive with a finite, non-zero sum. Equal
    normalisation weights every wavelength sample equally and is not quadrature for
    nonuniform sampling.
    """

    normalise: bool

    def __init__(
        self,
        basis,
        coeffs=None,
        shape=None,
        normalise=True,
        *,
        coefficients=None,
    ):
        """Initialise spectral weights from an explicit basis.

        Parameters
        ----------
        basis : Array
            Spectral basis with trailing wavelength axis.
        coeffs : Array or None
            Coefficients contracting the configured basis dimensions.
        shape : tuple[int, ...] or None
            Coefficient dimensions represented by the leading basis axes.
        normalise : bool
            Normalise resolved weights to unit sum over wavelength.
        coefficients : Array or None
            Deprecated alias for ``coeffs``.
        """
        coeffs = _resolve_coeffs(coeffs, coefficients)
        super().__init__(basis, coeffs, shape)
        self.normalise = bool(normalise)

    def evaluate(self, **context):
        """Evaluate and optionally normalise sampled spectral weights.

        Coefficient axes are contracted against the leading basis axes. The returned
        trailing axis is the sampled wavelength axis; any leading coefficient batch
        axes are preserved.
        """
        # Contract the coefficient and basis dimensions
        ndim = len(self.shape)
        b_ax = tuple(range(ndim))
        coeffs = self.coeffs
        c_ax = tuple(range(coeffs.ndim - ndim, coeffs.ndim))
        weights = np.tensordot(coeffs, self.basis, axes=(c_ax, b_ax))

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
    Use a blackbody photon spectrum directly in a source:

    ```python
    import jax.numpy as np

    import dLux as dl

    # Construct a blackbody photon spectrum
    spectrum = dl.Blackbody(temperature=6000)

    # Use the parameterisation directly in a source
    wavelengths = np.linspace(400e-9, 900e-9, 20)
    source = dl.Source(wavelengths, weights=spectrum)

    # Evaluate the unit-sum spectral weights
    weights = spectrum.evaluate(wavelengths=wavelengths)
    ```
    """

    temperature: Array
    normalise: bool

    def __init__(self, temperature, normalise=True):
        """Initialise blackbody spectral weights.

        Parameters
        ----------
        temperature : float or Array, kelvin
            Positive effective temperature; leading axes vectorise spectra.
        normalise : bool
            Normalise resolved weights to unit sum over wavelength.
        """
        temperature = dlu.to_value(temperature)
        if np.any(temperature <= 0):
            raise ValueError("temperature values must be positive.")
        self.temperature = temperature
        self.normalise = bool(normalise)

    def evaluate(self, *, wavelengths, **context):
        """Evaluate the blackbody photon spectrum at wavelengths in metres.

        The returned array has temperature leading axes followed by the wavelength
        axis. Unit-sum normalisation is applied along that final axis when configured.
        """
        # Evaluate the dimensionless Planck exponent
        wavelengths = np.asarray(wavelengths, dtype=float)
        c2 = 1.438776877e-2
        exponent = c2 / (wavelengths * self.temperature[..., None])

        # Evaluate stable logarithmic photon-number weights
        log_expm1 = exponent + np.log(-np.expm1(-exponent))
        log_weights = -4 * np.log(wavelengths) - log_expm1

        # Return normalised or absolute relative spectral weights
        if self.normalise:
            return jnn.softmax(log_weights)
        return np.exp(log_weights)
