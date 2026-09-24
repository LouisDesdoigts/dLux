"""Translate released relative weights to current integrated spectral samples."""

import jax.numpy as np
from jax import Array
import dLux.utils as dlu
from .warnings import warn_deprecated

BaseSpectrum = dlu.BaseSpectrum
__all__ = ["BaseSpectrum", "SimpleSpectrum", "Spectrum", "PolySpectrum"]


class _IntegratedSpectrum(dlu.BaseSpectrum):
    """Share the discrete-count boundary used by released spectral constructors."""

    def __init__(self, wavelengths):
        wavelengths = np.atleast_1d(np.asarray(wavelengths, dtype=float))
        if wavelengths.ndim != 1 or wavelengths.size == 0:
            raise ValueError("wavelengths must be a non-empty vector.")
        super().__init__(wavelengths, flux=1.0)

    @property
    def wavelengths(self):
        return self.bandpass

    @property
    def value(self):
        return self.weights

    def sample(self, wavelengths=None, **context):
        if wavelengths is not None:
            raise ValueError("Integrated samples cannot replace their sampling.")
        return self.weights

    def contributions(self, bandpass=None, **context):
        if bandpass is not None:
            raise ValueError("Integrated samples already define their sampling.")
        return self.bandpass, self.weights

    def normalise(self):
        """Calculated weights already sum to one."""
        return self


class Spectrum(_IntegratedSpectrum):
    """Released relative sample weights, separate from current spectral densities."""

    weights: Array

    def __init__(self, wavelengths, weights=None):
        warn_deprecated(
            "compat.Spectrum(wavelengths, weights)",
            "dlu.Spectrum",
            "Spectrum(value, bandpass=...); use Weights for integrated counts",
        )
        super().__init__(wavelengths)
        weights = (
            np.ones_like(self.bandpass)
            if weights is None
            else np.asarray(weights, dtype=float)
        )
        if weights.shape[-1:] != self.bandpass.shape:
            raise ValueError("weights must end in the wavelength dimension.")
        self.weights = weights / weights.sum(-1, keepdims=True)

    def normalise(self):
        """Normalise each spectrum after explicit updates to its weights."""
        return self.set(weights=self.weights / self.weights.sum(-1, keepdims=True))


class PolySpectrum(_IntegratedSpectrum):
    """Evaluate released coefficients in metre wavelengths as relative counts.

    This preserves the old constant term and coefficient convention. It does not
    reinterpret them as the normalised-coordinate coefficients of dlu.PolySpectrum.
    """

    coefficients: Array

    def __init__(self, wavelengths, coefficients):
        warn_deprecated(
            "compat.PolySpectrum",
            "dlu.PolySpectrum",
            "PolySpectrum(bandpass, coeffs=...); convert metre polynomial coefficients",
        )
        super().__init__(wavelengths)
        self.coefficients = np.asarray(coefficients, dtype=float)
        if self.coefficients.ndim != 1:
            raise ValueError("coefficients must be a one-dimensional array.")

    @property
    def weights(self):
        values = np.polyval(self.coefficients[::-1], self.bandpass)
        return values / values.sum()


def SimpleSpectrum(wavelengths):
    return Spectrum(wavelengths)
