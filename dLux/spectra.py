from __future__ import annotations
from abc import abstractmethod
import jax.numpy as np
from zodiax import Base
from jax import vmap, Array

__all__ = ["Spectrum", "PolySpectrum"]


class BaseSpectrum(Base):
    """
    Abstract base class for arbitrary spectral parametrisations.

    Attributes
    ----------
    wavelengths : Array, metres
        The array of wavelengths at which the spectrum is defined.
    """

    wavelengths: Array

    def __init__(self: Spectrum, wavelengths: Array):
        """
        Constructor for the Spectrum class.

        Parameters
        ----------
        wavelengths : Array, metres
            The array of wavelengths at which the spectrum is defined.
        """
        self.wavelengths = np.asarray(wavelengths, dtype=float)
        super().__init__()

    @abstractmethod
    def normalise(self: Spectrum) -> Spectrum:  # pragma: no cover
        """
        Abstract method to normalise the spectrum. Must be overwritten by child
        classes.
        """


class Spectrum(BaseSpectrum):
    """
    A Spectrum class that internally parametrises the spectrum via arrays (i.e.
    wavelengths and weights)

    Attributes
    ----------
    wavelengths : Array, metres
        The array of wavelengths at which the spectrum is defined.
    weights : Array
        The relative weights of each wavelength.
    """

    weights: Array

    def __init__(self: Spectrum, wavelengths: Array, weights: Array = None):
        """
        Constructor for the Spectrum class.

        Parameters
        ----------
        wavelengths : Array, metres
            The array of wavelengths at which the spectrum is defined.
        weights : Array = None
            The relative weights of each wavelength. Defaults to uniform
            spectrum. Weights are automatically normalised to a sum of 1.
        """
        super().__init__(wavelengths)
        if weights is None:
            in_shape = self.wavelengths.shape
            weights = np.ones(in_shape) / in_shape[-1]

        weights = np.asarray(weights, dtype=float)
        if weights.ndim == 2:
            self.weights = weights / weights.sum(-1)[:, None]
        else:
            self.weights = weights / weights.sum()

        if self.weights.ndim == 1:
            if self.wavelengths.shape != self.weights.shape:
                raise ValueError(
                    "wavelengths and weights must have the same " "shape."
                )
        else:
            if self.wavelengths.shape != self.weights.shape[-1:]:
                raise ValueError(
                    "wavelengths and weights must have the same " "shape."
                )

    def normalise(self: Spectrum) -> Spectrum:
        """
        Method for returning a new spectrum object with a normalised total
        spectrum.

        Returns
        -------
        spectrum : Spectrum
            The spectrum object with the normalised spectrum.
        """
        if self.weights.ndim == 2:
            weight_sum = self.weights.sum(-1)[:, None]
        else:
            weight_sum = self.weights.sum()
        return self.divide("weights", weight_sum)


class PolySpectrum(BaseSpectrum):
    """
    Implements a generic polynomial spectrum. This is likely not needed and
    will probably just be turned into LinearSpectrum in the future.

    This implements a polynomial as follows:
    f(x) = c0 + c1*x + c2*x^2 + ... + cn*x^n

    Attributes
    ----------
    wavelengths : Array, metres
        The array of wavelengths at which the spectrum is defined.
    coefficients : Array
        The array of polynomial coefficient values.
    """

    coefficients: Array

    def __init__(self: Spectrum, wavelengths: Array, coefficients: Array):
        """
        Constructor for the PolySpectrum class.

        Parameters
        ----------
        wavelengths : Array, metres
            The array of wavelengths at which the spectrum is defined.
        coefficients : Array
            The array of polynomial coefficient values.
        """
        super().__init__(wavelengths)
        self.coefficients = np.asarray(coefficients, dtype=float)

        if self.coefficients.ndim != 1:
            raise ValueError("Coefficients must be a 1d array.")

    def _eval_weight(self, wavelength):
        return np.array(
            [
                self.coefficients[i] * wavelength**i
                for i in range(len(self.coefficients))
            ]
        ).sum()

    @property
    def weights(self: Spectrum) -> Array:
        """
        Gets the relative spectral weights by evaluating the polynomial
        function at the internal wavelengths. This automatically normalises
        the weights to have unitary amplitude.

        Returns
        -------
        weights : Array
            The normalised relative weights of each wavelength.
        """
        weights = vmap(self._eval_weight)(self.wavelengths)
        return weights / weights.sum()

    def normalise(self: Spectrum) -> Spectrum:
        """
        Calculated weights are automatically normalised, but could be
        calculated from the shift term (ie b in y = mx + b)

        Returns
        --------
        spectrum : Spectrum
            The unmodified spectrum object
        """
        return self
