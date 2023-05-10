from __future__ import annotations
from abc import ABC, abstractmethod
import jax.numpy as np
from equinox import tree_at
from zodiax import Base
from jax import vmap, Array


__all__ = ["ArraySpectrum", "PolynomialSpectrum", "CombinedSpectrum"]


class Spectrum(Base, ABC):
    """
    Abstract base class for arbitrary spectral parametrisations.

    Attributes
    ----------
    wavelengths : Array, meters
        The array of wavelengths at which the spectrum is defined.
    """
    wavelengths : Array


    def __init__(self        : Spectrum,
                 wavelengths : Array) -> Spectrum:
        """
        Constructor for the Spectrum class.

        Parameters
        ----------
        wavelengths : Array, meters
            The array of wavelengths at which the spectrum is defined.
        """
        self.wavelengths = np.asarray(wavelengths, dtype=float)


    @abstractmethod
    def normalise(self : Spectrum) -> Spectrum: # pragma: no cover
        """
        Abstract method to normalise the spectrum. Must be overwitten by child
        classes.
        """
        return


    @abstractmethod
    def weights(self : Spectrum) -> Array: # pragma: no cover
        """
        Abstract getter method for the weights. Must be overwritten by child
        classes. Should be made into a property
        """
        return


class ArraySpectrum(Spectrum):
    """
    A Spectrum class that interally parametersises the spectrum via arrays (ie
    wavelengths and weights)

    Attributes
    ----------
    wavelengths : Array, meters
        The array of wavelengths at which the spectrum is defined.
    weights : Array
        The relative weights of each wavelength.
    """
    weights : Array


    def __init__(self        : Spectrum,
                 wavelengths : Array,
                 weights     : Array = None) -> Spectrum:
        """
        Constructor for the ArraySpectrum class.

        Parameters
        ----------
        wavelengths : Array, meters
            The array of wavelengths at which the spectrum is defined.
        weights : Array = None
            The relative weights of each wavelength. Defaults to uniform
            spectrum. Weights are automatically normalised to a sum of 1.
        """
        super().__init__(wavelengths)
        if weights is None:
            in_shape = self.wavelengths.shape
            weights = np.ones(in_shape)/in_shape[-1]
        weights = np.asarray(weights, dtype=float)
        self.weights = weights / weights.sum(-1)

        if self.wavelengths.shape != self.weights.shape:
            raise ValueError("wavelengths and weights must have the same "
                "shape.")


    @property
    def weights(self : Spectrum) -> Array:
        """
        Getter method for the weights.

        Returns
        -------
        weights : Array
            The relative weights of each wavelength.
        """
        return self.weights


    def normalise(self : Spectrum) -> Spectrum:
        """
        Method for returning a new spectrum object with a normalised total
        spectrum.

        Returns
        -------
        spectrum : Specturm
            The spectrum object with the normalised spectrum.
        """
        return self.divide('weights', self.weights.sum(-1))


class PolynomialSpectrum(Spectrum):
    """
    Implements a generic polynomial spectrum. This is likely not needed and
    will probably just be turned into LinearSpectrum in the future.

    This implements a polynomial as follows:
    f(x) = c0 + c1*x + c2*x^2 + ... + cn*x^n

    Attributes
    ----------
    wavelengths : Array, meters
        The array of wavelengths at which the spectrum is defined.
    coefficients : Array
        The array of polynomial coefficient values.
    """
    coefficients : Array


    def __init__(self         : Spectrum,
                 wavelengths  : Array,
                 coefficients : Array) -> Spectrum:
        """
        Constructor for the PolynomialSpectrum class.

        Parameters
        ----------
        wavelengths : Array, meters
            The array of wavelengths at which the spectrum is defined.
        coefficients : Array
            The array of polynomial coefficient values.
        """
        super().__init__(wavelengths)
        self.coefficients = np.asarray(coefficients, dtype=float)

        if self.coefficeints.ndim != 1:
            raise ValueError("Coefficients must be a 1d array.")


    def _eval_weight(self, wavelength):
        return np.array([self.coefficients[i] * wavelength**i 
            for i in range(len(self.coefficients))]).sum()

    @property
    def weights(self : Spectrum) -> Array:
        """
        Gets the relative spectral weights by evalutating the polynomial
        function at the internal wavelengths. This automaically normalises
        the weights to have unitary amplitude.

        Returns
        -------
        weights : Array
            The normalised relative weights of each wavelength.
        """
        weights = vmap(self._eval_weights)(self.wavelengths)
        return weights/weights.sum()


    def normalise(self : Spectrum) -> Spectrum:
        """
        Calculated weights are automatically normalised, but could be
        calculated from the shift term (ie b in y = mx + b) 

        Returns
        --------
        spectrum : Specturm
            The unmodified spectrum object
        """
        return self