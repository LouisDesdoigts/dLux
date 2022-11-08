from __future__ import annotations
from abc import ABC, abstractmethod
import jax.numpy as np
from equinox import tree_at
from jax import vmap
import dLux


__all__ = ["ArraySpectrum", "PolynomialSpectrum", "CombinedSpectrum"]


Array = np.ndarray


class Spectrum(dLux.base.ExtendedBase, ABC):
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

        # Input checking
        assert self.wavelengths.ndim == 1, \
        ("Wavelengths must be a 1d array.")
        assert len(self.wavelengths) > 0, \
        ("Length of wavelengths must be greater than 1.")


    def get_wavelengths(self : Spectrum) -> Array:
        """
        Getter method for the wavelengths.

        Returns
        -------
        wavelengths : Array, meters
            The array of wavelengths at which the spectrum is defined.
        """
        return self.wavelengths


    @abstractmethod
    def normalise(self : Spectrum) -> Spectrum:
        """
        Abstract method to normalise the spectrum. Must be overwitten by child
        classes.
        """
        return


    @abstractmethod
    def get_weights(self : Spectrum) -> Array:
        """
        Abstract getter method for the weights. Must be overwritten by child
        classes.
        """
        return


class ArraySpectrum(Spectrum):
    """
    A Spectrum class that interally parametersises the spectrum via arrays (ie
    wavelengths and weights)

    Attributes
    ----------
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
        weights = np.ones(len(self.wavelengths))/len(wavelengths) \
                                    if weights is None else weights
        weights = np.asarray(weights, dtype=float)
        self.weights = weights/np.sum(weights)

        # Input checks
        assert self.weights.ndim == 1, \
        ("weights must be a 1d array.")
        assert len(self.weights) > 0, \
        ("Length of weights must be greater than 1.")
        assert len(self.wavelengths) == len(self.weights), \
        ("wavelengths and weights must have the same length.")


    def get_weights(self : Spectrum) -> Array:
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
        total_power = self.get_weights().sum()
        norm_weights = self.get_weights()/total_power
        return tree_at(lambda spectrum: spectrum.weights, self, norm_weights)


class PolynomialSpectrum(Spectrum):
    """
    Implements a generic polynomial spectrum. This is likely not needed and
    will probably just be turned into LinearSpectrum in the future.

    This implements a polynomial as follows:
    f(x) = c0 + c1*x + c2*x^2 + ... + cn*x^n

    Attributes
    ----------
    degree : int
        The degree of the polynomial.
    coefficients : Array
        The array of polynomial coefficient values.
    """
    degree       : int # Just a helper
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

        assert self.coefficients.ndim == 1, "Coefficients must be a 1d array."
        assert not np.isnan(self.coefficients).any(), \
        ("Coefficients must not be nan.")
        assert not np.isinf(self.coefficients).any(), \
        ("Coefficients must be not be infinite.")

        # Input checks
        assert self.coefficients.ndim == 1, \
        ("coefficients must be a 1d array.")
        assert len(self.coefficients) > 0, \
        ("Length of coefficients must be greater than 1.")

        self.degree = int(len(coefficients) - 1)


    def get_weights(self : Spectrum) -> Array:
        """
        Gets the relative spectral weights by evalutating the polynomial
        function at the internal wavelengths. This automaically normalises
        the weights to have unitary amplitude.

        Returns
        -------
        weights : Array
            The normalised relative weights of each wavelength.
        """
        generate_polynomial = vmap(lambda wavelength :
                                np.array([self.coefficients[i] * wavelength**i 
                                for i in range(len(self.coefficients))]).sum())
        weights = generate_polynomial(self.wavelengths)
        return weights/weights.sum()


    def normalise(self : Spectrum) -> Spectrum:
        """
        This method currently does nothing becuase solving for normalised
        polynomial coefficients is difficut and the get_weights() method
        already returns normalised weights.

        Returns:
        spectrum : Specturm
            The unmodified spectrum object
        """
        return self


class CombinedSpectrum(ArraySpectrum):
    """
    Implements a combined spectrum, in order to have only a single spectrum
    object for parameterised sources such as binary stars.

    TODO: Expand to be arbitrary number of spectrums, store only a single
    wavelengths array and tile it on the get_wavelengths method. (ie make work
    for MultiPointSource)
    """


    def __init__(self        : Spectrum,
                 wavelengths : Array,
                 weights     : Array = None) -> Spectrum:
        """
        Constructor for the CombinedSpectrum class. Expects wavelengths and
        weights to have the same dimensionality, ie (nsources, nwavelengths).

        Parameters
        ----------
        wavelengths : Array, meters
            The (2, n) array of wavelengths at which the spectrum is defined.
            Input can also be a 1d
        weights : Array (optional)
            The (2, n) relative weights of each wavelength. Defaults to uniform
            throughput.
        """
        super() # Access methods but don't instatiate attributes
        self.wavelengths = np.asarray(wavelengths, dtype=float)

        # Wavelengths
        # Tile single dimension wavelengths inputs
        if self.wavelengths.ndim == 1:
            self.wavelengths = np.tile(self.wavelengths, (2, 1))

        # Input checking
        assert self.wavelengths.ndim == 2, \
        ("Wavelengths must be a 2d array.")
        assert len(self.wavelengths[0]) > 0, \
        ("Length of wavelengths must be greater than 1.")

        # Weights
        weights = np.ones(self.wavelengths.shape)/self.wavelengths.shape[1] \
                                                if weights is None else weights
        self.weights = np.asarray(weights, dtype=float)

        # Tile single dimension weights inputs
        if self.weights.ndim == 1:
            self.weights = np.tile(self.weights, (len(self.wavelengths), 1))

        # Input checking
        assert self.weights.ndim == 2, \
        ("weights must be a 2d array.")
        assert len(self.weights[0]) > 0, \
        ("Length of weights must be greater than 1.")

        # Check consistency between wavelenghts and weights
        assert self.wavelengths.shape == self.weights.shape, "Weights and \
        Wavelengths must have the same shape."


    def normalise(self : Spectrum) -> Spectrum:
        """
        Method for returning a new spectrum object with a normalised total
        spectrum for each individual source.

        Returns
        -------
        spectrum : Specturm
            The spectrum object with the normalised spectrums.
        """
        weights = self.get_weights()
        total_power = weights.sum(1).reshape((len(weights), 1))
        norm_weights = weights/total_power
        return tree_at(lambda spectrum: spectrum.weights, self, norm_weights)