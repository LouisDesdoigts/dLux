from __future__ import annotations
import jax
import jax.numpy as np
import equinox as eqx
import abc
import typing
import dLux

__author__ = "Louis Desdoigts"
__date__ = "30/08/2022"
__all__ = ["ArraySpectrum", "PolynomialSpectrum", "CombinedSpectrum"]

Array =  typing.NewType("Array",  np.ndarray)

class Spectrum(dLux.base.Base, abc.ABC):
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
        Parameters
        ----------
        wavelengths : Array, meters
            The array of wavelengths at which the spectrum is defined.
        """
        self.wavelengths = np.asarray(wavelengths, dtype=float)
    
    
    @abc.abstractmethod
    def get_weights(self : Spectrum) -> Array:
        """
        
        """
        return
    
    
    def get_wavelengths(self : Spectrum) -> Array:
        """
        Getter method for the wavelengths.
        
        Returns
        -------
        wavelengths : Array, meters
            The array of wavelengths at which the spectrum is defined.
        """
        return self.wavelengths
    
    
    @abc.abstractmethod
    def normalise(self : Spectrum) -> Spectrum:
        """
        
        """
        return
    
    
    def set_wavelengths(self : Spectrum, wavelengths : Array) -> Spectrum:
        """
        Setter method for the wavelengths.
        
        Parameters
        ----------
        wavelengths : Array, meters
            The new array of wavelengths at which the spectrum is defined.
        
        Returns
        -------
        spectrum : Specturm
            The spectrum object with the updated wavelengths.
        """
        return eqx.tree_at(
            lambda spectrum : spectrum.wavelengths, self, wavelengths)
    
    
    ### Formatted Output Methods ###
    def _get_wavelengths(self : Spectrum) -> Array:
        """
        Method for returning the wavelengths of the spectrum, formatted
        correctly for the `scene.decompose()` method.
        
        Returns
        -------
        wavelengths : Array, meters
            The formatted array of wavelengths at which the spectrum is defined.
        """
        return np.array([self.wavelengths])
    
    
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
        Parameters
        ----------
        wavelengths : Array, meters
            The array of wavelengths at which the spectrum is defined.
        weights : Array (optional)
            The relative weights of each wavelength. Defaults to uniform
            throughput.
        """
        super().__init__(wavelengths)
        
        weights = np.ones(len(self.wavelengths))/len(wavelengths) \
                                    if weights is None else weights
        self.weights = np.asarray(weights, dtype=float)
        
        assert len(self.wavelengths) == len(self.weights), "Wavelengths and \
        weights must have the same length"
    
    
    def get_weights(self : Spectrum) -> Array:
        """
        Getter method for the weights.
        
        Returns
        -------
        weights : Array
            The relative weights of each wavelength.
        """
        return self.weights
    
    
    def set_weights(self : Spectrum, weights : Array) -> Spectrum:
        """
        Setter method for the weights.
        
        Parameters
        ----------
        weights : Array
            The relative weights of each wavelength.
        
        Returns
        -------
        spectrum : Specturm
            The spectrum object with the updated weights.
        """
        return eqx.tree_at(
            lambda spectrum : spectrum.weights, self, weights)
    
    
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
        return self.set_weights(self.get_weights()/total_power)
    
    
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
    degree : int # Just a helper
    coefficients : Array
    
    
    def __init__(self         : Spectrum,
                 wavelengths  : Array,
                 coefficients : Array) -> Spectrum:
        """
        Parameters
        ----------
        wavelengths : Array, meters
            The array of wavelengths at which the spectrum is defined.
        coefficients : Array
            The array of polynomial coefficient values.
        """
        super().__init__(wavelengths)

        self.coefficients = np.asarray(coefficients, dtype=float)
        self.degree       = int(len(coefficients) - 1)
    
    
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
        # TODO: test this a bit more rigorously - at first pass seems fine
        generate_polynomial = jax.vmap(lambda wavelength : 
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
    """
    
    
    def __init__(self        : Spectrum,
                 wavelengths : Array,
                 weights     : Array) -> Spectrum:
        """
        Expects wavelengths and weights to have the same dimensionality, ie
        (nsources, nwavelengths)
        
        Parameters
        ----------
        wavelengths : Array, meters
            The (2, n) array of wavelengths at which the spectrum is defined.
        weights : Array (optional)
            The (2, n) relative weights of each wavelength. Defaults to uniform
            throughput.
        """
        super().__init__(wavelengths)
        
        assert len(wavelengths) == len(weights), "Weights and \
        Wavelengths must have the same dimnesionality"
        
        self.wavelengths = np.asarray(wavelengths, dtype=float)
        self.weights     = np.asarray(weights, dtype=float)
    
    
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
        return self.set_weights(norm_weights)