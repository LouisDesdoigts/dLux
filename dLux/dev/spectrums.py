import jax
import jax.numpy as np
import equinox as eqx
import abc
import typing

__author__ = "Louis Desdoigts"
__date__ = "30/08/2022"
__all__ = ["ArraySpectrum", "PolynomialSpectrum", "CombinedSpectrum"]

# Base Jax Types
Scalar = typing.NewType("Scalar", np.ndarray) # 0d
Vector = typing.NewType("Vector", np.ndarray) # 1d
Array =  typing.NewType("Array",  np.ndarray) # 2d +

Spectrum = typing.NewType("Spectrum", object)

class Spectrum(eqx.Module, abc.ABC):
    """
    Abstract base class for arbitrary spectral parametrisation
    
    Current issues: Values can not be internally normalised within a 
    jax/equinox framework since in-place updates aren't possible.
    
    The Best solution seen so far is to treat the class as a container
    and call some normalisation method before the get_spectrum method 
    is called (like wavefronts)
    
    Also how do we treat negative values
    
    Also how how are polynomials normalised (i assume though 
    their intercept/yshift value?)
    """
    wavelengths : Vector
    
    def __init__(self        : Spectrum, 
                 wavelengths : Vector) -> Spectrum:
        self.wavelengths = np.asarray(wavelengths, dtype=float)
    
    
    def get_weights(self : Spectrum) -> Vector:
        """
        Abstract method for returning weight Vector for sources
        """
        pass
    
    
    def get_wavelengths(self : Spectrum) -> Vector:
        """
        Concrete method for returning wavelength Vector for sources
        """
        return self.wavelengths
    
    
    def get_spectrum(self : Spectrum) -> dict:
        """
        Concrete method for returning spectral array for sources
        """
        return np.array([self.get_wavelengths(), self.get_weights()])
    
    
    def normalise(self : Spectrum) -> Spectrum:
        """
        Abstract methods returning a normalised Spectrum object 
        """
        pass
    
    
    def set_wavelengths(self : Spectrum, wavelengths : Vector) -> Spectrum:
        """
        Setter method
        """
        return eqx.tree_at(
            lambda spectrum : spectrum.wavelengths, self, wavelengths)

    
    ### Formatted Output Methods ###
    def _get_wavelengths(self : Spectrum) -> Vector:
        """
        Concrete method for returning wavelength Vector for sources, 
        correctly formatted for stacking
        """
        return np.array([self.wavelengths])
    
class ArraySpectrum(Spectrum):
    """
    
    """
    weights : Vector
    
    def __init__(self        : Spectrum,
                 wavelengths : Vector,
                 weights     : Vector = None) -> Spectrum:
        
        super().__init__(wavelengths)
        
        # TODO: Check what convention we want to use with 'typing'
        # and enforcing data types in constructors
        weights = np.ones(len(self.wavelengths))/len(wavelengths) \
                                    if weights is None else weights
        self.weights = np.asarray(weights, dtype=float)
        
    
    def get_weights(self : Spectrum) -> Vector:
        """
        Abstract method for returning weight Vector for sources
        """
        return self.weights
    
    
    ### Formatted Output Methods ###
    def _get_weights(self : Spectrum) -> Array:
        """
        Abstract method for returning weight Vector for sources,
        correctly formatted for stacking
        """
        return np.array([self.weights])
    
    
    def set_weights(self : Spectrum, weights : Vector) -> Spectrum:
        """
        Setter method
        """
        return eqx.tree_at(
            lambda spectrum : spectrum.weights, self, weights)
    
    
    def normalise(self : Spectrum) -> Spectrum:
        """
        Returns a normalised Spectrum object 
        """
        total_power = self.get_weights().sum()
        return self.set_weights(self.get_weights()/total_power)
    
    
class PolynomialSpectrum(Spectrum):
    """
    This is likely not the best way to parameterise spectrums using 
    polynomails but its not important right now
    """
    degree : int # Just a helper
    coefficients : Vector
    
    def __init__(self : Spectrum,
                 wavelengths : Vector,
                 coefficients : Vector) -> Spectrum:
        
        super().__init__(wavelengths)

        self.coefficients = np.asarray(coefficients, dtype=float)
        self.degree       = int(len(coefficients) - 1)
        
        
    def get_weights(self : Spectrum) -> Vector:
        """
        Gets the relative spectral weights by evalutating the polynomial
        function at the loaded wavelengths
        """
        # TODO: test this a bit more rigorously - at first pass seems fine
        generate_polynomial = jax.vmap(lambda wavelength : 
                                np.array([self.coefficients[i] * wavelength**i 
                                for i in range(len(self.coefficients))]).sum())
        weights = generate_polynomial(self.wavelengths)
        return weights/weights.sum()

    
    ### Formatted Output Methods ###
    def _get_weights(self : Spectrum) -> Array:
        """
        Gets the relative spectral weights by evalutating the polynomial
        function at the loaded wavelengths, correctly formatted for stacking
        """
        return np.array([self.get_weights()])
    
    def normalise(self : Spectrum) -> Spectrum:
        """
        Returns self as i'm not sure how to normalise a general polynomial,
        however the get_weights class always returns a normalised spectrum
        """
        return self
        
        
class CombinedSpectrum(ArraySpectrum):
    """
    Test overwriting of class attribute types
    """
    wavelengths : Array
    weights     : Array
    
    def __init__(self        : Spectrum,
                 wavelengths : Array,
                 weights     : Array) -> Spectrum:
        
        super().__init__(wavelengths)
        
        # TODO: Check what convention we want to use with 'typing'
        # and enforcing data types in constructors
        
        assert len(wavelengths) == len(weights), "Weights and \
        Wavelengths must have the same dimnesionality"
        
        self.wavelengths = np.asarray(wavelengths, dtype=float)
        self.weights     = np.asarray(weights, dtype=float)
    
    
    def normalise(self : Spectrum) -> Spectrum:
        """
        Returns a normalised Spectrum object 
        """
        weights = self.get_weights()
        total_power = weights.sum(1).reshape((len(weights), 1))
        norm_weights = weights/total_power
        return self.set_weights(norm_weights)
