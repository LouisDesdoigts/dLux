# import jax
import jax.numpy as np
import equinox as eqx
import abc
import typing
import dLux

__author__ = "Louis Desdoigts"
__date__ = "30/08/2022"
__all__ = ["Source", "PointSource", "ResolvedSource",
           "GaussianSource", "ExtendedSource", "BinarySource"]

# Base Jax Types
Scalar = typing.NewType("Scalar", np.ndarray) # 0d
Vector = typing.NewType("Vector", np.ndarray) # 1d
Array =  typing.NewType("Array",  np.ndarray) # 2d +

Spectrum = typing.NewType("Spectrum", object)
Source   = typing.NewType("Source",   object)

"""
TODO Build out resolved sources properly
"""

class Source(dLux.Base, abc.ABC):
    """
    Source: Base Class
    
    The idea of these source classes is to allow an arbitrary parametrisation of the
    underlying astrophyical objects, through which cartesain parameters are passed up 
    to the higher level classes which is then used to model them through the opticd
    
    High level - what do we want here?
    Source, Abstract
        Point Source, Concrete
        Resolved Source, abstract
            Gaussian source, concrete (param'd by some gaussian distribution)
            Extended Source, concrete (Arbitrary grid definition)

    """
    resolved          : bool # Does this need to exist? Likely not but it probs helps high level class logic
    position          : Vector # Union with float though?? 
    flux              : Vector # Union with float though?? 
    spectrum          : Spectrum
    name              : str = eqx.static_field()
    
    
    def __init__(self              : Source, 
                 position          : Vector,
                 flux              : Vector,
                 spectrum          : Spectrum,
                 resolved          : bool,
                 name              : str = 'Source') -> Source:
    
        self.position          = np.asarray(position, dtype=float)
        self.flux              = np.asarray(flux,     dtype=float)
        self.spectrum          = spectrum # Will this error if its not a 'Spectrum' class? I hope so...
        self.resolved          = bool(resolved)
        self.name              = name
        
        
    ### Start Getter Methods ###
    def get_flux(self : Source) -> Vector:
        """
        Concrete method for returning the flux of the source object
        """
        return self.flux
        
    def get_position(self : Source) -> Array:
        """
        Concrete method for returning the on-sky position of this source
        """
        return self.position
    
    def get_spectrum(self : Source) -> Array:
        """
        Concrete method for returning the single source internal spectrum
        """
        return self.spectrum.get_spectrum()
    
    def get_wavelengths(self : Source) -> Array:
        """
        Concrete method for returning the single source internal wavelengths
        """
        return self.spectrum.get_wavelengths()
    
    def get_weights(self : Source) -> Array:
        """
        Concrete method for returning the single source internal weights
        """
        return self.spectrum.get_weights()
    
    def is_resolved(self : Source) -> bool:
        """
        Concrete method for returning if the source is resolved or not
        """
        return self.resolved
    ### End Getter Methods ###
    
    
    ### Correctly Formatted Outputs ###
    def _get_wavelengths(self : Source) -> Array:
        """
        Concrete method for returning the single source internal wavelengths,
        correctly formatted for stacking
        """
        return self.spectrum._get_wavelengths()
    
    def _get_weights(self : Source) -> Array:
        """
        Concrete method for returning the single source internal weights,
        correctly formatted for stacking
        """
        return self.spectrum._get_weights()
    
    def _get_flux(self : Source) -> Vector:
        """
        Concrete method for returning the flux of the source object,
        correctly formatted for stacking
        """
        return np.array([self.flux])
    
    def _get_position(self : Source) -> Array:
        """
        Concrete method for returning the on-sky position of this source,
        correctly formatted for stacking
        """
        nwavels = self.get_wavelengths().shape[-1]
        return np.array([np.tile(self.get_position(), (nwavels, 1))])
        # return np.array([self.get_position()])
        
    def _is_resolved(self : Source) -> bool:
        """
        Concrete method for returning if the source is resolved or not,
        correctly formatted for stacking
        """
        nwavels = self.get_wavelengths().shape[-1]
        return np.tile(self.is_resolved(), (1, nwavels))
    
    
    ### Start Setter Methods ###
    def set_flux(self : Source, flux : Vector) -> Source:
        """
        Concrete method for setting the flux of the source object
        """
        return eqx.tree_at(
            lambda source : source.flux, self, flux)  
        
    def set_position(self : Source, position : Vector) -> Source:
        """
        Concrete method for returning the on-sky position of this source
        """
        return eqx.tree_at(
            lambda source : source.position, self, position)
    
    def set_spectrum(self : Source, spectrum : Spectrum) -> Source:
        """
        Concrete method for returning the single source internal spectrum
        """
        return eqx.tree_at(
            lambda source : source.spectrum, self, spectrum)
    
    def set_resolved(self : Source, resolved : bool) -> Source:
        """
        Concrete method for returning if the source is resolved or not
        """
        return eqx.tree_at(
            lambda source : source.resolved, self, resolved)
    ### End Setter Methods ###
    
    
    def normalise(self : Source) -> Source:
        """
        Returns the Source object with an internally normalised spectrum
        """
        normalised_spectrum = self.spectrum.normalise()
        return eqx.tree_at(
            lambda source : source.spectrum, self, normalised_spectrum)
    
    
class PointSource(Source):
    """
    Source
        PointSource
        
    Concrete Class for unresolved point source objects
    
    Essentially just passes the resolved boolean value to Source
    """
    
    def __init__(self     : Source, 
                 position : Vector,
                 flux     : Vector,
                 spectrum : Spectrum,
                 name     : str = 'Source') -> Source:
        
        super().__init__(position, flux, spectrum, False, name=name)
        
        
class ResolvedSource(Source, abc.ABC):
    """
    Source 
        ResolvedSource (abstract)
    
    Abstract Class for resolved source objects
    
    Essentially just passes the resolved boolean value to Source
    """
    
    def __init__(self : Source, 
                 position : Vector,
                 flux     : Vector,
                 spectrum : Spectrum,
                 name     : str = 'Source') -> Source:
        
        super().__init__(position, flux, spectrum, True, name=name)
        
    # TODO: Implement this, maybe do some clever array shape checking
    # in order to always implement the most efficient convolution type
    def convolve_source(self : Source, psf : Array) -> Array:
        """
        Possible have this exist as an external function that this 
        """
        pass
        
        
class GaussianSource(ResolvedSource):
    """
    Source
        ResolvedSource
            GaussianSource
    
    Concrete class for sources with gaussian flux distribution
    Assumed rotationally symmetric, ie parametrised by a single sigma value
    
    Sigma here can have physical 'on sky' units, allowing the convolution 
    to take place with a gaussian kernel sampled at the same spatial resoltuion
    as the psf at any given time
    
    """
    sigma : Vector
    
    def __init__(self : Source, 
                 position : Vector,
                 flux     : Vector,
                 spectrum : Spectrum,
                 sigma    : Vector,
                 name     : str = 'Source') -> Source:
        
        super().__init__(position, flux, spectrum, name=name)
        self.sigma = np.asarray(sigma, dtype=float)
        
    ## TODO: Implement the generation of the convolution
    ## Theres a few upstream qs to answer so I leave this for now
    ## Algorithmically this should be identical to ApplyJitter
    def generate_kernel(self : Source) -> Array:
        """
        
        """
        pass
    
    
class ExtendedSource(ResolvedSource):
    """
    Source
        ResolvedSource
            ExtendedSource
    
    Concrete class for sources with non-parametric (pixel based) on sky
    flux distribution
    
    This class does not adress normalisations yet - Similar to Spectrum
    I believe this should be added as a class method and called before
    being used in a convolution
    
    Basic idea is that there is some underlying source distribution defined
    on a pixel grid, with some defined pixel scale. A point sounce PSF is 
    modelled at the given RA and DEC and the resluting PSF is convolved
    with this distribution. 
    
    The distribution may need to be interpolated over in order to perform 
    the convolution with the PSF at the same sampling.
    
    Should pixel_scale be a jax type? Im really not sure. No for now, can 
    always be changed trivially
    """
    disribution : Array
    pixel_scale : float # Python type non optimisible, maybe change later
    
    def __init__(self         : Source, 
                 position     : Vector,
                 flux         : Vector,
                 spectrum     : Spectrum,
                 distribution : Array,
                 pixel_scale  : Vector,
                 name         : str = 'Source') -> Source:
        
        super().__init__(RA, DEC, flux, spectrum, name=name)
        self.distribution = np.asarray(distribution, dtype=float)
        self.pixel_scale = float(pixel_scale)
        
    ## TODO: Figure out what else we need to implement this fully
    
    
    
# TODO: Implement this - Requires some thinking
class BinarySource(Source):
    """
    
    """
    separation  : Scalar
    field_angle : Scalar
    flux_ratio  : Vector
    resolved    : Vector # Overwritten
    
    
    def __init__(self     : Source, 
                 position : Vector,
                 flux     : Vector,
                 
                 separation  : Scalar,
                 field_angle : Scalar,
                 flux_ratio  : Vector,
                 spectrum    : Spectrum, # Converted into dict
                 resolved    : list, # Converted into dict
                 name        : str = 'Source'
                 ) -> Source:
        
        super().__init__(position, flux, spectrum, None, name=name)
        
        self.separation  = np.asarray(separation, dtype=float)
        self.field_angle = np.asarray(field_angle, dtype=float)
        self.flux_ratio  = np.asarray(flux_ratio, dtype=float)
        
        assert len(resolved) == 2, "Resolved list must contain exactly two values"
        
        self.resolved = np.asarray(resolved, dtype=bool)
        
        
    ### Start Getter Methods ###
    def get_separation(self : Source) -> Array:
        """
        Concrete method for returning the on-sky separation of this source
        """
        return self.separation
    
    def get_field_angle(self : Source) -> Array:
        """
        Concrete method for returning the on-sky field_angle of this source
        """
        return self.field_angle
    
    def get_flux_ratio(self : Source) -> Array:
        """
        Concrete method for returning the on-sky flux_ratio of this source
        """
        return self.flux_ratio
    
    def get_flux(self : Source) -> Array:
        """
        Concrete method for returning the on-sky fluxes of these sources.
        This overwrites the base Source class `get_flux()` method.
        """
        flux_A = 2 * self.get_flux_ratio() * super().get_flux() / (1 + self.get_flux_ratio())
        flux_B = 2 * super().get_flux() / (1 + self.get_flux_ratio())
        return np.array([flux_A, flux_B])
    
    def get_position(self : Source) -> Array:
        """
        Concrete method for returning the on-sky positions of these sources.
        This overwrites the base Source class `get_position()` method.
        """
        sep_vec = dLux.utils.polar2cart(self.get_separation()/2, 
                                        self.get_field_angle())
        return np.array([super().get_position() + sep_vec, 
                         super().get_position() - sep_vec])
    ### End Getter Methods ###
    
    
    ### Correctly Formatted Outputs ###
    def _get_wavelengths(self : Source) -> Array:
        """
        Concrete method for returning the single source internal wavelengths,
        correctly formatted for stacking
        """
        return self.spectrum._get_wavelengths()
    
    def _get_weights(self : Source) -> Array:
        """
        Concrete method for returning the single source internal weights,
        correctly formatted for stacking
        """
        return self.spectrum._get_weights()
    
    def _get_flux(self : Source) -> Vector:
        """
        Concrete method for returning the flux of the source object,
        correctly formatted for stacking
        """
        # return self.get_flux()
        return np.expand_dims(self.get_flux(), -1)
    
    def _get_position(self : Source) -> Array:
        """
        Concrete method for returning the on-sky position of this source,
        correctly formatted for stacking
        """
        
        position_A, position_B = self.get_position()
        nwavels = self.get_wavelengths().shape[-1]
        return np.array([np.tile(position_A, (nwavels, 1)), 
                         np.tile(position_B, (nwavels, 1))])
    
    def _is_resolved(self : Source) -> bool:
        """
        Concrete method for returning if the source is resolved or not,
        correctly formatted for stacking
        """
        nwavels = self.get_wavelengths().shape[-1]
        resolved_A, resolved_B = self.is_resolved()
        return np.array([np.tile(resolved_A, (nwavels)), 
                         np.tile(resolved_B, (nwavels))])
    
    
    ### Start Setter Methods ###
    def set_separation(self : Source) -> Source:
        """
        Concrete method for setting the on-sky separation of this source
        """
        return eqx.tree_at(
           lambda source: source.separation, self, separation)
    
    def set_field_angle(self : Source) -> Source:
        """
        Concrete method for setting the on-sky field_angle of this source
        """
        return eqx.tree_at(
           lambda source: source.field_angle, self, field_angle)
    
    def set_flux_ratio(self : Source) -> Source:
        """
        Concrete method for setting the on-sky flux_ratio of this source
        """
        return eqx.tree_at(
           lambda source: source.flux_ratio, self, flux_ratio)
    ### End Setter Methods ###
    
