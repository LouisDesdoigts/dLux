from __future__ import annotations
import jax
import jax.numpy as np
import equinox as eqx
import abc
import typing
import dLux
import jax.scipy as jsp

__author__ = "Louis Desdoigts"
__date__ = "30/08/2022"
__all__ = ["PointSource", "ArrayDistribution", "BinarySource", 
           "PointExtendedSource", "PointAndExtendedSource"]

# Base Jax Types
Array    = typing.NewType("Array",  np.ndarray)

"""
Complex class inheritance: Parameterisation classes can only implement methods,
not attributes because python
"""

########################
### Abstract Classes ###
########################

class Source(dLux.base.Base, abc.ABC):
    """
    Base class for source objects. The idea of these source classes is to allow
    an arbitrary parametrisation of the underlying astrophyical objects. Each 
    source object requires a normalise(), format_inputs(), and model() methods,
    along with the regular getter and setter methods.
    
    The normalise() method should return a new instance of the class with all
    the appropriate attribues normalised (ie spectral weights, resolved source
    distribution).
    
    The format_inputs() method should return the relevant wavelengths, weights,
    positions etc, that are correctly formatted to be used by the model() 
    method. This should primarily interface with the specrum object, and should
    call the normalise() method.
    
    The model() method should return a single psf of the source. This is done 
    inside of the source object so that each class can be arbitrarily 
    parameterised as required. This method should call the format_inputs()
    method.
    
    A series of parameterisations have been proided to model a series of 
    different astrophysical objects. 
    
    Attributes
    ----------
    position : Array, radians
        The (x, y) on-sky position of this object. Units are currently in
        radians, but will likely be extended to RA/DEC.
    flux : Array, photons
        The flux of the object.
    spectrum : Spectrum
        The spectrum of this object, represented by a Spectrum object.
    name : str
        The name for this object.
    """
    position : Array
    flux     : Array
    spectrum : Spectrum
    name     : str = eqx.static_field()
    
    
    def __init__(self     : Source, 
                 position : Array,
                 flux     : Array,
                 spectrum : Spectrum,
                 name     : str = 'Source') -> Source:
        """
        Parameters
        ----------
        position : Array, radians
            The (x, y) on-sky position of this object. Units are currently in
            radians, but will likely be extended to RA/DEC.
        flux : Array, photons
            The flux of the object.
        spectrum : Spectrum
            The spectrum of this object, represented by a Spectrum object.
        name : str (optional)
            The name for this object. Defaults to 'Source'
        """
        self.position = np.asarray(position, dtype=float)
        self.flux     = np.asarray(flux,     dtype=float)
        self.spectrum = spectrum
        self.name     = name
    
    
    ### Start Getter Methods ###
    def get_flux(self : Source) -> Array:
        """
        Getter method for the flux.
        
        Returns
        -------
        flux : Array, photons
            The flux of the object.
        """
        return self.flux
    
    
    def get_position(self : Source) -> Array:
        """
        Getter method for the position.
        
        Returns
        -------
        position : Array, radians
            The (x, y) on-sky position of this object.
        """
        return self.position
    
    
    def get_wavelengths(self : Source) -> Array:
        """
        Getter method for the source internal wavelengths.
        
        Returns
        -------
        wavelengths : Array, meters
            The array of wavelengths at which the spectrum is defined.
        """
        return self.spectrum.get_wavelengths()
    
    
    def get_weights(self : Source) -> Array:
        """
        Getter method for the source internal weights.
        
        Returns
        -------
        weights : Array
            The relative weights of each wavelength.
        """
        return self.spectrum.get_weights()
    
    
    ### Start Setter Methods ###
    def set_flux(self : Source, flux : Array) -> Source:
        """
        Setter method for the flux.
        
        Parameters
        ----------
        flux : Array, photons
            The flux of the object.
        
        Returns
        -------
        source : Source
            The source object with the updated flux parameter.
        """
        return eqx.tree_at(
            lambda source : source.flux, self, flux)
    
    
    def set_position(self : Source, position : Array) -> Source:
        """
        Setter method for the position.
        
        Parameters
        ----------
        position : Array, radians
            The (x, y) on-sky position of this object.
        
        Returns
        -------
        source : Source
            The source object with the updated position parameter.
        """
        return eqx.tree_at(
            lambda source : source.position, self, position)
    
    
    def set_spectrum(self : Source, spectrum : Spectrum) -> Source:
        """
        Setter method for the specturm.
        
        Parameters
        ----------
        spectrum : Spectrum
            The spectrum of this object, represented by a Spectrum object.
        
        Returns
        -------
        source : Source
            The source object with the updated spectrum.
        """
        return eqx.tree_at(
            lambda source : source.spectrum, self, spectrum)
    ### End Setter Methods ###
    
    ### General Methods ###
    def normalise(self : Source) -> Source:
        """
        Method for returning a new normalised source object.
        
        Returns
        -------
        source : Source
            The normalised source object.
        """
        normalised_spectrum = self.spectrum.normalise()
        return eqx.tree_at(
            lambda source : source.spectrum, self, normalised_spectrum)
        
    
    def format_inputs(self : Source, filter_in : Filter = None) -> tuple:
        """
        Method for formatting the spectral and positional parameters of the 
        source object to be passed to the model() method.
        
        Parameters
        ----------
        filter_in : Filter (optional)
            The filter through which the source is being observed. Default is 
            None which is uniform throughput.
            
        Returns
        -------
        wavelengths : Array, meters
            The formatted wavelengths array to be passed to model()
        weights : Array
            The formatted weights array to be passed to model()
        positions : Array, radians
            The formatted positions array to be passed to model()
        """
        # Normalise source
        self = self.normalise()
        
        # Get wavelengths
        wavelengths = self.get_wavelengths()
        
        # Get positional info
        positions = self.get_position()
        
        # Get filter throughput
        if filter_in is not None:
            throughput = filter_in.get_throughput(wavelengths) 
        else:
            throughput = np.ones(wavelengths.shape)
        
        # Construct realtive weights
        weights = throughput * self.get_weights() * \
                                np.expand_dims(self.get_flux(), -1)
        
        return wavelengths, weights, positions
    
    @abc.abstractmethod
    def model(self      : Source,
              optics    : Optics,
              detector  : Detector = None,
              filter_in : Filter   = None) -> Array:
        """
        Abstract method to model the psf of the source through the optics.
        
        Parameters
        ----------
        optics : Optics
            The optics through which to model the source objects.
        detector : Detector (optional)
            The detector object that is observing the psf.
        filter_in : Filter (optional)
            The filter through which the source is being observed. Default is 
            None which is uniform throughput.
        
        Returns
        -------
        psf : Array
            The psf of the source source modelled through the optics
        """
        return
    
    
class ResolvedSource(Source, abc.ABC):
    """
    Base class for resolved source objects. This simply extends the base Source
    class by implementing an abstract get_distribution() method and a concrete 
    model() method. 
    """
    
    
    def get_distribution(self):
        """
        Abstract method for returning the distribution of the resolved source.
        
        Returns
        -------
        distribution : Array
            The distribution of the resolved source
        """
        pass
    
    
    def model(self      : Source, 
              optics    : Optics,
              detector  : Detector = None,
              filter_in : Filter = None) -> Array:
        """
        Method to model the psf of the source through the optics. Implements a
        basic convolution with the psf and source distribution.
        
        Parameters
        ----------
        optics : Optics
            The optics through which to model the source objects.
        detector : Detector (optional)
            The detector object that is observing the psf.
        filter_in : Filter (optional)
            The filter through which the source is being observed. Default is 
            None which is uniform throughput.
        
        Returns
        -------
        psf : Array
            The psf of the source modelled through the optics
        """
        # Format imputs
        wavelengths, weights, positions = \
                            self.format_inputs(filter_in=filter_in)
        
        # Vmap propagator
        propagator = jax.vmap(optics.propagate_mono, in_axes=(0, None, 0))
        
        # Model psf
        psf = propagator(wavelengths, positions, weights).sum(0)
        
        # Convolve distribution
        distribution = self.get_distribution()
        psf_out = jsp.signal.convolve(psf, distribution, mode='same')
    
        # Apply detector if supplied
        if detector is None:
            return psf_out
        else:
            return detector.apply_detector(psf_out)
           
           
class RelativeFluxSource(Source, abc.ABC):
    """
    Abstract class that extend the methods of Source to allow for binary-object
    sources to be parameterised by their relative flux. Classes that inherit 
    from this class must instantiate a flux_ratio attribute.
    """
    
    
    def get_flux_ratio(self : Source) -> Array:
        """
        Getter method for the source contrast ratio.
        
        Returns
        -------
        flux_ratio : Array
            The contrast ratio between the two sources.
        """
        return self.flux_ratio
    
    
    def get_flux(self : Source) -> Array:
        """
        Getter method for the fluxes. This paramterieses the source such that 
        flux refers to the mean_flux and flux_ratio is defined as the ratio of 
        the flux of the first entry divided by the second entry.
        
        Returns
        -------
        flux : Array, photons
            The flux (flux1, flux2) of the binary object.
        """
        flux_A = 2 * self.get_flux_ratio() * super().get_flux() / \
                                                    (1 + self.get_flux_ratio())
        flux_B = 2 * super().get_flux() / (1 + self.get_flux_ratio())
        return np.array([flux_A, flux_B])
    
    
    def set_flux_ratio(self : Source, flux_ratio : Array) -> Source:
        """
        Setter method for the source flux ratio.
        
        Parameters
        ----------
        flux_ratio : Array
            The contrast ratio between the two sources.
        
        Returns
        -------
        source : Source
            The source object with updated flux ratio.
        """
        return eqx.tree_at(
           lambda source: source.flux_ratio, self, flux_ratio)
    
    
class RelativePositionSource(Source, abc.ABC):
    """
    Abstract class that extend the methods of Source to allow for binary-object
    sources to be parameterised by their relative position. Classes that
    inherit from this class must instantiate a separation attribute.
    """
    
    
    def get_separation(self : Source) -> Array:
        """
        Getter method for the source separation.
        
        Returns
        -------
        separation : Array, radians
            The separation of the two sources in radians.
        """
        return self.separation
    
    
    def get_field_angle(self : Source) -> Array:
        """
        Getter method for the source field angle.
        
        Returns
        -------
        field_angle : Array, radians
            The field angle between the two sources measure from the positive
            x axis.
        """
        return self.field_angle
    
    
    def get_position(self : Source) -> Array:
        """
        Getter method for the position.
        
        Returns
        -------
        position : Array, radians
            The ((x, y), (x, y)) on-sky position of this object.
        """
        sep_vec = dLux.utils.polar2cart(self.get_separation()/2, 
                                        self.get_field_angle())
        return np.array([super().get_position() + sep_vec, 
                         super().get_position() - sep_vec])

    
    def set_separation(self : Source) -> Source:
        """
        Setter method for the source separation.
        
        Parameters
        ----------
        separation : Array, radians
            The separation of the two sources in radians.
        
        Returns
        -------
        source : Source
            The source object with updated separation.
        """
        return eqx.tree_at(
           lambda source: source.separation, self, separation)
    
    
    def set_field_angle(self : Source) -> Source:
        """
        Setter method for the source field angle.
        
        Parameters
        ----------
        field_angle : Array, radians
            The field angle between the two sources measure from the positive
            x axis.
        
        Returns
        -------
        source : Source
            The source object with updated field angle.
        """
        return eqx.tree_at(
           lambda source: source.field_angle, self, field_angle)
    
    
########################
### Concrete Classes ###
########################
           
class PointSource(Source):
    """
    Concrete Class for unresolved point source objects.
    """
    
    
    def __init__(self     : Source,
                 position : Array,
                 flux     : Array,
                 spectrum : Spectrum,
                 name     : str = 'Point Source') -> Source:
        """
        Parameters
        ----------
        position : Array, radians
            The (x, y) on-sky position of this object. Units are currently in
            radians, but will likely be extended to RA/DEC.
        flux : Array, photons
            The flux of the object.
        spectrum : Spectrum
            The spectrum of this object, represented by a Spectrum object.
        name : str (optional)
            The name for this object. Defaults to 'Point Source'
        """
        super().__init__(position, flux, spectrum, name=name)
        
        
    def model(self      : Source, 
              optics    : Optics,
              detector  : Detector = None,
              filter_in : Filter   = None) -> Array:
        """
        Method to model the psf of the point source through the optics.
        
        Parameters
        ----------
        optics : Optics
            The optics through which to model the source objects.
        detector : Detector (optional)
            The detector object that is observing the psf.
        filter_in : Filter (optional)
            The filter through which the source is being observed. Default is 
            None which is uniform throughput.
        
        Returns
        -------
        psf : Array
            The psf of the source source modelled through the optics.
        """
        # Format imputs
        wavelengths, weights, positions = \
                            self.format_inputs(filter_in=filter_in)
        
        # Vmap propagator
        propagator = jax.vmap(optics.propagate_mono, in_axes=(0, None, 0))
        
        # Model Psf
        psf = propagator(wavelengths, positions, weights).sum(0)
        
        # Apply detector if supplied
        if detector is None:
            return psf
        else:
            return detector.apply_detector(psf)
    
    
class ArrayDistribution(ResolvedSource):
    """
    A class for modelling resolved sources that parameterise their resolved 
    component using an array of intensities.
    
    Attributes
    ----------
    distribution : Array
        The array of intensities respresenting the resolved source.
    """
    distribution : Array
    
    
    def __init__(self         : Source, 
                 position     : Array,
                 flux         : Array,
                 spectrum     : Spectrum,
                 distribution : Array,
                 name         : str = 'ArrayDistribution') -> Source:
        """
        Parameters
        ----------
        position : Array, radians
            The (x, y) on-sky position of this object. Units are currently in
            radians, but will likely be extended to RA/DEC.
        flux : Array, photons
            The flux of the object.
        spectrum : Spectrum
            The spectrum of this object, represented by a Spectrum object.
        distribution : Array
            The array of intensities respresenting the resolved source.
        name : str (optional)
            The name for this object. Defaults to 'Source'
        """
        super().__init__(position, flux, spectrum, name=name)
        distribution = np.asarray(distribution, dtype=float)
        self.distribution = distribution/distribution.sum()
    
    
    def get_distribution(self : Source) -> Array:
        """
        Getter method for the source distribution.
        
        Returns
        -------
        distribution : Array, intensity
            The distribution of the sources intensity.
        """
        return self.distribution
    
    
    def normalise(self : Source):
        """
        Method for returning a new source object with a normalised total
        spectrum and source distribution.
        
        Returns
        -------
        source : Source
            The source object with the normalised spectrum and distribution.
        """
        normalised_spectrum = self.spectrum.normalise()
        normalised_distribution = self.distribution/self.distribution.sum()
        return eqx.tree_at(
            lambda source : (source.spectrum, source.distribution), self, \
                            (normalised_spectrum, normalised_distribution))
    
    
class BinarySource(RelativeFluxSource, RelativePositionSource):
    """
    A parameterised binary source.
    
    Attributes
    ----------
    separation : Array, radians
        The separation of the two sources in radians.
    field_angle : Array, radians
        The field angle between the two sources measure from the positive
        x axis.
    flux_ratio : Array
        The contrast ratio between the two sources.
    """
    separation  : Array
    field_angle : Array
    flux_ratio  : Array
    
    
    def __init__(self        : Source,
                 position    : Array,
                 flux        : Array,
                 separation  : Array,
                 field_angle : Array,
                 flux_ratio  : Array,
                 spectrum    : Spectrum,
                 name        : str = 'Binary Source') -> Source:
        """
        Parameters
        ----------
        position : Array, radians
            The (x, y) on-sky position of this object. Units are currently in
            radians, but will likely be extended to RA/DEC.
        flux : Array, photons
            The mean flux of the sources.
        spectrum : Spectrum
            The spectrum of this object, represented by a Spectrum object.
        separation : Array, radians
            The separation of the two sources in radians.
        field_angle : Array, radians
            The field angle between the two sources measure from the positive
            x axis.
        flux_ratio : Array
            The contrast ratio between the two sources.
        name : str (optional)
            The name for this object. Defaults to 'Binary Source'
        """
        assert isinstance(spectrum, dLux.CombinedSpectrum), "The input spectrum \
        must be a CombinedSpectrum object"
        super().__init__(position, flux, spectrum, name=name)
        
        self.separation  = np.asarray(separation, dtype=float)
        self.field_angle = np.asarray(field_angle, dtype=float)
        self.flux_ratio  = np.asarray(flux_ratio, dtype=float)
    
    
    def model(self      : Source, 
              optics    : Optics,
              detector  : Detector = None,
              filter_in : Filter   = None) -> Array:
        """
        Method to model the psf of the binary source through the optics.
        
        Parameters
        ----------
        optics : Optics
            The optics through which to model the soource objects
        detector : Detector (optional)
            The detector object that is observing the psf.
        filter_in : Filter (optional)
            The filter through which the source is being observed. Default is 
            None which is uniform throughput.
        
        Returns
        -------
        psf : Array
            The psf of the source source modelled through the optics
        """
        # Format imputs
        wavelengths, weights, positions = \
                            self.format_inputs(filter_in=filter_in)
        
        # Vmap propagator
        base_propagator = jax.vmap(optics.propagate_mono, in_axes=(0, None, 0))
        propagator = jax.vmap(base_propagator, in_axes=(0, 0, 0))
        
        # Model Psf
        psf = propagator(wavelengths, positions, weights).sum((0, 1))
    
        # Apply detector if supplied
        if detector is None:
            return psf
        else:
            return detector.apply_detector(psf)
        
        
class PointExtendedSource(ArrayDistribution, RelativeFluxSource):
    """
    A class for modelling a point source and a resolved source that is defined 
    relative to the point source. An example would be an unresolved star with 
    a resolved dust shell or debris disk. These two objects share the same 
    spectra but have their fluxes defined by flux (the mean flux) and the flux
    ratio (contrast) between the point source and resolved distribution. The
    resolved component is defined by an array (ie this class inherits from 
    ArrayDistribution).
    
    Attributes
    ----------
    flux_ratio : Array
        The contrast ratio between the point source and the resolved source.
    """
    flux_ratio : Array
    
    def __init__(self         : Source, 
                 position     : Array,
                 flux         : Array,
                 spectrum     : Spectrum,
                 distribution : Array,
                 flux_ratio   : Array,
                 name         : str = 'PointExtendedSource') -> Source:
        """
        Parameters
        ----------
        position : Array, radians
            The (x, y) on-sky position of this object. Units are currently in
            radians, but will likely be extended to RA/DEC.
        flux : Array, photons
            The flux of the object.
        spectrum : Spectrum
            The spectrum of this object, represented by a Spectrum object.
        distribution : Array
            The array of intensities respresenting the resolved source.
        flux_ratio : Array
            The contrast ratio between the point source and the resolved 
            source.
        name : str (optional)
            The name for this object. Defaults to 'Source'
        """
        super().__init__(position, flux, spectrum, distribution, name=name)
        self.flux_ratio = np.asarray(flux_ratio, dtype=float)

        
    def model(self      : Source, 
              optics    : Optics,
              detector  : Detector = None,
              filter_in : Filter   = None) -> Array:
        """
        Method to model the psf of the source through the optics. Implements a
        basic convolution with the psf and source distribution, while also 
        modelling the single point source psf.
        
        Parameters
        ----------
        optics : Optics
            The optics through which to model the soource objects.
        detector : Detector (optional)
            The detector object that is observing the psf.
        filter_in : Filter (optional)
            The filter through which the source is being observed. Default is 
            None which is uniform throughput.
        
        Returns
        -------
        psf : Array
            The psf of the source modelled through the optics.
        """
        # Format imputs
        wavelengths, weights, positions = \
                            self.format_inputs(filter_in=filter_in)
        
        # Vmap propagator
        propagator = jax.vmap(optics.propagate_mono, in_axes=(0, None))
        
        # Model psfs
        spectral_psf = propagator(wavelengths, positions)
        expanded_weights = np.expand_dims(weights, (-1, -2))
        spectral_psfs = expanded_weights * np.tile(spectral_psf, (2, 1, 1, 1))
        point_psf = spectral_psfs[0].sum(0)
        extended_psf = spectral_psfs[1].sum(0)
        
        # Convolve distribution
        convolved = jsp.signal.convolve(extended_psf, self.get_distribution(),\
                                        mode='same')
        psf = convolved + point_psf
    
        # Apply detector if supplied
        if detector is None:
            return psf
        else:
            return detector.apply_detector(psf)
    
    
class PointAndExtendedSource(ArrayDistribution, RelativeFluxSource):
    """
    A class for modelling a point source and a resolved source that is defined 
    relative to the point source, but with its own spectra. An example would be
    an unresolved quasar within a resolved galaxy. These two objects have 
    independent spectra but have their fluxes defined by flux (the mean flux) 
    and the flux ratio (contrast) between the point source and resolved 
    distribution. The resolved component is defined by an array (ie this class
    inherits from ArrayDistribution).
    
    Attributes
    ----------
    flux_ratio : Array
        The contrast ratio between the point source and the resolved source.
    """
    flux_ratio : Array
    
           
    def __init__(self         : Source, 
                 position     : Array,
                 flux         : Array,
                 spectrum     : Spectrum,
                 distribution : Array,
                 flux_ratio   : Array,
                 name         : str = 'Source') -> Source:
        """
        Parameters
        ----------
        position : Array, radians
            The (x, y) on-sky position of this object. Units are currently in
            radians, but will likely be extended to RA/DEC.
        flux : Array, photons
            The flux of the object.
        spectrum : Spectrum
            The spectrum of this object, represented by a Spectrum object.
        distribution : Array
            The array of intensities respresenting the resolved source.
        flux_ratio : Array
            The contrast ratio between the point source and the resolved 
            source.
        name : str (optional)
            The name for this object. Defaults to 'Source'
        """
        assert isinstance(spectrum, dLux.CombinedSpectrum), "The input spectrum \
        must be a CombinedSpectrum object"
        super().__init__(position, flux, spectrum, distribution, name=name)
        self.flux_ratio = np.asarray(flux_ratio, dtype=float)
    
        
    def model(self      : Source, 
              optics    : Optics,
              detector  : Detector = None,
              filter_in : Filter   = None) -> Array:
        """
        Method to model the psf of the source through the optics. Implements a
        basic convolution with the psf and source distribution, while also 
        modelling the single point source psf. Applied a different spectrum to 
        the point source and resolved source.
        
        Parameters
        ----------
        optics : Optics
            The optics through which to model the source objects.
        detector : Detector (optional)
            The detector object that is observing the psf.
        filter_in : Filter (optional)
            The filter through which the source is being observed. Default is 
            None which is uniform throughput.
        
        Returns
        -------
        psf : Array
            The psf of the source modelled through the optics.
        """
        # Format imputs
        wavelengths, weights, positions = \
                            self.format_inputs(filter_in=filter_in)
        
        # Vmap propagator
        propagator = jax.vmap(optics.propagate_mono, in_axes=(0, None))
        
        # Model psfs
        spectral_psf = propagator(wavelengths[0], positions)
        expanded_weights = np.expand_dims(weights, (-1, -2))
        spectral_psfs = expanded_weights * np.tile(spectral_psf, (2, 1, 1, 1))
        point_psf = spectral_psfs[0].sum(0)
        extended_psf = spectral_psfs[1].sum(0)
        
        # Convolve distribution
        convolved = jsp.signal.convolve(extended_psf, self.get_distribution(),\
                                        mode='same')
        psf = convolved + point_psf
    
        # Apply detector if supplied
        if detector is None:
            return psf
        else:
            return detector.apply_detector(psf)