from __future__ import annotations
import jax.numpy as np
from jax.scipy.signal import convolve
from jax import vmap
from equinox import tree_at, static_field
from abc import ABC, abstractmethod
import dLux


__all__ = ["PointSource", "ArrayDistribution", "BinarySource",
           "PointExtendedSource", "PointAndExtendedSource"]


Array = np.ndarray


"""
If you are confused about the class inheritance, please read this stack
overflow post on diamond inheritance in python, where each class instantiates
parameters: https://stackoverflow.com/questions/34884567/python-multiple-inheritance-passing-arguments-to-constructors-using-super
"""

########################
### Abstract Classes ###
########################
class Source(dLux.base.Base, ABC):
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
        The (x, y) on-sky position of this object.
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
    name     : str = static_field()
    
    
    def __init__(self     : Source,
                 position : Array,
                 flux     : Array,
                 spectrum : Spectrum,
                 name     : str = 'Source') -> Source:
        """
        Parameters
        ----------
        position : Array, radians
            The (x, y) on-sky position of this object.
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
        
        # Input position checking
        assert self.position.ndim == 1, \
        ("position must be a 1d array.")
        assert self.position.shape == (2,), \
        ("positions must be shape (2,), ie (x, y).")
        assert not np.isnan(self.position).any(), \
        ("position must not be nan.")
        assert not np.isinf(self.position).any(), \
        ("position must be not be infinite.")
        
        # Input flux checking
        assert self.flux.shape == (), \
        ("flux must be a scalar, (shape == ()).")
        assert not np.isnan(self.flux).any(), \
        ("flux must not be nan.")
        assert not np.isinf(self.flux).any(), \
        ("flux must be not be infinite.")
        
        # Input spectrum checking
        assert isinstance(self.spectrum, dLux.spectrums.Spectrum), \
        ("Spectrum must be dLux Spectrum object.")
        
        # Input name checking
        assert isinstance(self.name, str), "Name must be a string."
    
    
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
        assert isinstance(flux, Array) and flux.ndim == 0, \
        ("flux must be a scalar array.")
        return tree_at(
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
        assert isinstance(position, Array) and position.shape == (2,), \
        ("position must be a array of shape (2,), ie (x, y).")
        return tree_at(
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
        assert isinstance(spectrum, dLux.spectrums.Spectrum), \
        ("spectrum must be a dLux.spectrums.Spectrum object.")
        return tree_at(
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
        return tree_at(
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
    
    
    @abstractmethod
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
    
    
class ResolvedSource(Source, ABC):
    """
    Base class for resolved source objects. This simply extends the base Source
    class by implementing an abstract get_distribution() method and a concrete 
    model() method.
    """
    
    
    @abstractmethod
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
              filter_in : Filter   = None) -> Array:
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
        propagator = vmap(optics.propagate_mono, in_axes=(0, None, 0))
        
        # Model psf
        psf = propagator(wavelengths, positions, weights).sum(0)
        
        # Convolve distribution
        distribution = self.get_distribution()
        psf_out = convolve(psf, distribution, mode='same')
    
        # Apply detector if supplied
        if detector is None:
            return psf_out
        else:
            return detector.apply_detector(psf_out)
    
    
class RelativeFluxSource(Source, ABC):
    """
    Abstract class that extend the methods of Source to allow for binary-object
    sources to be parameterised by their relative flux. Classes that inherit 
    from this class must instantiate a flux_ratio attribute.
    
    Attributes
    ----------
    flux_ratio : Array
        The contrast ratio between the two sources.
    """
    flux_ratio : Array
    
    
    def __init__(self       : Source,
                 flux_ratio : Array,
                 **kwargs) -> Source:
        """
        Parameters
        ----------
        flux_ratio : Array
            The contrast ratio between the two sources.
        """
        super().__init__(**kwargs)
        self.flux_ratio = np.asarray(flux_ratio, dtype=float)
        
        # Input flux_ratio checking
        assert self.flux_ratio.shape == (), \
        ("Flux ratio must be a scalar, (shape == ()).")
        assert not np.isnan(self.flux_ratio).any(), \
        ("flux_ratio must not be nan.")
        assert not np.isinf(self.flux_ratio).any(), \
        ("flux_ratio must be not be infinite.")
    
    
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
        assert isinstance(flux_ratio, Array) and flux_ratio.ndim == 0, \
        ("flux_ratio must be a scalar array.")
        return tree_at(
           lambda source: source.flux_ratio, self, flux_ratio)
    
    
class RelativePositionSource(Source, ABC):
    """
    Abstract class that extend the methods of Source to allow for binary-object
    sources to be parameterised by their relative position. Classes that
    inherit from this class must instantiate a separation attribute.
    
    Attributes
    ----------
    separation : Array, radians
        The separation of the two sources in radians.
    field_angle : Array, radians
        The field angle between the two sources measure from the positive
        x axis.
    """
    separation  : Array
    field_angle : Array
    
    
    def __init__(self        : Source,
                 separation  : Array,
                 field_angle : Array,
                 **kwargs) -> Source:
        """
        Parameters
        ----------
        separation : Array, radians
            The separation of the two sources in radians.
        field_angle : Array, radians
            The field angle between the two sources measure from the positive
            x axis.
        """
        super().__init__(**kwargs)
        self.separation  = np.asarray(separation,  dtype=float)
        self.field_angle = np.asarray(field_angle, dtype=float)
        
        assert self.separation.shape == (), "Separation must be a scalar, \
        (shape == ())."
        
        assert self.field_angle.shape == (), "Field angle must be a scalar, \
        (shape == ())."
    
    
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
    
    
    def set_separation(self : Source, separation : Array) -> Source:
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
        assert isinstance(separation, Array) and separation.ndim == 0, \
        ("separation must be a scalar array.")
        return tree_at(
           lambda source: source.separation, self, separation)
    
    
    def set_field_angle(self : Source, field_angle : Array) -> Source:
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
        assert isinstance(field_angle, Array) and field_angle.ndim == 0, \
        ("field_angle must be a scalar array.")
        return tree_at(
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
            The (x, y) on-sky position of this object.
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
        propagator = vmap(optics.propagate_mono, in_axes=(0, None, 0))
        
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
                 name         : str = 'ArrayDistribution',
                 **kwargs) -> Source:
        """
        Parameters
        ----------
        position : Array, radians
            The (x, y) on-sky position of this object.
        flux : Array, photons
            The flux of the object.
        spectrum : Spectrum
            The spectrum of this object, represented by a Spectrum object.
        distribution : Array
            The array of intensities respresenting the resolved source.
        name : str (optional)
            The name for this object. Defaults to 'Source'
        """
        super().__init__(position, flux, spectrum, name=name, **kwargs)
        distribution = np.asarray(distribution, dtype=float)
        self.distribution = distribution/distribution.sum()
        
        # Input checking
        assert self.distribution.ndim == 2, \
        ("distribution must be a 2d array.")
        assert len(self.distribution) > 0, \
        ("Length of distribution must be greater than 1.")
        assert not np.isnan(self.distribution).any(), \
        ("distribution must not be nan.")
        assert not np.isinf(self.distribution).any(), \
        ("distribution must be not be infinite.")
    
    
    def get_distribution(self : Source) -> Array:
        """
        Getter method for the source distribution.
        
        Returns
        -------
        distribution : Array, intensity
            The distribution of the source intensity.
        """
        return self.distribution
    
    
    def set_distribution(self : Source, distribution : Array) -> Source:
        """
        Setter method for the source distribution.
        
        Parameters
        ----------
        distribution : Array, intensity
            The distribution of the source intensity.
        
        Returns
        -------
        source : Source
            The source object with updated distribution.
        """
        assert isinstance(distribution, Array) and distribution.ndim == 2, \
        ("distribution must be a 2d array.")
        return tree_at(
           lambda source: source.distribution, self, distribution)
    
    
    def normalise(self : Source) -> Source:
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
        return tree_at(
            lambda source : (source.spectrum, source.distribution), self, \
                            (normalised_spectrum, normalised_distribution))
    
    
class BinarySource(RelativePositionSource, RelativeFluxSource):
    """
    A parameterised binary source.
    """
    
    
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
            The (x, y) on-sky position of this object.
        flux : Array, photons
            The mean flux of the sources.
        spectrum : CombinedSpectrum
            The spectrum of this object, represented by a CombinedSpectrum \
            object.
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
        assert isinstance(spectrum, dLux.CombinedSpectrum), \
        ("The input spectrum must be a CombinedSpectrum object.")
        
        super().__init__(position=position,
                         flux=flux,
                         spectrum=spectrum,
                         separation=separation,
                         field_angle=field_angle,
                         flux_ratio=flux_ratio,
                         name=name)
    
    
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
        base_propagator = vmap(optics.propagate_mono, in_axes=(0, None, 0))
        propagator = vmap(base_propagator, in_axes=(0, 0, 0))
        
        # Model Psf
        psf = propagator(wavelengths, positions, weights).sum((0, 1))
    
        # Apply detector if supplied
        if detector is None:
            return psf
        else:
            return detector.apply_detector(psf)
    
    
class PointExtendedSource(RelativeFluxSource, ArrayDistribution):
    """
    A class for modelling a point source and a resolved source that is defined 
    relative to the point source. An example would be an unresolved star with 
    a resolved dust shell or debris disk. These two objects share the same 
    spectra but have their fluxes defined by flux (the mean flux) and the flux
    ratio (contrast) between the point source and resolved distribution. The
    resolved component is defined by an array (ie this class inherits from 
    ArrayDistribution).
    """
    
    
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
            The (x, y) on-sky position of this object.
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
            The name for this object. Defaults to 'PointExtendedSource'
        """
        super().__init__(position=position, flux=flux, spectrum=spectrum, \
                         distribution=distribution, flux_ratio=flux_ratio, \
                         name=name)
        
        
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
        propagator = vmap(optics.propagate_mono, in_axes=(0, None))
        
        # Model psfs
        spectral_psf = propagator(wavelengths, positions)
        expanded_weights = np.expand_dims(weights, (-1, -2))
        spectral_psfs = expanded_weights * np.tile(spectral_psf, (2, 1, 1, 1))
        point_psf = spectral_psfs[0].sum(0)
        extended_psf = spectral_psfs[1].sum(0)
        
        # Convolve distribution
        convolved = convolve(extended_psf, self.get_distribution(), \
                                        mode='same')
        psf = convolved + point_psf
    
        # Apply detector if supplied
        if detector is None:
            return psf
        else:
            return detector.apply_detector(psf)
    
    
class PointAndExtendedSource(RelativeFluxSource, ArrayDistribution):
    """
    A class for modelling a point source and a resolved source that is defined 
    relative to the point source, but with its own spectra. An example would be
    an unresolved quasar within a resolved galaxy. These two objects have 
    independent spectra but have their fluxes defined by flux (the mean flux) 
    and the flux ratio (contrast) between the point source and resolved 
    distribution. The resolved component is defined by an array (ie this class
    inherits from ArrayDistribution).
    """
    
    
    def __init__(self         : Source,
                 position     : Array,
                 flux         : Array,
                 spectrum     : Spectrum,
                 distribution : Array,
                 flux_ratio   : Array,
                 name         : str = 'PointAndExtendedSource') -> Source:
        """
        Parameters
        ----------
        position : Array, radians
            The (x, y) on-sky position of this object.
        flux : Array, photons
            The flux of the object.
        spectrum : CombinedSpectrum
            The spectrum of this object, represented by a CombinedSpectrum \
            object.
        distribution : Array
            The array of intensities respresenting the resolved source.
        flux_ratio : Array
            The contrast ratio between the point source and the resolved 
            source.
        name : str (optional)
            The name for this object. Defaults to 'PointAndExtendedSource'
        """
        assert isinstance(spectrum, dLux.CombinedSpectrum), \
        ("The input spectrum must be a CombinedSpectrum object.")
        
        super().__init__(position=position, flux=flux, spectrum=spectrum, \
                         distribution=distribution, flux_ratio=flux_ratio, \
                         name=name)
    
    
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
        propagator = vmap(optics.propagate_mono, in_axes=(0, None))
        
        # Model psfs
        spectral_psf = propagator(wavelengths[0], positions)
        expanded_weights = np.expand_dims(weights, (-1, -2))
        spectral_psfs = expanded_weights * np.tile(spectral_psf, (2, 1, 1, 1))
        point_psf = spectral_psfs[0].sum(0)
        extended_psf = spectral_psfs[1].sum(0)
        
        # Convolve distribution
        convolved = convolve(extended_psf, self.get_distribution(),\
                                        mode='same')
        psf = convolved + point_psf
    
        # Apply detector if supplied
        if detector is None:
            return psf
        else:
            return detector.apply_detector(psf)