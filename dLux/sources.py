from __future__ import annotations
import jax.numpy as np
from jax.scipy.signal import convolve
from jax import vmap
from equinox import tree_at, static_field
from abc import ABC, abstractmethod
import dLux


__all__ = ["PointSource", "MultiPointSource", "ArrayDistribution",
           "BinarySource", "PointExtendedSource", "PointAndExtendedSource"]


Array = np.ndarray


########################
### Abstract Classes ###
########################
class Source(dLux.base.ExtendedBase, ABC):
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
                 position : Array    = None,
                 flux     : Array    = None,
                 spectrum : Spectrum = None,
                 name     : str      = 'Source') -> Source:
        """
        Constructor for the Source class.

        Parameters
        ----------
        position : Array, radians = None
            The (x, y) on-sky position of this object.
        flux : Array, photons = None
            The flux of the object.
        spectrum : Spectrum = None
            The spectrum of this object, represented by a Spectrum object.
        name : str = 'Source'
            The name for this object.
        """
        self.position = np.asarray(position, dtype=float)
        self.flux     = np.asarray(flux,     dtype=float)
        self.spectrum = spectrum
        self.name     = name

        # Input position checking
        if position is not None:
            assert self.position.ndim == 1, \
            ("position must be a 1d array.")
            assert self.position.shape == (2,), \
            ("positions must be shape (2,), ie (x, y).")

        # Input flux checking
        if flux is not None:
            assert self.flux.shape == (), \
            ("flux must be a scalar, (shape == ()).")

        # Input spectrum checking
        assert isinstance(self.spectrum, dLux.spectrums.Spectrum), \
        ("Spectrum must be dLux Spectrum object.")

        # Input name checking
        assert isinstance(self.name, str), "Name must be a string."


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
        detector : Detector = None
            The detector object that is observing the psf.
        filter_in : Filter = None
            The filter through which the source is being observed.

        Returns
        -------
        psf : Array
            The psf of the source source modelled through the optics.
        """
        # Normalise and get parameters
        self        = self.normalise()
        wavelengths = self.get_wavelengths()
        weights     = self.get_weights()
        position    = self.get_position()
        flux        = self.get_flux()

        # Get filter throughput
        if filter_in is not None:
            raise NotImplementedError("Filter modelling is under development.")

        # Vmap propagator & model
        propagator = vmap(optics.propagate_mono, in_axes=(0, None))
        psfs = weights[:, None, None] * propagator(wavelengths, position)
        psf = flux * psfs.sum(0)

        # Apply detector if supplied
        return psf if detector is None else detector.apply_detector(psf)


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
        detector : Detector = None
            The detector object that is observing the psf.
        filter_in : Filter = None
            The filter through which the source is being observed.

        Returns
        -------
        psf : Array
            The psf of the source modelled through the optics
        """
        # Normalise and get parameters
        self         = self.normalise()
        wavelengths  = self.get_wavelengths()
        weights      = self.get_weights()
        position     = self.get_position()
        flux         = self.get_flux()
        distribution = self.get_distribution()

        # Get filter throughput
        if filter_in is not None:
            raise NotImplementedError("Filter modelling is under development.")

        # Vmap propagator & model
        propagator = vmap(optics.propagate_mono, in_axes=(0, None))
        psfs = weights[:, None, None] * propagator(wavelengths, position)
        psf = convolve(flux * psfs.sum(0), distribution, mode='same')

        # Apply detector if supplied
        return psf if detector is None else detector.apply_detector(psf)


class RelativeFluxSource(Source, ABC):
    """
    Abstract class that extend the methods of Source to allow for binary-object
    sources to be parameterised by their relative flux. Classes that inherit
    from this class must instantiate a contrast attribute.

    Attributes
    ----------
    contrast : Array
        The contrast ratio between the two sources.
    """
    contrast : Array


    def __init__(self      : Source,
                 contrast : Array,
                 **kwargs) -> Source:
        """
        Constructor for the RelativeFluxSource class.

        Parameters
        ----------
        contrast : Array
            The contrast ratio between the two sources.
        """
        super().__init__(**kwargs)
        self.contrast = np.asarray(contrast, dtype=float)

        # Input contrast checking
        assert self.contrast.shape == (), \
        ("Flux ratio must be a scalar, (shape == ()).")


    def get_flux(self : Source) -> Array:
        """
        Getter method for the fluxes. This paramterieses the source such that
        flux refers to the mean_flux and contrast is defined as the ratio of
        the flux of the first entry divided by the second entry.

        Returns
        -------
        flux : Array, photons
            The flux (flux1, flux2) of the binary object.
        """
        flux_A = 2 * self.contrast * self.flux / (1 + self.contrast)
        flux_B = 2 * self.flux / (1 + self.contrast)
        return np.array([flux_A, flux_B])


class RelativePositionSource(Source, ABC):
    """
    Abstract class that extend the methods of Source to allow for binary-object
    sources to be parameterised by their relative position. Classes that
    inherit from this class must instantiate a separation attribute.

    Attributes
    ----------
    separation : Array, radians
        The separation of the two sources in radians.
    position_angle : Array, radians
        The field angle between the two sources measure from the positive
        x axis.
    """
    separation     : Array
    position_angle : Array


    def __init__(self           : Source,
                 separation     : Array,
                 position_angle : Array,
                 **kwargs) -> Source:
        """
        Constructor for the RelativePositionSource class.

        Parameters
        ----------
        separation : Array, radians
            The separation of the two sources in radians.
        position_angle : Array, radians
            The field angle between the two sources measure from the positive
            x axis.
        """
        super().__init__(**kwargs)
        self.separation     = np.asarray(separation,     dtype=float)
        self.position_angle = np.asarray(position_angle, dtype=float)

        assert self.separation.shape == (), "Separation must be a scalar, \
        (shape == ())."

        assert self.position_angle.shape == (), "Field angle must be a scalar, \
        (shape == ())."


    def get_position(self : Source) -> Array:
        """
        Getter method for the position.

        Returns
        -------
        position : Array, radians
            The ((x, y), (x, y)) on-sky position of this object.
        """
        r, phi = self.separation/2, self.position_angle
        sep_vec = np.array([r*np.sin(phi), r*np.cos(phi)])
        return np.array([self.position + sep_vec,
                         self.position - sep_vec])


########################
### Concrete Classes ###
########################
class PointSource(Source):
    """
    Concrete Class for unresolved point source objects.
    """


    def __init__(self        : Source,
                 position    : Array    = np.array([0., 0.]),
                 flux        : Array    = np.array(1.),
                 spectrum    : Spectrum = None,
                 wavelengths : Array    = None,
                 name        : str      = 'PointSource') -> Source:
        """
        Constructor for the PointSource class.

        Parameters
        ----------
        position : Array, radians = np.array([0., 0.])
            The (x, y) on-sky position of this object.
        flux : Array, photons = np.array(1.)
            The flux of the object.
        spectrum : Spectrum = None
            The spectrum of this object, represented by a Spectrum object.
        wavelengths : Array, meters = None
            The array of wavelengths at which the spectrum is defined.
        name : str = 'PointSource'
            The name for this object.
        """
        # Check spectrum & wavelengths
        assert spectrum is not None or wavelengths is not None, \
        ("Either spectrum or wavelengths must be specified.")
        if spectrum is not None:
            assert wavelengths is None, \
            ("wavelengths can not be specified if spectrum is specified.")
        else:
            spectrum = dLux.spectrums.ArraySpectrum(wavelengths)
        super().__init__(position, flux, spectrum, name=name)


class MultiPointSource(Source):
    """
    Concrete Class for multiple unresolved point source objects.
    """


    def __init__(self        : Source,
                 position    : Array,
                 flux        : Array    = None,
                 spectrum    : Spectrum = None,
                 wavelengths : Array    = None,
                 name        : str      = 'MultiPointSource') -> Source:
        """
        Constructor for the MultiPointSource class.

        Parameters
        ----------
        position : Array, radians
            The ((x0, y0), (x1, y1), ...) on-sky positions of these sourcese.
        flux : Array, photons = None
            The fluxes of the sources.
        spectrum : Spectrum = None
            The spectrum of this object, represented by a Spectrum object.
            Every source in this class will have an identical spectrum.
        wavelengths : Array, meters = None
            The array of wavelengths at which the spectrum is defined.
        name : str = 'MultiPointSource'
            The name for this object.
        """
        # Check spectrum & wavelengths
        assert spectrum is not None or wavelengths is not None, \
        ("Either spectrum or wavelengths must be specified.")
        if spectrum is not None:
            assert wavelengths is None, \
            ("wavelengths can not be specified if spectrum is specified.")
        else:
            spectrum = dLux.spectrums.ArraySpectrum(wavelengths)

        # Only call super, not __init__ since we are overwriting all the attrs
        super().__init__(spectrum=spectrum, name=name)

        self.position = np.asarray(position, dtype=float)

        # Input position checking
        assert self.position.ndim == 2, \
        ("position must be a 2d array.")
        assert self.position.shape[-1] == 2, \
        ("positions must be shape (nstars, 2), ie [(x0, y0), (x1, y1), ...].")

        # Get flux
        self.flux = np.ones(len(self.positions)) if flux is None else \
                    np.asarray(flux, dtype=float)

        # Input flux checking
        assert self.flux.ndim == 1, \
        ("flux must be a 1d array.")

        # Ensure same dimensionality
        assert self.flux.shape[0] == self.position.shape[0], ("position and "
        "flux must have the same length leading dimension, ie nstars")


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
        detector : Detector = None
            The detector object that is observing the psf.
        filter_in : Filter = None
            The filter through which the source is being observed.

        Returns
        -------
        psf : Array
            The psf of the source source modelled through the optics.
        """
        # Normalise and get parameters
        self        = self.normalise()
        wavelengths = self.get_wavelengths()
        weights     = self.get_weights()
        positions   = self.get_position()
        fluxes      = self.get_flux()

        # Get filter throughput
        if filter_in is not None:
            raise NotImplementedError("Filter modelling is under development.")

        # Vmap propagator
        source_propagator = vmap(optics.propagate_mono, in_axes=(0, None))
        propagator = vmap(source_propagator, in_axes=(None, 0))

        # Model Psf
        psfs = propagator(wavelengths, positions)
        psfs *= weights[None, :, None, None] * fluxes[:, None, None, None]
        psf = psfs.sum((0, 1))

        # Apply detector if supplied
        return psf if detector is None else detector.apply_detector(psf)


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
                 position     : Array    = np.array([0., 0.]),
                 flux         : Array    = np.array(1.),
                 distribution : Array    = np.ones((3, 3)),
                 spectrum     : Spectrum = None,
                 wavelengths  : Array    = None,
                 name         : str      = 'ArrayDistribution',
                 **kwargs) -> Source:
        """
        Constructor for the ArrayDistribution class.

        Parameters
        ----------
        position : Array, radians = np.array([0., 0.])
            The (x, y) on-sky position of this object.
        flux : Array, photons = np.array(1.)
            The flux of the object.
        distribution : Array = np.ones((3, 3))
            The array of intensities respresenting the resolved source.
        spectrum : Spectrum = None
            The spectrum of this object, represented by a Spectrum object.
        wavelengths : Array, meters = None
            The array of wavelengths at which the spectrum is defined.
        name : str = 'ArrayDistribution'
            The name for this object.
        """
        # Check spectrum & wavelengths
        assert spectrum is not None or wavelengths is not None, \
        ("Either spectrum or wavelengths must be specified.")
        if spectrum is not None:
            assert wavelengths is None, \
            ("wavelengths can not be specified if spectrum is specified.")
        else:
            spectrum = dLux.spectrums.ArraySpectrum(wavelengths)

        super().__init__(position, flux, spectrum, name=name, **kwargs)
        distribution = np.asarray(distribution, dtype=float)
        self.distribution = distribution/distribution.sum()

        # Input checking
        assert self.distribution.ndim == 2, \
        ("distribution must be a 2d array.")
        assert len(self.distribution) > 0, \
        ("Length of distribution must be greater than 1.")


    def get_distribution(self : Source) -> Array:
        """
        Getter method for the source distribution.

        Returns
        -------
        distribution : Array, intensity
            The distribution of the source intensity.
        """
        return self.distribution


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


    def __init__(self           : Source,
                 position       : Array    = np.array([0., 0.]),
                 flux           : Array    = np.array(1.),
                 separation     : Array    = None,
                 position_angle : Array    = np.pi/2,
                 contrast       : Array    = np.array(1.),
                 spectrum       : Spectrum = None,
                 wavelengths    : Array    = None,
                 name           : str      = 'BinarySource') -> Source:
        """
        Parameters
        ----------
        position : Array, radians = np.array([0., 0.])
            The (x, y) on-sky position of this object.
        flux : Array, photons = np.array(1.)
            The mean flux of the sources.
        separation : Array, radians = None
            The separation of the two sources in radians.
        position_angle : Array, radians = np.pi/2
            The position angle between the two sources measured clockwise from
            the vertical axis.
        contrast : Array = np.array(1.)
            The contrast ratio between the two sources.
        spectrum : CombinedSpectrum = None
            The spectrum of this object, represented by a CombinedSpectrum.
        wavelengths : Array, meters = None
            The array of wavelengths at which the spectrum is defined.
        name : str = 'BinarySource'
            The name for this object.
        """
        # Check spectrum & wavelengths
        assert spectrum is not None or wavelengths is not None, \
        ("Either spectrum or wavelengths must be specified.")
        if spectrum is not None:
            assert wavelengths is None, \
            ("wavelengths can not be specified if spectrum is specified.")
            assert isinstance(spectrum, dLux.CombinedSpectrum), \
            ("The input spectrum must be a CombinedSpectrum object.")
        else:
            spectrum = dLux.spectrums.CombinedSpectrum(wavelengths)

        # Check separation
        assert separation is not None, ("separation must be provided.")

        super().__init__(position=position,
                         flux=flux,
                         spectrum=spectrum,
                         separation=separation,
                         position_angle=position_angle,
                         contrast=contrast,
                         name=name)


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
        detector : Detector = None
            The detector object that is observing the psf.
        filter_in : Filter = None
            The filter through which the source is being observed.

        Returns
        -------
        psf : Array
            The psf of the source source modelled through the optics.
        """
        # Normalise and get parameters
        self        = self.normalise()
        wavelengths = self.get_wavelengths()[0]
        weights     = self.get_weights()
        positions   = self.get_position()
        fluxes      = self.get_flux()

        # Get filter throughput
        if filter_in is not None:
            raise NotImplementedError("Filter modelling is under development.")

        # Vmap propagator
        source_propagator = vmap(optics.propagate_mono, in_axes=(0, None))
        propagator = vmap(source_propagator, in_axes=(None, 0))

        # Model Psf
        psfs = propagator(wavelengths, positions)
        psfs *= weights[:, :, None, None] * fluxes[:, None, None, None]
        psf = psfs.sum((0, 1))

        # Apply detector if supplied
        return psf if detector is None else detector.apply_detector(psf)


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
                 position     : Array    = np.array([0., 0.]),
                 flux         : Array    = np.array(1.),
                 distribution : Array    = np.ones((3, 3)),
                 contrast     : Array    = np.array(1.),
                 spectrum     : Spectrum = None,
                 wavelengths  : Array    = None,
                 name         : str      = 'PointExtendedSource') -> Source:
        """
        Parameters
        ----------
        position : Array, radians = np.array([0., 0.])
            The (x, y) on-sky position of this object.
        flux : Array, photons = np.array(1.)
            The flux of the object.
        distribution : Array = np.ones((3, 3))
            The array of intensities respresenting the resolved source.
        contrast : Array = np.array(1.)
            The contrast ratio between the point source and the resolved
            source.
        spectrum : Spectrum = None
            The spectrum of this object, represented by a Spectrum object.
        wavelengths : Array, meters = None
            The array of wavelengths at which the spectrum is defined.
        name : str = 'PointExtendedSource'
            The name for this object.
        """
        # Check spectrum & wavelengths
        assert spectrum is not None or wavelengths is not None, \
        ("Either spectrum or wavelengths must be specified.")
        if spectrum is not None:
            assert wavelengths is None, \
            ("wavelengths can not be specified if spectrum is specified.")
        else:
            spectrum = dLux.spectrums.Spectrum(wavelengths)

        super().__init__(position=position, flux=flux, spectrum=spectrum, \
                         distribution=distribution, contrast=contrast, \
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
        detector : Detector = None
            The detector object that is observing the psf.
        filter_in : Filter = None
            The filter through which the source is being observed. Default is 
            None which is uniform throughput.

        Returns
        -------
        psf : Array
            The psf of the source modelled through the optics.
        """
        # Normalise and get parameters
        self         = self.normalise()
        wavelengths  = self.get_wavelengths()
        weights      = self.get_weights()
        position     = self.get_position()
        fluxes       = self.get_flux()
        distribution = self.get_distribution()

        # Get filter throughput
        if filter_in is not None:
            raise NotImplementedError("Filter modelling is under development.")

        # Vmap propagator
        propagator = vmap(optics.propagate_mono, in_axes=(0, None))

        # Model psfs
        psfs = propagator(wavelengths, position)
        single_psf = (weights[:, None, None] * psfs).sum(0)
        point_psf = fluxes[0] * single_psf
        extended_psf = fluxes[1] * single_psf
        convolved = convolve(extended_psf, distribution, mode='same')
        psf = convolved + point_psf

        # Apply detector if supplied
        return psf if detector is None else detector.apply_detector(psf)



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
                 position     : Array    = np.array([0., 0.]),
                 flux         : Array    = np.array(1.),
                 distribution : Array    = np.ones((3, 3)),
                 contrast     : Array    = np.array(1.),
                 spectrum     : Spectrum = None,
                 wavelengths  : Array    = None,
                 name         : str      = 'PointAndExtendedSource') -> Source:
        """
        Parameters
        ----------
        position : Array, radians = np.array([0., 0.])
            The (x, y) on-sky position of this object.
        flux : Array, photons = np.array(1.)
            The flux of the object.
        distribution : Array = np.ones((3, 3))
            The array of intensities respresenting the resolved source.
        contrast : Array = np.array(1.)
            The contrast ratio between the point source and the resolved
            source.
        spectrum : CombinedSpectrum = None
            The spectrum of this object, represented by a CombinedSpectrum.
        wavelengths : Array, meters = None
            The array of wavelengths at which the spectrum is defined.
        name : str = 'PointAndExtendedSource'
            The name for this object.
        """
        # Check spectrum & wavelengths
        assert spectrum is not None or wavelengths is not None, \
        ("Either spectrum or wavelengths must be specified.")
        if spectrum is not None:
            assert wavelengths is None, \
            ("wavelengths can not be specified if spectrum is specified.")
            assert isinstance(spectrum, dLux.CombinedSpectrum), \
            ("The input spectrum must be a CombinedSpectrum object.")
        else:
            spectrum = dLux.spectrums.CombinedSpectrum(wavelengths)

        super().__init__(position=position, flux=flux, spectrum=spectrum, \
                         distribution=distribution, contrast=contrast, \
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
        detector : Detector = None
            The detector object that is observing the psf.
        filter_in : Filter = None
            The filter through which the source is being observed. Default is
            None which is uniform throughput.

        Returns
        -------
        psf : Array
            The psf of the source modelled through the optics.
        """
        # Normalise and get parameters
        self         = self.normalise()
        wavelengths  = self.get_wavelengths()[0]
        weights      = self.get_weights()
        position     = self.get_position()
        fluxes       = self.get_flux()
        distribution = self.get_distribution()

        # Get filter throughput
        if filter_in is not None:
            raise NotImplementedError("Filter modelling is under development.")

        # Vmap propagator
        propagator = vmap(optics.propagate_mono, in_axes=(0, None))

        # Model psfs
        psfs = propagator(wavelengths, position)
        point_psf    = fluxes[0] * (weights[0, :, None, None] * psfs).sum(0)
        extended_psf = fluxes[1] * (weights[1, :, None, None] * psfs).sum(0)
        convolved = convolve(extended_psf, distribution, mode='same')
        psf = convolved + point_psf

        # Apply detector if supplied
        return psf if detector is None else detector.apply_detector(psf)