from __future__ import annotations
from typing import Any
import jax.numpy as np
from jax.scipy.signal import convolve
from jax import vmap, Array
from zodiax import Base
from abc import abstractmethod
import dLux


__all__ = [
    "PointSource",
    "PointSources",
    "BinarySource",
    "ResolvedSource",
    "PointResolvedSource",
]


Spectrum = lambda: dLux.spectra.Spectrum
Optics = lambda: dLux.optics.BaseOptics


###################
# Private Classes #
###################
class BaseSource(Base):
    # TODO: Add this to allow custom sources

    @abstractmethod
    def normalise(self):  # pragma: no cover
        pass

    @abstractmethod
    def model(self, optics, detector=None):  # pragma: no cover
        pass


class Source(BaseSource):
    """
    Base class for source objects. The idea of these source classes is to allow
    an arbitrary parametrisation of the underlying astrophysical objects. Each
    source object requires a normalise(), format_inputs(), and model() methods,
    along with the regular getter and setter methods.

    The normalise() method should return a new instance of the class with all
    the appropriate attributes normalised (ie spectral weights, resolved source
    distribution).

    The format_inputs() method should return the relevant wavelengths, weights,
    positions etc., that are correctly formatted to be used by the model()
    method. This should primarily interface with the spectrum object, and
    should call the normalise() method.

    The model() method should return a single psf of the source. This is done
    inside the source object so that each class can be arbitrarily
    parameterised as required. This method should call the format_inputs()
    method.

    A series of parametrisations have been provided to model a series of
    different astrophysical objects.

    Attributes
    ----------
    position : Array, radians
        The (x, y) on-sky position of this object.
    flux : Array, photons
        The flux of the object.
    spectrum : Spectrum
        The spectrum of this object, represented by a Spectrum object.
    """

    position: Array
    flux: Array
    spectrum: Spectrum

    def __init__(
        self: Source,
        wavelengths: Array,
        position: Array = np.zeros(2),
        flux: Array = np.array(1.0),
        weights: Array = None,
        spectrum: Spectrum = None,
    ):
        """
        Constructor for the Source class.

        Parameters
        ----------
        wavelengths : Array, metres
            The array of wavelengths at which the spectrum is defined. Defaults
            to a PointSource with a flat spectrum.
        position : Array, radians = None
            The (x, y) on-sky position of this object.
        flux : Array, photons = None
            The flux of the object.
        weights : Array = None
            The spectral weights of the object.
        spectrum : Spectrum = None
            The spectrum of this object, represented by a Spectrum object.
            if provided it will overwrite the inputs wavelengths and weights.

        """
        # Position and Flux
        self.position = np.asarray(position, dtype=float)
        self.flux = np.asarray(flux, dtype=float)

        if self.position.shape != (2,):
            raise ValueError("position must be a 1d array of shape (2,).")

        if self.flux.shape != ():
            raise ValueError("flux must be a scalar, i.e. shape == ().")

        # Spectrum
        if spectrum is not None:
            if not isinstance(spectrum, dLux.spectra.Spectrum):
                raise ValueError("spectrum must be a dLux Spectrum object.")
            self.spectrum = spectrum
        else:
            self.spectrum = dLux.spectra.Spectrum(wavelengths, weights)

    def __getattr__(self: Source, key: str) -> Any:
        """
        Getter method for the spectrum object.

        Parameters
        ----------
        key : str
            The key to get from the spectrum object.

        Returns
        -------
        value : Any
            The value of the key.
        """
        if hasattr(self.spectrum, key):
            return getattr(self.spectrum, key)
        else:
            raise AttributeError(
                f"{self.__class__.__name__} has no " f"attribute {key}."
            )

    def normalise(self: Source) -> Source:
        """
        Method for returning a new normalised source object.

        Returns
        -------
        source : Source
            The normalised source object.
        """
        norm_spectrum = self.spectrum.normalise()
        return self.set("spectrum", norm_spectrum)

    def model(self: Source, optics: Optics) -> Array:
        """
        Method to model the psf of the point source through the optics.

        Parameters
        ----------
        optics : Optics
            The optics through which to model the source objects.
        detector : Detector = None
            The detector object that is observing the psf.

        Returns
        -------
        psf : Array
            The PSF of the source modelled through the optics.
        """
        self = self.normalise()
        weights = self.weights * self.flux
        return optics.propagate(self.wavelengths, self.position, weights)


class RelativeFluxSource(Source):
    """
    Abstract class that extend the methods of Source to allow for binary-object
    sources to be parameterised by their relative flux. Classes that inherit
    from this class must instantiate a contrast attribute.

    Attributes
    ----------
    position : Array, radians
        The (x, y) on-sky position of this object.
    flux : Array, photons
        The flux of the object.
    contrast : Array
        The contrast ratio between the two sources.
    spectrum : Spectrum
        The spectrum of this object, represented by a Spectrum object.
    """

    contrast: Array

    def __init__(self: Source, contrast: Array, **kwargs):
        """
        Constructor for the RelativeFluxSource class.

        Parameters
        ----------
        contrast : Array
            The contrast ratio between the two sources.
        """
        super().__init__(**kwargs)
        self.contrast = np.asarray(contrast, dtype=float)

        if self.contrast.shape != ():
            raise ValueError("contrast must have shape ().")

    @property
    def fluxes(self: Source) -> Array:
        """
        Getter method for the fluxes. This parametrises the source such that
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


class RelativePositionSource(Source):
    """
    Abstract class that extend the methods of Source to allow for binary-object
    sources to be parameterised by their relative position. Classes that
    inherit from this class must instantiate a separation attribute.

    Attributes
    ----------
    position : Array, radians
        The (x, y) on-sky position of this object.
    flux : Array, photons
        The flux of the object.
    separation : Array, radians
        The separation of the two sources in radians.
    position_angle : Array, radians
        The field angle between the two sources measure from the positive
        x-axis.
    spectrum : Spectrum
        The spectrum of this object, represented by a Spectrum object.
    """

    separation: Array
    position_angle: Array

    def __init__(
        self: Source, separation: Array, position_angle: Array, **kwargs
    ):
        """
        Constructor for the RelativePositionSource class.

        Parameters
        ----------
        separation : Array, radians
            The separation of the two sources in radians.
        position_angle : Array, radians
            The field angle between the two sources measure from the positive
            x-axis.
        """
        super().__init__(**kwargs)
        self.separation = np.asarray(separation, dtype=float)
        self.position_angle = np.asarray(position_angle, dtype=float)

        if self.separation.shape != ():
            raise ValueError("separation must have shape ().")

        if self.position_angle.shape != ():
            raise ValueError("position_angle must have shape ().")

    @property
    def positions(self: Source) -> Array:
        """
        Getter method for the position.

        Returns
        -------
        position : Array, radians
            The ((x, y), (x, y)) on-sky position of this object.
        """
        r, phi = self.separation / 2, self.position_angle
        sep_vec = np.array([r * np.sin(phi), r * np.cos(phi)])
        return np.array([self.position + sep_vec, self.position - sep_vec])


####################
# Concrete Classes #
####################
class PointSource(Source):
    """
    Concrete Class for unresolved point source objects.

    Attributes
    ----------
    position : Array, radians
        The (x, y) on-sky position of this object.
    flux : Array, photons
        The flux of the object.
    spectrum : Spectrum
        The spectrum of this object, represented by a Spectrum object.
    """


class PointSources(Source):
    """
    Concrete Class for multiple unresolved point source objects.

    Attributes
    ----------
    position : Array, radians
        The (x, y) on-sky positions of these sources.
    flux : Array, photons
        The fluxes of the sources.
    spectrum : Spectrum
        The spectrum of this object, represented by a Spectrum object.
        Every source in this class will have an identical spectrum.
    """

    def __init__(
        self: Source,
        wavelengths: Array,
        position: Array = np.zeros(2),
        flux: Array = None,
        weights: Array = None,
        spectrum: Spectrum = None,
    ):
        """
        Constructor for the PointSources class.

        Parameters
        ----------
        position : Array, radians
            The ((x0, y0), (x1, y1), ...) on-sky positions of these sources.
        flux : Array, photons = None
            The fluxes of the sources.
        spectrum : Spectrum = None
            The spectrum of this object, represented by a Spectrum object.
            Every source in this class will have an identical spectrum.
        wavelengths : Array, metres = None
            The array of wavelengths at which the spectrum is defined.
        """
        super().__init__(
            spectrum=spectrum, wavelengths=wavelengths, weights=weights
        )

        # More complex parameter checks here because of extra dims
        self.position = np.asarray(position, dtype=float)
        if self.position.ndim != 2:
            raise ValueError("position must be a 2d array.")

        if flux is None:
            self.flux = np.ones(len(self.position))
        else:
            self.flux = np.asarray(flux, dtype=float)

            if self.flux.ndim != 1:
                raise ValueError("flux must be a 1d array.")

            if len(self.flux) != len(self.position):
                raise ValueError(
                    "Length of flux must be equal to length of " "position."
                )

    def model(self: Source, optics: Optics) -> Array:
        """
        Method to model the psf of the point source through the optics.

        Parameters
        ----------
        optics : Optics
            The optics through which to model the source objects.

        Returns
        -------
        psf : Array
            The PSF of the source modelled through the optics.
        """
        self = self.normalise()
        weights = self.weights[None, :] * self.flux[:, None]
        propagator = vmap(optics.propagate, in_axes=(None, 0, 0))
        return propagator(self.wavelengths, self.position, weights).sum(0)


class ResolvedSource(Source):
    """
    A class for modelling resolved sources that parametrise their resolved
    component using an array of intensities.

    Attributes
    ----------
    position : Array, radians
        The (x, y) on-sky position of this object.
    flux : Array, photons
        The flux of the object.
    spectrum : Spectrum
        The spectrum of this object, represented by a Spectrum object.
    distribution : Array
        The array of intensities representing the resolved source.
    """

    distribution: Array

    def __init__(
        self: Source,
        wavelengths: Array,
        position: Array = np.zeros(2),
        flux: Array = np.array(1.0),
        distribution: Array = np.ones((3, 3)),
        weights: Array = None,
        spectrum: Spectrum = None,
    ):
        """
        Constructor for the ResolvedSource class.

        Parameters
        ----------
        position : Array, radians = np.array([0., 0.])
            The (x, y) on-sky position of this object.
        flux : Array, photons = np.array(1.)
            The flux of the object.
        distribution : Array = np.ones((3, 3))
            The array of intensities representing the resolved source.
        spectrum : Spectrum = None
            The spectrum of this object, represented by a Spectrum object.
        wavelengths : Array, metres = None
            The array of wavelengths at which the spectrum is defined.
        """
        distribution = np.asarray(distribution, dtype=float)
        self.distribution = distribution / distribution.sum()

        if self.distribution.ndim != 2:
            raise ValueError("distribution must be a 2d array.")

        super().__init__(
            position=position,
            flux=flux,
            spectrum=spectrum,
            wavelengths=wavelengths,
            weights=weights,
        )

    def normalise(self: Source) -> Source:
        """
        Method for returning a new source object with a normalised total
        spectrum and source distribution.

        Returns
        -------
        source : Source
            The source object with the normalised spectrum and distribution.
        """
        spectrum = self.spectrum.normalise()
        distribution_floor = np.maximum(self.distribution, 0.0)
        distribution = distribution_floor / distribution_floor.sum()
        return self.set(["spectrum", "distribution"], [spectrum, distribution])

    def model(self: Source, optics: Optics) -> Array:
        """
        Method to model the psf of the source through the optics. Implements a
        basic convolution with the psf and source distribution.

        Parameters
        ----------
        optics : Optics
            The optics through which to model the source objects.

        Returns
        -------
        psf : Array
            The psf of the source modelled through the optics.
        """
        # Normalise and get parameters
        self = self.normalise()
        psf = optics.propagate(self.wavelengths, self.position, self.weights)
        convolved = convolve(psf, self.distribution, mode="same")
        return self.flux * convolved


class BinarySource(RelativePositionSource, RelativeFluxSource):
    """
    A parameterised binary source.

    Attributes
    ----------
    position : Array, radians
        The (x, y) on-sky position of this object.
    flux : Array, photons
        The mean flux of the sources.
    separation : Array, radians
        The separation of the two sources in radians.
    position_angle : Array, radians
        The position angle between the two sources measured clockwise from
        the vertical axis.
    contrast : Array
        The contrast ratio between the two sources.
    spectrum : Spectrum
        The spectrum of this object, represented by a CombinedSpectrum object.
    """

    def __init__(
        self: Source,
        wavelengths: Array = None,
        position: Array = np.array([0.0, 0.0]),
        flux: Array = np.array(1.0),
        separation: Array = None,
        position_angle: Array = np.pi / 2,
        contrast: Array = np.array(1.0),
        spectrum: Spectrum = None,
        weights: Array = None,
    ):
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
        wavelengths : Array, metres = None
            The array of wavelengths at which the spectrum is defined.
        """
        wavelengths = np.asarray(wavelengths, dtype=float)
        if weights is None:
            weights = np.ones((2, len(wavelengths)))

        super().__init__(
            wavelengths=wavelengths,
            position=position,
            flux=flux,
            separation=separation,
            position_angle=position_angle,
            contrast=contrast,
            spectrum=spectrum,
            weights=weights,
        )

    def model(self: Source, optics: Optics) -> Array:
        """
        Method to model the psf of the point source through the optics.

        Parameters
        ----------
        optics : Optics
            The optics through which to model the source objects.

        Returns
        -------
        psf : Array
            The PSF of the source modelled through the optics.
        """
        self = self.normalise()
        weights = self.weights * self.fluxes[:, None]
        propagator = vmap(optics.propagate, in_axes=(None, 0, 0))
        return propagator(self.wavelengths, self.positions, weights).sum(0)


class PointResolvedSource(RelativeFluxSource, ResolvedSource):
    """
    A class for modelling a point source and a resolved source that is defined
    relative to the point source. An example would be an unresolved star with
    a resolved dust shell or debris disk. These two objects share the same
    spectra but have their fluxes defined by flux (the mean flux) and the flux
    ratio (contrast) between the point source and resolved distribution. The
    resolved component is defined by an array (ie this class inherits from
    ResolvedSource).

    Attributes
    ----------
    position : Array, radians
        The (x, y) on-sky position of this object.
    flux : Array, photons
        The mean flux of the point and resolved source.
    distribution : Array
        The array of intensities representing the resolved source.
    contrast : Array
        The contrast ratio between the point source and the resolved
        source.
    spectrum : Spectrum
        The spectrum of this object, represented by a Spectrum object.
    """

    def __init__(
        self: Source,
        wavelengths: Array = None,
        position: Array = np.zeros(2),
        flux: Array = np.array(1.0),
        distribution: Array = np.ones((3, 3)),
        contrast: Array = np.array(1.0),
        spectrum: Spectrum = None,
        weights: Array = None,
    ) -> Source:
        """
        Parameters
        ----------
        position : Array, radians = np.array([0., 0.])
            The (x, y) on-sky position of this object.
        flux : Array, photons = np.array(1.)
            The mean flux of the point and resolved source.
        distribution : Array = np.ones((3, 3))
            The array of intensities representing the resolved source.
        contrast : Array = np.array(1.)
            The contrast ratio between the point source and the resolved
            source.
        spectrum : Spectrum = None
            The spectrum of this object, represented by a Spectrum object.
        wavelengths : Array, metres = None
            The array of wavelengths at which the spectrum is defined.
        """
        super().__init__(
            wavelengths=wavelengths,
            position=position,
            flux=flux,
            distribution=distribution,
            spectrum=spectrum,
            weights=weights,
            contrast=contrast,
        )

    def model(self: Source, optics: Optics) -> Array:
        """
        Method to model the psf of the source through the optics. Implements a
        basic convolution with the psf and source distribution, while also
        modelling the single point source psf.

        Parameters
        ----------
        optics : Optics
            The optics through which to model the source objects.

        Returns
        -------
        psf : Array
            The psf of the source modelled through the optics.
        """
        # Normalise and get parameters
        self = self.normalise()
        psf = optics.propagate(self.wavelengths, self.position, self.weights)
        convolved = convolve(psf, self.distribution, mode="same")
        return self.fluxes[0] * psf + self.fluxes[1] * convolved
