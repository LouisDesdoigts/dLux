from __future__ import annotations
from typing import Any
import jax.numpy as np
from jax.scipy.signal import convolve
from jax import vmap, Array
from jax.tree_util import tree_map
from zodiax import Base
from abc import abstractmethod
import dLux
import dLux.utils as dlu


__all__ = [
    "Scene",
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
    def model(self, optics):  # pragma: no cover
        pass


class Scene(BaseSource):
    sources: dict

    def __init__(self, sources: list[Source]):
        if isinstance(sources, (Source(), tuple)):
            source = [sources]
        self.sources = dlu.list_to_dictionary(source, False, Source())

    def normalise(self: Scene) -> Scene:
        """
        Method for returning a new scene with normalised source objects.

        Returns
        -------
        scene : Scene
            The normalised scene object.
        """
        is_source = lambda leaf: isinstance(leaf, Source())
        norm_fn = lambda source: source.normalise()
        sources = tree_map(norm_fn, self.sources, is_leaf=is_source)
        return self.set("sources", sources)

    def __getattr__(self: Source, key: str) -> Any:
        """
        Magic method designed to allow accessing of the various items within
        the sub-dictionaries of this class via the 'class.attribute' method.
        It is recommended that each dictionary key in the optical layers,
        detector layers, and scene sources are unique to prevent unexpected
        behaviour. In the case they there are identical keys across the
        dictionaries This method prioritises searching for keys in the optical
        layers, then detector layers, and then the scene sources.

        Parameters
        ----------
        key : str
            The key of the item to be searched for in the sub-dictionaries.

        Returns
        -------
        item : object
            The item corresponding to the supplied key in the sub-dictionaries.
        """
        if key in self.sources.keys():
            return self.sources[key]
        for attribute in self.__dict__.values():
            if hasattr(attribute, key):
                return getattr(attribute, key)
        raise AttributeError(
            f"{self.__class__.__name__} has no attribute " f"{key}."
        )

    def model(
        self: Scene, optics: Optics, get_pixel_scale: bool = False
    ) -> Array:
        """
        Method for returning the model of the scene.

        Parameters
        ----------
        optics : Optics
            The optics object to be used to model the scene.
        get_pixel_scale : bool = False
            Whether to return the pixel scale of the scene.

        Returns
        -------
        psf : Array
            The psf of the scene.
        pixel_scale : Array
            The pixel scale of the scene. Only returned if get_pixel_scale is
            True.
        """
        self = self.normalise
        sources = list(self.source.values())

        if get_pixel_scale:
            psfs, pixel_scales = np.array(
                [source.model(self, True) for source in sources]
            )
            return psfs.sum(0), pixel_scales.mean()
        else:
            return np.array(
                [source.model(self, False) for source in sources]
            ).sum(0)


class Source(BaseSource):
    """
    Base source class that implements the spectra attribute.

    Attributes
    ----------
    spectrum : Spectrum
        The spectrum of this object, represented by a Spectrum object.
    """

    spectrum: Spectrum

    def __init__(
        self: Source,
        wavelengths: Array,
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
        weights : Array = None
            The spectral weights of the object.
        spectrum : Spectrum = None
            The spectrum of this object, represented by a Spectrum object.
            if provided it will overwrite the inputs wavelengths and weights.
        """
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

    position: Array
    flux: float

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
            If provided it will overwrite the inputs wavelengths and weights.

        """
        # Position and Flux
        self.position = np.asarray(position, dtype=float)
        self.flux = float(flux)

        if self.position.shape != (2,):
            raise ValueError("position must be a 1d array of shape (2,).")

        super().__init__(
            wavelengths=wavelengths, weights=weights, spectrum=spectrum
        )

    def model(
        self: Source, optics: Optics, get_pixel_scale: bool = False
    ) -> Array:
        """
        Method to model the psf of the point source through the optics.

        Parameters
        ----------
        optics : Optics
            The optics through which to model the source objects.
        get_pixel_scale : bool = False, radians
            Whether to also return the psf pixel scale.

        Returns
        -------
        psf : Array
            The PSF of the source modelled through the optics.
        pixel_scale : Array, radians
            The pixel scale of the psf. Only returned if
            `get_pixel_scale == True`.
        """
        self = self.normalise()
        weights = self.weights * self.flux
        return optics.propagate(
            self.wavelengths, self.position, weights, get_pixel_scale
        )


class PointSources(Source):
    """
    Class for multiple unresolved point source objects.

    Attributes
    ----------
    position : Array, radians
        The (x, y) on-sky positions of these sources.
    flux : Array, photons
        The fluxes of the sources.
    spectrum : Spectrum
        The spectrum of this object, represented by a Spectrum object.
    """

    position: Array
    flux: Array

    def __init__(
        self: Source,
        wavelengths: Array,
        position: Array = np.zeros((1, 2)),
        flux: Array = None,
        weights: Array = None,
        spectrum: Spectrum = None,
    ):
        """
        Constructor for the PointSources class.

        Parameters
        ----------
        wavelengths : Array, metres = None
            The array of wavelengths at which the spectrum is defined.
        position : Array, radians
            The ((x0, y0), (x1, y1), ...) on-sky positions of these sources.
        flux : Array, photons = None
            The fluxes of the sources.
        spectrum : Spectrum = None
            The spectrum of this object, represented by a Spectrum object.
            If provided it will overwrite the inputs wavelengths and weights.
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
                    "Length of flux must be equal to length of " "positions."
                )

    def model(
        self: Source, optics: Optics, get_pixel_scale: bool = False
    ) -> Array:
        """
        Method to model the psf of the point source through the optics.

        Parameters
        ----------
        optics : Optics
            The optics through which to model the source objects.
        get_pixel_scale : bool = False, radians
            Whether to also return the psf pixel scale.

        Returns
        -------
        psf : Array
            The PSF of the source modelled through the optics.
        pixel_scale : Array, radians
            The pixel scale of the psf. Only returned if
            `get_pixel_scale == True`.
        """
        self = self.normalise()
        weights = self.weights[None, :] * self.flux[:, None]
        propagator = vmap(optics.propagate, in_axes=(None, 0, 0, None))
        if get_pixel_scale:
            psfs, pixel_scales = propagator(
                self.wavelengths, self.position, weights, True
            )
            return psfs.sum(0), pixel_scales.mean(0)
        else:
            return propagator(
                self.wavelengths, self.position, weights, False
            ).sum(0)


class ResolvedSource(PointSource):
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
        wavelengths : Array, metres
            The array of wavelengths at which the spectrum is defined.
        position : Array, radians = np.array([0., 0.])
            The (x, y) on-sky position of this object.
        flux : Array, photons = np.array(1.)
            The flux of the object.
        distribution : Array = np.ones((3, 3))
            The array of intensities representing the resolved source.
        spectrum : Spectrum = None
            The spectrum of this object, represented by a Spectrum object.
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

    def model(
        self: Source, optics: Optics, get_pixel_scale: bool = False
    ) -> Array:
        """
        Method to model the psf of the point source through the optics.

        Parameters
        ----------
        optics : Optics
            The optics through which to model the source objects.
        get_pixel_scale : bool = False, radians
            Whether to also return the psf pixel scale.

        Returns
        -------
        psf : Array
            The PSF of the source modelled through the optics.
        pixel_scale : Array, radians
            The pixel scale of the psf. Only returned if
            `get_pixel_scale == True`.
        """
        # Normalise and get parameters
        self = self.normalise()
        if get_pixel_scale:
            psf, pixel_scale = optics.propagate(
                self.wavelengths, self.position, self.weights, get_pixel_scale
            )
        else:
            psf = optics.propagate(
                self.wavelengths, self.position, self.weights
            )
        convolved = convolve(psf, self.distribution, mode="same")
        if get_pixel_scale:
            return self.flux * convolved, pixel_scale
        else:
            return self.flux * convolved


class BinarySource(Source):
    """
    A parameterised binary source.

    Attributes
    ----------
    position : Array, radians
        The mean (x, y) on-sky position of this object.
    mean_flux : Array, photons
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

    position: Array
    mean_flux: float
    separation: float
    position_angle: float
    contrast: float

    def __init__(
        self: Source,
        wavelengths: Array = None,
        position: Array = np.zeros(2),
        mean_flux: float = 1.0,
        separation: float = None,
        position_angle: float = np.pi / 2,
        contrast: Array = 1.0,
        spectrum: Spectrum = None,
        weights: Array = None,
    ):
        """
        Parameters
        ----------
        position : Array, radians = np.zeros(2)
            The mean (x, y) on-sky position of this object.
        mean_flux : float, photons = 1.
            The mean flux of the sources.
        separation : float, radians = None
            The separation of the two sources in radians.
        position_angle : float, radians = np.pi/2
            The position angle between the two sources measured clockwise from
            the vertical axis.
        contrast : float = np.array(1.)
            The contrast ratio between the two sources.
        spectrum : CombinedSpectrum = None
            The spectrum of this object, represented by a CombinedSpectrum.
        wavelengths : Array, metres = None
            The array of wavelengths at which the spectrum is defined.
        """
        wavelengths = np.asarray(wavelengths, dtype=float)
        if weights is None:
            weights = np.ones((2, len(wavelengths)))

        # Position and Flux
        self.position = np.asarray(position, dtype=float)
        self.mean_flux = float(mean_flux)

        if self.position.shape != (2,):
            raise ValueError("position must be a 1d array of shape (2,).")

        # Binary values
        self.separation = float(separation)
        self.position_angle = float(position_angle)
        self.contrast = float(contrast)

        super().__init__(
            wavelengths=wavelengths,
            spectrum=spectrum,
            weights=weights,
        )

    def model(
        self: Source, optics: Optics, get_pixel_scale: bool = False
    ) -> Array:
        """
        Method to model the psf of the point source through the optics.

        Parameters
        ----------
        optics : Optics
            The optics through which to model the source objects.
        get_pixel_scale : bool = False, radians
            Whether to also return the psf pixel scale.

        Returns
        -------
        psf : Array
            The PSF of the source modelled through the optics.
        pixel_scale : Array, radians
            The pixel scale of the psf. Only returned if
            `get_pixel_scale == True`.
        """

        positions = dlu.positions_from_sep(
            self.position, self.separation, self.position_angle
        )
        flux = dlu.flux_from_contrast(self.mean_flux, self.contrast)

        self = self.normalise()
        weights = self.weights * flux[:, None]
        propagator = vmap(optics.propagate, in_axes=(None, 0, 0, None))
        if get_pixel_scale:
            psfs, pixel_scales = propagator(
                self.wavelengths, positions, weights, True
            )
            return psfs.sum(0), pixel_scales.mean(0)
        else:
            psfs = propagator(self.wavelengths, positions, weights, False)
            return psfs.sum(0)


class PointResolvedSource(ResolvedSource):
    """
    A class for modelling a point source and a resolved source that is defined
    relative to the point source. An example would be an unresolved star with
    a resolved dust shell or debris disk. These two objects share the same
    spectra but have their flux defined by flux (the mean flux) and the flux
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

    contrast: float

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

    def model(
        self: Source, optics: Optics, get_pixel_scale: bool = False
    ) -> Array:
        """
        Method to model the psf of the point source through the optics.

        Parameters
        ----------
        optics : Optics
            The optics through which to model the source objects.
        get_pixel_scale : bool = False, radians
            Whether to also return the psf pixel scale.

        Returns
        -------
        psf : Array
            The PSF of the source modelled through the optics.
        pixel_scale : Array, radians
            The pixel scale of the psf. Only returned if
            `get_pixel_scale == True`.
        """
        # Normalise and get parameters
        self = self.normalise()
        flux = dlu.flux_from_contrast(self.flux, self.contrast)
        weights = self.weights * flux[:, None]

        if get_pixel_scale:
            psf, pixel_scale = optics.propagate(
                self.wavelengths, self.position, weights, get_pixel_scale
            )
        else:
            psf = optics.propagate(self.wavelengths, self.position, weights)
        convolved = convolve(psf, self.distribution, mode="same")
        if get_pixel_scale:
            return (
                self.flux[0] * psf + self.flux[1] * convolved,
                pixel_scale,
            )
        else:
            return self.flux[0] * psf + self.flux[1] * convolved
