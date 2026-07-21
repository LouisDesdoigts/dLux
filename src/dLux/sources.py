from __future__ import annotations
from typing import Any, TYPE_CHECKING
from abc import abstractmethod
from jax import Array
import jax.numpy as np
import jax.scipy as jsp
import jax.tree as jtu
import zodiax as zdx
import equinox as eqx
import dLux.utils as dlu
from dLux import spectra


from .psfs import PSF

if TYPE_CHECKING:
    from .optical_systems import BaseOpticalSystem
    from .wavefronts import Wavefront

__all__ = [
    "BaseSource",
    "PointSource",
    "PointSources",
    "BinarySource",
    "ResolvedSource",
    "PointResolvedSource",
    "Scene",
]


def _validate_return_mode(return_wf: bool, return_psf: bool) -> None:
    """
    Raises a ValueError if both return_wf and return_psf are True.

    Parameters
    ----------
    return_wf : bool
        Whether to return a Wavefront object.
    return_psf : bool
        Whether to return a PSF object.
    """
    if return_wf and return_psf:
        raise ValueError(
            "Cannot return both Wavefront and PSF objects. Choose one: "
            "set return_wf=True for Wavefront, or return_psf=True for PSF object."
        )


def _as_wavelengths_1d(wavelengths: Array | None) -> Array | None:
    """
    Validates and returns wavelengths as a 1D float array, or None.

    Parameters
    ----------
    wavelengths : Array | None
        The wavelengths to validate.

    Returns
    -------
    wavelengths : Array | None
        The validated 1D wavelengths array, or None.
    """
    if wavelengths is None:
        return None
    wavelengths = np.asarray(wavelengths, dtype=float)
    if wavelengths.ndim != 1:
        raise ValueError("wavelengths must be a 1d array.")
    return wavelengths


def _infer_n_wavelengths(
    wavelengths: Array | None,
    spectrum: spectra.BaseSpectrum | None,
) -> int:
    """
    Infers the number of wavelengths from the wavelengths array or spectrum.

    Parameters
    ----------
    wavelengths : Array | None
        The wavelengths array. If provided, its length is returned.
    spectrum : BaseSpectrum | None
        The spectrum object. Used to infer the number of wavelengths if
        `wavelengths` is None.

    Returns
    -------
    n_wavelengths : int
        The number of wavelengths.
    """
    wavelengths = _as_wavelengths_1d(wavelengths)
    if wavelengths is not None:
        return len(wavelengths)

    if spectrum is None:
        raise ValueError(
            "wavelengths must be provided when weights is None and no spectrum "
            "is supplied."
        )

    if not isinstance(spectrum, spectra.BaseSpectrum):
        raise TypeError(
            "spectrum must be a dLux BaseSpectrum instance from dLux.spectra."
        )

    return len(np.asarray(spectrum.wavelengths, dtype=float))


def _as_position_2d(position: Array) -> Array:
    """
    Validates and returns a position as a shape-(2,) float array.

    Parameters
    ----------
    position : Array
        The (x, y) on-sky position in radians.

    Returns
    -------
    position : Array
        The validated shape-(2,) position array.
    """
    position = np.asarray(position, dtype=float)
    if position.shape != (2,):
        raise ValueError(
            f"position must be a 1d array of shape (2,), got shape {position.shape}. "
            "Pass position=[x, y] where x, y are on-sky angles in radians."
        )
    return position


class BaseSource(zdx.Base):
    """
    Abstract base class for source models.

    Concrete source classes define normalisation behavior and the optical
    modelling interface via `normalise(...)` and `model(...)`.

    ??? abstract "UML"
        ![UML](../assets/uml/BaseSource.png)
    """

    def __init_subclass__(cls, **kwargs):
        """Inherit docstrings from parent classes for model method."""
        super().__init_subclass__(**kwargs)
        dlu.helpers.inherit_docstrings(cls, ["model"])

    @abstractmethod
    def normalise(self: BaseSource) -> BaseSource:  # pragma: no cover
        pass

    @abstractmethod
    def model(
        self: BaseSource,
        optics: BaseOpticalSystem,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array | Wavefront | PSF:  # pragma: no cover
        """
        Models the source object through the provided optics.

        Parameters
        ----------
        optics : OpticalSystem
            The optics through which to model the source object.
        return_wf : bool = False
            Should the Wavefront object be returned instead of the PSF array?
        return_psf : bool = False
            Should the PSF object be returned instead of the PSF array?

        Returns
        -------
        result : Array | Wavefront | PSF
            If `return_wf` is False and `return_psf` is False, returns the PSF array.
            If `return_wf` is True and `return_psf` is False, returns the Wavefront
                object.
            If `return_wf` is False and `return_psf` is True, returns the PSF object.
        """


class Source(BaseSource):
    """
    Base source class that implements the spectra attribute.

    ??? abstract "UML"
        ![UML](../assets/uml/Source.png)

    Attributes
    ----------
    spectrum : Spectrum
        The spectrum of this object, represented by a Spectrum object.
    """

    spectrum: spectra.BaseSpectrum

    def __init__(
        self: Source,
        wavelengths: Array = None,
        weights: Array = None,
        spectrum: spectra.BaseSpectrum = None,
    ):
        """
        Parameters
        ----------
        wavelengths : Array, metres = None
            The array of wavelengths at which the spectrum is defined. This input is
            ignored if a Spectrum object is provided.
        weights : Array = None
            The spectral weights of the object.
        spectrum : Spectrum = None
            The spectrum of this object, represented by a Spectrum object.
        """
        # Spectrum
        if spectrum is not None:
            if not isinstance(spectrum, spectra.BaseSpectrum):
                raise TypeError(
                    f"spectrum must be a dLux BaseSpectrum instance from "
                    f"dLux.spectra, "
                    f"got {type(spectrum).__name__}. "
                    "Use a dLux spectrum type from dLux.spectra."
                )
            self.spectrum = spectrum
        else:
            self.spectrum = spectra.Spectrum(wavelengths, weights)

    def __getattr__(self: Source, key: str) -> Any:
        """
        Raises the parameters of the spectrum object to this class.

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
        spectrum_state = [
            a for a in self.spectrum.__dict__.keys() if not a.startswith("_")
        ]
        core_spectrum_fields = [
            name for name in ("wavelengths", "weights") if hasattr(self.spectrum, name)
        ]
        spectrum_attrs = sorted(set(spectrum_state + core_spectrum_fields))
        raise dlu.helpers.missing_attribute_error(
            self,
            key,
            spectrum_attrs,
            hint="This object forwards spectrum attributes via '.spectrum'.",
        )

    def normalise(self: Source) -> Source:
        """
        Returns a normalised source object.

        Returns
        -------
        source : Source
            The normalised source object.
        """
        norm_spectrum = self.spectrum.normalise()
        return self.set("spectrum", norm_spectrum)


class PointSource(Source):
    """
    A simple point source with a spectrum, position and flux.

    ??? abstract "UML"
        ![UML](../assets/uml/PointSource.png)

    Attributes
    ----------
    position : Array, radians
        The (x, y) on-sky position of this object.
    flux : float, photons
        The flux of the object.
    spectrum : Spectrum
        The spectrum of this object, represented by a Spectrum object.
    """

    position: Array
    flux: float

    def __init__(
        self: PointSource,
        wavelengths: Array = None,
        position: Array = np.zeros(2),
        flux: float = 1.0,
        weights: Array = None,
        spectrum: spectra.BaseSpectrum = None,
    ):
        """
        Parameters
        ----------
        wavelengths : Array, metres = None
            The array of wavelengths at which the spectrum is defined. This input is
            ignored if a Spectrum object is provided.
        position : Array, radians = np.zeros(2)
            The (x, y) on-sky position of this object.
        flux : float, photons = 1.
            The flux of the object.
        spectrum : Spectrum = None
            The spectrum of this object, represented by a Spectrum object.
        """
        # Position and Flux
        self.position = _as_position_2d(position)
        self.flux = float(flux)

        super().__init__(wavelengths=wavelengths, weights=weights, spectrum=spectrum)

    def model(
        self: PointSource,
        optics: BaseOpticalSystem,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array | Wavefront | PSF:
        _validate_return_mode(return_wf, return_psf)
        self = self.normalise()
        weights = self.weights * self.flux
        return optics.propagate(
            self.wavelengths, self.position, weights, return_wf, return_psf
        )


class PointSources(Source):
    """
    A set of point sources with the same spectrum, but different positions and fluxes.

    ??? abstract "UML"
        ![UML](../assets/uml/PointSources.png)

    Attributes
    ----------
    position : Array, radians
        The ((x0, y0), (x1, y1), ...) on-sky positions of these sources.
    flux : Array, photons
        The fluxes of the sources.
    spectrum : Spectrum
        The spectrum of this object, represented by a Spectrum object.
    """

    position: Array
    flux: Array

    def __init__(
        self: PointSources,
        wavelengths: Array = None,
        position: Array = np.zeros((1, 2)),
        flux: Array = None,
        weights: Array = None,
        spectrum: spectra.BaseSpectrum = None,
    ):
        """
        Parameters
        ----------
        wavelengths : Array, metres
            The array of wavelengths at which the spectrum is defined.
        position : Array, radians = np.zeros((1, 2))
            The (x, y) on-sky position of this object.
        flux : Array, photons = None
            The flux of the object.
        weights : Array = None
            The spectral weights of the object.
        spectrum : Spectrum = None
            The spectrum of this object, represented by a Spectrum object.
        """
        super().__init__(spectrum=spectrum, wavelengths=wavelengths, weights=weights)

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
        self: PointSources,
        optics: BaseOpticalSystem,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array | Wavefront | PSF:
        _validate_return_mode(return_wf, return_psf)
        self = self.normalise()
        weights = self.weights[None, :] * self.flux[:, None]
        prop_fn = lambda position, weight: optics.propagate(
            self.wavelengths, position, weight, return_wf=True
        )
        wfs = eqx.filter_vmap(prop_fn)(self.position, weights)

        if return_wf:
            return wfs
        if return_psf:
            return PSF(wfs.psf.sum((0, 1)), wfs.pixel_scale.mean())
        else:
            return wfs.psf.sum((0, 1))


class ResolvedSource(PointSource):
    """
    A single resolved source with a spectrum, position, flux, and distribution array
    that represents the resolved component.

    ??? abstract "UML"
        ![UML](../assets/uml/ResolvedSource.png)

    Attributes
    ----------
    position : Array, radians
        The (x, y) on-sky position of this object.
    flux : float, photons
        The flux of the object.
    distribution : Array
        The array of intensities representing the resolved source.
    spectrum : Spectrum
        The spectrum of this object, represented by a Spectrum object.
    """

    distribution: Array

    def __init__(
        self: ResolvedSource,
        wavelengths: Array = None,
        position: Array = np.zeros(2),
        flux: float = 1.0,
        distribution: Array = np.ones((3, 3)),
        weights: Array = None,
        spectrum: spectra.BaseSpectrum = None,
    ):
        """
        Parameters
        ----------
        wavelengths : Array, metres
            The array of wavelengths at which the spectrum is defined.
        position : Array, radians = np.zeros(2)
            The (x, y) on-sky position of this object.
        flux : float, photons = 1.
            The flux of the object.
        distribution : Array = np.ones((3, 3))
            The array of intensities representing the resolved source.
        weights : Array = None
            The spectral weights of the object.
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

    def normalise(self: ResolvedSource) -> ResolvedSource:
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
        self: ResolvedSource,
        optics: BaseOpticalSystem,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array | Wavefront | PSF:
        _validate_return_mode(return_wf, return_psf)
        # Normalise and get parameters
        self = self.normalise()
        weights = self.weights * self.flux

        # Note that we always return a wavefront here so we can convolve each
        # wavelength
        # individually if a chromatic wavefront output is required.
        wf = optics.propagate(self.wavelengths, self.position, weights, return_wf=True)

        # Returning wf is a special case
        if return_wf:
            raise NotImplementedError(
                "Wavefront information cannot be preserved through convolution. "
                "Convolution can only operate on PSFs (incoherent light). "
                "Please use return_wf=False to get the PSF array or return_psf=True "
                "to get a PSF object."
            )

        # Return PSF object
        conv_psf = jsp.signal.convolve(wf.psf.sum(0), self.distribution, mode="same")
        if return_psf:
            return PSF(conv_psf, wf.pixel_scale.mean())

        # Return PSF array
        return conv_psf


class BinarySource(Source):
    """
    A binary source parameterised by the position, flux, separation, position_angle,
    and contrast between the two sources.

    ??? abstract "UML"
        ![UML](../assets/uml/BinarySource.png)

    Attributes
    ----------
    position : Array, radians
        The mean (x, y) on-sky position of this object.
    mean_flux : float, photons
        The mean flux of the sources.
    separation : float, radians
        The separation of the two sources in radians.
    position_angle : float, radians
        The position angle between the two sources measured clockwise from the
        vertical axis.
    contrast : float
        The contrast ratio between the two sources.
    spectrum : Spectrum
        The spectrum of this object, represented by a Spectrum object.
    """

    position: Array
    mean_flux: float
    separation: float
    position_angle: float
    contrast: float

    def __init__(
        self: BinarySource,
        wavelengths: Array = None,
        position: Array = np.zeros(2),
        mean_flux: float = 1.0,
        separation: float = 0.0,
        position_angle: float = np.pi / 2,
        contrast: float = 1.0,
        spectrum: spectra.BaseSpectrum = None,
        weights: Array = None,
    ):
        """
        Parameters
        ----------
        wavelengths : Array, metres = None
            The array of wavelengths at which the spectrum is defined.
        position : Array, radians = np.zeros(2)
            The (x, y) on-sky position of this object.
        mean_flux : float, photons = 1.
            The mean flux of the sources.
        separation : float, radians = 0.
            The separation of the two sources in radians.
        position_angle : float, radians = np.pi / 2
            The position angle between the two sources measured clockwise from the
            vertical axis.
        contrast : float = 1.
            The contrast ratio between the two sources.
        spectrum : Spectrum = None
            The spectrum of this object, represented by a Spectrum object.
        """
        wavelengths = _as_wavelengths_1d(wavelengths)
        if weights is None:
            n_wavelengths = _infer_n_wavelengths(wavelengths, spectrum)
            weights = np.ones((2, n_wavelengths))

        # Position and Flux
        self.position = _as_position_2d(position)
        self.mean_flux = float(mean_flux)

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
        self: BinarySource,
        optics: BaseOpticalSystem,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array | Wavefront | PSF:
        _validate_return_mode(return_wf, return_psf)
        # Normalise and get input values
        self = self.normalise()
        positions = dlu.positions_from_sep(
            self.position, self.separation, self.position_angle
        )
        flux = dlu.fluxes_from_contrast(self.mean_flux, self.contrast)
        weights = self.weights * flux[:, None]

        # Return wf case is simple
        prop_fn = lambda position, weight: optics.propagate(
            self.wavelengths, position, weight, return_wf, return_psf
        )
        output = eqx.filter_vmap(prop_fn)(positions, weights)

        # Return wf is simple case
        if return_wf:
            return output

        # Return PSF case just requires constructing the object
        if return_psf:
            return PSF(output.data.sum(0), output.pixel_scale.mean())

        # Return array is simple
        return output.sum(0)


class PointResolvedSource(ResolvedSource):
    """
    A class for modelling a point source and a resolved source that is defined
    relative to the point source. An example would be an unresolved star with
    a resolved dust shell or debris disk. These two objects share the same
    spectra but have their flux defined by flux (the mean flux) and the flux
    ratio (contrast) between the point source and resolved distribution. The
    resolved component is defined by an array of intensities that represent
    the resolved distribution.

    ??? abstract "UML"
        ![UML](../assets/uml/PointResolvedSource.png)

    Attributes
    ----------
    position : Array, radians
        The (x, y) on-sky position of this object.
    flux : float, photons
        The mean flux of the point and resolved source.
    distribution : Array
        The array of intensities representing the resolved source.
    contrast : float
        The contrast ratio between the point source and the resolved source.
    spectrum : Spectrum
        The spectrum of this object, represented by a Spectrum object.
    """

    contrast: float

    def __init__(
        self: PointResolvedSource,
        wavelengths: Array = None,
        position: Array = np.zeros(2),
        flux: float = 1.0,
        distribution: Array = np.ones((3, 3)),
        contrast: float = 1.0,
        weights: Array = None,
        spectrum: spectra.BaseSpectrum = None,
    ) -> PointResolvedSource:
        """
        Parameters
        ----------
        wavelengths : Array, metres = None
            The array of wavelengths at which the spectrum is defined.
        position : Array, radians = np.zeros(2)
            The (x, y) on-sky position of this object.
        flux : float, photons = 1.
            The mean flux of the point and resolved source.
        distribution : Array = np.ones((3, 3))
            The array of intensities representing the resolved source.
        contrast : float = 1.
            The contrast ratio between the point source and the resolved source.
        weights : Array = None
            The spectral weights of the object.
        spectrum : Spectrum = None
            The spectrum of this object, represented by a Spectrum object.
        """
        wavelengths = _as_wavelengths_1d(wavelengths)
        if weights is None:
            n_wavelengths = _infer_n_wavelengths(wavelengths, spectrum)
            weights = np.ones((2, n_wavelengths))

        self.contrast = float(contrast)

        super().__init__(
            wavelengths=wavelengths,
            position=position,
            flux=flux,
            distribution=distribution,
            spectrum=spectrum,
            weights=weights,
        )

    def model(
        self: PointResolvedSource,
        optics: BaseOpticalSystem,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array | Wavefront | PSF:
        _validate_return_mode(return_wf, return_psf)
        # Normalise and get parameters
        self = self.normalise()
        flux = dlu.fluxes_from_contrast(self.flux, self.contrast)
        weights = self.weights * flux[:, None]

        # Note that we always return a wavefront here so we can convolve each
        # wavelength
        # individually if a chromatic wavefront output is required. We also
        # cannot propagate the weights since they have different values
        # for the point and resolved source.
        wf = optics.propagate(self.wavelengths, self.position, return_wf=True)

        # Returning wf is a special case
        if return_wf:
            raise NotImplementedError(
                "Wavefront information cannot be preserved through convolution. "
                "Convolution can only operate on PSFs (incoherent light). "
                "Please use return_wf=False to get the PSF array or return_psf=True "
                "to get a PSF object."
            )

        # Create single PSF-array object
        point_psf = (np.expand_dims(weights[0], (1, 2)) * wf.psf).sum(0)
        resolved_psf = (np.expand_dims(weights[1], (1, 2)) * wf.psf).sum(0)
        conv_psf = jsp.signal.convolve(resolved_psf, self.distribution, mode="same")
        psf = point_psf + conv_psf
        if return_psf:
            return PSF(psf, wf.pixel_scale.mean())

        # Return PSF array
        return psf


class Scene(BaseSource):
    """
    A source object that holds a set of sources that are model simultaneously.

    ??? abstract "UML"
        ![UML](../assets/uml/Scene.png)

    Attributes
    ----------
    sources : dict
        A dictionary of source objects to model simultaneously.
    """

    sources: dict

    def __init__(self: Scene, sources: list[Source]):
        """
        Parameters
        ----------
        sources : list[Source]
            A list of source objects to model simultaneously.
        """
        super().__init__()
        if isinstance(sources, BaseSource):
            sources = [sources]
        elif isinstance(sources, tuple):
            sources = list(sources)
        self.sources = dlu.list2dictionary(sources, False, BaseSource)

    def normalise(self: Scene) -> Scene:
        """
        Method for returning a new scene with normalised source objects.

        Returns
        -------
        scene : Scene
            The normalised scene object.
        """
        is_source = lambda leaf: isinstance(leaf, BaseSource)
        norm_fn = lambda source: source.normalise()
        sources = jtu.map(norm_fn, self.sources, is_leaf=is_source)
        return self.set("sources", sources)

    def __getattr__(self: Scene, key: str) -> Any:
        """
        Raises the individual sources via their keys.

        Parameters
        ----------
        key : str
            The key of the item to be searched for in the sub-dictionaries.

        Returns
        -------
        item : Any
            The item corresponding to the supplied key in the sub-dictionaries.
        """
        if key in self.sources.keys():
            return self.sources[key]
        raise dlu.helpers.missing_attribute_error(
            self,
            key,
            list(self.sources.keys()),
        )

    def model(
        self: Scene,
        optics: BaseOpticalSystem,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array | Wavefront | PSF:
        _validate_return_mode(return_wf, return_psf)
        self = self.normalise()

        # Define leaf_fn and map across sources
        leaf_fn = lambda leaf: isinstance(leaf, BaseSource)
        output = jtu.map(
            lambda s: s.model(optics, return_wf, return_psf),
            self.sources,
            is_leaf=leaf_fn,
        )

        # Return wf case is simple
        if return_wf:
            return output

        # Return PSF case requires mapping across the PSF outputs
        if return_psf:
            # Define mapping function
            leaf_fn = lambda leaf: isinstance(leaf, PSF)
            get_psfs = lambda psf: psf.data.sum(tuple(range(psf.ndim)))
            get_pscales = lambda psf: psf.pixel_scale.mean()

            # Get values and return PSF
            psf = dlu.map2array(get_psfs, output, leaf_fn).sum(0)
            pixel_scale = dlu.map2array(get_pscales, output, leaf_fn).mean()
            return PSF(psf, pixel_scale)

        # Return array is simple
        return dlu.map2array(lambda x: x, output).sum(0)
