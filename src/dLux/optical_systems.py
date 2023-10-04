from __future__ import annotations
from collections import OrderedDict
from abc import abstractmethod
import jax.numpy as np
from jax import Array
from zodiax import filter_vmap, Base
from typing import Union, Any
import dLux.utils as dlu


__all__ = [
    "BaseOpticalSystem",
    "AngularOpticalSystem",
    "CartesianOpticalSystem",
    "LayeredOpticalSystem",
]

from .layers.optical_layers import OpticalLayer
from .wavefronts import Wavefront
from .sources import BaseSource as Source
from .psfs import PSF


###################
# Private Classes #
###################
class BaseOpticalSystem(Base):
    @abstractmethod
    def propagate_mono(
        self: BaseOpticalSystem,
        wavelength: float,
        offset: Array = np.zeros(2),
        return_wf: bool = False,
    ) -> Array:  # pragma: no cover
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : float, metres
            The wavelength of the wavefront to propagate through the optical layers.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        return_wf: bool = False
            Should the Wavefront object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront
            if `return_wf` is False, returns the psf Array.
            if `return_wf` is True, returns the Wavefront object.
        """

    @abstractmethod
    def propagate(
        self: OpticalSystem,
        wavelengths: Array,
        offset: Array = np.zeros(2),
        weights: Array = None,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array:  # pragma: no cover
        """
        Propagates a Polychromatic point source through the optics.

        Parameters
        ----------
        wavelengths : Array, metres
            The wavelengths of the wavefronts to propagate through the optics.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        weights : Array = None
            The weight of each wavelength. If None, all weights are equal.
        return_wf : bool = False
            Should the Wavefront object be returned instead of the psf Array?
        return_psf : bool = False
            Should the PSF object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront, PSF
            if `return_wf` is False and `return_psf` is False, returns the psf Array.
            if `return_wf` is True and `return_psf` is False, returns the Wavefront
                object.
            if `return_wf` is False and `return_psf` is True, returns the PSF object.

        """

    @abstractmethod
    def model(
        self: OpticalSystem,
        source: Source,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array:  # pragma: no cover
        """
        Models the input Source object through the optics.

        Parameters
        ----------
        source : Source
            The Source object to model through the optics.
        return_wf : bool = False
            Should the Wavefront object be returned instead of the psf Array?
        return_psf : bool = False
            Should the PSF object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront, PSF
            if `return_wf` is False and `return_psf` is False, returns the psf Array.
            if `return_wf` is True and `return_psf` is False, returns the Wavefront
                object.
            if `return_wf` is False and `return_psf` is True, returns the PSF object.
        """


class OpticalSystem(BaseOpticalSystem):
    """
    Base optics class implementing both the `propagate` and `model` methods that are
    universal to all optics classes.
    """

    def propagate(
        self: OpticalSystem,
        wavelengths: Array,
        offset: Array = np.zeros(2),
        weights: Array = None,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array:
        """
        Propagates a Polychromatic point source through the optics.

        Parameters
        ----------
        wavelengths : Array, metres
            The wavelengths of the wavefronts to propagate through the optics.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        weights : Array = None
            The weight of each wavelength. If None, all weights are equal.
        return_wf : bool = False
            Should the Wavefront object be returned instead of the psf Array?
        return_psf : bool = False
            Should the PSF object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront, PSF
            if `return_wf` is False and `return_psf` is False, returns the psf Array.
            if `return_wf` is True and `return_psf` is False, returns the Wavefront
                object.
            if `return_wf` is False and `return_psf` is True, returns the PSF object.
        """
        if return_wf and return_psf:
            raise ValueError(
                "return_wf and return_psf cannot both be True. "
                "Please choose one."
            )

        wavelengths = np.atleast_1d(wavelengths)
        if weights is None:
            weights = np.ones_like(wavelengths) / len(wavelengths)
        else:
            weights = np.atleast_1d(weights)

        # Check wavelengths and weights
        if weights.shape != wavelengths.shape:
            raise ValueError(
                "wavelengths and weights must have the "
                f"same shape, got {wavelengths.shape} and {weights.shape} "
                "respectively."
            )

        # Check offset
        offset = np.array(offset) if not isinstance(offset, Array) else offset
        if offset.shape != (2,):
            raise ValueError(
                "offset must be a 2-element array, got "
                f"shape {offset.shape}."
            )

        # Calculate - note we multiply by sqrt(weight) to account for the
        # fact that the PSF is the square of the amplitude
        prop_fn = lambda wavelength, weight: self.propagate_mono(
            wavelength, offset, return_wf=True
        ).multiply("amplitude", weight**0.5)
        wf = filter_vmap(prop_fn)(wavelengths, weights)

        # Return PSF, Wavefront, or array psf
        if return_wf:
            return wf
        if return_psf:
            return PSF(wf.psf.sum(0), wf.pixel_scale.mean())
        return wf.psf.sum(0)

    def model(
        self: OpticalSystem,
        source: Source,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array:
        """
        Models the input Source object through the optics.

        Parameters
        ----------
        source : Source
            The Source object to model through the optics.
        return_wf : bool = False
            Should the Wavefront object be returned instead of the psf Array?
        return_psf : bool = False
            Should the PSF object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront, PSF
            if `return_wf` is False and `return_psf` is False, returns the psf Array.
            if `return_wf` is True and `return_psf` is False, returns the Wavefront
                object.
            if `return_wf` is False and `return_psf` is True, returns the PSF object.
        """
        return source.model(self, return_wf, return_psf)


class ParametricOpticalSystem(OpticalSystem):
    """
    Implements the attributes required for an optical system with a specific output
    pixel scale and number of pixels.

    Attributes
    ----------
    psf_npixels : int
        The number of pixels of the final PSF.
    oversample : int
        The oversampling factor of the final PSF. Decreases the psf_pixel_scale
        parameter while increasing the psf_npixels parameter.
    psf_pixel_scale : float
        The pixel scale of the final PSF.
    """

    psf_npixels: int
    oversample: int
    psf_pixel_scale: float

    def __init__(
        self: OpticalSystem,
        psf_npixels: int,
        psf_pixel_scale: float,
        oversample: int = 1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        psf_npixels : int
            The number of pixels of the final PSF.
        psf_pixel_scale : float
            The pixel scale of the final PSF.
        oversample : int = 1.
            The oversampling factor of the final PSF. Decreases the psf_pixel_scale
            parameter while increasing the psf_npixels parameter.
        """
        self.psf_npixels = int(psf_npixels)
        self.oversample = int(oversample)
        self.psf_pixel_scale = float(psf_pixel_scale)
        super().__init__(**kwargs)


##################
# Public Classes #
##################
class LayeredOpticalSystem(OpticalSystem):
    """
    A flexible optical system that allows for the arbitrary chaining of OpticalLayers.

    ??? abstract "UML"
        ![UML](../../assets/uml/LayeredOpticalSystem.png)

    Attributes
    ----------
    wf_npixels : int
        The size of the initial wavefront to propagate.
    diameter : float, metres
        The diameter of the wavefront to propagate.
    layers : OrderedDict
        A series of `OpticalLayer` transformations to apply to wavefronts.
    """

    wf_npixels: int
    diameter: float
    layers: OrderedDict

    def __init__(
        self: OpticalSystem,
        wf_npixels: int,
        diameter: float,
        layers: list[OpticalLayer, tuple],
    ):
        """
        Parameters
        ----------
        wf_npixels : int
            The size of the initial wavefront to propagate.
        diameter : float
            The diameter of the wavefront to propagate.
        layers : list[OpticalLayer, tuple]
            A list of `OpticalLayer` transformations to apply to wavefronts. The list
            entries can be either `OpticalLayer` objects or tuples of (key, layer) to
            specify a key for the layer in the layers dictionary.
        """
        self.wf_npixels = int(wf_npixels)
        self.diameter = float(diameter)
        self.layers = dlu.list2dictionary(layers, True, OpticalLayer)

    def __getattr__(self: OpticalSystem, key: str) -> Any:
        """
        Raises both the individual layers and the attributes of the layers via
        their keys.

        Parameters
        ----------
        key : str
            The key of the item to be searched for in the layers dictionary.

        Returns
        -------
        item : object
            The item corresponding to the supplied key in the layers dictionary.
        """
        if key in self.layers.keys():
            return self.layers[key]
        for layer in list(self.layers.values()):
            if hasattr(layer, key):
                return getattr(layer, key)
        raise AttributeError(
            f"{self.__class__.__name__} has no attribute " f"{key}."
        )

    def propagate_mono(
        self: OpticalSystem,
        wavelength: Array,
        offset: Array = np.zeros(2),
        return_wf: bool = False,
    ) -> Array:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : float, metres
            The wavelength of the wavefront to propagate through the optical layers.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        return_wf: bool = False
            Should the Wavefront object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront
            if `return_wf` is False, returns the psf Array.
            if `return_wf` is True, returns the Wavefront object.
        """
        # Initialise wavefront
        wavefront = Wavefront(self.wf_npixels, self.diameter, wavelength)
        wavefront = wavefront.tilt(offset)

        # Apply layers
        for layer in list(self.layers.values()):
            wavefront *= layer

        # Return PSF or Wavefront
        if return_wf:
            return wavefront
        return wavefront.psf

    def insert_layer(
        self: OpticalSystem, layer: Union[OpticalLayer, tuple], index: int
    ) -> OpticalSystem:
        """
        Inserts a layer into the layers dictionary at a specified index. This function
        calls the list2dictionary function to ensure all keys remain unique. Note that
        this can result in some keys being modified if they are duplicates. The input
        'layer' can be a tuple of (key, layer) to specify a key, else the key is taken
        as the class name of the layer.

        Parameters
        ----------
        layer : Any
            The layer to be inserted.
        index : int
            The index at which to insert the layer.

        Returns
        -------
        optical_system : OpticalSystem
            The updated optical system.
        """
        return self.set(
            "layers", dlu.insert_layer(self.layers, layer, index, OpticalLayer)
        )

    def remove_layer(self: OpticalLayer, key: str) -> OpticalSystem:
        """
        Removes a layer from the layers dictionary, specified by its key.

        Parameters
        ----------
        key : str
            The key of the layer to be removed.

        Returns
        -------
        optical_system : OpticalSystem
            The updated optical system.
        """
        return self.set("layers", dlu.remove_layer(self.layers, key))


class AngularOpticalSystem(ParametricOpticalSystem, LayeredOpticalSystem):
    """
    An extension to the LayeredOpticalSystem class that propagates a wavefront to an
    image plane with `psf_pixel_scale` in units of arcseconds.

    ??? abstract "UML"
        ![UML](../../assets/uml/AngularOpticalSystem.png)

    Attributes
    ----------
    wf_npixels : int
        The number of pixels representing the wavefront.
    diameter : Array, metres
        The diameter of the initial wavefront to propagate.
    layers : OrderedDict
        A series of `OpticalLayer` transformations to apply to wavefronts.
    psf_npixels : int
        The number of pixels of the final PSF.
    psf_pixel_scale : float, arcseconds
        The pixel scale of the final PSF.
    oversample : int
        The oversampling factor of the final PSF. Decreases the psf_pixel_scale
        parameter while increasing the psf_npixels parameter.
    """

    def __init__(
        self: OpticalSystem,
        wf_npixels: int,
        diameter: float,
        layers: list[OpticalLayer, tuple],
        psf_npixels: int,
        psf_pixel_scale: float,
        oversample: int = 1,
    ):
        """
        Parameters
        ----------
        wf_npixels : int
            The number of pixels representing the wavefront.
        diameter : Array, metres
            The diameter of the initial wavefront to propagate.
        layers : list[OpticalLayer, tuple]
            A list of `OpticalLayer` transformations to apply to wavefronts. The list
            entries can be either `OpticalLayer` objects or tuples of (key, layer) to
            specify a key for the layer in the layers dictionary.
        psf_npixels : int
            The number of pixels of the final PSF.
        psf_pixel_scale : float, arcseconds
            The pixel scale of the final PSF in units of arcseconds.
        oversample : int
            The oversampling factor of the final PSF. Decreases the psf_pixel_scale
            parameter while increasing the psf_npixels parameter.
        """
        super().__init__(
            wf_npixels=wf_npixels,
            diameter=diameter,
            layers=layers,
            psf_npixels=psf_npixels,
            psf_pixel_scale=psf_pixel_scale,
            oversample=oversample,
        )

    def propagate_mono(
        self: OpticalSystem,
        wavelength: Array,
        offset: Array = np.zeros(2),
        return_wf: bool = False,
    ) -> Array:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : float, metres
            The wavelength of the wavefront to propagate through the optical layers.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        return_wf: bool = False
            Should the Wavefront object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront
            if `return_wf` is False, returns the psf Array.
            if `return_wf` is True, returns the Wavefront object.
        """
        wf = super().propagate_mono(wavelength, offset, return_wf=True)

        # Propagate
        true_pixel_scale = self.psf_pixel_scale / self.oversample
        pixel_scale = dlu.arcsec2rad(true_pixel_scale)
        psf_npixels = self.psf_npixels * self.oversample
        wf = wf.propagate(psf_npixels, pixel_scale)

        # Return PSF or Wavefront
        if return_wf:
            return wf
        return wf.psf


class CartesianOpticalSystem(ParametricOpticalSystem, LayeredOpticalSystem):
    """
    An extension to the LayeredOpticalSystem class that propagates a wavefront to an
    image plane with `psf_pixel_scale` in units of microns.

    ??? abstract "UML"
        ![UML](../../assets/uml/CartesianOpticalSystem.png)

    Attributes
    ----------
    wf_npixels : int
        The number of pixels representing the wavefront.
    diameter : Array, metres
        The diameter of the initial wavefront to propagate.
    layers : OrderedDict
        A series of `OpticalLayer` transformations to apply to wavefronts.
    focal_length : float, metres
        The focal length of the system.
    psf_npixels : int
        The number of pixels of the final PSF.
    psf_pixel_scale : float, microns
        The pixel scale of the final PSF.
    oversample : int
        The oversampling factor of the final PSF. Decreases the psf_pixel_scale
        parameter while increasing the psf_npixels parameter.
    """

    focal_length: None

    def __init__(
        self: OpticalSystem,
        wf_npixels: int,
        diameter: float,
        layers: list[OpticalLayer, tuple],
        focal_length: float,
        psf_npixels: int,
        psf_pixel_scale: float,
        oversample: int = 1,
    ):
        """
        Parameters
        ----------
        wf_npixels : int
            The number of pixels representing the wavefront.
        diameter : Array, metres
            The diameter of the initial wavefront to propagate.
        layers : list[OpticalLayer, tuple]
            A list of `OpticalLayer` transformations to apply to wavefronts. The list
            entries can be either `OpticalLayer` objects or tuples of (key, layer) to
            specify a key for the layer in the layers dictionary.
        focal_length : float, metres
            The focal length of the system.
        psf_npixels : int
            The number of pixels of the final PSF.
        psf_pixel_scale : float, microns
            The pixel scale of the final PSF in units of microns.
        oversample : int
            The oversampling factor of the final PSF. Decreases the psf_pixel_scale
            parameter while increasing the psf_npixels parameter.
        """
        self.focal_length = float(focal_length)

        super().__init__(
            wf_npixels=wf_npixels,
            diameter=diameter,
            layers=layers,
            psf_npixels=psf_npixels,
            psf_pixel_scale=psf_pixel_scale,
            oversample=oversample,
        )

    def propagate_mono(
        self: OpticalSystem,
        wavelength: Array,
        offset: Array = np.zeros(2),
        return_wf: bool = False,
    ) -> Array:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : float, metres
            The wavelength of the wavefront to propagate through the optical layers.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        return_wf: bool = False
            Should the Wavefront object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront
            if `return_wf` is False, returns the psf Array.
            if `return_wf` is True, returns the Wavefront object.
        """
        wf = super().propagate_mono(wavelength, offset, return_wf=True)

        # Propagate
        true_pixel_scale = self.psf_pixel_scale / self.oversample
        pixel_scale = 1e-6 * true_pixel_scale
        psf_npixels = self.psf_npixels * self.oversample
        wf = wf.propagate(psf_npixels, pixel_scale, self.focal_length)

        # Return PSF or Wavefront
        if return_wf:
            return wf
        return wf.psf
