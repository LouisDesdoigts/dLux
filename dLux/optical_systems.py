from __future__ import annotations
from typing import Any
from collections import OrderedDict
from abc import abstractmethod
import jax.numpy as np
from jax import Array
from zodiax import filter_vmap, Base
from typing import Union
import dLux.utils as dlu


__all__ = [
    "BaseOpticalSystem",
    "AngularOptics",
    "CartesianOptics",
    "LayeredOptics",
]

from .layers.optical_layers import OpticalLayer
from .containers.wavefronts import Wavefront
from .sources import BaseSource as Source
from .containers.psfs import PSF


###################
# Private Classes #
###################
class BaseOpticalSystem(Base):
    @abstractmethod
    def propagate_mono(
        self: BaseOpticalSystem,
        wavelength: Array,
        offset: Array,
        return_wf: bool,
    ) -> Array:  # pragma: no cover
        pass

    @abstractmethod
    def propagate(
        self: BaseOpticalSystem,
        wavelengths: Array,
        offset: Array,
        weights: Array,
        return_wf: bool,
    ):
        pass

    @abstractmethod
    def model(
        self: BaseOpticalSystem,
        # source: BaseSourceObject,
        return_wf: bool = False,
    ) -> Array:
        pass


class OpticalSystem(BaseOpticalSystem):
    """
    The Base Optics class that all optics classes inherit from. Can be used to
    create your own optics classes that will integrate seamlessly with the rest
    of dLux.

    This class implements three concrete methods and on abstract one. The
    concrete methods are `model(sources)`, which models dLux sources through
    the optics, `propagate(wavelengths, offset, weights)`, which propagates a
    polychromatic point source through the optics, and `__getattr__`, which
    allows for easy access to the attributes of the class.

    The abstract method is `propagate_mono(wavelength, offset, return_wf)`,
    which propagates a monochromatic point source through the optics. This
    is where the actual operations on the wavefront are performed. This
    method must be implemented by any class that inherits from `OpticalSystem`.
    """

    def __getattr__(self: OpticalSystem, key: str) -> Any:
        """
        Accessor for attributes of the class to simplify zodiax paths by
        searching for parameters in the attributes of the class.

        Parameters
        ----------
        key : str
            The key of the item to be searched for in the class.

        Returns
        -------
        item : object
            The item corresponding to the supplied key.
        """
        for attribute in self.__dict__.values():
            if hasattr(attribute, key):
                return getattr(attribute, key)
        else:
            raise AttributeError(
                f"{self.__class__.__name__} has no attribute " f"{key}."
            )

    # TODO: Move to base?
    def propagate_mono(
        self: AngularOptics,
        wavelength: float,
        offset: Array = np.zeros(2),
        return_wf: bool = False,
    ) -> Array:  # pragma: no cover
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : Array, metres
            The wavelength of the wavefront to propagate through the optical
            layers.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.
        get_pixel_scale : bool = False

        Returns
        -------
        psf : Array
            The monochromatic point spread function after being propagated
            though the optical layers.
        pixel_scale : float, radians

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
        offset : Array, radians, = np.zeros(2)
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.
        weights : Array, = None
            The weights of each wavelength. If None, all wavelengths are
            weighted equally.
        get_pixel_scale : bool = False


        Returns
        -------
        psf : Array
            The chromatic point spread function after being propagated
            though the optical layers. Only returned if return_object is False.
        pixel_scale : float, radians

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
        Models the input sources through the optics. The sources input can be
        a single Source object, or a list of Source objects.

        Parameters
        ----------
        source : Source
            The Source or Scene to model.
        return_wf : bool = False

        Returns
        -------
        psf : Array
            The sum of the individual sources modelled through the optics.

        """
        return source.model(self, return_wf, return_psf)


class ParametricOptics(OpticalSystem):
    """
    Implements the basics required for an optical system with a parametric PSF
    output sampling. Adds the `psf_npixels`, `psf_pixel_scale`, and
    `oversample` attributes.
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
            The oversampling factor of the final PSF.
        """
        self.psf_npixels = int(psf_npixels)
        self.oversample = int(oversample)
        self.psf_pixel_scale = float(psf_pixel_scale)
        super().__init__(**kwargs)


##################
# Public Classes #
##################
class LayeredOptics(OpticalSystem):
    """
    A fully flexible optical system that allows for the arbitrary chaining of
    dLux OpticalLayers.

    Attributes
    ----------
    wf_npixels : int
        The size of the initial wavefront to propagate.
    diameter : Array
        The diameter of the wavefront to model through the system in metres.
    layers : OrderedDict
        A collections.OrderedDict of 'layers' that define the transformations
        and operations upon some input wavefront through an optical system.
    """

    wf_npixels: int
    diameter: Array
    layers: OrderedDict

    def __init__(
        self: OpticalSystem, wf_npixels: int, diameter: float, layers: list
    ) -> OpticalSystem:
        """
        Constructor for the Optics class.

        Parameters
        ----------
        wf_npixels : int
            The number of pixels to use when propagating the wavefront through
            the optical system.
        diameter : float
            The diameter of the wavefront to model through the system in
            metres.
        layers : list
            A list of dLux 'layers' that define the transformations and
            operations upon some input wavefront through an optical system.
            The entries can either be dLux OpticalLayers, or tuples of the
            form (OpticalLayer, key), with the key being used as the dictionary
            key for the layer.
        """
        self.wf_npixels = int(wf_npixels)
        self.diameter = float(diameter)
        self.layers = dlu.list2dictionary(layers, True, OpticalLayer)

    def __getattr__(self: OpticalSystem, key: str) -> object:
        """
        Magic method designed to allow accessing of the various items within
        the layers dictionary of this class via the 'class.attribute' method.

        Parameters
        ----------
        key : str
            The key of the item to be searched for in the layers dictionary.

        Returns
        -------
        item : object
            The item corresponding to the supplied key in the layers
            dictionary.
        """
        if key in self.layers.keys():
            return self.layers[key]
        for layer in list(self.layers.values()):
            if hasattr(layer, key):
                return getattr(layer, key)
        super().__getattr__(key)

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
        wavelength : Array, metres
            The wavelength of the wavefront to propagate through the optical
            layers.
        offset : Array, radians, = np.zeros(2)
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.
        get_pixel_scale : bool = False

        Returns
        -------
        psf : Array
            The monochromatic point spread function after being propagated
            though the optical layers.
        pixel_scale : float, radians

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
        Inserts a layer into the layers dictionary at the given index using the
        list.insert method. Note this method may require the names of some
        parameters to be

        Parameters
        ----------
        layer : Union[OpticalLayer, tuple]
            The layer to insert into the layers dictionary. Can either be a
            single OpticalLayer, or you can specify the layers key by passing
            in a tuple of (OpticalLayer, key).
        index : int
            The index to insert the layer at.
        """
        return self.set("layers", dlu.insert_layer(layer, index, OpticalLayer))

    def remove_layer(self: OpticalLayer, key: str) -> OpticalSystem:
        """
        Removes a layer from the layers dictionary indexed at 'key' using the
        dict.pop(key) method.

        Parameters
        ----------
        key : str
            The key of the layer to remove.
        """
        return self.set("layers", dlu.remove_layer(key))


class AngularOptics(ParametricOptics, LayeredOptics):
    """
    A simple optical system that propagates a wavefront to an image plane
    with `psf_pixel_scale` in units of arcseconds.

    Attributes
    ----------
    wf_npixels : int
        The number of pixels representing the wavefront.
    diameter : Array, metres
        The diameter of the initial wavefront to propagate.
    aperture : Union[Array, OpticalLayer]
        The aperture of the system. Can be an Array or a OpticalLayer.
    mask : Union[Array, OpticalLayer]
        The mask to apply to the wavefront. Can be an Array or an OpticalLayer.
        If an Array it is treated as a transmissive mask.
    psf_pixel_scale : float
        The pixel scale of the final PSF.
    oversample : int
        The oversampling factor of the final PSF.
    psf_npixels : int
        The number of pixels of the final PSF.
    """

    def __init__(
        self: AngularOptics,
        wf_npixels: int,
        diameter: float,
        layers: list,
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
        aperture : Union[Array, OpticalLayer]
            The aperture of the system. Can be an Array or a OpticalLayer.
        psf_npixels : int
            The number of pixels of the final PSF.
        psf_pixel_scale : float, arcseconds
            The pixel scale of the final PSF in units of arcseconds.
        oversample : int
            The oversampling factor of the final PSF.
        mask : Union[Array, OpticalLayer] = None
            The mask to apply to the wavefront. Can be an Array or an
            OpticalLayer. If an Array it is treated as a transmissive mask.
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
        self: AngularOptics,
        wavelength: Array,
        offset: Array = np.zeros(2),
        return_wf: bool = False,
    ) -> Array:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : Array, metres
            The wavelength of the wavefront to propagate through the optical
            layers.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.
        get_pixel_scale : bool = False

        Returns
        -------
        psf : Array
            The monochromatic point spread function after being propagated
            though the optical layers.
        pixel_scale : float, radians

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


class CartesianOptics(ParametricOptics, LayeredOptics):
    """
    A simple optical system that propagates a wavefront to an image plane
    with `psf_pixel_scale` in units of microns.

    Attributes
    ----------
    wf_npixels : int
        The number of pixels representing the wavefront.
    diameter : Array, metres
        The diameter of the initial wavefront to propagate.
    focal_length : float, metres
        The focal length of the optical system.
    aperture : Union[Array, OpticalLayer]
        The aperture of the system. Can be an Array or a OpticalLayer.
    mask : Union[Array, OpticalLayer]
        The mask to apply to the wavefront. Can be an Array or an OpticalLayer.
        If an Array it is treated as a transmissive mask.
    psf_pixel_scale : float, microns
        The pixel scale of the final PSF.
    oversample : int
        The oversampling factor of the final PSF.
    psf_npixels : int
        The number of pixels of the final PSF.
    """

    focal_length: None

    def __init__(
        self: CartesianOptics,
        wf_npixels: int,
        diameter: float,
        layers: list,
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
        aperture : Union[Array, OpticalLayer]
            The aperture of the system. Can be an Array or a OpticalLayer.
        focal_length : float, metres
            The focal length of the optical system.
        psf_npixels : int
            The number of pixels of the final PSF.
        psf_pixel_scale : float, microns
            The pixel scale of the final PSF in units of microns.
        oversample : int
            The oversampling factor of the final PSF.
        mask : Union[Array, OpticalLayer] = None
            The mask to apply to the wavefront. Can be an Array or an
            OpticalLayer. If an Array it is treated as a transmissive mask.
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
        self: CartesianOptics,
        wavelength: Array,
        offset: Array = np.zeros(2),
        return_wf: bool = False,
    ) -> Array:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : Array, metres
            The wavelength of the wavefront to propagate through the optical
            layers.
        offset : Array, radians, = np.zeros(2)
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.
        get_pixel_scale : bool = False

        Returns
        -------
        psf : Array
            The monochromatic point spread function after being propagated
            though the optical layers.
        pixel_scale : float, radians

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
