from __future__ import annotations
from abc import abstractmethod
from typing import Any
from collections import OrderedDict
import jax.numpy as np
from jax import vmap, Array
from zodiax import Base
from typing import Union
import dLux.utils as dlu
import dLux

__all__ = [
    "AngularOptics",
    "CartesianOptics",
    "FlexibleOptics",
    "LayeredOptics",
]

# Alias classes for simplified type-checking
OpticalLayer = lambda: dLux.optical_layers.OpticalLayer
Propagator = lambda: dLux.propagators.Propagator
Source = lambda: dLux.sources.BaseSource
Wavefront = lambda: dLux.wavefronts.Wavefront


###################
# Private Classes #
###################
class BaseOptics(Base):
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
    method must be implemented by any class that inherits from `BaseOptics`.
    """

    def __getattr__(self: BaseOptics, key: str) -> Any:
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

    @abstractmethod
    def propagate_mono(
        self: BaseOptics,
        wavelength: Array,
        offset: Array = np.zeros(2),
        return_wf: bool = False,
    ) -> Array:  # pragma: no cover
        """
        Propagates a monochromatic point source through the optics.

        Parameters
        ----------
        wavelength : Array, metres
            The wavelength of the wavefront to propagate through the optical
            layers.
        offset : Array, radians, = np.zeros(2)
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.
        return_wf : bool, = False
            If True, the wavefront object after propagation is returned.

        Returns
        -------
        psf : Array
            The monochromatic point spread function after being propagated
            though the optical layers.
        wavefront : Wavefront
            The wavefront object after propagation. Only returned if
            return_wf is True.
        """

    def propagate(
        self: BaseOptics,
        wavelengths: Array,
        offset: Array = np.zeros(2),
        weights: Array = None,
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

        Returns
        -------
        psf : Array
            The chromatic point spread function after being propagated
            though the optical layers.
        """
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

        # Calculate
        propagator = vmap(self.propagate_mono, in_axes=(0, None))
        psfs = propagator(wavelengths, offset)
        if weights is not None:
            psfs *= weights[:, None, None]
        return psfs.sum(0)

    def model(self: BaseOptics, sources: Union[list, Source]) -> Array:
        """
        Models the input sources through the optics. The sources input can be
        a single Source object, or a list of Source objects.

        Parameters
        ----------
        sources : Union[list, Source]
            The sources to model.

        Returns
        -------
        psf : Array
            The sum of the individual sources modelled through the optics.
        """
        if not isinstance(sources, list):
            sources = [sources]

        for source in sources:
            if not isinstance(source, Source()):
                raise TypeError(
                    "All input sources must be a Source "
                    f"object. Got type: {type(sources)})"
                )

        return np.array([source.model(self) for source in sources]).sum(0)


class SimpleOptics(BaseOptics):
    """
    A Simple Optical system that initialises a wavefront based on the wavefront
    diameter and npixels. It adds two attributes, `wf_npixels` and `diameter`,
    as well as the `_construct_wavefront` method that constructs and tilts the
    initial wavefront.

    Attributes
    ----------
    wf_npixels : int
        The number of pixels of the initial wavefront to propagate.
    diameter : Array, metres
        The diameter of the initial wavefront to propagate.
    """

    wf_npixels: int
    diameter: Array

    def __init__(self: BaseOptics, wf_npixels: int, diameter: float, **kwargs):
        """
        Parameters
        ----------
        wf_npixels : int
            The number of pixels representing the wavefront.
        diameter : Array, metres
            The diameter of the initial wavefront to propagate.
        """
        self.wf_npixels = int(wf_npixels)
        self.diameter = float(diameter)

        super().__init__(**kwargs)

    def _construct_wavefront(
        self: BaseOptics, wavelength: Array, offset: Array = np.zeros(2)
    ) -> Array:
        """
        Constructs the appropriate tilted wavefront object for the optical
        system.

        Parameters
        ----------
        wavelength : Array, metres
            The wavelength of the wavefront to propagate through the optics.
        offset : Array, radians, = np.zeros(2)
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.

        Returns
        -------
        wavefront : Wavefront
            The wavefront object to propagate through the optics.
        """
        wf = dLux.Wavefront(self.wf_npixels, self.diameter, wavelength)
        return wf.tilt(offset)


class NonPropagatorOptics(BaseOptics):
    """
    Implements the basics required for an optical system with a parametric PSF
    output sampling. Adds the `psf_npixels`, `psf_pixel_scale`, and
    `psf_oversample` attributes.

    Attributes
    ----------
    psf_npixels : int
        The number of pixels of the final PSF.
    psf_pixel_scale : float
        The pixel scale of the final PSF.
    psf_oversample : float
        The oversampling factor of the final PSF.
    """

    psf_npixels: int
    psf_oversample: float
    psf_pixel_scale: float

    def __init__(
        self: BaseOptics,
        psf_npixels: int,
        psf_pixel_scale: float,
        psf_oversample: float = 1.0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        psf_npixels : int
            The number of pixels of the final PSF.
        psf_pixel_scale : float
            The pixel scale of the final PSF.
        psf_oversample : float = 1.
            The oversampling factor of the final PSF.
        """
        self.psf_npixels = int(psf_npixels)
        self.psf_oversample = float(psf_oversample)
        self.psf_pixel_scale = float(psf_pixel_scale)
        super().__init__(**kwargs)

    @property
    def true_pixel_scale(self):
        """
        Returns the true pixel scale of the PSF.
        """
        return dlu.arcsec_to_rad(self.psf_pixel_scale / self.psf_oversample)


class AperturedOptics(BaseOptics):
    """
    Constructs a simple optical system  with an aperture and an optional
    'mask'.

    Attributes
    ----------
    aperture : Union[Array, OpticalLayer]
        The aperture of the system. Can be an Array or a OpticalLayer.
    mask : Union[Array, OpticalLayer]
        The mask to apply to the wavefront. Can be an Array or an OpticalLayer.
        If an Array it is treated as a transmissive mask.
    """

    aperture: Union[Array, OpticalLayer()]
    mask: Union[Array, OpticalLayer()]

    def __init__(
        self: BaseOptics,
        aperture: Union[Array, OpticalLayer()],
        mask: Union[Array, OpticalLayer()] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        aperture : Union[Array, OpticalLayer]
            The aperture of the system. Can be an Array or a OpticalLayer.
        mask : Union[Array, OpticalLayer], = None
            The mask to apply to the wavefront. Can be an Array or an
            OpticalLayer. If an Array it is treated as a transmissive mask.
            Default is None.
        """
        if not isinstance(aperture, (Array, OpticalLayer())):
            raise TypeError(
                "aperture must be an Array or "
                f"OpticalLayer, got {type(aperture)}."
            )
        self.aperture = aperture

        if mask is not None:
            if not isinstance(mask, (Array, OpticalLayer())):
                raise TypeError(
                    "mask must be an Array or "
                    f"OpticalLayer, got {type(aperture)}."
                )
        self.mask = mask

        super().__init__(**kwargs)

    def _apply_aperture(
        self: BaseOptics, wavelength: float, offset: Array
    ) -> Wavefront():
        """
        Constructs the wavefront, applies the aperture and mask, and returns
        the wavefront.

        Parameters
        ----------
        wavelength : Array, metres
            The wavelength of the wavefront to propagate through the optics.
        offset : Array, radians
            The (x, y) offset from the optical axis of the source.
        """
        wf = self._construct_wavefront(wavelength, offset)
        wf *= self.aperture
        wf = wf.normalise()
        wf *= self.mask
        return wf


##################
# Public Classes #
##################
class AngularOptics(NonPropagatorOptics, AperturedOptics, SimpleOptics):
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
    psf_oversample : float
        The oversampling factor of the final PSF.
    psf_npixels : int
        The number of pixels of the final PSF.
    """

    def __init__(
        self: AngularOptics,
        wf_npixels: int,
        diameter: float,
        aperture: Union[Array, OpticalLayer()],
        psf_npixels: int,
        psf_pixel_scale: float,
        psf_oversample: float = 1,
        mask: Union[Array, OpticalLayer()] = None,
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
        psf_oversample : float
            The oversampling factor of the final PSF.
        mask : Union[Array, OpticalLayer] = None
            The mask to apply to the wavefront. Can be an Array or an
            OpticalLayer. If an Array it is treated as a transmissive mask.
        """
        super().__init__(
            wf_npixels=wf_npixels,
            diameter=diameter,
            aperture=aperture,
            psf_npixels=psf_npixels,
            mask=mask,
            psf_pixel_scale=psf_pixel_scale,
            psf_oversample=psf_oversample,
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
        offset : Array, radians, = np.zeros(2)
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.
        return_wf : bool, = False
            If True, the wavefront object after propagation is returned.

        Returns
        -------
        psf : Array
            The monochromatic point spread function after being propagated
            though the optical layers.
        wavefront : Wavefront
            The wavefront object after propagation. Only returned if
            return_wf is True.
        """
        wf = self._apply_aperture(wavelength, offset)

        # Propagate
        pixel_scale = self.psf_pixel_scale / self.psf_oversample
        pixel_scale_radians = dlu.arcsec_to_rad(pixel_scale)
        wf = wf.MFT(self.psf_npixels, pixel_scale_radians)

        # Return PSF or Wavefront
        if return_wf:
            return wf
        return wf.psf


class CartesianOptics(NonPropagatorOptics, AperturedOptics, SimpleOptics):
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
    psf_oversample : float
        The oversampling factor of the final PSF.
    psf_npixels : int
        The number of pixels of the final PSF.
    """

    focal_length: None

    def __init__(
        self: CartesianOptics,
        wf_npixels: int,
        diameter: float,
        aperture: Union[Array, OpticalLayer()],
        focal_length: float,
        psf_npixels: int,
        psf_pixel_scale: float,
        psf_oversample: int = 1,
        mask: Union[Array, OpticalLayer()] = None,
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
        psf_oversample : float
            The oversampling factor of the final PSF.
        mask : Union[Array, OpticalLayer] = None
            The mask to apply to the wavefront. Can be an Array or an
            OpticalLayer. If an Array it is treated as a transmissive mask.
        """
        self.focal_length = float(focal_length)

        super().__init__(
            wf_npixels=wf_npixels,
            diameter=diameter,
            aperture=aperture,
            psf_npixels=psf_npixels,
            psf_pixel_scale=psf_pixel_scale,
            psf_oversample=psf_oversample,
            mask=mask,
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
        return_wf : bool, = False
            If True, the wavefront object after propagation is returned.

        Returns
        -------
        psf : Array
            The monochromatic point spread function after being propagated
            though the optical layers.
        wavefront : Wavefront
            The wavefront object after propagation. Only returned if
            return_wf is True.
        """
        wf = self._apply_aperture(wavelength, offset)

        # Propagate
        pixel_scale = 1e-6 * self.psf_pixel_scale / self.psf_oversample
        wf = wf.MFT(
            self.psf_npixels, pixel_scale, focal_length=self.focal_length
        )

        # Return PSF or Wavefront
        if return_wf:
            return wf
        return wf.psf


class FlexibleOptics(AperturedOptics, SimpleOptics):
    """
    A simple optical system that propagates a wavefront to an image plane
    using the user-supplied propagator. This allows for propagation of fresnel
    wavefronts.

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
    propagator : Propagator
        The propagator to use to propagate the wavefront through the optics.
    """

    propagator: None

    def __init__(
        self: BaseOptics,
        wf_npixels: int,
        diameter: float,
        aperture: Union[Array, OpticalLayer()],
        propagator: Propagator(),
        mask: Union[Array, OpticalLayer()] = None,
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
        propagator : Propagator
            The propagator to use to propagate the wavefront through the
            optics.
        mask : Union[Array, OpticalLayer] = None
            The mask to apply to the wavefront. Can be an Array or an
            OpticalLayer. If an Array it is treated as a transmissive mask.
        """
        if not isinstance(propagator, Propagator()):
            raise TypeError(
                "propagator must be a Propagator object, "
                f"got {type(propagator)}."
            )
        self.propagator = propagator
        super().__init__(
            wf_npixels=wf_npixels,
            diameter=diameter,
            aperture=aperture,
            mask=mask,
        )

    @property
    def true_pixel_scale(self):
        """
        Returns the true pixel scale of the PSF.
        """
        return self.propagator.pixel_scale

    def _construct_wavefront(
        self: BaseOptics, wavelength: Array, offset: Array = np.zeros(2)
    ) -> Array:
        """
        Constructs the appropriate tilted wavefront object for the optical
        system.

        Parameters
        ----------
        wavelength : Array, metres
            The wavelength of the wavefront to propagate through the optics.
        offset : Array, radians, = np.zeros(2)
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.

        Returns
        -------
        wavefront : Wavefront
            The wavefront object to propagate through the optics.
        """
        if isinstance(self.propagator, dLux.propagators.FarFieldFresnel):
            wf = dLux.FresnelWavefront(
                self.wf_npixels, self.diameter, wavelength
            )
        else:
            wf = dLux.Wavefront(self.wf_npixels, self.diameter, wavelength)
        return wf.tilt(offset)

    def propagate_mono(
        self: BaseOptics,
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
        return_wf : bool, = False
            If True, the wavefront object after propagation is returned.

        Returns
        -------
        psf : Array
            The monochromatic point spread function after being propagated
            though the optical layers.
        wavefront : Wavefront
            The wavefront object after propagation. Only returned if
            return_wf is True.
        """
        wf = self._apply_aperture(wavelength, offset)
        wf = self.propagator(wf)

        # Return PSF or Wavefront
        if return_wf:
            return wf
        return wf.psf


class LayeredOptics(SimpleOptics):
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

    layers: OrderedDict

    def __init__(
        self: BaseOptics, wf_npixels: int, diameter: float, layers: list
    ) -> BaseOptics:
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
        super().__init__(wf_npixels=wf_npixels, diameter=diameter)
        self.layers = dlu.list_to_dictionary(layers, True, OpticalLayer())

    def __getattr__(self: BaseOptics, key: str) -> object:
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
        super().__getattr__(key)

    @property
    def true_pixel_scale(self):
        """
        Returns the true pixel scale of the PSF.
        """
        # Note: This is a bit inefficient, but should work
        for layer in self.layers.values():
            if isinstance(layer, Propagator()):
                propagator = layer
        return propagator.pixel_scale

    def propagate_mono(
        self: BaseOptics,
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
        return_wf : bool, = False
            If True, the wavefront object after propagation is returned.

        Returns
        -------
        psf : Array
            The monochromatic point spread function after being propagated
            though the optical layers.
        wavefront : Wavefront
            The wavefront object after propagation. Only returned if
            return_wf is True.
        """
        wavefront = self._construct_wavefront(wavelength, offset)
        for layer in list(self.layers.values()):
            wavefront *= layer

        if return_wf:
            return wavefront
        return wavefront.psf
