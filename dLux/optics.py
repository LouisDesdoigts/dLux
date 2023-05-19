from __future__ import annotations
from abc import abstractmethod
import jax.numpy as np
from jax import vmap, Array
from jax.tree_util import tree_map, tree_flatten
from equinox import tree_at
from zodiax import Base
from typing import Union
import dLux.utils as dlu
import dLux


__all__ = ["AngularOptics", "CartesianOptics", "FlexibleOptics", 
    "LayeredOptics"]


# Alias classes for simplified type-checking
OpticalLayer  = lambda : dLux.optical_layers.OpticalLayer
Propagator    = lambda : dLux.propagators.Propagator
Source        = lambda : dLux.sources.BaseSource


#######################
### Private Classes ###
#######################
class BaseOptics(Base):
    """
    A base class for all Optics classes that implements a few usefull methods.

    All child classes must implement a `propagate_mono` method, with a
    signature matching the abstract method of this class.
    """


    def __getattr__(self : BaseOptics, key : str) -> Any:
        """
        Accessor for attributes of the class to simplify zodiax paths.

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
            raise AttributeError(f"{self.__class__.__name__} has no attribute "
            f"{key}.")


    @abstractmethod
    def propagate_mono(
        self       : BaseOptics,
        wavelength : Array,
        offset     : Array = np.zeros(2),
        return_wf  : bool = False) -> Array: # pragma: no cover
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : Array, meters
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


    def propagate(self        : BaseOptics, 
                  wavelengths : Array,
                  offset      : Array = np.zeros(2),
                  weights     : Array = None) -> Array:
        """
        Propagates a Polychromatic point source through the optics.

        Parameters
        ----------
        wavelengths : Array, meters
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
            weights = np.ones_like(wavelengths)/len(wavelengths)
        else:
            weights = np.atleast_1d(weights)

        # Check wavelengths and weights
        if weights.shape != wavelengths.shape:
            raise ValueError("wavelengths and weights must have the "
                f"same shape, got {wavelengths.shape} and {weights.shape} "
                "respectively.")
        
        # Check offset
        offset = np.array(offset) if not isinstance(offset, Array) \
            else offset
        if offset.shape != (2,):
            raise ValueError("offset must be a 2-element array, got "
                f"shape {offset.shape}.")

        # Caluculate
        propagator = vmap(self.propagate_mono, in_axes=(0, None))
        psfs = propagator(wavelengths, offset)
        if weights is not None:
            psfs *= weights[:, None, None]
        return psfs.sum(0)


    def model(self    : BaseOptics,
              sources : Union[list, Source]) -> Array:
        """
        Models the sources through the optics.
        Parameters
        ----------
        sources : Union[list, Source]
            The sources to model.

        Returns
        -------
        image : Array
            The sum of the individual source modelled through the optics.
        """
        if not isinstance(sources, list):
            sources = [sources]
        
        for source in sources:
            if not isinstance(source, Source()):
                raise TypeError("All input sources must be a Source "
                    f"object. Got type: {type(sources)})")
        
        return np.array([source.model(self) for source in sources]).sum(0)

class SimpleOptics(BaseOptics):
    """
    A Simple Optical system that initialises a wavefront based on the wavefront
    diameter and npixels.

    Attributes
    ----------
    wf_npixels : int
        The nuber of pixels of the initial wavefront to propagte.
    diameter : Array, meters
        The diameter of the initial wavefront to propagte.
    """
    wf_npixels  : int
    diameter    : Array


    def __init__(
        self        : BaseOptics, 
        wf_npixels  : int,
        diameter    : float,
        **kwargs):
        """

        Parameters
        ----------
        wf_npixels : int
            The number of pixels representing the wavefront.
        diameter : Array, meters
            The diameter of the initial wavefront to propagte.
        """
        self.wf_npixels = int(wf_npixels)
        self.diameter = float(diameter)

        super().__init__(**kwargs)


    def _construct_wavefront(
        self       : BaseOptics,
        wavelength : Array,
        offset     : Array = np.zeros(2)) -> Array:
        """
        Constructs the appropriate tilted wavefront object for the optical
        system.

        Parameters
        ----------
        wavelength : Array, meters
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
    
    """
    psf_npixels     : int
    psf_oversample  : float
    psf_pixel_scale : float


    def __init__(
        self,
        psf_npixels,
        psf_pixel_scale,
        psf_oversample=1,
        **kwargs):
        """

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
    """
    aperture : Union[Array, OpticalLayer()]
    mask     : Union[Array, OpticalLayer()]


    def __init__(
        self     : BaseOptics, 
        aperture : Union[Array, OpticalLayer()],
        mask     : Union[Array, OpticalLayer()] = None,
        **kwargs):
        """
        Constructs a simple optical system with an aperutre and a mask.

        Note this class automatically converts aperture input into an array.

        Parameters
        ----------
        aperture : Union[Array, OpticalLayer]
            The aperture of the system. Can be an Array or a OpticalLayer.
        mask : Union[Array, OpticalLayer], = None
            The mask to apply to the wavefront. Can be an Array or an
            OpticalLayer. Default is None.
        """
        if not isinstance(aperture, (Array, OpticalLayer())):
            raise TypeError("aperture must be an Array or "
                f"OpticalLayer, got {type(aperture)}.")
        self.aperture = aperture

        if mask is not None:
            if not isinstance(mask, (Array, OpticalLayer())):
                raise TypeError("mask must be an Array or "
                    f"OpticalLayer, got {type(aperture)}.")
        self.mask = mask

        super().__init__(**kwargs)


    def _apply_aperture(self, wavelength, offset):
        """

        """
        wf = self._construct_wavefront(wavelength, offset)
        wf *= self.aperture
        wf = wf.normalise()
        wf *= self.mask
        return wf

######################
### Public Classes ###
######################
class AngularOptics(NonPropagatorOptics, AperturedOptics, SimpleOptics):
    """

    """


    def __init__(self, 
        wf_npixels,
        diameter,
        aperture,
        psf_npixels,
        psf_pixel_scale,
        psf_oversample = 1,
        mask = None):
        """

        """
        super().__init__(wf_npixels=wf_npixels, diameter=diameter, 
            aperture=aperture, psf_npixels=psf_npixels, mask=mask,
            psf_pixel_scale=psf_pixel_scale, psf_oversample=psf_oversample)


    def propagate_mono(self       : SimpleToliman,
                       wavelength : Array,
                       offset     : Array = np.zeros(2),
                       return_wf  : bool = False) -> Array:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : Array, meters
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
    
    """
    focal_length : None


    def __init__(
        self,
        wf_npixels,
        diameter,
        aperture,
        focal_length,
        psf_npixels,
        psf_pixel_scale,
        psf_oversample = 1,
        mask = None):
        """

        """
        self.focal_length = float(focal_length)

        super().__init__(wf_npixels=wf_npixels, diameter=diameter,
            aperture=aperture, psf_npixels=psf_npixels,
            psf_pixel_scale=psf_pixel_scale, psf_oversample=psf_oversample,
            mask=mask)


    def propagate_mono(self       : SimpleToliman,
                       wavelength : Array,
                       offset     : Array = np.zeros(2),
                       return_wf  : bool = False) -> Array:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : Array, meters
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
        wf = wf.MFT(self.psf_npixels, pixel_scale, 
            focal_length=self.focal_length)

        # Return PSF or Wavefront
        if return_wf:
            return wf
        return wf.psf


class FlexibleOptics(AperturedOptics, SimpleOptics):
    propagator : None

    def __init__(
        self, 
        wf_npixels,
        diameter, 
        aperture, 
        propagator, 
        mask = None):
        """

        """
        if not isinstance(propagator, Propagator()):
            raise TypeError("propagator must be a Propagator object, "
                f"got {type(propagator)}.")
        self.propagator = propagator
        super().__init__(wf_npixels=wf_npixels, diameter=diameter,
            aperture=aperture, mask=mask)


    @property
    def true_pixel_scale(self):
        """
        Returns the true pixel scale of the PSF.
        """
        return self.propagator.pixel_scale

    def propagate_mono(self       : SimpleToliman,
                       wavelength : Array,
                       offset     : Array = np.zeros(2),
                       return_wf  : bool = False) -> Array:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : Array, meters
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
    A high level class desgined to model the behaviour of some optical systems
    response to wavefronts.

    Attributes
    ----------
    wf_npixels : int
        The size of the initial wavefront to propagte.
    diameter : Array
        The diameter of the wavefront to model through the system in meters.
    layers: dict
        A collections.OrderedDict of 'layers' that define the transformations
        and operations upon some input wavefront through an optical system.
    """
    layers : OrderedDict


    def __init__(self       : Optics, 
                 wf_npixels : int, 
                 diameter   : float, 
                 layers     : list) -> Optics:
        """
        Constructor for the Optics class.

        Parameters
        ----------
        wf_npixels : int
            The number of pixels to use when propagating the wavefront through
            the optical system.
        diameter : float
            The diameter of the wavefront to model through the system in meters.
        layers : list
            A list of ∂Lux 'layers' that define the transformations and
            operations upon some input wavefront through an optical system.
            The entried can either be dLux OtpicalLayers, or tuples of the
            form (OpticalLayer, key), with the key being used as the dictionary
            key for the layer.
        """
        super().__init__(wf_npixels=wf_npixels, diameter=diameter)
        self.layers = dlu.list_to_dictionary(layers, True, OpticalLayer())


    def __getattr__(self : Optics, key : str) -> object:
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
            The item corresponding to the supplied key in the layers dictionary.
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
        self       : BaseOptics,
        wavelength : Array,
        offset     : Array = np.zeros(2),
        return_wf  : bool = False) -> Array:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : Array, meters
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