from __future__ import annotations
import jax.numpy as np
from jax import vmap, Array
from jax.tree_util import tree_map, tree_flatten
from equinox import tree_at
from zodiax import Base
from collections import OrderedDict
from copy import deepcopy
from inspect import signature
from typing import Union
from warnings import warn
from abc import abstractmethod
import dLux
import dLux.utils as dlu


__all__ = ["AngularOptics", "CartesianOptics", "FlexibleOptics", 
    "LayeredOptics"]


# Alias classes for simplified type-checking
CreateWavefront   = lambda : dLux.optical_layers.CreateWavefront
TransmissiveOptic = lambda : dLux.optical_layers.TransmissiveOptic
AberrationLayer   = lambda : dLux.optical_layers.AberrationLayer
OpticalLayer      = lambda : dLux.optical_layers.OpticalLayer
AddOPD            = lambda : dLux.optical_layers.AddOPD
AddPhase          = lambda : dLux.optical_layers.AddPhase
Propagator        = lambda : dLux.propagators.Propagator
FarFieldFresnel   = lambda : dLux.propagators.FarFieldFresnel
Source            = lambda : dLux.sources.BaseSource
ShapedOptic       = lambda : dLux.optical_layer.ShapedOptic


# Base
# # LayeredOptics
# # SimpleOptics
# # # Angular
# # # Cartesian
# # # Flexible (propagator)

class BaseOptics(Base):
    """
    A base class for all Optics classes that implements a few usefull methods.

    All child classes must implement a `propagate_mono` method, with a
    signature matching the abstract method of this class.
    """


    def __getattr__(self : BaseOptics, key : str) -> Any:
        """
        Accessor for attributes of the class to simplify zodiax paths
        
        NOTE: Will not work properly if multiple attributes have the same 
        parmeter name. Also works with @property methods

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
        pass


    def _format_input(self        : BaseOptics,
                      wavelengths : Array,
                      offset      : Array = None,
                      weights     : Array = None) -> Array:
        """
        Formats the weights and wavelengths of the polychromatic wavefronts.

        Parameters
        ----------
        wavelengths : Array, meters
            The wavelengths of the wavefronts to propagate through the optics.
        weights : Array
            The weights of each wavelength. If None, all wavelengths are
            weighted equally.
        offset : Array, radians
            The (x, y) offset from the optical axis of the source.

        Returns
        -------
        wavelengths : Array
            The wavelengths of the wavefronts to propagate through the optics.
        offset : Array, radians, = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        weights : Array
            The weights of each wavelength. If None, all wavelengths are
            weighted equally.
        """
        # Check wavelengths
        if isinstance(wavelengths, float) or \
            (isinstance(wavelengths, Array) and wavelengths.shape == ()):
            wavelengths = np.array([wavelengths])
        elif isinstance(wavelengths, list):
            wavelengths = np.array(wavelengths)

        # Check weights
        if weights is not None:
            weights = np.array(weights, dtype=float) \
                if not isinstance(weights, np.ndarray) else weights
            if len(weights) != len(wavelengths):
                raise ValueError("wavelengths and weights must have the "
                    f"same length, got {len(wavelengths)} and {len(weights)} "
                    "respectively.")
        
        # Check offset
        offset = np.array(offset) if not isinstance(offset, Array) \
            else offset
        if offset.shape != (2,):
            raise ValueError("offset must be a 2-element array, got "
                f"shape {offset.shape}.")

        # Return
        return wavelengths, weights, offset


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
        wavelengths, weights, offset = self._format_input(wavelengths, offset, 
            weights)

        # Construct Propagate
        propagator = vmap(self.propagate_mono, in_axes=(0, None))

        # Calc and Return
        psfs = propagator(wavelengths, offset)
        if weights is not None:
            psfs *= weights[:, None, None]
        return psfs.sum(0)


    def model(self        : BaseOptics,
              sources     : Union[dict, list, Source],
              return_tree : bool = False) -> Array:
        """
        A base level modelling function designed to robustly handle the different
        combinations of inputs. Models the sources through the instrument optics
        and detector. Users must provide optics and source.

        Parameters
        ----------
        sources : Union[dict, list, Source]
            The sources to model.
        return_tree : bool = False
            Whether to return a Pytree like object with matching tree structure as
            the input scene/sources/source. Default is False.

        Returns
        -------
        image : Array, Pytree
            The image of the scene modelled through the optics with detector and
            filter effects applied if they are supplied. Returns either as a single
            array (if return_tree is false), or a pytree like object with matching
            tree strucutre as the input scene/sources/source.
        """
        # Check valid inputs
        if isinstance(sources, Source()):
            sources = [sources]
        elif isinstance(sources, (dict, list, tuple)):
            source_vals = sources.values() if isinstance(sources, dict) else sources
            for source in source_vals:
                if not isinstance(source, Source()):
                    raise ValueError("sources must be a Source object, dict, list, "
                        f"or tuple object of sources. Got type: {type(sources)})")

        # Call the source.model() method to generate the psfs
        model_fn = lambda source: source.model(self)
        _is_source = lambda leaf: isinstance(leaf, Source())
        psfs = tree_map(model_fn, sources, is_leaf=_is_source)
        return psfs if return_tree else np.array(tree_flatten(psfs)[0]).sum(0)


class SimpleOptics(BaseOptics):
    """
    
    Attributes
    ----------
    diameter : Array
        The diameter of the wavefront to model through the system in meters.
    aperture : Union[Array, TransmissiveOptic()]
        The aperture of the system. Can be an Array or a TransmissiveOptic.
    propagator : Propagator()
        The propagator to use to propagate the wavefront through the system.
    aberrations : Union[Array, AberrationLayer()]
        The aberrations to apply to the wavefront. Can be an Array or an
        AberrationLayer, or defaults to None.
    """
    diameter    : Array
    aperture    : Union[Array, TransmissiveOptic()]
    aberrations : Union[Array, AberrationLayer()]
    mask        : Union[Array, OpticalLayer()]


    def __init__(self : SimpleOptics, 
                 diameter : Array, 
                 aperture : Union[Array, TransmissiveOptic()],
                 aberrations : Union[Array, AberrationLayer()] = None,
                 mask  : Union[Array, OpticalLayer()]= None
                 ) -> SimpleOptics:
        """
        Constructs a simple optical system with a static aperture and
        aberrations.

        Note this class automatically converts aperture input into an array.

        Parameters
        ----------
        diameter : Array
            The diameter of the wavefront to model through the system in meters.
        aperture : Union[Array, TransmissiveOptic()]
            The aperture of the system. Can be an Array or a TransmissiveOptic.
        propagator : Propagator()
            The propagator to use to propagate the wavefront through the system.
        aberrations : Union[Array, AberrationLayer()] = None
            The aberrations to apply to the wavefront. Can be an Array or an
            AberrationLayer. If None, no aberrations are applied.
        """
        super().__init__()
        
        # Diameter Checking
        if isinstance(diameter, Array):
            if diameter.ndim != 0:
                raise ValueError("diameter must be a scalar, got shape"
                    f"{diameter.shape}.")
        elif isinstance(diameter, int):
            diameter = float(diameter)
        elif not isinstance(diameter, float):
            raise TypeError("diameter must be a scalar, got type"
                f"{type(diameter)}.")
        self.diameter = diameter

        ### Aperture ###
        # Type Check
        if not isinstance(aperture, (Array, TransmissiveOptic())):
            raise TypeError("aperture must be an Array or "
                f"TransmissveOptic, got {type(aperture)}.")
        
        # Automatically Convert transmissive optics to arrays for simplicity
        if hasattr(aperture, 'transmission'):
            aperture = aperture.transmission
        self.aperture = aperture
        ap_shape = self.aperture.shape

        ### Aberrations ###
        if aberrations is not None:

            # Type Check
            if not isinstance(aberrations, (AberrationLayer(), AddOPD(), 
                AddPhase())):
                raise TypeError("aberrations must be an AberrationLayer, "
                    f"AddPhase, or AddOPD got {type(aberrations)}.")

            # Shape Check
            if (hasattr(aberrations, 'shape') and 
                (ap_shape != aberrations.shape)):
                raise ValueError("Inconsistent array sizes found: aperture has "
                    f"shape {ap_shape} and aberrations has shape {mask.shape}")
        self.aberrations = aberrations


        ### Mask ###
        if mask is not None:
            
            # Type check
            if not isinstance(mask, (Array, OpticalLayer())):
                raise ValueError("mask must be an Array or OpticalLayer, "
                    f"got {type(mask)}.")

            # Shape Check
            if hasattr(mask, 'shape') and (ap_shape != mask.shape):
                raise ValueError("Inconsistent array sizes found: aperture has "
                    f"shape {ap_shape} and mask has shape {mask.shape}")
        self.mask = mask


    def _construct_wavefront(self       : BaseOptics,
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
        # Get correct wavefront type
        if isinstance(self.propagator, FarFieldFresnel()):
            wf_constructor = dLux.FresnelWavefront
        else:
            wf_constructor = dLux.Wavefront
        
        # Construct and tilt
        wf = wf_constructor(self.aperture.shape[-1], self.diameter, wavelength)
        return wf.tilt(offset)


    def _apply_aperture(self, wavelength, offset):
        # Construct and tilt the Wavefront
        wf = self._construct_wavefront(wavelength, offset)

        # Apply aperture and normalise
        wf *= self.aperture
        wf = wf.normalise()

        # Apply Mask and Aberrations
        wf *= self.mask
        wf *= self.aberrations

        return wf


class NonPropagatorOptics(SimpleOptics):
    psf_npixels     : int
    psf_oversample  : float
    psf_pixel_scale : float

    def __init__(self, 
        diameter,
        aperture,
        psf_npixels,
        psf_pixel_scale, # arcseconds
        psf_oversample = 1,
        aberrations = None,
        mask = None,
        ) -> SimpleOptics:
        """
        """
        # Propagator Properties
        self.psf_npixels = int(psf_npixels)
        self.psf_oversample = float(psf_oversample)
        self.psf_pixel_scale = float(psf_pixel_scale)

        super().__init__(diameter=diameter, aperture=aperture, 
            aberrations=aberrations, mask=mask)

    def _construct_wavefront(self       : BaseOptics,
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
        # Construct and tilt
        wf = dLux.Wavefront(self.aperture.shape[-1], self.diameter, wavelength)
        return wf.tilt(offset)

class AngularOptics(NonPropagatorOptics):
    """

    """


    def __init__(self, 
        diameter,
        aperture,
        psf_npixels,
        psf_pixel_scale, # arcseconds
        psf_oversample = 1,
        aberrations = None,
        mask = None,
        ) -> SimpleOptics:
        """
        """
        super().__init__(diameter=diameter, aperture=aperture, 
            psf_npixels=psf_npixels, psf_pixel_scale=psf_pixel_scale, 
            psf_oversample=psf_oversample, aberrations=aberrations, mask=mask)


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
        # wf = self._construct_wavefront(wavelength, offset)
        wf = self._apply_aperture(wavelength, offset)

        # Propagate
        pixel_scale = self.psf_pixel_scale / self.psf_oversample
        pixel_scale_radians = dlu.arcsec_to_rad(pixel_scale)
        wf = wf.MFT(self.psf_npixels, pixel_scale_radians)

        # Return PSF or Wavefront
        if return_wf:
            return wf
        return wf.psf

class CartesianOptics(SimpleOptics):
    focal_length : None

    def __init__(self, diameter, aperture, propagator, focal_length, 
        aberrations = None, mask = None):
        """

        """
        self.focal_length = focal_length
        super().__init__(diameter=diameter, aperture=aperture,
            aberrations=aberrations, mask=mask)

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
        # wf = self._construct_wavefront(wavelength, offset)
        wf = self._apply_aperture(wavelength, offset)

        # Propagate
        pixel_scale = self.psf_pixel_scale / self.psf_oversample
        wf = wf.MFT(self.psf_npixels, pixel_scale, 
            focal_length=self.focal_length)

        # Return PSF or Wavefront
        if return_wf:
            return wf
        return wf.psf

class FlexibleOptics(SimpleOptics):
    propagator : None

    def __init__(self, diameter, aperture, propagator, aberrations = None, 
        mask = None):
        """

        """
        self.propagator = propagator
        super().__init__(diameter=diameter, aperture=aperture,
            aberrations=aberrations, mask=mask)


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


class LayeredOptics(BaseOptics):
    """
    A high level class desgined to model the behaviour of some optical systems
    response to wavefronts.

    Attributes
    ----------
    layers: dict
        A collections.OrderedDict of 'layers' that define the transformations
        and operations upon some input wavefront through an optical system.
    """
    wf_npixels : int
    diameter   : Array
    layers     : OrderedDict


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
        super().__init__()
        if not isinstance(layers, list):
            raise ValueError("Input layers must be a list, it is"
                " automatically converted to a dictionary.")
        
        # Ensure all entries are dLux layers
        for layer in layers:
            if isinstance(layer, tuple):
                layer = layer[0]
            if not isinstance(layer, OpticalLayer()):
                raise ValueError("All entries within layers must be an "
                    "OpticalLayer object.")

        self.layers = dLux.utils.list_to_dictionary(layers)
        self.diameter = np.asarray(diameter, dtype=float)
        self.wf_npixels = int(wf_npixels)

        if self.diameter.ndim != 0:
            raise ValueError("diameter must be a scalar, got shape"
                f"{diameter.shape}.")


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
        for attribute in self.layers.values():
            if hasattr(attribute, key):
                return getattr(attribute, key)
        raise AttributeError(f"{self.__class__.__name__} has no attribute "
        f"{key}.")


    def _construct_wavefront(self       : BaseOptics,
                             wavelength : Array,
                             offset     : Array = np.zeros(2)) -> Array:
        """
        Constructs the appropriate tilted wavefront object for the optical
        system.

        TODO: Possibly seach for propagator type and construct the correct
        wavefront type. Possibly search and check for propagator consistency.
        Also do a shape check for layers.

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
            wavefront = layer(wavefront)
        
        if return_wf:
            return wavefront
        return wavefront.psf