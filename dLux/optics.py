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


__all__ = ["model", "Instrument", "SimpleOptics", "MaskedOptics", "Optics", 
    "Detector"]


# Alias classes for simplified type-checking
CreateWavefront   = lambda : dLux.optics.CreateWavefront
TransmissiveOptic = lambda : dLux.optics.TransmissiveOptic
AberrationLayer   = lambda : dLux.optics.AberrationLayer
OpticalLayer      = lambda : dLux.optics.OpticalLayer
AddOPD            = lambda : dLux.optics.AddOPD
AddPhase          = lambda : dLux.optics.AddPhase
Propagator        = lambda : dLux.propagators.Propagator
FarFieldFresnel   = lambda : dLux.propagators.FarFieldFresnel
Source            = lambda : dLux.sources.BaseSource
ShapedOptic       = lambda : dLux.optical_layer.ShapedOptic


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
        for keys, value in self.__dict__.items():
            if hasattr(value, 'name') and value.name == key:
                return value
            if hasattr(value, key):
                return getattr(value, key)
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute "
            f"{key}.")


    @abstractmethod
    def propagate_mono(self       : BaseOptics,
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
                      weights     : Array = None,
                      offset      : Array = None) -> Array:
        """
        Formats the weights and wavelengths of the polychromatic wavefronts.

        Parameters
        ----------
        wavelengths : Array, meters
            The wavelengths of the wavefronts to propagate through the optics.
        weights : Array, = None
            The weights of each wavelength. If None, all wavelengths are
            weighted equally.

        Returns
        -------
        wavelengths : Array
            The wavelengths of the wavefronts to propagate through the optics.
        weights : Array
            The weights of each wavelength. If None, all wavelengths are
            weighted equally.
        offset : Array, radians, = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
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
        wavelengths, weights, offset = self._format_input(wavelengths, weights, 
            offset)

        # Construct Propagate
        propagator = vmap(self.propagate_mono, in_axes=(0, None))

        # Calc and Return
        psfs = propagator(wavelengths, offset)
        if weights is not None:
            psfs *= weights[:, None, None]
        return psfs.sum(0)
    
    
    def model(self              : BaseOptics,
              sources           : Union[Source, dict, list],
              normalise_sources : bool = True,
              flatten           : bool = False,
              return_tree       : bool = False) -> Union(Array, dict):
        """
        A base level modelling function for modelling the optical system.
        Models the source or sources through the optics.

        Parameters
        ----------
        sources : Union[Source, dict, list]
            The source or sources to observe.
        normalise_sources : bool = True
            Whether to normalise the sources before modelling.
        flatten : bool = False
            Whether the output image should be flattened.
        return_tree : bool = False
            Whether to return a Pytree like object with matching tree structure
            as the input sources (ie dict).

        Returns
        -------
        image : Array, dict
            The image of the scene modelled through the optics. Returns either
            as a single array (if return_tree is false), or a dict of the output
            for each source.
        """
        # None input is for the detector - prevents declaring each parameter 
        # kwarg
        return model(self, sources, None, normalise_sources, flatten,
            return_tree)
    

    # Potential future implementation for an optics class with a source,
    # Current issue is this will not be deserialisable becuase its definition
    # Will never be able to founc in the namespcae becuase it is generated
    # dynamically.
    # def add_source(self : BaseOptics, source : Source) -> SourceOptics:

    #     # Define new class
    #     class SourceOptics(self.__class__):
    #         source : Source

    #         # def __init__(self):
    #         #     super().__init__()
        
    #         def model(self):
    #             return super().model(self.source)

    #     source_class = SourceOptics.__new__(SourceOptics)

    #     # Set existing attributes
    #     for key, value in self.__dict__.items():
    #         object.__setattr__(source_class, key, value)

    #     # Set source attribute
    #     object.__setattr__(source_class, 'source', source)

    #     return source_class



# Base
# # Optics
# # SimpleOptics
# # # Angular
# # # Cartesian
# # # Flexible (propagator)

# class SimpleOptics(BaseOptics):
class SimpleOptics(BaseOptics):
    """
    A simple class designed to model an optical system with a single pupil and
    focal plane. This class is designed to take advantage of the `Factory`
    classes to make construction simple. Lets look at an example of a simple
    unaberrated optical system:

    ```python
    # Set up the componenets
    diameter = 1 # meters
    wf_npixels = 256
    det_npixles = 128
    pixel_scale = 2e-7 # radians

    # Construct Components
    aperture = dl.ApertureFactory(wf_npixels)
    propagator = dl.PropagatorFactory(det_npixles, pixel_scale)
    optics = SimpleOptics(diameter, aperture, propagator)
    ```

    We can now model a polychromatic PSF like so:

    ```python
    wavelengths = np.linspace(1e-6, 2e-6, 20)
    psf = optics.propagate(wavelengths)
    ```

    Its that simple!

    What if we wanted to add aberrations? We can do that too! Lets add some
    random zernike aberrations and model the psf:

    ```python
    zernikes = np.arange(4, 11)
    coeffs = 1e-7 * jr.normal(jr.PRNGKey(0), zernikes.shape)
    aberrations = dl.AberrationFactory(wf_npixels, zernikes, coeffs)

    optics = SimpleOptics(diameter, aperture, propagator, aberrations)
    psf = optics.propagate(wavelengths)
    ```

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
        return wf.tilt_wavefront(offset)


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

class AngularOptics(SimpleOptics):
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
        wf = self._initialise_wavefront(wavelength, offset)

        # Propagate
        pixel_scale = self.psf_pixel_scale / self.psf_oversample
        pixel_scale_radians = dlu.arcseconds_to_radians(pixel_scale)
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
        wf = self._initialise_wavefront(wavelength, offset)

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
        wf = self._initialise_wavefront(wavelength, offset)
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
    layers : OrderedDict


    def __init__(self : Optics, layers : list) -> Optics:
        """
        Constructor for the Optics class.

        Parameters
        ----------
        layers : list
            A list of ∂Lux 'layers' that define the transformations and
            operations upon some input wavefront through an optical system.
        """
        # Ensure input is a list
        if not isinstance(layers, list):
            raise ValueError("Input layers must be a list, it is"
                " automatically converted to a dictionary.")

        # Check for CreateWavefront layer
        if not isinstance(layers[0], CreateWavefront()):
            raise ValueError("First layer must be a CreateWavefront object.")
        
        # Ensure all entries are dLux layers & propagator
        has_propagator = False
        for layer in layers:
            if not isinstance(layer, OpticalLayer()):
                raise ValueError("All entries within layers must be an "
                    "OpticalLayer object.")
            if isinstance(layer, Propagator()):
                has_propagator = True
        
        if not has_propagator:
            warn("No propagator found in layers, wavefront will remain in the "
                "Pupil plane.")

        self.layers = dLux.utils.list_to_dictionary(layers)


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
        else:
            raise AttributeError("'{}' object has no attribute '{}'"\
                                 .format(type(self), key))


    def propagate_mono(self       : Optics,
                       wavelength : Array,
                       offset     : Array = np.zeros(2),
                       return_wf  : bool = False,
                       return_all : bool = False) -> Array:
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
        return_wf : bool = False
            Whether to return the wavefront object after propagation.
        return_all : bool = False
            Whether to return the all intermediate wavefront objects.

        Returns
        -------
        psf : Array
            The monochromatic point spread function after being propagated
            though the optical layers.
        wavefront : Wavefront
            The wavefront object after propagation. Only returned if
            return_wf is True.
        wavefront_list : List[Wavefront]
            A list of all the wavefront objects after propagation. Only
            returned if return_all is True.
        """
        # Ensure jax arrays
        wavelength = np.asarray(wavelength, dtype=float) \
            if not isinstance(wavelength, np.ndarray) else wavelength
        offset = np.asarray(offset, dtype=float) \
            if not isinstance(offset, np.ndarray) else offset

        # Ensure dimensionality
        assert wavelength.shape == (), "wavelength must be a scalar."
        assert offset.shape == (2,), "offset must be shape (2,), ie (x, y)."

        layers = list(self.layers.values())
        WF = layers[0](wavelength, offset)

        # Construct parameters
        parameters = {"optics" : self}

        # Propagate though the rest of the layers
        if not return_all:
            for layer in layers[1:]:
                WF, parameters = layer.apply(WF, parameters)
            if return_wf:
                return WF
            else:
                return WF.psf
        
        else:
            WF_list = [WF]
            for layer in layers[1:]:
                WF, parameters = layer.apply(WF, parameters)
                WF_list.append(WF)
            return WF_list