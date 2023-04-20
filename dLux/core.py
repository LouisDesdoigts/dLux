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
TransmissiveOptic = lambda : dLux.optics.TransmissiveOptic
AberrationLayer = lambda : dLux.optics.AberrationLayer
OpticalLayer = lambda : dLux.optics.OpticalLayer
AddOPD = lambda : dLux.optics.AddOPD
AddPhase = lambda : dLux.optics.AddPhase
Propagator = lambda : dLux.propagators.Propagator
Propagator = lambda : dLux.propagators.Propagator
FarFieldFresnel = lambda : dLux.propagators.FarFieldFresnel


###############
### Methods ###
###############
def model(optics      : Optics,
          sources     : Union[dict, list, Source],
          detector    : Detector = None,
          normalise   : bool     = True,
          flatten     : bool     = False,
          return_tree : bool     = False) -> Array:
    """
    A base level modelling function designed to robustly handle the different
    combinations of inputs. Models the sources through the instrument optics
    and detector. Users must provide optics and source.

    Parameters
    ----------
    optics : Optics
        The optics through which to model the source objects.
    sources : Union[dict, list, Source]
        The sources to observe.
    detector : Detector = None
        The detector to use with the observation.
    normalise : bool = None
        Whether to normalise the sources before modelling. Default is True.
    flatten : bool = False
        Whether the output image should be flattened. Default is False.
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
    '''Input checking and formatting'''
    # Check that optics input is an Optics object.
    # assert isinstance(optics, (Optics)), ("optics must be an Optics object.")
    assert isinstance(optics, BaseOptics), ("optics must be an Optics object.")

    # Check that detector input is a Detector object if specified.
    assert isinstance(detector, (Detector, type(None))), \
    ("detector must be a Detector object.")

    # Check that sources is a dict object.
    assert isinstance(sources, (dict, list, tuple, Source())), \
    ("sources must be a Source object, dict, list, or tuple object.")

    # Check that all inputs are Source objects
    if not isinstance(sources, Source()):
        source_vals = sources.values() if isinstance(sources, dict) else sources
        for source in source_vals:
            assert isinstance(source, Source()), \
            ("All entries within sources must be a Source object.")
    
    # Turn single source object into a list for mapping
    else:
        sources = [sources]

    # Normalise Sources
    if normalise:
        # Define the normalisation function
        normalise_fn = lambda source: source.normalise()

        # Map the normalisation function across the sources
        sources_in = tree_map(normalise_fn, sources, \
                is_leaf = lambda leaf: isinstance(leaf, Source()))
    else:
        sources_in = sources

        # Get sources
        sources_in = source.normalise() if normalise else source

    '''Begin modelling'''
    # Apply optional inputs
    model_fn = lambda source: source.model(optics)

    # Map the model_source function across the sources
    psf_tree = tree_map(model_fn, sources_in, 
            is_leaf = lambda leaf: isinstance(leaf, Source()))

    # Return psfs in the same structure as the sources
    if return_tree:

        # Apply detector if required
        if detector is not None:
            detector_fn = lambda psf: detector.apply_detector(psf)
            image_tree = tree_map(detector_fn, psf_tree, 
                is_leaf = lambda leaf: isinstance(leaf, np.ndarray))
        else:
            image_tree = psf_tree

        # flatten if required
        if flatten:
            flatten_fn = lambda image: image.flatten()
            tree_out = tree_map(flatten_fn, image_tree,
                is_leaf = lambda leaf: isinstance(leaf, np.ndarray))
        else:
            tree_out = image_tree

        # Return psfs with matching tree strucutre as input
        return tree_out


    # Return a single summed psf
    else:
        # Get flatten tree and sum to single psf
        psf = np.array(tree_flatten(psf_tree)[0]).sum(0)

        # Apply detector
        image = detector.apply_detector(psf) if detector is not None else psf

        # Flatten
        return image.flatten() if flatten else image



######################
### Optics Classes ###
######################
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
        pass


    def _format_inputs(self       : BaseOptics,
                       wavelength : Array,
                       offset     : Array = np.zeros(2),
                       return_wf  : bool = False) -> Array:
        """
        Checks that all inputs are of the correct type and shape, returning
        Array verions of all float/list inputs.

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
        wavelength : Array, meters
            The wavelength of the wavefront to propagate through the optical
            layers.
        offset : Array, radians
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.
        return_wf : bool
            If True, the wavefront object after propagation is returned.
        """
        # Check offset
        if not isinstance(offset, Array):
            offset = np.asarray(offset, dtype=float)
            if offset.shape != (2,):
                raise ValueError("offset must be a 2-element array, got "
                    f"shape {offset.shape}.")
        
        # Check wavelength
        if not isinstance(wavelength, Array):
            wavelength = np.asarray(wavelength, dtype=float)
            if wavelength.shape != ():
                raise ValueError("wavelength must be a scalar, got "
                    f"shape {wavelength.shape}.")
        
        # Check return_wf
        if not isinstance(return_wf, bool):
            raise ValueError("return_wf must be a bool, got "
                f"type {type(return_wf)}.")
        
        return wavelength, offset, return_wf


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


    def _format_weights(self        : BaseOptics,
                        wavelengths : Array,
                        weights     : Array = None) -> Array:
        """
        Formats the weights of the polychromatic wavefronts.

        Parameters
        ----------
        wavelengths : Array, meters
            The wavelengths of the wavefronts to propagate through the optics.
        weights : Array, = None
            The weights of each wavelength. If None, all wavelengths are
            weighted equally.

        Returns
        -------
        : None
            If weights is None.
        weights : Array
            The weights of each wavelength if weights is not None.
        """
        # Check weights
        if weights is None:
            return
        else:
            weights = np.array(weights, dtype=float) \
                if not isinstance(weights, np.ndarray) else weights
            if len(weights) != len(wavelengths):
                raise ValueError("wavelengths and weights must have the "
                    f"same length, got {len(wavelengths)} and {len(weights)} "
                    "respectively.")
        return weights


    def propagate(self        : BaseOptics, 
                  wavelengths : Array,
                  offset      : Array = np.zeros(2),
                  weights     : Array = None,
                  return_wfs  : bool = False) -> Array:
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
        return_wfs : bool, = False
            If True, the wavefront objects after propagation are returned.

        Returns
        -------
        psf : Array
            The monochromatic point spread function after being propagated
            though the optical layers.
        wavefronts : list[Wavefront]
            The wavefront object after propagation. Only returned if
            return_wf is True.
        """
        weights = self._format_weights(wavelengths, weights)

        # Construct Propagate
        propagator = vmap(self.propagate_mono, in_axes=(0, None))

        # Calc and Return
        if return_wfs:
            return propagator(wavelengths, offset, return_wf=True)
        else:
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
    propagator  : Propagator()


    def __init__(self : SimpleOptics, 
                 diameter : Array, 
                 aperture : Union[Array, TransmissiveOptic()],
                 propagator : Propagator(), 
                 aberrations : Union[Array, AberrationLayer()] = None
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
            raise ValueError("diameter must be a scalar, got type"
                f"{type(diameter)}.")
        self.diameter = diameter

        # Aperture Checking
        if not isinstance(aperture, (Array, TransmissiveOptic())):
            raise ValueError("aperture must be an Array or "
                f"TransmissveOptic, got {type(aperture)}.")
        self.aperture = aperture.transmission

        # Aberrations Checking
        if aberrations is not None:
            if not isinstance(aberrations, AberrationLayer()):
                raise ValueError("aberrations must be an AberrationLayer, got "
                    f"{type(aberrations)}.")
            
            # Check for consistent array sizes of basis
            if hasattr(aberrations, 'basis') and \
                isinstance(aberrations.basis, Array):
                if self.aperture.shape != aberrations.basis.shape[-2:]:
                    raise ValueError("aperture and aberration basis must have "
                        f"the same shape, got {self.aperture.shape} and "
                        f"{aberrations.basis.shape} respectively.")

            # Check for consistent array sizes of opd
            elif hasattr(aberrations, 'opd') and \
                isinstance(aberrations.opd, Array):
                if self.aperture.shape != aberrations.opd.shape:
                    raise ValueError("aperture and aberration opd must have "
                        f"the same shape, got {self.aperture.shape} and "
                        f"{aberrations.opd.shape} respectively.")
        self.aberrations = aberrations
    
        # Propagator Checking
        if not isinstance(propagator, Propagator()):
            raise ValueError("propagator must be a Propagator, got "
                f"{type(propagator)}.")
        self.propagator = propagator


    def propagate_mono(self       : BaseOptics,
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
        # Check valid inputs
        wavelength, offset, return_wf = self._format_inputs(wavelength, offset, 
            return_wf)

        # Construct and tilt the Wavefront
        wf = self._construct_wavefront(wavelength, offset)

        # Apply aperture and normalise
        wf *= self.aperture
        wf = wf.normalise()

        # Apply aberrations
        wf *= self.aberrations

        # Propagate
        wf *= self.propagator

        # Return PSF or Wavefront
        if return_wf:
            return wf
        return wf.psf


class MaskedOptics(SimpleOptics):
    """
    A simple extension of the SimpleOptics class than can hold a mask. This
    mask can be transmissive like an aperture mask, or it can be some form of
    phase mask, either adding a phase or an OPD. If the input mask is an array
    it will be treated as a transmissive mask that is multiplied by the
    wavefront amplitude. 
    
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
    mask : Union[Array, TransmissiveOptic(), AddPhase(), AddOPD()]
        The mask to apply to the wavefront. Can be an Array, a 
        TransmissiveOptic, an AddPhase or an AddOPD.
    """
    mask : Union[Array, OpticalLayer()]
    

    def __init__(self, diameter, aperture, mask, propagator, aberrations=None):
        """
        Constructs a simple optical system with a static aperture, mask and
        aberrations.

        Note this class automatically converts aperture input into an array.

        Parameters
        ----------
        diameter : Array
            The diameter of the wavefront to model through the system in meters.
        aperture : Union[Array, TransmissiveOptic()]
            The aperture of the system. Can be an Array or a TransmissiveOptic.
        mask : Union[Array, TransmissiveOptic(), AddPhase(), AddOPD()]
            The mask to apply to the wavefront. Can be an Array, a
            TransmissiveOptic, an AddPhase or an AddOPD.
        propagator : Propagator()
            The propagator to use to propagate the wavefront through the system.
        aberrations : Union[Array, AberrationLayer()] = None
            The aberrations to apply to the wavefront. Can be an Array or an
            AberrationLayer. If None, no aberrations are applied.
        """
        super().__init__(diameter, aperture, propagator, aberrations)

        mask_like = (Array, TransmissiveOptic(), AddPhase(), AddOPD())
        if not isinstance(mask, (Array, OpticalLayer())):
            raise ValueError("mask must be an Array or OpticalLayer, "
                f"got {type(mask)}.")

        # Check for consistent array sizes of mask
        if isinstance(mask, Array):
            if self.aperture.shape != mask.shape:
                raise ValueError("aperture and mask must have "
                    f"the same shape, got {self.aperture.shape} and "
                    f"{mask.shape} respectively.")

        # Tranmissive Optics
        elif isinstance(mask, TransmissiveOptic()):
            if self.aperture.shape != mask.tranmission.shape:
                raise ValueError("aperture and mask transmission must have "
                    f"the same shape, got {self.aperture.shape} and "
                    f"{mask.transmission.shape} respectively.")
        
        # Add OPD
        elif isinstance(mask, AddOPD()):
            if self.aperture.shape != mask.opd.shape:
                raise ValueError("aperture and mask opd must have "
                    f"the same shape, got {self.aperture.shape} and "
                    f"{mask.opd.shape} respectively.")
        
        # Add Phase
        elif isinstance(mask, AddPhase()):
            if self.aperture.shape != mask.phase.shape:
                raise ValueError("aperture and mask phase must have "
                    f"the same shape, got {self.aperture.shape} and "
                    f"{mask.phase.shape} respectively.")

        # Finally, set the mask
        self.mask = mask


    def propagate_mono(self       : BaseOptics,
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
        # Check valid inputs
        wavelength, offset, return_wf = self._format_inputs(wavelength, offset, 
            return_wf)

        # Construct and tilt the Wavefront
        wf = self._construct_wavefront(wavelength, offset)

        # Apply aperture and normalise
        wf *= self.aperture
        wf = wf.normalise()

        # Apply Mask
        wf *= self.mask

        # Apply aberrations
        wf *= self.aberrations

        # Propagate
        wf *= self.propagator

        # Return PSF or Wavefront
        if return_wf:
            return wf
        return wf.psf


class Optics(BaseOptics):
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

        # Runtime check for CreateWavefront layer (Maybe move to constructor?)
        layers = list(self.layers.values())
        if not isinstance(layers[0], CreateWavefront()):
            raise ValueError("First layer must be a CreateWavefront layer")
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


    # def propagate(self        : Optics,
    #               wavelengths : Array,
    #               offset      : Array = np.zeros(2),
    #               weights     : Array = None,
    #               return_wfs  : bool = False) -> Array:
    #     """
    #     Propagates a broadband point source through the optical layers.

    #     Parameters
    #     ----------
    #     wavelengths : Array, meters
    #         The wavelengths of the wavefront to propagate through the optical
    #         layers.
    #     offset : Array, radians = np.zeros(2)
    #         The (x, y) offset from the optical axis of the source. Default
    #         value is (0, 0), on axis.
    #     weights : Array = None
    #         The relative to apply to the monochromatic psfs.
    #     return_wfs : bool = False
    #         Whether to return the wavefront objects after propagation.

    #     Returns
    #     -------
    #     psf : Array
    #         The broadband point spread function after being propagated
    #         though the optical layers.
    #     wavefronts : List[Wavefront]
    #         A list of all the wavefront objects after propagation. Only
    #         returned if return_wfs is True.
    #     """
    #     # Format weights input
    #     wavelengths = np.asarray(wavelengths, dtype=float) \
    #               if not isinstance(wavelengths, np.ndarray) else wavelengths
    #     assert wavelengths.ndim == 1, "wavelengths must be 1 dimensional.."

    #     # Format weights input
    #     if weights is None:
    #         weights = np.ones(len(wavelengths))/len(wavelengths)
    #     elif not isinstance(weights, np.ndarray):
    #         weights = np.asarray(weights, dtype=float)
    #     assert weights.ndim == 1, "weights must be 1 dimensional."

    #     # Ensure matching dimensionality
    #     assert wavelengths.shape == weights.shape, \
    #     ("wavelengths and weights must have the same shape.")

    #     # Offset checking
    #     offset = np.asarray(offset, dtype=float) \
    #              if not isinstance(offset, np.ndarray) else offset
    #     assert offset.shape == (2,), "offset must be shape (2,), ie (x, y)."

    #     # Construct Propagate
    #     propagator = vmap(self.propagate_mono, in_axes=(0, None))

    #     # TODO: Test if speed can be improved by initialising a single wavefront
    #     # here and vmapping a 'propagate_wavefront' function, rather than
    #     # vmapping the 'propagate_mono' function which initialises the WF.
    #     # This poses a potential issue with the tilting of the wavefront, which
    #     # presently done in the CreateWavefront layer. Possibly a small lambda
    #     # function can be used to do this. Should primarily check if this has 
    #     # any speed and memory benefits.

    #     # For now this should work fine, but it is something to keep in mind.

    #     # Calc and Return
    #     if not return_wfs:
    #         psfs = propagator(wavelengths, offset)
    #         psfs *= weights[:, None, None]
    #         return psfs.sum(0)
    #     else:
    #         psfs, wfs = propagator(wavelengths, offset, return_wf=True)
    #         psfs *= weights[:, None, None]
    #         return psfs.sum(0), wfs
    

    def get_planes(self : Optics) -> list:
        """
        Breaks the optical layers into planes, where each plane is a list of
        layers.

        Returns
        -------
        planes : list
            A list of lists, with the inner lists being optical layers, and the
            outer list being planes.
        """
        planes = []
        plane = []
        keys = self.layers.keys()
        for key in keys:
            layer = self.layers[key]
            plane.append(layer)
            if isinstance(layer, dLux.propagators.Propagator):
                planes.append(plane)
                plane = []
        return planes


    def summarise(self : Optics) -> None:
        """
        Prints a summary of all the planes in the optical system.
        """
        planes = self.get_planes()
        # TODO: Add plane type (Plane 0: Pupil)
        print("Text summary:")
        for i in range(len(planes)):
            print(f'Plane {i}')
            for layer in planes[i]:
                print(f"  {layer.summary(angular_units='arcseconds')}")
        print('\n')


    def plot(self       : Optics, 
             wavelength : Array, 
             offset     : Array = np.zeros(2)) -> None:
        """
        Prints the summary of all the planes and then plots a wavefront as it
        propagates through the optics.

        Parameters
        ----------
        wavelength : Array, meters
            The wavelength of the wavefront to propagate through the optical
            layers.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        
        Returns
        -------
        wf : Wavefront
            The final wavefront after being propagated through the optical
            layers.
        """
        planes = self.get_planes()
        self.summarise()

        for i in range(len(planes)):
            print(f'Plane {i}')
            for layer in planes[i]:
                print(f"  {layer.summary()}")
                if isinstance(layer, dLux.CreateWavefront):
                    wf, parameters = layer(None, 
                                  {"wavelength": wavelength, 'offset': offset},returns_parameters=True)
                else:
                    # Inspect apply function to see if it takes/returns the parameters dict
                    input_parameters = signature(layer).parameters

                    # Method does not take in the parameters, update in place
                    if 'parameters' not in input_parameters:
                        wf = layer(wf)

                    # Method takes and return updated parameters
                    elif input_parameters['returns_parameters'].default == True:
                        wf, parameters = layer(wf, parameters)

                    # Method takes but does not return parameters
                    else:
                        wf = layer(wf, parameters)
                layer.display(wf)
        return wf


    # def model(self              : Optics,
    #           sources           : Union[Source, dict, list],
    #           normalise_sources : bool = True,
    #           flatten           : bool = False,
    #           return_tree       : bool = False) -> Union(Array, dict):
    #     """
    #     A base level modelling function for modelling the optical system.
    #     Models the source or sources through the optics.

    #     Parameters
    #     ----------
    #     sources : Union[Source, dict, list]
    #         The source or sources to observe.
    #     normalise_sources : bool = True
    #         Whether to normalise the sources before modelling.
    #     flatten : bool = False
    #         Whether the output image should be flattened.
    #     return_tree : bool = False
    #         Whether to return a Pytree like object with matching tree structure
    #         as the input sources (ie dict).

    #     Returns
    #     -------
    #     image : Array, dict
    #         The image of the scene modelled through the optics. Returns either
    #         as a single array (if return_tree is false), or a dict of the output
    #         for each source.
    #     """
    #     # None input is for the detector
    #     return model(self, sources, None, normalise_sources, flatten,
    #         return_tree)


#####################
### Other Classes ###
#####################
class Instrument(Base):
    """
    A high level class desgined to model the behaviour of a telescope. It
    stores a series different ∂Lux objects, and primarily passes the relevant
    information between these objects in order to coherently model some
    telescope observation.

    Attributes
    ----------
    optics : Optics
        A Optics object that defines some optical configuration.
    sources : dict
        A dictionary of the various source objects that the instrument is
        observing.
    detector : Detector
        A Detector object that is used to model the various
        instrumental effects on a psf.
    observation : Observation
        An class that inherits from Observation. This is to allow flexibility
        in the different kind of observations, ie applying dithers, switching
        filters, etc.
    """
    optics      : Optics
    sources     : dict
    detector    : Detector
    observation : Observation

    
    def __init__(self        : Instrument,
                 optics      : Optics,
                 sources     : Union[list, Source],
                 detector    : Detector    = None,
                 observation : Observation = None,
                 ) -> Instrument:
        """
        Constructor for the Instrument class.

        Parameters
        ----------
        optics : Optics
            A pre-configured Optics object.
        sources : Union[list, Source]
            Either a list of sources or an individual Source object.
        detector : Detector = None
            A pre-configured Detector object.
        observation : Observation = None
            An class that inherits from Observation. This is to allow
            flexibility in the different kind of observations, ie applying
            dithers, switching filters, etc.
        """
        # Optics
        if not isinstance(optics, Optics):
            raise ValueError("optics must be an Optics object.")
        self.optics = optics
        
        # Sources
        if isinstance(sources, Source()):
            sources = [sources]
        elif isinstance(sources, list):
            for source in sources:
                if not isinstance(source, Source()):
                    raise ValueError(
                        "sources must be a list of Source objects.")
        self.sources = dLux.utils.list_to_dictionary(sources, ordered=False)

        # Detector
        if not isinstance(detector, (Detector, type(None))):
            raise ValueError("detector must be an Detector object. "
                f"Got type {type(detector)}")
        self.detector = detector

        # Observation
        if not isinstance(observation, (Observation(), type(None))):
            raise ValueError("observation must be an Observation object.")
        self.observation = observation


    def observe(self : Instrument, **kwargs) -> Any:
        """
        Calls the `observe` method of the stored observation class, passing in
        any extra keyword arguments.

        Returns
        -------
         : Any
            The output of the stored observation class.
        """
        return self.observation.observe(self, **kwargs)


    def __getattr__(self : Instrument, key : str) -> object:
        """
        Magic method designed to allow accessing of the various items within
        the sub-dictionaries of this class via the 'class.attribute' method.
        It is recommended that each dictionary key in the optical layers,
        detector layers, and scene sources are unique to prevent unexpected
        behaviour. In the case they there are idenitcal keys across the
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
        if key in self.optics.layers.keys():
            return self.optics.layers[key]
        elif key in self.sources.keys():
            return self.sources[key]
        elif self.detector is not None and key in self.detector.layers.keys():
            return self.detector.layers[key]
        elif self.observation is not None and hasattr(self.observation, key):
            return getattr(self.observation, key)
        else:
            raise AttributeError("'{}' object has no attribute '{}'"\
                                 .format(type(self), key))


    def normalise(self : Instrument) -> Instrument:
        """
        Normalises the internally stored scene by calling the scene.normalise()
        method.

        Returns
        -------
        instrument : Instrument
            A new version of the instrument with the interally stored scene
            normalised.
        """
        leaf_fn = lambda source: isinstance(source, Source())
        normalise_fn = lambda source: source.normalise()
        return tree_at(lambda instrument: instrument.sources, self,
            tree_map(normalise_fn, self.sources, is_leaf=leaf_fn))
    

    def summarise(self : Instrument) -> None:
        """
        Prints a summary of all instrument
        """
        print("Sources summary:")
        summary_fn = lambda source: source.summarise()
        tree_map(summary_fn, self.sources)

        print("Optics summary:")
        self.optics.summarise()
        
        if self.detector is not None:
            print("Detector summary:")
            self.detector.summarise()


    def plot(self       : Optics, 
             wavelength : Array, 
             offset     : Array = np.zeros(2)) -> None:
        """
        Prints the summary of all the planes and then plots a wavefront as it
        propagates through the optics.

        Parameters
        ----------
        wavelength : Array, meters
            The wavelength of the wavefront to propagate through the optical
            layers.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        """
        wf = self.optics.plot(wavelength, offset)
        self.detector.plot(wf.psf)


    def model(self              : Instrument,
              normalise_sources : bool = True,
              flatten           : bool = False,
              return_tree       : bool = False) -> Union(Array, dict):
        """
        A base level modelling function designed to robustly handle the
        different combinations of inputs. Models the sources through the
        instrument optics and detector.

        Parameters
        ----------
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
            The image of the scene modelled through the optics with detector and
            filter effects applied if they are supplied. Returns either as a
            single array (if return_tree is false), or a dict of the output for
            each source.
        """
        return model(self.optics, self.sources, self.detector, 
            normalise_sources, flatten, return_tree)


class Detector(Base):
    """
    A high level class desgined to model the behaviour of some detectors
    response to some psf.

    Attributes
    ----------
    layers: dict
        A collections.OrderedDict of 'layers' that define the transformations
        and operations upon some input psf as it interacts with the detector.
    """
    layers : OrderedDict


    def __init__(self : Detector, layers : list) -> Instrument:
        """
        Constructor for the Detector class.

        Parameters
        ----------
        layers : list
            An list of dLux detector layer classes that define the instrumental
            effects for some detector.
        """
        # Ensure input is a list
        assert isinstance(layers, list), ("Input layers must be a list, it is" \
        " automatically converted to a dictionary.")

        # Ensure all entries are dLux layers
        for layer in layers:
            assert isinstance(layer, dLux.detectors.DetectorLayer), \
            ("All entries within layers must be a "
             "dLux.detectors.DetectorLayer object.")

        # Construct layers
        self.layers = dLux.utils.list_to_dictionary(layers)


    def __getattr__(self : Detector, key : str) -> object:
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


    def apply_detector(self : Instrument, image : Array) -> Array:
        """
        Applied the stored detector layers to the input image.

        Parameters
        ----------
        image : Array
            The input 'image' to the detector to be transformed.

        Returns
        -------
        image : Array
            The ouput 'image' after being transformed by the detector layers.
        """
        # Input type checking
        assert isinstance(image, np.ndarray), "Input must be a jax array."
        assert image.ndim == 2, "Input image must a 2d array."

        # Apply detector layers
        for key, layer in self.layers.items():
            image = layer(image)
        return image


    def debug_apply_detector(self  : Instrument, 
                             image : Array) -> Array:
        """
        Applied the stored detector layers to the input image, storing and
        returning the intermediate states of the image and layers.

        Parameters
        ----------
        image : Array
            The input 'image' to the detector to be transformed.

        Returns
        -------
        image : Array
            The ouput 'image' after being transformed by the detector layers.
        intermediate_images : list
            The intermediate states of the image.
        intermediate_layers : list
            The intermediate states of each layer after being applied to the
            image.
        """
        # Input type checking
        assert isinstance(image, np.ndarray), "Input must be a jax array."
        assert image.ndim == 2, "Input image must a 2d array."

        # Apply detector layers
        intermediate_images = []
        intermediate_layers = []
        for key, layer in self.layers.items():
            image = layer(image)
            intermediate_images.append(image.copy())
            intermediate_layers.append(deepcopy(layer))
        return image, intermediate_images, intermediate_layers
    

    def summarise(self : Detector) -> None:
        """
        Prints a summary of all the layers in the detector.
        """
        print("Text summary:")
        keys = self.layers.keys()
        for key in keys:
            layer = self.layers[key]
            print(f"  {layer.summary()}")
        print('\n')


    def plot(self : Optics, image : Array) -> None:
        """
        Prints the summary of all the layers and then plots a image as it
        propagates through the detector layer.

        Parameters
        ----------
        iamge : Array
            The image to propagate through the detector.
        """
        self.summarise()
        keys = self.layers.keys()
        for key in keys:
            layer = self.layers[key]
            print(f"{layer.summary()}")
            image = layer(image)
            layer.display(image)


    def model(self : Detector, image: Array) -> Array:
        """
        A function to apply the detector layers to the input image.

        Parameters
        ----------
        image: Array
            The image to be transformed by the detector layers.

        Returns
        -------
        image : Array
            The image after being transformed by the detector layers.
        """
        return self.apply_detector(image)


class Filter(Base):
    """
    NOTE: This class is under development.

    A class for modelling optical filters.

    Attributes
    ----------
    wavelengths : Array
        The wavelengths at which the filter is defined.
    throughput : Array
        The throughput of the filter at the corresponding wavelength.
    filter_name : str
        A string identifier that can be used to initialise specific filters.
    """
    wavelengths  : Array
    throughput   : Array
    filter_name  : str


    def __init__(self        : Filter,
                 wavelengths : Array = None,
                 throughput  : Array = None,
                 filter_name : str   = None) -> Filter:
        """
        Constructor for the Filter class. All inputs are optional and defaults
        to uniform unitary throughput. If filter_name is specified then
        wavelengths and weights must not be specified.

        Parameters
        ----------
        wavelengths : Array = None
            The wavelengths at which the filter is defined.
        throughput : Array = None
            The throughput of the filter at the corresponding wavelength.
        filter_name : str = None
            A string identifier that can be used to initialise specific filters.
            Currently no pre-built filters are implemented.
        """
        # Take the filter name as the priority input
        if filter_name is not None:
            # TODO: Pre load filters
            raise NotImplementedError("You know what this means.")
            pass

            # Check that wavelengths and throughput are not specified
            if wavelengths is not None or throughput is not None:
                raise ValueError("If filter_name is specified, wavelengths "
                "and throughput can not be specified.")

        # Check that both wavelengths and throughput are specified
        elif (wavelengths is     None and throughput is not None) or \
             (wavelengths is not None and throughput is     None):
            raise ValueError("If either wavelengths or throughput is "
            "specified, then both must be specified.")

        # Neither is specified
        elif wavelengths is None and throughput is None:
            self.wavelengths = np.array([0., np.inf])
            self.throughput  = np.array([1., 1.])
            self.filter_name = 'Unitary'

        # Both wavelengths and throughputs are specified
        else:
            self.wavelengths = np.asarray(wavelengths, dtype=float)
            self.throughput  = np.asarray(throughput,  dtype=float)
            self.filter_name = 'Custom'

            # Check bounds
            assert self.wavelengths.ndim == 1 and self.throughput.ndim == 1, \
            "Both wavelengths and throughput must be 1 dimensional."
            assert self.wavelengths.shape == self.throughput.shape, \
            ("wavelengths and throughput must have the same length.")
            assert np.min(self.wavelengths) >= 0, \
            ("wavelengths can not be less than 0.")
            assert (self.throughput >= 0).all() and \
            (self.throughput <= 1).all(), ("throughput must be between 0-1.")
            assert np.min(wavelengths) < np.max(wavelengths), \
            ("wavelengths must be in-order from small to large.")


    def get_throughput(self : Filter, sample_wavelenghts : Array) -> Array:
        """
        Gets the average throughput of the bandpass defined the the differences
        between each sample wavelength, ie if sample wavelengths are:
            [10, 20, 30, 40],
        the bandpasses for each sample wavelength will be
            [5-15, 15-25, 25-30, 35-40].
        The throughput is calculated as the average throughput over that
        bandpass.

        Parameters
        ----------
        sample_wavelengths : Array, meters
            The wavelengths at which to sample the filter. Must contain at
            least two values.

        Returns
        -------
        throughputs : Array
            The average throughput for each bandpass defined by
            sample_wavelengths.
        """
        mids = (sample_wavelenghts[1:] + sample_wavelenghts[:-1]) / 2
        diffs = np.diff(sample_wavelenghts)

        start = np.array([sample_wavelenghts[0] - diffs[0]/2])
        end = np.array([sample_wavelenghts[-1] + diffs[-1]/2])
        min_val = np.array([self.wavelengths.min()])
        max_val = np.array([self.wavelengths.max()])
        bounds = np.concatenate([start, mids, end])

        # Translate input wavelengths to indexes
        min_wavelength = self.wavelengths.min()
        max_wavelength = self.wavelengths.max()
        num_wavelength = len(self.wavelengths)
        wavelength_range = max_wavelength - min_wavelength
        bnd_indxs = num_wavelength * (bounds - min_wavelength)/wavelength_range
        bnd_indxs = np.clip(bnd_indxs, a_min=0, a_max=len(self.wavelengths))
        bnd_inds = np.round(bnd_indxs, decimals=0).astype(int)

        def nan_div(y, x):
            x_new = np.where(x == 0, 1, x)
            return np.where(x == 0, 0., y/x_new)

        def get_tp(start, end, weights, indexes):
            size = (end - start)
            val = np.where((indexes <= start) | (indexes >= end), \
                           0., weights).sum()
            return nan_div(val, size)

        starts = bnd_inds[:-1]
        ends   = bnd_inds[1:]
        # dwavelength = self.wavelengths[1] - self.wavelengths[0]
        indexes = np.arange(len(self.wavelengths))

        # weights = self.throughput/self.throughput.sum()
        weights = self.throughput
        out = vmap(get_tp, in_axes=(0, 0, None, None))(starts, ends, weights, indexes)
        return out


    def model(self : Filter, optics : Optics, **kwargs):
        """
        A base level modelling function designed to robustly handle the
        different combinations of inputs. Models the sources through the
        instrument optics and detector. Users must provide optics and some form
        of source, either via a scene, sources or single source input, but not
        multiple.

        Parameters
        ----------
        optics : Optics
            The optics through which to model the source objects.
        detector : Detector = None
            The detector to use with the observation. Defaults to the
            internally stored value.
        scene : Scene = None
            The scene to observe. Defaults to the internally stored value.
        sources : Union[dict, list, tuple) = None
            The sources to observe.
        source : dLux.sources.Source = None
            The source to observe.
        normalise_sources : bool = True
            Whether to normalise the sources before modelling. Default is True.
        flatten : bool = False
            Whether the output image should be flattened. Default is False.
        return_tree : bool = False
            Whether to return a Pytree like object with matching tree structure
            as the input scene/sources/source. Default is False.

        Returns
        -------
        image : Array, Pytree
            The image of the scene modelled through the optics with detector and
            filter effects applied if they are supplied. Returns either as a
            single array (if return_tree is false), or a pytree like object
            with matching tree strucutre as the input scene/sources/source.
        """
        return model(optics, filter=self, **kwargs)