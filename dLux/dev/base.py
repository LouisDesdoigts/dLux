import jax
import jax.numpy as np
import equinox as eqx
import typing
import dLux
from collections import OrderedDict

__author__ = "Louis Desdoigts"
__date__ = "30/08/2022"
__all__ = ["Telescope", "Optics", "Scene",
           "Filter", "Detector", "Observation"]

# Base Jax Types
Array =  typing.NewType("Array",  np.ndarray)

# Classes
Telescope   = typing.NewType("Telescope",   object)
Optics      = typing.NewType("Optics",      object)
Scene       = typing.NewType("Scene",       object)
Filter      = typing.NewType("Filter",      object)
Detector    = typing.NewType("Detector",    object)
# Observation = typing.NewType("Observation", object)\

"""
High level notes:

There should be some way to cache psf calculations during observations in order
to only calculte novel psfs for large observations.

TODO: Build the observation object, open questions here.

Q: Should the modelling_dict object store single dimension arrays, with 
recombinations happening after via some indexing parameter?

Q: For inputs like wavelenght, should the input type be Array or float? Its a 
single valued float 'array' of shape (0,)....
"""


class Telescope(dLux.base.Base):
    """
    A high level class desgined to model the behaviour of a telescope. It 
    stores a series different ∂Lux objects, and primarily passes the relevant 
    information between these objects in order to coherently model some 
    telescope observation.
    
    Attributes
    ----------
    optics : Optics
        An Optics object that is used to propagate wavefronts through some
        some optical configuration to generate a psf.
    scene : Scene
        A Scene object that stores the various source objects that the
        telescope is observing.
    detector : Detector
        A Detector object that is used to model the various instrumental 
        effects on a psf.
    filter : Filter
        A Filter object that is used to model the effective throughput of each
        wavelength though the optical system.
    """
    scene    : Scene
    optics   : Optics
    detector : Detector
    filter   : Filter
    # Observation: Observation
    
    
    def __init__(self     : Telescope,
                 optics   : Optics,
                 scene    : Scene,
                 detector : Detector = None,
                 filter   : Filter   = None,
                 # Observation :
                ) -> Telescope:
        """
        Parameters
        ----------
        optics : Optics, list
            An Optics object that is used to propagate wavefronts through some
            some optical configuration to generate a psf.
        scene : Scene, list
            A Scene object that stores the various source objects that the
            telescope is observing.
        detector : Detector (optional)
            A Detector object that is used to model the various instrumental
            effects on a psf.
        filter : Filter (optional)
            A Filter object that is used to model the effective throughput of
            each wavelength though the optical system.
        """
        self.scene    = Secene(scene)  if isinstance(list, scene) else scene
        self.optics   = Optics(optics) if isinstance(list, optics) else optics
        self.detector = Detector()     if detector is None else detector
        self.filter   = Filter()       if filter   is None else filter
        
        
    def model_scene(self : Telescope, scene : Scene = None) -> Array:
        """
        Models the scene through the telescope optics.
        
        Parameters
        ----------
        scene : Scene (optional)
            The scene to observe, defaults to using the internally stored scene
            if no value is passed.
        
        Returns
        -------
        psfs : Array
            The summed psfs of the scene modelled through the telescope.
        """
        # Get the scene
        scene = self.scene if scene is None else scene
        
        # vmap the appropriate functions
        vector_prop = jax.vmap(self.optics.propagate_mono, in_axes=(0, 0))
        source_prop = jax.vmap(vector_prop, in_axes=(0, 0))
        throughput_mapped = jax.vmap(self.filter.get_throughput, in_axes=0)
        
        # Decompose the scene and format the observation wavelengths
        modelling_dict = scene.decompose()
        wavelengths = modelling_dict['wavelengths']
        offsets = modelling_dict['positions']
        weights = np.expand_dims(modelling_dict['weights'] * \
                                 throughput_mapped(wavelengths), (-1, -2))

        # Model and combine the individual psfs
        out = weights * source_prop(wavelengths, offsets)
        psfs = out.sum((0, 1))
        return psfs
    
    
    def model_image(self : Telescope, detector : Detector = None,
                    scene : Scene = None) -> Array:
        """
        Models the scene through the telescope optics and detector.
        
        Parameters
        ----------
        detector : Detector (optional)
            The detector to use with the observation, defaults to using the
            internally stored detector if no value is passed.
        scene : Scene (optional)
            The scene to observe, defaults to using the internally stored scene
            if no value is passed.
        
        Returns
        -------
        image : Array
            The image of the scene modelled through the telescope with detector
            effects applied.
        """
        psfs = self.model_scene(scene=scene)
        detector = self.detector if detector is None else detector
        image = detector.apply_detector_layers(psfs)
        return image
    
    
class Optics(dLux.base.Base):
    """
    A high level class desgined to model the behaviour of some optical systems
    response to wavefronts.
    
    Attributes
    ----------
    layers: dict
        A collections.OrderedDict of 'layers' that define the transformations
        and operations upon some input wavefront through an optical system.
    """
    layers : dict
    
    
    def __init__(self : Optics, layers : list) -> Optics:
        """
        Parameters
        ----------
        layers : list
            A list of ∂Lux 'layers' that define the transformations and
            operations upon some input wavefront through an optical system.
        """
        self.layers = dLux.utils.list_to_dict(layers)
    
    
    def propagate_mono(self : Optics, wavelength : Array,
                       offset : Array = np.zeros(2)) -> Array:
        """
        Propagates a monochromatic point source through the optical layers.
        
        Parameters
        ----------
        wavelength : Array, meters
            The wavelength of the wavefront to propagate through the optical
            layers.
        offset : Array, radians, (optional)
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.
        
        Returns
        -------
        psf : Array
            The monochromatic point spread function after being propagated
            though the optical layers.
        """
        params_dict = {"OpticalSystem": self,
                       "wavelength": wavelenght,
                       "offset": offset}
        
        for key, layer in self.layers.items():
            params_dict = layer(params_dict)
        return params_dict["Wavefront"].wavefront_to_psf()
    
    
    def propagate_single(self : Optics, wavelenghts : Array,
                         offset  : Array = np.zeros(2),
                         weights : Array = np.ones(1),
                         flux    : Array = np.ones(1)) -> Array:
        """
        Propagates a single broadband point source through the optical layers.
        
        Parameters
        ----------
        wavelengths : Array, meters
            The wavelengths of the wavefront to propagate through the optical
            layers.
        offset : Array, radians, (optional)
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.
        weights : Array, (optional)
            The relative weights of each wavelength from the source
        flux : Array, (optional)
            The total flux of the source
        
        Returns
        -------
        psf : Array
            The point spread function of the point source after being
            propagated though the optical layers.
        """
        # Mapping propagator over wavelengths
        prop_wf_map = jax.vmap(self.propagate_mono, in_axes=(0, None))
        
        # Propagate wavelengths
        monochromatic_psfs = prop_wf_map(wavelenghts, offset)/len(wavels)
        
        # Sum into single psf
        psf = flux * (weights * monochromatic_psfs).sum(0)
        return psf
    
    
    def debug_prop(self : Optics, wavelength : Array,
                   offset : Array = np.zeros(2)) -> Array:
        """
        Propagates a monochromatic point source through the optical layers,
        while also returning the intermediate state of the parameter dictionary
        and layers after each layer application.
        
        Parameters
        ----------
        wavelength : Array, meters
            The wavelength of the wavefront to propagate through the optical
            layers.
        offset : Array, radians, (optional)
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.
        
        Returns
        -------
        psf : Array
            The monochromatic point spread function after being propagated
            though the optical layers.
        intermediate_dicts : list
            The intermediate states of the parameters dictionary.
        intermediate_layers : list
            The intermediate states of each layer after being applied to the
            wavefront.
        """
        params_dict = {"OpticalSystem": self,
                       "wavelength": wavel,
                       "offset": offset}
        
        intermediate_dicts = []
        intermediate_layers = []
        for key, layer in self.layers.items():
            params_dict = layer(params_dict)
            intermediate_dicts.append(params_dict.copy())
            intermediate_layers.append(deepcopy(layer))
        
        return params_dict["Wavefront"].wavefront_to_psf(), \
                                intermediate_dicts, intermediate_layers
    
    
class Scene(dLux.base.Base):
    """
    A high level class representing some 'astrophysical scene', which is
    composed of Sources. This class mainly serves as an interface between the
    individual source objects and the Optics/Telescope classes.
    
    Attributes
    ----------
    sources : dict
        A collections.OrderedDict
    """
    
    sources: dict
    
    def __init__(self, sources : list):
        """
        
        """
        self.sources = dLux.utils.list_to_dict(sources)
        
    def get_positions(self : Scene) -> Array:
        """
        
        """
        pass
    
        
    def get_wavelengths(self : Scene) -> Array:
        """
        
        """
        pass
    
    def decompose(self : Scene) -> dict:
        """
        Note currectly only works with source spectrum of the save length
        
        This could be expanded to arbitrary using .flatten() methods 
        
        
        """
        keys = list(self.sources.keys())
        source = self.sources[keys[0]].normalise()
        
        # Correctly shaped arrays must exist in order to be correctly
        # appended to, so we must initiliase the source dictionary 
        # outside of the itterative loop
        source_dict = {"wavelengths": source._get_wavelengths(), 
                       "weights":     source._get_weights() * \
                                           source._get_flux(),
                       "positions":   source._get_position(),
                       "resolved":    source._is_resolved(),
                       "source_key":  [keys[0]],}
        
        for i in range(1, len(keys)):
            key = keys[i]
            source = self.sources[key].normalise()
            
            wavelengths = source._get_wavelengths()
            source_dict['wavelengths'] = np.append(
                source_dict['wavelengths'], wavelengths, axis=0)

            weights = source._get_weights() * source._get_flux()
            source_dict['weights'] = np.append(
                source_dict['weights'], weights, axis=0)
            
            positions = source._get_position()
            source_dict['positions'] = np.append(
                source_dict['positions'], positions, axis=0)
            
            resolved = source._is_resolved()
            source_dict['resolved'] = np.append(
                source_dict['resolved'], resolved, axis=0)
            
            source_dict['source_key'] = \
                source_dict['source_key'] + [key]

        return source_dict
    
    
class Filter(dLux.base.Base):
    """
    
    """
    wavelengths : Array
    throughput  : Array
    filter_name : str = eqx.static_field()
        
    def __init__(self        : Filter, 
                 wavelengths : Array = None,
                 throughput  : Array = None,
                 filter_name : str     = None) -> Filter:
        """
        
        """
        
        # Take the filter name as the priority input
        if filter_name is not None:
            # TODO: Pre load filters
            raise NotImplementedError("You know what this means")
            pass
            
            # Check that wavelengths and throughput are not specified
            if wavelengths is not None or throughput is not None:
                raise ValueError("If filter_name is specified, wavelengths \
                and throughput can not be specified")
        
        # Neither is specified
        elif wavelengths is None and throughput is None:
            self.wavelengths = np.array([1.])
            self.throughput  = np.array([1.])
            self.filter_name = 'custom'
                
        # Check that both wavelengths and throughput are specified
        elif (wavelengths is     None and throughput is not None) or \
             (wavelengths is not None and throughput is     None):
            raise ValueError("If either wavelengths or throughput is\
            specified, then both must be specified")
                
        # Both wavelengths and throughput are specified
        else:
            assert len(wavelengths) != len(throughput), "wavelengths and \
            throughput must have the same dimension"
            self.wavelengths = np.asarray(wavelengths, dtype=float)
            self.throughput  = np.asarray(throughput,  dtype=float)
            self.filter_name = 'custom'
    
    def get_throughput(self : Filter, wavelengths : Array):
        """
        
        """
        # Translate input wavelengths to indexes 
        min_wavelength = self.wavelengths.min()
        max_wavelength = self.wavelengths.max()
        num_wavelength = self.wavelengths.shape[0]
        indxs = num_wavelength * (wavelengths - min_wavelength)/max_wavelength
        return jax.scipy.ndimage.map_coordinates(self.throughput, \
                                        np.array([indxs]), 1, 'nearest')



class Detector(dLux.base.Base):
    """
    Contains a list (dictionary?) of detector 'layers'

    Can also contain a spectral sensitivty value for each wavelength
    This would imply that the detecotr should in some cases recieve 
    the full per-wavelength psf. Should be a simple logic call

    I guess then it should take in spectral channels and have a 
    'combine spectra' layer that puts them all together once 
    all the spectrally responsive operations are done 

    Each layer should also take a params dict, not an image for full 
    flexibility
    """

    layers: dict

    def __init__(self: Detector, layers : list = []):
        """

        """
        self.layers = dLux.utils.list_to_dict(layers)


    def apply_detector_layers(self, image):
        """

        """

        for key, layer in self.layers.items():
            image = layer(image)
        return image


class Observation(dLux.base.Base):
    """

    """
    pointing : Array
    roll_angle : Array

    def __init__(self):
        raise NotImplementedError("Duh")


