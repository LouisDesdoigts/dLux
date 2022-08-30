import jax
import jax.numpy as np
import equinox as eqx
import abc
import typing
from collections import OrderedDict
import dLux

__author__ = "Louis Desdoigts"
__date__ = "30/08/2022"
__all__ = ["Telescope", "Optics", "Scene",
           "Filter", "Detector", "Observation"]

# Base Jax Types
Scalar = typing.NewType("Scalar", np.ndarray) # 0d
Vector = typing.NewType("Vector", np.ndarray) # 1d
Array =  typing.NewType("Array",  np.ndarray) # 2d +

Telescope   = typing.NewType("Telescope",   object)
Optics      = typing.NewType("Optics",      object)
Scene       = typing.NewType("Scene",       object)
Filter      = typing.NewType("Filter",      object)
Detector    = typing.NewType("Detector",    object)
Observation = typing.NewType("Observation", object)



class Telescope(eqx.Module):
    """
    A high level class to store the various compoennets needed to model a 
    real astrophysical scene with a telescope.
    
    Scene: Contains a list (dictionary?) of sources and a set of functions 
    needed to interface the parameterised sources with the Optics
    
    Optics: an OpticalSystem object used to model Source Wavefront through
    the optial system
    
    Detector: models detector effects
    
    Filter: Optional? Optics layer? Interpolate source to filter or filter 
    to source?? 
    I think source to filter to allow for low dimensional 
    spectral representations of objects - Makes array formatting hard tho...
    Under the assumption that arbitrary stack of vmap are equally efficent
    then we can vamp over objects, then wavelengths, letting us the filter 
    function only as a modifier to the spectral weight - this should be 
    simplest
    Using a simple interpolat should allow for easy zeroing of wavelengths
    outside of its tranmissive bandpass. This could possibly also be used
    to prevenet calculations of wavelengths with insignificant relative
    weights 
    
    This class handles the primary interfacing between all of these obejcts
    
    Observation: Optional? Probably doesnt need to be an eqx module. 
    Cache PSF calcuations for background objects between dithers??
    
    Other things: Observation scheme -> Not sure yet, but will likely be 
    needed to correctly parameterise objects properties over some data set
    ie. do we want fluxes to be constant or variable over the set of produced
    images.
    
    
    Questions:
        Where should the modelled wavelengths be defined? by the source or
        filter?
        
    Thoughts:
        Under the assumption that arbitrary stacks of vmap perform identically
        to shaped/formatted inputs we should model objects under their source
        spectrums and interpolate the filter weight values
        
        I believe that Filter need to be a property of the Telescope as it only
        affect the *relative* weights of each spectral feature. If it was a
        layer then 
    
    """
    
    scene    : Scene
    optics   : Optics # Can instansiate with a list of layers?
    detector : Detector
    filter   : Filter 
    # Observation: Observation # Who tf knows at this point - A dictionary?
    
    
    def __init__(self     : Telescope, 
                 optics   : Optics,
                 scene    : Scene, 
                 detector : Detector = None,
                 filter   : Filter   = None,
                 # Observation : 
                ) -> Telescope:
        
        self.scene    = scene
        self.optics   = optics
        self.detector = Detector() if detector is None else detector
        self.filter   = Filter()   if filter   is None else filter
        
    # def model_scene(self : Telescope, scene : Scene = None) -> Array:
    def model_scene(self : Telescope) -> Array:
        """
        
        """
        
        vector_prop = jax.vmap(self.optics.propagate_mono, in_axes=(0, 0))
        source_prop = jax.vmap(vector_prop, in_axes=(0, 0))
        throughput_mapped = jax.vmap(self.filter.get_throughput, in_axes=0)
        
        modelling_dict = self.scene.decompose()
        wavelengths = modelling_dict['wavelengths']
        offsets = modelling_dict['positions']
        weights = np.expand_dims(modelling_dict['weights'] * \
                                 throughput_mapped(wavelengths), (-1, -2))

        out = weights * source_prop(wavelengths, offsets)
        psfs = out.sum(1)
        image = psfs.sum(0)
        
        return image
    
    
class Optics(eqx.Module):
    """ Optical System class, Equinox Modle
    
    Attributes
    ----------
    layers: list, required
        - A list of layers that defines the tranformaitons and operations of 
        the system (typically optical)
    """
    layers : dict

    def __init__(self, layers):
        
        # TODO: Make this into a function like in scene
        # also maybe make into a utils function since its
        # used in Scene, Optics, and Detector
        # Construct names list and identify repeats
        names, repeats = [], []
        for i in range(len(layers)):
            
            if hasattr(layers[i], 'name') and layers[i].name is not None:
                name = layers[i].name
            else:
                name = str(layers[i]).split('(')[0]
                
            # Check for Repeats
            if name in names:
                repeats.append(name)
            names.append(name)

        # Add indexes to repeats
        repeats = list(set(repeats))
        for i in range(len(repeats)):
            idx = 0
            for j in range(len(names)):
                if repeats[i] == names[j]:
                    names[j] = names[j] + '_{}'.format(idx)
                    idx += 1
        
        # Turn list into Dictionary
        layers_dict = OrderedDict()
        for i in range(len(names)):
            layers_dict[names[i]] = layers[i]
        self.layers = layers_dict
            
        # Register layers as DotMap object
        # self.layers = DotMap(layers_dict)
        
    def debug_prop(self, wavel, offset=np.zeros(2)):        
        """
        
        """
        params_dict = {"OpticalSystem": self, 
                       "wavelength": wavel,
                       "offset": offset}
        
        intermed_dicts = []
        intermed_lays = []
            
        for key, layer in self.layers.items():
            params_dict = layer(params_dict)
            intermed_dicts.append(params_dict.copy())
            intermed_lays.append(deepcopy(layer))
            
        return params_dict["Wavefront"].wavefront_to_psf(), intermed_dicts, intermed_lays
    
    def propagate_mono(self, wavel, offset=np.zeros(2)):        
        """
        
        """
        params_dict = {"OpticalSystem": self, 
                       "wavelength": wavel,
                       "offset": offset}
        
        for key, layer in self.layers.items():
            params_dict = layer(params_dict)
        return params_dict["Wavefront"].wavefront_to_psf()
    
    def propagate_single(self, wavels, offset=np.zeros(2), weights=1., flux=1.):
        """
        Only propagates a single star, allowing wavelength input
        sums output to single array
        
        Wavels must be an array and the same shape as weights if provided
        """
        
        # Mapping over wavelengths
        prop_wf_map = jax.vmap(self.propagate_mono, in_axes=(0, None))
        
        # Apply spectral weighting
        psfs = weights * prop_wf_map(wavels, offset)/len(wavels)
        
        # Sum into single psf and apply flux
        image = flux * psfs.sum(0)
        
        return image
    
    
class Scene(eqx.Module):
    """
    Contains a list (dictionary?) of sources and a set of functions needed to 
    interface the parameterised sources with the Optics
    
    Requires a _decompose_sources() method to return a list of positions
    for each source, along with the wavelengths/weights for each of
    these sources. This information is then fed through the telescope Filter
    to generate true relative weights
    
    Should implement a .decompose() method that returns a dictionary with 
    the following structure:
        {'Wavelength' : [...],
         'Offset'     : [...], 
         'Flux'       : [...], 
         'Resolved'   : [...]}
    Ideally this could be vmapped over, so that each image can have its own
    dictionary object.
    """
    
    sources: dict
    
    def __init__(self, sources : list):
        """
        
        """
        self.sources = self._list_to_dict(sources)
        
    def _list_to_dict(self, list_in):
        """
        
        """
        # Construct names list and identify repeats
        names, repeats = [], []
        for i in range(len(list_in)):
            
            if hasattr(list_in[i], 'name') and list_in[i].name is not None:
                name = list_in[i].name
            else:
                name = str(list_in[i]).split('(')[0]
                
            # Check for Repeats
            if name in names:
                repeats.append(name)
            names.append(name)

        # Add indexes to repeats
        repeats = list(set(repeats))
        for i in range(len(repeats)):
            idx = 0
            for j in range(len(names)):
                if repeats[i] == names[j]:
                    names[j] = names[j] + '_{}'.format(idx)
                    idx += 1
        
        # Turn list into Dictionary
        dict_out = OrderedDict()
        for i in range(len(names)):
            dict_out[names[i]] = list_in[i]
        
        return dict_out
        
        
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
    
    
class Filter(eqx.Module):
    """
    
    """
    wavelengths : Vector
    throughput  : Vector
    filter_name : str = eqx.static_field()
        
    def __init__(self        : Filter, 
                 wavelengths : Vector = None,
                 throughput  : Vector = None,
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
    
    def get_throughput(self : Filter, wavelengths : Vector):
        """
        
        """
        # Translate input wavelengths to indexes 
        min_wavelength = self.wavelengths.min()
        max_wavelength = self.wavelengths.max()
        num_wavelength = self.wavelengths.shape[0]
        indxs = num_wavelength * (wavelengths - min_wavelength)/max_wavelength
        return jax.scipy.ndimage.map_coordinates(self.throughput, \
                                        np.array([indxs]), 1, 'nearest')


    
class Detector(eqx.Module):
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
    
    layers: list # To be made into a dict
    
    def __init__(self: Detector, layers : list = []):
        """
        
        """
        self.layers = layers
    
    def apply_detector_layers(self, image):
        """
        
        """
        for i in range(len(self.detector_layers)):
            image = self.detector_layers[i](image)
        return image

    
    
class Observation(eqx.Module):
    """
    
    """
    pointing : Vector
    roll_angle : Scalar
    
    def __init__(self):
        raise NotImplementedError("Duh")
        
        
