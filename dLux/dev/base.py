import jax
import jax.numpy as np
import equinox as eqx
import abc
import typing
import dLux
import optax
from collections import OrderedDict

__author__ = "Louis Desdoigts"
__date__ = "30/08/2022"
__all__ = ["Base", "Telescope", "Optics", "Scene",
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

# class Base(abc.ABC, eqx.Module):
#     def get_leaf(self, pytree, path):
#         """
#         Recuses down the path of the pytree
#         """
#         key = path[0]
#         pytree = pytree.__dict__[key] if isinstance(pytree, eqx.Module) else \
#                  pytree[key]

#         # Return param if at the end of path, else recurse
#         return pytree if len(path) == 1 else self.get_leaf(pytree, path[1:])
    
#     def update_pytree(self, paths, values):
#         """
#         Updates the `pytree` leaves specificied by params_paths with values
#         """
#         # Returns a tuple of leaves specified by paths
#         get_leaves = lambda pytree : tuple([self.get_leaf(pytree, paths[i]) \
#                                             for i in range(len(paths))])

#         # Updates the leaf if passed a function
#         update_leaf = lambda leaf, leaf_update: leaf_update(leaf)\
#                 if isinstance(leaf_update, typing.Callable) else leaf_update

#         # Updates the leaves specified by paths
#         update_values = tuple([update_leaf(self.get_leaf(self, paths[i]), \
#                                 values[i]) for i in range(len(paths))])

#         return eqx.tree_at(get_leaves, self, update_values)
    
#     def update_params(self, update_fn, **kwargs):
#         """
#         Update the paramaters with the given input value
#         """
#         params, param_paths = update_fn(**kwargs)
        
#         # Get value & paths to be updated
#         update_values = tuple([params[i] for i in range(len(params)) \
#                               if params[i] is not None])
        
#         # Get paths to be parameters to be updated
#         update_paths = tuple([param_paths[i] for i in range(len(params)) \
#                               if params[i] is not None])
        
#         # Return updated pytree
#         return self.update_pytree(update_paths, update_values)
    
#     def update_and_model(self, update_fn, model_fn, **kwargs):
#         """
#         Updates the given input parameters, and then models the psf
#         """
#         return getattr(self.update_params(update_fn, **kwargs), model_fn)()
    
    
class Base(abc.ABC, eqx.Module):
    
    ######################
    ### Hidden methods ###
    ######################
    def _get_leaf(self, pytree, path):
        """
        Recuses down the path of the pytree
        pytree must be passed in as an input in order to be able to recurse 
        properly
        """
        key = path[0]
        pytree = pytree.__dict__[key] if isinstance(pytree, eqx.Module) else \
                 pytree[key]
        
        # Return param if at the end of path, else recurse
        return pytree if len(path) == 1 else self._get_leaf(pytree, path[1:])
    
    def _unwrap_paths(self, paths, values=None, path_dict=None):
        """
        Helper function to unwrap nested paths and approriately format
        different inputs
        """
        new_paths = []
        keys = list(path_dict.keys()) if path_dict is not None else []
        
        # Only unwrap paths
        if values is None:
            for path in paths:
                # Check if path has sub-paths
                if isinstance(path[0], typing.Union[list, tuple]) or path[0] in keys:
                    for sub_path in path:
                        if sub_path in keys:
                            new_paths.append(path_dict[sub_path])
                        else:
                            new_paths.append(sub_path)
                else:
                    if path in keys:
                        new_paths.append(path_dict[path])
                    else:
                        new_paths.append(path)
                    
        # Unwrap paths and values
        else:
            new_values = []
            for path, value in zip(paths, values):
                # Check if path has sub-paths
                if isinstance(path[0], typing.Union[list, tuple]) or path[0] in keys:
                    for sub_path in path:
                        if sub_path in keys:
                            new_paths.append(path_dict[sub_path])
                        else:
                            new_paths.append(sub_path)
                        new_values.append(value)
                else:
                    if path in keys:
                        new_paths.append(path_dict[path])
                    else:
                        new_paths.append(path)
                    new_values.append(value)
                    
        # Wrap non-list path objects in list
        new_paths = [[path] if not isinstance(path, typing.Union[list, tuple]) \
                 else path for path in new_paths]
        
        # Return values
        if values is None:
            return new_paths
        else:
            return new_paths, new_values

        
    ########################
    ### Accessor methods ###
    ########################
    def get_leaf(self, path, path_dict=None):
        """
        Used to get a single leaf
        Kinda useless, but why not have the functionality
        """
        # Check if path dict exists
        if path_dict is not None:
            # Allow for string and single entry list inputs
            if isinstance(path, str):
                path = path_dict[path]
            else:
                path = path_dict[path[0]]
        
        # Get the leaf
        return self._get_leaf(self, path)
    
    def get_leaves(self, paths, path_dict=None):
        """
        Used to get multiple leaves from a list of paths
        
        self._unrwap_paths automatically applies the path_dict, hence it is 
        not passed to self.get_leaf
        """
        # Unwrap paths
        new_paths = self._unwrap_paths(paths, path_dict=path_dict)
        return tuple([self.get_leaf(path) for path in new_paths])
    
    
    #######################
    ### Updater methods ###
    #######################
    def update_pytree(self, paths, values, path_dict=None):
        """
        Updates the `pytree` leaves specificied by params_paths with values
        """
        # Unwrap paths
        new_paths, new_values = self._unwrap_paths(paths, values=values, path_dict=path_dict)
        
        # Define 'where' function and update pytree
        get_leaves_fn = lambda pytree: pytree.get_leaves(new_paths)
        return eqx.tree_at(get_leaves_fn, self, list(new_values))
    
    def apply_fns(self, paths, fns, path_dict=None):
        """
        Updates the `pytree` leaves specificied by params_paths with values
        """
        # Unwrap paths
        new_paths, new_fns = self._unwrap_paths(paths, values=fns, path_dict=path_dict)
        
        # Call using the get_leaves function in order to properly apply path dictionary
        new_values = [fn(leaf) for fn, leaf in zip(new_fns, self.get_leaves(new_paths))]                
                
        # Define 'where' function and update pytree
        get_leaves_fn = lambda pytree: pytree.get_leaves(new_paths)
        return eqx.tree_at(get_leaves_fn, self, list(new_values))
    
    
    #######################
    ### Optax functions ###
    #######################
    def get_filter_spec(self, paths, path_dict=None):
        """
        Returns a filter_spec 
        """
        filter_spec = jax.tree_map(lambda _: False, self)
        values = len(paths) * [True]
        return filter_spec.update_pytree(paths, values, path_dict=path_dict)
    
    def get_param_spec(self, paths, groups, get_filter_spec=False, path_dict=None):
        """
        Returns a params_spec
        """
        param_spec = jax.tree_map(lambda _: 'null', self)
        param_spec = param_spec.update_pytree(paths, groups, path_dict=path_dict)
        
        # For some weird ass reason this works correctly but single liner doesnt
        if not get_filter_spec:
            return param_spec
        else:
            return param_spec, self.get_filter_spec(paths, path_dict=path_dict)

    def get_pytree_optimiser(self, paths, optimisers, get_filter_spec=False, path_dict=None):
        """
        Colates optimisers for each leaf into a coherent optax pytree optimiser
        """
        # Construct groups and get param_spec
        groups = [str(i) for i in range(len(paths))]
        param_spec = self.get_param_spec(paths, groups, path_dict=path_dict)
            
        # Generate optimiser dictionary
        opt_dict = dict([(groups[i], optimisers[i]) \
                         for i in range(len(groups))])
        
        # Assign the null group
        opt_dict['null'] = optax.adam(0.)

        # Get optimiser object
        optim = optax.multi_transform(opt_dict, param_spec)
    
        # For some weird ass reason this works correctly but single liner doesnt
        if not get_filter_spec:
            return optim
        else:
            return optim, self.get_filter_spec(paths, path_dict=path_dict)
    
    
    #########################
    ### Numpyro functions ###
    #########################
    def update_and_model(self, model_fn, paths, values, path_dict=None, *args, **kwargs):
        """
        Updates the given input parameters, and then models the psf
        """
        return getattr(self.update_pytree(paths, values, path_dict=path_dict), model_fn)(*args, **kwargs)



class Telescope(Base):
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
    
    def model_scene_flat(self : Telescope) -> Array:
        return self.model_scene().flatten()
    
    # def model_image(self : Telescope, flatten=False : bool) -> Array:
    def model_image(self : Telescope, flatten : bool=False) -> Array:
        if not flatten:
            return self.detector.apply_detector_layers(self.model_scene())
        else:
            return self.detector.apply_detector_layers(self.model_scene()) \
                    .flatten()
    
    # def model_image_flat(self : Telescope) -> Array:
    #     return self.model_image().flatten()
    
    
class Optics(Base):
    """ Optical System class, Equinox Modle
    
    Attributes
    ----------
    layers: list, required
        - A list of layers that defines the tranformaitons and operations of 
        the system (typically optical)
    """
    layers : dict

    def __init__(self, layers):
        self.layers = dLux.utils.list_to_dict(layers)
        
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
    
    
class Scene(Base):
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
    
    
class Filter(Base):
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



class Detector(Base):
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


class Observation(eqx.Module):
    """

    """
    pointing : Vector
    roll_angle : Scalar

    def __init__(self):
        raise NotImplementedError("Duh")


