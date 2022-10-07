from __future__ import annotations
import jax
import jax.numpy as np
from jax import vmap
import equinox as eqx
from copy import deepcopy
import dLux
import typing
import optax
import abc
from collections import OrderedDict
import warnings
import jax.tree_util as jtu


__all__ = ["OpticalSystem", "Instrument", "Optics", "Scene",
           "Filter", "Detector"]

# Types
Array     = typing.NewType("Array",  np.ndarray)
list_like = typing.Union[list, tuple]
Path      = typing.Union[list, tuple]
Pytree    = typing.NewType("Pytree", object)
Leaf      = typing.Any


class Base(abc.ABC, eqx.Module):
    """
    An abstract base class that is used to give a user-friendly API for working
    with PyTrees, specifically using Equniox. This can be thought of as
    extending the equinox.Module class. It also has some methods with a 
    user-friendly interface for some other usefull packages such as optax
    and numpyro.
    
    If you are unfamiliar with PyTrees, check out this jax tutorial
    and have a look a the equinox docs to see how they can be extending to give
    object-oriented jax:
     - https://jax.readthedocs.io/en/latest/jax-101/05.1-pytrees.html
     - https://docs.kidger.site/equinox/
     
    New concept: Paths
        In order to make working with PyTrees easier, there is one concept that
        is introduced, the paths object. a path object is not a unique or new
        class of object type, just a helpfull way of thinking about navigating
        PyTrees. We define a 'path' here as list/tuple that contains either
        strings or integers. Each path object refers to a unique 'leaf' in the
        PyTree. Since each PyTrees is constructed from nested lists, tuples and
        dictionaries (and since we're using Equinox, Equinox.Modules too), we 
        can refer to any unique leaf via a chain of strings and intergers. 
        
        Example path objects:
            path1 = ['param1', 3, 'nested_param3', 5]
            path2 = ('p1', 'p2', 'p3')
            path3 = (1, 4, 6, 3, 6, 2, 'param')
            
        These objects are quite simple, but are worth clarifying since they are
        integral to this new PyTree interface.
        
    New concept: Path Dictionary
        Since PyTrees can have both very deep and wide structures it would be 
        impractical to always refer to each leaf via its full absolute path. 
        The path dictionary is simply a dictionary that allows a simple key to 
        be used to refer to the full path to some leaf. Note the key must be 
        unique to and and all parameter names within the PyTree structure!
    """
    
    ######################
    ### Hidden methods ###
    ######################
    def _get_leaf(self : Pytree, pytree : Pytree, path : Path) -> Leaf:
        """
        A hidden class desinged to recurse down a pytree following the path, 
        returning the leaf at the end of the path.
        
        Base case: len(path) == 1
            In this case the leaf referred to by the single path entry is 
            returned (and hence recursively sent up to the initial call)
            
        Recursive case: len(path) > 1
            In this case the function takes the PyTree like object referred to
            by the first entry in path, and recursively calls this function
            with this new pytree object and the path without the first entry
        
        Parameters
        ----------
        pytree : Pytree
            The pytee object to recurse though
        path : Path
            The path to recurse down
            
        Returns
        -------
        leaf : Leaf
            The leaf object specified at the end of the path object
        """
        key = path[0]
        pytree = pytree.__dict__[key] if isinstance(pytree, eqx.Module) else \
                 pytree[key]

        # Return param if at the end of path, else recurse
        return pytree if len(path) == 1 else self._get_leaf(pytree, path[1:])

    # TODO: Re-write the logic in this a bit nice for optional values input
    def _unwrap_paths(self : Pytree, paths : list_like, 
                      values : list_like = None, 
                      path_dict : dict = None) -> list_like:
        """
        Helper function designed to unwrap nested paths, while also extracting 
        the absolute paths from the path dictionary. It similarly will tile out
        the correct value to apply for each nested path object. It outputs 
        a flat, non-nested set of absolute paths and values, as this is the
        format required by Equinox in order to simultaneously update multiple 
        parameters.
        
        Parameters
        ----------
        paths : list_like
            A list/tuple of nested paths. Note path objects can only be 
            nested a single time
        values : list_like (optional)
            A list/tuple of values (which can be functions) to be 
            updated/applied to the leaves specified by paths. These can not 
            be nested
        path_dict : dict (optional)
            A dictionary of absolute paths 
            
        Returns
        -------
        new_paths : list
            A flat list of absolute paths
        new_values : list (if values is not None)
            A flat list of values/functions to be updated/applied to the 
            leaves specified by new_paths
        """
        new_paths = []
        keys = list(path_dict.keys()) if path_dict is not None else []

        # Only unwrap paths
        if values is None:
            for path in paths:
                # Check if path has sub-paths
                if isinstance(path[0], typing.Union[list, tuple]) or \
                                                            path[0] in keys:
                    # Un-nest sub paths
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
                if isinstance(path[0], typing.Union[list, tuple]) or \
                                                            path[0] in keys:
                    # Un-nest sub-paths
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
        new_paths = [
            [path] if not isinstance(path, typing.Union[list, tuple]) else path
            for path in new_paths
        ]

        # Return values
        if values is None:
            return new_paths
        else:
            return new_paths, new_values

    ########################
    ### Accessor methods ###
    ########################
    def get_leaf(self : Pytree, path : Path, path_dict : dict = None) -> Leaf:
        """
        Returns the leaf specified by path
        
        Parameters
        ----------
        path : Path
            The path to recurse down
        path_dict : dict (optional)
            A dictionary of absolute paths 
            
        Returns
        -------
        leaf : Leaf
            The leaf object specified at the end of the path object
        """
        new_path = self._unwrap_paths([path], path_dict=path_dict)[0]

        # Get the leaf
        return self._get_leaf(self, new_path)

    def get_leaves(self : Pytree, paths : list_like, 
                    path_dict : dict = None) -> list:
        """
        Returns a list of leaves specified by the paths
        
        Parameters
        ----------
        paths : list_like
            A list/tuple of nested paths. Note path objects can only be 
            nested a single time
        path_dict : dict (optional)
            A dictionary of absolute paths 
            
        Returns
        -------
        leaves : list
            The list of leaf objects specified by the paths object
        """        
        # Unwrap paths
        new_paths = self._unwrap_paths(paths, path_dict=path_dict)
        return [self._get_leaf(self, path) for path in new_paths]

    #######################
    ### Updater methods ###
    #######################
    def update_leaves(self : Pytree, paths : list_like, values : list_like, \
                      path_dict : dict = None) -> Pytree:
        """
        Returns an updated version of the pytree with the leaves speficied by 
        paths updated with the values in values.
        
        Parameters
        ----------
        paths : list_like
            A list/tuple of nested paths. Note path objects can only be 
            nested a single time.
        values : list_like
            A list/tuple of new values to be applied to the leaves 
            specified by paths. These can not be nested.
        path_dict : dict (optional)
            A dictionary of absolute paths .
            
        Returns
        -------
        pytree : Pytree
            An updated version of the current pytree
        """
        # Unwrap paths
        new_paths, new_values = self._unwrap_paths(
            paths, values=values, path_dict=path_dict
        )

        # Define 'where' function and update pytree
        get_leaves_fn = lambda pytree: pytree.get_leaves(new_paths)
        return eqx.tree_at(get_leaves_fn, self, new_values)


    def apply_to_leaves(self : Pytree, paths : list_like, fns : list_like, \
                        path_dict : dict = None) -> Pytree:
        """
        Returns an updated version of the pytree with the the input functions 
        applied to the leaves speficied by the paths.
        
        Parameters
        ----------
        paths : list_like
            A list/tuple of nested paths. Note path objects can only be 
            nested a single time.
        fns : list_like
            A list/tuple of functions to be applied to the leaves specified
            by paths. These can not be nested.
        path_dict : dict (optional)
            A dictionary of absolute paths.
            
        Returns
        -------
        pytree : Pytree
            An updated version of the current pytree
        """        
        # Unwrap paths
        new_paths, new_fns = self._unwrap_paths(paths, values=fns, \
                                                path_dict=path_dict)

        # Call using the get_leaves function in order to properly apply path 
        # dictionary
        new_values = [fn(leaf) for fn, leaf in zip(new_fns, \
                                                   self.get_leaves(new_paths))]

        # Define 'where' function and update pytree
        get_leaves_fn = lambda pytree: pytree.get_leaves(new_paths)
        return eqx.tree_at(get_leaves_fn, self, new_values)
    
    
    #########################
    ### Equinox functions ###
    #########################
    def get_filter_spec(self : Pytree, paths : list_like, \
                        path_dict : dict = None) -> Pytree:
        """
        Returns 'filter_spec' object, to be used in conjunction with the 
        Equinox filter functions. A 'filter_spec' is a Pytree with a matching
        tree strucutre, but with boolean values at the leaves. This is 
        primarily used by the Equinox.filter_grad() and 
        Equinox.filter_value_and_grad() functions, passed in as the optional 
        'arg' argument. It is used to either turn on or off gradient 
        calculations with respect to each leaf.
        
        Parameters
        ----------
        paths : list_like
            A list/tuple of nested paths. Note path objects can only be 
            nested a single time.
        path_dict : dict (optional)
            A dictionary of absolute paths.
            
        Returns
        -------
        filter_spec : Pytree
            An pytree of matching structre with boolean values at the 
            leaves
        """
        filter_spec = jax.tree_map(lambda _: False, self)
        values = len(paths) * [True]
        return filter_spec.update_leaves(paths, values, path_dict=path_dict)
    
    
    #######################
    ### Optax functions ###
    #######################
    def get_param_spec(self : Pytree, paths : list_like, groups : list_like, \
                       get_filter_spec : bool = False, \
                       path_dict : dict = None) -> Pytree:
        """
        Returns 'param_spec' object, to be used in conjunction with the 
        Optax.multi_transform functions. The param_spec is a pytree of matching
        strucutre that has strings assigned to every node, denoting the group
        that is belongs to. Each of these groups can then have unique optimiser
        objects assigned to them. This is typically used to assign different
        learning rates to different parameters.
        
        Note this sets the default or non-trainable group to 'null'.

        Parameters
        ----------
        paths : list_like
            A list/tuple of nested paths. Note path objects can only be 
            nested a single time.
        groups : list_like
            A list/tuple of strings, denoting which group to assign the 
            corresponding leaves denoted by paths to.
        get_filter_spec : bool = False
            Boolean defining whether to return a corresponding filter_spec
            object.
        path_dict : dict (optional)
            A dictionary of absolute paths.
            
        Returns
        -------
        param_spec : Pytree
            An pytree of matching structre with string values at the 
            leaves specified by groups. 
        """
        param_spec = jax.tree_map(lambda _: "null", self)
        param_spec = param_spec.update_leaves(paths, groups, \
                                              path_dict=path_dict)

        # For some weird ass reason this works correctly but single liner 
        # doesn't
        if not get_filter_spec:
            return param_spec
        else:
            return param_spec, self.get_filter_spec(paths, path_dict=path_dict)

    
    def get_optimiser(self : Pytree, paths : list_like, \
                      optimisers : list_like, get_filter_spec : bool = False, \
                      path_dict : dict = None) -> object:
        """
        Returns an Optax.GradientTransformion object, with the optimisers 
        specified by optimisers applied to the leaves specified by paths.

        Parameters
        ----------
        paths : list_like
            A list/tuple of nested paths. Note path objects can only be 
            nested a single time.
        optimisers : list_like
            A list/tuple of optax.GradientTransformation objects to be 
            applied to the leaves specified by paths.
        get_filter_spec : bool = False
            Boolean defining whether to return a corresponding filter_spec
            object.
        path_dict : dict (optional)
            A dictionary of absolute paths.
            
        Returns
        -------
        optimiser : Optax.GradientTransformion
            An Optax.GradientTransformion object, with the optimisers 
            specified by optimisers applied to the leaves specified 
            by paths.
        """
        # Construct groups and get param_spec
        groups = [str(i) for i in range(len(paths))]
        param_spec = self.get_param_spec(paths, groups, path_dict=path_dict)

        # Generate optimiser dictionary
        opt_dict = dict([(groups[i], optimisers[i]) \
                         for i in range(len(groups))])

        # Assign the null group
        opt_dict["null"] = optax.adam(0.0)

        # Get optimiser object
        optim = optax.multi_transform(opt_dict, param_spec)

        # For some weird ass reason this works correctly but single liner 
        # doesn't
        if not get_filter_spec:
            return optim
        else:
            return optim, self.get_filter_spec(paths, path_dict=path_dict)

    
    #########################
    ### Numpyro functions ###
    #########################
    def update_and_model(self : Pytree, model_fn : str, paths : list_like, \
                         values : list_like, path_dict : dict = None, \
                         *args, **kwargs) -> object:
        """
        Updates the leaves speficied by paths with values, and then calls the
        function specified by the string model_fn, returning whatever is 
        returnd by the model_fn. Any extra positional arguments or key-word 
        arguments are passed through to the modelling function.
        
        This function is desigend to be used in conjunction with numpyro. 
        Please go through the 'Pytree interface' tutorial to see how this
        is used.
        
        Parameters
        ----------
        model_fn : str
            A string specifying which model function to call
        paths : list_like
            A list/tuple of nested paths. Note path objects can only be nested
            a single time.
        values : list_like
            A list/tuple of new values to be applied to the leaves specified by
            paths. These can not be nested.
        path_dict : dict (optional)
            A dictionary of absolute paths.

        Returns
        -------
        out : object 
            Whatever object is returned by model_fn
        """
        return getattr(
            self.update_leaves(paths, values, path_dict=path_dict), \
                model_fn)(*args, **kwargs)


class OpticalSystem(Base):
    """ Optical System class, Equinox Modle
    
    DOCSTRING NOT COMPLETE
    
    A Class to store and apply properties external to the optical system
    Ie: stellar positions and spectra
    
    positions: (Nstars, 2) array
    wavels: (Nwavels) array
    weights: (Nwavel)/(Nwavels, Nstars) array
    
    dLux currently does not check that inputs are correctly shaped/formatted!

    Notes:
     - Take in layers in order to re-intialise the model every call?
    
    General images output shape: (Nimages, Nstars, Nwavels, Npix, Npix)
    
     - Currently doesnt allow temporal variation in spectrum 
     - Currently doesnt allow temporal variation in flux
    
    ToDo: Add getter methods for accessing weights and fluxes attributes that
    use np.squeeze to remove empy axes

    
    Attributes
    ----------
    layers: list, required
        - A list of layers that defines the tranformaitons and operations of the system (typically optical)
     
    wavels: ndarray
        - An array of wavelengths in meters to simulate
        - The shape must be 1d - stellar spectrums are controlled through the weights parameter
        - No default value is set if not provided and this will throw an error if you try to call functions that depend on this parameter
        - It is left as optional so that functions that allow wavelength input can be called on objects without having to pre-input wavelengths
    positions: ndarray, optional
        - An array of (x,y) stellar positions in units of radians, measured as deviation of the optical axis. 
        - Its input shape should be (Nstars, 2), defining an x, y position for each star. 
        - If not provided, the value defaults to (0, 0) - on axis
    fluxes: ndarray, optional
        - An array of stellar fluxes, its length must match the positions inputs size to work properly
        - Theoretically this has arbitrary units, but we think of it as photons
        - Defaults to 1 (ie, returning a unitary flux psf if not specified)
    weights: ndarray, optional
        - An array of stellar spectral weights (arb units)
        - This can take multiple shapes
        - Default is to weight all wavelengths equally (top-hat)
        - If a 1d array is provided this is applied to all stars, shape (Nwavels)
        - if a 2d array is provided each is applied to each individual star, shape (Nstars, Nwavels)
        - Note the inputs values are always normalised and will not directly change total output flux (inderectly it can change it by weighting more flux to wavelengths with more aperture losses, for example)
    dithers: ndarray, optional
        - An arary of (x, y) positional dithers in units of radians
        - Its input shape should be (Nims, 2), defining the (x,y) dither for each image
        - if not provided, defualts to no-dither
    detector_layers: list, optional
        - A second list of layer objects designed to allow processing of psfs, rather than wavefronts
        - It is applied to each image after psfs have been approraitely weighted and summed
    
    
    """

    # Helpers, Determined from inputs, not real params
    Nstars:  int
    Nwavels: int
    Nims:    int
    
    wavels:          np.ndarray
    positions:       np.ndarray
    fluxes:          np.ndarray
    weights:         np.ndarray
    dithers:         np.ndarray
    layers:          list
    detector_layers: list
    
    # To Do - add asset conditions to ensure that everything is formatted correctly 
    # To Do - pass in positions for multiple images, ignoring dither (ie multi image)
    def __init__(self, layers, wavels, positions=None, fluxes=None, 
                       weights=None, dithers=None, detector_layers=None):
        
        # Required Inputs
        self.layers = layers
        self.wavels = np.array(wavels).astype(float)
        self.Nwavels = len(self.wavels)
        
        # Set to default values
        self.positions = np.zeros([1, 2]) if positions is None else np.array(positions)
        self.fluxes = np.ones(len(self.positions)) if fluxes is None else np.array(fluxes)
        self.weights = np.ones(len(self.wavels)) if weights is None else np.array(weights)
        self.dithers = np.zeros([1, 2]) if dithers is None else dithers
        self.detector_layers = [] if detector_layers is None else detector_layers
        
        if self.fluxes.shape == ():
            self.fluxes = np.array([self.fluxes])
        
        # Determined from inputs - treated as static
        self.Nstars =  len(self.positions)
        self.Nims =    len(self.dithers)
        
        # Check Input shapes
        assert self.positions.shape[-1] == 2, """Input positions must be 
        of shape (Nstars, 2)"""
        
        assert self.fluxes.shape[0] == self.Nstars, """Input fluxes must be
        match input positions."""
        
        weight_shape = self.weights.shape
        if len(weight_shape) == 1 and weights is not None:
            assert weight_shape[0] == self.Nwavels, """Inputs weights shape 
            must be either (len(wavels)) or  (len(positions), len(wavels)), 
            got shape: {}""".format(self.weights.shape)
        elif len(weight_shape) == 2:
            assert weight_shape == [self.Nstars, self.Nwavels], """Inputs 
            weights shape must be either (len(wavels)) or  (len(positions), 
            len(wavels))"""

    def debug_prop(self, wavel, offset=np.zeros(2)):        
        """
        I believe this is diffable but there is no reason to force it to be
        """
        params_dict = {"wavelength" : wavel, 
                       "offset" : offset,
                       "Optical System" : self}
        intermeds = []
        layers_applied = []
        for i in range(len(self.layers)):
            params_dict = self.layers[i](params_dict)
            intermeds.append(deepcopy(params_dict))
            layers_applied.append(self.layers[i].__str__())
        return params_dict["Wavefront"].wavefront_to_psf(), intermeds, layers_applied
            
        
        
    """################################"""
    ### DIFFERENTIABLE FUNCTIONS BELOW ###
    """################################"""
    
    
    
    def propagate(self):
        """
        Maps the wavelength and position calcualtions across multiple dimesions
        
        To Do: Reformat the vmaps such that we only vmap over wavelengths and
        positions in order to simplify the dimensionality
        """
        
        # Mapping over wavelengths
        propagate_single = vmap(self.propagate_mono, in_axes=(0, None))
        
        # Then over the positions 
        propagator = vmap(propagate_single, in_axes=(None, 0))

        # Generate input positions vector
        dithered_positions = self.dither_positions()
        
        # Calculate PSFs
        psfs = propagator(self.wavels, dithered_positions)
        
        # Reshape output into images
        psfs = self.reshape_psfs(psfs)
        
        # Weight PSFs and sum into images
        psfs = self.weight_psfs(psfs).sum([1, 2])
        
        # Vmap detector operations over each image
        detector_vmap = vmap(self.apply_detector_layers, in_axes=0)
        images = detector_vmap(psfs)
        
        return np.squeeze(images)
    
    def propagate_mono(self, wavel, offset=np.zeros(2)):        
        """
        
        """
        params_dict = {"wavelength" : wavel, 
                       "offset" : offset,
                       "Optical System" : self}
        
        for i in range(len(self.layers)):
            params_dict = self.layers[i](params_dict)
            
        return params_dict["Wavefront"].wavefront_to_psf()
    
    def propagate_single(self, wavels, offset=np.zeros(2), weights=1., flux=1.,
                         apply_detector=False):
        """
        Only propagates a single star, allowing wavelength input
        sums output to single array
        
        Wavels must be an array and the same shape as weights if provided
        """
        
        # Mapping over wavelengths
        prop_wf_map = vmap(self.propagate_mono, in_axes=(0, None))
        
        # Apply spectral weighting
        psfs = weights * prop_wf_map(wavels, offset)/len(wavels)
        
        # Sum into single psf and apply flux
        image = flux * psfs.sum(0)
        
        if apply_detector:
            image = self.apply_detector_layers(image)
        
        return image
    
    def apply_detector_layers(self, image):
        """
        
        """
        for i in range(len(self.detector_layers)):
            image = self.detector_layers[i](image)
        return image
    
    def reshape_psfs(self, psfs):
        """
        
        """
        npix = psfs.shape[-1]
        return psfs.reshape([self.Nims, self.Nstars, self.Nwavels, npix, npix])
    
    def dither_positions(self):
        """
        Dithers the input positions, returned with shape (Npsfs, 2)
        """
        Npsfs = self.Nstars * self.Nims
        shaped_pos = self.positions.reshape([1, self.Nstars, 2])
        shaped_dith = self.dithers.reshape([self.Nims, 1, 2])
        dithered_positions = (shaped_pos + shaped_dith).reshape([Npsfs, 2])
        return dithered_positions
    
    
    def weight_psfs(self, psfs):
        """
        Normalise Weights, and format weights/fluxes
        Psfs output shape: (Nims, Nstars, Nwavels, npix, npix)
        We want weights shape: (1, 1, Nwavels, 1, 1)
        We want fluxes shape: (1, Nstars, 1, 1, 1)
        """
        # Get values
        Nims = self.Nims
        Nstars = self.Nstars
        Nwavels = self.Nwavels
        
        # Format and normalise weights
        if len(self.weights.shape) == 3:
            weights_in = self.weights.reshape([Nims, Nstars, Nwavels, 1, 1])
            weights_in /= np.expand_dims(weights_in.sum(2), axis=2) 
        elif len(self.weights.shape) == 2:
            weights_in = self.weights.reshape([1, Nstars, Nwavels, 1, 1])
            weights_in /= np.expand_dims(weights_in.sum(2), axis=2) 
        elif self.weights.shape[0] == self.Nwavels:
            weights_in = self.weights.reshape([1, 1, Nwavels, 1, 1])
            weights_in /= np.expand_dims(weights_in.sum(2), axis=2) 
        else:
            weights_in = self.weights
        
        # Format Fluxes
        if len(self.fluxes) == 1:
            fluxes = self.fluxes
        else:
            fluxes = self.fluxes.reshape([1, Nstars, 1, 1, 1])
        
        
        # Apply weights and fluxus
        psfs *= weights_in
        psfs *= fluxes
        return psfs



"""
High level notes:

These classes are still in early development and are still subject to change
both at the internals and API. Ideally the API should be relatively stable,
but no guarantee can be made!

There should be some way to cache psf calculations during observations in order
to only calculte novel psfs for large observations.

TODO: Build the observation object, open questions here.

Q: Should the modelling_dict object store single dimension arrays, with
recombinations happening after via some indexing parameter?

Q: For inputs like wavelenght, should the input type be Array or float? Its a
single valued float 'array' of shape (0,)....
"""


class Instrument(Base):
    """
    A high level class desgined to model the behaviour of a telescope. It
    stores a series different ∂Lux objects, and primarily passes the relevant
    information between these objects in order to coherently model some
    telescope observation.
    
    Attributes
    ----------
    optics : Optics
        A dLux.base.Optics object that defines some optical configuration.
    sources : Scene
        A dLux.base.Scene that stores the various source objects that the 
        instrument is observing.
    detector : Detector
        A dLux.base.Detector object that is used to model the various 
        instrumental effects on a psf.
    filter : Filter
        A dLux.base.Filter object that is used to model the effective 
        throughput of each wavelength though the optical system.
    """
    optics   : Optics
    scene    : Scene
    detector : Detector
    filter   : Filter
    # Observation: Observation
    
    
    def __init__(self     : Instrument,
                 
                 # Class inputs
                 optics   : Optics    = None,
                 scene    : Scene     = None,
                 detector : Detector  = None,
                 filter   : Filter    = None,
                 
                 # List inputs
                 optical_layers  : list = [],
                 sources         : list = [],
                 detector_layers : list = [],
                 
                 # Observation :
                 ) -> Instrument:
        """
        Parameters
        ----------
        optics : Optics, (optional)
            A pre-configured dLux.base.Optics object. Can not be specified if
            optical layers in specified.
        optical_layers : list, (optional)
            A list of dLux optical layer classes that define the optical 
            transformations within some optical configuration. Can not be 
            specified if optics is specified.
        scene : Scene, (optional)
            A pre-configured dLux.base.Scene object. Can not be specified if
            sources is specified.
        sources : list, (optional)
            A list of dLux source objects that the telescope is observing.
        detector : Detector (optional)
            A pre-configured dLux.base.Detector object. Can not be specified if
            detector_layers is specified.
        detector_layers : list (optional)
            An list of dLux detector layer classes that define the instrumental 
            effects for some detector. Can not be specified if detector is 
            specified.
        filter : Filter (optional)
            A Filter object that is used to model the effective throughput of
            each wavelength though the Instrument.
        """ 
        # Optics            
        if optics is None and optical_layers == []:
            self.optics = Optics([])
        elif optics is not None and optical_layers != []:
            raise ValueError("Either optics OR optical_layers can be \
                                                        specified, not both.")
        elif optics is not None and optical_layers == []:
            assert isinstance(optics, Optics), "If optics is specified \
            it must a dLux.base.Optics object."
            self.optics = optics
        elif optics is None and optical_layers != []:
            assert isinstance(optical_layers, list), "If optical_layers is \
            specified it must be a list."
            self.optics = Optics(optical_layers)
        else:
            raise ValueError("How did you get here? Please raise a bug report \
                                                to help improve the software.")
        
        # Detector            
        if detector is None and detector_layers == []:
            self.detector = Detector([])
        elif detector is not None and detector_layers != []:
            raise ValueError("Either detector OR detector_layers can be \
                                                        specified, not both.")
        elif detector is not None and detector_layers == []:
            assert isinstance(detector, Detector), "If detector is specified \
            it must a dLux.base.Detector object."
            self.detector = detector
        elif detector is None and detector_layers != []:
            assert isinstance(detector_layers, list), "If detector_layers is \
            specified it must be a list."
            self.detector = Detector(detector_layers)
        else:
            raise ValueError("How did you get here? Please raise a bug report \
                                                to help improve the software.")
        
        # Scene            
        if scene is None and sources == []:
            self.scene = Scene([])
        elif scene is not None and sources != []:
            raise ValueError("Either scene OR sources can be \
                                                        specified, not both.")
        elif scene is not None and sources == []:
            assert isinstance(scene, Scene), "If scene is specified it must a \
            dLux.base.Scene object."
            self.scene = scene
        elif scene is None and sources != []:
            assert isinstance(sources, list), "If sources is specified it \
            must be a list."
            self.scene = Scene(sources)
        else:
            raise ValueError("How did you get here? Please raise a bug report \
                                                to help improve the software.")
        
        # Filter
        if filter is None:
            self.filter = Filter() 
        elif isinstance(filter, Filter):
            self.filter = filter
        else:
            raise ValueError("filter input must be None or a Filter object")
    
    
    def normalise(self : Instrument) -> Instrument:
        """
        Normalises the internally stored scene.
        
        Returns
        -------
        instrument : Instrument
            A new version of the instrument with the interally stored scene
            normalised.
        """
        return eqx.tree_at(lambda instrument: instrument.scene, self, \
                           self.scene.normalise())
    
    
    def model_source(self      : Instrument, 
                     source    : dLux.sources.Source,
                     optics    : Optics   = None,
                     detector  : Detector = None,
                     filter_in : Filter   = None) -> Array:
        """
        Models the source through the optics.
        
        Parameters
        ----------
        source : dict
            The source to observe
        optics : Optics
            The optics through which to model the source object.
        detector : Detector (optional)
            The detector object that is observing the psf.
        filter_in : Filter (optional)
            The filter through which the source is being observed. Default is 
            None which is uniform throughput.
        
        Returns
        -------
        psf : Array
            The summed psfs of the sources modelled through the telescope.
        """
        optics = self.optics if optics is None else optics
        source = source.normalise()
        return source.model(optics, detector=detector, filter_in=filter_in)
    
    
    def model_sources(self      : Optics,
                      sources   : dict,
                      optics    : Optics   = None,
                      detector  : Detector = None,
                      filter_in : Filter   = None) -> Array:
        """
        Models the sources through the optics.
        
        Parameters
        ----------
        sources : dict
            The sources to observe
        optics : Optics
            The optics through which to model the sources.
        detector : Detector (optional)
            The detector object that is observing the psf.
        filter_in : Filter (optional)
            The filter through which the source is being observed. Default is 
            None which is uniform throughput.
        
        Returns
        -------
        psf : Array
            The summed psfs of the sources modelled through the intrument.
        """
        # Apply optional inputs (check this work properly)
        model_fn = lambda source: self.model_source(source, 
                                                        filter_in=filter_in)
        
        # Map the model_source function across the sources
        psf_tree = jtu.tree_map(model_fn, sources, \
                        is_leaf = lambda leaf: isinstance(leaf, dLux.Source))
        
        # Get psfs
        psf = np.array(jtu.tree_flatten(psf_tree)[0]).sum(0)
        
        # apply detector
        if detector is None:
            return psf
        else:
            return detector.apply_detector(psf)

    def model(self      : Instrument, 
              optics    : Optics   = None,
              scene     : Scene    = None, 
              detector  : Detector = None,
              filter_in : Filter   = None,
              flatten   : bool     = False) -> Array:
        """
        Models the sources through the instrument optics and detector.
        
        Parameters
        ----------
        optics : Optics (optional)
            The optics through which to model the source objects, defaults to 
            using the internally stored optics if no value is passed.
        scene : Scene (optional)
            The scene to observe, defaults to using the internally stored scene
            if no value is passed.
        detector : Detector (optional)
            The detector to use with the observation, defaults to using the
            internally stored detector if no value is passed.
        filter_in : Filter (optional)
            The filter through which the source is being observed, defaults to 
            using the internally stored detfilter ector if no value is passed.
        flatten : bool (optional)
            Whether the output image should be flattened
        
        Returns
        -------
        image : Array
            The image of the scene modelled through the telescope with detector
            effects applied.
        """
        # Format inputs
        optics   = self.optics   if optics   is None else optics
        scene    = self.scene    if scene    is None else scene
        detector = self.detector if detector is None else detector
        
        # Apply optional inputs (check this work properly)
        model_fn = lambda source: self.model_source(source, 
                                                    optics=optics,
                                                    filter_in=filter_in)
        
        # Map the model_source function across the sources
        psf_tree = jtu.tree_map(model_fn, scene.sources, \
                        is_leaf = lambda leaf: isinstance(leaf, dLux.Source))
        
        # Get psfs
        psf = np.array(jtu.tree_flatten(psf_tree)[0]).sum(0)
        
        # Apply detector
        image = detector.apply_detector(psf)
        
        # Possibly flatten
        return image.flatten() if flatten else image 
    
    
class Optics(Base):
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
        Parameters
        ----------
        layers : list
            A list of ∂Lux 'layers' that define the transformations and
            operations upon some input wavefront through an optical system.
        """
        self.layers = dLux.utils.list_to_dict(layers)
    

    def propagate_mono(self       : Optics,
                       wavelength : Array,
                       offset     : Array = np.zeros(2),
                       weight     : Array = np.ones(1)) -> Array:
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
        weight : Array, (optional)
            The relative weighting of the wavelength. Simply scales the output
            psf.
        
        Returns
        -------
        psf : Array
            The monochromatic point spread function after being propagated
            though the optical layers.
        """
        # Construct parameters dictionary
        params_dict = {"optics"     : self,
                       "wavelength" : wavelength,
                       "offset"     : offset}
        
        # Propagate though layers
        for key, layer in self.layers.items():
            params_dict = layer(params_dict)
        psf = params_dict["Wavefront"].wavefront_to_psf()
        return weight * psf
    
    
    def propagate_multi(self        : Optics,
                        wavelengths : Array,
                        offset      : Array = np.zeros(2),
                        weights     : Array = np.ones(1)) -> Array:
        """
        Propagates a broadband point source through the optical layers.
        
        Parameters
        ----------
        wavelengths : Array, meters
            The wavelengths of the wavefront to propagate through the optical
            layers.
        offset : Array, radians, (optional)
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.
        weights : Array, (optional)
            The relative weighting of the wavelengths. Simply scales the output
            psf.
        
        Returns
        -------
        psf : Array
            The broadband point spread function after being propagated
            though the optical layers.
        """
        propagator = jax.vmap(self.propagate_mono, in_axes=(0, None, 0, None))
        psfs = propagator(wavelengths, offset, weights)
        return psfs.sum(0)
    
    
    def debug_prop(self       : Optics,
                   wavelength : Array,
                   offset     : Array = np.zeros(2)) -> Array:
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
    
    
    def model_source(self      : Optics, 
                     source    : dLux.Source, 
                     detector  : Detector = None,
                     filter_in : Filter   = None) -> Array:
        """
        Models the source through the optics.
        
        Parameters
        ----------
        sources : dict
            The sources to observe, defaults to using the internally stored 
            sources if no value is passed.
        detector : Detector (optional)
            The detector object that is observing the psf.
        filter_in : Filter (optional)
            The filter through which the source is being observed. Default is 
            None which is uniform throughput.
        
        Returns
        -------
        psf : Array
            The summed psfs of the sources modelled through the telescope.
        """
        source = source.normalise()
        return source.model(self, detector=detector, filter_in=filter_in)
    
    
    def model_sources(self      : Optics,
                      sources   : dict,
                      detector  : Detector = None,
                      filter_in : Filter   = None) -> Array:
        """
        Models the sources through the optics.
        
        Parameters
        ----------
        sources : dict
            The sources to observe, defaults to using the internally stored 
            sources if no value is passed.
        detector : Detector (optional)
            The detector object that is observing the psf.
        filter_in : Filter (optional)
            The filter through which the source is being observed. Default is 
            None which is uniform throughput.
        
        Returns
        -------
        psf : Array
            The summed psfs of the sources modelled through the telescope.
        """
        # Apply optional inputs (check this work properly)
        model_fn = lambda source: self.model_source(source, 
                                                        filter_in=filter_in)
        
        # Map the model_source function across the sources
        psf_tree = jtu.tree_map(model_fn, sources, \
                        is_leaf = lambda leaf: isinstance(leaf, dLux.Source))
        
        # Get psfs
        psf = np.array(jtu.tree_flatten(psf_tree)[0]).sum(0)
        
        # apply detector
        if detector is None:
            return psf
        else:
            return detector.apply_detector(psf)
    
    
class Scene(Base):
    """
    A high level class representing some 'astrophysical scene', which is
    composed of Sources.
    
    Attributes
    ----------
    sources : dict
        A dictionary containing all the of sources that comprise the
        astrophysical scene.
    """
    sources: dict
    
    
    def __init__(self, sources : list) -> Scene:
        """
        Parameters
        ----------
        sources : list
            a list of individual source objects that is automatically converted
            into a dictionary
        """
        self.sources = dLux.utils.list_to_dict(sources, ordered=False)
        
    
    def normalise(self : Scene) -> Scene:
        """
        Normalises the internally stores sources of the scene.
        
        Returns
        -------
        scene : Scene
            A new version of the scene with the interally stored sources
            normalised.
        """
        normalised_sources = self.normalise_sources(self.sources)
        return eqx.tree_at(lambda scene: scene.sources, self, \
                                                    normalised_sources)
    
    
    # Pretty sure this should be moved to utils
    def normalise_sources(self    : Scene, 
                          sources : dict) -> dict:
        """
        Normalises a dictionary of source objects.
        
        Parameters
        ----------
        sources : dict
            The sources to normalise.
            
        Returns
        -------
        sources : dict
            The normalised sources.
        """
        normalise_fn = lambda source: source.normalise()
        
        # Map the model_source function across the sources
        return jtu.tree_map(normalise_fn, sources, \
                        is_leaf = lambda leaf: isinstance(leaf, dLux.Source))
    
        
    
    def model_optics(self      : Scene, 
                     optics    : Optics,
                     detector  : Detector = None,
                     filter_in : Filter   = None) -> Array:
        """
        Models the scene through some optics.
        
        Parameters
        ----------
        optics : Optics
            The optics through which to model the source objects.
        detector : Detector (optional)
            The detector object that is observing the psf.
        filter_in : Filter (optional)
            The filter through which the source is being observed. Default is 
            None which is uniform throughput.
        
        Returns
        -------
        psfs : Array
            The summed psfs of the sources modelled through the telescope.
        """
        # Apply optional inputs (check this work properly)
        model_fn = lambda source: optics.model_source(source, 
                                                        filter_in=filter_in)
        
        # Map the model_source function across the sources
        psf_tree = jtu.tree_map(model_fn, sources, \
                        is_leaf = lambda leaf: isinstance(leaf, dLux.Source))
        
        
        # Get psf
        psf = np.array(jtu.tree_flatten(psf_tree)[0]).sum(0)
        
        # apply detector
        if detector is None:
            return psf
        else:
            return detector.apply_detector(psf)
    
    
class Filter(Base):
    """
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
    wavelengths : Array
    throughput  : Array
    filter_name : str = eqx.static_field()
    
    
    def __init__(self        : Filter,
                 wavelengths : Array = None,
                 throughput  : Array = None,
                 filter_name : str   = None) -> Filter:
        """
        Initialises the filter. All inputs are optional and defaults to uniform
        unitary throughput. If filter_name is specified then wavelengths and
        weights must not be specified.
        
        Parameters
        ----------
        wavelengths : Array (optional)
            The wavelengths at which the filter is defined.
        throughput : Array (optional)
            The throughput of the filter at the corresponding wavelength.
        filter_name : str (optional)
            A string identifier that can be used to initialise specific filters.
            Currently no pre-built filters are implemented.
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
            self.filter_name = 'Unitary'
                
        # Check that both wavelengths and throughput are specified
        elif (wavelengths is     None and throughput is not None) or \
             (wavelengths is not None and throughput is     None):
            raise ValueError("If either wavelengths or throughput is\
            specified, then both must be specified")
                
        # Both wavelengths and throughput are specified
        else:
            assert len(wavelengths) == len(throughput), "wavelengths and \
            throughput must have the same dimension"
            self.wavelengths = np.asarray(wavelengths, dtype=float)
            self.throughput  = np.asarray(throughput,  dtype=float)
            self.filter_name = 'custom'
            
            # Check bounds
            nlarge = np.sum(self.throughput > 1)
            nsmall = np.sum(self.throughput < 0)
            if nlarge + nsmall >= 1:
                warnings.warn("Filter throughputs should be between 0-1, {} \
                throughputs are above 1 or below 0.".format(nlarge + nsmall))
            
    # Cache this? It will likely be called many times with the same inputs
    # TODO: Make an integrated, rather than interpolated filter
    def get_throughput(self : Filter, wavelengths : Array) -> Array:
        """
        Get the correspondning throughput for the filter at the specified
        wavelengths. Currently uses a linear interpolation method, but is
        planned to use an integration method in the future. Any wavelengths
        outside of the defined wavelength range are taken as zero (except for
        'Unitary' throughput which is uniform)
        
        Parameters
        ----------
        wavelengths: Array
            An array of wavelengths to sample the filter at.
        
        Returns:
        throughputs : Array
            An array of the corresponding throughputs at the given wavlengths.
        """
        # Translate input wavelengths to indexes 
        min_wavelength = self.wavelengths.min()
        max_wavelength = self.wavelengths.max()
        wavelength_range = max_wavelength - min_wavelength
        num_wavelength = self.wavelengths.shape[0]
        indxs = num_wavelength*(wavelengths - min_wavelength)/wavelength_range
        throughputs = jax.scipy.ndimage.map_coordinates(self.throughput, \
                                        np.array([indxs]), 1, 'nearest')
        return throughputs


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
        Parameters
        ----------
        layers : list
            An list of dLux detector layer classes that define the instrumental 
            effects for some detector.
        """
        self.layers = dLux.utils.list_to_dict(layers)
    
    
    def apply_detector(self  : Instrument, 
                       image : Array) -> Array:
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
        intermediate_images = []
        intermediate_layers = []
        
        # Apply detector layers
        for key, layer in self.layers.items():
            image = layer(image)
            intermediate_images.append(image.copy())
            intermediate_layers.append(deepcopy(layer))
        return image, intermediate_images, intermediate_layers