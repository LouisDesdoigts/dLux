from __future__ import annotations
import jax.numpy as np
from jax import vmap
from jax.tree_util import tree_map, tree_flatten
from jax.scipy.ndimage import map_coordinates
from equinox import tree_at, static_field
from collections import OrderedDict
from copy import deepcopy
from functools import partial
import dLux


__all__ = ["model", "OpticalSystem", "Instrument", "Optics", "Scene",
           "Filter", "Detector"]


Array = np.ndarray


###############
### Methods ###
###############
def model(optics      : Optics,
          detector    : Detector            = None,
          filter      : Filter              = None,
          scene       : Scene               = None,
          sources     : Union[dict, list]   = None,
          source      : dLux.sources.Source = None,
          normalise   : bool                = True,
          flatten     : bool                = False,
          return_tree : bool                = False) -> Array:
    """
    A base level modelling function designed to robustly handle the different
    combinations of inputs. Models the sources through the instrument optics
    and detector. Users must provide optics and some form of source, either via
    a scene, sources or single source input, but not multiple.

    Parameters
    ----------
    optics : Optics
        The optics through which to model the source objects.
    detector : Detector = None
        The detector to use with the observation.
    filter : Filter = None
        The filter through which the source is being observed.
    scene : Scene = None
        The scene to observe.
    sources : Union[dict, list) = None
        The sources to observe.
    source : dLux.sources.Source = None
        The source to observe.
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
    assert isinstance(optics, (Optics)), ("optics must be an Optics object.")

    # Check that detector input is a Detector object if specified.
    assert isinstance(detector, (Detector, type(None))), \
    ("detector must be a Detector object.")

    # Check that filter input is a Filter object if specified.
    assert isinstance(filter, (Filter, type(None))), \
    ("filter must be a Filter object.")

    # Make sure that some form of source is speficied
    assert scene is not None or sources is not None or source is not None, \
    ("Either a scene, source, or sources must be specified")

    # Make sure that input types are correct
    # scene is specified
    if scene is not None:
        # Check for other inputs
        assert sources is None and source is None, \
        ("If scene is specified, sources and source can not be specified.")

        # Check that scene is a Scehen object.
        assert isinstance(scene, Scene), \
        ("scene must be a Scene object.")

        # Get sources
        sources_in = scene.normalise().sources if normalise \
                                                       else scene.sources

    # Check sources as next input
    elif sources is not None:
        # Check for other inputs
        assert source is None, \
        ("If sources is specified, scene and source can not be specified.")

        # Check that sources is a dict object.
        assert isinstance(sources, (dict, list, tuple)), \
        ("sources must be a dict, list, or tuple object.")

        # Check that all inputs are dLux.sources.Source objects
        source_vals = sources.values() if isinstance(sources, dict) else sources
        for source in source_vals:
            assert isinstance(source, dLux.sources.Source), \
            ("All entries within sources must be a dLux.sources.Source object.")

        # Get sources
        if normalise:
            # Define the normalisation function
            normalise_fn = lambda source: source.normalise()

            # Map the normalisation function across the sources
            sources_in = tree_map(normalise_fn, sources, \
                   is_leaf = lambda leaf: isinstance(leaf, dLux.sources.Source))
        else:
            sources_in = sources

    # source is provided
    else:
        assert isinstance(source, dLux.sources.Source), \
        ("source must be a dLux.sources.Source object.")

        # Get sources
        sources_in = source.normalise() if normalise else source

    '''Begin modelling'''
    # Apply optional inputs
    model_fn = lambda source: source.model(optics, filter_in=filter)

    # Map the model_source function across the sources
    psf_tree = tree_map(model_fn, sources_in, 
            is_leaf = lambda leaf: isinstance(leaf, dLux.sources.Source))

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


class Instrument(dLux.base.ExtendedBase):
    """
    A high level class desgined to model the behaviour of a telescope. It
    stores a series different ∂Lux objects, and primarily passes the relevant
    information between these objects in order to coherently model some
    telescope observation.

    Attributes
    ----------
    optics : Optics
        A Optics object that defines some optical configuration.
    sources : Scene
        A Scene that stores the various source objects that the
        instrument is observing.
    detector : Detector
        A Detector object that is used to model the various
        instrumental effects on a psf.
    filter : Filter
        A Filter object that is used to model the effective
        throughput of each wavelength though the optical system.
    observation : dict
        A dictionary that must have the keys 'fn' and 'args'. The 'fn' key must
        refer to a function that takes in the argument stored in 'args'. This
        is to allow flexibility in the different kind of observations, ie
        applying dithers, switching filters, etc.
    """
    optics      : Optics
    scene       : Scene
    detector    : Detector
    filter      : Filter
    observation : dict


    def __init__(self : Instrument,

                 # Class inputs
                 optics   : Optics    = None,
                 scene    : Scene     = None,
                 detector : Detector  = None,
                 filter   : Filter    = None,

                 # List inputs
                 optical_layers  : list = None,
                 sources         : list = None,
                 detector_layers : list = None,

                 # Obervation
                 observation : dict = None,
                 ) -> Instrument:
        """
        Constructor for the Instrument class.

        Parameters
        ----------
        optics : Optics, = None
            A pre-configured Optics object. Can not be specified if
            optical layers in specified.
        optical_layers : list, = None
            A list of dLux optical layer classes that define the optical
            transformations within some optical configuration. Can not be
            specified if optics is specified.
        scene : Scene, = None
            A pre-configured Scene object. Can not be specified if
            sources is specified.
        sources : list, = None
            A list of dLux source objects that the telescope is observing.
        detector : Detector = None
            A pre-configured Detector object. Can not be specified if
            detector_layers is specified.
        detector_layers : list = None
            An list of dLux detector layer classes that define the instrumental
            effects for some detector. Can not be specified if detector is
            specified.
        filter : Filter = None
            A Filter object that is used to model the effective throughput of
            each wavelength though the Instrument.
        observation : dict = None
            A dictionary that must have the keys 'fn' and 'args'. The 'fn' key
            must refer to a function that takes in the argument stored in
            'args'. This is to allow flexibility in the different kind of
            observations, ie applying dithers, switching filters, etc.
        """
        # Optics
        if optics is None and optical_layers is None:
            self.optics = None
        elif optics is not None and optical_layers is not None:
            raise ValueError("Either optics OR optical_layers can be "
            "specified, not both.")
        elif optics is not None and optical_layers is None:
            assert isinstance(optics, Optics), "If optics is specified "
            "it must a Optics object."
            self.optics = optics
        elif optics is None and optical_layers is not None:
            assert isinstance(optical_layers, list), "If optical_layers is "
            "specified it must be a list."
            self.optics = Optics(optical_layers)
        else:
            raise ValueError("How did you get here? Please raise a bug report "
            "to help improve the software.")

        # Detector
        if detector is None and detector_layers is None:
            self.detector = None
        elif detector is not None and detector_layers is not None:
            raise ValueError("Either detector OR detector_layers can be "
            "specified, not both.")
        elif detector is not None and detector_layers is None:
            assert isinstance(detector, Detector), "If detector is specified "
            "it must a Detector object."
            self.detector = detector
        elif detector is None and detector_layers is not None:
            assert isinstance(detector_layers, list), "If detector_layers is "
            "specified it must be a list."
            self.detector = Detector(detector_layers)
        else:
            raise ValueError("How did you get here? Please raise a bug report "
            "to help improve the software.")

        # Scene
        if scene is None and sources is None:
            self.scene = None
        elif scene is not None and sources is not None:
            raise ValueError("Either scene OR sources can be "
            "specified, not both.")
        elif scene is not None and sources is None:
            assert isinstance(scene, Scene), "If scene is specified it must a "
            "Scene object."
            self.scene = scene
        elif scene is None and sources is not None:
            assert isinstance(sources, list), "If sources is specified it "
            "must be a list."
            self.scene = Scene(sources)
        else:
            raise ValueError("How did you get here? Please raise a bug report "
            "to help improve the software.")

        # Filter
        if filter is None:
            self.filter = None
        else:
            assert isinstance(filter, Filter), \
            ("filter must be a Filter object.")
            self.filter = filter

        if observation is not None:
            assert isinstance(observation, dict) and \
            'fn' in observation.keys() and 'args' in observation.keys(), \
            ("observation must be a dictionary with the keys 'fn' and 'args'.")
        self.observation = observation


    def observe(self : Instrument, **kwargs) -> Any:
        """
        Call the function stored within the observation attribute at key 'fn',
        passing in the instrument (self) as the first argument and 'args' as
        the second.

        Returns
        -------
         : Any
            The output of the function store in the observation at 'fn' with
            arguments (instrument, observation['args']).
        """
        return self.observation['fn'](self, self.observation['args'], **kwargs)


    def dither_position(self : Instrument, dither : Array) -> Instrument:
        """
        Dithers the position of the source objects by dither.

        Parameters
        ----------
        dither : Array, radians
            The (x, y) dither to apply to the source positions.

        Returns
        -------
        instrument : Instrument
            The instrument with the sources dithered.
        """
        assert dither.shape == (2,), ("dither must have shape (2,) ie (x, y)")
        # Define the dither function
        dither_fn = lambda source: source.add('position', dither)

        # Map the dithers across the sources
        dithered_sources = tree_map(dither_fn, self.scene.sources, \
                is_leaf = lambda leaf: isinstance(leaf, dLux.sources.Source))

        # Apply updates
        return tree_at(lambda instrument: instrument.scene.sources, self, \
                       dithered_sources)


    def dither_and_model(self    : Instrument,
                         dithers : Array,
                         **kwargs) -> Any:
        """
        Applies a series of dithers to the instrument sources and calls the
        .model() method after applying each dither.

        Parameters
        ----------
        dithers : Array, radians
            The array of dithers to apply to the source positions.

        Returns
        -------
        psfs : Array
            The psfs generated after applying the dithers to the source
            positions.
        """
        dith_fn = lambda dither: self.dither_position(dither).model(**kwargs)
        return vmap(dith_fn, in_axes=(0))(dithers)



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

        if self.optics is not None and hasattr(self.optics, 'layers') and \
                                        key in self.optics.layers.keys():
            return self.optics.layers[key]
        elif self.detector is not None and hasattr(self.detector, 'layers') and\
                                        key in self.detector.layers.keys():
            return self.detector.layers[key]
        elif self.scene is not None and hasattr(self.scene, 'sources') and \
                                        key in self.scene.sources.keys():
            return self.scene.sources[key]
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
        return tree_at(lambda instrument: instrument.scene, self, \
                           self.scene.normalise())


    def model(self : Instrument, **kwargs):
        """
        A base level modelling function designed to robustly handle the
        different combinations of inputs. Models the sources through the
        instrument optics and detector. Users must provide optics and some form
        of source, either via a scene, sources or single source input, but not
        multiple.

        Parameters
        ----------
        optics : Optics = None
            The optics through which to model the source objects. Defaults to
            the internally stored value.
        detector : Detector = None
            The detector to use with the observation. Defaults to the
            internally stored value.
        filter : Filter = None
            The filter through which the source is being observed. Defaults to
            the internally stored value.
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
        optics = optics if 'optics' in kwargs else self.optics
        kwargs['detector'] = self.detector if 'detector' not in kwargs \
                                                        else kwargs['detector']
        kwargs['filter']   = self.filter   if 'filter'   not in kwargs \
                                                        else kwargs['filter']
        if 'scene' not in kwargs and \
           'source' not in kwargs and \
           'sources' not in kwargs:
            kwargs['scene'] = self.scene

        return model(optics,
                     **kwargs)


class Optics(dLux.base.ExtendedBase):
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
        assert isinstance(layers, list), ("Input layers must be a list, it is" \
        " automatically converted to a dictionary")

        # Ensure all entries are dLux layers
        for layer in layers:
            assert isinstance(layer, dLux.optics.OpticalLayer), ("All entries" \
            " within layers must be an dLux.optics.OpticalLayer object")

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
                       weight     : Array = np.array(1.)) -> Array:
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
        weight : Array, = np.array(1.)
            The relative weighting of the wavelength. Simply scales the output
            psf.

        Returns
        -------
        psf : Array
            The monochromatic point spread function after being propagated
            though the optical layers.
        """
        # Ensure jax arrays
        wavelength = np.asarray(wavelength, dtype=float) \
                    if not isinstance(wavelength, np.ndarray) else wavelength
        offset = np.asarray(offset, dtype=float) \
                    if not isinstance(offset, np.ndarray) else offset
        weight = np.asarray(weight, dtype=float) \
                    if not isinstance(weight, np.ndarray) else weight

        # Ensure dimensionality
        assert wavelength.shape == (), "wavelength must be a scalar."
        assert offset.shape == (2,), "offset must be shape (2,), ie (x, y)."
        assert weight.shape == (), "weight must be a scalar."

        # Construct parameters dictionary
        params_dict = {"Wavefront"  : None,
                       "optics"     : self,
                       "wavelength" : wavelength,
                       "offset"     : offset}

        # Propagate though layers
        for key, layer in self.layers.items():
            params_dict = layer.apply(params_dict)
        psf = params_dict["Wavefront"].wavefront_to_psf()
        return weight * psf


    def propagate_multi(self        : Optics,
                        wavelengths : Array,
                        offset      : Array = np.zeros(2),
                        weights     : Array = None) -> Array:
        """
        Propagates a broadband point source through the optical layers.

        Parameters
        ----------
        wavelengths : Array, meters
            The wavelengths of the wavefront to propagate through the optical
            layers.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.
        weights : Array = None
            The relative weighting of the wavelengths. Simply scales the output
            psf.

        Returns
        -------
        psf : Array
            The broadband point spread function after being propagated
            though the optical layers.
        """
        # Format weights input
        wavelengths = np.asarray(wavelengths, dtype=float) \
                  if not isinstance(wavelengths, np.ndarray) else wavelengths
        assert wavelengths.ndim == 1, "wavelengths must be 1 dimensional.."

        # Format weights input
        if weights is None:
            weights = np.ones(len(wavelengths))
        elif not isinstance(weights, np.ndarray):
            weights = np.asarray(weights, dtype=float)
        assert weights.ndim == 1, "weights must be 1 dimensional."

        # Ensure matching dimensionality
        assert wavelengths.shape == weights.shape, \
        ("wavelengths and weights must have the same shape.")

        # Offset checking
        offset = np.asarray(offset, dtype=float) \
                 if not isinstance(offset, np.ndarray) else offset
        assert offset.shape == (2,), "offset must be shape (2,), ie (x, y)."

        # Propagate
        propagator = vmap(self.propagate_mono, in_axes=(0, None, 0))
        psfs = propagator(wavelengths, offset, weights)
        return psfs.sum(0)


    def debug_prop(self       : Optics,
                   wavelength : Array,
                   offset     : Array = np.zeros(2),
                   weight     : Array = np.array(1.)) -> Array:
        """
        Propagates a monochromatic point source through the optical layers,
        while also returning the intermediate state of the parameter dictionary
        and layers after each layer application.

        Parameters
        ----------
        wavelength : Array, meters
            The wavelength of the wavefront to propagate through the optical
            layers.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.
        weight : Array, = np.array(1.)
            The relative weighting of the wavelength. Simply scales the output
            psf.

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
        # Ensure jax arrays
        wavelength = np.asarray(wavelength, dtype=float) \
                    if not isinstance(wavelength, np.ndarray) else wavelength
        offset = np.asarray(offset, dtype=float) \
                    if not isinstance(offset, np.ndarray) else offset
        weight = np.asarray(weight, dtype=float) \
                    if not isinstance(weight, np.ndarray) else weight

        # Ensure dimensionality
        assert wavelength.shape == (), "wavelength must be a scalar."
        assert offset.shape == (2,), "offset must be shape (2,), ie (x, y)."
        assert weight.shape == (), "weight must be a scalar."

        # Construct parameters dictionary
        params_dict = {"Wavefront"  : None,
                       "optics"     : self,
                       "wavelength" : wavelength,
                       "offset"     : offset}

        intermediate_dicts = []
        intermediate_layers = []
        for key, layer in self.layers.items():
            params_dict = layer.apply(params_dict)
            intermediate_dicts.append(params_dict.copy())
            intermediate_layers.append(deepcopy(layer))

        return params_dict["Wavefront"].wavefront_to_psf(), \
                                intermediate_dicts, intermediate_layers


    def model(self : Optics, **kwargs):
        """
        A base level modelling function designed to robustly handle the
        different combinations of inputs. Models the sources through the
        instrument optics and detector. Users must provide optics and some form
        of source, either via a scene, sources or single source input, but not
        multiple.

        Parameters
        ----------
        detector : Detector = None
            The detector to use with the observation. Defaults to the
            internally stored value.
        filter : Filter = None
            The filter through which the source is being observed. Defaults to
            the internally stored value.
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
        return model(self, **kwargs)


class Scene(dLux.base.ExtendedBase):
    """
    A high level class representing some 'astrophysical scene', which is
    composed of dLux.sources.Sources.

    Attributes
    ----------
    sources : dict
        A dictionary containing all the of sources that comprise the
        astrophysical scene.
    """
    sources: dict


    def __init__(self, sources : list) -> Scene:
        """
        Constructor for the Scene class.

        Parameters
        ----------
        sources : list
            a list of individual source objects that is automatically converted
            into a dictionary
        """
        assert isinstance(sources, list), ("Input sources must be a list, it" \
        " is automatically converted to a dictionary.")

        # Ensure all entries are dLux dLux.sources.Sources
        for source in sources:
            assert isinstance(source, dLux.sources.Source), ("All entries " \
            "within sources must be a dLux.sources.Source object.")

        self.sources = dLux.utils.list_to_dictionary(sources, ordered=False)


    def __getattr__(self : Scene, key : str) -> object:
        """
        Magic method designed to allow accessing of the various items within
        the sources dictionary of this class via the 'class.attribute' method.

        Parameters
        ----------
        key : str
            The key of the item to be searched for in the sources dictionary.

        Returns
        -------
        item : object
            The item corresponding to the supplied key in the sources
            dictionary.
        """
        if key in self.sources.keys():
            return self.sources[key]
        else:
            raise AttributeError("'{}' object has no attribute '{}'"\
                                 .format(type(self), key))


    def normalise(self : Scene) -> Scene:
        """
        Normalises the internally stores sources of the scene.

        Returns
        -------
        scene : Scene
            A new version of the scene with the interally stored sources
            normalised.
        """
        # Define the normalisation function
        normalise_fn = lambda source: source.normalise()

        # Map the model_source function across the sources
        normalised_sources = tree_map(normalise_fn, self.sources, \
                is_leaf = lambda leaf: isinstance(leaf, dLux.sources.Source))

        return tree_at(lambda scene: scene.sources, self, normalised_sources)


    def model(self : Scene, optics : Optics, **kwargs):
        """
        A base level modelling function designed to robustly handle the
        different combinations of inputs. Models the sources through the
        instrument optics and detector. Users must provide optics and some form
        of source, either via a scene, sources or single source input, but not
        multiple.

        Parameters
        ----------
        optics : Optics
            The optics through which to model the source objects. Defaults to
            the internally stored value.
        detector : Detector = None
            The detector to use with the observation. Defaults to the
            internally stored value.
        filter : Filter = None
            The filter through which the source is being observed. Defaults to
            the internally stored value.
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
        return model(optics, scene=self, **kwargs)


class Detector(dLux.base.ExtendedBase):
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


    def model(self : Detector, optics : Optics, **kwargs):
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
        filter : Filter = None
            The filter through which the source is being observed. Defaults to
            the internally stored value.
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
        return model(optics, detector=self, **kwargs)


class Filter(dLux.base.ExtendedBase):
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
    wavelengths  : Array
    throughput   : Array
    filter_name  : str = static_field()


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



"""Legacy code"""
class OpticalSystem(dLux.base.ExtendedBase):
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
        params_dict = {"Wavefront" : None,
                       "wavelength" : wavel, 
                       "offset" : offset,
                       "Optical System" : self}
        
        for i in range(len(self.layers)):
            # params_dict = self.layers[i](params_dict)
            params_dict = self.layers[i].apply(params_dict)
            
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
