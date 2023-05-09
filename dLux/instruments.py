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


__all__ = ["BaseInstrument", "Instrument"]


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
Observation       = lambda : dLux.observations.AbstractObservation


#####################
### Other Classes ###
#####################
class BaseInstrument(Base):
    
    @abstractmethod
    def normalise(self): # pragma: no cover
        pass

    @abstractmethod
    def model(self, optics, detector=None): # pragma: no cover
        pass


class Instrument(BaseInstrument):
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
        if not isinstance(optics, BaseOptics):
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
        if hasattr(self.optics, key):
            return getattr(self.optics, key)
        # if key in self.optics.layers.keys():
            # return self.optics.layers[key]
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
    

    def summarise(self : Instrument) -> None:  # pragma: no cover
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
             offset     : Array = np.zeros(2)) -> None:  # pragma: no cover
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
        return self._model(self.optics, self.sources, self.detector, 
            normalise_sources, flatten, return_tree)
    

    def _model(
        self        : Instrument,
        optics      : Optics,
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