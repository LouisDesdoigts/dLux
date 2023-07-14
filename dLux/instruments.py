from __future__ import annotations
from typing import Any
from abc import abstractmethod
import jax.numpy as np
from jax import Array
from jax.tree_util import tree_map, tree_flatten
from zodiax import Base
from typing import Union
import dLux.utils as dlu
import dLux

__all__ = ["Instrument"]

# Alias classes for simplified type-checking
Optics = lambda: dLux.optics.BaseOptics
Detector = lambda: dLux.detectors.BaseDetector
Source = lambda: dLux.sources.BaseSource
Observation = lambda: dLux.observations.BaseObservation
Image = lambda: dLux.images.Image


class BaseInstrument(Base):
    """
    The Base Instrument class that all instrument classes inherit from. Can be
    used to create your own instrument classes that will integrate seamlessly
    with the rest of dLux.
    """

    @abstractmethod
    def model(self):  # pragma: no cover
        pass


class Instrument(Base):
    """
    A high level class designed to model the behaviour of a telescope. It
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
        in the different kind of observations, i.e. applying dithers, switching
        filters, etc.
    """

    optics: Optics()
    sources: dict
    detector: Detector()
    observation: Observation()

    def __init__(
        self: Instrument,
        optics: Optics(),
        sources: Union[list, Source()],
        detector: Detector() = None,
        observation: Observation = None,
    ):
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
            flexibility in the different kind of observations, i.e. applying
            dithers, switching filters, etc.
        """
        # Optics
        if not isinstance(optics, Optics()):
            raise TypeError("optics must be an Optics object.")
        self.optics = optics

        # Sources
        if isinstance(sources, (Source(), tuple)):
            sources = [sources]
        self.sources = dlu.list_to_dictionary(sources, False, Source())

        # Detector
        if not isinstance(detector, Detector()) and detector is not None:
            raise TypeError(
                "detector must be an Detector object. "
                f"Got type {type(detector)}"
            )
        self.detector = detector

        # Observation
        if (
            not isinstance(observation, Observation())
            and observation is not None
        ):
            raise TypeError("observation must be an Observation object.")
        self.observation = observation

    def observe(self: Instrument) -> Any:
        """
        Calls the `observe` method of the stored observation class, passing in
        any extra keyword arguments.

        Returns
        -------
         : Any
            The output of the stored observation class.
        """
        return self.observation.model(self)

    def __getattr__(self: Instrument, key: str) -> object:
        """
        Magic method designed to allow accessing of the various items within
        the sub-dictionaries of this class via the 'class.attribute' method.
        It is recommended that each dictionary key in the optical layers,
        detector layers, and scene sources are unique to prevent unexpected
        behaviour. In the case they there are identical keys across the
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
        for attribute in self.__dict__.values():
            if hasattr(attribute, key):
                return getattr(attribute, key)
        if key in self.sources.keys():
            return self.sources[key]
        for source in self.sources.values():
            if hasattr(source, key):
                return getattr(source, key)
        raise AttributeError(
            f"{self.__class__.__name__} has no attribute " f"{key}."
        )

    def normalise(self: Instrument) -> Instrument:
        """
        Method for returning a new instrument with normalised source objects.

        Returns
        -------
        instrument : Instrument
            The normalised instrument object.
        """
        is_source = lambda leaf: isinstance(leaf, Source())
        norm_fn = lambda source: source.normalise()
        sources = tree_map(norm_fn, self.sources, is_leaf=is_source)
        return self.set("sources", sources)

    def model(self: Instrument) -> Union[Array, dict]:
        """
        A base level modelling function designed to robustly handle the
        different combinations of inputs. Models the sources through the
        instrument optics and detector.

        Returns
        -------
        image : Array, dict
            The image of the scene modelled through the optics with detector
            and filter effects applied if they are supplied. Returns either as
            a single array (if return_tree is false), or a dict of the output
            for each source.
        """
        psf = self.optics.model(list(self.sources.values()))
        image = Image()(psf, self.optics.true_pixel_scale)
        image = (
            self.detector.model(image) if self.detector is not None else psf
        )
        return np.array(tree_flatten(image)[0]).sum(0)
