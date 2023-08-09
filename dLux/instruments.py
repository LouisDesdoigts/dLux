from __future__ import annotations
from abc import abstractmethod
import jax.numpy as np
from jax import Array, vmap
from jax.tree_util import tree_map, tree_flatten
from equinox import tree_at
from zodiax import Base
from typing import Union
import dLux

__all__ = ["Instrument"]

# Alias classes for simplified type-checking
Optics = lambda: dLux.optics.BaseOptics
Detector = lambda: dLux.detectors.BaseDetector
Source = lambda: dLux.sources.BaseSource
PSF = lambda: dLux.psfs.PSF


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
    source : Source
        A dictionary of the various source objects that the instrument is
        observing.
    detector : Detector
        A Detector object that is used to model the various
        instrumental effects on a psf.
    """

    optics: Optics()
    source: Source()
    detector: Detector()

    def __init__(
        self: Instrument,
        optics: Optics(),
        source: Union[list, Source()],
        detector: Detector() = None,
    ):
        """
        Constructor for the Instrument class.

        Parameters
        ----------
        optics : Optics
            A pre-configured Optics object.
        source : Union[list, Source]
            Either a Scene, list of Sources, or an individual Source object.
        detector : Detector = None
            A pre-configured Detector object.
        """
        # Optics
        if not isinstance(optics, Optics()):
            raise TypeError("optics must be an Optics object.")
        self.optics = optics

        # Sources
        if isinstance(source, Source()):
            self.source = source
        elif isinstance(source, tuple):
            # If its a (source, key) tuple, we ignore the key
            self.source = source[0]
        else:
            self.source = dLux.sources.Scene(source)

        # Detector
        if not isinstance(detector, Detector()) and detector is not None:
            raise TypeError(
                "detector must be an Detector object. "
                f"Got type {type(detector)}"
            )
        self.detector = detector

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
        return self.set("source", self.source.normalise())

    def model(self: Instrument) -> Union[Array, dict]:
        """
        A base level modelling function designed to robustly handle the
        different combinations of inputs. Models the sources through the
        instrument optics and detector.

        Returns
        -------
        psf : Array, dict
            The psf of the scene modelled through the optics with detector
            and filter effects applied if they are supplied. Returns either as
            a single array (if return_tree is false), or a dict of the output
            for each source.
        """
        psf, pixel_scale = self.optics.model(self.sources, True)
        psf = PSF()(psf, pixel_scale)
        psf = self.detector.model(psf) if self.detector is not None else psf
        return np.array(tree_flatten(psf)[0]).sum(0)


# TODO: Test and re-write the Dither class
class Dither(Instrument):
    """
    Instrument class designed to apply a series of dithers to the instrument
    and return the corresponding PSFs.

    Attributes
    ----------
    dithers : Array, (radians)
        The array of dithers to apply to the source positions. The shape of the
        array should be (ndithers, 2) where ndithers is the number of dithers
        and the second dimension is the (x, y) dither in radians.
    """

    dithers: Array

    def __init__(
        self: Instrument,
        dithers: Array,
        optics: Optics(),
        sources: Union[list, Source()],
        detector: Detector() = None,
    ):
        """
        Constructor for the Dither class.

        Parameters
        ----------
        dithers : Array, radians
            The array of dithers to apply to the source positions. The shape of
            the array should be (ndithers, 2) where ndithers is the number of
            dithers and the second dimension is the (x, y) dither in radians.
        optics : Optics
            A pre-configured Optics object.
        sources : Union[list, Source]
            Either a list of sources or an individual Source object.
        detector : Detector = None
            A pre-configured Detector object.
        """
        self.dithers = np.asarray(dithers, float)
        if self.dithers.ndim != 2 or self.dithers.shape[1] != 2:
            raise ValueError("dithers must be an array of shape (ndithers, 2)")
        super().__init__(optics=optics, sources=sources, detector=detector)

    def dither_position(
        self: Instrument, instrument: Instrument, dither: Array
    ) -> Instrument:
        """
        Dithers the position of the source objects by dither.

        Parameters
        ----------
        instrument : Instrument
            The instrument to dither.
        dither : Array, radians
            The (x, y) dither to apply to the source positions.

        Returns
        -------
        instrument : Instrument
            The instrument with the sources dithered.
        """
        # Define the dither function
        dither_fn = lambda source: source.add("position", dither)

        # Map the dithers across the sources
        dithered_sources = tree_map(
            dither_fn,
            instrument.sources,
            is_leaf=lambda leaf: isinstance(leaf, dLux.sources.Source),
        )

        # Apply updates
        return tree_at(
            lambda instrument: instrument.sources, instrument, dithered_sources
        )

    def model(self: Dither, instrument: Instrument, *args, **kwargs) -> Array:
        """
        Applies a series of dithers to the instrument sources and calls the
        .model() method after applying each dither.

        Parameters
        ----------
        instrument : Instrument
            The array of dithers to apply to the source positions.

        Returns
        -------
        psfs : Array
            The psfs generated after applying the dithers to the source
            positions.
        """
        dith_fn = lambda dither: self.dither_position(
            instrument, dither
        ).model(*args, **kwargs)
        return vmap(dith_fn, 0)(self.dithers)
