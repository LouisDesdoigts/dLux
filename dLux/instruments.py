from __future__ import annotations
from abc import abstractmethod
from zodiax import Base
from jax import Array, vmap
import jax.numpy as np
from typing import Union
import dLux.utils as dlu

from .optical_systems import BaseOpticalSystem as OpticalSystem
from .detectors import BaseDetector as Detector
from .sources import BaseSource as Source, Scene
from .containers.psfs import PSF


__all__ = ["Instrument", "Telescope", "Dither"]


class Instrument(Base):
    @abstractmethod
    def model(self):  # pragma: no cover
        pass


# TODO: re-name to Telescope
class Telescope(Instrument):
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

    optics: OpticalSystem
    source: Source
    detector: Detector

    def __init__(
        self: Telescope,
        optics: OpticalSystem,
        source: Union[list, Source],
        detector: Detector = None,
    ):
        """
        Constructor for the Telescope class.

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
        if not isinstance(optics, OpticalSystem):
            raise TypeError("optics must be an Optics object.")
        self.optics = optics

        # Sources
        if isinstance(source, Source):
            self.source = source
        elif isinstance(source, tuple):
            # If its a (key, source) tuple, we ignore the key
            self.source = source[1]
        else:
            self.source = Scene(source)

        # Detector
        if not isinstance(detector, Detector) and detector is not None:
            raise TypeError(
                "detector must be an Detector object. "
                f"Got type {type(detector)}"
            )
        self.detector = detector

    def __getattr__(self: Telescope, key: str) -> object:
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

    def model(self: Telescope, return_psf: bool = False) -> Union[Array, dict]:
        """
        A base level modelling function designed to robustly handle the
        different combinations of inputs. Models the  through the
        instrument optics and detector.

        Returns
        -------
        psf : Array, dict
            The psf of the scene modelled through the optics with detector
            and filter effects applied if they are supplied. Returns either as
            a single array (if return_tree is false), or a dict of the output
            for each source.
        """
        # Model optics: return_psf=True for more efficient source calculations
        psfs = self.optics.model(self.source, return_psf=True)

        # Check for tree-like output from scene
        if not isinstance(psfs, PSF):
            # Define functions
            leaf_fn = lambda x: isinstance(x, PSF)
            get_psfs = lambda psf: psf.data.sum(tuple(range(psf.ndim)))
            get_pscales = lambda psf: psf.pixel_scale.mean()

            # Get values
            psf = dlu.map2array(get_psfs, psfs, leaf_fn).sum()
            pixel_scale = dlu.map2array(get_pscales, psfs, leaf_fn).mean()

        # Array based output
        else:
            psf = psfs.data.sum(tuple(range(psfs.ndim)))
            pixel_scale = psfs.pixel_scale.mean()

        # Pass through detector transformations if it exists
        psf_obj = PSF(psf, pixel_scale)
        if self.detector is not None:
            return self.detector.model(psf_obj, return_psf=return_psf)

        # Return psf
        if return_psf:
            return psf_obj
        return psf_obj.data


# TODO: Test and re-write the Dither class
class Dither(Telescope):
    """
    Telescope class designed to apply a series of dithers to the instrument
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
        self: Telescope,
        dithers: Array,
        optics: OpticalSystem,
        source: Union[list, Source],
        detector: Detector = None,
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
        sourcs : Union[list, Source]
            Either a list of sources or an individual Source object.
        detector : Detector = None
            A pre-configured Detector object.
        """
        self.dithers = np.asarray(dithers, float)
        if self.dithers.ndim != 2 or self.dithers.shape[1] != 2:
            raise ValueError("dithers must be an array of shape (ndithers, 2)")
        super().__init__(optics=optics, source=source, detector=detector)

    def model(self: Telescope) -> Array:
        """
        Applies a series of dithers to the instrument sources and calls the
        .model() method after applying each dither.

        Parameters
        ----------
        instrument : Telescope
            The array of dithers to apply to the source positions.

        Returns
        -------
        psfs : Array
            The psfs generated after applying the dithers to the source
            positions.
        """

        def dither_and_model(dither, instrument):
            instrument = instrument.add("source.position", dither)
            return super(type(instrument), instrument).model()

        return vmap(dither_and_model, (0, None))(self.dithers, self)
