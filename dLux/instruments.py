from __future__ import annotations
from abc import abstractmethod
from zodiax import Base
from jax import Array, vmap
import jax.numpy as np
from typing import Union

from .optical_systems import BaseOpticalSystem as OpticalSystem
from .detectors import BaseDetector as Detector
from .sources import BaseSource as Source, Scene
from .psfs import PSF


__all__ = ["Instrument", "Telescope", "Dither"]


class Instrument(Base):
    @abstractmethod
    def model(self):  # pragma: no cover
        pass


class Telescope(Instrument):
    """
    Class that represents a telescope instrument, holding an optical system, a source
    object and (optionally) a detector object, automating the process of modelling
    all three in conjunction.

    To generate more complex instruments or a set of observations, the `Telescope`
    class can be inherited and modified to suit the needs of the user.

    Attributes
    ----------
    optics : OpticalSystem
        An `OpticalSystem` object that defines the optical transformations of the
        instrument.
    source : Source
        A `Source` or `Scene` to objects to model through the instrument.
    detector : Detector
        A `Detector` object that defines the detector transformations of the
        instrument.
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
        Parameters
        ----------
        optics : OpticalSystem
            An `OpticalSystem` object that defines the optical transformations of the
            instrument.
        source : Source
            A `Source` or `Scene` to objects to model through the instrument. Can be
            either a single `Source` object, or a list of `Source` objects which is
            then converted to a `Scene` object. The list entries can also be a tuple of
            (key, source)  in order to specify a key for the source in the scene.
        detector : Detector = None
            A `Detector` object that defines the detector transformations of the
            instrument.
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
        Raises the attributes from the optics, source and detector to the top level of
        the class.

        Parameters
        ----------
        key : str
            The key of the item to be searched for.

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

    def model(self: Telescope, return_psf: bool = False) -> Array:
        """
        Models the source objects through the optical system and detector.

        Parameters
        ----------
        return_psf : bool = False
            Should the PSF object be returned instead of the psf Array?

        Returns
        -------
        object : Array, PSF
            if `return_psf` is False, the psf Array is returned.
            If `return_psf` is True, the PSF object is returned.

        """
        # Model optics: return_psf=True for more efficient source calculations
        psfs = self.optics.model(self.source, return_psf=True)

        # Array based output
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


class Dither(Telescope):
    """
    Simple extension of the `Telescope` class that applies a series of dithers to the
    source positions before modelling the instrument. Serves both as a demonstration
    of how to extend the `Telescope` class and as a useful tool for modelling
    dithered observations.

    Attributes
    ----------
    optics : OpticalSystem
        An `OpticalSystem` object that defines the optical transformations of the
        instrument.
    source : Source
        A `Source` or `Scene` to objects to model through the instrument.
    detector : Detector
        A `Detector` object that defines the detector transformations of the
        instrument.
    dithers : Array, radians
        The array of dithers to apply to the source positions. The shape of the
        array should be (ndithers, 2).
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
        Parameters
        ----------
        dithers : Array, radians
            The array of dithers to apply to the source positions. The shape of the
            array should be (ndithers, 2).
        optics : OpticalSystem
            An `OpticalSystem` object that defines the optical transformations of the
            instrument.
        source : Source
            A `Source` or `Scene` to objects to model through the instrument. Can be
            either a single `Source` object, or a list of `Source` objects which is
            then converted to a `Scene` object. The list entries can also be a tuple of
            (key, source)  in order to specify a key for the source in the scene.
        detector : Detector = None
            A `Detector` object that defines the detector transformations of the
            instrument.
        """
        self.dithers = np.asarray(dithers, float)
        if self.dithers.ndim != 2 or self.dithers.shape[1] != 2:
            raise ValueError("dithers must be an array of shape (ndithers, 2)")
        super().__init__(optics=optics, source=source, detector=detector)

    def model(self: Telescope, return_psf: bool = False) -> Array:
        """
        Models the source objects through the optical system and detector, while also
        applying the dithers to the source positions.

        Parameters
        ----------
        return_psf : bool = False
            Should the PSF object be returned instead of the psf Array?

        Returns
        -------
        object : Array, PSF
            if `return_psf` is False, the psf Array is returned.
            If `return_psf` is True, the PSF object is returned.
        """

        def dither_and_model(dither, instrument):
            instrument = instrument.add("source.position", dither)
            return super(type(instrument), instrument).model(return_psf)

        return vmap(dither_and_model, (0, None))(self.dithers, self)
