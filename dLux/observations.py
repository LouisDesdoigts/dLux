from __future__ import annotations
from zodiax import Base
from jax.tree_util import tree_map
import jax.numpy as np
from jax import vmap, Array
from abc import abstractmethod
from equinox import tree_at
from typing import Any
import dLux


__all__ = ["BaseObservation", "Dither"]


Instrument = lambda : dLux.core.BaseInstrument


class BaseObservation(Base):
    """
    Abstract base class for observations. All observations should inherit from
    this class and must implement an `.observe()` method that only takes in a
    single instance of `dLux.Instrument`.
    """


    def __init__(self : BaseObservation):
        """
        Constructor for the BaseObservation class.
        """
        super().__init__()


    @abstractmethod
    def observe(self       : BaseObservation, 
                instrument : Instrument()) -> Any: # pragma: no cover
        """
        Abstract method for the observation function.
        """
        pass


class Dither(BaseObservation):
    """
    Observation class designed to apply a series of dithers to the insturment
    and return the corresponding psfs.

    Attributes
    ----------
    dithers : Array, (radians)
        The array of dithers to apply to the source positions. The shape of the
        array should be (ndithers, 2) where ndithers is the number of dithers
        and the second dimension is the (x, y) dither in radians.
    """
    dithers : Array


    def __init__(self : Dither, dithers : Array):
        """
        Constructor for the Dither class.

        Parameters
        ----------
        dithers : Array, (radians)
            The array of dithers to apply to the source positions. The shape of
            the array should be (ndithers, 2) where ndithers is the number of
            dithers and the second dimension is the (x, y) dither in radians.
        """
        super().__init__()
        self.dithers = np.asarray(dithers, float)
        dLux.exceptions.validate_bc_attr_dims(self.dithers.shape, (1, 2), 
            'dithers')


    def dither_position(self       : Dither, 
                        instrument : Instrument, 
                        dither     : Array) -> Instrument:
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
        # Define the dither function
        dither_fn = lambda source: source.add('position', dither)

        # Map the dithers across the sources
        dithered_sources = tree_map(dither_fn, instrument.sources, \
            is_leaf = lambda leaf: isinstance(leaf, dLux.sources.Source))

        # Apply updates
        return tree_at(lambda instrument: instrument.sources, instrument, 
            dithered_sources)


    def observe(self       : Dither,
                instrument : Instrument,
                *args, 
                **kwargs) -> Array:
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
        dith_fn = lambda dither: self.dither_position(instrument, 
            dither).model(*args, **kwargs)
        return vmap(dith_fn, 0)(self.dithers)