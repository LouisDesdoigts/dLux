import jax
import jax.numpy as np
import equinox as eqx
import typing
import dLux
from collections import OrderedDict

__author__ = "Louis Desdoigts"
__date__ = "30/08/2022"
__all__ = ["Telescope", "Optics", "Scene",
           "Filter", "Detector", "Observation"]

# Base Jax Types
Array =  typing.NewType("Array",  np.ndarray)

# Classes
Telescope   = typing.NewType("Telescope",   object)
Optics      = typing.NewType("Optics",      object)
Scene       = typing.NewType("Scene",       object)
Filter      = typing.NewType("Filter",      object)
Detector    = typing.NewType("Detector",    object)
# Observation = typing.NewType("Observation", object)\



class Observation(dLux.base.Base):
    """

    """
    pointing : Array
    roll_angle : Array

    def __init__(self):
        raise NotImplementedError("Duh")


