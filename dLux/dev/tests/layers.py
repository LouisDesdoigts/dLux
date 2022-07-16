from layers import *
from matplotlib import pyplot
from jax import numpy as np
from jax.config import config
from typing import TypeVar

config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)


class TestHexagonalAperture(object):
    """
    Blank stub, since tests are best performed with visual comparison 
    to poppy. Will likely be filled in future releases.
    """
