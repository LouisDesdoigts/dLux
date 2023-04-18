import dLux 
import jax.numpy as np
import typing
from dLux.exceptions import DimensionError
import pytest
from jax import config, Array
config.update("jax_debug_nans", True)


Dither = dLux.observations.Dither


class TestDither(object):
    """
    Contains the unit tests for the `UniformSpider` class.
    """

    def test_constructor(self, create_dither: callable) -> None:

        with pytest.raises(DimensionError):
            dither = create_dither(np.array([1., 1., 1.]))
        
        dither = create_dither()