import dLux 
import jax.numpy as np
import pytest
from jax import config, Array
config.update("jax_debug_nans", True)



class TestDither(object):
    """Tests the Dither class."""

    def test_constructor(self, create_dither):
        """Tests the constructor."""
        create_dither()
        with pytest.raises(ValueError):
            dither = create_dither(np.array([1., 1., 1.]))
    
    def test_model(self, create_dither, create_instrument):
        """Tests the model method."""
        create_dither().model(create_instrument())