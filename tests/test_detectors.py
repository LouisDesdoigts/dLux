import jax.numpy as np
import pytest
from jax import config

config.update("jax_debug_nans", True)


class TestLayeredDetector:
    """Tests the LayeredDetector class."""

    def test_constructor(self, create_layered_detector):
        """Tests the constructor."""
        create_layered_detector()
        with pytest.raises(TypeError):
            create_layered_detector(layers=[np.ones(1)])

    def test_model(self, create_layered_detector, create_image):
        """Tests the model method."""
        create_layered_detector().model(create_image())

    def test_getattr(self, create_layered_detector):
        """Tests the __getattr__ method."""
        create_layered_detector().AddConstant
        with pytest.raises(AttributeError):
            create_layered_detector().nonexistent_attribute
