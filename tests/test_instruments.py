import pytest
from jax import config

config.update("jax_debug_nans", True)


class TestInstrument(object):
    """Tests the Instrument class."""

    def test_constructor(self, create_instrument):
        """Tests the constructor."""
        create_instrument()
        with pytest.raises(TypeError):
            create_instrument(sources=["Not a source"])
        with pytest.raises(TypeError):
            create_instrument(optics="Not an optics")
        with pytest.raises(TypeError):
            create_instrument(observation="Not an observation")
        with pytest.raises(TypeError):
            create_instrument(detector="Not a detector")

    def test_observe(self, create_instrument):
        """Tests the normalise method."""
        create_instrument().observe()

    def test_normalise(self, create_instrument):
        """Tests the normalise method."""
        create_instrument().normalise()

    def test_model(self, create_instrument):
        """Tests the model method."""
        create_instrument().model()

    def test_getattr(self, create_instrument):
        """Tests the __getattr__ method."""
        create_instrument().diameter
        create_instrument().PointSource
        create_instrument().flux
        with pytest.raises(AttributeError):
            create_instrument().nonexistent_attribute
