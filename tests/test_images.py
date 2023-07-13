import jax.numpy as np
from jax import config

config.update("jax_debug_nans", True)


class TestImage:
    """Test the Image class."""

    def test_constructor(self, create_image):
        """Tests the constructor."""
        create_image()

    def test_npixels(self, create_image):
        """Tests the npixels property."""
        create_image().npixels

    def test_downsample(self, create_image):
        """Tests the downsample method."""
        create_image().downsample(2)

    def test_convolve(self, create_image):
        """Tests the convolve method."""
        create_image().convolve(np.ones((2, 2)))

    def test_rotate(self, create_image):
        """Tests the rotate method."""
        create_image().rotate(np.pi, order=1)

    def test_magic(self, create_image):
        """Tests the magic methods."""
        im = create_image()
        im *= np.array(1)
        im /= np.array(1)
        im += np.array(1)
        im -= np.array(1)
