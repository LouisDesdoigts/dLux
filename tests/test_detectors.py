import jax.numpy as np
import pytest
from dLux import LayeredDetector, PSF
from dLux.layers import Downsample


psf = PSF(np.ones((16, 16)), 1 / 16)


def test_layered_detector():
    det = LayeredDetector([Downsample(4)])

    # Test getattr
    det.Downsample
    with pytest.raises(AttributeError):
        det.not_an_attr

    # Test model
    assert isinstance(det.model(psf), np.ndarray)
    assert isinstance(det.model(psf, return_psf=True), PSF)

    # Test insert and remove layer
    det.insert_layer(Downsample(4), 1)
    det.remove_layer("Downsample")
