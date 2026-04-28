from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest
from dLux import LayeredDetector, PSF
from dLux.layers import Downsample

psf = PSF(np.ones((16, 16)), 1 / 16)


def test_layered_detector():
    det = LayeredDetector([Downsample(4)])

    # Test getattr
    det.Downsample
    assert det.kernel_size == 4
    with pytest.raises(AttributeError):
        det.not_an_attr

    # Test model
    assert isinstance(det.model(psf), np.ndarray)
    assert isinstance(det.model(psf, return_psf=True), PSF)

    # Test insert and remove layer
    inserted = det.insert_layer(Downsample(4), 1)
    assert isinstance(inserted, LayeredDetector)
    assert list(inserted.layers.keys()) == ["Downsample_0", "Downsample_1"]

    removed = inserted.remove_layer("Downsample_0")
    assert isinstance(removed, LayeredDetector)
    assert list(removed.layers.keys()) == ["Downsample_1"]
