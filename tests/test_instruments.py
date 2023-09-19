import jax.numpy as np
import pytest
from dLux import (
    Telescope,
    Dither,
    LayeredOptics,
    LayeredDetector,
    PointSource,
    PSF,
    Optic,
    AddConstant,
)


@pytest.fixture
def optics():
    return LayeredOptics(16, 1.0, [Optic()])


@pytest.fixture
def detector():
    return LayeredDetector([AddConstant(1)])


@pytest.fixture
def source():
    return PointSource([1e-6])


def _test_model(inst):
    assert isinstance(inst.model(), np.ndarray)
    assert isinstance(inst.model(return_psf=True), PSF)


def test_instrument(optics, detector, source):
    # All inputs
    _test_model(Telescope(optics, source, detector))

    # No detector
    _test_model(Telescope(optics, source))

    # Single source with key
    _test_model(Telescope(optics, ("source", source)))

    # Multiple sources
    _test_model(Telescope(optics, [source, source]))

    # Test failures
    with pytest.raises(TypeError):
        Telescope(1, source, detector)
    with pytest.raises(TypeError):
        Telescope(optics, source, 1)

    # Test getattr
    inst = Telescope(optics, source, detector)
    inst.opd
    with pytest.raises(AttributeError):
        inst.not_an_attr


def test_dither(optics, detector, source):
    _test_model(Dither(np.zeros((1, 2)), optics, source, detector))

    with pytest.raises(ValueError):
        Dither(np.zeros((1, 3)), optics, source, detector)
