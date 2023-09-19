import jax.numpy as np
import pytest
from dLux import (
    LayeredOptics,
    AngularOptics,
    CartesianOptics,
    PointSource,
    Wavefront,
    PSF,
)
from dLux.layers import Optic


@pytest.fixture
def wf_npixels():
    return 16


@pytest.fixture
def diameter():
    return 1.0


@pytest.fixture
def layers():
    return [Optic()]


def _test_model(optics):
    source = PointSource([1e-6])
    assert isinstance(optics.model(source), np.ndarray)
    assert isinstance(optics.model(source, return_wf=True), Wavefront)
    assert isinstance(optics.model(source, return_psf=True), PSF)
    with pytest.raises(ValueError):
        optics.model(source, return_wf=True, return_psf=True)


def _test_propagate(optics):
    wavels = np.ones(2)
    assert isinstance(optics.propagate(wavels), np.ndarray)
    assert isinstance(optics.propagate(wavels, return_wf=True), Wavefront)
    assert isinstance(optics.propagate(wavels, return_psf=True), PSF)

    with pytest.raises(ValueError):
        optics.propagate(wavels, return_wf=True, return_psf=True)
    with pytest.raises(ValueError):
        optics.propagate(wavels, weights=np.ones(3))
    with pytest.raises(ValueError):
        optics.propagate(wavels, offset=np.ones(3))


def _test_propagate_mono(optics):
    assert isinstance(optics.propagate_mono(1e-6), np.ndarray)
    assert isinstance(optics.propagate_mono(1e-6, return_wf=True), Wavefront)


def test_layered_optics(wf_npixels, diameter, layers):
    optics = LayeredOptics(wf_npixels, diameter, layers)
    _test_model(optics)
    _test_propagate(optics)
    _test_propagate_mono(optics)

    # Test getattr
    optics.Optic
    optics.opd
    with pytest.raises(AttributeError):
        optics.not_an_attr

    # Test insert and remove layer
    optics.insert_layer(Optic(), 1)
    optics.remove_layer("Optic")


@pytest.fixture
def psf_npixels():
    return 8


@pytest.fixture
def psf_pixel_scale():
    return 1 / 8


@pytest.fixture
def oversample():
    return 2


def test_angular_optics(
    wf_npixels, diameter, layers, psf_npixels, psf_pixel_scale, oversample
):
    optics = AngularOptics(
        wf_npixels, diameter, layers, psf_npixels, psf_pixel_scale, oversample
    )
    _test_model(optics)
    _test_propagate(optics)
    _test_propagate_mono(optics)


def test_cartesian_optics(
    wf_npixels, diameter, layers, psf_npixels, psf_pixel_scale, oversample
):
    optics = CartesianOptics(
        wf_npixels, diameter, layers, psf_npixels, psf_pixel_scale, oversample
    )
    _test_model(optics)
    _test_propagate(optics)
    _test_propagate_mono(optics)
