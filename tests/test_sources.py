from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest
from dLux import (
    Scene,
    PointSource,
    PointSources,
    BinarySource,
    ResolvedSource,
    PointResolvedSource,
    Spectrum,
    LayeredOpticalSystem,
    Wavefront,
    Optic,
    PSF,
)
import dLux.sources as sources_module


def _test_model(source):
    optics = LayeredOpticalSystem(16, 1.0, [Optic()])
    assert isinstance(source.model(optics), np.ndarray)
    if isinstance(source, (ResolvedSource, PointResolvedSource)):
        with pytest.raises(NotImplementedError):
            source.model(optics, return_wf=True)
    else:
        assert isinstance(source.model(optics, return_wf=True), Wavefront)
    assert isinstance(source.model(optics, return_psf=True), PSF)
    with pytest.raises(ValueError):
        source.model(optics, return_wf=True, return_psf=True)


@pytest.fixture
def wavelengths():
    return np.ones(2)


@pytest.fixture
def spectrum():
    return Spectrum([1e-6])


def test_point_source(wavelengths, spectrum):
    _test_model(PointSource(wavelengths))
    _test_model(PointSource(wavelengths, spectrum=spectrum))

    with pytest.raises(ValueError):
        PointSource(wavelengths, position=np.ones(1))
    with pytest.raises(TypeError):
        PointSource(wavelengths, spectrum=1)


def test_point_sources(wavelengths, spectrum):
    _test_model(PointSources(wavelengths))
    _test_model(PointSources(wavelengths, spectrum=spectrum))
    _test_model(PointSources(wavelengths, flux=np.ones(1)))

    with pytest.raises(ValueError):
        PointSources(wavelengths, position=np.ones(1))
    with pytest.raises(ValueError):
        PointSources(wavelengths, flux=np.ones(3))
    with pytest.raises(ValueError):
        PointSources(wavelengths, flux=np.ones((1, 3)))


def test_resolved_source(wavelengths, spectrum):
    _test_model(ResolvedSource(wavelengths))

    with pytest.raises(ValueError):
        ResolvedSource(wavelengths, distribution=np.ones(1))


def test_binary_source(wavelengths, spectrum):
    _test_model(BinarySource(wavelengths, separation=1.0))

    with pytest.raises(ValueError):
        BinarySource(wavelengths, separation=1.0, position=np.ones(1))


def test_point_resolved_source(wavelengths, spectrum):
    _test_model(PointResolvedSource(wavelengths))


def test_scene():
    keyed_scene = Scene([("source", PointSource([1e-6]))])
    assert isinstance(keyed_scene.source, PointSource)

    single_scene = Scene(PointSource([1e-6]))
    assert isinstance(single_scene.PointSource, PointSource)

    tuple_scene = Scene((PointSource([1e-6]), PointSource([1e-6])))
    assert isinstance(tuple_scene.PointSource_0, PointSource)

    scene = Scene([PointSource([1e-6]), PointSource([1e-6])])
    optics = LayeredOpticalSystem(16, 1.0, [Optic()])

    # In this case we have an output dictionary of each source
    assert isinstance(scene.model(optics), np.ndarray)
    output = scene.model(optics, return_wf=True).values()
    for o in output:
        assert isinstance(o, Wavefront)

    # In this case we have a single PSF object
    assert isinstance(scene.model(optics, return_psf=True), PSF)

    # Test getattr
    scene.PointSource_0
    with pytest.raises(AttributeError):
        scene.not_an_attr


def test_source_helper_functions(spectrum):
    assert sources_module._as_wavelengths_1d(None) is None

    with pytest.raises(ValueError):
        sources_module._as_wavelengths_1d(np.ones((2, 2)))

    assert sources_module._infer_n_wavelengths(np.ones(3), None) == 3
    assert sources_module._infer_n_wavelengths(None, spectrum) == 1

    with pytest.raises(ValueError):
        sources_module._infer_n_wavelengths(None, None)

    with pytest.raises(TypeError):
        sources_module._infer_n_wavelengths(None, 1)
