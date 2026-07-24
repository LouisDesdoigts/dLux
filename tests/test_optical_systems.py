from jax import numpy as np, config
import equinox as eqx

config.update("jax_debug_nans", True)
import pytest
from dLux import (
    LayeredOpticalSystem,
    AngularOpticalSystem,
    CartesianOpticalSystem,
    PointSource,
    Wavefront,
    PolarisedWavefront,
    PSF,
)
from dLux.layers import LinearPolariser, Optic


class MonochromaticOptic(Optic):
    """Test layer that rejects vectorised wavefront inputs."""

    def __call__(self, wavefront):
        assert wavefront.wavelength.ndim == 0
        assert wavefront.ndim == 0
        return wavefront


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


def _test_apply_wavefront(optics, wf_npixels, diameter):
    with pytest.raises(TypeError, match="wavefront must be a Wavefront instance"):
        optics(np.ones((wf_npixels, wf_npixels)))

    scalar = Wavefront(1e-6, wf_npixels, diameter)
    assert isinstance(optics(scalar), Wavefront)
    assert isinstance(optics.apply(scalar), Wavefront)

    wavelengths = np.array([0.9e-6, 1.0e-6, 1.1e-6])
    chromatic = Wavefront(wavelengths, wf_npixels, diameter)
    actual = optics(chromatic)
    expected = eqx.filter_vmap(
        lambda wavelength: optics.propagate_mono(wavelength, return_wf=True)
    )(wavelengths)

    assert actual.is_chromatic
    assert actual.wavelength.shape == wavelengths.shape
    assert np.allclose(actual.phasor, expected.phasor)
    assert np.allclose(actual.pixel_scale, expected.pixel_scale)

    polarised = PolarisedWavefront(wavelengths, wf_npixels, diameter)
    assert polarised._mapped_axis.phasor == 0
    assert polarised._mapped_axis.wavelength == 0
    assert polarised._mapped_axis.pixel_scale is None
    assert polarised._mapped_axis.center is None

    polarised_output = optics(polarised)
    assert isinstance(polarised_output, PolarisedWavefront)
    assert polarised_output.phasor.shape[:-4] == wavelengths.shape


def _test_debug_propagate_mono(optics):
    wavefront, outputs = optics.debug_propagate_mono(1e-6)

    assert isinstance(wavefront, Wavefront)
    assert isinstance(outputs, dict)
    assert isinstance(outputs["initial_wavefront"], Wavefront)


def test_layered_optics(wf_npixels, diameter, layers):
    optics = LayeredOpticalSystem(wf_npixels, diameter, layers)
    _test_model(optics)
    _test_propagate(optics)
    _test_propagate_mono(optics)
    _test_apply_wavefront(optics, wf_npixels, diameter)
    _test_debug_propagate_mono(optics)

    # Test getattr
    optics.Optic
    optics.opd
    with pytest.raises(AttributeError):
        optics.not_an_attr

    # Test insert and remove layer
    inserted = optics.insert_layer(Optic(), 1)
    assert isinstance(inserted, LayeredOpticalSystem)
    assert len(inserted.layers) == 2
    assert list(inserted.layers.keys()) == ["Optic_0", "Optic_1"]

    removed = inserted.remove_layer("Optic_0")
    assert isinstance(removed, LayeredOpticalSystem)
    assert len(removed.layers) == 1


def test_chromatic_call_maps_monochromatic_wavefronts(wf_npixels, diameter):
    optics = LayeredOpticalSystem(wf_npixels, diameter, [MonochromaticOptic()])
    wavelengths = np.ones(2) * 1e-6

    output = optics(Wavefront(wavelengths, wf_npixels, diameter))
    polarised_output = optics(PolarisedWavefront(wavelengths, wf_npixels, diameter))

    assert output.wavelength.shape == wavelengths.shape
    assert output.phasor.shape == wavelengths.shape + (wf_npixels, wf_npixels)
    assert polarised_output.wavelength.shape == wavelengths.shape
    assert polarised_output.phasor.shape == wavelengths.shape + (
        2,
        2,
        wf_npixels,
        wf_npixels,
    )


def test_chromatic_call_maps_shared_polarised_phasor(wf_npixels, diameter):
    optics = LayeredOpticalSystem(wf_npixels, diameter, [MonochromaticOptic()])
    wavelengths = np.ones(2) * 1e-6
    phasor = PolarisedWavefront(1e-6, wf_npixels, diameter).phasor
    wavefront = PolarisedWavefront(wavelengths, wf_npixels, diameter).set(phasor=phasor)

    assert wavefront._mapped_axis.phasor is None
    assert wavefront._mapped_axis.wavelength == 0

    output = optics(wavefront)

    assert output.wavelength.shape == wavelengths.shape
    assert output.phasor.shape == wavelengths.shape + (
        2,
        2,
        wf_npixels,
        wf_npixels,
    )


def test_chromatic_call_preserves_polarisation_promotion(wf_npixels, diameter):
    optics = LayeredOpticalSystem(wf_npixels, diameter, [LinearPolariser(0.0)])
    wavelengths = np.ones(2) * 1e-6

    output = optics(Wavefront(wavelengths, wf_npixels, diameter))

    assert isinstance(output, PolarisedWavefront)
    assert output.wavelength.shape == wavelengths.shape
    assert output.phasor.shape == wavelengths.shape + (
        2,
        2,
        wf_npixels,
        wf_npixels,
    )


@pytest.fixture
def psf_npixels():
    return 8


@pytest.fixture
def psf_pixel_scale():
    return 1 / 8


@pytest.fixture
def oversample():
    return 2


@pytest.fixture
def focal_length():
    return 1.0


def test_angular_optics(
    wf_npixels, diameter, layers, psf_npixels, psf_pixel_scale, oversample
):
    optics = AngularOpticalSystem(
        wf_npixels, diameter, layers, psf_npixels, psf_pixel_scale, oversample
    )
    _test_model(optics)
    _test_propagate(optics)
    _test_propagate_mono(optics)
    _test_apply_wavefront(optics, wf_npixels, diameter)
    _test_debug_propagate_mono(optics)

    assert optics.fov == psf_npixels * psf_pixel_scale
    assert optics.diameter.shape == ()
    assert optics.psf_pixel_scale.shape == ()


def test_cartesian_optics(
    wf_npixels,
    diameter,
    layers,
    focal_length,
    psf_npixels,
    psf_pixel_scale,
    oversample,
):
    optics = CartesianOpticalSystem(
        wf_npixels,
        diameter,
        layers,
        focal_length,
        psf_npixels,
        psf_pixel_scale,
        oversample,
    )
    _test_model(optics)
    _test_propagate(optics)
    _test_propagate_mono(optics)
    _test_apply_wavefront(optics, wf_npixels, diameter)
    _test_debug_propagate_mono(optics)

    assert optics.fov == psf_npixels * psf_pixel_scale
    assert optics.diameter.shape == ()
    assert optics.focal_length.shape == ()
    assert optics.psf_pixel_scale.shape == ()
