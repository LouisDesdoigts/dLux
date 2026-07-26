"""Tests for dLux.systems."""

import jax.numpy as np
import pytest

import dLux as dl

from tests.helpers import assert_differentiable, assert_jittable


@pytest.fixture
def input_spec():
    return dl.GridSpec(n=(8, 8), d=(0.125, 0.125), unit="m")


@pytest.fixture
def focal_spec():
    return dl.GridSpec(n=(6, 8), d=(2e-7, 3e-7), unit="rad")


@pytest.fixture
def system(input_spec, focal_spec):
    return dl.OpticalSystem(
        layers=[
            (
                "pupil",
                dl.DynamicOptic(
                    transmission=dl.Circle(0.8, softening=0.02),
                    opd=dl.DynamicZernikeBasis(
                        js=[4],
                        coefficients=[1e-8],
                        diameter=0.8,
                    ),
                ),
            ),
            ("propagator", dl.Fraunhofer(focal_spec)),
            ("normalise", dl.Normalise()),
        ],
        spec=input_spec,
    )


def test_system_composition_contract(system):
    wavefront = system.initialise_wavefront(1e-6)
    output = assert_jittable(system, wavefront, rtol=1e-5, atol=1e-5)

    assert output.spec.unit == "rad"
    assert output.phasor.shape == (8, 6)
    assert np.allclose(output.power, 1)
    assert np.allclose(system.apply(wavefront).phasor, output.phasor)


def test_nested_optical_system(input_spec):
    subsystem = dl.OpticalSystem([dl.Optic(transmission=0.5)], input_spec)
    system = dl.OpticalSystem([subsystem, dl.Optic(transmission=0.5)], input_spec)
    wavefront = system.initialise_wavefront(1e-6)

    output = assert_jittable(system, wavefront)

    assert isinstance(subsystem, dl.BaseOpticalLayer)
    assert np.allclose(output.phasor, 0.25 * wavefront.phasor)


def test_layered_system_contract(make_psf):
    system = dl.LayeredSystem(
        [
            ("offset", dl.AddConstant(1)),
            ("normalise", dl.Normalise()),
        ]
    )
    psf = make_psf()

    output = assert_jittable(system, psf)
    debugged, intermediate = system.debug(psf)
    inserted = system.insert_layer(("flip", dl.Flip(0)), 1)
    removed = inserted.remove_layer("flip")

    assert isinstance(output, dl.PSF)
    assert np.allclose(debugged.data, output.data)
    assert list(intermediate) == ["input", "offset", "normalise"]
    assert list(inserted.layers) == ["offset", "flip", "normalise"]
    assert list(removed.layers) == list(system.layers)


def test_propagation_interfaces(system):
    wavelengths = np.array([0.9e-6, 1.1e-6])
    weights = np.array([0.4, 0.6])

    mono = assert_jittable(
        lambda value: system.propagate_mono(value, return_all=True),
        np.asarray(1e-6),
        rtol=1e-5,
        atol=1e-5,
    )
    chromatic = assert_jittable(
        lambda value: system.propagate(
            value,
            weights=weights,
            return_all=True,
        ),
        wavelengths,
        rtol=1e-5,
        atol=1e-5,
    )
    results = system.propagate(wavelengths, weights=weights, return_all=True)
    array = system.propagate(wavelengths, weights=weights)
    mono_wavefront = system.propagate_mono(1e-6, return_wf=True)
    wavefront = system.propagate(wavelengths, weights=weights, return_wf=True)

    assert isinstance(mono["Wavefront"], dl.Wavefront)
    assert isinstance(mono_wavefront, dl.Wavefront)
    assert isinstance(wavefront, dl.Wavefront)
    assert chromatic["Wavefront"].wavelength.shape == wavelengths.shape
    assert isinstance(results["PSF"], dl.PSF)
    assert array.shape == results["PSF"].data.shape
    assert np.allclose(array, results["PSF"].data)


def test_detector_uses_common_system_contract(make_psf):
    detector = dl.DetectorSystem(
        [
            dl.ApplyPixelResponse(np.ones((8, 8))),
            dl.AddConstant(1),
            dl.Normalise(),
        ]
    )
    psf = make_psf()

    transformed = assert_jittable(detector, psf)
    image = assert_jittable(detector.model, psf)
    output = assert_jittable(lambda value: detector.model(value, return_all=True), psf)

    assert isinstance(detector, dl.LayeredSystem)
    assert isinstance(transformed, dl.PSF)
    assert isinstance(image, dl.Image)
    assert isinstance(output["PSF"], dl.PSF)
    assert isinstance(output["Image"], dl.Image)
    assert np.allclose(image.data, output["Image"].data)


def test_chromatic_and_polarised_execution(system):
    wavelengths = np.array([0.9e-6, 1.1e-6])
    chromatic = system.initialise_wavefront(wavelengths)
    polarising = system.insert_layer(dl.LinearPolariser(0.2), 1)

    output = assert_jittable(system, chromatic, rtol=1e-5, atol=1e-5)
    polarised = assert_jittable(
        polarising,
        chromatic,
        rtol=1e-5,
        atol=1e-5,
    )

    assert output.phasor.shape[:1] == wavelengths.shape
    assert isinstance(polarised, dl.PolarisedWavefront)
    assert polarised.phasor.shape[:3] == wavelengths.shape + (2, 2)


def test_nested_parameter_gradients(system):
    coefficients = system.pupil.opd.coefficients

    assert_differentiable(
        lambda value: np.real(
            system.set(
                "layers.pupil.opd.coefficients",
                value,
            )
            .propagate_mono(1e-6, return_all=True)["Wavefront"]
            .phasor
        ),
        coefficients,
        rtol=1e-5,
        atol=1e-5,
    )


def test_layer_management_and_debugging(system):
    inserted = system.insert_layer(("flip", dl.Flip(0)), 2)
    removed = inserted.remove_layer("flip")
    output, intermediate = system.debug_propagate_mono(1e-6)

    assert list(inserted.layers) == [
        "pupil",
        "propagator",
        "flip",
        "normalise",
    ]
    assert list(removed.layers) == list(system.layers)
    assert list(intermediate) == [
        "initial_wavefront",
        "pupil",
        "propagator",
        "normalise",
    ]
    assert np.allclose(output.phasor, intermediate["normalise"].phasor)
    assert system.pupil is system.layers["pupil"]
    assert system.opd is system.pupil.opd


def test_model_interface(system):
    spectrum = dl.Spectrum([0.9e-6, 1.1e-6], [0.25, 0.75])
    source = dl.Source(
        spectrum.wavelengths,
        position=[0.1, -0.2],
        weights=spectrum.weights,
    )
    binary = dl.BinarySource(
        spectrum.wavelengths,
        separation=0.1,
        contrast=2.0,
        weights=spectrum.weights,
    )

    assert isinstance(system.model(spectrum), dl.PSF)
    results = system.model(spectrum, return_all=True)
    assert isinstance(results["Wavefront"], dl.Wavefront)
    assert isinstance(results["PSF"], dl.PSF)
    assert np.allclose(results["Wavefront"].wavelength, spectrum.wavelengths)
    sourced = system.model(source, return_all=True)
    assert np.allclose(sourced["Wavefront"].wavelength, spectrum.wavelengths)
    binary = system.model(binary, return_all=True)
    assert binary["PSF"].data.shape == (2, 8, 6)
    assert binary["PSF"].spec.d.shape == (2, 2)


@pytest.mark.parametrize(
    "operation",
    [
        lambda system: system(np.ones((8, 8))),
        lambda system: system.propagate([1e-6, 2e-6], weights=[1.0]),
        lambda system: system.propagate([1e-6], offset=[0.0]),
        lambda system: system.not_an_attribute,
        lambda system: dl.OpticalSystem([], np.ones(2)),
        lambda system: dl.OpticalSystem([], dl.GridSpec(n=8, unit="m")),
        lambda system: dl.OpticalSystem([], dl.GridSpec(n=8, d=0.1, unit="rad")),
        lambda system: dl.DetectorSystem([dl.Optic()]),
        lambda system: dl.DetectorSystem([])(np.ones((8, 8))),
        lambda system: dl.OpticalSystem([dl.AddConstant(1)], system.spec),
        lambda system: system.insert_layer(dl.AddConstant(1), 0, dl.BaseOpticalLayer),
        lambda system: dl.DetectorSystem([]).insert_layer(
            dl.Optic(), 0, dl.BaseDetectorLayer
        ),
        lambda system: system.model(np.ones(2)),
        lambda system: system.propagate_mono(1e-6, return_wf=True, return_all=True),
        lambda system: system.propagate([1e-6], return_wf=True, return_all=True),
    ],
)
def test_validation(system, operation):
    with pytest.raises((AttributeError, TypeError, ValueError)):
        operation(system)
