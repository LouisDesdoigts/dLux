"""Tests for deprecated public compatibility contracts."""

import importlib

import jax.numpy as np
import pytest

import dLux as dl
import dLux.utils as dlu


@pytest.mark.parametrize("wavefront_type", [dl.Wavefront, dl.PolarisedWavefront])
def test_from_phasor_constructor_alias(wavefront_type):
    grid = dl.GridSpec(n=4, d=0.1, unit="m").broadcast(2)
    phasor = np.ones((4, 4), dtype=complex)

    with pytest.warns(DeprecationWarning) as record:
        wavefront = wavefront_type.from_phasor(phasor, 500e-9, grid)

    message = str(record[0].message)
    assert "removed in dLux 0.17.0" in message
    assert f"{wavefront_type.__name__}(w, g, p)" in message
    assert isinstance(wavefront, wavefront_type)


def test_coefficients_constructor_alias():
    basis = np.ones((2, 3, 3))

    with pytest.warns(DeprecationWarning) as record:
        parametric = dl.Basis(basis, coefficients=[1.0, 2.0])

    message = str(record[0].message)
    assert "removed in dLux 0.17.0" in message
    assert "Class(coefficients=value)` -> `Class(coeffs=value)" in message
    assert np.allclose(parametric.coeffs, np.array([1.0, 2.0]))


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: dl.Polynomial(degree=1, coefficients=[1.0, 2.0]),
        lambda: dl.SpectralPolynomial(degree=1, coefficients=[0.2]),
        lambda: dl.CauchyIndex(coefficients=[1.5, 0.01]),
    ],
)
def test_coefficients_constructor_aliases(constructor):
    with pytest.warns(DeprecationWarning, match="Class\\(coefficients=value\\)"):
        parametric = constructor()

    assert hasattr(parametric, "coeffs")


@pytest.mark.parametrize(
    "parametric, migration",
    [
        (
            dl.Basis(np.ones((2, 3, 3)), coeffs=[1.0, 2.0]),
            "basis.coefficients` -> `basis.coeffs",
        ),
        (
            dl.CauchyIndex(coeffs=[1.5, 0.01]),
            "model.coefficients` -> `model.coeffs",
        ),
    ],
)
def test_coefficients_attribute_alias(parametric, migration):

    with pytest.warns(DeprecationWarning) as record:
        coeffs = parametric.coefficients

    message = str(record[0].message)
    assert "removed in dLux 0.17.0" in message
    assert migration in message
    assert np.allclose(coeffs, parametric.coeffs)


@pytest.mark.parametrize(
    "legacy, current, kwargs",
    [
        (dl.CoordSpec, dl.GridSpec, {"n": 8, "d": 0.1}),
        (dl.PadSpec, dl.ResizeSpec, {"pad": 2, "crop": 1}),
        (dl.DistortedCoords, dl.Distortion, {"order": 2}),
    ],
)
def test_legacy_coordinate_interfaces(legacy, current, kwargs):
    with pytest.warns(DeprecationWarning) as record:
        value = legacy(**kwargs)

    message = str(record[0].message)
    assert "removed in dLux 0.17.0" in message
    assert current.__name__ in message
    assert isinstance(value, current)


def test_legacy_coordinate_behaviour():
    with pytest.warns(DeprecationWarning):
        spec = dl.CoordSpec(8, 0.1)
    with pytest.warns(DeprecationWarning):
        transform = dl.DistortedCoords(order=1)

    assert spec.xs.shape == (8,)
    assert np.isclose(spec.fov, 0.8)
    assert transform.calculate(8, 1.0).shape == (2, 8, 8)


def test_legacy_coord_transform_behaviour():
    coordinates = dlu.pixel_coords(8, 1.0)

    with pytest.warns(DeprecationWarning, match="CoordTransform"):
        transform = dl.CoordTransform(
            translation=[0.1, -0.2],
            rotation=0.2,
            compression=[0.9, 1.1],
            shear=[0.05, -0.1],
        )

    expected = dlu.translate_coords(coordinates, transform.translation)
    expected = dlu.shear_coords(expected, transform.shear)
    expected = dlu.compress_coords(expected, transform.compression)
    expected = dlu.rotate_coords(expected, transform.rotation)

    assert np.allclose(transform(coordinates), expected)


def test_legacy_detector_return_contract():
    with pytest.warns(DeprecationWarning):
        psf = dl.PSF(np.ones((4, 4)), dl.GridSpec(4, 0.1, unit="rad"))

    with pytest.warns(DeprecationWarning):
        detector = dl.LayeredDetector([])

    assert isinstance(detector(psf), type(psf.data))
    assert isinstance(detector(psf, return_psf=True), dl.PSF)


@pytest.mark.parametrize(
    ("legacy", "current", "args", "path"),
    [
        (dl.ApplyPixelResponse, dl.Sensitivity, (np.ones((4, 4)),), "pixel_response"),
        (dl.ApplyJitter, dl.Jitter, (0.5,), "sigma"),
        (dl.ApplySaturation, dl.Saturation, (10.0,), "threshold"),
        (dl.AddConstant, dl.Bias, (1.0,), "value"),
    ],
)
def test_legacy_detector_layers(legacy, current, args, path):
    intensity = dl.Intensity(np.arange(16.0).reshape(4, 4), dl.GridSpec(4, 0.1))

    with pytest.warns(DeprecationWarning) as record:
        layer = legacy(*args)

    replacement = current(*args)
    message = str(record[0].message)

    assert "removed in dLux 0.17.0" in message
    assert current.__name__ in message
    assert " -> " in message
    assert hasattr(layer, path)
    if legacy is not dl.ApplyJitter:
        assert np.allclose(layer(intensity).data, replacement(intensity).data)


def test_psf_alias():
    grid = dl.GridSpec(4, 0.1, unit="rad")

    with pytest.warns(DeprecationWarning) as record:
        intensity = dl.PSF(np.ones((4, 4)), grid)

    message = str(record[0].message)
    assert "removed in dLux 0.17.0" in message
    assert "`dl.PSF(data, pixel_scale)` -> `dl.Intensity(data, grid)`" in message
    assert isinstance(intensity, dl.Intensity)


def test_released_psf_constructor():
    with pytest.warns(DeprecationWarning):
        psf = dl.PSF(np.ones((4, 4)), 0.1)

    assert np.isclose(psf.pixel_scale, 0.1)
    assert psf.ndim == 0
    assert np.allclose(psf.downsample(2).pixel_scale, 0.2)


def test_released_jitter_contract():
    with pytest.warns(DeprecationWarning):
        jitter = dl.ApplyJitter(0.5, kernel_size=4, oversample=2)

    expected = dlu.gaussian(np.zeros(2), np.ones(2) * 0.5, 8)
    expected = dlu.downsample(expected, 2, mean=False)

    assert jitter.kernel_size == 4
    assert jitter.oversample == 2
    assert jitter.kernel.shape == (4, 4)
    assert np.allclose(jitter.kernel, expected)


def test_legacy_optical_system_grid_contract():
    with pytest.warns(DeprecationWarning):
        system = dl.LayeredOpticalSystem(8, 2.0, [])

    assert system.wf_npixels == 8
    assert np.isclose(system.diameter, 2.0)
    assert system.grid.unit == "m"


def test_legacy_source_normalisation():
    with pytest.warns(DeprecationWarning):
        source = dl.PointSource([1.0, 2.0], weights=[1.0, 3.0])

    _, weights = source.spectrum_params()
    assert np.allclose(weights, np.array([0.25, 0.75]))


def test_legacy_point_sources_contract():
    with pytest.warns(DeprecationWarning):
        source = dl.PointSources(
            [1.0, 2.0],
            position=[[0.0, 0.0], [0.1, 0.2]],
            flux=[2.0, 3.0],
            weights=[[1.0, 1.0], [1.0, 3.0]],
        )

    parameters = source.params()
    assert parameters["position"].shape == (2, 2)
    assert np.allclose(parameters["flux"], np.array([2.0, 3.0]))
    assert np.allclose(parameters["weights"].sum(-1), np.ones(2))


@pytest.mark.parametrize(
    "name",
    [
        "ASMPropagator",
        "BaseDetector",
        "BaseOpticalSystem",
        "BaseSpectrum",
        "BasisLayer",
        "ParametricLayeredOpticalSystem",
        "Scene",
    ],
)
def test_removed_contract_migration_errors(name):
    with pytest.raises(TypeError, match="cannot be translated") as record:
        getattr(dl, name)()

    assert "Use" in str(record.value)


@pytest.mark.parametrize(
    "module, name",
    [
        ("coordinates", "CoordSpec"),
        ("detectors", "LayeredDetector"),
        ("instruments", "Instrument"),
        ("optical_systems", "LayeredOpticalSystem"),
        ("psfs", "PSF"),
        ("spectra", "Spectrum"),
        ("wavefronts", "Wavefront"),
    ],
)
def test_legacy_module_paths(module, name):
    imported = importlib.import_module(f"dLux.{module}")

    with pytest.warns(DeprecationWarning, match="module"):
        value = getattr(imported, name)

    assert value is not None


def test_legacy_detector_layer_module():
    module = importlib.import_module("dLux.layers.detector_layers")

    with pytest.warns(DeprecationWarning, match="module"):
        detector_layer = module.DetectorLayer
    with pytest.warns(DeprecationWarning, match="module"):
        downsample = module.Downsample(2)

    class CustomLayer(detector_layer):
        def __call__(self, intensity):
            return intensity + 1

    intensity = dl.Intensity(np.ones((4, 4)), dl.GridSpec(4, 0.1))
    assert np.allclose(CustomLayer().apply(intensity).data, 2)
    assert downsample.kernel_size == 2
    assert downsample(intensity).data.shape == (2, 2)
