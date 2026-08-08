"""Tests for deprecated public compatibility contracts."""

import importlib

import jax.numpy as np
import pytest

import dLux as dl


def test_coefficients_constructor_alias():
    basis = np.ones((2, 3, 3))

    with pytest.warns(DeprecationWarning) as record:
        parametric = dl.Basis(basis, coefficients=[1.0, 2.0])

    message = str(record[0].message)
    assert "removed in dLux 0.17.0" in message
    assert "Class(coefficients=value)` -> `Class(coeffs=value)" in message
    assert np.allclose(parametric.coeffs, np.array([1.0, 2.0]))


def test_coefficients_attribute_alias():
    basis = dl.Basis(np.ones((2, 3, 3)), coeffs=[1.0, 2.0])

    with pytest.warns(DeprecationWarning) as record:
        coeffs = basis.coefficients

    message = str(record[0].message)
    assert "removed in dLux 0.17.0" in message
    assert "basis.coefficients` -> `basis.coeffs" in message
    assert np.allclose(coeffs, basis.coeffs)


@pytest.mark.parametrize(
    "legacy, current, kwargs",
    [
        (dl.CoordSpec, dl.GridSpec, {"n": 8, "d": 0.1}),
        (dl.PadSpec, dl.ResizeSpec, {"pad": 2, "crop": 1}),
        (dl.DistortedCoords, dl.DistortCoords, {"order": 2}),
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


def test_legacy_detector_return_contract():
    psf = dl.PSF(np.ones((4, 4)), dl.GridSpec(4, 0.1, unit="rad"))

    with pytest.warns(DeprecationWarning):
        detector = dl.LayeredDetector([])

    assert isinstance(detector(psf), type(psf.data))
    assert isinstance(detector(psf, return_psf=True), dl.PSF)


def test_legacy_optical_system_grid_contract():
    with pytest.warns(DeprecationWarning):
        system = dl.LayeredOpticalSystem(8, 2.0, [])

    assert system.wf_npixels == 8
    assert np.isclose(system.diameter, 2.0)
    assert system.spec.unit == "m"


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
