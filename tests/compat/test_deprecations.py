"""Released interfaces have explicit ownership and do not replace current models."""

import importlib

import jax.numpy as np
import pytest

import dLux as dl
from dLux import compat as old
import dLux.utils as dlu


def test_from_phasor_retains_its_current_constructor_and_warns():
    grid = dl.Grid(4, 0.1)
    phasor = np.ones((4, 4), complex)
    with pytest.warns(DeprecationWarning, match="from_phasor"):
        value = dl.Wavefront.from_phasor(phasor, 500e-9, grid)
    assert type(value) is dl.Wavefront
    assert np.array_equal(value.phasor, phasor)


@pytest.mark.parametrize(
    "name",
    (
        "CoordSpec",
        "PadSpec",
        "LayeredDetector",
        "AngularOpticalSystem",
        "CircularAperture",
    ),
)
def test_missing_core_names_warn_and_resolve_to_released_objects(name):
    with pytest.warns(DeprecationWarning, match="compat"):
        retained = getattr(dl, name)
    assert retained is getattr(old, name)


@pytest.mark.parametrize("name", ("BasisOptic", "MFT", "Rotate", "CircularAperture"))
def test_missing_layer_names_warn_without_replacing_current_layers(name):
    with pytest.warns(DeprecationWarning, match="compat"):
        retained = getattr(dl.layers, name)
    assert retained is getattr(old.layers, name)


@pytest.mark.parametrize("name", ("PointSource", "ResolvedSource", "Zernike"))
def test_conflicting_contracts_remain_distinct(name):
    assert getattr(dl, name) is not getattr(old, name)
    assert "compat" not in getattr(dl, name).__module__


@pytest.mark.parametrize(
    "path",
    (
        "coordinates",
        "wavefronts",
        "psfs",
        "spectra",
        "optical_systems",
        "detectors",
        "instruments",
        "layers.optics",
        "layers.apertures",
        "layers.detector_layers",
    ),
)
def test_released_module_paths_use_the_compatibility_implementation(path):
    retained = importlib.import_module(f"dLux.{path}")
    explicit = importlib.import_module(f"dLux.compat.{path}")
    assert retained is explicit


def test_released_psf_retains_pixel_spacing_and_immutable_updates():
    psf = old.PSF(np.ones((4, 4)), 0.1)
    changed = psf.set(array=psf.data * 2)
    assert np.array_equal(changed.pixel_scale, psf.pixel_scale)
    assert np.all(psf.data == 1)
    assert np.all(changed.data == 2)
    assert dl.PSF(np.ones((4, 4)), dl.Grid(4, 0.1)).grid.n == (4, 4)


def test_released_sources_and_detectors_form_a_complete_model():
    optics = old.AngularOpticalSystem(8, 1.0, [old.Optic(normalise=True)], 8, 0.1)
    source = old.PointSource([500e-9, 600e-9], flux=3.0)
    detector = old.LayeredDetector([old.AddConstant(2.0)])
    telescope = old.Telescope(optics, source, detector)
    image = telescope.model()
    assert isinstance(image, dl.Image)
    assert np.allclose(image.data, optics.model(source) + 2)


@pytest.mark.parametrize("name", ("Wavefront", "PSF"))
def test_fields_are_shared_classes(name):
    assert getattr(dl, name) is getattr(old, name)


def test_current_and_released_geometry_use_their_own_contracts():
    coordinates = np.array([[[0.0, 0.75, 1.25]], [[0.0, 0.0, 0.0]]])
    assert np.array_equal(old.utils.circle(coordinates, radius=1), [[1, 1, 0]])
    assert np.array_equal(
        dlu.Circle(diameter=1).resolve(coordinates=coordinates), [[1, 0, 0]]
    )


def test_current_coordinate_conversion_preserves_batches():
    coordinates = np.ones((3, 2, 4, 4))
    assert dlu.cart2polar(coordinates).shape == coordinates.shape
    assert old.utils.cart2polar(np.array([3.0, 4.0])).shape == (2,)


def test_unknown_public_names_raise_attribute_errors():
    for module in (dl, dl.layers):
        with pytest.raises(AttributeError):
            getattr(module, "not_a_dlux_class")


def test_current_import_does_not_require_the_optional_backend():
    import os
    from pathlib import Path
    import subprocess
    import sys

    program = """
import importlib.abc
import sys
class MissingBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "abcdLux" or fullname.startswith("abcdLux."):
            raise ModuleNotFoundError("abcdLux is unavailable", name="abcdLux")
sys.meta_path.insert(0, MissingBackend())
import dLux
assert dLux.Wavefront.__module__ == "dLux.fields.wavefront"
assert "abcdLux" not in sys.modules
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    subprocess.run([sys.executable, "-c", program], env=environment, check=True)


def test_released_optic_adapts_broadcasting_without_duplicating_fields():
    from dataclasses import fields

    assert issubclass(old.Optic, dl.Optic)
    assert {field.name for field in fields(old.Optic)} == {
        field.name for field in fields(dl.Optic)
    }
