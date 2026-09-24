"""Canonical public names must not be shadowed by compatibility exports."""

import dLux as dl
import dLux.utils as dlu
from dLux.sources import PointSource, ResolvedSource, Scene
from dLux.layers.optical import PropagatorLayer
from dLux.utils.grids.specs import Spec


def test_sampling_base_is_the_canonical_spec():
    assert dl.Spec is dlu.Spec is Spec
    assert isinstance(dlu.Grid(4, d=1), dl.Spec)


def test_current_source_names_are_not_legacy_adapters():
    assert dl.PointSource is PointSource
    assert dl.ResolvedSource is ResolvedSource
    assert dl.Scene is Scene


def test_source_module_import_forms_agree():
    import importlib
    import dLux.sources as sources

    assert sources is dl.sources is dl.systems.sources
    assert importlib.import_module("dLux.sources") is sources
    assert sources.PointSource is PointSource
    assert sources.BinarySource is dl.BinarySource
    assert sources is not dl.compat.sources


def test_numerical_source_definitions_belong_to_utilities():
    assert dlu.Point.__module__ == "dLux.utils.astrometry"
    assert dlu.Binary.__module__ == "dLux.utils.astrometry"
    assert dlu.Companions.__module__ == "dLux.utils.astrometry"
    assert dlu.Blackbody.__module__ == "dLux.utils.photometry.spectral"
    assert dlu.ContrastFlux.__module__ == "dLux.utils.photometry.flux"


def test_propagation_layer_inherits_flat_numerical_definition():
    assert dl.PropagatorLayer is dl.layers.PropagatorLayer is PropagatorLayer
    grid = dlu.Grid(4, d=1e-7, unit="rad")
    layer = dl.PropagatorLayer(grid)
    assert layer.grid_out is grid
    assert isinstance(layer, dlu.Propagator)
    assert "propagator" not in type(layer).__dataclass_fields__
    assert "PropagatorLayer" not in vars(dlu)
    for name in (
        "BasePropagator",
        "Propagator",
        "FFTPropagator",
        "SpectralPropagator",
        "FreeSpace",
    ):
        assert getattr(dlu, name).__module__ == "dLux.utils.propagation.propagators"
        assert getattr(dl, name) is getattr(dlu, name)
        assert name not in vars(dl.layers)


def test_current_zernike_is_not_shadowed_by_a_removed_compatibility_name():
    assert dl.Zernike is dlu.Zernike
    assert dl.Zernike.__module__ == "dLux.utils.optical.zernikes"


def test_public_exports_are_unique_and_importable():
    for module in (
        dl,
        dlu,
        dl.fields,
        dl.layers,
        dl.prebuilt,
        dl.systems,
        dlu.optical,
        dlu.propagation,
    ):
        assert len(module.__all__) == len(set(module.__all__))
        assert all(hasattr(module, name) for name in module.__all__)


def test_compatibility_facades_do_not_modify_canonical_exports():
    assert "rad2deg" not in dlu.numerical.units.__all__
    assert "l1_norm" not in dlu.numerical.norms.__all__
    assert "paste" not in dlu.grids.array_ops.__all__
    assert dlu.units.unit_factor is dlu.numerical.units.unit_factor
    assert dlu.array_ops.paste is dlu.grids.paste
    assert dlu.units is not dlu.numerical.units


def test_numerical_operations_and_field_layers_have_distinct_names():
    from dLux.utils.propagation import FFT, MFT, Resize
    from dLux.layers.unified import ResizeLayer

    assert dl.Resize is dlu.Resize is Resize
    assert dl.ResizeLayer is dl.layers.ResizeLayer is ResizeLayer
    assert dl.FFT is dlu.FFT is FFT
    assert dl.MFT is dlu.MFT is MFT
    assert dl.compat.Resize is ResizeLayer
    assert dl.compat.FFT is not FFT
    assert dl.compat.MFT is not MFT
    assert dl.compat.utils.FFT is not FFT
    assert dl.compat.utils.MFT is not MFT


def test_concrete_operation_layers_use_the_layer_suffix():
    names = (
        "Resize",
        "Downsample",
        "Flip",
        "Interpolate",
        "Normalise",
        "Lambda",
        "Sensitivity",
        "Convolve",
        "Bias",
        "Gain",
        "Saturation",
        "Tilt",
        "Wedge",
        "SoummerFPM",
    )
    for name in names:
        cls = getattr(dl, name + "Layer")
        assert cls is getattr(dl.layers, name + "Layer")
        assert cls.__name__ == name + "Layer"
        assert name not in vars(dl.layers)


def test_all_utility_exports_are_available_in_main():
    # Current core helpers win over their legacy utility copies. The old
    # coordinates module keeps its released import path rather than being shadowed.
    overrides = {
        "coordinates": dl.compat.coordinates,
        "segmented_hex_cens": dl.prebuilt.segmented_hex_cens,
        "non_redundant_support": dl.prebuilt.non_redundant_support,
    }
    assert set(dlu.__all__) <= set(dl.__all__)
    for name in dlu.__all__:
        assert getattr(dl, name) is overrides.get(name, getattr(dlu, name))
    assert dl.optical is dlu.optical
    assert dl.layers.optical is not dl.optical


def test_main_spectra_are_current_definitions_without_deprecation_warnings():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        bandpass = dl.BandPass(min=500, max=700, n=4, unit="nm")
        spectrum = dl.Spectrum(bandpass=bandpass, flux=8)
        polynomial = dl.PolySpectrum(bandpass, coeffs=[0.1])
        model = dl.Photometry(spectrum=spectrum)
    assert dl.Spectrum is dlu.Spectrum
    assert dl.PolySpectrum is dlu.PolySpectrum
    assert dl.Filter is dlu.Filter
    assert dl.Blackbody is dlu.Blackbody
    assert type(polynomial) is dlu.PolySpectrum
    assert model.spectrum is spectrum
    assert dl.compat.Spectrum is not dl.Spectrum
    assert dl.compat.PolySpectrum is not dl.PolySpectrum


def test_main_module_exports_preserve_released_coordinate_imports():
    import importlib

    assert importlib.import_module("dLux.coordinates") is dl.coordinates
    assert dl.coordinates is dl.compat.coordinates
