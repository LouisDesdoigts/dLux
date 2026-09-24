"""Default representations expose model structure without incidental settings."""

import jax.numpy as np
import pytest

import dLux as dl
import dLux.utils as dlu


@pytest.mark.parametrize(
    "module",
    [
        dlu.ResizeSpec(),
        dlu.FFTSpec(),
        dlu.Grid(),
        dlu.Affine(),
        dl.Image(np.ones((2, 2)), dlu.Grid(n=2, d=1)),
        dl.Optic(),
        dl.SparseOptic([[0.0, 0.0]]),
        dlu.Jones.HalfWavePlate(),
        dl.PropagatorLayer(dlu.Grid(n=2, d=1, unit="rad")),
        dlu.Interp(),
        dl.Source(),
        dl.BinarySource(),
    ],
)
def test_none_defaults_are_omitted(module):
    assert "=None" not in repr(module)


@pytest.mark.parametrize(
    ("module", "field"),
    [
        (dlu.Grid(n=2), "n"),
        (dl.Optic(transmission=0.5), "transmission"),
        (dl.Optic(opd=1e-7), "opd"),
        (dl.Source(flux=2.0), "flux"),
        (dl.ResolvedSource(profile=dlu.Gaussian(1e-7)), "profile"),
    ],
)
def test_configured_fields_remain_visible(module, field):
    assert f"{field}=" in repr(module)


def test_binary_shows_its_astrometry_and_brightness_definitions():
    text = repr(dl.BinarySource(separation=1e-7, primary_flux=100, contrast=0.2))
    assert "positions=Binary(" in text
    assert "flux=ContrastFlux(" in text
    assert "units=" not in text


@pytest.mark.parametrize("module", [dl.Optic(), dl.SparseOptic([[0.0, 0.0]])])
def test_unconfigured_normalisation_is_omitted(module):
    assert module.normalise is None
    assert "normalise=" not in repr(module)


@pytest.mark.parametrize("normalise", [False, True])
def test_explicit_normalisation_remains_visible(normalise):
    module = dl.Optic(normalise=normalise)
    assert f"normalise={normalise}" in repr(module)


def test_shape_default_and_nondefault_options():
    assert "invert=" not in repr(dlu.Circle(1.0))
    assert "invert=True" in repr(dlu.Circle(1.0, invert=True))


@pytest.mark.parametrize("method", ["cubic2", "linear"])
@pytest.mark.parametrize(
    "constructor",
    [
        dlu.Interp,
        dlu.Spectrum,
        dlu.Filter,
        lambda **kwargs: dlu.InterpolatedIndex([5e-7, 7e-7], [1.5, 1.4], **kwargs),
    ],
)
def test_interpolation_method_is_always_visible(constructor, method):
    assert f"method='{method}'" in repr(constructor(method=method))
