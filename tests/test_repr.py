"""Tests for concise dLux module representations."""

import jax.numpy as np
import pytest

import dLux as dl


@pytest.mark.parametrize(
    "module",
    [
        dl.ResizeSpec(),
        dl.GridSpec(),
        dl.Affine(),
        dl.Image(np.ones((2, 2)), dl.GridSpec(n=2, d=1)),
        dl.Optic(),
        dl.DynamicOptic(),
        dl.SparseOptic([[0.0, 0.0]]),
        dl.PolarisationLayer(),
        dl.UniformPolarisingOptic(np.eye(2)),
        dl.Fraunhofer(dl.GridSpec(n=2, d=1)),
        dl.DynamicZernikeBasis(js=[1]),
        dl.ZernikeDef(nolls=[1]),
        dl.ApertureBuilder(dl.Circle(1)),
        dl.Source(1e-6),
        dl.BinarySource(1e-6),
    ],
)
def test_none_defaults_are_omitted(module):
    assert "=None" not in repr(module)


@pytest.mark.parametrize(
    ("module", "field"),
    [
        (dl.GridSpec(n=2), "n"),
        (dl.Optic(transmission=0.5), "transmission"),
        (dl.DynamicOptic(opd=1e-7), "opd"),
        (dl.Source(1e-6, flux=2.0), "flux"),
    ],
)
def test_non_default_fields_remain_visible(module, field):
    assert f"{field}=" in repr(module)
