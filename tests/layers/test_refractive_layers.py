"""Refractive material and wedge behaviour through ordinary Optic layers."""

import jax.numpy as np
import pytest

import dLux as dl
import dLux.utils as dlu

from tests.helpers import assert_differentiable, assert_jittable


@pytest.fixture
def chromatic_wavefront(make_wavefront):
    return make_wavefront(wavelength=np.array([0.5e-6, 1e-6, 2e-6]))


@pytest.mark.parametrize(
    "layer",
    [
        dl.Optic(thickness=1e-7, n=1.5),
        dl.Optic(thickness=1e-7, n=dlu.CauchyIndex([1.5, 0.01])),
        dl.WedgeLayer([1e-9, -2e-9], 1.5),
        dl.WedgeLayer([1e-9, 0.0], dlu.InterpolatedIndex([0.5e-6, 2e-6], [1.6, 1.5])),
    ],
)
def test_refractive_layer_contract(layer, chromatic_wavefront):
    output = assert_jittable(layer, chromatic_wavefront)
    assert output.phasor.shape == chromatic_wavefront.phasor.shape


def test_lens_applies_residual_material_opd(make_wavefront):
    wavefront = make_wavefront()
    thickness = np.ones(wavefront.grid.shape) * 1e-7
    output = dl.Optic(thickness=thickness, n=1.5)(wavefront)
    expected = wavefront.add_opd(0.5 * thickness)

    assert np.allclose(output.phasor, expected.phasor)


def test_lens_is_chromatic(chromatic_wavefront):
    index = dlu.CauchyIndex([1.5, 0.01])
    output = dl.Optic(thickness=1e-7, n=index)(chromatic_wavefront)

    assert output.phasor.shape == chromatic_wavefront.phasor.shape
    assert not np.allclose(output.phasor[0], output.phasor[-1])


@pytest.mark.parametrize(
    ("layer", "path"),
    [
        (dl.Optic(thickness=1e-7, n=1.5), "thickness"),
        (dl.Optic(thickness=1e-7, n=1.5), "n"),
        (
            dl.Optic(thickness=1e-7, n=dlu.CauchyIndex([1.5, 0.01])),
            "n.coeffs",
        ),
        (dl.WedgeLayer([1e-9, -2e-9], 1.5), "angle"),
        (dl.WedgeLayer([1e-9, -2e-9], 1.5), "n"),
    ],
)
def test_refractive_layer_gradients(layer, path, chromatic_wavefront):
    value = layer.get(path)

    assert_differentiable(
        lambda replacement: np.real(
            layer.set(path, replacement)(chromatic_wavefront).phasor
        ),
        value,
    )


@pytest.mark.parametrize("angle", [1e-3, [1e-3], [1e-3, 0, 0]])
def test_wedge_validation(angle):
    with pytest.raises(ValueError, match="shape"):
        dl.WedgeLayer(angle, n=1.5)


def test_wedge_accepts_spatial_refractive_index(make_wavefront):
    field = make_wavefront()
    index = np.linspace(1.4, 1.6, 64).reshape(8, 8)
    wedge = dl.WedgeLayer([1e-6, -2e-6], index)
    result = assert_jittable(wedge, field)
    x, y = field.coordinates
    thickness = x * np.tan(1e-6) + y * np.tan(-2e-6)
    expected = field.add_opd((index - 1) * thickness)
    assert np.allclose(result.phasor, expected.phasor)
