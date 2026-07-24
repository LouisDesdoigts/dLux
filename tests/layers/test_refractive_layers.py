"""Tests for dLux.layers.refractive_layers."""

import jax.numpy as np
import pytest

import dLux as dl

from tests.helpers import assert_differentiable, assert_jittable


@pytest.fixture
def chromatic_wavefront(make_wavefront):
    return make_wavefront(wavelength=np.array([0.5e-6, 1e-6, 2e-6]))


@pytest.mark.parametrize(
    "layer",
    [
        dl.Lens(1e-7, 1.5),
        dl.Lens(1e-7, dl.CauchyIndex([1.5, 0.01])),
        dl.Wedge([1e-9, -2e-9], 1.5),
        dl.Wedge(
            [1e-9, 0.0],
            dl.InterpolatedIndex([0.5e-6, 2e-6], [1.6, 1.5]),
            reference_wavelength=1e-6,
        ),
    ],
)
def test_refractive_layer_contract(layer, chromatic_wavefront):
    output = assert_jittable(layer, chromatic_wavefront)
    assert output.phasor.shape == chromatic_wavefront.phasor.shape


def test_lens_applies_residual_material_opd(make_wavefront):
    wavefront = make_wavefront()
    thickness = np.ones(wavefront.spec.shape) * 1e-7
    output = dl.Lens(thickness, n=1.5)(wavefront)
    expected = wavefront.add_opd(0.5 * thickness)

    assert np.allclose(output.phasor, expected.phasor)


def test_lens_is_chromatic(chromatic_wavefront):
    index = dl.CauchyIndex([1.5, 0.01])
    output = dl.Lens(1e-7, index)(chromatic_wavefront)

    assert output.phasor.shape == chromatic_wavefront.phasor.shape
    assert not np.allclose(output.phasor[0], output.phasor[-1])


def test_wedge_reference_removes_common_deviation(make_wavefront):
    wavelength = 1.5e-6
    wavefront = make_wavefront(wavelength=wavelength)
    index = dl.InterpolatedIndex([1e-6, 2e-6], [1.6, 1.5])
    wedge = dl.Wedge(
        [1e-3, 0],
        index,
        reference_wavelength=wavelength,
    )

    assert np.allclose(wedge(wavefront).phasor, wavefront.phasor)


@pytest.mark.parametrize(
    ("layer", "path"),
    [
        (dl.Lens(1e-7, 1.5), "thickness"),
        (dl.Lens(1e-7, 1.5), "n"),
        (dl.Lens(1e-7, dl.CauchyIndex([1.5, 0.01])), "n.coefficients"),
        (dl.Wedge([1e-9, -2e-9], 1.5), "angle"),
        (dl.Wedge([1e-9, -2e-9], 1.5), "n"),
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
        dl.Wedge(angle, n=1.5)
