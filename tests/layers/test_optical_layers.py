"""Tests for dLux.layers.optical."""

import jax.numpy as np
import pytest

import dLux as dl

from tests.helpers import assert_differentiable, assert_jittable, assert_tree_allclose


@pytest.fixture
def wavefront(make_wavefront):
    return make_wavefront()


@pytest.mark.parametrize(
    "layer",
    [
        dl.TransmissiveLayer(),
        dl.TransmissiveLayer(0.5),
        dl.TransmissiveLayer(0.5, normalise=True),
        dl.AberratedLayer(),
        dl.AberratedLayer(opd=1e-7, phase=0.2),
        dl.Optic(transmission=0.5, opd=1e-7, phase=0.2),
        dl.Optic(transmission=0.5, normalise=True),
        dl.Tilt([0.1, -0.2], unit="arcsec"),
    ],
)
def test_optical_layer_contract(layer, wavefront):
    output = assert_jittable(layer, wavefront)
    assert output.phasor.shape == wavefront.phasor.shape


def test_layer_alias_and_inheritance(wavefront):
    layer = dl.Optic(transmission=0.5, opd=1e-7, phase=0.2)

    assert isinstance(layer, dl.TransmissiveLayer)
    assert isinstance(layer, dl.AberratedLayer)


def test_monochromatic_layer_mapping(make_grid):
    class MonochromaticLayer(dl.OpticalLayer):
        def apply_mono(self, wavefront):
            if wavefront.batch_ndim:
                raise ValueError("Expected a monochromatic wavefront.")
            return wavefront.add_phase(wavefront.wavelength / 1e-6)

    wavefront = dl.Wavefront([0.9e-6, 1.1e-6], make_grid())
    layer = MonochromaticLayer()
    output = layer(wavefront)

    assert output.phasor.shape == wavefront.phasor.shape
    assert np.allclose(output.phasor, layer.apply(wavefront).phasor)
    assert np.allclose(output.phase[:, 0, 0], np.array([0.9, 1.1]))

    monochromatic = dl.Wavefront(1e-6, make_grid())
    assert np.allclose(
        layer(monochromatic).phasor,
        layer.apply_mono(monochromatic).phasor,
    )


def test_transmissive_layer_contract(wavefront):
    unchanged = dl.TransmissiveLayer()(wavefront)
    attenuated = dl.TransmissiveLayer(0.5)(wavefront)
    normalised = dl.TransmissiveLayer(0.5, normalise=True)(wavefront)

    assert np.allclose(unchanged.phasor, wavefront.phasor)
    assert np.allclose(attenuated.power, 0.25 * wavefront.power)
    assert np.allclose(normalised.power, 1)


def test_aberrated_layer_contract(wavefront):
    layer = dl.AberratedLayer(opd=1e-7, phase=0.2)
    expected = wavefront.add_opd(1e-7).add_phase(0.2)

    assert np.allclose(layer(wavefront).phasor, expected.phasor)


def test_optic_phasor_contract(wavefront):
    optic = dl.Optic(transmission=0.5, opd=1e-7, phase=0.2)
    resolved = optic.resolve(wavefront=wavefront)

    assert optic.context(wavefront) == {"wavefront": wavefront}
    assert optic.phasor(wavefront).shape == (1, 1)
    assert np.allclose(optic.phasor(wavefront), resolved.phasor(wavefront))


@pytest.mark.parametrize(
    ("layer", "attribute"),
    [
        (dl.TransmissiveLayer(0.5), "transmission"),
        (dl.AberratedLayer(opd=1e-7), "opd"),
        (dl.AberratedLayer(phase=0.2), "phase"),
        (dl.Optic(transmission=0.5), "transmission"),
        (dl.Optic(opd=1e-7), "opd"),
        (dl.Optic(phase=0.2), "phase"),
        (dl.Tilt([0.1, -0.2]), "angles"),
    ],
)
def test_optical_layer_gradients(layer, attribute, wavefront):
    def apply(value):
        output = layer.set(attribute, value)(wavefront)
        if attribute in ("opd", "phase", "angles"):
            return np.real(output.phasor)
        return output

    assert_differentiable(apply, getattr(layer, attribute))


def test_parametric_optic(wavefront):
    optic = dl.Optic(
        transmission=0.5,
        opd=dl.DynamicZernikeBasis(js=[4], coeffs=[1e-7], diameter=0.5),
    )

    resolved = optic.resolve(wavefront=wavefront)
    assert isinstance(resolved, dl.Optic)
    assert isinstance(optic.opd, dl.Parametric)
    assert not isinstance(resolved.opd, dl.Parametric)
    assert_jittable(optic, wavefront)
    assert_differentiable(
        lambda coeffs: np.real(
            optic.set("opd.coeffs", coeffs)(wavefront).phasor
        ),
        optic.opd.coeffs,
    )


@pytest.mark.parametrize(
    "layer",
    [
        dl.TransmissiveLayer(np.linspace(0.5, 1.0, 64).reshape(8, 8)),
        dl.AberratedLayer(opd=np.ones((8, 8)) * 1e-7, phase=0.2),
        dl.Optic(transmission=0.5, opd=1e-7, phase=0.2),
        dl.Tilt([0.1, -0.2], unit="arcsec"),
        dl.RefractiveOptic(np.ones((8, 8)) * 1e-3, 1.5),
        dl.Wedge([1e-6, -2e-6], 1.5),
    ],
)
def test_optical_layers_preserve_leading_axes(layer, make_grid):
    grid = make_grid(n=(8, 8), d=(0.1, 0.1), c=(0.2, -0.1))
    wavefront = dl.Wavefront(1e-6, grid, np.ones((2, 3, 8, 8), complex))
    output = assert_jittable(layer, wavefront)

    assert output.phasor.shape == wavefront.phasor.shape
    assert output.grid == wavefront.grid


def test_tilt_validation():
    with pytest.raises(ValueError, match="shape"):
        dl.Tilt([1])


class TestSoummerFPM:
    @pytest.fixture
    def focal_spec(self):
        return dl.GridSpec(n=(6, 8), d=(2e-7, 3e-7), unit="rad")

    def test_complex_optic_matches_direct_mft(self, focal_spec, make_wavefront):
        wavefront = make_wavefront()
        phase = np.linspace(0.0, np.pi, 48).reshape(8, 6)
        optic = dl.Optic(transmission=0.7, phase=phase)
        layer = dl.SoummerFPM(optic, dl.Fraunhofer(focal_spec))

        focal = dl.utils.MFT(
            wavefront.phasor,
            wavefront.wavelength,
            wavefront.axes,
            focal_spec.axes,
        )
        modified = focal * 0.7 * np.exp(1j * phase)
        difference = dl.utils.MFT(
            focal - modified,
            wavefront.wavelength,
            focal_spec.axes,
            wavefront.axes,
            inverse=True,
        )
        expected = wavefront.set(phasor=wavefront.phasor - difference)

        output = assert_jittable(layer, wavefront, rtol=1e-5, atol=1e-5)
        assert_tree_allclose(output, expected, rtol=1e-5, atol=1e-5)

    def test_parametric_optic(self, focal_spec, make_wavefront):
        wavefront = make_wavefront()
        optic = dl.Optic(transmission=dl.Complement(dl.Circle(diameter=8e-7)))
        layer = dl.SoummerFPM(optic, dl.Fraunhofer(focal_spec))

        assert isinstance(layer.optic.transmission, dl.Parametric)
        assert_jittable(layer, wavefront, rtol=1e-5, atol=1e-5)
        assert_differentiable(
            lambda diameter: layer.set(
                "optic.transmission.shape.diameter", diameter
            )(wavefront),
            layer.optic.transmission.shape.diameter,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_polarising_optic(self, focal_spec, make_wavefront):
        wavefront = make_wavefront()
        layer = dl.SoummerFPM(
            dl.LinearPolariser(np.pi / 4), dl.Fraunhofer(focal_spec)
        )

        output = assert_jittable(layer, wavefront, rtol=1e-5, atol=1e-5)

        assert isinstance(output, dl.PolarisedWavefront)
        assert output.phasor.shape == (2, 2, *wavefront.phasor.shape)

    def test_focal_length_gradient(self, make_wavefront):
        wavefront = make_wavefront()
        grid = dl.GridSpec(n=(6, 8), d=(2e-6, 3e-6), unit="m")
        layer = dl.SoummerFPM(
            dl.Optic(transmission=0.5), dl.Fraunhofer(grid, focal_length=2.0)
        )

        assert_differentiable(
            lambda value: layer.set("propagator.focal_length", value)(wavefront),
            layer.propagator.focal_length,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_vectorised_wavefronts(self, focal_spec, make_wavefront):
        layer = dl.SoummerFPM(
            dl.Optic(transmission=0.5), dl.Fraunhofer(focal_spec)
        )
        chromatic = make_wavefront(wavelength=np.asarray([1e-6, 1.1e-6]))
        polarised = make_wavefront(polarised=True)

        assert_jittable(layer, chromatic, rtol=1e-5, atol=1e-5)
        assert_jittable(layer, polarised, rtol=1e-5, atol=1e-5)

    def test_validation(self, focal_spec, make_wavefront):
        with pytest.raises(TypeError, match="BaseOpticalLayer"):
            dl.SoummerFPM(np.ones(focal_spec.shape), dl.Fraunhofer(focal_spec))
        with pytest.raises(TypeError, match="Fraunhofer"):
            dl.SoummerFPM(dl.Optic(), dl.FreeSpace(0.1))
        with pytest.raises(ValueError, match="MFT"):
            dl.SoummerFPM(
                dl.Optic(), dl.Fraunhofer(dl.ResizeSpec(), method="fft")
            )
        with pytest.raises(ValueError, match="forward"):
            dl.SoummerFPM(dl.Optic(), dl.Fraunhofer(focal_spec, inverse=True))
