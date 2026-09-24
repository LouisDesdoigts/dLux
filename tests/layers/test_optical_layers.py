"""Tests for dLux.layers.optical."""

import jax.numpy as np
import pytest
import zodiax as zdx
from abcdLux import lct

import dLux as dl
import dLux.utils as dlu

from tests.helpers import assert_differentiable, assert_jittable, assert_tree_allclose


@pytest.fixture
def wavefront(make_wavefront):
    return make_wavefront()


@pytest.mark.parametrize(
    "kind", [dl.TiltLayer, lambda value: dl.WedgeLayer(value, 1.5)]
)
def test_angular_layers_preserve_expression_operands(kind, wavefront):
    angles = np.asarray([1e-7, -2e-7])
    definition = kind(zdx.Map(x=angles, s=2.0))
    expected = kind(2.0 * angles)(wavefront)
    actual = assert_jittable(lambda layer: layer(wavefront), definition)
    assert np.allclose(actual.phasor, expected.phasor)


@pytest.mark.parametrize(
    "layer",
    [
        dl.Optic(),
        dl.Optic(0.5),
        dl.Optic(0.5, normalise=False),
        dl.Optic(0.5, normalise=True),
        dl.Optic(),
        dl.Optic(opd=1e-7, phase=0.2),
        dl.Optic(transmission=0.5, opd=1e-7, phase=0.2),
        dl.Optic(transmission=0.5, normalise=True),
        dl.TiltLayer([0.1, -0.2], unit="arcsec"),
    ],
)
def test_optical_layer_contract(layer, wavefront):
    output = assert_jittable(layer, wavefront)
    assert output.phasor.shape == wavefront.phasor.shape


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
    unchanged = dl.Optic()(wavefront)
    layer = dl.Optic(0.5)
    attenuated = layer(wavefront)
    disabled = dl.Optic(0.5, normalise=False)(wavefront)
    normalised = dl.Optic(0.5, normalise=True)(wavefront)

    assert layer.normalise is None
    assert np.allclose(unchanged.phasor, wavefront.phasor)
    assert np.allclose(attenuated.power, 0.25 * wavefront.power)
    assert np.allclose(disabled.phasor, attenuated.phasor)
    assert np.allclose(normalised.power, 1)


def test_optic_normalisation_update(wavefront):
    optic = dl.Optic(transmission=0.5)
    normalised = optic.set("normalise", True)
    unconfigured = normalised.set("normalise", None)

    assert optic.normalise is None
    assert normalised.normalise is True
    assert unconfigured.normalise is None
    assert np.allclose(assert_jittable(normalised, wavefront).power, 1)
    assert np.allclose(
        assert_jittable(unconfigured, wavefront).power, 0.25 * wavefront.power
    )


def test_aberrated_layer_contract(wavefront):
    layer = dl.Optic(opd=1e-7, phase=0.2)
    expected = wavefront.add_opd(1e-7).add_phase(0.2)

    assert np.allclose(layer(wavefront).phasor, expected.phasor)


def test_optic_phasor_contract(wavefront):
    optic = dl.Optic(transmission=0.5, opd=1e-7, phase=0.2)
    resolved = optic.resolve(**optic.context(wavefront))

    context = optic.context(wavefront)
    assert np.allclose(context["coordinates"], wavefront.coordinates)
    assert set(context) == {"coordinates", "pixel_scale", "wavelength", "wavelengths"}
    assert np.allclose(context["wavelengths"], wavefront.wavelength)
    assert optic.phasor(wavefront).shape == (1, 1)
    assert np.allclose(optic.phasor(wavefront), resolved.phasor(wavefront))


@pytest.mark.parametrize(
    ("layer", "attribute"),
    [
        (dl.Optic(0.5), "transmission"),
        (dl.Optic(opd=1e-7), "opd"),
        (dl.Optic(phase=0.2), "phase"),
        (dl.Optic(transmission=0.5), "transmission"),
        (dl.Optic(opd=1e-7), "opd"),
        (dl.Optic(phase=0.2), "phase"),
        (dl.TiltLayer([0.1, -0.2]), "angles"),
    ],
)
def test_optical_layer_gradients(layer, attribute, wavefront):
    def apply(value):
        output = layer.set(attribute, value)(wavefront)
        if attribute.split(".")[-1] in ("opd", "phase", "angles"):
            return np.real(output.phasor)
        return output

    assert_differentiable(apply, layer.get(attribute))


def test_parametric_optic(wavefront):
    optic = dl.Optic(
        transmission=0.5,
        opd=dlu.Basis(dlu.Zernike(nolls=[4], diameter=0.5), coeffs=[1e-7]),
    )

    resolved = optic.resolve(**optic.context(wavefront))
    assert isinstance(resolved, dl.Optic)
    assert isinstance(optic.opd, zdx.Expression)
    assert not isinstance(resolved.opd, zdx.Expression)
    assert_jittable(optic, wavefront)
    assert_differentiable(
        lambda coeffs: np.real(optic.set("opd.coeffs", coeffs)(wavefront).phasor),
        optic.opd.coeffs,
    )


@pytest.mark.parametrize(
    "layer",
    [
        dl.Optic(np.linspace(0.5, 1.0, 64).reshape(8, 8)),
        dl.Optic(opd=np.ones((8, 8)) * 1e-7, phase=0.2),
        dl.Optic(transmission=0.5, opd=1e-7, phase=0.2),
        dl.TiltLayer([0.1, -0.2], unit="arcsec"),
        dl.Optic(thickness=np.ones((8, 8)) * 1e-3, n=1.5),
        dl.WedgeLayer([1e-6, -2e-6], 1.5),
    ],
)
def test_optical_layers_preserve_leading_axes(layer, make_grid):
    grid = make_grid(n=(8, 8), d=(0.1, 0.1), c=(0.2, -0.1))
    wavefront = dl.Wavefront(1e-6, grid, np.ones((2, 3, 8, 8), complex))
    output = assert_jittable(layer, wavefront)

    assert output.phasor.shape == wavefront.phasor.shape
    assert output.grid.n == wavefront.grid.n
    assert np.allclose(output.grid.d, wavefront.grid.d)
    assert np.allclose(output.grid.c, wavefront.grid.c)


def test_tilt_validation():
    assert dl.TiltLayer([0.0, 0.0], unit="arcseconds").unit == "arcsec"

    with pytest.raises(ValueError, match="shape"):
        dl.TiltLayer([1])
    with pytest.raises(ValueError, match="tilt unit"):
        dl.TiltLayer([0.0, 0.0], unit="m")


class TestSoummerFPM:
    @pytest.fixture
    def focal_spec(self):
        return dl.Grid(n=(6, 8), d=(2e-7, 3e-7), unit="rad")

    def test_complex_optic_matches_direct_mft(self, focal_spec, make_wavefront):
        wavefront = make_wavefront()
        phase = np.linspace(0.0, np.pi, 48).reshape(8, 6)
        optic = dl.Optic(transmission=0.7, phase=phase)
        propagator = dlu.Propagator(focal_spec)
        layer = dl.SoummerFPMLayer(optic, propagator)

        focal = lct.lct_prop(
            wavefront.phasor,
            wavefront.axes,
            focal_spec.axes,
            wavefront.wavelength,
            propagator.resolve(
                wavelength=wavefront.wavelength, grid_in=wavefront.grid
            ).matrix,
        )
        modified = focal * 0.7 * np.exp(1j * phase)
        difference = lct.lct_prop(
            focal - modified,
            focal_spec.axes,
            wavefront.axes,
            wavefront.wavelength,
            propagator.set(inverse=True)
            .resolve(wavelength=wavefront.wavelength, grid_in=focal_spec)
            .matrix,
        )
        expected = wavefront.set(phasor=wavefront.phasor - difference)

        output = assert_jittable(layer, wavefront, rtol=1e-5, atol=1e-5)
        assert_tree_allclose(output, expected, rtol=1e-5, atol=1e-5)

    def test_parametric_optic(self, focal_spec, make_wavefront):
        wavefront = make_wavefront()
        optic = dl.Optic(transmission=dlu.Complement(dlu.Circle(diameter=8e-7)))
        layer = dl.SoummerFPMLayer(optic, dlu.Propagator(focal_spec))

        assert isinstance(layer.optic.transmission, zdx.Expression)
        assert_jittable(layer, wavefront, rtol=1e-5, atol=1e-5)
        assert_differentiable(
            lambda diameter: layer.set("optic.transmission.shape.diameter", diameter)(
                wavefront
            ),
            layer.optic.transmission.shape.diameter,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_polarising_optic(self, focal_spec, make_wavefront):
        wavefront = make_wavefront()
        layer = dl.SoummerFPMLayer(
            dl.Optic(jones=dlu.Jones.LinearPolariser(np.pi / 4)),
            dlu.Propagator(focal_spec),
        )

        output = assert_jittable(layer, wavefront, rtol=1e-5, atol=1e-5)

        assert isinstance(output, dl.Wavefront)
        output = output.materialise()
        assert output.phasor.shape == (2, 2, *wavefront.phasor.shape)

    def test_focal_length_gradient(self, make_wavefront):
        wavefront = make_wavefront()
        grid = dl.Grid(n=(6, 8), d=(2e-6, 3e-6), unit="m")
        layer = dl.SoummerFPMLayer(
            dl.Optic(transmission=0.5), dlu.Propagator(grid, focal_length=2.0)
        )

        assert_differentiable(
            lambda value: layer.set("propagator.focal_length", value)(wavefront),
            layer.propagator.focal_length,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_vectorised_wavefronts(self, focal_spec, make_wavefront):
        layer = dl.SoummerFPMLayer(
            dl.Optic(transmission=0.5), dlu.Propagator(focal_spec)
        )
        chromatic = make_wavefront(wavelength=np.asarray([1e-6, 1.1e-6]))
        polarised = make_wavefront(polarised=True)

        assert_jittable(layer, chromatic, rtol=1e-5, atol=1e-5)
        assert_jittable(layer, polarised, rtol=1e-5, atol=1e-5)

    def test_validation(self, focal_spec, make_wavefront):
        with pytest.raises(TypeError, match="callable optical layer"):
            dl.SoummerFPMLayer(np.ones(focal_spec.shape), dlu.Propagator(focal_spec))
        with pytest.raises(TypeError, match="Propagator"):
            dl.SoummerFPMLayer(dl.Optic(), dlu.FreeSpace(0.1))
        with pytest.raises(TypeError, match="Propagator"):
            dl.SoummerFPMLayer(dl.Optic(), dlu.FFTPropagator(dl.ResizeSpec()))
        with pytest.raises(ValueError, match="forward"):
            dl.SoummerFPMLayer(
                dl.Optic(), dlu.Propagator(focal_spec.set(unit="m"), inverse=True)
            )


def test_soummer_requires_an_optical_operation():
    with pytest.raises(TypeError, match="optic"):
        dl.SoummerFPMLayer(1, dlu.Propagator(dl.Grid(3, 1e-7, unit="rad")))


def test_soummer_rejects_nonconjugate_propagation():
    propagation = dlu.Propagator(dl.Grid(3, 1e-7, unit="rad"), defocus=0.1)
    with pytest.raises(ValueError, match="undefocused conjugate"):
        dl.SoummerFPMLayer(dl.Optic(), propagation)
