"""Tests for dLux.layers.propagation_layers."""

import jax.numpy as np
import pytest

import dLux as dl

from tests.helpers import (
    assert_differentiable,
    assert_jittable,
    assert_tree_allclose,
)


@pytest.fixture
def angular_spec():
    return dl.CoordSpec(n=(6, 8), d=(2e-7, 3e-7), unit="rad")


@pytest.fixture
def physical_spec():
    return dl.CoordSpec(n=(6, 8), d=(2e-6, 3e-6), unit="m")


class TestPropagation:
    @pytest.mark.parametrize(
        "make_layer",
        [
            lambda angular, physical: dl.Fraunhofer(angular, method="mft"),
            lambda angular, physical: dl.Fraunhofer(
                physical, focal_length=2.0, method="mft"
            ),
            lambda angular, physical: dl.Fraunhofer(
                dl.PadSpec(pad=2, crop=2), method="fft"
            ),
            lambda angular, physical: dl.Fresnel(
                physical, defocus=0.1, focal_length=2.0, method="mft"
            ),
            lambda angular, physical: dl.Fresnel(
                physical, defocus=0.1, focal_length=2.0, method="lct"
            ),
            lambda angular, physical: dl.Fresnel(
                dl.PadSpec(), defocus=0.1, focal_length=2.0, method="fft"
            ),
            lambda angular, physical: dl.ABCDPropagator(
                [dl.ABCDFreeSpace(0.1)], physical, method="lct"
            ),
            lambda angular, physical: dl.ABCDPropagator(
                [dl.ABCDConjugatePlane(2.0)], dl.PadSpec(), method="fft"
            ),
            lambda angular, physical: dl.ASM(0.1),
        ],
    )
    def test_propagator_contract(
        self, make_layer, angular_spec, physical_spec, make_wavefront
    ):
        layer = make_layer(angular_spec, physical_spec)
        assert_jittable(layer, make_wavefront(), rtol=1e-5, atol=1e-5)

    def test_vectorised_wavefronts(self, angular_spec, make_wavefront):
        layer = dl.Fraunhofer(angular_spec)

        chromatic = make_wavefront(wavelength=np.asarray([1e-6, 1.1e-6]))
        polarised = make_wavefront(polarised=True)

        assert_jittable(layer, chromatic, rtol=1e-5, atol=1e-5)
        assert_jittable(layer, polarised, rtol=1e-5, atol=1e-5)

    def test_output_sampling(self, angular_spec, make_wavefront):
        requested = dl.Fraunhofer(angular_spec)(make_wavefront())
        assert_tree_allclose(requested.spec, angular_spec)

        native = dl.Fraunhofer(dl.PadSpec(), method="fft")(make_wavefront())
        assert native.spec.unit == "rad"
        assert native.spec.n == (8, 8)

    def test_field_gradient(self, angular_spec, make_wavefront):
        wavefront = make_wavefront()
        layer = dl.Fraunhofer(angular_spec)

        assert_differentiable(
            lambda phasor: layer(wavefront.set(phasor=phasor)),
            wavefront.phasor,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_physical_parameter_gradients(self, physical_spec, make_wavefront):
        wavefront = make_wavefront()

        fraunhofer = dl.Fraunhofer(physical_spec, focal_length=2.0)
        assert_differentiable(
            lambda focal_length: fraunhofer.set(focal_length=focal_length)(wavefront),
            fraunhofer.focal_length,
            rtol=1e-5,
            atol=1e-5,
        )

        fresnel = dl.Fresnel(physical_spec, defocus=0.1, focal_length=2.0)
        assert_differentiable(
            lambda defocus: fresnel.set(defocus=defocus)(wavefront),
            fresnel.defocus,
            rtol=1e-5,
            atol=1e-5,
        )

        angular_spectrum = dl.ASM(0.1)
        assert_differentiable(
            lambda distance: angular_spectrum.set(distance=distance)(wavefront),
            angular_spectrum.distance,
            rtol=1e-5,
            atol=1e-5,
        )


class TestValidation:
    @pytest.mark.parametrize(
        "constructor",
        [
            lambda spec: dl.Fraunhofer(spec, method="invalid"),
            lambda spec: dl.Fraunhofer(dl.PadSpec(), method="mft"),
            lambda spec: dl.Fresnel(spec, method="fft"),
            lambda spec: dl.ABCDPropagator([], spec),
            lambda spec: dl.ABCDPropagator(
                [dl.ABCDFreeSpace(1.0)],
                dl.PadSpec(),
                method="lct",
            ),
            lambda spec: dl.ASM(1.0, spec),
        ],
    )
    def test_construction(self, constructor, physical_spec):
        with pytest.raises((TypeError, ValueError)):
            constructor(physical_spec)

    def test_coordinate_compatibility(
        self,
        angular_spec,
        physical_spec,
        make_wavefront,
    ):
        wavefront = make_wavefront()

        with pytest.raises(ValueError, match="without a focal length"):
            dl.Fraunhofer(physical_spec)(wavefront)
        with pytest.raises(ValueError, match="with a focal length"):
            dl.Fraunhofer(angular_spec, focal_length=2.0)(wavefront)
        with pytest.raises(ValueError, match="physical units"):
            dl.ABCDPropagator(
                [dl.ABCDFreeSpace(0.1)],
                angular_spec,
            )(wavefront)

        angular_input = make_wavefront(spec=dl.CoordSpec(n=8, d=0.1, unit="rad"))
        with pytest.raises(ValueError, match="physical units"):
            dl.Fraunhofer(angular_spec)(angular_input)

    def test_fft_requires_monochromatic_wavefront(self, make_wavefront):
        layer = dl.Fraunhofer(dl.PadSpec(), method="fft")
        wavefront = make_wavefront(wavelength=np.asarray([1e-6, 1.1e-6]))

        with pytest.raises(ValueError, match="monochromatic"):
            layer(wavefront)
