"""Tests for dLux.layers.propagation."""

import jax.numpy as np
import pytest

import dLux as dl

from tests.helpers import (
    assert_differentiable,
    assert_jittable,
    assert_tree_allclose,
)


@pytest.mark.parametrize(
    "element",
    [
        dl.ABCDFreeSpace(1.0),
        dl.ABCDLens(2.0),
        dl.ABCDMirror(3.0),
        dl.ABCDFraunhofer(4.0),
    ],
)
def test_abcd_element_contract(element):
    matrix = assert_jittable(lambda value: value.abcd, element)
    assert matrix.shape == (2, 2)
    assert_differentiable(lambda value: value.abcd, element)


@pytest.fixture
def angular_spec():
    return dl.GridSpec(n=(6, 8), d=(2e-7, 3e-7), unit="rad")


@pytest.fixture
def physical_spec():
    return dl.GridSpec(n=(6, 8), d=(2e-6, 3e-6), unit="m")


class TestPropagation:
    def test_abcd_parameter_raising(self, physical_spec):
        propagator = dl.ABCDPropagator(
            {"space": dl.ABCDFreeSpace(0.1), "lens": dl.ABCDLens(2.0)},
            physical_spec,
        )
        updated = propagator.set("distance", 0.2)

        assert propagator.space is propagator.ABCDs["space"]
        assert np.isclose(propagator.get("distance"), 0.1)
        assert np.isclose(updated.space.distance, 0.2)

    @pytest.mark.parametrize(
        "make_layer",
        [
            lambda angular, physical: dl.Fraunhofer(angular, method="mft"),
            lambda angular, physical: dl.Fraunhofer(
                physical, focal_length=2.0, method="mft"
            ),
            lambda angular, physical: dl.Fraunhofer(
                dl.ResizeSpec(pad=2, crop=2), method="fft"
            ),
            lambda angular, physical: dl.Fresnel(
                physical, defocus=0.1, focal_length=2.0, method="mft"
            ),
            lambda angular, physical: dl.Fresnel(
                physical, defocus=0.1, focal_length=2.0, method="lct"
            ),
            lambda angular, physical: dl.Fresnel(
                dl.ResizeSpec(), defocus=0.1, focal_length=2.0, method="fft"
            ),
            lambda angular, physical: dl.ABCDPropagator(
                [dl.ABCDFreeSpace(0.1)], physical, method="lct"
            ),
            lambda angular, physical: dl.ABCDPropagator(
                [dl.ABCDFraunhofer(2.0)], dl.ResizeSpec(), method="fft"
            ),
            lambda angular, physical: dl.FreeSpace(0.1),
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
        assert_tree_allclose(requested.grid, angular_spec)

        native = dl.Fraunhofer(dl.ResizeSpec(), method="fft")(make_wavefront())
        assert native.grid.unit == "rad"
        assert native.grid.n == (8, 8)

    @pytest.mark.parametrize(
        ("make_fft", "make_mft"),
        [
            (
                lambda grid: dl.Fraunhofer(dl.ResizeSpec(), method="fft"),
                lambda grid: dl.Fraunhofer(grid, method="mft"),
            ),
            (
                lambda grid: dl.Fresnel(
                    dl.ResizeSpec(),
                    defocus=0.1,
                    focal_length=2.0,
                    method="fft",
                ),
                lambda grid: dl.Fresnel(
                    grid,
                    defocus=0.1,
                    focal_length=2.0,
                    method="mft",
                ),
            ),
        ],
    )
    def test_fft_matches_explicit_propagation(
        self,
        make_fft,
        make_mft,
        make_wavefront,
    ):
        wavefront = make_wavefront(grid=dl.GridSpec(n=8, d=1e-3, unit="m").broadcast(2))
        fft_output = make_fft(None)(wavefront)
        mft_output = make_mft(fft_output.grid)(wavefront)

        assert_tree_allclose(fft_output, mft_output, rtol=2e-5, atol=2e-6)

    def test_fraunhofer_mft_inverse_roundtrip(self, make_wavefront):
        wavefront = make_wavefront()
        focal_spec = dl.Fraunhofer(dl.ResizeSpec(), method="fft")(wavefront).grid

        focal = dl.Fraunhofer(focal_spec)(wavefront)
        recovered = dl.Fraunhofer(wavefront.grid, inverse=True)(focal)

        assert_tree_allclose(recovered, wavefront, rtol=2e-5, atol=2e-6)

    def test_fraunhofer_fft_inverse_roundtrip_with_pad_and_crop(
        self, make_wavefront
    ):
        wavefront = make_wavefront()

        focal = dl.Fraunhofer(
            dl.ResizeSpec(pad=2, c=np.zeros(2)), method="fft"
        )(wavefront)
        recovered = dl.Fraunhofer(
            dl.ResizeSpec(crop=2, c=np.zeros(2)), method="fft", inverse=True
        )(focal)

        assert np.allclose(recovered.phasor, wavefront.phasor, rtol=2e-5, atol=2e-6)
        assert recovered.grid.n == wavefront.grid.n
        assert recovered.grid.unit == wavefront.grid.unit
        assert np.allclose(recovered.grid.d, wavefront.grid.d)
        assert np.allclose(recovered.grid.c, 0, atol=1e-7)

    def test_chromatic_fft_inverse_roundtrip(self, make_wavefront):
        wavefront = make_wavefront(wavelength=np.asarray([1e-6, 1.1e-6]))

        focal = dl.Fraunhofer(
            dl.ResizeSpec(pad=2, c=np.asarray((2e-6, -1e-6))),
            method="fft",
        )(wavefront)
        recovered = dl.Fraunhofer(
            dl.ResizeSpec(crop=2, c=np.zeros(2)),
            method="fft",
            inverse=True,
        )(focal)

        assert_tree_allclose(recovered, wavefront, rtol=2e-5, atol=2e-6)

    @pytest.mark.parametrize("method", ["mft", "lct"])
    def test_fresnel_inverse_roundtrip(self, method, make_wavefront):
        wavefront = make_wavefront()
        kwargs = {"defocus": 1e-3, "focal_length": 2.0, "method": method}
        focal_spec = dl.Fresnel(
            dl.ResizeSpec(), defocus=1e-3, focal_length=2.0, method="fft"
        )(wavefront).grid

        focal = dl.Fresnel(focal_spec, **kwargs)(wavefront)
        recovered = dl.Fresnel(wavefront.grid, inverse=True, **kwargs)(focal)

        assert_tree_allclose(recovered, wavefront, rtol=2e-5, atol=2e-6)

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

        angular_spectrum = dl.FreeSpace(0.1)
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
            lambda grid: dl.Fraunhofer(grid, method="invalid"),
            lambda grid: dl.Fraunhofer(dl.ResizeSpec(), method="mft"),
            lambda grid: dl.Fresnel(grid, method="fft"),
            lambda grid: dl.ABCDPropagator([], grid),
            lambda grid: dl.ABCDPropagator(
                [dl.ABCDFreeSpace(1.0)],
                dl.ResizeSpec(),
                method="lct",
            ),
            lambda grid: dl.FreeSpace(1.0, grid),
        ],
    )
    def test_construction(self, constructor, physical_spec):
        with pytest.raises((TypeError, ValueError)):
            constructor(physical_spec)

    def test_inverse_fresnel_fft_is_rejected(self):
        with pytest.raises(ValueError, match="not supported"):
            dl.Fresnel(dl.ResizeSpec(), method="fft", inverse=True)

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

        angular_input = make_wavefront(grid=dl.GridSpec(n=8, d=0.1, unit="rad"))
        with pytest.raises(ValueError, match="physical units"):
            dl.Fraunhofer(angular_spec)(angular_input)

    def test_fft_supports_chromatic_wavefront(self, make_wavefront):
        layer = dl.Fraunhofer(dl.ResizeSpec(), method="fft")
        wavefront = make_wavefront(wavelength=np.asarray([1e-6, 1.1e-6]))

        output = layer(wavefront)

        assert output.phasor.shape == (2, 8, 8)
        assert output.grid.d.shape == (2, 2)
