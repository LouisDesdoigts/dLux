"""Wavefront adapters preserve numerical propagation routes and batching."""

import jax.numpy as np
from jax import Array
import pytest
import zodiax as zdx

import dLux as dl
import dLux.utils as dlu

from tests.helpers import (
    assert_differentiable,
    assert_jittable,
    assert_tree_allclose,
)


@pytest.mark.parametrize(
    "element",
    [
        dlu.ABCD(free_space=1.0),
        dlu.ABCD(lens=2.0),
        dlu.ABCD(radius=3.0),
        dlu.ABCD(fraunhofer=4.0),
    ],
)
def test_abcd_expression_contract(element):
    matrix = assert_jittable(lambda value: value.resolve(), element)
    assert matrix.shape == (2, 2)
    assert_differentiable(lambda value: value.resolve(), element)


@pytest.fixture
def angular_spec():
    return dl.Grid(n=(6, 8), d=(2e-7, 3e-7), unit="rad")


@pytest.fixture
def physical_spec():
    return dl.Grid(n=(6, 8), d=(2e-6, 3e-6), unit="m")


@pytest.mark.parametrize("polarised", [False, True])
def test_custom_numerical_propagator_uses_wavefront_adapter(polarised, make_wavefront):
    class PhaseDelay(dlu.BasePropagator):
        phase: Array

        def __init__(self, phase):
            self.phase = zdx.as_array(phase, dtype=float)
            super().__init__()

        def _prepare(self, grid_in, wavelength):
            x_factor = np.full(grid_in.n[0], np.exp(1j * self.phase))
            y_factor = np.ones(grid_in.n[1])
            return dlu.Chirp((x_factor, y_factor)), grid_in

    wavefront = make_wavefront(
        wavelength=np.asarray([1e-6, 1.1e-6]), polarised=polarised
    )
    layer = PhaseDelay(0.2)
    updated = layer.set("phase", 0.4)
    for operation in (layer, updated):
        actual = assert_jittable(lambda field: field.propagate(operation), wavefront)
        expected = wavefront.set(phasor=wavefront.phasor * np.exp(1j * operation.phase))
        assert_tree_allclose(actual, expected)
    assert np.isclose(layer.phase, 0.2)


@pytest.mark.parametrize("counts", [(2, 3), (2, 2), (1, 3)])
@pytest.mark.parametrize("polarised", [False, True])
def test_spectral_focal_length_keeps_wavelength_and_aperture_axes(counts, polarised):
    n_wavelengths, n_apertures = counts
    wavelengths = np.linspace(0.9e-6, 1.1e-6, n_wavelengths)
    centers = np.stack((np.linspace(-0.1, 0.1, n_apertures), np.zeros(n_apertures)), -1)
    pupil = dlu.Grid((4, 4), d=0.2, c=centers[:, None], unit="m")
    focal = dlu.Grid((4, 4), d=1e-6, unit="m")
    wavefront = dl.Wavefront(wavelengths, pupil, jones=True if polarised else None)

    class FocalLength(zdx.Expression):
        def evaluate(self, *, wavelengths, **context):
            return 1 + 0.1 * (wavelengths - 1e-6) / 1e-6

    focal_length = FocalLength()
    layer = dl.PropagatorLayer(focal, focal_length=focal_length)

    actual = assert_jittable(layer, wavefront, rtol=2e-5, atol=2e-5)
    expected = []
    for w, wavelength in enumerate(wavelengths):
        f = 1 + 0.1 * (wavelength - 1e-6) / 1e-6
        propagator = dlu.Propagator(focal, focal_length=f)
        fields = [
            propagator(wavefront.phasor[a, w], pupil.set(c=centers[a]), wavelength)
            for a in range(n_apertures)
        ]
        expected.append(np.stack(fields))
    assert actual.phasor.shape == wavefront.phasor.shape
    assert np.allclose(actual.phasor, np.stack(expected, axis=1), rtol=2e-5, atol=2e-5)
    assert_tree_allclose(actual.grid, focal)


class TestPropagation:
    def test_direct_parameters_match_explicit_numerical_definition(
        self, physical_spec, make_wavefront
    ):
        wavefront = make_wavefront()
        direct = dl.PropagatorLayer(physical_spec, focal_length=2.0)
        updated = direct.set("focal_length", np.asarray(2.5))
        numerical = dlu.Propagator(physical_spec, focal_length=2.5)
        assert isinstance(direct, dlu.Propagator)
        assert "propagator" not in type(direct).__dataclass_fields__
        assert_tree_allclose(
            assert_jittable(updated, wavefront), wavefront.propagate(numerical)
        )
        assert np.allclose(direct.focal_length, 2.0)
        assert_differentiable(
            lambda focal_length: direct.set(focal_length=focal_length)(wavefront),
            direct.focal_length,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_abcd_parameters_have_explicit_nested_paths(self, physical_spec):
        propagator = dl.PropagatorLayer(
            physical_spec,
            abcd=dlu.ABCDChain(
                {"space": dlu.ABCD(free_space=0.1), "lens": dlu.ABCD(lens=2.0)}
            ),
        )
        updated = propagator.set("abcd.elements.space.free_space", 0.2)

        assert np.isclose(propagator.abcd.elements["space"].free_space, 0.1)
        assert np.isclose(updated.abcd.elements["space"].free_space, 0.2)

    @pytest.mark.parametrize(
        "make_layer",
        [
            lambda angular, physical: dl.PropagatorLayer(angular),
            lambda angular, physical: dl.PropagatorLayer(physical, focal_length=2.0),
            lambda angular, physical: dlu.FFTPropagator(dl.FFTSpec(pad=2, crop=2)),
            lambda angular, physical: dl.PropagatorLayer(
                physical, defocus=0.1, focal_length=2.0
            ),
            lambda angular, physical: dlu.FFTPropagator(
                dl.FFTSpec(), defocus=0.1, focal_length=2.0
            ),
            lambda angular, physical: dl.PropagatorLayer(
                physical, abcd=dlu.ABCDChain([dlu.ABCD(free_space=0.1)])
            ),
            lambda angular, physical: dlu.FFTPropagator(
                dl.FFTSpec(), abcd=dlu.ABCDChain([dlu.ABCD(fraunhofer=2.0)])
            ),
            lambda angular, physical: dlu.FreeSpace(0.1),
        ],
    )
    def test_propagator_contract(
        self, make_layer, angular_spec, physical_spec, make_wavefront
    ):
        layer = make_layer(angular_spec, physical_spec)
        assert_jittable(
            lambda operation, field: field.propagate(operation),
            layer,
            make_wavefront(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_vectorised_wavefronts(self, angular_spec, make_wavefront):
        layer = dl.PropagatorLayer(angular_spec)

        chromatic = make_wavefront(wavelength=np.asarray([1e-6, 1.1e-6]))
        polarised = make_wavefront(polarised=True)

        assert_jittable(layer, chromatic, rtol=1e-5, atol=1e-5)
        assert_jittable(layer, polarised, rtol=1e-5, atol=1e-5)

    def test_chromatic_spatial_vectorisation(self, angular_spec, make_grid):
        wavelengths = np.asarray((0.9e-6, 1.1e-6))
        centers = np.asarray(((-0.2, 0.0), (0.0, 0.2), (0.2, 0.0)))
        wavefront = dl.Wavefront(wavelengths, make_grid().set(c=centers[:, None]))

        propagated = assert_jittable(
            dl.PropagatorLayer(angular_spec),
            wavefront,
            rtol=1e-5,
            atol=1e-5,
        )
        interfered = assert_jittable(lambda wf: wf.coherent_sum(), propagated)

        assert propagated.phasor.shape == (3, 2, 8, 6)
        assert interfered.phasor.shape == (2, 8, 6)

    def test_output_sampling(self, angular_spec, make_wavefront):
        requested = dl.PropagatorLayer(angular_spec)(make_wavefront())
        assert_tree_allclose(requested.grid, angular_spec)

        native = make_wavefront().propagate(dlu.FFTPropagator(dl.FFTSpec()))
        assert native.grid.unit == "rad"
        assert native.grid.n == (8, 8)

    @pytest.mark.parametrize(
        ("make_fft", "make_mft"),
        [
            (
                lambda grid: dlu.FFTPropagator(dl.FFTSpec()),
                lambda grid: dl.PropagatorLayer(grid),
            ),
            (
                lambda grid: dlu.FFTPropagator(
                    dl.FFTSpec(), defocus=0.1, focal_length=2.0
                ),
                lambda grid: dl.PropagatorLayer(grid, defocus=0.1, focal_length=2.0),
            ),
        ],
    )
    def test_fft_matches_explicit_propagation(
        self,
        make_fft,
        make_mft,
        make_wavefront,
    ):
        wavefront = make_wavefront(grid=dl.Grid(n=8, d=1e-3, unit="m"))
        fft_output = wavefront.propagate(make_fft(None))
        mft_output = make_mft(fft_output.grid)(wavefront)

        assert_tree_allclose(fft_output, mft_output, rtol=2e-5, atol=2e-6)

    def test_fraunhofer_mft_inverse_roundtrip(self, make_wavefront):
        wavefront = make_wavefront()
        focal_spec = wavefront.propagate(dlu.FFTPropagator(dl.FFTSpec())).grid

        focal = dl.PropagatorLayer(focal_spec)(wavefront)
        recovered = dl.PropagatorLayer(wavefront.grid, inverse=True)(focal)

        assert_tree_allclose(recovered, wavefront, rtol=2e-5, atol=2e-6)

    def test_fraunhofer_fft_inverse_roundtrip_with_pad_and_crop(self, make_wavefront):
        wavefront = make_wavefront()

        focal = wavefront.propagate(dlu.FFTPropagator(dl.FFTSpec(pad=2, c=np.zeros(2))))
        recovered = focal.propagate(
            dlu.FFTPropagator(dl.FFTSpec(crop=2, c=np.zeros(2)), inverse=True)
        )

        assert np.allclose(recovered.phasor, wavefront.phasor, rtol=2e-5, atol=2e-6)
        assert recovered.grid.n == wavefront.grid.n
        assert recovered.grid.unit == wavefront.grid.unit
        assert np.allclose(recovered.grid.d, wavefront.grid.d)
        assert np.allclose(recovered.grid.c, 0, atol=1e-7)

    def test_chromatic_fft_inverse_roundtrip(self, make_wavefront):
        wavefront = make_wavefront(wavelength=np.asarray([1e-6, 1.1e-6]))

        focal = wavefront.propagate(
            dlu.FFTPropagator(dl.FFTSpec(pad=2, c=np.asarray((2e-06, -1e-06))))
        )
        recovered = focal.propagate(
            dlu.FFTPropagator(dl.FFTSpec(crop=2, c=np.zeros(2)), inverse=True)
        )

        assert_tree_allclose(recovered, wavefront, rtol=2e-5, atol=2e-6)

    def test_fresnel_inverse_roundtrip(self, make_wavefront):
        wavefront = make_wavefront()
        kwargs = {"defocus": 1e-3, "focal_length": 2.0}
        focal_spec = wavefront.propagate(
            dlu.FFTPropagator(dl.FFTSpec(), defocus=0.001, focal_length=2.0)
        ).grid

        focal = dl.PropagatorLayer(focal_spec, **kwargs)(wavefront)
        recovered = dl.PropagatorLayer(wavefront.grid, inverse=True, **kwargs)(focal)

        assert_tree_allclose(recovered, wavefront, rtol=2e-5, atol=2e-6)

    def test_field_gradient(self, angular_spec, make_wavefront):
        wavefront = make_wavefront()
        layer = dl.PropagatorLayer(angular_spec)

        assert_differentiable(
            lambda phasor: layer(wavefront.set(phasor=phasor)),
            wavefront.phasor,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_physical_parameter_gradients(self, physical_spec, make_wavefront):
        wavefront = make_wavefront(grid=dl.Grid(n=8, d=1e-3, unit="m"))

        fraunhofer = dl.PropagatorLayer(physical_spec, focal_length=2.0)
        assert_differentiable(
            lambda focal_length: fraunhofer.set("focal_length", focal_length)(
                wavefront
            ),
            fraunhofer.focal_length,
            rtol=1e-5,
            atol=1e-5,
        )

        fresnel = dl.PropagatorLayer(physical_spec, defocus=0.1, focal_length=2.0)
        assert_differentiable(
            lambda defocus: fresnel.set("defocus", defocus)(wavefront),
            fresnel.defocus,
            rtol=1e-5,
            atol=1e-5,
        )

        angular_spectrum = dlu.FreeSpace(0.1)
        assert_differentiable(
            lambda distance: wavefront.propagate(
                angular_spectrum.set(distance=distance)
            ),
            angular_spectrum.distance,
            rtol=1e-5,
            atol=1e-5,
        )


class TestValidation:
    @pytest.mark.parametrize(
        "operation", [dlu.propagation.FFT(4), dlu.FreeSpace(0.1), dl.Optic()]
    )
    def test_layer_requires_output_grid_not_nested_operation(self, operation):
        with pytest.raises(TypeError, match="Grid"):
            dl.PropagatorLayer(operation)

    @pytest.mark.parametrize(
        "constructor",
        [
            lambda grid: dl.PropagatorLayer(grid, method="invalid"),
            lambda grid: dl.PropagatorLayer(dl.FFTSpec()),
            lambda grid: dlu.FFTPropagator(grid, defocus=0.0),
            lambda grid: dl.PropagatorLayer(grid, abcd=dlu.ABCDChain([])),
            lambda grid: dl.PropagatorLayer(
                dl.FFTSpec(), abcd=dlu.ABCDChain([dlu.ABCD(free_space=1.0)])
            ),
            lambda grid: dlu.FreeSpace(1.0, grid),
        ],
    )
    def test_construction(self, constructor, physical_spec):
        with pytest.raises((TypeError, ValueError)):
            constructor(physical_spec)

    def test_inverse_fresnel_fft_roundtrip(self, make_wavefront):
        wavefront = make_wavefront()
        forward = dlu.FFTPropagator(
            dl.FFTSpec(pad=2, c=(0.0, 0.0)), defocus=0.001, focal_length=2.0
        )
        reverse = dlu.FFTPropagator(
            dl.FFTSpec(crop=2, c=(0.0, 0.0)),
            defocus=0.001,
            focal_length=2.0,
            inverse=True,
        )
        recovered = wavefront.propagate(forward).propagate(reverse)
        assert_tree_allclose(recovered, wavefront, rtol=2e-5, atol=2e-6)

    def test_coordinate_compatibility(
        self,
        angular_spec,
        physical_spec,
        make_wavefront,
    ):
        wavefront = make_wavefront()

        with pytest.raises(ValueError, match="angle"):
            dl.PropagatorLayer(physical_spec)(wavefront)
        with pytest.raises(ValueError, match="length"):
            dl.PropagatorLayer(angular_spec, focal_length=2.0)(wavefront)
        with pytest.raises(ValueError, match="length"):
            dl.PropagatorLayer(
                angular_spec, abcd=dlu.ABCDChain([dlu.ABCD(free_space=0.1)])
            )(wavefront)

        angular_input = make_wavefront(grid=dl.Grid(n=8, d=0.1, unit="rad"))
        with pytest.raises(ValueError, match="length"):
            dl.PropagatorLayer(angular_spec)(angular_input)

    def test_fft_supports_chromatic_wavefront(self, make_wavefront):
        layer = dlu.FFTPropagator(dl.FFTSpec())
        wavefront = make_wavefront(wavelength=np.asarray([1e-6, 1.1e-6]))

        output = wavefront.propagate(layer)

        assert output.phasor.shape == (2, 8, 8)
        assert output.grid.d.shape == (2, 2)


def test_propagator_layer_mono_extension_method_matches_call(
    make_wavefront, angular_spec
):
    field = make_wavefront()
    layer = dl.PropagatorLayer(angular_spec)
    assert_tree_allclose(layer.apply_mono(field), layer(field))
