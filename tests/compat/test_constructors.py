"""Released calls construct current objects, with explicit migration boundaries."""

import equinox as eqx
import jax
import jax.numpy as np
import pytest

import dLux as dl
import dLux.utils as dlu
from dLux import compat as old

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_old_sampling_builds_current_fields_with_released_input_power():
    legacy = dl.Wavefront(600e-9, 8, 2.0)
    native = dl.Wavefront(600e-9, dlu.Grid(8, diam=2.0, unit="m"))
    assert type(legacy) is dl.Wavefront
    assert np.allclose(legacy.intensity.sum(), 1 / 64)
    assert np.allclose(native.intensity.sum(), 1)
    assert np.allclose(legacy.grid.d, native.grid.d)
    phasor = np.ones((8, 8), complex) * (1 + 2j)
    rebuilt = dl.Wavefront.from_phasor(phasor, 600e-9, pixel_scale=0.25)
    assert np.array_equal(rebuilt.phasor, phasor)
    with pytest.raises(TypeError, match="not both"):
        dl.Wavefront(600e-9, native.grid, phasor=phasor, pixel_scale=0.25)


def test_psf_sampling_keeps_batch_spacing_and_current_immutable_updates():
    image = dl.PSF(np.ones((3, 4, 4)), pixel_scale=np.array([0.1, 0.2, 0.3]))
    assert type(image) is dl.PSF
    assert image.grid.d.shape == (3, 2)
    changed = image.set(array=2 * image.array)
    assert np.all(image.array == 1)
    assert np.all(changed.array == 2)


@pytest.mark.parametrize(
    "factory, expected",
    [
        (lambda: old.TransmissiveLayer(0.3), dl.Optic),
        (lambda: old.AberratedLayer(opd=1e-8), dl.Optic),
        (lambda: old.BasisLayer(np.ones((2, 8, 8)), [1e-8, 2e-8]), dl.Optic),
        (
            lambda: old.BasisOptic(np.ones((2, 8, 8)), coefficients=[1e-8, 2e-8]),
            dl.Optic,
        ),
        (lambda: old.CircularAperture(0.4), dl.Optic),
        (lambda: old.SquareAperture(0.4), dl.Optic),
        (lambda: old.RectangularAperture(0.4, 0.5), dl.Optic),
        (lambda: old.RegPolyAperture(6, 0.4), dl.Optic),
        (lambda: old.Spider(0.05, [0.0, 90.0]), dl.Optic),
        (lambda: old.ApplyPixelResponse(np.ones((2, 2))), dl.SensitivityLayer),
        (lambda: old.ApplyJitter(0.3), dl.ConvolveLayer),
        (lambda: old.ApplySaturation(3.0), dl.SaturationLayer),
        (lambda: old.AddConstant(2.0), dl.BiasLayer),
        (lambda: old.Rotate(0.2), dl.InterpolateLayer),
        (lambda: old.CoordSpec(8, 0.1), dlu.Grid),
        (lambda: old.PadSpec(2), dlu.FFTSpec),
        (lambda: old.CoordTransform(rotation=0.2), dlu.AffineMap),
        (lambda: old.DistortedCoords(2), dlu.Distortion),
        (lambda: old.ABCDLens(2.0), dlu.ABCD),
    ],
)
def test_factories_return_current_classes(factory, expected):
    # Thin compatibility subclasses add read-only views and old method signatures.
    assert isinstance(factory(), expected)


def test_apertures_and_basis_compose_with_current_layers():
    wavefront = dl.Wavefront(600e-9, dlu.Grid(8, diam=1.0, unit="m"))
    circular = old.CircularAperture(0.4, softening=None)
    assert np.array_equal(
        circular(wavefront).phasor,
        wavefront.phasor * old.utils.circle(wavefront.coordinates, 0.4),
    )
    composite = old.CompoundAperture(
        [circular, old.SquareAperture(0.5, softening=None)]
    )
    separate = old.MultiAperture([circular, old.SquareAperture(0.5, softening=None)])
    a = circular.transmission.resolve(grid=wavefront.grid)
    b = dlu.Rectangle(0.5, 0.5).resolve(grid=wavefront.grid)
    assert np.allclose(composite(wavefront).phasor, wavefront.phasor * a * b)
    assert np.allclose(separate(wavefront).phasor, wavefront.phasor * (a + b))
    phase = old.BasisLayer(np.ones((2, 8, 8)), [0.1, 0.2], as_phase=True)
    assert np.allclose(phase(wavefront).phasor, wavefront.phasor * np.exp(0.3j))
    assert isinstance(
        dl.OpticalSystem([circular, phase, dl.NormaliseLayer()])(wavefront),
        dl.Wavefront,
    )


def test_transform_factory_preserves_released_affine_convention():
    coordinates = dlu.Grid(8, 0.2).coordinates
    translation, shear, compression, rotation = [0.1, -0.2], [0.1, 0.2], [0.8, 1.1], 0.3
    transform = old.CoordTransform(translation, rotation, compression)
    expected = old.utils.translate_coords(coordinates, np.array(translation))
    expected = old.utils.compress_coords(expected, np.array(compression))
    expected = old.utils.rotate_coords(expected, rotation)
    assert np.allclose(transform(coordinates), expected, atol=2e-7)
    with pytest.raises(ValueError, match="transposes grid samples"):
        old.CoordTransform(shear=shear)


def test_nested_apertures_store_combine_expressions():
    ring = old.CompoundAperture(
        [old.CircularAperture(0.4), old.CircularAperture(0.2, occulting=True)]
    )
    optic = old.MultiAperture({"ring": ring, "square": old.SquareAperture(0.3)})

    assert type(optic) is dl.Optic
    assert type(optic.transmission) is dlu.Combine
    assert type(optic.transmission.parametrics["ring"]) is dlu.Combine
    assert optic.transmission.operation == "sum"
    assert optic.transmission.parametrics["ring"].operation == "product"


def test_sources_preserve_photon_counts_and_binary_midpoint_conventions():
    source = dl.PointSource(wavelengths=[500e-9, 700e-9], weights=[1, 3], flux=8.0)
    optics = old.AngularOpticalSystem(4, 1.0, [], 4, 0.1)
    wavelengths, weights = optics.photometry(source).weights()
    assert np.allclose(wavelengths, np.array([500e-9, 700e-9]))
    assert np.allclose(weights, np.array([[2.0, 6.0]]))
    binary = old.BinarySource(
        [600e-9],
        position=[1.0, 2.0],
        mean_flux=3.0,
        separation=0.4,
        position_angle=0.2,
        contrast=2.0,
    )
    assert type(binary) is dl.Source
    assert np.allclose(binary.positions.mean(0), np.array([1.0, 2.0]))
    assert np.allclose(binary.spectrum.flux, old.utils.fluxes_from_contrast(3.0, 2.0))
    native = dl.BinarySource(
        position=[1.0, 2.0], unit="rad", primary_flux=3.0, contrast=0.2, log=False
    ).resolve()
    assert np.allclose(native.positions[0], np.array([1.0, 2.0]))
    assert np.allclose(native.spectrum.flux, np.array([3.0, 0.6]))
    with pytest.raises(TypeError, match="positional arrays"):
        dl.PointSource([600e-9])
    with pytest.raises(ValueError, match="Per-component"):
        old.BinarySource([600e-9], weights=[[1.0], [1.0]])


def test_normal_imports_and_released_constructors_mix_in_one_model():
    system = old.AngularOpticalSystem(8, 1.0, [dl.Optic(normalise=True)], 8, 0.1)
    source = dl.PointSource(wavelengths=[500e-9, 700e-9], flux=3.0)
    assert isinstance(system, dl.Imager)
    assert float(system.diameter) == 1.0
    detector = old.LayeredDetector([dl.BiasLayer(0.2), dl.DownsampleLayer(2)])
    model = old.Telescope(system, source, detector)
    assert isinstance(model, dl.Imager)
    expected = detector(dl.PSF(system.model(source), system.image_grid))
    assert np.allclose(model.model().data, expected)
    actual = eqx.filter_jit(lambda x: x.model().data)(model)
    assert np.allclose(actual, expected, rtol=2e-5)
    flux_grad = jax.grad(
        lambda flux: model.set("source.spectrum.flux", flux).model().data.sum()
    )(3.0)
    assert np.allclose(flux_grad, system.model(source).sum() / 3.0, rtol=2e-5)


def test_telescope_keeps_released_unnormalised_entrance_power():
    system = old.AngularOpticalSystem(8, 1.0, [dl.Optic(transmission=0.5)], 8, 0.1)
    source = old.PointSource([600e-9])
    assert np.allclose(
        old.Telescope(system, source).model().data,
        system.model(source),
        rtol=2e-5,
        atol=1e-8,
    )


def test_layered_system_and_custom_released_call_share_current_wavefronts():
    class Attenuate(dl.OpticalLayer):
        def __call__(self, wavefront):
            return wavefront.set(phasor=wavefront.phasor * 0.5)

    system = old.LayeredOpticalSystem(8, 1.0, [dl.Optic(normalise=True), Attenuate()])
    source = old.PointSource([500e-9, 700e-9], weights=[0.2, 0.8], flux=4.0)
    assert isinstance(system, dl.OpticalSystem)
    image = source.model(system)
    assert np.allclose(image.sum(), 1.0)
    field = source.model(system, return_wf=True)
    assert field.phasor.shape == (2, 8, 8)
    assert np.allclose(field.intensity.sum(), image.sum())


def test_propagation_factories_use_current_kernels():
    field = dl.Wavefront(600e-9, dlu.Grid(8, diam=1.0, unit="m"))
    for operation in [
        old.MFT(8, 1e-7),
        old.FFT(),
        old.MFTPropagator([old.ABCDConjugatePlane(1.0)], dlu.Grid(8, 1e-7, unit="m")),
    ]:
        actual = operation(field)
        expected = field.propagate(operation.operation)
        assert np.allclose(actual.phasor, expected.phasor * operation.phase)


@pytest.mark.parametrize(
    "call, message",
    [
        (
            lambda: old.ResolvedSource([600e-9], distribution=np.ones((3, 3))),
            "profile_grid",
        ),
        (lambda: old.PointResolvedSource(), "Scene"),
        (
            lambda: old.AberratedAperture(old.CircularAperture(0.4), [4]),
            "basis support",
        ),
        (lambda: old.Dither(), "vmap"),
        (
            lambda: old.AngularOpticalSystem(8, 1.0, [old.FFT()], 8, 0.1),
            "Intermediate propagation",
        ),
    ],
)
def test_untranslatable_contracts_fail_with_migration_guidance(call, message):
    with pytest.raises((TypeError, ValueError), match=message):
        call()


@pytest.mark.parametrize("inverse", [False, True])
def test_mft_layer_preserves_released_complex_phase(inverse):
    grid = dlu.Grid(8, 0.1, unit="rad" if inverse else "m")
    coordinates = grid.coordinates
    phasor = (1 + coordinates[0]) * np.exp(0.3j * coordinates[1])
    field = dl.Wavefront(600e-9, grid, phasor)
    actual = old.MFT(8, 1e-7, inverse=inverse)(field)
    expected = old.utils.MFT(phasor, 600e-9, 0.1, 8, 1e-7, inverse=inverse)
    assert np.allclose(actual.phasor, expected, rtol=3e-6, atol=1e-7)


@pytest.mark.parametrize("oversample", [1, 2, 3])
def test_released_detector_owns_oversampled_pixel_integration(oversample):
    optics = old.AngularOpticalSystem(
        8, 1.0, [dl.Optic(normalise=True)], 6, 0.1, oversample
    )
    source = old.PointSource([600e-9], flux=4.0)
    detector = old.LayeredDetector(
        [old.AddConstant(0.1), dl.DownsampleLayer(oversample)]
    )
    telescope = old.Telescope(optics, source, detector)
    expected = detector(dl.PSF(optics.model(source), optics.propagation_grid))
    image = telescope.model()
    assert image.data.shape == (6, 6)
    assert np.allclose(image.data, expected, rtol=2e-5)


def test_telescope_accepts_native_imager_without_changing_its_normalisation():
    optics = dl.Imager(
        dlu.Grid(8, diam=1.0, unit="m"),
        dlu.Grid(6, 0.1, unit="arcsec"),
        pupil=dl.Optic(transmission=0.5),
        oversample=2,
    )
    band = dlu.BandPass(600, unit="nm")
    source = dl.PointSource(dlu.Spectrum(bandpass=band, flux=4.0))
    telescope = old.Telescope(optics, source)
    assert np.allclose(telescope.model().data, optics.model(source).data, rtol=2e-5)
