from jax import numpy as np, config
import equinox as eqx

config.update("jax_debug_nans", True)
import pytest
from dLux import Wavefront, PolarisedWavefront
from dLux.psfs import PSF
from dLux.coordinates import CoordSpec
from dLux.utils import pixel_coords
import dLux.utils as dlu


def complex_pattern(npixels):
    y, x = np.mgrid[:npixels, :npixels]
    amplitude = 1 + 0.1 * x + 0.2 * y
    phase = 0.3 * x - 0.2 * y + 0.05 * x * y
    return amplitude * np.exp(1j * phase)


def jones_pattern(npixels):
    weights = np.array([[1.0, 0.2 - 0.1j], [0.4 + 0.3j, 0.7]])
    return weights[..., None, None] * complex_pattern(npixels)


def assert_polarised_identity(phasor, scalar_phasor):
    assert np.allclose(phasor[..., 0, 0, :, :], scalar_phasor)
    assert np.allclose(phasor[..., 1, 1, :, :], scalar_phasor)
    assert np.allclose(phasor[..., 0, 1, :, :], 0.0)
    assert np.allclose(phasor[..., 1, 0, :, :], 0.0)


def polarised_componentwise_expected(wavefront, operation, *args):
    leading = wavefront.phasor.shape[:-2]

    def match_leading(array):
        array = np.asarray(array)
        return array.reshape(array.shape + (1,) * (len(leading) - array.ndim))

    def apply(phasor, wavelength, pixel_scale, center):
        component = Wavefront.from_phasor(
            phasor,
            wavelength,
            pixel_scale=pixel_scale,
            center=center,
        )
        out = operation(component, *args)
        return out.phasor, out.pixel_scale, out.center

    apply = np.vectorize(apply, signature="(n,n),(),(),()->(m,m),(),()")
    phasor, pixel_scale, center = apply(
        wavefront.phasor,
        match_leading(wavefront.wavelength),
        match_leading(wavefront.pixel_scale),
        match_leading(wavefront.center),
    )
    return wavefront.set(
        phasor=phasor,
        pixel_scale=wavefront._from_vec(pixel_scale),
        center=wavefront._from_vec(center),
    )


class BaseWavefrontTests:
    """
    Abstract base test suite for all Wavefront variants.
    Pytest will not run this directly because the name does not start with 'Test'.
    """

    def assert_matches(self, actual, expected):
        assert isinstance(actual, Wavefront)
        assert actual.phasor.shape == expected.phasor.shape
        assert np.allclose(actual.phasor, expected.phasor, atol=1e-6, rtol=1e-6)
        assert np.allclose(actual.wavelength, expected.wavelength)
        assert np.allclose(actual.pixel_scale, expected.pixel_scale)
        assert np.allclose(actual.center, expected.center)

    def test_properties(self, wavefront, ndim, chromatic):
        assert wavefront.real.shape == wavefront.phasor.shape
        assert wavefront.imaginary.shape == wavefront.phasor.shape
        assert wavefront.amplitude.shape == wavefront.phasor.shape
        assert wavefront.phase.shape == wavefront.phasor.shape
        assert wavefront.complex.shape[:1] == (2,)
        assert wavefront.polar.shape[:1] == (2,)
        assert np.allclose(wavefront.complex[0], wavefront.real)
        assert np.allclose(wavefront.complex[1], wavefront.imaginary)
        assert np.allclose(wavefront.polar[0], wavefront.amplitude)
        assert np.allclose(wavefront.polar[1], wavefront.phase)
        assert isinstance(wavefront.to_psf(), PSF)
        assert wavefront.ndim == ndim
        assert wavefront.is_chromatic is chromatic
        if not chromatic:
            assert wavefront._mapped_axis is None

    @pytest.mark.parametrize(
        "operation",
        [
            lambda wavefront: wavefront.add_phase(0.3),
            lambda wavefront: wavefront.add_phase(None),
            lambda wavefront: wavefront.add_opd(1.0e-8),
            lambda wavefront: wavefront.add_opd(None),
            lambda wavefront: wavefront.tilt(np.array([1.0e-7, -2.0e-7])),
            lambda wavefront: wavefront.flip(-1),
            lambda wavefront: wavefront.resize(32),
            lambda wavefront: wavefront.resize(8),
            lambda wavefront: wavefront.downsample(2),
        ],
        ids=[
            "add_phase",
            "add_phase_none",
            "add_opd",
            "add_opd_none",
            "tilt",
            "flip",
            "pad",
            "crop",
            "downsample",
        ],
    )
    def test_methods(self, wavefront, expected, operation):
        self.assert_matches(operation(wavefront), expected(operation))

    def test_normalise_power(self, wavefront):
        out = wavefront.normalise()
        assert isinstance(out, Wavefront)
        assert np.allclose(np.sum(out.psf), 1.0)

    def test_normalise_peak(self, wavefront):
        out = wavefront.normalise(mode="peak")
        assert isinstance(out, Wavefront)
        assert np.allclose(np.max(out.psf), 1.0)

    @pytest.mark.parametrize("complex", [True, False])
    def test_scale_to(self, wavefront, expected, output_pixel_scale, complex):
        operation = lambda wavefront, pixel_scale: wavefront.scale_to(
            8, pixel_scale, complex=complex
        )
        self.assert_matches(
            operation(wavefront, output_pixel_scale),
            expected(operation, output_pixel_scale),
        )

    def test_scale_to_legacy_complex_argument(
        self, wavefront, expected, output_pixel_scale
    ):
        operation = lambda wavefront, pixel_scale: wavefront.scale_to(
            8, pixel_scale, False
        )
        self.assert_matches(
            operation(wavefront, output_pixel_scale),
            expected(operation, output_pixel_scale),
        )

    @pytest.mark.parametrize("complex", [True, False])
    def test_interpolate(self, wavefront, expected, complex):
        coords_in = pixel_coords(16, diameter=1.0)
        coords_out = pixel_coords(8, diameter=1.0)
        operation = lambda wavefront: wavefront.interpolate(
            coords_in, coords_out, complex=complex
        )
        self.assert_matches(operation(wavefront), expected(operation))

    @pytest.mark.parametrize("angle", [np.pi / 8])
    @pytest.mark.parametrize("complex", [True, False])
    def test_rotate(self, wavefront, expected, angle, complex):
        operation = lambda wavefront: wavefront.rotate(angle, complex=complex)
        self.assert_matches(operation(wavefront), expected(operation))

    def test_propagation(self, wavefront, expected, output_pixel_scale):
        operation = lambda wavefront, pixel_scale: wavefront.propagate(8, pixel_scale)
        self.assert_matches(
            operation(wavefront, output_pixel_scale),
            expected(operation, output_pixel_scale),
        )

    def test_propagation_with_focal_length(
        self, wavefront, expected, output_pixel_scale
    ):
        operation = lambda wavefront, pixel_scale: wavefront.propagate(
            8, pixel_scale, 1.0
        )
        self.assert_matches(
            operation(wavefront, output_pixel_scale),
            expected(operation, output_pixel_scale),
        )

    def test_mft_propagation_with_spec(self, wavefront, expected, output_pixel_scale):
        operation = lambda wavefront, pixel_scale: wavefront.propagate_MFT(
            CoordSpec(n=8, d=pixel_scale, c=0.0)
        )
        self.assert_matches(
            operation(wavefront, output_pixel_scale),
            expected(operation, output_pixel_scale),
        )

    @pytest.mark.parametrize("spec_out", [None, CoordSpec(c=0.0)])
    def test_fft_propagation(self, wavefront, expected, spec_out):
        operation = lambda wavefront: wavefront.propagate_FFT(pad=2, spec_out=spec_out)
        self.assert_matches(operation(wavefront), expected(operation))

    def test_fft_propagation_with_focal_length(self, wavefront, expected):
        operation = lambda wavefront: wavefront.propagate_FFT(pad=2, focal_length=1.0)
        self.assert_matches(operation(wavefront), expected(operation))

    def test_coordinates(self, wavefront, expected):
        coords = wavefront.coordinates(polar=True)
        expected_coords = expected(lambda wavefront: wavefront.coordinates(polar=True))
        assert coords.shape[-3:] == (2, 16, 16)
        assert np.allclose(coords, expected_coords)

    def test_set_spec(self, wavefront, expected):
        operation = lambda wavefront: wavefront.set_spec(
            CoordSpec(n=16, d=1 / 16, c=0.0)
        )
        self.assert_matches(operation(wavefront), expected(operation))

    def test_magic_add(self, wavefront, expected):
        operation = lambda wavefront: wavefront + 1
        self.assert_matches(operation(wavefront), expected(operation))

    def test_magic_mul(self, wavefront, expected):
        operation = lambda wavefront: wavefront * np.exp(1j)
        self.assert_matches(operation(wavefront), expected(operation))

    def test_magic_sub(self, wavefront, expected):
        operation = lambda wavefront: wavefront - 1
        self.assert_matches(operation(wavefront), expected(operation))

    def test_magic_div(self, wavefront, expected):
        operation = lambda wavefront: wavefront / 2
        self.assert_matches(operation(wavefront), expected(operation))

    def test_magic_none(self, wavefront, expected):
        operation = lambda wavefront: wavefront * None
        self.assert_matches(operation(wavefront), expected(operation))

    def test_magic_wavefront_operand(self, wavefront, expected):
        operation = lambda wavefront: wavefront + wavefront
        self.assert_matches(operation(wavefront), expected(operation))

    @pytest.mark.parametrize(
        "operation",
        [
            lambda wavefront: wavefront.__iadd__(1),
            lambda wavefront: wavefront.__isub__(1),
            lambda wavefront: wavefront.__imul__(2),
            lambda wavefront: wavefront.__itruediv__(2),
        ],
        ids=["iadd", "isub", "imul", "itruediv"],
    )
    def test_magic_inplace_methods(self, wavefront, expected, operation):
        self.assert_matches(operation(wavefront), expected(operation))

    def test_psf_from_stokes(self, wavefront):
        assert np.allclose(wavefront.psf_from_stokes(), wavefront.psf)
        stokes = np.array([2.0, 0.0, 0.0, 0.0])
        assert np.allclose(wavefront.psf_from_stokes(stokes), 2 * wavefront.psf)

    def test_method_errors(self, wavefront):
        with pytest.raises(ValueError):
            wavefront.tilt(np.zeros(3))
        with pytest.raises(ValueError, match="mode must be"):
            wavefront.normalise(mode="invalid")
        with pytest.raises(TypeError, match="Unsupported type"):
            wavefront * "1"
        with pytest.raises(ValueError, match="Unsupported operation"):
            wavefront._magic_unified_op(np.ones(1), "invalid")
        with pytest.raises(ValueError, match="cannot specify d"):
            wavefront.propagate_FFT(spec_out=CoordSpec(c=0.0, d=1.0))
        with pytest.raises(ValueError, match="cannot specify n"):
            wavefront.propagate_FFT(spec_out=CoordSpec(c=0.0, n=16))

    def test_fresnel_propagation_not_implemented(self, wavefront):
        with pytest.raises(NotImplementedError):
            wavefront.propagate_ASM()
        with pytest.raises(NotImplementedError):
            wavefront.propagate_fresnel()
        with pytest.raises(NotImplementedError):
            wavefront.propagate_fresnel_fft()
        with pytest.raises(NotImplementedError):
            wavefront.propagate_fraunhofer()
        with pytest.raises(NotImplementedError):
            wavefront.propagate_fraunhofer_fft()


class TestWavefront(BaseWavefrontTests):
    """Concrete implementation running core tests for standard Wavefront."""

    @pytest.fixture
    def ndim(self):
        return 0

    @pytest.fixture
    def chromatic(self):
        return False

    @pytest.fixture
    def wavefront(self):
        return Wavefront(npixels=16, diameter=1.0, wavelength=1e-6)

    @pytest.fixture
    def output_pixel_scale(self):
        return 1 / 32

    @pytest.fixture
    def expected(self, wavefront):
        return lambda operation, *args: operation(wavefront, *args)

    def test_constructor(self):
        wavefront = Wavefront(npixels=16, diameter=1.0, wavelength=1e-6)
        assert wavefront.npixels == 16
        assert wavefront.diameter == 1.0
        assert wavefront.wavelength == 1e-6
        assert wavefront.phasor.shape == (16, 16)

        wf_px = Wavefront(wavelength=1e-6, npixels=16, pixel_scale=1 / 16)
        assert wf_px.pixel_scale == 1 / 16

        wf_center = Wavefront(
            wavelength=1e-6,
            npixels=16,
            diameter=1.0,
            center=0.0,
        )
        assert np.allclose(wf_center.center, 0.0)

        with pytest.raises(ValueError, match="Provide one"):
            Wavefront(wavelength=1e-6, npixels=16)

        with pytest.raises(ValueError, match="Cannot specify both"):
            Wavefront(
                wavelength=1e-6,
                npixels=16,
                diameter=1.0,
                pixel_scale=1 / 16,
            )

        with pytest.raises(ValueError, match="center must be scalar"):
            Wavefront(
                wavelength=1e-6,
                npixels=16,
                diameter=1.0,
                center=np.array([0.0, 0.0]),
            )

    def test_from_phasor(self):
        phasor = np.ones((8, 8), dtype=complex)
        wf = Wavefront.from_phasor(phasor=phasor, wavelength=1e-6, pixel_scale=1 / 8)
        assert isinstance(wf, Wavefront)
        assert wf.npixels == 8

    def test_apply_jones_promotes_to_polarised(self):
        wavefront = Wavefront(1.0e-6, npixels=8, diameter=1.0)
        actual = wavefront.apply_jones(dlu.horizontal_polariser())
        expected = np.zeros((4, 8, 8))
        expected = expected.at[0].set(0.5 / 8**4)
        expected = expected.at[1].set(0.5 / 8**4)

        assert isinstance(actual, PolarisedWavefront)
        assert actual.phasor.shape == (2, 2, 8, 8)
        assert np.allclose(actual.stokes(), expected)


class TestChromaticWavefront(BaseWavefrontTests):
    """Run the standard Wavefront tests on chromatic Wavefront construction paths."""

    @pytest.fixture
    def ndim(self):
        return 1

    @pytest.fixture
    def chromatic(self):
        return True

    @pytest.fixture
    def wavelengths(self):
        return np.array([1.0e-6, 1.5e-6])

    @pytest.fixture(params=["direct", "vmapped"])
    def construction(self, request):
        return request.param

    @pytest.fixture
    def wavefront(self, wavelengths, construction):
        weights = np.array([0.4, 0.6])

        if construction == "direct":
            return Wavefront(wavelengths, npixels=16, diameter=1.0)

        def make_wavefront(wavelength, weight):
            wavefront = Wavefront(wavelength, npixels=16, diameter=1.0)
            return wavefront.set(phasor=complex_pattern(16) * np.sqrt(weight))

        return eqx.filter_vmap(make_wavefront)(wavelengths, weights)

    @pytest.fixture
    def output_pixel_scale(self, construction):
        if construction == "direct":
            return 1 / 32
        return np.array([1 / 32, 1 / 40])

    @pytest.fixture
    def expected(self, wavefront, wavelengths, construction):
        if construction == "direct":
            wavefront = eqx.filter_vmap(
                lambda wavelength: Wavefront(wavelength, npixels=16, diameter=1.0)
            )(wavelengths)
        return lambda operation, *args: eqx.filter_vmap(operation)(wavefront, *args)

    def test_constructor(self, wavefront, wavelengths, construction):
        assert wavefront.phasor.shape == wavelengths.shape + (16, 16)
        assert wavefront.wavelength.shape == wavelengths.shape
        if construction == "direct":
            assert wavefront.pixel_scale.shape == ()
            assert wavefront.center.shape == ()
        else:
            assert wavefront.pixel_scale.shape == wavelengths.shape
            assert wavefront.center.shape == wavelengths.shape

    def test_mapped_axis(self, wavefront, construction):
        mapped_axis = wavefront._mapped_axis

        assert mapped_axis.phasor == 0
        assert mapped_axis.wavelength == 0
        expected_metadata_axis = None if construction == "direct" else 0
        assert mapped_axis.pixel_scale == expected_metadata_axis
        assert mapped_axis.center == expected_metadata_axis

    def test_from_phasor(self, wavelengths):
        phasor = np.ones((8, 8), dtype=complex)
        wf = Wavefront.from_phasor(phasor, wavelengths, pixel_scale=1 / 8)
        assert wf.phasor.shape == wavelengths.shape + (8, 8)
        assert np.allclose(wf.phasor, phasor)


class TestPolarisedWavefront(BaseWavefrontTests):
    """Run the standard Wavefront tests on a polarised Wavefront."""

    @pytest.fixture
    def ndim(self):
        return 0

    @pytest.fixture
    def chromatic(self):
        return False

    @pytest.fixture
    def wavefront(self):
        return PolarisedWavefront.from_phasor(
            jones_pattern(16),
            wavelength=1.0e-6,
            pixel_scale=1 / 16,
        )

    @pytest.fixture
    def output_pixel_scale(self):
        return 1 / 32

    @pytest.fixture
    def expected(self, wavefront):
        return lambda operation, *args: polarised_componentwise_expected(
            wavefront, operation, *args
        )

    def test_coordinates(self, wavefront):
        coords = wavefront.coordinates(polar=True)
        expected = Wavefront(1.0e-6, npixels=16, pixel_scale=1 / 16).coordinates(
            polar=True
        )
        assert coords.shape == (2, 16, 16)
        assert np.allclose(coords, expected)


class TestPolarisedWavefrontConstruction:
    def test_constructor(self):
        wavefront = PolarisedWavefront(1.0e-6, npixels=16, diameter=1.0)
        assert wavefront.phasor.shape == (2, 2, 16, 16)
        assert wavefront.ndim == 0
        assert wavefront.npixels == 16
        assert wavefront.is_chromatic is False
        assert_polarised_identity(
            wavefront.phasor, np.ones((16, 16), dtype=complex) / 16**2
        )

    def test_chromatic_constructor(self):
        wavelengths = np.array([1.0e-6, 1.5e-6, 2.0e-6])
        wavefront = PolarisedWavefront(wavelengths, npixels=16, diameter=1.0)
        assert wavefront.phasor.shape == (3, 2, 2, 16, 16)
        assert wavefront.ndim == 1
        assert wavefront.npixels == 16
        assert wavefront.is_chromatic is True
        assert_polarised_identity(
            wavefront.phasor, np.ones((3, 16, 16), dtype=complex) / 16**2
        )


class TestPolarisedWavefrontFromWavefront:
    def test_from_wavefront(self):
        wavefront = Wavefront(1.0e-6, npixels=16, diameter=1.0)
        wavefront = wavefront.set(phasor=np.arange(16**2).reshape(16, 16))

        actual = PolarisedWavefront.from_wavefront(wavefront)
        assert actual.phasor.shape == (2, 2, 16, 16)
        assert actual.ndim == 0
        assert np.allclose(actual.wavelength, wavefront.wavelength)
        assert np.allclose(actual.pixel_scale, wavefront.pixel_scale)
        assert np.allclose(actual.center, wavefront.center)
        assert_polarised_identity(actual.phasor, wavefront.phasor)

    def test_from_chromatic_wavefront(self):
        wavelengths = np.array([1.0e-6, 1.5e-6, 2.0e-6])
        wavefront = Wavefront(wavelengths, npixels=16, diameter=1.0)
        wavefront = wavefront.set(
            phasor=wavefront.phasor * np.arange(1, wavelengths.size + 1)[:, None, None]
        )

        actual = PolarisedWavefront.from_wavefront(wavefront)
        assert actual.phasor.shape == (3, 2, 2, 16, 16)
        assert actual.ndim == 1
        assert np.allclose(actual.wavelength, wavefront.wavelength)
        assert np.allclose(actual.pixel_scale, wavefront.pixel_scale)
        assert np.allclose(actual.center, wavefront.center)
        assert_polarised_identity(actual.phasor, wavefront.phasor)


class TestPolarisedWavefrontMagicMethods:
    @pytest.mark.parametrize(
        ("method", "operation"),
        [
            ("__add__", np.add),
            ("__sub__", np.subtract),
            ("__mul__", np.multiply),
            ("__iadd__", np.add),
            ("__isub__", np.subtract),
            ("__imul__", np.multiply),
        ],
    )
    def test_promotes_mixed_wavefront_operands(self, method, operation):
        wavefront = Wavefront(1.0e-6, npixels=8, diameter=1.0)
        polarised = PolarisedWavefront(1.0e-6, npixels=8, diameter=1.0)
        polarised = polarised.set(phasor=np.ones_like(polarised.phasor) * (2 + 1j))

        normal_first = getattr(wavefront, method)(polarised)
        polarised_first = getattr(polarised, method)(wavefront)

        assert isinstance(normal_first, PolarisedWavefront)
        assert isinstance(polarised_first, PolarisedWavefront)
        assert np.allclose(
            normal_first.phasor,
            operation(wavefront.phasor, polarised.phasor),
        )
        assert np.allclose(
            polarised_first.phasor,
            operation(polarised.phasor, wavefront.phasor),
        )
        assert np.all(np.isfinite(normal_first.phasor))
        assert np.all(np.isfinite(polarised_first.phasor))

    @pytest.mark.parametrize("method", ["__truediv__", "__itruediv__"])
    def test_rejects_wavefront_division(self, method):
        wavefront = Wavefront(1.0e-6, npixels=8, diameter=1.0)
        polarised = PolarisedWavefront(1.0e-6, npixels=8, diameter=1.0)

        with pytest.raises(TypeError):
            getattr(wavefront, method)(polarised)
        with pytest.raises(TypeError):
            getattr(polarised, method)(wavefront)

    @pytest.mark.parametrize("nwavelengths", [2, 3])
    def test_chromatic_mixed_polarisation_dimensions(self, nwavelengths):
        wavelengths = np.linspace(1.0e-6, 2.0e-6, nwavelengths)
        wavefront = Wavefront(wavelengths, npixels=8, diameter=1.0)
        polarised = PolarisedWavefront(wavelengths, npixels=8, diameter=1.0)

        normal_first = wavefront * polarised
        polarised_first = polarised * wavefront
        expected = wavefront.phasor[..., None, None, :, :] * polarised.phasor

        assert isinstance(normal_first, PolarisedWavefront)
        assert isinstance(polarised_first, PolarisedWavefront)
        assert normal_first.phasor.shape == (nwavelengths, 2, 2, 8, 8)
        assert polarised_first.phasor.shape == (nwavelengths, 2, 2, 8, 8)
        assert np.allclose(normal_first.phasor, expected)
        assert np.allclose(polarised_first.phasor, expected)

    def test_monochromatic_operand_broadcasts_over_chromatic_base(self):
        wavelengths = np.array([1.0e-6, 1.5e-6, 2.0e-6])
        wavefront = Wavefront(wavelengths, npixels=8, diameter=1.0)
        polarised = PolarisedWavefront(1.0e-6, npixels=8, diameter=1.0)

        output = wavefront * polarised

        assert isinstance(output, PolarisedWavefront)
        assert output.wavelength.shape == wavelengths.shape
        assert output.phasor.shape == (3, 2, 2, 8, 8)

    def test_rejects_incompatible_chromatic_operand(self):
        monochromatic = Wavefront(1.0e-6, npixels=8, diameter=1.0)
        chromatic = PolarisedWavefront(
            np.array([1.0e-6, 1.5e-6]), npixels=8, diameter=1.0
        )
        incompatible = Wavefront(
            np.array([1.0e-6, 1.5e-6, 2.0e-6]), npixels=8, diameter=1.0
        )

        with pytest.raises(ValueError, match="requires a chromatic base"):
            monochromatic * chromatic
        with pytest.raises(ValueError, match="same wavelength shape"):
            incompatible * chromatic


class TestWavefrontArrayOperandDimensions:
    @pytest.mark.parametrize(
        "shape",
        [
            (),
            (3,),
            (8, 8),
            (3, 8, 8),
            (2, 2),
            (3, 2, 2),
            (2, 2, 8, 8),
            (3, 2, 2, 8, 8),
        ],
    )
    def test_chromatic_polarised_array_layouts(self, shape):
        wavelengths = np.array([1.0e-6, 1.5e-6, 2.0e-6])
        wavefront = PolarisedWavefront(wavelengths, npixels=8, diameter=1.0)

        output = wavefront * np.ones(shape)

        assert isinstance(output, PolarisedWavefront)
        assert output.phasor.shape == (3, 2, 2, 8, 8)

    @pytest.mark.parametrize("shape", [(3,), (8, 8), (3, 8, 8)])
    def test_chromatic_regular_array_layouts(self, shape):
        wavelengths = np.array([1.0e-6, 1.5e-6, 2.0e-6])
        wavefront = Wavefront(wavelengths, npixels=8, diameter=1.0)

        output = wavefront * np.ones(shape)

        assert type(output) is Wavefront
        assert output.phasor.shape == (3, 8, 8)

    def test_spatial_array_broadcasts_over_single_pixel_wavefront(self):
        wavefront = Wavefront(1.0e-6, npixels=1, diameter=1.0)

        output = wavefront * np.ones((8, 8))

        assert output.phasor.shape == (8, 8)

    @pytest.mark.parametrize(
        "shape", [(2, 2), (3, 2, 2), (2, 2, 8, 8), (3, 2, 2, 8, 8)]
    )
    def test_jones_arrays_promote_chromatic_regular_wavefront(self, shape):
        wavelengths = np.array([1.0e-6, 1.5e-6, 2.0e-6])
        wavefront = Wavefront(wavelengths, npixels=8, diameter=1.0)

        output = wavefront * np.ones(shape)

        assert isinstance(output, PolarisedWavefront)
        assert output.phasor.shape == (3, 2, 2, 8, 8)

    @pytest.mark.parametrize("shape", [(8, 8), (2, 2), (2, 2, 8, 8)])
    def test_monochromatic_array_layouts(self, shape):
        wavefront = PolarisedWavefront(1.0e-6, npixels=8, diameter=1.0)

        output = wavefront * np.ones(shape)

        assert isinstance(output, PolarisedWavefront)
        assert output.phasor.shape == (2, 2, 8, 8)

    def test_invalid_array_layouts(self):
        monochromatic = Wavefront(1.0e-6, npixels=8, diameter=1.0)
        wavelengths = np.array([1.0e-6, 1.5e-6, 2.0e-6])
        chromatic = Wavefront(wavelengths, npixels=8, diameter=1.0)

        with pytest.raises(ValueError, match="vector operand"):
            monochromatic * np.ones(3)
        with pytest.raises(ValueError, match="vector operand"):
            chromatic * np.ones(4)
        with pytest.raises(ValueError, match="Unsupported array shape"):
            monochromatic * np.ones((3, 8, 8))
        with pytest.raises(ValueError, match="Unsupported array shape"):
            chromatic * np.ones((4, 5))

    def test_ambiguous_array_layouts(self):
        monochromatic = Wavefront(1.0e-6, npixels=2, diameter=1.0)
        chromatic = Wavefront(
            np.array([1.0e-6, 1.5e-6, 2.0e-6]), npixels=2, diameter=1.0
        )

        with pytest.raises(ValueError, match="ambiguous for npixels=2"):
            monochromatic * np.ones((2, 2))
        with pytest.raises(ValueError, match="ambiguous between"):
            chromatic * np.ones((3, 2, 2))

    def test_wavefront_operands_require_matching_spatial_shapes(self):
        wavefront = Wavefront(1.0e-6, npixels=8, diameter=1.0)
        other = Wavefront(1.0e-6, npixels=4, diameter=1.0)

        with pytest.raises(ValueError, match="matching spatial shapes"):
            wavefront * other


class TestPolarisedWavefrontFromPhasor:
    def test_from_regular_phasor(self):
        phasor = np.ones((8, 8), dtype=complex)
        actual = PolarisedWavefront.from_phasor(phasor, 1.0e-6, pixel_scale=1 / 8)
        assert actual.phasor.shape == (2, 2, 8, 8)
        assert actual.ndim == 0
        assert_polarised_identity(actual.phasor, phasor)

    def test_from_chromatic_regular_phasor(self):
        wavelengths = np.array([1.0e-6, 1.5e-6, 2.0e-6])
        phasor = np.ones((8, 8), dtype=complex)
        actual = PolarisedWavefront.from_phasor(phasor, wavelengths, pixel_scale=1 / 8)
        assert actual.phasor.shape == (3, 2, 2, 8, 8)
        assert actual.ndim == 1
        assert_polarised_identity(actual.phasor, np.ones((3, 8, 8), dtype=complex))

    def test_from_jones_phasor(self):
        phasor = np.ones((2, 2, 8, 8), dtype=complex)
        actual = PolarisedWavefront.from_phasor(phasor, 1.0e-6, pixel_scale=1 / 8)
        assert actual.phasor.shape == phasor.shape
        assert actual.ndim == 0
        assert np.allclose(actual.phasor, phasor)

    def test_from_chromatic_jones_phasor(self):
        wavelengths = np.array([1.0e-6, 1.5e-6, 2.0e-6])
        phasor = np.ones((2, 2, 8, 8), dtype=complex)
        actual = PolarisedWavefront.from_phasor(phasor, wavelengths, pixel_scale=1 / 8)
        assert actual.phasor.shape == (3, 2, 2, 8, 8)
        assert actual.ndim == 1
        assert np.allclose(actual.phasor, np.ones((3, 2, 2, 8, 8)))


class TestPolarisedWavefrontStokes:
    def test_stokes(self):
        wavefront = PolarisedWavefront(1.0e-6, npixels=8, diameter=1.0)
        expected = np.zeros((4, 8, 8))
        expected = expected.at[0].set(1 / 8**4)
        assert wavefront.stokes().shape == (4, 8, 8)
        assert np.allclose(wavefront.stokes(), expected)

    def test_chromatic_stokes(self):
        wavelengths = np.array([1.0e-6, 1.5e-6, 2.0e-6])
        wavefront = PolarisedWavefront(wavelengths, npixels=8, diameter=1.0)
        expected = np.zeros((3, 4, 8, 8))
        expected = expected.at[:, 0].set(1 / 8**4)
        assert wavefront.stokes().shape == (3, 4, 8, 8)
        assert np.allclose(wavefront.stokes(), expected)

    def test_psf(self):
        wavefront = PolarisedWavefront(1.0e-6, npixels=8, diameter=1.0)
        expected = np.ones((8, 8)) / 8**4
        assert wavefront.psf.shape == (8, 8)
        assert np.allclose(wavefront.psf, expected)
        assert np.allclose(wavefront.psf_from_stokes(), expected)
        assert np.allclose(wavefront.psf, wavefront.stokes()[0])

    def test_chromatic_psf(self):
        wavelengths = np.array([1.0e-6, 1.5e-6, 2.0e-6])
        wavefront = PolarisedWavefront(wavelengths, npixels=8, diameter=1.0)
        expected = np.ones((3, 8, 8)) / 8**4
        assert wavefront.psf.shape == (3, 8, 8)
        assert np.allclose(wavefront.psf, expected)
        assert np.allclose(wavefront.psf_from_stokes(), expected)
        assert np.allclose(wavefront.psf, wavefront.stokes()[:, 0])

    def test_input_stokes(self):
        wavefront = PolarisedWavefront(1.0e-6, npixels=8, diameter=1.0)
        stokes = np.array([1.0, 1.0, 0.0, 0.0])
        expected = np.zeros((4, 8, 8))
        expected = expected.at[0].set(1 / 8**4)
        expected = expected.at[1].set(1 / 8**4)
        assert np.allclose(wavefront.stokes(stokes), expected)


class TestPolarisedWavefrontApplyJones:
    def test_apply_jones(self):
        wavefront = PolarisedWavefront(1.0e-6, npixels=8, diameter=1.0)
        actual = wavefront.apply_jones(dlu.horizontal_polariser())
        expected = np.zeros((4, 8, 8))
        expected = expected.at[0].set(0.5 / 8**4)
        expected = expected.at[1].set(0.5 / 8**4)
        assert isinstance(actual, PolarisedWavefront)
        assert actual.phasor.shape == (2, 2, 8, 8)
        assert np.allclose(actual.stokes(), expected)

    def test_chromatic_apply_jones(self):
        wavelengths = np.array([1.0e-6, 1.5e-6, 2.0e-6])
        wavefront = PolarisedWavefront(wavelengths, npixels=8, diameter=1.0)
        actual = wavefront.apply_jones(dlu.vertical_polariser())
        expected = np.zeros((3, 4, 8, 8))
        expected = expected.at[:, 0].set(0.5 / 8**4)
        expected = expected.at[:, 1].set(-0.5 / 8**4)
        assert isinstance(actual, PolarisedWavefront)
        assert actual.phasor.shape == (3, 2, 2, 8, 8)
        assert np.allclose(actual.stokes(), expected)

    def test_apply_spatial_jones(self):
        wavefront = PolarisedWavefront(1.0e-6, npixels=8, diameter=1.0)
        jones = dlu.linear_polariser(np.zeros((8, 8)))
        actual = wavefront.apply_jones(jones)
        expected = np.zeros((4, 8, 8))
        expected = expected.at[0].set(0.5 / 8**4)
        expected = expected.at[1].set(0.5 / 8**4)
        assert actual.phasor.shape == (2, 2, 8, 8)
        assert np.allclose(actual.stokes(), expected)

    def test_chromatic_apply_spatial_jones(self):
        wavelengths = np.array([1.0e-6, 1.5e-6, 2.0e-6])
        wavefront = PolarisedWavefront(wavelengths, npixels=8, diameter=1.0)
        jones = dlu.linear_polariser(np.zeros((8, 8)))
        actual = wavefront.apply_jones(jones)
        expected = np.zeros((3, 4, 8, 8))
        expected = expected.at[:, 0].set(0.5 / 8**4)
        expected = expected.at[:, 1].set(0.5 / 8**4)
        assert actual.phasor.shape == (3, 2, 2, 8, 8)
        assert np.allclose(actual.stokes(), expected)


class TestChromaticPolarisedWavefront(BaseWavefrontTests):
    """Run the standard Wavefront tests on a chromatic polarised Wavefront."""

    @pytest.fixture
    def ndim(self):
        return 1

    @pytest.fixture
    def chromatic(self):
        return True

    @pytest.fixture
    def wavelengths(self):
        return np.array([1.0e-6, 1.5e-6, 2.0e-6])

    @pytest.fixture
    def wavefront(self, wavelengths):
        weights = np.array([0.2, 0.3, 0.5])
        phasor = weights[:, None, None, None, None] * jones_pattern(16)
        return PolarisedWavefront.from_phasor(
            phasor,
            wavelength=wavelengths,
            pixel_scale=1 / 16,
        )

    @pytest.fixture
    def output_pixel_scale(self):
        return 1 / 32

    @pytest.fixture
    def expected(self, wavefront):
        return lambda operation, *args: polarised_componentwise_expected(
            wavefront, operation, *args
        )

    def test_coordinates(self, wavefront):
        coords = wavefront.coordinates(polar=True)
        expected = Wavefront(1.0e-6, npixels=16, pixel_scale=1 / 16).coordinates(
            polar=True
        )
        assert coords.shape == (2, 16, 16)
        assert np.allclose(coords, expected)
