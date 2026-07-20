from jax import numpy as np, config
import equinox as eqx

config.update("jax_debug_nans", True)
import pytest
from dLux import Wavefront
from dLux.psfs import PSF
from dLux.coordinates import CoordSpec
from dLux.utils import pixel_coords


def complex_pattern(npixels):
    y, x = np.mgrid[:npixels, :npixels]
    amplitude = 1 + 0.1 * x + 0.2 * y
    phase = 0.3 * x - 0.2 * y + 0.05 * x * y
    return amplitude * np.exp(1j * phase)


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
        assert wavefront.chromatic is chromatic

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
        operation = lambda wavefront: wavefront.set(
            phasor=wavefront.coordinates(polar=True).astype(complex)
        )
        self.assert_matches(operation(wavefront), expected(operation))

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


class TestChromaticWavefront(BaseWavefrontTests):
    """Run the standard Wavefront tests on a filter-vmapped chromatic Wavefront."""

    @pytest.fixture
    def ndim(self):
        return 1

    @pytest.fixture
    def chromatic(self):
        return True

    @pytest.fixture
    def wavefront(self):
        wavelengths = np.array([1.0e-6, 1.5e-6])
        weights = np.array([0.4, 0.6])

        def make_wavefront(wavelength, weight):
            wavefront = Wavefront(wavelength, npixels=16, diameter=1.0)
            return wavefront.set(phasor=complex_pattern(16) * np.sqrt(weight))

        return eqx.filter_vmap(make_wavefront)(wavelengths, weights)

    @pytest.fixture
    def output_pixel_scale(self):
        return np.array([1 / 32, 1 / 40])

    @pytest.fixture
    def expected(self, wavefront):
        return lambda operation, *args: eqx.filter_vmap(operation)(wavefront, *args)
