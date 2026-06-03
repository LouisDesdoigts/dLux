from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest
from dLux import Wavefront
from dLux.psfs import PSF
from dLux.coordinates import CoordSpec


class BaseWavefrontTests:
    """
    Abstract base test suite for all Wavefront variants.
    Pytest will not run this directly because the name does not start with 'Test'.
    """

    def test_constructor(self, wavefront, wavefront_cls, wavefront_factory):
        assert wavefront.npixels == 16
        assert wavefront.diameter == 1.0
        assert wavefront.wavelength == 1e-6

        wf_px = wavefront_factory(wavelength=1e-6, npixels=16, pixel_scale=1 / 16)
        assert wf_px.pixel_scale == 1 / 16

        wf_center = wavefront_factory(
            wavelength=1e-6,
            npixels=16,
            diameter=1.0,
            center=np.array([0.0]),
        )
        assert np.allclose(wf_center.center, np.array([0.0]))

        with pytest.raises(ValueError, match="Provide one"):
            wavefront_factory(wavelength=1e-6, npixels=16)

        with pytest.raises(ValueError, match="Cannot specify both"):
            wavefront_factory(
                wavelength=1e-6,
                npixels=16,
                diameter=1.0,
                pixel_scale=1 / 16,
            )

        with pytest.raises(ValueError, match="center must have shape"):
            wavefront_factory(
                wavelength=1e-6,
                npixels=16,
                diameter=1.0,
                center=np.array([0.0, 0.0]),
            )

    def test_from_phasor(self, wavefront_cls):
        phasor = np.ones((8, 8), dtype=complex)
        wf = wavefront_cls.from_phasor(phasor=phasor, wavelength=1e-6, pixel_scale=1 / 8)
        assert isinstance(wf, wavefront_cls)
        assert wf.npixels == 8

    def test_properties(self, wavefront):
        assert wavefront.real.shape == wavefront.phasor.shape
        assert wavefront.imaginary.shape == wavefront.phasor.shape
        assert isinstance(wavefront.to_psf(), PSF)
        assert wavefront.ndim == 0

    def test_methods(self, wavefront, wavefront_cls):
        assert isinstance(wavefront.add_opd(0), wavefront_cls)
        assert isinstance(wavefront.add_phase(0), wavefront_cls)
        assert isinstance(wavefront.tilt(np.zeros(2)), wavefront_cls)
        with pytest.raises(ValueError):
            wavefront.tilt(np.zeros(3))
        assert isinstance(wavefront.normalise(), wavefront_cls)
        assert isinstance(wavefront.normalise(mode="peak"), wavefront_cls)
        with pytest.raises(ValueError, match="mode must be"):
            wavefront.normalise(mode="invalid")
        assert isinstance(wavefront.flip(0), wavefront_cls)
        assert wavefront.scale_to(8, 1 / 32).npixels == 8
        assert np.allclose(wavefront.scale_to(8, 1 / 32).pixel_scale, 1 / 32)
        assert np.allclose(wavefront.scale_to(8, 1 / 32, False).pixel_scale, 1 / 32)
        assert isinstance(wavefront.rotate(np.pi), wavefront_cls)
        assert isinstance(wavefront.rotate(np.pi, complex=False), wavefront_cls)
        assert isinstance(wavefront.resize(8), wavefront_cls)
        assert wavefront.coordinates(polar=True).shape[0] == 2
        assert isinstance(
            wavefront.set_spec(CoordSpec(n=16, d=1 / 16, c=0.0)), wavefront_cls
        )

    def test_magic_add(self, wavefront, wavefront_cls):
        # Test arrays
        wavefront += np.ones(1)
        assert isinstance(wavefront, wavefront_cls)

        # Test numeric
        wavefront += 1
        assert isinstance(wavefront, wavefront_cls)

        # Test invalid
        with pytest.raises(TypeError):
            wavefront += "1"

    def test_magic_mul(self, wavefront, wavefront_cls):
        # Test Nones
        wavefront *= None
        assert isinstance(wavefront, wavefront_cls)

        # Test arrays
        wavefront *= np.ones(1)
        assert isinstance(wavefront, wavefront_cls)

        # Test complex arrays
        wavefront *= np.ones(1) * np.exp(1j)
        assert isinstance(wavefront, wavefront_cls)

        # Test numeric
        wavefront *= 1
        assert isinstance(wavefront, wavefront_cls)

        # Test invalid
        with pytest.raises(TypeError):
            wavefront *= "1"

    def test_magic_sub_and_div(self, wavefront, wavefront_cls, wavefront_factory):
        other = wavefront_factory(npixels=16, diameter=1.0, wavelength=1e-6)

        out_sub = wavefront - other
        assert isinstance(out_sub, wavefront_cls)

        out_div = wavefront / other
        assert isinstance(out_div, wavefront_cls)

        wavefront -= np.ones(1)
        assert isinstance(wavefront, wavefront_cls)

        wavefront /= np.ones(1)
        assert isinstance(wavefront, wavefront_cls)

        with pytest.raises(ValueError, match="Unsupported operation"):
            wavefront._magic_unified_op(np.ones(1), "invalid")

    def test_propagation(self, wavefront, wavefront_cls):
        npix = 8
        pscale = 1 / 32
        focal_length = 1.0

        focal_wf = wavefront.propagate(npix, pscale)
        assert isinstance(focal_wf, wavefront_cls)

        back_wf = focal_wf.propagate(npix, pscale)
        assert isinstance(back_wf, wavefront_cls)

        focal_wf_fl = wavefront.propagate(npix, pscale, focal_length)
        assert isinstance(focal_wf_fl, wavefront_cls)

        back_wf_fl = focal_wf_fl.propagate(npix, pscale, focal_length)
        assert isinstance(back_wf_fl, wavefront_cls)

    def test_fft_propagation(self, wavefront, wavefront_cls):
        pad = 2
        focal_length = 1.0
        npix_out = wavefront.npixels * pad

        focal_wf = wavefront.propagate_FFT()
        assert isinstance(focal_wf, wavefront_cls)
        assert focal_wf.npixels == npix_out

        pupil_wf = focal_wf.propagate_FFT()
        assert isinstance(pupil_wf, wavefront_cls)
        assert pupil_wf.npixels == npix_out * pad

        focal_wf = wavefront.propagate_FFT(focal_length=focal_length)
        assert isinstance(focal_wf, wavefront_cls)
        assert focal_wf.npixels == npix_out

        with pytest.raises(ValueError, match="cannot specify d"):
            wavefront.propagate_FFT(spec_out=CoordSpec(c=0.0, d=1.0))

        with pytest.raises(ValueError, match="cannot specify n"):
            wavefront.propagate_FFT(spec_out=CoordSpec(c=0.0, n=16))

    def test_mft_propagation_with_spec(self, wavefront, wavefront_cls):
        out = wavefront.propagate_MFT(CoordSpec(n=8, d=1 / 32, c=0.0))
        assert isinstance(out, wavefront_cls)
        assert out.npixels == 8

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
    def wavefront_cls(self):
        return Wavefront

    @pytest.fixture
    def wavefront_factory(self, wavefront_cls):
        return lambda **kwargs: wavefront_cls(**kwargs)

    @pytest.fixture
    def wavefront(self, wavefront_factory):
        return wavefront_factory(npixels=16, diameter=1.0, wavelength=1e-6)