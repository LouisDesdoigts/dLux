from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest
from dLux import Wavefront
from dLux.psfs import PSF
from dLux.coordinates import CoordSpec


@pytest.fixture
def wavefront():
    return Wavefront(npixels=16, diameter=1.0, wavelength=1e-6)


class TestWavefront:
    def test_constructor(self, wavefront):
        assert wavefront.npixels == 16
        assert wavefront.diameter == 1.0
        assert wavefront.wavelength == 1e-6

        wf_px = Wavefront(wavelength=1e-6, npixels=16, pixel_scale=1 / 16)
        assert wf_px.pixel_scale == 1 / 16

        wf_center = Wavefront(
            wavelength=1e-6,
            npixels=16,
            diameter=1.0,
            center=np.array([0.0]),
        )
        assert np.allclose(wf_center.center, np.array([0.0]))

        with pytest.raises(ValueError, match="Provide one"):
            Wavefront(wavelength=1e-6, npixels=16)

        with pytest.raises(ValueError, match="Cannot specify both"):
            Wavefront(
                wavelength=1e-6,
                npixels=16,
                diameter=1.0,
                pixel_scale=1 / 16,
            )

        with pytest.raises(ValueError, match="center must have shape"):
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

    def test_properties(self, wavefront):
        assert wavefront.real.shape == wavefront.phasor.shape
        assert wavefront.imaginary.shape == wavefront.phasor.shape
        assert isinstance(wavefront.to_psf(), PSF)
        assert wavefront.ndim == 0

    def test_methods(self, wavefront):
        assert isinstance(wavefront.add_opd(0), Wavefront)
        assert isinstance(wavefront.add_phase(0), Wavefront)
        assert isinstance(wavefront.tilt(np.zeros(2)), Wavefront)
        with pytest.raises(ValueError):
            wavefront.tilt(np.zeros(3))
        assert isinstance(wavefront.normalise(), Wavefront)
        assert isinstance(wavefront.normalise(mode="peak"), Wavefront)
        with pytest.raises(ValueError, match="mode must be"):
            wavefront.normalise(mode="invalid")
        assert isinstance(wavefront.flip(0), Wavefront)
        assert wavefront.scale_to(8, 1 / 32).npixels == 8
        assert np.allclose(wavefront.scale_to(8, 1 / 32).pixel_scale, 1 / 32)
        assert np.allclose(wavefront.scale_to(8, 1 / 32, False).pixel_scale, 1 / 32)
        assert isinstance(wavefront.rotate(np.pi), Wavefront)
        assert isinstance(wavefront.rotate(np.pi, complex=False), Wavefront)
        assert isinstance(wavefront.resize(8), Wavefront)
        assert wavefront.coordinates(polar=True).shape[0] == 2
        assert isinstance(
            wavefront.set_spec(CoordSpec(n=16, d=1 / 16, c=0.0)), Wavefront
        )

    def test_set_spec_normalises_coordinates(self, wavefront):
        spec = CoordSpec(n=16, d=1 / 16, c=0.0).set(d=0.5, c=1.0)

        result = wavefront.set_spec(spec)

        assert result.pixel_scale.shape == ()
        assert result.center.shape == ()
        assert np.allclose(result.pixel_scale, 0.5)
        assert np.allclose(result.center, 1.0)

    def test_magic_add(self, wavefront):
        # Test arrays
        wavefront += np.ones(1)
        assert isinstance(wavefront, Wavefront)

        # Test numeric
        wavefront += 1
        assert isinstance(wavefront, Wavefront)

        # Test invalid
        with pytest.raises(TypeError):
            wavefront += "1"

    def test_magic_mul(self, wavefront):
        # Test Nones
        wavefront *= None
        assert isinstance(wavefront, Wavefront)

        # Test arrays
        wavefront *= np.ones(1)
        assert isinstance(wavefront, Wavefront)

        # Test complex arrays
        wavefront *= np.ones(1) * np.exp(1j)
        assert isinstance(wavefront, Wavefront)

        # Test numeric
        wavefront *= 1
        assert isinstance(wavefront, Wavefront)

        # Test invalid
        with pytest.raises(TypeError):
            wavefront *= "1"

    def test_magic_sub_and_div(self, wavefront):
        other = Wavefront(npixels=16, diameter=1.0, wavelength=1e-6)

        out_sub = wavefront - other
        assert isinstance(out_sub, Wavefront)

        out_div = wavefront / other
        assert isinstance(out_div, Wavefront)

        wavefront -= np.ones(1)
        assert isinstance(wavefront, Wavefront)

        wavefront /= np.ones(1)
        assert isinstance(wavefront, Wavefront)

        with pytest.raises(ValueError, match="Unsupported operation"):
            wavefront._magic_unified_op(np.ones(1), "invalid")

    def _test_propagated(self, wf, plane, units, npix, pscale):
        assert wf.plane == plane
        assert wf.units == units

    def test_propagation(self, wavefront):
        npix = 8
        pscale = 1 / 32
        focal_length = 1.0

        focal_wf = wavefront.propagate(npix, pscale)
        assert isinstance(focal_wf, Wavefront)

        back_wf = focal_wf.propagate(npix, pscale)
        assert isinstance(back_wf, Wavefront)

        focal_wf_fl = wavefront.propagate(npix, pscale, focal_length)
        assert isinstance(focal_wf_fl, Wavefront)

        back_wf_fl = focal_wf_fl.propagate(npix, pscale, focal_length)
        assert isinstance(back_wf_fl, Wavefront)

    def test_fft_propagation(self, wavefront):
        pad = 2
        focal_length = 1.0
        npix_out = wavefront.npixels * pad

        focal_wf = wavefront.propagate_FFT()
        assert isinstance(focal_wf, Wavefront)
        assert focal_wf.npixels == npix_out

        pupil_wf = focal_wf.propagate_FFT()
        assert isinstance(pupil_wf, Wavefront)
        assert pupil_wf.npixels == npix_out * pad

        focal_wf = wavefront.propagate_FFT(focal_length=focal_length)
        assert isinstance(focal_wf, Wavefront)
        assert focal_wf.npixels == npix_out

        with pytest.raises(ValueError, match="cannot specify d"):
            wavefront.propagate_FFT(spec_out=CoordSpec(c=0.0, d=1.0))

        with pytest.raises(ValueError, match="cannot specify n"):
            wavefront.propagate_FFT(spec_out=CoordSpec(c=0.0, n=16))

    def test_mft_propagation_with_spec(self, wavefront):
        out = wavefront.propagate_MFT(CoordSpec(n=8, d=1 / 32, c=0.0))
        assert isinstance(out, Wavefront)
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
