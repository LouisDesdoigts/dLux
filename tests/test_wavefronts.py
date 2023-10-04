from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest
from dLux import Wavefront, Optic


@pytest.fixture
def wavefront():
    return Wavefront(npixels=16, diameter=1.0, wavelength=1e-6)


# TODO: Test magic
class TestWavefront:
    def test_constructor(self, wavefront):
        assert wavefront.npixels == 16
        assert wavefront.diameter == 1.0
        assert wavefront.wavelength == 1e-6

    def test_properties(self, wavefront):
        assert wavefront.ndim == 0
        assert wavefront.real.shape == wavefront.amplitude.shape
        assert wavefront.imaginary.shape == wavefront.amplitude.shape
        assert wavefront.phasor.shape == wavefront.amplitude.shape
        assert wavefront.psf.shape == wavefront.amplitude.shape
        assert wavefront.coordinates.shape == (2, 16, 16)
        assert np.array(wavefront.wavenumber).shape == ()
        assert np.array(wavefront.fringe_size).shape == ()

    def test_methods(self, wavefront):
        assert isinstance(wavefront.add_opd(0), Wavefront)
        assert isinstance(wavefront.add_phase(0), Wavefront)
        assert isinstance(wavefront.tilt(np.zeros(2)), Wavefront)
        with pytest.raises(ValueError):
            wavefront.tilt(np.zeros(3))
        assert isinstance(wavefront.normalise(), Wavefront)
        assert isinstance(wavefront.flip(0), Wavefront)
        assert wavefront.scale_to(8, 1 / 32).npixels == 8
        assert np.allclose(wavefront.scale_to(8, 1 / 32).pixel_scale, 1 / 32)
        assert np.allclose(
            wavefront.scale_to(8, 1 / 32, True).pixel_scale, 1 / 32
        )
        assert isinstance(wavefront.rotate(np.pi), Wavefront)
        assert isinstance(wavefront.resize(8), Wavefront)

    def test_magic_add(self, wavefront):
        # Test Nones
        wavefront += None
        assert isinstance(wavefront, Wavefront)

        # Test optical layers
        wavefront += Optic()
        assert isinstance(wavefront, Wavefront)

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

        # Test optical layers
        wavefront *= Optic()
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

    def _test_propagated(self, wf, plane, units, npix, pscale):
        assert wf.plane == plane
        assert wf.units == units
        assert wf.npixels == npix
        assert np.allclose(wf.pixel_scale, pscale)

    @pytest.mark.parametrize("pixel", [True, False])
    def test_propagation(self, wavefront, pixel):
        npix = 8
        pscale = 1 / 32
        focal_length = 1.0

        # No focal length to focal plane
        focal_wf = wavefront.propagate(npix, pscale, pixel=pixel)
        self._test_propagated(focal_wf, "Focal", "Angular", npix, pscale)

        # No focal length to pupil plane
        pupil_wf = focal_wf.propagate(npix, pscale, pixel=pixel)
        self._test_propagated(pupil_wf, "Pupil", "Cartesian", npix, pscale)

        # From an angular focal plane, specifying fl
        with pytest.raises(ValueError):
            focal_wf.propagate(npix, pscale, focal_length, pixel=pixel)

        # Focal length to focal plane
        focal_wf = wavefront.propagate(npix, pscale, focal_length, pixel=pixel)
        self._test_propagated(focal_wf, "Focal", "Cartesian", npix, pscale)

        # Focal length to pupil plane
        pupil_wf = focal_wf.propagate(npix, pscale, focal_length, pixel=pixel)
        self._test_propagated(pupil_wf, "Pupil", "Cartesian", npix, pscale)

    def test_fft_propagation(self, wavefront):
        pad = 2
        focal_length = 1.0
        npix_out = wavefront.npixels * pad
        pscale_in = wavefront.pixel_scale
        pscale_out = wavefront.fringe_size / pad

        # No focal length to focal plane
        focal_wf = wavefront.propagate_FFT()
        self._test_propagated(
            focal_wf, "Focal", "Angular", npix_out, pscale_out
        )

        # No focal length to pupil plane
        pupil_wf = focal_wf.resize(wavefront.npixels).propagate_FFT()
        self._test_propagated(
            pupil_wf, "Pupil", "Cartesian", npix_out, pscale_in
        )

        # From an angular focal plane, specifying fl
        with pytest.raises(ValueError):
            focal_wf.propagate_FFT(focal_length)

        # With focal length
        pscale_in *= focal_length

        # Focal length to focal plane
        focal_wf = wavefront.propagate_FFT(focal_length)
        self._test_propagated(
            focal_wf, "Focal", "Cartesian", npix_out, pscale_out
        )

        # Focal length to pupil plane
        pupil_wf = focal_wf.resize(wavefront.npixels).propagate_FFT(
            focal_length
        )
        self._test_propagated(
            pupil_wf, "Pupil", "Cartesian", npix_out, pscale_in
        )

    @pytest.mark.parametrize("pixel", [True, False])
    def test_fresnel_propagation(self, wavefront, pixel):
        npix = 8
        pscale = 1 / 32
        focal_length = 1.0
        focal_shift = 1e-2

        # Simple test
        focal_wf = wavefront.propagate_fresnel(
            npix, pscale, focal_length, focal_shift, pixel=pixel
        )
        self._test_propagated(
            focal_wf, "Intermediate", "Cartesian", npix, pscale
        )

        # Test error from focal plane
        focal_wf = wavefront.propagate(npix, pscale, pixel=pixel)
        with pytest.raises(ValueError):
            focal_wf.propagate_fresnel(
                npix, pscale, focal_length, focal_shift, pixel=pixel
            )
