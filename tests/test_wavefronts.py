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

    @pytest.mark.parametrize("pad", [1, 2, 3])
    def test_fresnel_AS_propagation(self, wavefront, pad):
        """
        Fresnel AS: correct plane/units, shape & pixel scale; deterministic and
        (approximately) energy-preserving.
        """
        prop_dist = 0.75  # any nonzero distance

        # Forward propagate
        out1 = wavefront.propagate_fresnel_AS(prop_dist, pad=pad)
        out2 = wavefront.propagate_fresnel_AS(prop_dist, pad=pad)

        # Plane/units and structure
        assert isinstance(out1, Wavefront)
        assert out1.plane == "Intermediate"
        assert out1.units == "Cartesian"
        assert out1.npixels == wavefront.npixels * pad
        assert np.allclose(out1.pixel_scale, wavefront.pixel_scale)

        # No NaNs
        assert not np.isnan(out1.amplitude).any()
        assert not np.isnan(out1.phase).any()

        # Determinism
        assert np.allclose(out1.amplitude, out2.amplitude, atol=1e-12)
        assert np.allclose(out1.phase, out2.phase, atol=1e-12)

        # Energy conservation (unitary up to numerical error)
        Ein = np.sum(wavefront.amplitude**2)
        Eout = np.sum(out1.amplitude**2)
        assert np.isclose(Ein, Eout, rtol=1e-10)

    def test_fresnel_AS_identity_and_roundtrip(self, wavefront):
        """
        Fresnel AS: z=0 is identity (with pad=1), and forward/backward round-trip
        returns to the original field (same size) up to numerical error.
        """
        # z = 0 ⇒ identity (no padding so shapes match)
        z0 = wavefront.propagate_fresnel_AS(0.0, pad=1)
        assert np.allclose(z0.amplitude, wavefront.amplitude, atol=1e-12)
        assert np.allclose(z0.phase, wavefront.phase, atol=1e-12)

        # Round-trip: +z then -z (pad=1 keeps constant size)
        z = 0.5
        fwd = wavefront.propagate_fresnel_AS(z, pad=1)
        bwd = fwd.propagate_fresnel_AS(-z, pad=1)

        # Compare complex fields to avoid phase-wrapping edge cases
        wf0 = wavefront.amplitude * np.exp(1j * wavefront.phase)
        wfB = bwd.amplitude * np.exp(1j * bwd.phase)
        assert np.allclose(wf0, wfB, atol=1e-10)
