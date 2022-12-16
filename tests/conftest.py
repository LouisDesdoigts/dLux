import pytest
import jax.numpy as np
import dLux


Array = np.ndarray
Wavefront = dLux.wavefronts.Wavefront


@pytest.fixture
def create_wavefront() -> callable:
    """
    Returns:
    --------
    create_wavefront: callable
        A function that has all keyword arguments and can be 
        used to create a wavefront for testing.
    """
    def _create_wavefront(
            wavelength: Array = np.array(550e-09),
            pixel_scale: Array = np.array(1.),
            plane_type: int = dLux.PlaneType.Pupil,
            amplitude: Array = np.ones((1, 16, 16)),
            phase: Array = np.zeros((1, 16, 16))) -> Wavefront:
        return dLux.wavefronts.Wavefront(
            wavelength, pixel_scale, amplitude, phase, plane_type)
    return _create_wavefront


@pytest.fixture
def create_create_wavefront() -> callable:
    """
    Returns:
    --------
    create_create_wavefront: callable 
        A function that has all keyword arguments and can be 
        used to create a `CreateWavefront` layer for testing.
    """
    def _create_create_wavefront(
            npixels = 16,
            diameter = np.array(1.),
            wavefront_type = "Cartesian") -> OpticalLayer:
        return dLux.optics.CreateWavefront(npixels, diameter, wavefront_type)
    return _create_create_wavefront


class TiltWavefrontUtility(Utility):
    """
    Utility for TiltWavefront class.
    """
    tilt_angles : Array


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the TiltWavefront Utility.
        """
        self.tilt_angles = np.ones(2)


    def construct(self : Utility, tilt_angles : Array = None) -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        tilt_angles = self.tilt_angles if tilt_angles is None else tilt_angles
        return dLux.optics.TiltWavefront(tilt_angles)
