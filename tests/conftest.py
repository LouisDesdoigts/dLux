import pytest
import jax.numpy as np
import dLux


Array = np.ndarray
Wavefront = dLux.wavefronts.Wavefront


@pytest.fixture
def create_wavefront():
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


class CreateWavefrontUtility(Utility):
    """
    Utility for CreateWavefront class.
    """
    npixels        : int
    diameter       : Array
    wavefront_type : str


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the CreateWavefront Utility.
        """
        self.npixels = 16
        self.diameter = np.array(1.)
        self.wavefront_type = "Cartesian"


    def construct(self            : Utility,
                  npixels         : int   = None,
                  diameter        : Array = None,
                  wavefront_type  : str   = None) -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        npixels  = self.npixels  if npixels  is None else npixels
        diameter = self.diameter if diameter is None else diameter
        wavefront_type = self.wavefront_type if wavefront_type is None else \
        wavefront_type
        return dLux.optics.CreateWavefront(npixels, diameter, wavefront_type)
