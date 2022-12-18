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


@pytest.fixture
def create_tilt_wavefront() -> callable:
    """
    Returns:
    --------
    create_tilt_wavefront: callable 
        A function that has all keyword arguments and can be 
        used to create a `TiltWavefront` layer for testing.
    """
    def _create_tilt_wavefront(
            tilt_angles: Array = np.ones(2)) -> OpticalLayer:
        return dLux.optics.TiltWavefront(tilt_angles)
    return _create_tilt_wavefront


@pytest.fixture
def create_normalise_wavefront() -> callable:
    """
    Returns:
    --------
    create_normalise_wavefront: callable 
        A function that has all keyword arguments and can be 
        used to create a `NormaliseWavefront` layer for testing.
    """
    def _create_normalise_wavefront():
        return dLux.optics.NormaliseWavefront()
    return _create_normalise_wavefront


@pytest.fixture
def create_apply_basis_opd():
    """
    Returns:
    --------
    create_apply_basis_opd: callable 
        A function that has all keyword arguments and can be 
        used to create a `ApplyBasisOPD` layer for testing.
    """
    def _create_apply_basis_opd(
            basis = np.ones((3, 16, 16)),
            coefficients = np.ones(3)) -> OpticalLayer:
        return dLux.optics.ApplyBasisOPD(basis, coefficients)
    return _create_apply_basis_opd


@pytest.fixture
def create_add_phase():
    """
    Returns:
    --------
    create_add_phase: callable 
        A function that has all keyword arguments and can be 
        used to create a `AddPhase` layer for testing.
    """
    def _create_add_phase(phase: Array = np.ones((16, 16))) -> OpticalLayer:
        return dLux.optics.AddPhase(phase)
    return _create_add_phase    


@pytest.fixture
def create_add_opd():
    """
    Returns:
    --------
    create_add_opd: callable
        a function that has all keyword arguments and can be
        used to create a `AddOPD` layer for testing.
    """
    def _create_add_opd(opd: Array = np.ones((16, 16))) -> OpticalLayer:
        return dLux.optics.AddOPD(opd)
    return _create_add_opd


@pytest.fixture
def create_transmissive_optic():
    """
    Returns:
    --------
    create_transmissive_optic: callable
        a function that has all keyword arguments and can be
        used to create a `TransmissiveOptic` layer for testing.
    """
    def _create_trans_optic(trans: Array = np.ones((16, 16))) -> OpticalLayer:
        return dLux.optics.TransmissiveOptic(trans)
    return _create_trans_optic 


@pytest.fixture
def create_basis_climb():
    """
    Returns:
    --------
    create_basis_climb: callable
        a function that has all keyword arguments and can be
        used to create a `BasisCLIMB` layer for testing.
    """
    def _create_basis_climb(
            basis: Array = np.ones((3, 16, 16)),
            coefficients: Array = np.ones(3),
            ideal_wavelength: Array = np.array(5e-7)) -> OpticalLayer:
        return dLux.optics.ApplyBasisCLIMB(
            basis, ideal_wavelength, coefficients)
    return _create_basis_climb


