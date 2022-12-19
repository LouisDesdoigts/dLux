import pytest
import jax.numpy as np
import dLux


Array = np.ndarray
Wavefront = dLux.wavefronts.Wavefront
OpticalLayer = dLux.optics.OpticalLayer


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


@pytest.fixture
def create_rotate() -> callable:
    """
    Returns:
    --------
    create_rotate: callable
        a function that has all keyword arguments and can be
        used to create a `Rotate` layer for testing.
    """
    def _create_rotate(
            angle: Array = np.array(np.pi),
            real_imaginary: bool = False,
            fourier: bool = False,
            padding: int = 2) -> OpticalLayer:
        return dLux.optics.Rotate(angle, real_imaginary, fourier, padding)
    return _create_rotate


@pytest.fixture
def create_propagator() -> callable:
    """
    Returns:
    --------
    _create_propagator: callable
        a function that has all keyword arguments and can be
        used to create a `Propagator` layer for testing.
    """
    def _create_propagator(
            inverse : bool = False) -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        inverse = inverse if inverse is None else False
        return dLux.propagators.Propagator(inverse=inverse)
    return _create_propagator


@pytest.fixture
def create_fixed_sampling_propagator() -> callable:
    """
    Returns:
    --------
    _create_fixed_sampling_propagator_utility: callable
        a function that has all keyword arguments and can be
        used to create a `FixedSamplingPropagator` layer for testing.
    """
    def _create_fixed_sampling_propagator_utility() -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return dLux.propagators.FixedSamplingPropagator()
    return _create_fixed_sampling_propagator_utility


@pytest.fixture
def create_variable_sampling_propagator() -> callable:
    """
    Returns:
    --------
    create_variable_sampling_propagator: callable
        a function that has all keyword arguments and can be
        used to create a `VariableSamplingPropagator` layer for testing.
    """
    def _create_variable_sampling_propagator_utility(
                                                     npixels_out     : int   = 16,
                                                     pixel_scale_out : Array = np.array(1.),
                                                     shift           : Array = np.zeros(2),
                                                     pixel_shift     : bool  = False,
                                                     inverse         : bool  = None
                                                     ) -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return dLux.propagators.VariableSamplingPropagator(pixel_scale_out,
                            npixels_out, shift, pixel_shift)
    return _create_variable_sampling_propagator_utility



@pytest.fixture
def create_cartesian_propagator() -> callable:
    """
    Returns:
    --------
    _create_cartesian_propagator: callable
        a function that has all keyword arguments and can be
        used to create a `CartesianPropagator` layer for testing.
    """

    def _create_cartesian_propagator(
                                     focal_length : Array = np.array(1.),
                                     inverse      : bool  = False) -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return dLux.propagators.CartesianPropagator(focal_length, \
                                                    inverse=inverse)
    return _create_cartesian_propagator


@pytest.fixture
def create_angular_propagator() -> callable:
    """
    Returns:
    --------
    _create_angular_propagator: callable
        a function that has all keyword arguments and can be
        used to create a `AngularPropagator` layer for testing.
    """

    def _create_angular_propagator(inverse : bool = False) -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return dLux.propagators.AngularPropagator(inverse=inverse)
    return _create_angular_propagator



@pytest.fixture
def create_far_field_fresnel() -> callable:
    """
    Returns:
    --------
    _create_far_field_fresnel: callable
        a function that has all keyword arguments and can be
        used to create a `FarFieldFresnel` layer for testing.
    """

    def _create_far_field_fresnel(
                  propagation_shift : Array = np.array(1e-3),
                  inverse           : bool  = False) -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return dLux.propagators.FarFieldFresnel(propagation_shift, \
                                                inverse=inverse)
    return _create_far_field_fresnel


@pytest.fixture
def create_cartesian_mft() -> callable:
    """
    Returns:
    --------
    _create_cartesian_mft: callable
        a function that has all keyword arguments and can be
        used to create a `CartesianMFT` layer for testing.
    """

    def _create_cartesian_mft(
                            npixels_out     : int   = 16,
                            pixel_scale_out : Array = np.array(1.),
                            focal_length    : Array = np.array(1.),
                            shift           : Array = np.zeros(2),
                            pixel_shift     : bool  = False,
                            inverse         : bool  = False) -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        pixel_scale_out = pixel_scale_out if pixel_scale_out is None \
                                                        else pixel_scale_out
        focal_length = focal_length if focal_length is None \
                                                        else focal_length
        npixels_out = npixels_out if npixels_out is None else npixels_out
        shift       = shift       if shift       is None else shift
        pixel_shift = pixel_shift if pixel_shift is None else pixel_shift
        inverse     = inverse     if inverse     is None else inverse
        return dLux.propagators.CartesianMFT(npixels_out, pixel_scale_out,
                                     focal_length, inverse, shift, pixel_shift)
    return _create_cartesian_mft


@pytest.fixture
def create_angular_mft() -> callable:
    """
    Returns:
    --------
    _create_angular_mft: callable
        a function that has all keyword arguments and can be
        used to create a `AngularMFT` layer for testing.
    """
    def _create_angular_mft(
                  npixels_out     : int   = 16,
                  pixel_scale_out : float = np.array(1.),
                  inverse         : bool  = False,
                  shift           : Array = np.zeros(2),
                  pixel_shift     : bool  = False) -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return dLux.propagators.AngularMFT(npixels_out, pixel_scale_out,
                                           inverse, shift, pixel_shift)
    return _create_angular_mft


@pytest.fixture
def create_cartesian_fft():
    """
    Returns:
    --------
    _create_angular_mft: callable
        a function that has all keyword arguments and can be
        used to create a `CartesianFFT` layer for testing.
    """

    def _create_cartesian_fft(
                  focal_length : Array = np.array(1.),
                  inverse      : bool  = False) -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        focal_length = focal_length if focal_length is None \
                                                        else focal_length
        inverse = inverse if inverse is None else inverse
        return dLux.propagators.CartesianFFT(focal_length, inverse)
    return _create_cartesian_fft

@pytest.fixture
def create_angular_fft() -> callable:
    """
    Returns:
    --------
    _create_angular_fft: callable
        a function that has all keyword arguments and can be
        used to create a `AngularFFT` layer for testing.
    """

    def _create_angular_fft(inverse : bool = False) -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return dLux.propagators.AngularFFT(inverse)
    return _create_angular_fft


@pytest.fixture
def create_cartesian_fresnel() -> callable:
    """
    Returns:
    --------
    _create_cartesian_fresnel: callable
        a function that has all keyword arguments and can be
        used to create a `CartesianFresnel` layer for testing.
    """
    def _create_cartesian_fresnel(
                  npixels_out       : int   = 16,
                  pixel_scale_out   : float = np.array(1.),
                  focal_length      : Array = np.array(1.),
                  propagation_shift : Array = np.array(1e-3),
                  inverse           : bool  = False,
                  shift             : Array = np.zeros(2),
                  pixel_shift       : bool  = False) -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return dLux.propagators.CartesianFresnel(npixels_out, pixel_scale_out,
                 focal_length, propagation_shift, inverse, shift, pixel_shift)
    return _create_cartesian_fresnel


@pytest.fixture
def create_source() -> callable:
    """
    Returns:
    --------
    _create_source: callable
        a function that has all keyword arguments and can be
        used to create a `Source` layer for testing.
    """

    def _create_source(
                  position : Array    = np.array([0., 0.]),
                  flux     : Array    = np.array(1.),
                  spectrum : OpticalLayer = dLux.spectrums.ArraySpectrum(np.linspace(500e-9, \
                                                                 600e-9, 10)),
                  name     : str      = "Source") -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return dLux.sources.Source(position, flux, spectrum, name=name)
    return _create_source


@pytest.fixture
def create_relative_flux_source() -> callable:
    """
    Returns:
    --------
    _create_relative_flux_source: callable
        a function that has all keyword arguments and can be
        used to create a `RelativeFluxSource` layer for testing.
    """
    
    def _create_relative_flux_source(
                  position : Array    = np.array([0., 0.]),
                  flux     : Array    = np.array(1.),
                  spectrum : OpticalLayer = dLux.spectrums.ArraySpectrum(np.linspace(500e-9, \
                                                                 600e-9, 10)),
                  contrast   : Array    = np.array(2.),
                  name       : str      = "RelativeSource") -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return dLux.sources.RelativeFluxSource(contrast, position=position,
                                               flux=flux, spectrum=spectrum,
                                               name=name)
    return _create_relative_flux_source


@pytest.fixture
def create_relative_position_source() -> callable:
    """
    Returns:
    --------
    _create_relative_position_source: callable
        a function that has all keyword arguments and can be
        used to create a `RelativePositionSource` layer for testing.
    """

    def _create_relative_position_source(
                  position       : Array    = np.array([0., 0.]),
                  flux           : Array    = np.array(1.),
                  spectrum       : OpticalLayer = dLux.spectrums.ArraySpectrum(np.linspace(500e-9, \
                                                                 600e-9, 10)),
                  separation     : Array    = np.array(1.),
                  position_angle : Array    = np.array(0.),
                  name           : str      = "RelativePositionSource") -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return dLux.sources.RelativePositionSource(separation, position_angle,
                                                   position=position, flux=flux,
                                                   spectrum=spectrum, name=name)
    return _create_relative_position_source

@pytest.fixture
def create_point_source() -> callable:
    """
    Returns:
    --------
    _create_point_source: callable
        a function that has all keyword arguments and can be
        used to create a `PointSource` layer for testing.
    """


    def _create_point_source(
                  position    : Array    = np.array([0., 0.]),
                  flux        : Array    = np.array(1.),
                  spectrum    : OpticalLayer = dLux.spectrums.ArraySpectrum(np.linspace(500e-9, \
                                                                 600e-9, 10)),
                  name        : str      = "PointSource") -> OpticalLayer:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return dLux.sources.PointSource(position, flux, spectrum, name=name)
    return _create_point_source


def create_optics() -> callable:
    """
    Returns:
    --------
    _create_optics: callable
        A function that has all keyword arguments and can be
        used to create a `Optics` layer for testing.
    """
    def _create_optics(
            layers = [
                dLux.optics.CreateWavefront(16, 1),
                dLux.optics.CompoundAperture([0.5]),
                dLux.optics.NormaliseWavefront(),
                dLux.propagators.CartesianMFT(16, 1., 1e-6)
            ]) -> OpticalLayer:
        return dLux.core.Optics(layers)
    return _create_optics


@pytest.fixture
def create_detector() -> callable:
    """
    Returns:
    --------
    _create_detector: callable
        A function that has all keyword arguments and can be
        used to create a `Detector` layer for testing.
    """
    def _create_detector(
            layers = [dLux.detectors.AddConstant(1.)]) -> OpticalLayer:
        return dLux.core.Detector(layers)
    return _create_detector


@pytest.fixture
def create_scene() -> callable:
    """
    Returns:
    --------
    _create_scene: callable
        A function that has all keyword arguments and can be
        used to create a `Scene` layer for testing.
    """
    def _create_scene(
            sources = [dLux.sources.PointSource()]) -> OpticalLayer:
        return dLux.core.Scene(sources)
    return _create_scene


@pytest.fixture
def create_filter() -> callable:
    """
    Returns:
    --------
    _create_filter: callable
        A function that has all keyword arguments and can be
        used to create a `Filter` layer for testing.
    """
    def _create_filter( 
            wavelengths = np.linspace(1e-6, 10e-6, 10),
            throughput = np.linspace(0, 1, 10)) -> OpticalLayer:
        return dLux.core.Filter(wavelengths, throughput)
    return _create_filter


@pytest.fixture
def create_instrument(
        create_optic: callable, 
        create_scene: callable, 
        create_detector: callable,
        create_filter: callable) -> callable:
    """
    Returns:
    --------
    _create_instrument: callable
        A function that has all keyword arguments and can be
        used to create a `Instrument` layer for testing.
    """
    def _create_instrument(
            optics          : OpticalLayer  = OpticsUtility().construct(),
            scene           : OpticalLayer  = SceneUtility().construct(),
            detector        : OpticalLayer  = DetectorUtility().construct(),
            # filter        : OpticalLayer  = FilterUtility().construct()
            filter          : OpticalLayer  = None,
            optical_layers  : list          = None,
            sources         : list          = None,
            detector_layers : list          = None,
            input_layers    : bool          = False,
            input_both      : bool          = False) -> Instrument:

        if input_both:
            return dLux.core.Instrument(
                optics=optics,
                scene=scene,
                detector=detector,
                filter=filter,
                optical_layers=optical_layers,
                sources=sources,
                detector_layers=detector_layers)
        elif not input_layers:
            return dLux.core.Instrument(
                optics=optics,
                scene=scene,
                detector=detector,
                filter=filter)
        else:
            return dLux.core.Instrument(
                filter=filter,
                optical_layers=optical_layers,
                sources=sources,
                detector_layers=detector_layers)
    return _create_instrument
