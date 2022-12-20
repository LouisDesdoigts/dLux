import pytest
import jax.numpy as np
import dLux


Array = np.ndarray
Wavefront = dLux.wavefronts.Wavefront
OpticalLayer = dLux.optics.OpticalLayer
Spectrum = dLux.spectrums.Spectrum
Source = dLux.sources.Source
Aperture = dLux.apertures.ApertureLayer


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
            basis: Array = np.ones((3, 768, 768)),
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
                  name     : str      = "Source") -> Source:
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
                  name       : str      = "RelativeSource") -> Source:
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
                  name           : str      = "RelativePositionSource") -> Source:
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
                  name        : str      = "PointSource") -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return dLux.sources.PointSource(position, flux, spectrum, name=name)
    return _create_point_source

@pytest.fixture
def create_multi_point_source() -> callable:
    """
    Returns:
    --------
    _create_point_source: callable
        a function that has all keyword arguments and can be
        used to create a `MultiPointSource` layer for testing.
    """


    def _create_multi_point_source(
                  position : Array    = np.zeros((3, 2)),
                  flux     : Array    = np.ones(3),
                  spectrum : Spectrum = dLux.spectrums.ArraySpectrum(np.linspace(500e-9, \
                                                                 600e-9, 10)),
                  name     : str      = "Source") -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return dLux.sources.MultiPointSource(position, flux, spectrum, name=name)
    return _create_multi_point_source


@pytest.fixture
def create_array_distribution():
    """
    Returns:
    --------
    _create_array_distribution: callable
        a function that has all keyword arguments and can be
        used to create a `ArrayDistribution` layer for testing.
    """


    def _create_array_distribution(
                  position     : Array    = np.array([0., 0.]),
                  flux         : Array    = np.array(1.),
                  spectrum     : Spectrum = dLux.spectrums.ArraySpectrum(np.linspace(500e-9, \
                                                                 600e-9, 10)),
                  distribution : Array    = np.ones((5, 5))/np.ones((5, 5)).sum(),
                  name         : str      = "Source") -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return dLux.sources.ArrayDistribution(position, flux, distribution,
                                              spectrum, name=name)
    return _create_array_distribution

@pytest.fixture
def create_binary_source():
    """
    Returns:
    --------
    _create_binary_source: callable
        a function that has all keyword arguments and can be
        used to create a `BinarySource` layer for testing.
    """


    def _create_binary_source(
                  position       : Array    = np.array([0., 0.]),
                  flux           : Array    = np.array(1.),
                  spectrum       : Spectrum = None,
                  separation     : Array    = np.array(1.),
                  position_angle : Array    = np.array(0.),
                  contrast       : Array    = np.array(2.),
                  name           : str      = "BinarySource") -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        if spectrum is None:
            wavelengths = np.tile(np.linspace(500e-9, 600e-9, 10), (2, 1))
            weights     = np.tile(np.arange(10), (2, 1))
            spectrum = dLux.spectrums.CombinedSpectrum(wavelengths, weights)
            
        return dLux.sources.BinarySource(position, flux, separation, \
                                  position_angle, contrast, spectrum, name=name)
    return _create_binary_source


@pytest.fixture
def create_point_extended_source() -> callable:
    """
    Returns:
    --------
    _create_point_extended_source: callable
        a function that has all keyword arguments and can be
        used to create a `PointExtendedSource` layer for testing.
    """

    def _create_point_extended_source(
                  position     : Array    = np.array([0., 0.]),
                  flux         : Array    = np.array(1.),
                  spectrum     : Spectrum = dLux.spectrums.ArraySpectrum(np.linspace(500e-9, \
                                                                 600e-9, 10)),
                  contrast     : Array    = np.array(2.),
                  distribution : Array    = np.ones((5, 5))/np.ones((5, 5)).sum(),
                  name         : str      = "Source") -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return dLux.sources.PointExtendedSource(position, flux, distribution,
                                                contrast, spectrum, name=name)
    return _create_point_extended_source

@pytest.fixture
def create_point_and_extended_source():
    """
    Returns:
    --------
    _create_point_and_extended_source: callable
        a function that has all keyword arguments and can be
        used to create a `PointAndExtendedSource` layer for testing.
    """


    def _create_point_and_extended_source(
                  position     : Array    = np.array([0., 0.]),
                  flux         : Array    = np.array(1.),
                  spectrum     : Spectrum = None,
                  contrast     : Array    = np.array(2.),
                  distribution : Array    = np.ones((5, 5))/np.ones((5, 5)).sum(),
                  name         : str      = "Point and extended source") -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        if spectrum is None:
            wavelengths = np.tile(np.linspace(500e-9, 600e-9, 10), (2, 1))
            weights = np.tile(np.arange(10), (2, 1))
            spectrum = dLux.spectrums.CombinedSpectrum(wavelengths, weights)
            
        return dLux.sources.PointAndExtendedSource(position, flux, distribution,
                                                contrast, spectrum, name=name)
    return _create_point_and_extended_source


@pytest.fixture
def create_array_spectrum() -> callable:
    """
    Returns:
    --------
    _create_array_spectrum: callable
        a function that has all keyword arguments and can be
        used to create a `ArraySpectrum` layer for testing.
    """

    def _create_array_spectrum(
                  wavelengths : Array = np.linspace(500e-9, 600e-9, 10),
                  weights     : Array = np.arange(10)) -> Spectrum:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return dLux.spectrums.ArraySpectrum(wavelengths, weights)
    return _create_array_spectrum


@pytest.fixture
def create_polynomial_spectrum() -> callable:
    """
    Returns:
    --------
    _create_polynomial_spectrum: callable
        a function that has all keyword arguments and can be
        used to create a `ArraySpectrum` layer for testing.
    """

    def _create_polynomial_spectrum(
                  wavelengths  : Array = np.linspace(500e-9, 600e-9, 10),
                  coefficients : Array = np.arange(3)) -> Spectrum:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return dLux.spectrums.PolynomialSpectrum(wavelengths, coefficients)
    return _create_polynomial_spectrum

@pytest.fixture
def create_combined_spectrum() -> callable:
    """
    Returns:
    --------
    _create_combined_spectrum: callable
        a function that has all keyword arguments and can be
        used to create a `ArraySpectrum` layer for testing.
    """

    def _create_combined_spectrum(
                  wavelengths : Array = np.tile(np.linspace(500e-9, 600e-9, 10), (2, 1)),
                  weights     : Array = np.tile(np.arange(10), (2, 1))) -> Spectrum:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return dLux.spectrums.CombinedSpectrum(wavelengths, weights)
    return _create_combined_spectrum


@pytest.fixture
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
def create_scene(create_point_source) -> callable:
    """
    Returns:
    --------
    _create_scene: callable
        A function that has all keyword arguments and can be
        used to create a `Scene` layer for testing.
    """
    def _create_scene(
            sources = [create_point_source()]) -> OpticalLayer:
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
            throughput = np.linspace(0, 1, 10),
            filter_name = None) -> OpticalLayer:
        return dLux.core.Filter(wavelengths, throughput, filter_name=filter_name)
    return _create_filter


@pytest.fixture
def create_instrument(
        create_optics: callable, 
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
            optics          : OpticalLayer  = create_optics(),
            scene           : OpticalLayer  = create_scene(),
            detector        : OpticalLayer  = create_detector(),
            # filter        : OpticalLayer  = create_filter(),
            filter          : OpticalLayer  = None,
            optical_layers  : list          = None,
            sources         : list          = None,
            detector_layers : list          = None,
            input_layers    : bool          = False,
            input_both      : bool          = False) -> OpticalLayer:

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


#####  Apertures ######

@pytest.fixture
def create_square_aperture() -> callable:
    """
    Returns:
    --------
    _create_filter: callable
        A function that has all keyword arguments and can be
        used to create a `Filter` layer for testing.
    """

    def _create_square_aperture(  
            width       : Array = 1., 
            centre      : Array = [0., 0.],
            strain      : Array = [0., 0.],
            compression : Array = [1., 1.],
            rotation    : Array = 0.,
            occulting   : bool = False, 
            softening   : bool = False) -> Aperture:
        """
        Construct's an instance of `SquareAperture` making 
        it easy to toggle a single parameter. 
        Parameters:
        -----------
        centre: float, meters
            The centre of the coordinate system in the paraxial coordinates.
        strain: Array
            Linear stretching of the x and y axis representing a 
            strain of the coordinate system.
        compression: Array 
            The x and y compression of the coordinate system. This 
            is a constant. 
        rotation: float, radians
            The rotation of the aperture away from the positive 
            x-axis. 
        softening: bool 
            True if the aperture is soft edged otherwise False. A
            soft edged aperture has a small layer of non-binary 
            pixels. This is to prevent undefined gradients. 
        occulting: bool 
            True if the aperture is occulting else False. An 
            occulting aperture is zero inside and one outside. 
        theta: float, radians
            The rotation of the aperture away from the positive 
            x-axis. 
        width: float, meters
            The side length of the square. 
        """
        return dLux.SquareAperture(width,
                                    centre,
                                    strain,
                                    compression,
                                    rotation,
                                    occulting,
                                    softening)

    return _create_square_aperture



@pytest.fixture
def create_rectangular_aperture()-> callable:
    """
    Returns:
    --------
    _create_rectangular_aperture: callable
        A function that has all keyword arguments and can be
        used to create a `RectangularAperture` layer for testing.
    """

    def _create_rectangular_aperture(
                                    length      : Array = 0.5, 
                                    width       : Array = 1., 
                                    centre      : Array = [0., 0.],
                                    strain      : Array = [0., 0.],
                                    compression : Array = [1., 1.],
                                    rotation    : Array = 0.,
                                    occulting   : bool = False, 
                                    softening   : bool = False) -> Aperture:
        """
        Construct's an instance of `SquareAperture` making 
        it easy to toggle a single parameter. 
        Parameters:
        -----------
        occulting: bool = None
            True if light passes through the aperture. 
            False if light is blocked by the aperture. 
        softening: bool = None
            True is the aperture has soft pixels on the 
            edges else false. 
        x_offset: float = None, meters
            The distance along the x axis that the aperture 
            is moved from the centre of the coordinate system. 
        y_offset: float = None, meters
            The distance along the y axis that the aperture 
            is moved from the centre of the coordinate system.
        width: float = None, meters
            The width of the aperture.
        length: float = None, meters 
            The length of the aperture.
        theta: float = None, radians
            The rotation of the aperture.
        """
        return dLux.apertures.RectangularAperture(
                                                length      , 
                                                width       , 
                                                centre      ,
                                                strain      ,
                                                compression ,
                                                rotation    ,
                                                occulting   , 
                                                softening)
    return _create_rectangular_aperture


@pytest.fixture
def create_circular_aperture() -> callable:
    """
    Returns:
    --------
    _create_circular_aperture: callable
        A function that has all keyword arguments and can be
        used to create a `CircularAperture` layer for testing.
    """


    def _create_circular_aperture( 
                                radius      : Array = 1., 
                                centre      : Array = [0., 0.],
                                strain      : Array = [0., 0.],
                                compression : Array = [1., 1.],
                                occulting   : bool = False, 
                                softening   : bool = False) -> Aperture:
        return dLux.apertures.CircularAperture(
                                radius,
                                centre,
                                strain,
                                compression,
                                occulting,
                                softening)
    return _create_circular_aperture

@pytest.fixture
def create_hexagonal_aperture() -> callable:
    """
    Returns:
    --------
    _create_hexagonal_aperture: callable
        A function that has all keyword arguments and can be
        used to create a `HexagonalAperture` layer for testing.
    """


    def _create_hexagonal_aperture( 
                                radius      : Array = 1., 
                                centre      : Array = [0., 0.],
                                strain      : Array = [0., 0.],
                                compression : Array = [1., 1.],
                                rotation    : Array = 0.,
                                occulting   : bool = False, 
                                softening   : bool = False) -> Aperture:
        return dLux.apertures.HexagonalAperture(
                                radius,
                                centre,
                                strain,
                                compression,
                                rotation,
                                occulting,
                                softening)
    return _create_hexagonal_aperture


@pytest.fixture
def create_annular_aperture() -> callable:
    """
    Returns:
    --------
    _create_annular_aperture: callable
        A function that has all keyword arguments and can be
        used to create a `AnnularAperture` layer for testing.
    """


    def _create_annular_aperture( 
                                rmax        : Array = 1.2, 
                                rmin        : Array = 0.5, 
                                centre      : Array = [0., 0.],
                                strain      : Array = [0., 0.],
                                compression : Array = [1., 1.],
                                occulting   : bool = False, 
                                softening   : bool = False) -> Aperture:
        return dLux.apertures.AnnularAperture(
                                rmax,
                                rmin,
                                centre,
                                strain,
                                compression,
                                occulting,
                                softening)
    return _create_annular_aperture


@pytest.fixture
def create_pixel_response() -> callable:
    """
    Returns:
    --------
    _create_pixel_response: callable
        A function that has all keyword arguments and can be
        used to create a `ApplyPixelResponse` layer for testing.
    """
    def _create_pixel_response(
            pixel_response: Array = np.ones((16, 16))) -> OpticalLayer:
        return dLux.detectors.ApplyPixelResponse(pixel_response)
    return _create_pixel_response


@pytest.fixture
def create_jitter() -> None:
    """
    Returns:
    --------
    _create_jitter: callable
        A function that has all keyword arguments and can be
        used to create a `ApplyJitter` layer for testing.
    """
    def _create_jitter(
            sigma: Array = np.array(1.),
            kernel_size: int = 10) -> OpticalLayer:
        return dLux.detectors.ApplyJitter(sigma, kernel_size)
    return _create_jitter


@pytest.fixture
def create_saturation() -> None:
    """
    Returns:
    --------
    _create_saturation: callable
        A function that has all keyword arguments and can be
        used to create a `ApplySaturation` layer for testing.
    """
    def _create_saturation(saturation: Array = np.array(1.)) -> OpticalLayer:
        return dLux.detectors.ApplySaturation(saturation)
    return _create_saturation


@pytest.fixture
def create_constant() -> None:
    """
    Returns:
    --------
    _create_constant: callable
        A function that has all keyword arguments and can be
        used to create a `ApplyConstant` layer for testing.
    """
    def _create_constant(value: Array = np.array(1.)) -> OpticalLayer:
        return dLux.detectors.AddConstant(value)
    return _create_constant


@pytest.fixture
def create_integer_downsample() -> callable:
    """
    Returns:
    --------
    _create_integer_downsample: callable
        A function that has all keyword arguments and can be
        used to create a `IntegerDownsample` layer for testing.
    """
    def _create_integer_downsample(kernel_size: int = 4) -> OpticalLayer:
        return dLux.detectors.IntegerDownsample(kernel_size)
    return _create_integer_downsample    


@pytest.fixture
def create_rotate_detector() -> callable:
    """
    Returns:
    --------
    _create_rotate_detector: callable
        A function that has all keyword arguments and can be
        used to create a `Rotate` layer for testing.
    """
    def _create_rotate_detector(
            angle: Array = np.array(np.pi),
            fourier: bool = False,
            padding: int = 2) -> OpticalLayer:
        return dLux.detectors.Rotate(angle, fourier, padding)
    return _create_rotate_detector


@pytest.fixture
def create_uniform_spider() -> callable:
    """
    Returns:
    --------
    _create_uniform_spider: callable
        A function that has all keyword arguments and can be
        used to create a `UniformSpider` layer for testing.
    """
    def _create_uniform_spider(
            number_of_struts: int = 4,
            width_of_struts: float = .05,
            centre      : Array = [0., 0.], 
            strain      : Array = [0., 0.],
            compression : Array = [1., 1.],
            rotation    : Array = 0.,
            occulting   : bool = False, 
            softening   : bool = False) -> OpticalLayer:
        return dLux.apertures.UniformSpider(
            number_of_struts, 
            width_of_struts,
            centre=centre,
            strain=strain,
            compression=compression,
            rotation=rotation,
            occulting=occulting,
            softening=softening)
    return _create_uniform_spider


@pytest.fixture
def create_aberrated_aperture(create_circular_aperture: callable) -> callable:
    """
    Returns:
    --------
    _create_aberrated_aperture: callable
        A function that has all keyword arguments and can be
        used to create a `AberratedAperture` layer for testing.
    """
    shape: int = 6

    def _create_aberrated_aperture(
            noll_inds: list = np.arange(shape, dtype=int),
            coeffs: Array = np.ones(shape, dtype=float),
            aperture: object = create_circular_aperture()) -> OpticalLayer:
        return dLux.apertures.AberratedAperture(
            noll_inds=noll_inds,
            coeffs=coeffs,
            aperture=aperture)
    
    return _create_aberrated_aperture
    

