import pytest
import jax.numpy as np
from jax import Array
import dLux

from dLux import (
    wavefronts,
    optical_layers,
    propagators,
    apertures,
    aberrations,
    optics,
    images,
    detectors,
    detector_layers,
    instruments,
    observations,
    sources,
    spectra,
    models,
    utils,
    )

"""
There are Four main parts of dLux:

Classes that operate on/interact with wavefronts
    wavefronts.py     : 2 Classes
    aberrations.py    : 3 Classes
    optics.py         : 4 Classes
    optical_layers.py : 8 Classes
    propagators.py    : 4 Classes
    apertures.py      :   Classes

Classes that operate on/interact with images
    images.py          : 1 Classes
    detectors.py       : 1 Classes
    detector_layers.py :   Classes

Other main class types
    instruments.py  : 1 Classes
    observations.py : 1 Classes
    sources.py      :   Classes
    spectra.py      :   Classes

Sub Modules
    models.py
    utils.py


Presently no explicit tests exist for utils as the methods are implicity tested
though the other modules. 

Models is also new, and may be tested in the future.
"""






'''=============================================================================
================================================================================
# Wavefronts
------------

wavefronts.py classes:
    Wavefront
    FresnelWavefront

'''
@pytest.fixture
def create_wavefront():
    def _create_wavefront(
            npixels : int = 16,
            diameter : Array = np.array(1.),
            wavelength : Array = np.array(1e-6)) -> Wavefront:
        return wavefronts.Wavefront(
            npixels=npixels, 
            diameter=diameter, 
            wavelength=wavelength)
    return _create_wavefront

@pytest.fixture
def create_fresnel_wavefront():
    def _create_fresnel_wavefront(
            npixels : int = 16,
            diameter : Array = np.array(1.),
            wavelength : Array = np.array(1e-6)) -> Wavefront:
        return wavefronts.FresnelWavefront(
            npixels=npixels, 
            diameter=diameter, 
            wavelength=wavelength)
    return _create_fresnel_wavefront


'''
================================================================================
================================================================================
# Optical Layers
----------------

optical_layers.py classes:
    Tilt
    Normalise
    AddPhase
    AddOPD
    TransmissiveOptic
    GenericOptic
    Rotate
    ApplyBasisOPD

'''
@pytest.fixture
def create_tilt():
    """Constructs the Tilt class for testing."""
    def _create_tilt(angles : Array = np.ones(2)):
        return optical_layers.Tilt(tilt_angles)
    return _create_tilt

@pytest.fixture
def create_normalise():
    """Constructs the Normalise class for testing."""
    def _create_normalise():
        return optical_layers.Normalise()
    return _create_normalise_wavefront

@pytest.fixture
def create_add_phase():
    """Constructs the AddPhase class for testing."""
    def _create_add_phase(phase : Array = np.ones((16, 16))):
        return optical_layers.AddPhase(phase=phase)
    return _create_add_phase    

@pytest.fixture
def create_add_opd():
    """Constructs the AddOPD class for testing."""
    def _create_add_opd(opd : Array = np.ones((16, 16))):
        return optical_layers.AddOPD(opd=opd)
    return _create_add_opd

@pytest.fixture
def create_transmissive_optic():
    """Constructs the TransmissiveOptic class for testing."""
    def _create_transmissive_optic(transmission : Array = np.ones((16, 16))):
        return optical_layers.TransmissiveOptic(transmission=transmission)
    return _create_transmissive_optic 

@pytest.fixture
def create_generic_optic():
    """Constructs the GenericOptic class for testing."""
    def _create_generic_optic(
        transmission : Array = np.ones((16, 16)),
        opd          : Array = np.zeros((16, 16)),
        normalise    : bool  = False):
        return optical_layers.GenericOptic(
            transmission=transmission, 
            opd=opd, 
            normalise=normalise)
    return _create_generic_optic 

@pytest.fixture
def create_rotate():
    """Constructs the Rotate class for testing."""
    def _create_rotate(
        angle   : Array = np.array(np.pi), 
        order   : int   = 1, 
        complex : bool  = False):
        return optical_layers.Rotate(angle=angle, order=order, complex=complex)
    return _create_rotate

@pytest.fixture
def create_apply_basis_opd():
    """Constructs the ApplyBasisOPD class for testing."""
    def _create_apply_basis_opd(
        basis = np.ones((3, 16, 16)), 
        coefficients = np.ones(3)):
        return optical_layers.ApplyBasisOPD(basis, coefficients)
    return _create_apply_basis_opd


'''
================================================================================
================================================================================
# Aberrations
-------------

aberrations.py classes:
    Zernike
    ZernikeBasis
    AberrationFactory

'''
@pytest.fixture
def create_zernike():
    """Constructs the Zernike class for testing."""
    def _create_zernike(j : int = 1):
        return aberrations.Zernike(j=j)
    return _create_zernike

@pytest.fixture
def create_zernike_basis():
    """Constructs the ZernikeBasis class for testing."""
    def _create_zernike_basis(
            js : Array = np.arange(1, 4)):
        return aberrations.ZernikeBasis(js=js)
    return _create_zernike_basis

@pytest.fixture
def create_aberration_factory():
    """Constructs the AberrationFactory class for testing."""
    def _create_aberration_factory(
            npixels        : int = 16,
            radial_orders  : Array = [1, 2],
            coefficients   : Array = np.ones(3), 
            aperutre_ratio : float = 1.,
            nsides         : int  = 0,
            rotation       : float = 0.,
            noll_indices   : Array = None):
        return aberrations.AberrationFactory(
            npixels=npixels, 
            radial_orders=radial_orders,
            coefficients=coefficients,
            aperutre_ratio=aperutre_ratio, 
            nsides=nsides, 
            rotation=rotation,
            noll_indices=noll_indices)
    return _create_aberration_factory


'''
================================================================================
================================================================================
# Optics
--------

optics.py classes:
    LayeredOptics
    AngularOptics
    CartesianOptics
    FlexibleOptics

'''
aperture = apertures.ApertureFactory(16)
aberrations = aberrations.AberrationFactory(16, [2, 3])
propagator = propagators.PropagatorFactory(16, 1)
mask = np.ones((16, 16))

@pytest.fixture
def create_angular_optics():
    """Constructs the AngularOptics class for testing."""
    def _create_angular_optics(
        diameter        : float = 1.,
        aperture        : Array = aperture,
        psf_npixels     : int   = 16,
        psf_pixel_scale : float = 1.,
        psf_oversample  : int   = 1,
        aberrations     : Array = aberrations,
        mask            : Array = mask):
        return optics.AngularOptics(
            diameter=diameter,
            aperture=aperture,
            psf_npixels=psf_npixels,
            psf_pixel_scale=psf_pixel_scale,
            psf_oversample=psf_oversample,
            aberrations=aberrations,
            mask=mask)
    return _create_angular_optics

@pytest.fixture
def create_cartesian_optics():
    """Constructs the CartesianOptics class for testing."""
    def _create_cartesian_optics(
        diameter        : float = 1.,
        aperture        : Array = aperture,
        focal_length    : float = 10.,
        psf_npixels     : int = 16,
        psf_pixel_scale : float = 1.,
        psf_oversample  : int = 1,
        aberrations     : Array = aberration,
        mask            : Array = mask):
        return optics.CartesianOptics(
            diameter=diameter,
            aperture=aperture,
            focal_length=focal_length,
            psf_npixels=psf_npixels,
            psf_pixel_scale=psf_pixel_scale,
            psf_oversample=psf_oversample,
            aberrations=aberrations,
            mask=mask)
    return _createcartesianr_optics

@pytest.fixture
def create_flexible_optics():
    """Constructs the FlexibleOptics class for testing."""
    def _create_flexible_optics(
        diameter    = 1.,
        aperture    = aperture,
        propagator  = propagator,
        aberrations = aberration,
        mask        = mask):
        return optics.FlexibleOptics(
            diameter=diameter,
            aperture=aperture,
            propagator=propagator,
            aberrations=aberrations,
            mask=mask)
    return _create_flexible_optics

@pytest.fixture
def create_layered_optics():
    """Constructs the LayeredOptics class for testing."""
    def _create_layered_optics(
        diameter   : float = 1,
        wf_npixels : int   = 16,
        layers     : list  = [aperture, propagator]):
        return optics.LayeredOptics(
            diameter=diameter,
            wf_npixels=wf_npixels, 
            layers=layers)
    return _create_layered_optics


'''
================================================================================
================================================================================
# Propagators
-------------

propagators.py classes:
    MFT
    FFT
    ShiftedMFT
    FarFieldFresnel

'''
@pytest.fixture
def create_fft():
    def _create_fft(
            focal_length : Array = 1.,
            pad          : int = 2,
            inverse      : bool = False):
        return propagators.FFT(focal_length=focal_length, pad=pad, 
            inverse=inverse)
    return _create_fft

@pytest.fixture
def create_mft():
    def _create_mft(
        pixel_scale  : Array = 1.,
        npixels      : int   = 16,
        focal_length : Array = 1.,
        inverse      : bool = False):
        return propagators.MFT(npixels=npixels, pixel_scale=pixel_scale,
            focal_length=focal_length, inverse=inverse)
    return _create_mft

@pytest.fixture
def create_shifted_mft():
    def _create_shifted_mft(
        pixel_scale  : Array = 1.,
        npixels      : int   = 16,
        shift        : Array = np.zeros(2),
        focal_length : Array = None,
        pixel        : bool = False,
        inverse      : bool = False):
        return propagators.ShiftedMFT(npixels=npixels, pixel_scale=pixel_scale,
            focal_length=focal_length, inverse=inverse)
    return _create_shifted_mft


@pytest.fixture
def create_far_field_fresnel():
    def _create_far_field_fresnel(
                  npixels      : int   = 16,
                  pixel_scale  : float = np.array(1.),
                  focal_length : Array = np.array(1.),
                  focal_shift  : Array = np.array(1e-3),
                  shift        : Array = np.zeros(2),
                  pixel        : bool  = False):
        return propagators.FarFieldFresnel(npixels, pixel_scale,
                 focal_length, focal_shift, shift, pixel)
    return _create_far_field_fresnel







### Apertures ###
@pytest.fixture
def create_square_aperture():

    def _create_square_aperture(  
            width       : Array = 1., 
            centre      : Array = [0., 0.],
            shear       : Array = [0., 0.],
            compression : Array = [1., 1.],
            rotation    : Array = 0.,
            occulting   : bool = False, 
            softening   : Array = 0.,
            normalise   : bool = False) -> Aperture:
        return SquareAperture(width,
            centre,
            shear,
            compression,
            rotation,
            occulting,
            softening,
            normalise)

    return _create_square_aperture


@pytest.fixture
def create_rectangular_aperture()-> callable:

    def _create_rectangular_aperture(
                                    length      : Array = 0.5, 
                                    width       : Array = 1., 
                                    centre      : Array = [0., 0.],
                                    shear       : Array = [0., 0.],
                                    compression : Array = [1., 1.],
                                    rotation    : Array = 0.,
                                    occulting   : bool = False, 
                                    softening   : Array = 0.,
                                    normalise   : bool = False) -> Aperture:
        return apertures.RectangularAperture(
                                                length,
                                                width,
                                                centre,
                                                shear,
                                                compression,
                                                rotation,
                                                occulting,
                                                softening,
                                                normalise)
    return _create_rectangular_aperture


@pytest.fixture
def create_circular_aperture():

    def _create_circular_aperture( 
            radius      : Array = 1., 
            centre      : Array = [0., 0.],
            shear       : Array = [0., 0.],
            compression : Array = [1., 1.],
            occulting   : bool = False, 
            softening   : Array = 0.,
            normalise   : bool = False) -> Aperture:
        return apertures.CircularAperture(
            radius,
            centre,
            shear,
            compression,
            occulting,
            softening,
            normalise)

    return _create_circular_aperture


@pytest.fixture
def create_hexagonal_aperture():

    def _create_hexagonal_aperture( 
            radius      : Array = 1., 
            centre      : Array = [0., 0.],
            shear       : Array = [0., 0.],
            compression : Array = [1., 1.],
            rotation    : Array = 0.,
            occulting   : bool = False, 
            softening   : Array = 0.,
            normalise   : bool = False) -> Aperture:
        return apertures.HexagonalAperture(
            radius,
            centre,
            shear,
            compression,
            rotation,
            occulting,
            softening,
            normalise)

    return _create_hexagonal_aperture


@pytest.fixture
def create_annular_aperture():

    def _create_annular_aperture( 
            rmax        : Array = 1.2, 
            rmin        : Array = 0.5, 
            centre      : Array = [0., 0.],
            shear       : Array = [0., 0.],
            compression : Array = [1., 1.],
            occulting   : bool = False, 
            softening   : Array = 0.,
            normalise   : bool = False) -> Aperture:
        return apertures.AnnularAperture(
            rmax,
            rmin,
            centre,
            shear,
            compression,
            occulting,
            softening,
            normalise)

    return _create_annular_aperture


@pytest.fixture
def create_irregular_polygonal_aperture():

    def _create_irregular_polygonal_aperture( 
            vertices    : Array = np.array([[0.5,   0.5], [0.5, -0.5], 
                                            [-0.5, -0.5], [-0.5, 0.5]]),
            centre      : Array = [0., 0.],
            shear       : Array = [0., 0.],
            compression : Array = [1., 1.],
            rotation    : Array = 0.,
            occulting   : bool = False, 
            softening   : Array = 0.,
            normalise   : bool = False) -> Aperture:
        return apertures.IrregularPolygonalAperture(
            vertices,
            centre,
            shear,
            compression,
            rotation,
            occulting,
            softening,
            normalise)

    return _create_irregular_polygonal_aperture


@pytest.fixture
def create_uniform_spider():

    def _create_uniform_spider(
            number_of_struts: int = 4,
            width_of_struts: float = .05,
            centre      : Array = [0., 0.], 
            shear       : Array = [0., 0.],
            compression : Array = [1., 1.],
            rotation    : Array = 0., 
            softening   : bool = False,
            normalise   : bool = False):
        return apertures.UniformSpider(
            number_of_struts, 
            width_of_struts,
            centre=centre,
            shear=shear,
            compression=compression,
            rotation=rotation,
            softening=softening,
            normalise=normalise)
    return _create_uniform_spider


@pytest.fixture
def create_aberrated_aperture(create_circular_aperture: callable):
    nterms = 6
    def _create_aberrated_aperture(
            aperture: object = create_circular_aperture(),
            noll_inds: list = np.arange(1, nterms+1, dtype=int),
            coefficients: Array = np.ones(nterms, dtype=float),
            ):
        return apertures.AberratedAperture(
            noll_inds=noll_inds,
            coefficients=coefficients,
            aperture=aperture)
    
    return _create_aberrated_aperture


@pytest.fixture
def create_static_aperture(create_circular_aperture: callable):

    def _create_static_aperture(
            aperture: object = create_circular_aperture(),
            npixels: int = 16,
            diameter: float = 1.,
            coordinates: Array = None,
            ):
        return apertures.StaticAperture(
            aperture=aperture, npixels=npixels, diameter=diameter, 
            coordinates=coordinates)
    
    return _create_static_aperture


@pytest.fixture
def create_static_aberrated_aperture(create_aberrated_aperture: callable):

    def _create_static_aberrated_aperture(
            aperture: object = create_aberrated_aperture(),
            npixels: int = 16,
            diameter: float = 1.,
            coordinates: Array = None,
            ):
        return apertures.StaticAberratedAperture(
            aperture=aperture, npixels=npixels, diameter=diameter, 
            coordinates=coordinates)
    
    return _create_static_aberrated_aperture


@pytest.fixture
def create_compound_aperture(create_circular_aperture: callable):

    def _create_compound_aperture(
            apertures: list = [create_circular_aperture()],
            normalise: bool = False,
            ):
        return apertures.CompoundAperture(
            apertures=apertures, normalise=normalise)
    
    return _create_compound_aperture


@pytest.fixture
def create_multi_aperture(create_circular_aperture: callable):

    def _create_multi_aperture(
            apertures: list = [create_circular_aperture()],
            normalise: bool = False,
            ):
        return apertures.MultiAperture(
            apertures=apertures, normalise=normalise)
    
    return _create_multi_aperture


@pytest.fixture
def create_aperture_factory(create_circular_aperture: callable):

    def _create_aperture_factory(
        npixels          : int   = 16, 
        aperture_ratio   : float = 1.0,
        secondary_ratio  : float = 0.,
        nsides           : int   = 0,
        secondary_nsides : int   = 0,
        rotation         : float = 0., 
        nstruts          : int   = 0,
        strut_ratio      : float = 0.,
        strut_rotation   : float = 0.,
        normalise        : bool  = True):
        return apertures.ApertureFactory(
            npixels=npixels,
            aperture_ratio=aperture_ratio,
            secondary_ratio=secondary_ratio,
            nsides=nsides,
            secondary_nsides=secondary_nsides,
            rotation=rotation,
            nstruts=nstruts,
            strut_ratio=strut_ratio,
            strut_rotation=strut_rotation,
            normalise=normalise)
    
    return _create_aperture_factory








### Sources ###
@pytest.fixture
def create_source():
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
                  spectrum : OpticalLayer = spectra.ArraySpectrum(np.linspace(500e-9, \
                                                                 600e-9, 10)),
                  name     : str      = "Source") -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return sources.Source(position, flux, spectrum, name=name)
    return _create_source


@pytest.fixture
def create_relative_flux_source():
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
                  spectrum : OpticalLayer = spectra.ArraySpectrum(np.linspace(500e-9, \
                                                                 600e-9, 10)),
                  contrast   : Array    = np.array(2.),
                  name       : str      = "RelativeSource") -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return sources.RelativeFluxSource(contrast, position=position,
                                               flux=flux, spectrum=spectrum,
                                               name=name)
    return _create_relative_flux_source


@pytest.fixture
def create_relative_position_source():
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
                  spectrum       : Spectrum = spectra.ArraySpectrum(
                    np.linspace(500e-9, 600e-9, 10)),
                  separation     : Array    = np.array(1.),
                  position_angle : Array    = np.array(0.),
                  name           : str      = "RelativePositionSource") -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return sources.RelativePositionSource(separation, position_angle,
                                                   position=position, flux=flux,
                                                   spectrum=spectrum, name=name)
    return _create_relative_position_source


@pytest.fixture
def create_point_source():
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
                  spectrum    : Spectrum = spectra.ArraySpectrum(
                    np.linspace(500e-9, 600e-9, 10)),
                  name        : str      = "PointSource") -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return sources.PointSource(position, flux, spectrum, name=name)
    return _create_point_source


@pytest.fixture
def create_multi_point_source():
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
                  spectrum : Spectrum = spectra.ArraySpectrum(np.linspace(500e-9, \
                                                                 600e-9, 10)),
                  name     : str      = "Source") -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return sources.MultiPointSource(position, flux, spectrum, name=name)
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
                  spectrum     : Spectrum = spectra.ArraySpectrum(np.linspace(500e-9, \
                                                                 600e-9, 10)),
                  distribution : Array    = np.ones((5, 5))/np.ones((5, 5)).sum(),
                  name         : str      = "Source") -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return sources.ArrayDistribution(position, flux, distribution,
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
            spectrum = spectra.CombinedSpectrum(wavelengths, weights)
            
        return sources.BinarySource(position, flux, separation, \
                                  position_angle, contrast, spectrum, name=name)
    return _create_binary_source


@pytest.fixture
def create_point_extended_source():
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
                  spectrum     : Spectrum = spectra.ArraySpectrum(np.linspace(500e-9, \
                                                                 600e-9, 10)),
                  contrast     : Array    = np.array(2.),
                  distribution : Array    = np.ones((5, 5))/np.ones((5, 5)).sum(),
                  name         : str      = "Source") -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        return sources.PointExtendedSource(position, flux, distribution,
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
            spectrum = spectra.CombinedSpectrum(wavelengths, weights)
            
        return sources.PointAndExtendedSource(position, flux, distribution,
                                                contrast, spectrum, name=name)
    return _create_point_and_extended_source











### Spectrums ###
@pytest.fixture
def create_array_spectrum():
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
        return spectra.ArraySpectrum(wavelengths, weights)
    return _create_array_spectrum


@pytest.fixture
def create_polynomial_spectrum():
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
        return spectra.PolynomialSpectrum(wavelengths, coefficients)
    return _create_polynomial_spectrum


@pytest.fixture
def create_combined_spectrum():
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
        return spectra.CombinedSpectrum(wavelengths, weights)
    return _create_combined_spectrum




@pytest.fixture
def create_detector():
    """
    Returns:
    --------
    _create_detector: callable
        A function that has all keyword arguments and can be
        used to create a `Detector` layer for testing.
    """
    def _create_detector(
            layers = [detectors.AddConstant(1.)]):
        return core.Detector(layers)
    return _create_detector







@pytest.fixture
def create_instrument(
        create_optics: callable, 
        create_point_source: callable, 
        create_detector: callable,
        create_dither: callable):
    """
    Returns:
    --------
    _create_instrument: callable
        A function that has all keyword arguments and can be
        used to create a `Instrument` layer for testing.
    """
    def _create_instrument(
            optics          : OpticalLayer  = create_optics(),
            sources         : OpticalLayer  = create_point_source(),
            detector        : OpticalLayer  = create_detector(),
            observation     : Observation   = create_dither(),
            ):

        # if input_both:
        return core.Instrument(
            optics=optics,
            detector=detector,
            sources=sources,
            observation=observation,)
    return _create_instrument













### Detectors ###
@pytest.fixture
def create_pixel_response():
    """
    Returns:
    --------
    _create_pixel_response: callable
        A function that has all keyword arguments and can be
        used to create a `ApplyPixelResponse` layer for testing.
    """
    def _create_pixel_response(
            pixel_response: Array = np.ones((16, 16))):
        return detectors.ApplyPixelResponse(pixel_response)
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
            kernel_size: int = 10):
        return detectors.ApplyJitter(sigma, kernel_size)
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
    def _create_saturation(saturation: Array = np.array(1.)):
        return detectors.ApplySaturation(saturation)
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
    def _create_constant(value: Array = np.array(1.)):
        return detectors.AddConstant(value)
    return _create_constant


@pytest.fixture
def create_integer_downsample():
    """
    Returns:
    --------
    _create_integer_downsample: callable
        A function that has all keyword arguments and can be
        used to create a `IntegerDownsample` layer for testing.
    """
    def _create_integer_downsample(kernel_size: int = 4):
        return detectors.IntegerDownsample(kernel_size)
    return _create_integer_downsample    


@pytest.fixture
def create_rotate_detector():
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
            padding: int = 2):
        return detectors.Rotate(angle, fourier, padding)
    return _create_rotate_detector


### Observations ###
@pytest.fixture
def create_dither():
    """
    Returns:
    --------
    _create_dither: callable
        A function that has all keyword arguments and can be
        used to create a `Dither` class for testing.
    """
    def _create_dither(
            dithers: Array = np.ones((5, 2))) -> BaseObservation:
        return observations.Dither(dithers)
    
    return _create_dither


# Possibly for models later on
# @pytest.fixture
# def create_basis_climb():
#     def _create_basis_climb(
#             basis: Array = np.ones((3, 768, 768)),
#             coefficients: Array = np.ones(3),
#             ideal_wavelength: Array = np.array(5e-7)):
#         return optical_layers.ApplyBasisCLIMB(
#             basis, ideal_wavelength, coefficients)
#     return _create_basis_climb

