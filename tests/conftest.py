import pytest
import jax.numpy as np
from jax import Array

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
)

"""
There are Four main parts of dLux:

Classes that operate on/interact with wavefronts
    wavefronts.py     : 2 Classes
    aberrations.py    : 2 Classes
    optics.py         : 4 Classes
    propagators.py    : 4 Classes
    optical_layers.py : 8 Classes
    apertures.py      : 9 Classes

Classes that operate on/interact with images
    images.py          : 1 Classes
    detectors.py       : 1 Classes
    detector_layers.py : 6 Classes

Source objects the operate on optics classes
    sources.py      : 5 Classes
    spectra.py      : 2 Classes

Other main class types
    instruments.py  : 1 Classes
    observations.py : 1 Classes

Sub Modules
    utils.py
"""


"""=============================================================================
================================================================================
# Wavefronts
------------

wavefronts.py classes:
    Wavefront
    FresnelWavefront

"""


@pytest.fixture
def create_wavefront():
    def _create_wavefront(
        npixels: int = 16,
        diameter: Array = np.array(1.0),
        wavelength: Array = np.array(1e-6),
    ):
        return wavefronts.Wavefront(
            npixels=npixels, diameter=diameter, wavelength=wavelength
        )

    return _create_wavefront


@pytest.fixture
def create_fresnel_wavefront():
    def _create_fresnel_wavefront(
        npixels: int = 16,
        diameter: Array = np.array(1.0),
        wavelength: Array = np.array(1e-6),
    ):
        return wavefronts.FresnelWavefront(
            npixels=npixels, diameter=diameter, wavelength=wavelength
        )

    return _create_fresnel_wavefront


"""
================================================================================
================================================================================
# Aberrations
-------------

aberrations.py classes:
    Zernike
    ZernikeBasis

"""


@pytest.fixture
def create_zernike():
    """Constructs the Zernike class for testing."""

    def _create_zernike(j: int = 1):
        return aberrations.Zernike(j=j)

    return _create_zernike


@pytest.fixture
def create_zernike_basis():
    """Constructs the ZernikeBasis class for testing."""

    def _create_zernike_basis(js: Array = np.arange(1, 4)):
        return aberrations.ZernikeBasis(js=js)

    return _create_zernike_basis


"""
================================================================================
================================================================================
# Optics
--------

optics.py classes:
    LayeredOptics
    AngularOptics
    CartesianOptics
    FlexibleOptics

"""
aperture = apertures.ApertureFactory(16)
propagator = propagators.MFT(16, 1)
mask = np.ones((16, 16))


@pytest.fixture
def create_angular_optics():
    """Constructs the AngularOptics class for testing."""

    def _create_angular_optics(
        wf_npixels: int = 16,
        diameter: float = 1.0,
        aperture: Array = aperture,
        psf_npixels: int = 16,
        psf_pixel_scale: float = 1.0,
        psf_oversample: int = 1,
        mask: Array = mask,
    ):
        return optics.AngularOptics(
            wf_npixels=wf_npixels,
            diameter=diameter,
            aperture=aperture,
            psf_npixels=psf_npixels,
            psf_pixel_scale=psf_pixel_scale,
            psf_oversample=psf_oversample,
            mask=mask,
        )

    return _create_angular_optics


@pytest.fixture
def create_cartesian_optics():
    """Constructs the CartesianOptics class for testing."""

    def _create_cartesian_optics(
        wf_npixels: int = 16,
        diameter: float = 1.0,
        aperture: Array = aperture,
        focal_length: float = 10.0,
        psf_npixels: int = 16,
        psf_pixel_scale: float = 1.0,
        psf_oversample: int = 1,
        mask: Array = mask,
    ):
        return optics.CartesianOptics(
            wf_npixels=wf_npixels,
            diameter=diameter,
            aperture=aperture,
            focal_length=focal_length,
            psf_npixels=psf_npixels,
            psf_pixel_scale=psf_pixel_scale,
            psf_oversample=psf_oversample,
            mask=mask,
        )

    return _create_cartesian_optics


@pytest.fixture
def create_flexible_optics():
    """Constructs the FlexibleOptics class for testing."""

    def _create_flexible_optics(
        wf_npixels: int = 16,
        diameter: float = 1.0,
        aperture: Array = aperture,
        propagator: Array = propagator,
        mask: Array = mask,
    ):
        return optics.FlexibleOptics(
            wf_npixels=wf_npixels,
            diameter=diameter,
            aperture=aperture,
            propagator=propagator,
            mask=mask,
        )

    return _create_flexible_optics


@pytest.fixture
def create_layered_optics():
    """Constructs the LayeredOptics class for testing."""

    def _create_layered_optics(
        wf_npixels: int = 16,
        diameter: float = 1,
        layers: list = [aperture, propagator],
    ):
        return optics.LayeredOptics(
            wf_npixels=wf_npixels, diameter=diameter, layers=layers
        )

    return _create_layered_optics


"""
================================================================================
================================================================================
# Propagators
-------------

propagators.py classes:
    MFT
    FFT
    ShiftedMFT
    FarFieldFresnel

"""


@pytest.fixture
def create_fft():
    def _create_fft(
        focal_length: Array = 1.0, pad: int = 2, inverse: bool = False
    ):
        return propagators.FFT(
            focal_length=focal_length, pad=pad, inverse=inverse
        )

    return _create_fft


@pytest.fixture
def create_mft():
    def _create_mft(
        pixel_scale: Array = 1.0,
        npixels: int = 16,
        focal_length: Array = 1.0,
        inverse: bool = False,
    ):
        return propagators.MFT(
            npixels=npixels,
            pixel_scale=pixel_scale,
            focal_length=focal_length,
            inverse=inverse,
        )

    return _create_mft


@pytest.fixture
def create_shifted_mft():
    def _create_shifted_mft(
        pixel_scale: Array = 1.0,
        npixels: int = 16,
        shift: Array = np.zeros(2),
        focal_length: Array = None,
        pixel: bool = False,
        inverse: bool = False,
    ):
        return propagators.ShiftedMFT(
            npixels=npixels,
            pixel_scale=pixel_scale,
            shift=shift,
            focal_length=focal_length,
            inverse=inverse,
        )

    return _create_shifted_mft


@pytest.fixture
def create_far_field_fresnel():
    def _create_far_field_fresnel(
        npixels: int = 16,
        pixel_scale: float = np.array(1.0),
        focal_length: Array = np.array(1.0),
        focal_shift: Array = np.array(1e-3),
        shift: Array = np.zeros(2),
        pixel: bool = False,
        inverse: bool = False,
    ):
        return propagators.FarFieldFresnel(
            npixels=npixels,
            pixel_scale=pixel_scale,
            focal_length=focal_length,
            inverse=inverse,
            focal_shift=focal_shift,
            pixel=pixel,
        )

    return _create_far_field_fresnel


"""
================================================================================
================================================================================
# Optical Layers
----------------

optical_layers.py classes:
    Tilt
    Normalise
    Rotate
    Flip
    Resize
    Optic
    BasisOptic
    PhaseOptic
    PhaseBasisOptic

"""


@pytest.fixture
def create_tilt():
    """Constructs the Tilt class for testing."""

    def _create_tilt(angles: Array = np.ones(2)):
        return optical_layers.Tilt(angles=angles)

    return _create_tilt


@pytest.fixture
def create_normalise():
    """Constructs the Normalise class for testing."""

    def _create_normalise():
        return optical_layers.Normalise()

    return _create_normalise


@pytest.fixture
def create_rotate():
    """Constructs the Rotate class for testing."""

    def _create_rotate(
        angle: Array = np.array(np.pi), order: int = 1, complex: bool = False
    ):
        return optical_layers.Rotate(angle=angle, order=order, complex=complex)

    return _create_rotate


@pytest.fixture
def create_flip():
    """Constructs the Flip class for testing."""

    def _create_flip(axes: int = 0):
        return optical_layers.Flip(axes=axes)

    return _create_flip


@pytest.fixture
def create_resize():
    """Constructs the Resize class for testing."""

    def _create_resize(npixels: int = 16):
        return optical_layers.Resize(npixels=npixels)

    return _create_resize


@pytest.fixture
def create_optic():
    """Constructs the Optic class for testing."""

    def _create_optic(
        transmission: Array = np.ones((16, 16)),
        opd: Array = np.zeros((16, 16)),
        normalise: bool = True,
    ):
        return optical_layers.Optic(
            transmission=transmission, opd=opd, normalise=normalise
        )

    return _create_optic


@pytest.fixture
def create_basis_optic():
    """Constructs the BasisOptic class for testing."""

    def _create_basis_optic(
        transmission: Array = np.ones((16, 16)),
        basis: Array = np.ones((3, 16, 16)),
        coefficients: Array = np.zeros(3),
        normalise: bool = True,
    ):
        return optical_layers.BasisOptic(
            transmission=transmission,
            basis=basis,
            coefficients=coefficients,
            normalise=normalise,
        )

    return _create_basis_optic


@pytest.fixture
def create_phase_optic():
    """Constructs the PhaseOptic class for testing."""

    def _create_phase_optic(
        transmission: Array = np.ones((16, 16)),
        phase: Array = np.zeros((16, 16)),
        normalise: bool = True,
    ):
        return optical_layers.PhaseOptic(
            transmission=transmission, phase=phase, normalise=normalise
        )

    return _create_phase_optic


@pytest.fixture
def create_phase_basis_optic():
    """Constructs the PhaseBasisOptic class for testing."""

    def _create_phase_basis_optic(
        transmission: Array = np.ones((16, 16)),
        basis: Array = np.ones((3, 16, 16)),
        coefficients: Array = np.zeros(3),
        normalise: bool = True,
    ):
        return optical_layers.PhaseBasisOptic(
            transmission=transmission,
            basis=basis,
            coefficients=coefficients,
            normalise=normalise,
        )

    return _create_phase_basis_optic


"""
================================================================================
================================================================================
# Apertures
-----------

aperture.py classes:
    CircularAperture
    RectangularAperture
    RegPolyAperture
    IrregPolyAperture
    AberratedAperture
    UniformSpider
    CompoundAperture
    MultiAperture
    ApertureFactory

"""


@pytest.fixture
def create_circular_aperture():
    def _create_circular_aperture(
        radius: Array = 1.0,
        centre: Array = [0.0, 0.0],
        shear: Array = [0.0, 0.0],
        compression: Array = [1.0, 1.0],
        occulting: bool = False,
        softening: Array = 0.0,
        normalise: bool = True,
    ):
        return apertures.CircularAperture(
            radius=radius,
            centre=centre,
            shear=shear,
            compression=compression,
            occulting=occulting,
            softening=softening,
            normalise=normalise,
        )

    return _create_circular_aperture


@pytest.fixture
def create_rectangular_aperture():
    def _create_rectangular_aperture(
        height: Array = 0.5,
        width: Array = 1.0,
        centre: Array = [0.0, 0.0],
        shear: Array = [0.0, 0.0],
        compression: Array = [1.0, 1.0],
        rotation: Array = 0.0,
        occulting: bool = False,
        softening: Array = 0.0,
        normalise: bool = True,
    ):
        return apertures.RectangularAperture(
            height=height,
            width=width,
            centre=centre,
            shear=shear,
            compression=compression,
            rotation=rotation,
            occulting=occulting,
            softening=softening,
            normalise=normalise,
        )

    return _create_rectangular_aperture


@pytest.fixture
def create_reg_poly_aperture():
    def _create_reg_poly_aperture(
        nsides: int = 4,
        rmax: float = 0.5,
        centre: Array = [0.0, 0.0],
        shear: Array = [0.0, 0.0],
        compression: Array = [1.0, 1.0],
        rotation: Array = 0.0,
        occulting: bool = False,
        softening: Array = 0.0,
        normalise: bool = True,
    ):
        return apertures.RegPolyAperture(
            nsides=nsides,
            rmax=rmax,
            centre=centre,
            shear=shear,
            compression=compression,
            rotation=rotation,
            occulting=occulting,
            softening=softening,
            normalise=normalise,
        )

    return _create_reg_poly_aperture


@pytest.fixture
def create_irreg_poly_aperture():
    def _create_irreg_poly_aperture(
        vertices: Array = np.array(
            [[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5]]
        ),
        centre: Array = [0.0, 0.0],
        shear: Array = [0.0, 0.0],
        compression: Array = [1.0, 1.0],
        rotation: Array = 0.0,
        occulting: bool = False,
        softening: Array = 0.0,
        normalise: bool = True,
    ):
        return apertures.IrregPolyAperture(
            vertices=vertices,
            centre=centre,
            shear=shear,
            compression=compression,
            rotation=rotation,
            occulting=occulting,
            softening=softening,
            normalise=normalise,
        )

    return _create_irreg_poly_aperture


@pytest.fixture
def create_aberrated_aperture(create_circular_aperture):
    nterms = 6

    def _create_aberrated_aperture(
        aperture: object = create_circular_aperture(),
        noll_inds: list = np.arange(1, nterms + 1, dtype=int),
        coefficients: Array = np.zeros(nterms, dtype=float),
    ):
        return apertures.AberratedAperture(
            aperture=aperture, noll_inds=noll_inds, coefficients=coefficients
        )

    return _create_aberrated_aperture


@pytest.fixture
def create_uniform_spider():
    def _create_uniform_spider(
        nstruts: int = 4,
        strut_width: float = 0.05,
        centre: Array = [0.0, 0.0],
        shear: Array = [0.0, 0.0],
        compression: Array = [1.0, 1.0],
        rotation: Array = 0.0,
        softening: bool = False,
        normalise: bool = True,
    ):
        return apertures.UniformSpider(
            nstruts=nstruts,
            strut_width=strut_width,
            centre=centre,
            shear=shear,
            compression=compression,
            rotation=rotation,
            softening=softening,
            normalise=normalise,
        )

    return _create_uniform_spider


# Note the apertures input is mapped to 'apers' since apertures is already
# taken in the namespace for the aperture module
@pytest.fixture
def create_compound_aperture(create_circular_aperture):
    def _create_compound_aperture(
        apers: list = [create_circular_aperture()],
        centre: Array = np.array([0.0, 0.0]),
        shear: Array = np.array([0.0, 0.0]),
        compression: Array = np.array([1.0, 1.0]),
        rotation: Array = np.array(0.0),
        normalise: bool = True,
    ):
        return apertures.CompoundAperture(apertures=apers, normalise=normalise)

    return _create_compound_aperture


@pytest.fixture
def create_multi_aperture(create_circular_aperture):
    def _create_multi_aperture(
        apers: list = [create_circular_aperture()],
        centre: Array = np.array([0.0, 0.0]),
        shear: Array = np.array([0.0, 0.0]),
        compression: Array = np.array([1.0, 1.0]),
        rotation: Array = np.array(0.0),
        normalise: bool = True,
    ):
        return apertures.MultiAperture(apertures=apers, normalise=normalise)

    return _create_multi_aperture


@pytest.fixture
def create_aperture_factory(create_circular_aperture):
    def _create_aperture_factory(
        npixels: int = 16,
        radial_orders: Array = [0, 1, 2],
        coefficients: Array = None,
        noll_indices: Array = None,
        aperture_ratio: float = 1.0,
        secondary_ratio: float = 0.0,
        nsides: int = 0,
        secondary_nsides: int = 0,
        rotation: float = 0.0,
        nstruts: int = 0,
        strut_ratio: float = 0.0,
        strut_rotation: float = 0.0,
        normalise: bool = True,
    ):
        return apertures.ApertureFactory(
            npixels=npixels,
            radial_orders=radial_orders,
            coefficients=coefficients,
            noll_indices=noll_indices,
            aperture_ratio=aperture_ratio,
            secondary_ratio=secondary_ratio,
            nsides=nsides,
            secondary_nsides=secondary_nsides,
            rotation=rotation,
            nstruts=nstruts,
            strut_ratio=strut_ratio,
            strut_rotation=strut_rotation,
            normalise=normalise,
        )

    return _create_aperture_factory


"""=============================================================================
================================================================================
# Images
--------

images.py classes:
    Image

"""


@pytest.fixture
def create_image():
    def _create_image(
        image: Array = np.ones((16, 16)), pixel_scale: float = 1 / 16
    ):
        return images.Image(image=image, pixel_scale=pixel_scale)

    return _create_image


"""=============================================================================
================================================================================
# Detectors
-----------

Detectors.py classes:
    LayeredDetector

"""


@pytest.fixture
def create_layered_detector(create_constant):
    def _create_layered_detector(layers=[create_constant()]):
        return detectors.LayeredDetector(layers=layers)

    return _create_layered_detector


"""=============================================================================
================================================================================
# Image Layers
--------------

image_layers.py classes:
    ApplyPixelResponse
    ApplyJitter
    ApplySaturation
    AddConstant
    IntegerDownsample
    Rotate

"""


@pytest.fixture
def create_pixel_response():
    def _create_pixel_response(pixel_response: Array = np.ones((16, 16))):
        return detector_layers.ApplyPixelResponse(
            pixel_response=pixel_response
        )

    return _create_pixel_response


@pytest.fixture
def create_jitter():
    def _create_jitter(sigma: float = 1.0, kernel_size: int = 10):
        return detector_layers.ApplyJitter(
            sigma=sigma, kernel_size=kernel_size
        )

    return _create_jitter


@pytest.fixture
def create_saturation():
    def _create_saturation(saturation: float = 1.0):
        return detector_layers.ApplySaturation(saturation=saturation)

    return _create_saturation


@pytest.fixture
def create_constant():
    def _create_constant(value: Array = np.array(1.0)):
        return detector_layers.AddConstant(value=value)

    return _create_constant


@pytest.fixture
def create_integer_downsample():
    def _create_integer_downsample(kernel_size: int = 2):
        return detector_layers.IntegerDownsample(kernel_size=kernel_size)

    return _create_integer_downsample


@pytest.fixture
def create_rotate_detector():
    def _create_rotate_detector(
        angle: Array = np.array(np.pi), order: int = 1
    ):
        return detector_layers.Rotate(angle=angle, order=order)

    return _create_rotate_detector


"""
================================================================================
================================================================================
Sources
-------

sources.py classes:
    Source
    PointSources
    BinarySource
    ResolvedSource
    PointResolvedSource

"""


@pytest.fixture
def create_point_source():
    def _create_point_source(
        wavelengths: Array = np.array([1e-6]),
        position: Array = np.zeros(2),
        flux: Array = np.array(1.0),
        weights: Array = None,
        spectrum: spectra.Spectrum = None,
    ):
        return sources.PointSource(
            wavelengths=wavelengths,
            position=position,
            flux=flux,
            spectrum=spectrum,
            weights=weights,
        )

    return _create_point_source


@pytest.fixture
def create_point_sources():
    def _create_point_sources(
        wavelengths: Array = np.array([1e-6]),
        position: Array = np.zeros((3, 2)),
        flux: Array = np.ones(3),
        weights: Array = None,
        spectrum: spectra.Spectrum = None,
    ):
        return sources.PointSources(
            wavelengths=wavelengths,
            position=position,
            flux=flux,
            spectrum=spectrum,
            weights=weights,
        )

    return _create_point_sources


@pytest.fixture
def create_resolved_source():
    def _create_resolved_source(
        wavelengths: Array = np.array([1e-6]),
        position: Array = np.zeros(2),
        flux: Array = np.array(1.0),
        distribution: Array = np.ones((5, 5)),
        weights: Array = None,
        spectrum: spectra.Spectrum = None,
    ):
        return sources.ResolvedSource(
            wavelengths=wavelengths,
            position=position,
            flux=flux,
            distribution=distribution,
            spectrum=spectrum,
            weights=weights,
        )

    return _create_resolved_source


@pytest.fixture
def create_binary_source():
    def _create_binary_source(
        wavelengths: Array = np.array([1e-6]),
        position: Array = np.array([0.0, 0.0]),
        flux: Array = np.array(1.0),
        separation: Array = np.array(1.0),
        position_angle: Array = np.array(0.0),
        contrast: Array = np.array(2.0),
        spectrum: spectra.Spectrum = None,
        weights: Array = None,
    ):
        return sources.BinarySource(
            position=position,
            flux=flux,
            separation=separation,
            position_angle=position_angle,
            contrast=contrast,
            spectrum=spectrum,
            wavelengths=wavelengths,
            weights=weights,
        )

    return _create_binary_source


@pytest.fixture
def create_point_resolved_source():
    def _create_point_resolved_source(
        wavelengths: Array = np.array([1e-6]),
        position: Array = np.array([0.0, 0.0]),
        flux: Array = np.array(1.0),
        distribution: Array = np.ones((5, 5)),
        contrast: Array = np.array(2.0),
        spectrum: spectra.Spectrum = None,
        weights: Array = None,
    ):
        return sources.PointResolvedSource(
            position=position,
            flux=flux,
            distribution=distribution,
            contrast=contrast,
            spectrum=spectrum,
            wavelengths=wavelengths,
        )

    return _create_point_resolved_source


"""
================================================================================
================================================================================
Spectra
-------

spectra.py classes:
    Spectrum
    PolySpectrum

"""


@pytest.fixture
def create_spectrum():
    def _create_spectrum(
        wavelengths: Array = np.linspace(1e-6, 1.2e-6, 5),
        weights: Array = np.arange(5),
    ):
        return spectra.Spectrum(wavelengths, weights)

    return _create_spectrum


@pytest.fixture
def create_poly_spectrum():
    def _create_poly_spectrum(
        wavelengths: Array = np.linspace(1e-6, 1.2e-6, 5),
        coefficients: Array = np.arange(3),
    ):
        return spectra.PolySpectrum(wavelengths, coefficients)

    return _create_poly_spectrum


"""
================================================================================
================================================================================
Instruments
-----------

instruments.py classes:
    Instrument

"""


@pytest.fixture
def create_instrument(
    create_angular_optics,
    create_point_source,
    create_layered_detector,
    create_dither,
):
    def _create_instrument(
        optics=create_angular_optics(),
        sources=create_point_source(),
        detector=create_layered_detector(),
        observation=create_dither(),
    ):
        return instruments.Instrument(
            optics=optics,
            detector=detector,
            sources=sources,
            observation=observation,
        )

    return _create_instrument


"""
================================================================================
================================================================================
Observations
------------

observations.py classes:
    Dither

"""


@pytest.fixture
def create_dither():
    def _create_dither(dithers: Array = np.ones((5, 2))):
        return observations.Dither(dithers=dithers)

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
