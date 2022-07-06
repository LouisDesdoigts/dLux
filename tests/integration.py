"""
tests/integration.py
--------------------
This file generates _P_oint _S_pread _F_unctions through a subset of 
the layers and the propagators. It generates plots of the outputs 
that then need to be manually checked. 
"""

from dLux import *

# NOTE: So I plan to be using the Toliman. Let's get this running 
# and then fix the tilt problem in the propagators.

class Toliman(OpticalSystem):
    """
    A complete parametrisation of the Toliman mission telescope. 
    This is just used for generating PSFs for testing purposes.
    May be updated to a separate file later on.

    Attributes
    ---------- 
    wavefront_pixels : int = 256
        The number of pixels in the input wavefront.
    detector_pixels : int = 256
        The number of pixels in the detector plane.
    outer_aperture_radius : float = 0.125
        The outer radius of the toliman aperture in meters.
    inner_aperture_radius : flaot = 0.02
        The inner radius of the toliman aperture in meters.
    focal_shift : float = 0.
        The shift of the detector from the focal plane.
    focal_length : float = 0.
        The distance to the focal plane of the telescope. The 
        lens is implicit. 
    pixel_oversample : int = 5
        The ultimate resolution of the simulation. The number 
        of pixels to simulate per measurable pixel.
    pixel_scale_out : float = 6.5e-06
        The scale of the pixels in the detector plane in meters.
    """
    wavefront_pixels : int
    detector_pixels : int
    outer_aperture_radius : float
    inner_aperture_radius : float
    focal_shift : float
    focal_length : float
    pixel_oversample : int
    pixel_scale_out : float


    def __init__(self : Telescope, 
            extra_layers : list = [],
            wavefront_pixels : int = 256,
            detector_pixels : int = 256,
            outer_aperture_radius : float = 0.125
            inner_aperture_radius : float = 0.02,
            focal_shift : float = 0.,
            focal_length : float = 1.32,
            pixel_oversample : int = 5, 
            pixel_scale_out : float = 6.5e-06,
            in_focus : bool = True):
        """
        Parameters
        ----------
        extra_layers : list = []
            Additional operations to add to the optical train. By 
            default it is an empty list.
        wavefront_pixels : int = 256
            The number of pixels in the input wavefront.
        detector_pixels : int = 256
            The number of pixels in the detector plane.
        outer_aperture_radius : float = 0.125
            The outer radius of the toliman aperture in meters.
        inner_aperture_radius : flaot = 0.02
            The inner radius of the toliman aperture in meters.
        focal_shift : float = 0.
            The shift of the detector from the focal plane.
        focal_length : float = 0.
            The distance to the focal plane of the telescope. The 
            lens is implicit. 
        pixel_oversample : int = 5
            The ultimate resolution of the simulation. The number 
            of pixels to simulate per measurable pixel.
        pixel_scale_out : float = 6.5e-06
            The scale of the pixels in the detector plane in meters.
        in_focus : bool = True
            Defines if the `PhysicalFresnel` propagator is to be used.
        """
        self.wavefront_pixels = wavefront_pixels
        self.detector_pixels = detector_pixels
        self.outer_aperture_radius = outer_aperture_radius
        self.inner_aperture_radius = inner_aperture_radius
        self.focal_shift = focal_shift
        self.focal_length = focal_length
        self.pixel_oversample = pixel_oversample
        self.pixel_scale_out = pixel_scale_out

        layers = [
            CreateWavefront(wavefront_pixels, outer_aperture_radius),
            TiltWavefront(),
            CircularAperture(wavefront_pixels, 
                rmin = inner_aperture_radius / outer_aperture_radius),
            NormaliseWavefront()]
    
        layers.extend(extra_layers)
    
        if in_focus:
            layers.append(PhysicalMFT(
                pixels_out = detector_pixels,
                focal_length = focal_length, 
                pixel_scale_out = pixel_scale_out / pixel_oversample, 
                inverse = False, 
                tilt = False))
        else:
            layers.append(PhysicalFresnel(
                pixels_out = detector_pixels, 
                focal_length = focal_length, 
                focal_shift = focal_shift,
                pixel_scale_out = pixel_scale_out / pixel_oversample, 
                inverse = False,
                tilt = False))
    
        self.layers = layers
