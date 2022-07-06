"""
tests/integration.py
--------------------
This file generates _P_oint _S_pread _F_unctions through a subset of 
the layers and the propagators. It generates plots of the outputs 
that then need to be manually checked. 
"""

from dLux import *
from typing import NewType
from matplotlib import pyplot

Telescope = NewType("Telescope", OpticalSystem)

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
            outer_aperture_radius : float = 0.125,
            inner_aperture_radius : float = 0.02,
            focal_shift : float = 0.,
            focal_length : float = 1.32,
            pixel_oversample : int = 5, 
            pixel_scale_out : float = 6.5e-06):
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
        super().__init__(layers, wavels=[550e-09])


    def get_wavefront_pixels(self : Telescope) -> int:
        """
        Returns
        -------
        wavefront_pixels : int
            The number of pixels in the input layer.
        """
        return self.wavefront_pixels


    def get_detector_pixels(self : Telescope) -> int:
        """
        Returns
        -------
        detector_pixels : int
            The number of pixels in the ouput layer
        """
        return self.detector_layers


    def get_outer_aperture_radius(self : Telescope) -> float:
        """
        Returns
        -------
        outer_aperture_radius : float
            The radius of the outer rim of the annular toliman 
            aperture in meters.
        """
        return self.outer_aperture_radius


    def get_inner_aperture_radius(self : Telescope) -> float:
        """
        Returns
        -------
        inner_aperture_radius : float
            The radius of the inner rim of the annular toliman 
            aperture.
        """
        return self.inner_aperture_radius


    def get_focal_shift(self : Telescope) -> float:
        """
        Returns 
        -------
        focal_shift : float
            The displacement of the detector layer from the focal
            plane of the toliman telescope.
        """
        return self.focal_shift

 
    def get_focal_length(self : Telescope) -> float:
        """
        Returns
        -------
        focal_length : float
            The focal length of the toliman telescope in meters.
        """
        return self.focal_length


    def get_pixel_oversample(self : Telescope) -> float:
        """
        Returns
        -------
        oversample : int
            The number of simulated pixels per physical pixel in this 
            simulation of the toliman optical train.
        """
        return self.pixel_oversample


    def get_pixel_scale_out(self : Telescope) -> float:
        """
        Returns
        -------
        pixel_scale_out : float 
            The side length of a physical pixel in the toliman detector
            plane in meters.
        """
        returns self.pixel_scale_out


    # TODO: This is very shit.
    def set_propagator(self : Telescope, in_focus : bool, fixed : bool) -> None:
        """
        Choose your propagator.

        Parameters
        ----------
        in_focus : bool
            True if the MFT is to be used else the fresnel will be used.
        fixed : bool
            Can only be set if in focus is True. True if the FFT is to 
            be used.
        """
        if fixed:
            layers.append(PhysicalFFT(
                inverse = False,
                focal_length = self.get_focal_length()))
        else:
            if in_focus:
                layers.append(PhysicalMFT(
                    pixels_out = self.get_detector_pixels(),
                    focal_length = self.get_focal_length(), 
                    pixel_scale_out = self.get_pixel_scale_out() / \
                        self.get_pixel_oversample(), 
                    inverse = False, 
                    tilt = False))
            else:
                layers.append(PhysicalFresnel(
                    pixels_out = self.get_detector_pixels(), 
                    focal_length = self.get_focal_length(), 
                    focal_shift = self.get_focal_shift(),
                    pixel_scale_out = self.get_pixel_scale_out() /\
                        self.get_pixel_oversample(), 
                    inverse = False,
                    tilt = False))

    
image = Toliman()() ** 0.5

pyplot.figure(figsize = (10, 10))
pyplot.title("PSF")
pyplot.imshow(image)
pyplot.colorbar()
pyplot.show()
