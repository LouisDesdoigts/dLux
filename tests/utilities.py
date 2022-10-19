from __future__ import annotations
import jax.numpy as np
import dLux
import typing
import abc

__author__ = "Jordan Dennis"
__date__ = "28/06/2022"

Array = typing.NewType("Array", np.ndarray)
PlaneType = typing.NewType("PlaneType", dLux.wavefronts.PlaneType)
Wavefront = typing.NewType("Wavefront", dLux.wavefronts.Wavefront)
Propagator = typing.NewType("Propagator", dLux.propagators.Propagator)


class UtilityUser():
    """
    The base utility class. These utility classes are designed to 
    define safe constructors and constants for testing. These   
    classes are for testing purposes only. 
    """
    utility : Utility


    def get_utility(self : UtilityUser) -> Utility:
        """
        Accessor for the utility. 

        Returns 
        -------
        utility : Utility
            The utility
        """
        return self.utility


class Utility():
    """
    """
    def __init__(self : Utility) -> Utility:
        """
        Construct a new Utility.

        Returns
        : Utility 
            The utility. 
        """
        pass

    
    def construct(self : Utility) -> object:
        """
        Safe constructor for the dLuxModule, associated with 
        this utility.

        Returns
        -------
        : dLuxModule
            A safe dLuxModule for testing.
        """
        pass


    def approx(self : Utility, result : Array, comparator : Array) -> Array:
        """
        Compare two arrays to within floating point precision.

        Parameters
        ----------
        result : Array
            The result that you want to test for nearness to
            comparator.
        comparator : Array
            The comparison array.

        Returns
        -------
        : Array[bool]
            True if the array elements are similar to float 
            error, False otherwise. 
        """
        lower_bound = (result - 0.0005) <= comparator
        upper_bound = (result + 0.0005) >= comparator
        return lower_bound & upper_bound 


class WavefrontUtility(Utility):
    """
    Defines safe state constants and a simple constructor for a safe
    `Wavefront` object. 

    Attributes
    ----------
    offset : Array[float]
        A simple array defining the angular displacement of the 
        wavefront. 
    wavelength : float
        A safe wavelength for the testing wavefronts in meters
    """
    wavelength : float
    offset : Array
    size : int
    amplitude : Array 
    phase : Array
    pixel_scale : float
    plane_type : PlaneType


    def __init__(self : Utility, /,
            wavelength : float = None, 
            offset : Array = None,
            size : int = None,
            amplitude : Array = None,
            phase : Array = None,
            pixel_scale : float = None,
            plane_type : PlaneType = None) -> Utility:
        """
        Parameters
        ----------
        wavelength : float = 550e-09
            The safe wavelength for the utility in meters.
        offset : Array = [0., 0.]
            The safe offset for the utility in meters.
        size : int
            A parameter for defining consistent wavefront pixel arrays 
            without causing errors.
        amplitude : Array[float] 
            A simple array defining electric field amplitudes without 
            causing errors.
        phase : Array[float]
            A simple array defining the pixel phase for a wavefront, 
            defined to be safe. 
        pixel_scale : float
            The scale of the pixels in the wavefront in units of 
            (radians) meters per pixel.

        Returns 
        -------
        utility : Utility 
            The new utility for generating test cases.
        """
        self.wavelength = 550e-09 if not wavelength else wavelength
        self.offset = np.array([0., 0.]).astype(float) if not \
            offset else np.array(offset).astype(float)           
        self.size = 128 if not size else size
        self.amplitude = np.ones((1, self.size, self.size)) if not \
            amplitude else amplitude
        self.phase = np.zeros((1, self.size, self.size)) if not \
            phase else phase
        self.pixel_scale = 1. if not pixel_scale else pixel_scale
        self.plane_type = dLux.PlaneType.Pupil if not \
            plane_type else pixel_scale

        assert self.size == self.amplitude.shape[-1]
        assert self.size == self.amplitude.shape[-2]
        assert self.size == self.phase.shape[-1]
        assert self.size == self.phase.shape[-2]          


    def construct(self : Utility) -> Wavefront:
        """
        Build a safe wavefront for testing.

        Returns 
        -------
        wavefront : Wavefront
            The safe testing wavefront.
        """
        return dLux.Wavefront(self.wavelength, self.offset, \
              self.pixel_scale, self.plane_type, self.amplitude, self.phase)


    def get_wavelength(self : Utility) -> float:
        """
        Accessor for the wavelength associated with this utility.

        Returns
        -------
        wavelength : int
            The wavelength of the utility and hence any `Wavefront` 
            objects it creates in meters.
        """
        return self.wavelength


    def get_size(self : Utility) -> int:
        """
        Accessor for the `size` constant.

        Returns
        -------
        size : int
            The side length of a pixel array currently stored.
        """
        return self.size


    def get_amplitude(self : Utility) -> Array:
        """
        Accessor for the `amplitude` constant.

        Returns 
        -------
        amplitude : Array
            The square array of pixel amplitudes in SI units of 
            electric field.
        """
        return self.amplitude


    def get_phase(self : Utility) -> Array:
        """
        Accessor for the `phase` constant.

        Returns
        -------
        phase : Array
            The square array of pixel phases in radians.
        """
        return self.phase


    def get_offset(self : Utility) -> Array:
        """
        Accessor for the `offset` constant.

        Returns
        -------
        offset : Array
            The angle that the wavefront makes with the x and 
            y planes in radians.
        """
        return self.offset


    def get_pixel_scale(self : Utility) -> Array:
        """
        Accessor for the `pixel_scale` constant.

        Returns
        -------
        pixel_scale : Array
            The `pixel_scale` associated with the wavefront.
        """
        return self.pixel_scale
    
    def get_plane_type(self : Utility) -> PlaneType:
        """
        Accessor for the `plane_type` attribute.

        Returns
        -------
        plane_type : Array
            The `plane_type` associated with the wavefront.
        """
        return self.plane_type


class CartesianWavefrontUtility(WavefrontUtility):
    """
    Defines useful safes state constants as well as a basic 
    constructor for a safe `CartesianWavefront`.
    """
    def __init__(self : Utility, /,
            wavelength : float = None, 
            offset : Array = None,
            size : int = None, 
            amplitude : Array = None, 
            phase : Array = None,
            pixel_scale : float = None,
            plane_type : PlaneType = None) -> Utility:
        """
        Parameters
        ----------
        wavelength : float 
            The safe wavelength to use for the constructor in meters.
        offset : Array[float]
            The safe offset to use for the constructor in radians.
        size : int
            The static size of the pixel arrays.
        amplitude : Array[float]
            The electric field amplitudes in SI units for electric
            field.
        phase : Array[float]
            The phases of each pixel in radians. 
        pixel_scale : float
            The scale of the output pixels in units of (radians) meters
            per pixel

        Returns
        -------
        utility : CartesianWavefrontUtility 
            A helpful class for implementing the tests. 
        """
        super().__init__(wavelength, offset, size, amplitude, phase, 
            pixel_scale, plane_type)


class AngularWavefrontUtility(WavefrontUtility):
    """
    Defines useful safes state constants as well as a basic 
    constructor for a safe `CartesianWavefront`.
    """
    def __init__(self : Utility, /,
            wavelength : float = None, 
            offset : Array = None,
            size : int = None, 
            amplitude : Array = None, 
            phase : Array = None,
            pixel_scale : float = None,
            plane_type : PlaneType = None) -> Utility:
        """
        Parameters
        ----------
        wavelength : float 
            The safe wavelength to use for the constructor in meters.
        offset : Array[float]
            The safe offset to use for the constructor in radians.
        size : int
            The static size of the pixel arrays.
        amplitude : Array[float]
            The electric field amplitudes in SI units for electric
            field.
        phase : Array[float]
            The phases of each pixel in radians. 
        pixel_scale : float
            The scale of the output pixels in units of (radians) meters
            per pixel

        Returns
        -------
        wavefront : AngularWavefrontUtility 
            A helpful class for implementing the tests. 
        """
        super().__init__(wavelength, offset, size, amplitude, phase,
            pixel_scale, plane_type)


# class GaussianWavefrontUtility(CartesianWavefrontUtility):
#     """
#     Defines safe state constants and a simple constructor for a 
#     safe state `GaussianWavefront` object. 

#     Attributes
#     ----------
#     beam_radius : float
#         A safe radius for the GaussianWavefront in meters.
#     phase_radius : float
#         A safe phase radius for the GaussianWavefront in radians.
#     position : float
#         A safe position for the GaussianWavefront in meters.
#     """
#     beam_radius : float 
#     phase_radius : float
#     position : float


#     def __init__(self : Utility, 
#             wavelength : float = None,
#             offset : Array = None,
#             size : int = None,
#             amplitude : Array = None,
#             phase : Array = None,
#             beam_radius : float = None,
#             phase_radius : float = None,
#             position : float = None) -> Utility:
#         """
#         Parameters
#         ----------
#         wavelength : float 
#             The safe wavelength to use for the constructor in meters.
#         offset : Array[float]
#             The safe offset to use for the constructor in radians.
#         size : int
#             The static size of the pixel arrays.
#         amplitude : Array[float]
#             The electric field amplitudes in SI units for electric
#             field.
#         phase : Array[float]
#             The phases of each pixel in radians.
#         beam_radius : float 
#             The radius of the gaussian beam in meters.
#         phase_radius : float
#             The phase radius of the gaussian beam in radians.
#         position : float
#             The position of the gaussian beam in meters.

#         Returns
#         -------
#         utility : GaussianWavefrontUtility 
#             A helpful class for implementing the tests. 
#         """
#         super().__init__(wavelength, offset, size, amplitude, phase)
#         self.beam_radius = 1. if not beam_radius else beam_radius
#         self.phase_radius = np.inf if not phase_radius else phase_radius
#         self.position = 0. if not position else position


#     # TODO: get_beam_radius and get_position
#     def get_phase_radius(self : Utility) -> float:
#         """
#         Returns
#         -------
#         phase_radius : float
#             The phase radius safe state.
#         """
#         return self.phase_radius


#     def get_beam_radius(self : Utility) -> float:
#         """
#         Returns
#         -------
#         beam_radius : float
#             The safe beam radius in meters.
#         """
#         return self.beam_radius


#     def get_position(self : Utility) -> float:
#         """
#         Returns
#         -------
#         position : float
#             The safe position in meters
#         """
#         return self.position


#     def construct(self : Utility) -> Wavefront:
#         """
#         Build a safe wavefront for testing.

#         Returns 
#         -------
#         wavefront : CartesianWavefront
#             The safe testing wavefront.
#         """
#         wavefront = dLux\
#             .GaussianWavefront(self.offset,
#                self.wavelength)\
#             .update_phasor(self.amplitude, self.phase)\
#             .set_pixel_scale(self.pixel_scale)\
#             .set_position(self.position)\
#             .set_phase_radius(np.inf)\
#             .set_beam_radius(self.beam_radius)

#         return wavefront


class PropagatorUtility(Utility):
    """
    Testing utility for the Propagator (abstract) class.

    Attributes 
    ----------
    inverse : bool
        The directionality of the generated propagators. 
    """
    inverse : bool
    dLux.propagators.Propagator.__abstractmethods__ = ()


    def __init__(self : Utility) -> Utility:
        """
        Initialises a safe state for the Propagator attributes 
        stored as attributes in this Utility.
        """
        self.inverse = False


    def construct(self : Utility, inverse : bool = None) -> Propagator:
        """
        Returns 
        -------
        propagator : Propagator 
            A safe propagator for testing purposes. 
        """
        return dLux.propagators.Propagator(
            self.inverse if inverse is None else inverse)


    def is_inverse(self : Utility) -> bool:
        """
        Returns
        -------
        inverse : bool
            The safe inverse setting of the Utility.
        """
        return self.inverse


class VariableSamplingUtility(PropagatorUtility, UtilityUser):
    """
    Container of useful functions and constructors for testing the 
    VariableSamplingPropagator (abstract) class.

    Attributes
    ----------
    utility : Utility 
        A utility for building `Wavefront` objects that interact 
        with the `VariableSamplingPropagator`.
    pixels_out : int
        The safe number of pixels in the output plane for the 
        `Propagator`.
    pixel_scale_out : float
        The safe pixel scale in the output plane for the `Propagator`.
        The units are (radians) meters per pixel.
    """
    utility : Utility = WavefrontUtility()
    pixels_out : int
    pixel_scale_out : float
    dLux.propagators.VariableSamplingPropagator.__abstractmethods__ = ()
   

    def __init__(self : Utility) -> Utility:
        """
        Initialises a safe state for the Propagator attributes 
        stored as attributes in this Utility.
        """
        super().__init__()
        self.pixels_out = 256
        self.pixel_scale_out = 1.e-3


    def construct(self : Utility, /, inverse : bool = None, 
            pixels_out : int = None, 
            pixel_scale_out : float = None, tilt : bool = False) -> Propagator:
        """
        Build a safe `VariableSamplingPropagator` for testing purposes.

        Parameters
        ----------
        inverse : bool
            True if the inverse `Propagtor` is to be set.
        pixels_out : int
            The number of pixels in the output plane.
        pixel_scale_out : float
            The pixel scale in the output plane in units of (radians)
            meters per pixel.
        
        Returns
        -------
        propagator : Propagator
            The safe testing `Propagator`
        """
        # TODO: These should not be accessible in the importable 
        # dLux. need to confer with @LouisDesdoigts.
        return dLux.propagators.VariableSamplingPropagator(
            tilt = tilt,
            inverse = self.is_inverse() if inverse is None else inverse,
            pixels_out = self.pixels_out if pixels_out is None \
                else pixels_out,
            pixel_scale_out = self.pixel_scale_out if pixel_scale_out \
                is None else pixel_scale_out) 


    def get_pixels_out(self : Utility) -> int:
        """
        Returns
        -------
        pixels_out : int
            The number of pixels in the output plane for the safe
            `Propagator`
        """
        return self.pixels_out


    def get_pixel_scale_out(self : Utility) -> float:
        """
        Returns
        -------
        pixel_scale_out : float
            The pixel scale in the output plane for the safe `Propagator`
        """
        return self.pixel_scale_out


class FixedSamplingUtility(PropagatorUtility, UtilityUser):
    """
    Container of useful functions and constructors for testing the 
    FixedSamplingPropagator (abstract) class.

    Attributes
    ----------
    utility : Utility 
        A utility for building `Wavefront` objects that interact 
        with the `FixedSamplingPropagator`.
    """
    utility : Utility = WavefrontUtility()
    dLux.propagators.FixedSamplingPropagator.__abstractmethods__ = ()


    def __init__(self : Utility) -> Utility:
        """
        Initialises a safe state for the Propagator attributes 
        stored as attributes in this Utility.
        """
        super().__init__()


    def construct(self : Utility, inverse : bool = False) -> Propagator:
        """
        Build a safe `FixedSamplingPropagator` for testing purposes.

        Parameters
        ----------
        inverse : bool
            True if the inverse `Propagtor` is to be set.
        
        Returns
        -------
        propagator : Propagator
            The safe testing `Propagator`
        """
        return dLux.propagators.FixedSamplingPropagator(
            self.is_inverse() if inverse is None else inverse)


class PhysicalMFTUtility(VariableSamplingUtility, UtilityUser):
    """
    Container of useful functions and constructors for testing the 
    PhysicalMFT class.

    Attributes
    ----------
    utility : Utility 
        A utility for building `Wavefront` objects that interact 
        with the `PhysicalMFT`.
    focal_length : float
        The safe focal length of the lens or mirror associated with 
        the porpagation.
    """
    utility : Utility = CartesianWavefrontUtility()
    focal_length : float


    def __init__(self : Utility) -> Utility:
        """
        Initialises a safe state for the Propagator attributes 
        stored as attributes in this Utility.
        """
        super().__init__()
        self.focal_length = 1.


    def construct(self : Utility, inverse : bool = None, 
            pixels_out : int = None, pixel_scale_out : float = None, 
            focal_length = None, tilt : bool = False) -> Propagator:
        """
        Build a safe `PhysicalMFT` for testing purposes.

        Parameters
        ----------
        inverse : bool
            True if the inverse `Propagtor` is to be set.
        pixels_out : int
            The number of pixels in the output plane.
        pixel_scale_out : float
            The pixel scale in the output plane in units of (radians)
            meters per pixel.
        focal_length : float
            The focal length associated with the mirror or lens 
            associated with the propagation.
        
        Returns
        -------
        propagator : Propagator
            The safe testing `Propagator`
        """
        return dLux.PhysicalMFT(
            tilt = tilt,
            inverse = self.is_inverse() if inverse is None else inverse,
            pixels_out = self.get_pixels_out() if pixels_out is None else pixels_out,
            pixel_scale_out = self.get_pixel_scale_out() if pixel_scale_out is None else pixel_scale_out,
            focal_length = self.get_focal_length() if focal_length is None else focal_length)


    def get_focal_length(self : Utility) -> float:
        """
        Returns
        -------
        focal_length : float
            The focal length in meters of the mirror or lens associated
            with the propagation.
        """
        return self.focal_length


class PhysicalFFTUtility(FixedSamplingUtility, UtilityUser):
    """
    Container of useful functions and constructors for testing the 
    PhysicalFFT class.

    Attributes
    ----------
    utility : Utility 
        A utility for building `Wavefront` objects that interact 
        with the `PhysicalFFT`.
    focal_length : float
        The safe focal length of the lens or mirror associated with 
        the porpagation.
    """

    utility : Utility = CartesianWavefrontUtility()
    focal_length : float


    def __init__(self : Utility) -> Utility:
        """
        Initialises a safe state for the Propagator attributes 
        stored as attributes in this Utility.
        """
        super().__init__()
        self.focal_length = 1.


    def construct(self : Utility, inverse : bool = None, 
            focal_length = None) -> Propagator:
        """
        Build a safe `PhysicalFFT` for testing purposes.

        Parameters
        ----------
        inverse : bool
            True if the inverse `Propagtor` is to be set.
        focal_length : float
            The focal length associated with the mirror or lens 
            associated with the propagation.
        
        Returns
        -------
        propagator : Propagator
            The safe testing `Propagator`
        """
        return dLux.PhysicalFFT(
            inverse = self.is_inverse() if inverse is None else inverse,
            focal_length = self.get_focal_length() if focal_length is None else focal_length)


    # TODO: Set up a FocalPlane abstract class. 
    def get_focal_length(self : Utility) -> float:
        """
        Returns
        -------
        focal_length : float
            The focal length in meters of the mirror or lens associated
            with the propagation.
        """
        return self.focal_length


# class PhysicalFresnelUtility(VariableSamplingUtility, UtilityUser):
#     """
#     Container of useful functions and constructors for testing the 
#     PhysicalFresnel class.

#     Attributes
#     ----------
#     utility : Utility 
#         A utility for building `Wavefront` objects that interact 
#         with the `PhysicalFresnel`.
#     focal_length : float
#         The safe focal length of the lens or mirror associated with 
#         the porpagation.
#     focal_shift : float
#         The shift away from focus that the Fresnel approximation is
#         to be applied to.
#     """
#     utility : Utility = CartesianWavefrontUtility()
#     focal_length : float
#     focal_shift : float


#     def __init__(self : Utility) -> Utility:
#         """
#         Initialises a safe state for the Propagator attributes 
#         stored as attributes in this Utility.
#         """
#         super().__init__()
#         self.focal_length = 1.
#         self.focal_shift = -.01


#     def construct(self : Utility, inverse : bool = None, 
#             pixels_out : int = None, pixel_scale_out : float = None, 
#             focal_length = None, focal_shift :float = None,
#             tilt = False) -> Propagator:
#         """
#         Build a safe `PhysicalFresnel` for testing purposes.

#         Parameters
#         ----------
#         inverse : bool
#             True if the inverse `Propagtor` is to be set.
#         pixels_out : int
#             The number of pixels in the output plane.
#         pixel_scale_out : float
#             The pixel scale in the output plane in units of (radians)
#             meters per pixel.
#         focal_length : float
#             The focal length associated with the mirror or lens 
#             associated with the propagation.
#         focal_shift : float
#             The disparity from the focal length to which the Fresnel
#             approximation is to be applied.
        
#         Returns
#         -------
#         propagator : Propagator
#             The safe testing `Propagator`
#         """
#         return dLux.PhysicalFresnel(
#             tilt = tilt,
#             inverse = self.is_inverse() if inverse is None else inverse,
#             pixels_out = self.get_pixels_out() if pixels_out is None else pixels_out,
#             pixel_scale_out = self.get_pixel_scale_out() if pixel_scale_out is None else pixel_scale_out,
#             focal_length = self.get_focal_length() if focal_length is None else focal_length,
#             focal_shift = self.get_focal_shift() if focal_shift is None else focal_shift)


#     def get_focal_length(self : Utility) -> float:
#         """
#         Returns
#         -------
#         focal_length : float
#             The focal length in meters of the mirror or lens associated
#             with the propagation.
#         """
#         return self.focal_length


#     def get_focal_shift(self : Utility) -> float:
#         """
#         Returns
#         -------
#         focal_shift : float
#             The shift from the focal plane of the detector in meters.
#         """
#         return self.focal_shift


class AngularMFTUtility(VariableSamplingUtility, UtilityUser):
    """
    Container of useful functions and constructors for testing the 
    AngluarMFT class.

    Attributes
    ----------
    utility : Utility 
        A utility for building `Wavefront` objects that interact 
        with the `AngularMFT`.
    """
    utility : Utility = AngularWavefrontUtility()


    def __init__(self : Utility) -> Utility:
        """
        Initialises a safe state for the Propagator attributes 
        stored as attributes in this Utility.
        """
        super().__init__()


    def construct(self : Utility, inverse : bool = None, 
            pixels_out : int = None, pixel_scale_out : float = None, 
            focal_length = None, focal_shift :float = None,
            tilt : bool = False) -> Propagator:
        """
        Build a safe `PhysicalMFT` for testing purposes.

        Parameters
        ----------
        inverse : bool
            True if the inverse `Propagtor` is to be set.
        pixels_out : int
            The number of pixels in the output plane.
        pixel_scale_out : float
            The pixel scale in the output plane in units of (radians)
            meters per pixel.
        
        Returns
        -------
        propagator : Propagator
            The safe testing `Propagator`
        """
        return dLux.AngularMFT(
            tilt = tilt,
            inverse = self.is_inverse() if inverse is None else inverse,
            pixels_out = self.get_pixels_out() if pixels_out is None else pixels_out,
            pixel_scale_out = self.get_pixel_scale_out() if pixel_scale_out is None else pixel_scale_out)


class AngularFFTUtility(FixedSamplingUtility, UtilityUser):
    """
    Container of useful functions and constructors for testing the 
    AngluarFFT class.

    Attributes
    ----------
    utility : Utility 
        A utility for building `Wavefront` objects that interact 
        with the `AngularFFT`.
    """
    utility : Utility = AngularWavefrontUtility()


    def __init__(self : Utility) -> Utility:
        """
        Initialises a safe state for the Propagator attributes 
        stored as attributes in this Utility.
        """
        super().__init__()


    def construct(self : Utility, inverse : bool = None) -> Propagator:
        """
        Build a safe `PhysicalMFT` for testing purposes.

        Parameters
        ----------
        inverse : bool
            True if the inverse `Propagtor` is to be set.
        
        Returns
        -------
        propagator : Propagator
            The safe testing `Propagator`
        """
        return dLux.AngularFFT(
            inverse = self.is_inverse() if inverse is None else inverse)


# class GaussianPropagatorUtility(Utility, UtilityUser):
#     """
#     Container of useful functions and constructors for testing the 
#      `GaussianPropagator` class.

#     Attributes
#     ----------
#     utility : Utility 
#         A utility for building `Wavefront` objects that interact 
#         with the `GaussianPropagator`.
#     """
#     utility : Utility = GaussianWavefrontUtility()


#     def __init__(self : Utility) -> Utility:
#         """
#         Initialises a safe state for the Propagator attributes 
#         stored as attributes in this Utility.
#         """
#         pass         


#     def construct(self : Utility, distance : float) -> Propagator:
#         """
#         Build a safe `GaussianPropagator` for testing purposes.

#         Parameters
#         ----------
#         distance : float
#             The distance of the propagation in meters.        
        
#         Returns
#         -------
#         propagator : Propagator
#             The safe testing `Propagator`
#         """
#         return dLux.GaussianPropagator(distance)


##########################
### Spectrum Utilities ###
##########################
class SpectrumUtility(Utility):
    """
    Utility for the Spectrum class.
    """
    wavelengths : Array
    dLux.spectrums.Spectrum.__abstractmethods__ = ()
    
    
    def __init__(self : Utility) -> Utility:
        """
        Constrcutor for the Spectrum Utility.
        """
        self.wavelengths = np.linspace(500e-9, 600e-9, 10)
    
    
    def construct(self : Utility, wavelengths : Array = None) -> Spectrum:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        wavelengths = self.wavelengths if wavelengths is None else wavelengths
        return dLux.spectrums.Spectrum(wavelengths)
    
    
class SpectrumUtility(Utility):
    """
    Utility for the Spectrum class.
    """
    wavelengths : Array
    dLux.spectrums.Spectrum.__abstractmethods__ = ()
    
    
    def __init__(self : Utility) -> Utility:
        """
        Constrcutor for the Spectrum Utility.
        """
        self.wavelengths = np.linspace(500e-9, 600e-9, 10)
    
    
    def construct(self : Utility, wavelengths : Array = None) -> Spectrum:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        wavelengths = self.wavelengths if wavelengths is None else wavelengths
        return dLux.spectrums.Spectrum(wavelengths)
    
    
class ArraySpectrumUtility(SpectrumUtility):
    """
    Utility for the ArraySpectrum class.
    """
    weights : Array
    
    
    def __init__(self : Utility) -> Utility:
        """
        Constrcutor for the ArraySpectrum Utility.
        """
        super().__init__()
        self.weights = np.arange(10)
    
    
    def construct(self : Utility, wavelengths : Array = None,
                  weights : Array = None) -> Spectrum:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        wavelengths = self.wavelengths if wavelengths is None else wavelengths
        weights = self.weights if weights is None else weights
        return dLux.spectrums.ArraySpectrum(wavelengths, weights)
    
    
class PolynomialSpectrumUtility(SpectrumUtility):
    """
    Utility for the PolynomialSpectrum class.
    """
    coefficients : Array
    
    
    def __init__(self : Utility) -> Utility:
        """
        Constrcutor for the PolynomialSpectrum Utility.
        """
        super().__init__()
        self.coefficients = np.arange(3)
    
    
    def construct(self : Utility, wavelengths : Utility = None,
                  coefficients : Utility = None) -> Spectrum:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        wavelengths = self.wavelengths if wavelengths is None else wavelengths
        coefficients = self.coefficients if coefficients is None \
                                                            else coefficients
        return dLux.spectrums.PolynomialSpectrum(wavelengths, coefficients)
    
    
class CombinedSpectrumUtility(SpectrumUtility):
    """
    Utility for the ArraySpectrum class.
    """
    wavelengths : Array
    weights     : Array
    
    
    def __init__(self : Utility) -> Utility:
        """
        Constrcutor for the ArraySpectrum Utility.
        """
        super()
        self.wavelengths = np.tile(np.linspace(500e-9, 600e-9, 10), (2, 1))
        self.weights = np.tile(np.arange(10), (2, 1))
    
    
    def construct(self : Utility, wavelengths : Utility = None,
                  weights : Utility = None) -> Spectrum:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        wavelengths = self.wavelengths if wavelengths is None else wavelengths
        weights = self.weights if weights is None else weights
        return dLux.spectrums.CombinedSpectrum(wavelengths, weights)


########################
### Source Utilities ###
########################
class SourceUtility(Utility):
    """
    Utility for the Source class.
    """
    position : Array
    flux     : Array
    spectrum : dLux.spectrums.Spectrum
    name     : str
    dLux.sources.Source.__abstractmethods__ = ()
    
    
    def __init__(self : Utility) -> Utility:
        """
        Constructor for the Source Utility.
        """
        self.position = np.array([0., 0.])
        self.flux     = np.array(1.)
        self.spectrum = dLux.spectrums.ArraySpectrum(np.linspace(500e-9, \
                                                                 600e-9, 10))
        self.name = "Source"
    
    
    def construct(self     : Utility,
                  position : Array    = None,
                  flux     : Array    = None,
                  spectrum : Spectrum = None,
                  name     : str      = None) -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        position = self.position if position is None else position
        flux     = self.flux     if flux     is None else flux
        spectrum = self.spectrum if spectrum is None else spectrum
        name     = self.name     if name     is None else name
        return dLux.sources.Source(position, flux, spectrum, name=name)
    
    
class ResolvedSourceUtility(SourceUtility):
    """
    Utility for the ResolvedSource class.
    """
    dLux.sources.ResolvedSource.__abstractmethods__ = ()
    
    
    def __init__(self : Utility) -> Utility:
        """
        Constructor for the ResolvedSource Utility.
        """
        pass
    
    
    def construct(self     : Utility,
                  position : Array    = None,
                  flux     : Array    = None,
                  spectrum : Spectrum = None,
                  name     : str      = None) -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        position = self.position if position is None else position
        flux     = self.flux     if flux     is None else flux
        spectrum = self.spectrum if spectrum is None else spectrum
        name     = self.name     if name     is None else name
        return dLux.sources.Source(position, flux, spectrum, name=name)
    
    
class RelativeFluxSourceUtility(SourceUtility):
    """
    Utility for the RelativeFluxSource class.
    """
    flux_ratio : Array
    dLux.sources.RelativeFluxSource.__abstractmethods__ = ()
    
    
    def __init__(self : Utility) -> Utility:
        """
        Constructor for the RelativeFluxSource Utility.
        """
        super().__init__()
        self.flux_ratio = np.array(2.)
    
    
    def construct(self       : Utility,
                  position   : Array    = None,
                  flux       : Array    = None,
                  spectrum   : Spectrum = None,
                  flux_ratio : Array    = None,
                  name       : str      = None) -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        position   = self.position   if position   is None else position
        flux       = self.flux       if flux       is None else flux
        spectrum   = self.spectrum   if spectrum   is None else spectrum
        flux_ratio = self.flux_ratio if flux_ratio is None else flux_ratio
        name       = self.name       if name       is None else name
        return dLux.sources.RelativeFluxSource(flux_ratio, position=position,
                                               flux=flux, spectrum=spectrum,
                                               name=name)
    
    
class RelativePositionSourceUtility(SourceUtility):
    """
    Utility for the RelativePositionSource class.
    """
    separation : Array
    field_angle : Array
    dLux.sources.RelativePositionSource.__abstractmethods__ = ()
    
    
    def __init__(self : Utility) -> Utility:
        """
        Constructor for the RelativePositionSource Utility.
        """
        super().__init__()
        self.separation  = np.array(1.)
        self.field_angle = np.array(0.)
    
    
    def construct(self        : Utility,
                  position    : Array    = None,
                  flux        : Array    = None,
                  spectrum    : Spectrum = None,
                  separation  : Array    = None,
                  field_angle : Array    = None,
                  name        : str      = None) -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        position    = self.position    if position    is None else position
        flux        = self.flux        if flux        is None else flux
        spectrum    = self.spectrum    if spectrum    is None else spectrum
        separation  = self.separation  if separation  is None else separation
        field_angle = self.field_angle if field_angle is None else field_angle
        name        = self.name        if name        is None else name
        return dLux.sources.RelativePositionSource(separation, field_angle,
                                                   position=position, flux=flux,
                                                   spectrum=spectrum, name=name)
    
    
class PointSourceUtility(SourceUtility):
    """
    Utility for the PointSource class.
    """
    
    
    def __init__(self : Utility) -> Utility:
        """
        Constructor for the PointSource Utility.
        """
        super().__init__()
    
    
    def construct(self     : Utility,
                  position : Array    = None,
                  flux     : Array    = None,
                  spectrum : Spectrum = None,
                  name     : str      = None) -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        position = self.position if position is None else position
        flux     = self.flux     if flux     is None else flux
        spectrum = self.spectrum if spectrum is None else spectrum
        name     = self.name     if name     is None else name
        return dLux.sources.PointSource(position, flux, spectrum, name=name)
    
    
class ArrayDistributionUtility(SourceUtility):
    """
    Utility for the ArrayDistribution class.
    """
    distribution : Array
    
    
    def __init__(self : Utility) -> Utility:
        """
        Constructor for the ArrayDistribution Utility.
        """
        super().__init__()
        distribution = np.ones((5, 5))
        self.distribution = distribution/distribution.sum()
    
    
    def construct(self         : Utility,
                  position     : Array    = None,
                  flux         : Array    = None,
                  spectrum     : Spectrum = None,
                  distribution : Array    = None,
                  name         : str      = None) -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        position     = self.position     if position     is None else position
        flux         = self.flux         if flux         is None else flux
        spectrum     = self.spectrum     if spectrum     is None else spectrum
        name         = self.name         if name         is None else name
        distribution = self.distribution if distribution is None \
                                                            else distribution
        return dLux.sources.ArrayDistribution(position, flux, spectrum, \
                                              distribution, name=name)
    
    
class BinarySourceUtility(RelativePositionSourceUtility, \
                          RelativeFluxSourceUtility):
    """
    Utility for the BinarySource class.
    """
    
    
    def __init__(self : Utility) -> Utility:
        """
        Constructor for the BinarySource Utility.
        """
        super().__init__()
        wavelengths = np.tile(np.linspace(500e-9, 600e-9, 10), (2, 1))
        weights     = np.tile(np.arange(10), (2, 1))
        self.spectrum = dLux.spectrums.CombinedSpectrum(wavelengths, weights)
    
    
    def construct(self        : Utility,
                  position    : Array    = None,
                  flux        : Array    = None,
                  spectrum    : Spectrum = None,
                  separation  : Array    = None,
                  field_angle : Array    = None,
                  flux_ratio  : Array    = None,
                  name        : str      = None) -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        position    = self.position    if position    is None else position
        flux        = self.flux        if flux        is None else flux
        spectrum    = self.spectrum    if spectrum    is None else spectrum
        separation  = self.separation  if separation  is None else separation
        field_angle = self.field_angle if field_angle is None else field_angle
        flux_ratio  = self.flux_ratio  if flux_ratio  is None else flux_ratio
        name        = self.name        if name        is None else name
        return dLux.sources.BinarySource(position, flux, separation, \
                                  field_angle, flux_ratio, spectrum, name=name)
    
    
class PointExtendedSourceUtility(RelativeFluxSourceUtility, \
                                 ArrayDistributionUtility):
    """
    Utility for the PointExtendedSource class.
    """
    
    
    def __init__(self : Utility) -> Utility:
        """
        Constructor for the PointExtendedSource Utility.
        """
        super().__init__()
    
    
    def construct(self         : Utility,
                  position     : Array    = None,
                  flux         : Array    = None,
                  spectrum     : Spectrum = None,
                  flux_ratio   : Array    = None,
                  distribution : Array    = None,
                  name         : str      = None) -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        position     = self.position     if position     is None else position
        flux         = self.flux         if flux         is None else flux
        spectrum     = self.spectrum     if spectrum     is None else spectrum
        flux_ratio   = self.flux_ratio   if flux_ratio   is None else flux_ratio
        name         = self.name         if name         is None else name
        distribution = self.distribution if distribution is None \
                                                            else distribution
        return dLux.sources.PointExtendedSource(position, flux, spectrum, \
                                         distribution, flux_ratio, name=name)
    
    
class PointAndExtendedSourceUtility(RelativeFluxSourceUtility, \
                                    ArrayDistributionUtility):
    """
    Utility for the PointAndExtendedSource class.
    """
    
    
    def __init__(self : Utility) -> Utility:
        """
        Constructor for the PointAndExtendedSource Utility.
        """
        super().__init__()
        wavelengths = np.tile(np.linspace(500e-9, 600e-9, 10), (2, 1))
        weights = np.tile(np.arange(10), (2, 1))
        self.spectrum = dLux.spectrums.CombinedSpectrum(wavelengths, weights)
    
    
    def construct(self         : Utility,
                  position     : Array    = None,
                  flux         : Array    = None,
                  spectrum     : Spectrum = None,
                  flux_ratio   : Array    = None,
                  distribution : Array    = None,
                  name         : str      = None) -> Source:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        position     = self.position     if position     is None else position
        flux         = self.flux         if flux         is None else flux
        spectrum     = self.spectrum     if spectrum     is None else spectrum
        flux_ratio   = self.flux_ratio   if flux_ratio   is None else flux_ratio
        name         = self.name         if name         is None else name
        distribution = self.distribution if distribution is None \
                                                            else distribution
        return dLux.sources.PointAndExtendedSource(position, flux, spectrum, \
                                         distribution, flux_ratio, name=name)

######################
### Base Utilities ###
######################
class BaseUtility(Utility):
    """
    Utility for the Base class.
    """
    param1 : float
    param2 : float
    
    
    class A(dLux.base.Base):
        """
        Test subclass to test the Base methods
        """
        param : float
        b     : B
        
        
        def __init__(self, param, b):
            """
            Constructor for the Base testing class
            """
            self.param = param
            self.b = b
        
        
        def model(self):
            """
            Sample modelling function
            """
            return self.param**2 + self.b.param**2
    
    
    class B(dLux.base.Base):
        """
        Test subclass to test the Base methods
        """
        param : float
        
        
        def __init__(self, param):
            """
            Constructor for the Base testing class
            """
            self.param = param
    
    
    def __init__(self : Utility):
        """
        Constructor for the Optics Utility.
        """ 
        self.param1 = 1.
        self.param2 = 1.
    
    
    def construct(self : Utility, 
                  param1 : float = None, 
                  param2 : float = None):
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        param1 = self.param1 if param1 is None else param1
        param2 = self.param2 if param2 is None else param2
        return self.A(param1, self.B(param2))


class OpticsUtility(Utility):
    """
    Utility for the Optics class.
    """
    layers : list
    
    
    def __init__(self : Utility) -> Utility:
        """
        Constructor for the Optics Utility.
        """    
        self.layers = [
            dLux.layers.CreateWavefront(16, 1),
            dLux.layers.CompoundAperture([0.5]),
            dLux.layers.NormaliseWavefront(),
            dLux.propagators.PhysicalMFT(16, 1., 1e-6)
        ]
    
    
    def construct(self : Utility, layers : list = None) -> Optics:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        layers = self.layers if layers is None else layers
        return dLux.base.Optics(layers)


class DetectorUtility(Utility):
    """
    Utility for the Detector class.
    """
    layers : list
    
    
    def __init__(self : Utility) -> Utility:
        """
        Constructor for the Detector Utility.
        """    
        self.layers = [
            dLux.detectors.AddConstant(1.)
        ]
    
    
    def construct(self : Utility, layers : list = None) -> Detector:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        layers = self.layers if layers is None else layers
        return dLux.base.Detector(layers)


class SceneUtility(Utility):
    """
    Utility for the Scene class.
    """
    sources : list
    
    
    def __init__(self : Utility) -> Utility:
        """
        Constructor for the Scene Utility.
        """
        self.sources = [
            PointSourceUtility().construct()
        ]
    
    
    def construct(self : Utility, sources : list = None) -> Scene:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        sources = self.sources if sources is None else sources
        return dLux.base.Scene(sources)


class FilterUtility(Utility):
    """
    Utility for the Filter class.
    """
    wavelengths : Array
    throughput  : Array
    order       : int
    
    
    def __init__(self : Utility) -> Utility:
        """
        Constructor for the Filter Utility.
        """
        self.wavelengths = np.linspace(1e-6, 10e-6, 10)
        self.throughput  = np.linspace(0, 1, len(self.wavelengths))
        self.order       = int(1)
    
    
    def construct(self        : Utility, 
                  wavelengths : Array = None, 
                  throughput  : Array = None,
                  order       : int   = 1,
                  filter_name : str   = None) -> Filter:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        wavelengths = self.wavelengths if wavelengths is None else wavelengths
        throughput  = self.throughput  if throughput  is None else throughput
        order       = self.order       if order       is None else order
        
        if filter_name is None:
            return dLux.base.Filter(wavelengths, throughput, order=order)
        else:
            return dLux.base.Filter(wavelengths, throughput, order=order, 
                                    filter_name=filter_name)


class InstrumentUtility(Utility):
    """
    Utility for the Instrument class.
    """
    optics   : Optics
    scene    : Scene
    detector : Detector
    filter   : Filter
    
    
    def __init__(self : Utility) -> Utility:
        """
        Constructor for the Instrument Utility.
        """    
        self.optics   = OpticsUtility().construct()
        self.scene    = SceneUtility().construct()
        self.detector = DetectorUtility().construct()
        self.filter   = FilterUtility().construct()
    
    
    def construct(self            : Utility,
                  optics          : Optics   = None,
                  scene           : Scene    = None,
                  detector        : Detector = None,
                  filter          : Filter   = None,
                  optical_layers  : list     = None,
                  sources         : list     = None,
                  detector_layers : list     = None,
                  input_layers    : bool     = False,
                  input_both      : bool     = False) -> Instrument:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        optics   = self.optics   if optics   is None else optics
        scene    = self.scene    if scene    is None else scene
        detector = self.detector if detector is None else detector
        filter   = self.filter   if filter   is None else filter
        
        if input_both:
            return dLux.base.Instrument(optics=optics,
                                        scene=scene,
                                        detector=detector,
                                        filter=filter,
                                        optical_layers=optical_layers,
                                        sources=sources,
                                        detector_layers=detector_layers)
        elif not input_layers:
            return dLux.base.Instrument(optics=optics,
                                        scene=scene,
                                        detector=detector,
                                        filter=filter)
        else:
            return dLux.base.Instrument(filter=filter,
                                        optical_layers=optical_layers,
                                        sources=sources,
                                        detector_layers=detector_layers)