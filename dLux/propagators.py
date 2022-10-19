"""
This file contains the propagator classes. These sit low on the 
cohesion spectrum and high on the coupling spectrum as they are 
typically specific to a Wavefront class and implement something 
that is a behaviour of the Wavefront; that is to __propagate__.
They were separated so that propagation distances could be made
class attributes and hence differentiated using Patrick Kidgers
`equinox` package.


To avoid code duplication at the expense of coupling between 
propagtors, a complex system of abstract classes was created
by Jordan Dennis. The abstract Propagator class is primarily 
for self documentation of the code and implements only a few 
simple helper methods. Concrete methods are added within the 
next level of classes; which are VariableSamplingPropagator 
and FixedSamplingPropagtor. 


The primary difference at this level is the style of Fourier 
transform algorithm that is used. The VariableSamplingPropagator
uses the Soummer et. al. 2007 algorithm which allows the scale
of the plane of propagation to be varied. On the other hand the,
FixedSamplingPropagator uses the numpy.fft module, which has the
fixed sample of one diffraction fringe in the output plane per
pixel in the input plane. 


The units are handled at the next level of classes which also
implement the exact algorithm. For both angular units and 
physical units there is a FixedSampling and a VariableSampling
option for far-field diffraction as well as a VairableSampling
near-field algorithm. There is also a special intermediate 
plane Fresnel algorithm based on the GaussianWavefront class.


There is a total of four non-abstract propagators defined in 
this file that are listed below:
 - PhysicalVariableFraunhofer
 - PhysicalFixedFraunhofer
 - AngularVariableFraunhofer
 - AngularFixedFraunhofer

Propagator
    Concrete Methods
    - _normalising_factor
    - get_pixel_positions

    Abstract Methods
    - __init__
    - __call__
    - _fourier_transform()
    - _inverse_fourier_transform()

    VariableSampling (MFT)
        PhysicalMFT
        AngularMFT

    FixedSampling (FFT)
        PhysicalFFT
        AngularFFT
"""
from __future__ import annotations
import jax.numpy as np
import jax
import equinox as eqx
import typing
import dLux
import abc


__all__ = ["PhysicalMFT", "AngularMFT", "PhysicalFFT", "AngularFFT",
           "PhysicalFresnel"]
Array =  typing.NewType("Array",  np.ndarray)
Wavefront =  typing.NewType("Wavefront",  dLux.wavefronts.Wavefront)


class Propagator(dLux.layers.OpticalLayer, abc.ABC):
    """
    An abstract class indicating a spatial transfromation of the
    `Wavefront`. This is a separate class because it allows
    us to take gradients with respect to the fields of the
    propagator and hence optimise distances ect.

    Attributes
    ----------
    inverse : bool
        True if the inverse algorithm is to be used else false.
    tilt : bool
        True if the offset of the `Wavefront` is to be considered
        otherwise false.
    """
    inverse : bool
    
    
    def __init__(self : Propagator, inverse : bool = False) -> Propagator:
        """
        Parameters
        ----------
        inverse : bool = False
            True if the inverse algorithm is to be used else False.
        """
        self.inverse = bool(inverse)
        

    def is_inverse(self : Propagator) -> bool:
        """
        Returns 
        -------
        inverse : bool
            Whether or not the inverse algorithm is to be used.
        """
        return self.inverse
    
    
    @abc.abstractmethod
    def _fourier_transform(self : Propagator, wavefront : Wavefront) -> Array:
        """
        The implementation of the Fourier transform that is to 
        be used for the optical propagation. This is for propagation 
        from the input plane to the plane of propagation. This 
        method is abstract and defines a behaviour that must be 
        implemented by the subclasses.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront that is to be propagted.

        Returns 
        -------
        field : Array
            The complex electric field amplitude following the 
            propagation. 
        """
        return
    
    
    @abc.abstractmethod
    def _inverse_fourier_transform(self : Propagator,
            wavefront : Wavefront) -> Array:
        """
        The implementation of the inverse Fourier transform that is 
        to be used for the optical propagation. The inverse represents
        propagtion from the plane of propagation to the input plane.
        This method is abstract and defines a behaviour that must 
        be implemented by the subclasses.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront that is getting propagated. 

        Returns 
        -------
        field : Array
            The complex electric field amplitude following the 
            propagation. 
        """  
        return


    def _propagate(self : Propagator, wavefront : Wavefront) -> Array:
        """
        Performs the propagation as a directional wrapper to the 
        fourier methods of the class.

        Parameters
        ----------
        wavefront : Wavefront 
            The `Wavefront to propagate.
        
        Returns
        -------
        field : Array   
            The electric field of the wavefronts.
        """
        field = jax.lax.cond(self.is_inverse(),
            lambda wavefront : self._inverse_fourier_transform(wavefront),
            lambda wavefront : self._fourier_transform(wavefront),
            wavefront)

        field *= self._normalising_factor(wavefront)
        return field
    
    
    @abc.abstractmethod
    def _normalising_factor(self : Propagator, wavefront : Wavefront) -> float:
        """
        Apply a normalisation to the wavefront to ensure that it 
        conserves flux through the propagation. This method is 
        abstract and must be implemented by the subclasses. 

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront that has been propagated.

        Returns
        -------
        wavefront : Wavefront
            The `Wavefront` where the electric field power has 
            been conserved.
        """
        return


class VariableSamplingPropagator(Propagator, abc.ABC):
    """
    A propagator that users the Soummer et. al. 2007 MFT algorithm 
    to implement variable sampling in the plane of propagation 
    rather than the enforced; one pixel in the input plane = one 
    diffraction fringe in the output plane, fast fourier transform 
    algorithm.

    Attributes
    ----------
    pixel_scale_out : float, meters/pixel or radians/pixel
        The pixel scale in the plane of propagation measured in 
        meters (radians) per pixel.
    pixels_out : int
        The number of pixels in the plane of propagation. 
    """
    pixel_scale_out : float
    pixels_out : int
    tilt : bool 
    pixel_tilt : bool


    def __init__(self : Propagator, pixel_scale_out : float, 
            pixels_out : int, inverse : bool = False,
            tilt : bool = False, pixels : bool = False) -> Propagator:
        """
        Constructor for VariableSampling propagators.

        Parameters
        ----------
        pixels_out : int
            The number of pixels in the output plane (side length).
        pixel_scale_out : float, meters/pixel or radians/pixel
            The pixel scale in the output plane in units of meters
            (radians) per pixel.
        inverse : bool
            True if the inverse algorithm is to be used else False.
        tilt : bool 
            True if the tilt of the `Wavefront` is to be considered 
            else False. 
        pixel_tilt : bool
            True if the tilt of the wavefront is to be considered in 
            units of pixel, else the shift is considered in units of
            radians. Redundant is tilt is False
        """
        super().__init__(inverse)
        self.pixel_scale_out = np.array(pixel_scale_out).astype(float)
        self.pixels_out = int(pixels_out)
        self.tilt = bool(tilt)
        self.pixel_tilt = bool(pixels)


    def is_tilted(self : Propagator) -> bool:
        """
        Returns
        -------
        tilt : bool
            Whether or not the tilt of the `Wavefront` is to be 
            considered. 
        """
        return self.tilt
    
    def is_pixel_tilted(self : Propagator) -> bool:
        """
        Returns
        -------
        pixel_tilt : bool
            Whether or not the tilt of the `Wavefront` is to be 
            considered in units of pixels or radians.
        """
        return self.tilt


    def get_pixel_scale_out(self : Propagator) -> float:
        """
        Accessor for the pixel scale in the output plane. 

        Returns
        -------
        pixel_scale_out : float, meters/pixel or radians/pixel
            The pixel scale in the output plane.
        """
        return self.pixel_scale_out


    def get_pixels_out(self : Propagator) -> int:
        """
        Accessor for the `pixels_out` parameter.

        Returns
        -------
        pixels_out : int
            The number of pixels in the plane of propagation.
        """
        return self.pixels_out
    
    
    def get_pixel_offsets(self : Propagator, 
            wavefront : Wavefront) -> Array:
        """
        The offset(s) in units of pixels.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate.

        Returns
        -------
        pixel_offset : Array, pixels
            The offset from the x and y plane.
        """
        return jax.lax.cond(self.is_tilted(),     
            lambda wavefront : self.get_offset_value(wavefront),
            lambda _ : np.array([0., 0.]).astype(float),
            wavefront)
    

    def get_offset_value(self : Propagator,
                        wavefront : Wavefront) -> Array:
        """
        Returns the offset value either as-is or scaled by
        the physical focal length and pixel scale, depending
        on the boolean value of pixel_tilt. Used to handle cases
        where the offset value is given in either units of pixels
        or radians
        
        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate.

        Returns
        -------
        offset : Array
            The offset from the x and y plane in units of pixels
        """
        return jax.lax.cond(self.is_pixel_tilted(),
                        lambda wavefront : wavefront.get_offset(),
                        lambda wavefront : wavefront.get_offset() * \
                        self.get_focal_length() / self.get_pixel_scale_out(),
                        wavefront)


    def _generate_twiddle_factors(self : Propagator,            
            pixel_offset : float, pixel_scales : tuple, 
            pixels : tuple, sign : int) -> Array:
        """
        The twiddle factors for the fourier transforms.

        Parameters
        ----------
        pixel_offset : float, pixels
            The offset in units of pixels.
        pixel_scales : tuple
            The input and output pixel scales. 
        pixels : tuple
            The number of input and output pixels.
        sign : int
            1. for Fourier transform and -1. for inverse Fourier 
            transform. 

        Returns
        -------
        twiddle_factors : Array
            The twiddle factors.
        """
        input_scale, output_scale = pixel_scales
        pixels_input, pixels_output = pixels

        input_coordinates = dLux.utils.coordinates.get_coordinates_vector(
                                                pixels_input, input_scale, 
                                                pixel_offset/input_scale)

        output_coordinates = dLux.utils.coordinates.get_coordinates_vector(
                                                pixels_output, output_scale, 
                                                pixel_offset/output_scale)

        input_to_output = np.outer(input_coordinates, output_coordinates)

        return np.exp(-2. * sign * np.pi * 1j * input_to_output)


    def _matrix_fourier_transform(self : Propagator, 
            wavefront : Wavefront, sign : int) -> Array:
        """
        Take the paraxial fourier transform of the wavefront in the 
        complex representation.

        Parameters
        ----------
        wavefront : Wavefront
            The `Wavefront` object that we want to Fourier transform.
        sign : int
            1. if forward Fourier transform else -1.

        Returns
        -------
        field : Array[Complex]
            The complex un-normalised electric field after the 
            propagation.
        """
        field = wavefront.get_complex_form()
        nfields = field.shape[0]
 
        input_scale = 1.0 / wavefront.number_of_pixels()
        output_scale = self.number_of_fringes(wavefront) / \
            self.get_pixels_out()
        pixels_input = wavefront.number_of_pixels()
        pixels_output = self.get_pixels_out()
        
        x_offset, y_offset = self.get_pixel_offsets(wavefront)

        x_twiddle_factors = np.tile(self._generate_twiddle_factors(
            x_offset, (input_scale, output_scale), 
            (pixels_input, pixels_output), sign), (nfields, 1, 1))

        y_twiddle_factors = np.tile(self._generate_twiddle_factors(
            y_offset, (input_scale, output_scale), 
            (pixels_input, pixels_output), sign).T, (nfields, 1, 1))
        
        return (y_twiddle_factors @ field) \
            @ x_twiddle_factors


    def _fourier_transform(self : Propagator, wavefront : Wavefront) -> Array:
        """
        Compute the fourier transform of the electric field of the 
        input wavefront. This represents propagation from the input
        plane to the plane of propagation.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront that is getting propagated. 

        Returns
        -------
        field : Array[Complex]
            The complex electric field after propagation. Not yet normalised.
        """
        return self._matrix_fourier_transform(wavefront, sign = 1)


    def _inverse_fourier_transform(self : Propagator, 
            wavefront : Wavefront) -> Array:
        """
        The inverse fourier transform of the wavefront. This represents
        propagation from the focal plane to the pupil plane. 

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront that is to be propagated. 

        Returns
        -------
        field : Array[Complex]
            The complex electric field.
        """
        return self._matrix_fourier_transform(wavefront, sign = -1)

    
    def _normalising_factor(self : Propagator, 
            wavefront : Wavefront) -> Wavefront:
        """
        Normalise the `Wavefront` according to the `poppy` convention. 

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront that is getting propagated. 

        Returns
        -------
        normalising_factor : Wavefront
            The normalised `Wavefront`.
        """
        return np.exp(np.log(self.number_of_fringes(wavefront)) - \
            (np.log(wavefront.number_of_pixels()) + \
                np.log(self.get_pixels_out())))


    def __call__(self : Propagator, parameters : dict) -> dict:
        """
        Propagate a `Wavefront` to the focal_plane of the lens or 
        mirror.

        Parameters
        ----------
        parameters : dict
            A dictionary that must contain a "Wavefront" key.

        Returns 
        -------
        parameters : dict 
            A dictionary with the updated "Wavefront" key; value
        """
        wavefront = parameters["Wavefront"]

        new_wavefront = self._propagate(wavefront)

        new_amplitude = np.abs(new_wavefront)
        new_phase = np.angle(new_wavefront)
        new_plane_type = jax.lax.cond(self.is_inverse(),
                                     lambda : dLux.PlaneType.Pupil,
                                     lambda : dLux.PlaneType.Focal)
        
        new_wavefront = wavefront\
            .update_phasor(new_amplitude, new_phase)\
            .set_plane_type(new_plane_type)\
            .set_pixel_scale(self.get_pixel_scale_out())

        parameters["Wavefront"] = new_wavefront
        return parameters           
    
    
    @abc.abstractmethod
    def number_of_fringes(self : Propagator, wavefront : Wavefront) -> float:
        """
        The number of diffraction fringes in the output plane. 
        This is an abstract method and needs to be implemented 
        by the subclasses.

        Parameters
        ----------
        wavefront : Wavefront
            The `Wavefront` that is getting propagated to the output
            plane.

        Returns
        -------
        fringes : float
            The number of diffraction fringes in the output plane. 
        """
        return


class FixedSamplingPropagator(Propagator, abc.ABC):
    """
    A propagator that samples the electric field in the output plane
    at the rate of one fringe per pixel where a fringe is a wavelength 
    divided by the aperture diameter. 


    These propagators are implemented using the numpy.fft sub-sub-package
    and cannot be modified elsewise. 
    """
    
    
    def _fourier_transform(self : Propagator, 
            wavefront : Wavefront) -> Array:
        """
        Perfrom a fast fourier transform on the wavefront.

        Parameters
        ----------
        wavefront : Wavefront 
            The wavefront to propagate.

        Returns
        -------
        field : Array[Complex]
            The complex electric field units following propagation.
        """
        return np.fft.fftshift(np.fft.ifft2(wavefront.get_complex_form()))


    def _inverse_fourier_transform(self : Propagator,
            wavefront : Wavefront) -> Array:
        """
        Perfrom an inverse fourier transform of a wavefront.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to be propagated.

        Returns
        -------
        field : Array[Complex]
            The complex electric field units following propagation.
        """
        return np.fft.fft2(np.fft.ifftshift(wavefront.get_complex_form()))


    def _normalising_factor(self : Propagator, 
            wavefront : Wavefront) -> float:
        """
        The normalising factor associated with the propagtion.
        This is a unitless quanitity.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront that is getting propagated.

        Returns
        -------
        normalising_factor : float
            The normalising factor that is appropriate to the 
            method of propagation.
        """
        return self.is_inverse() / wavefront.number_of_pixels() + \
            (1 - self.is_inverse()) * wavefront.number_of_pixels()
    
    
    @abc.abstractmethod
    def get_pixel_scale_out(self : Propagator, 
            wavefront : Wavefront) -> float:
        """
        The pixel scale in the output plane. Calculated based on the 
        wavefront and the units of the wavefront. 

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront that is getting propagated.

        Returns 
        -------
        pixel_scale : float, meters/pixel or radians/pixel
            The pixel scale in the output plane in units of meters
            (radians) per pixel.
        """
        return


    def __call__(self : Propagator, parameters : dict) -> dict:
        """
        Propagate a `Wavefront` to the focal_plane of the lens or 
        mirror.

        Parameters
        ----------
        parameters : dict
            A dictionary that must contain a "Wavefront" key.

        Returns 
        -------
        parameters : dict 
            A dictionary with the updated "Wavefront" key; value
        """
        wavefront = parameters["Wavefront"]

        new_wavefront = self._propagate(wavefront)

        new_amplitude = np.abs(new_wavefront)
        new_phase = np.angle(new_wavefront)
        new_plane_type = jax.lax.cond(self.is_inverse(),
                                     lambda : dLux.PlaneType.Pupil,
                                     lambda : dLux.PlaneType.Focal)
        
        new_wavefront = wavefront\
            .update_phasor(new_amplitude, new_phase)\
            .set_plane_type(new_plane_type)\
            .set_pixel_scale(self.get_pixel_scale_out(wavefront))

        parameters["Wavefront"] = new_wavefront
        return parameters


### VariableSamplingPropagators sub-classes ###

class PhysicalMFT(VariableSamplingPropagator):
    """
    Fraunhofer propagation based on the a matrix fourier transfrom 
    with adjustable scaling. 

    Attributes 
    ----------
    focal_length : float, meters
        The focal length of the propagation distance.    
    """
    focal_length : float


    def __init__(self : Propagator, pixels_out : float, 
            focal_length : int, pixel_scale_out : float, 
            inverse : bool = False, tilt : bool = False) -> Propagator:
        """
        Parameters
        ----------
        focal_length : float, meters
            The focal length of the mirror or lens that the Wavefront
            is propagating away from.
        pixels_out : int
            The number of pixels in the output image. 
        pixel_scale_out : float, meters/pixel
            The pixel scale in the output plane. 
        inverse : bool = False
            Whether or not the propagation is input plane to output 
            plane or output plane to input plane. The inverse algorithm
            is used if True is provided.
        tilt : bool = False
            Whether or not to use the propagation is to use the tilt 
            of the wavefront. The tilt is used if True and not if 
            False.
        """
        super().__init__(inverse = inverse, tilt = tilt,
            pixel_scale_out = pixel_scale_out, 
            pixels_out = pixels_out) 
        self.focal_length = np.array(focal_length).astype(float)


    def number_of_fringes(self : Propagator, 
            wavefront : Wavefront) -> float:
        """
        Determines the number of diffraction fringes in the plane of 
        propagation.

        Parameters 
        ----------
        wavefront : Wavefront
            The `Wavefront` that is getting propagated.

        Returns 
        -------
        fringes : float
            The floating point number of diffraction fringes in the 
            plane of propagation.
        """
        size_in = wavefront.get_pixel_scale() * \
            wavefront.number_of_pixels()        
        size_out = self.get_pixel_scale_out() * \
            self.get_pixels_out()
        
        return size_in * size_out / self.get_focal_length() /\
            wavefront.get_wavelength()


    def get_focal_length(self : Propagator) -> float:
        """
        Accessor for the focal_length.

        Returns 
        -------
        : float, meters
            The focal length. 
        """
        return self.focal_length


class AngularMFT(VariableSamplingPropagator):
    """
    Propagation of an AngularWavefront by a paraxial matrix fourier
    transform.
    """
    def __init__(self : Propagator, pixel_scale_out : float, 
            pixels_out : int, inverse : bool = False,
            tilt : bool = False) -> Propagator:
        """
        Parameters
        ----------
        pixel_scale_out : float, radians/pixel or meters/pixel
            The scale of the pixels in the output plane in radians 
            per pixel. Pixel scales have units of meters/pixel in pupil
            planes and units of radians/pixel in focal planes.
        pixels_out : int
            The number of pixels in the output plane.
        inverse : bool = False
            True if the inverse transformation is to be applied else 
            False.
        tilt : bool = False
            True if the tilt of the wavefront is to be considered else
            False.
        """
        super().__init__(pixel_scale_out, pixels_out, inverse, tilt)


    def number_of_fringes(self : Propagator, 
                          wavefront : Wavefront) -> float:
        """
        Determines the number of diffraction fringes in the plane of 
        propagation.

        Parameters 
        ----------
        wavefront : Wavefront
            The `Wavefront` that is getting propagated.

        Returns 
        -------
        : float
            The floating point number of diffraction fringes in the 
            plane of propagation.
        """
        diameter = wavefront.number_of_pixels() * wavefront.get_pixel_scale()
        fringe_size = wavefront.get_wavelength() / diameter
        detector_size = self.pixels_out * self.pixel_scale_out 
        num_fringe = detector_size / fringe_size
        return num_fringe


    def get_pixel_offsets(self : Propagator, 
            wavefront : Wavefront) -> Array:
        """
        The offset(s) in units of pixels.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate.

        Returns
        -------
        : Array
            The offset from the x and y plane in units of pixels.
        """
        return jax.lax.cond(self.is_tilted(),
            lambda wavefront : wavefront.get_offset() /\
                 self.get_pixel_scale_out(),
            lambda _ : np.array([0., 0.]).astype(float),
            wavefront)


### FixedSamplingPropagators sub-classes ###

class PhysicalFFT(FixedSamplingPropagator):
    """
    Perfrom an FFT propagation on a `PhysicalWavefront`. This is the 
    same as the MFT however it samples 1 diffraction fringe per pixel.

    Attributes
    ----------
    focal_length : float, meters 
        The focal length of the lens or mirror.
    """
    focal_length : float


    def __init__(self : Propagator, focal_length : float, 
            inverse : bool = False, tilt : bool = False) -> Propagator:
        """
        Parameters
        ----------
        focal_length : float, meters
            The focal length of the lens or mirror that is getting 
            propagated away from.
        inverse : bool = False
            Propagation direction. True for forwards and False for 
            backwards. 
        tilt : bool
            Whether or not to use the tilt of the `Wavefront`. True
            if the tilt is to be accounted for. 

        Returns
        -------
        propagator : Propagator
            A `Propagator` object representing the optical path from 
            the mirror to the focal plane. 
        """
        super().__init__(inverse)
        self.focal_length = focal_length

            
    def get_focal_length(self : Propagator) -> float:
        """
        Accessor for the focal length.

        Returns
        -------
        focal_length : float, meters
            The focal length of the lens or mirror.
        """
        return self.focal_length


    def get_pixel_scale_out(self : Propagator, 
            wavefront : Wavefront) -> float:
        """
        The pixel scale in the focal plane

        Parameters
        ----------
        wavefront : Wavefront
            The `Wavefront` that is propagting.

        Returns
        -------
        pixel_scale_out : float, meters/pixel
            The pixel scale.
        """
        return wavefront.get_wavelength() * \
            self.get_focal_length() / (wavefront.get_pixel_scale() \
                * wavefront.number_of_pixels())    
    

class AngularFFT(FixedSamplingPropagator):
    """
    Propagation of an Angular wavefront by a non-paraxial fast Fourier
    transform. 
    """
    def __init__(self : Propagator, inverse : bool = False) -> Propagator:
        """
        Parameters
        ----------
        inverse : bool = False
            True if the inverse algorithm is to be used else False
        """
        super().__init__(inverse)


    def get_pixel_scale_out(self : Propagator, wavefront : Wavefront) -> float:
        """
        Calculate the pixel scale in the output plane in units of 
        radians per pixel.

        Parameters
        ----------
        wavefront : Wavefront 
            The wavefront that is getting propagated.

        Overrides
        ---------
        FixedSamplingPropagator::get_pixel_scale_out

        Returns
        -------
        pixel_scale : float
            The pixel scale in the ouptut plane in units of radians 
            per pixel. 
        """
        return wavefront.get_wavelength() / wavefront.get_diameter()


class PhysicalFresnel(VariableSamplingPropagator):
    """
    far-field diffraction based on the Frensel approximation.
    This implementation approximately conserves flux because the 
    normalisation is based on focal plane MFT normalisation.

    Attributes
    ----------
    focal_length : float, meters
        The focal length of the lens or mirror in meters. This 
        is a differentiable parameter.
    focal_shift : float, meters
        The displacement of the plane of propagation from the 
        the focal_length in meters. This is a differentiable
        parameter. The focal shift is positive if the plane of 
        propagation is beyond the focal length and negative 
        if the plane of propagation is inside the focal length.
    """
    focal_length : float
    focal_shift : float


    def __init__(self : Propagator, pixels_out : float,
            focal_length : float, focal_shift : float, 
            pixel_scale_out : float, inverse : bool = False,
            tilt : bool = False) -> Propagator:
        """
        Parameters
        ----------
        focal_length : float, meters
            The focal length of the mirror or the lens.
        focal_shift : float, meters
            The distance away from the focal plane to be propagated
            to.
        pixels_out : int
            The number if pixels in the plane of propagation.
        pixel_scale_out : float, meters/pixel
            The scale of a pixel in the plane of propagation in 
            units of meters per pixel. 
        inverse : bool
            True if propagating from the plane of propagation and
            False if propagating to the plane of propagation. 
        tilt : bool 
            True if the tilt of the `Wavefront` is to be considered.
            False if the tilt is to be discarded.
        """
        self.focal_shift = np.asarray(focal_shift).astype(float)
        self.focal_length = np.asarray(focal_length).astype(float)
        super().__init__(inverse = inverse, tilt = tilt,
            pixel_scale_out = pixel_scale_out, pixels_out = pixels_out)       


    def get_focal_length(self : Propagator) -> float:
        """
        Returns
        -------
        focal_length : float, meters
            The focal length of the mirror or lens.
        """
        return self.focal_length


    def get_focal_shift(self : Propagator) -> float:
        """
        Returns 
        -------
        shift : float, meters
            The shift away from focus of the detector.
        """
        return self.focal_shift


    def number_of_fringes(self : Propagator, 
            wavefront : Wavefront) -> float:
        """
        The number of diffraction fringes in the plane of propagation.

        Overrides
        ---------
        VariableSamplingPropagator : number_of_fringes()
            Adds somplexity to deal with the near field. 

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront that is getting propagated. 

        Returns
        -------
        fringes : float
            The number of diffraction fringes visible in the plane of 
            propagation.
        """
        propagation_distance = self.get_focal_length() + self.get_focal_shift()
        focal_ratio = self.get_focal_length() / propagation_distance

        size_in = wavefront.get_pixel_scale() * \
            wavefront.number_of_pixels()        
        size_out = self.get_pixel_scale_out() * \
            self.get_pixels_out()
        
        number_of_fringes = size_in * size_out / \
            self.get_focal_length() / wavefront.get_wavelength() * \
            focal_ratio

        return number_of_fringes
               

    # TODO: Room for optimisation by pasing the radial parameters
    # instead.
    def quadratic_phase(self : Propagator, x_coordinates : Array, 
            y_coordinates : Array, wavelength : float, 
            distance : float) -> Array:
        """
        A convinience function for calculating quadratic phase factors
        
        Parameters
        ----------
        x_coordinates : Array 
            The x coordinates of the pixels in meters. This will be 
            different in the plane of propagation and the initial 
            plane.
        y_coordinates : Array
            The y coordinates of the pixels in meters. This will be 
            different in the plane of propagation and the initial 
            plane.
        wavelength : float, meters
            The wavelength of the wavefront.
        distance : float, meters
            The distance that is to be propagated in meters. 

        Returns
        -------
        quadratic_phase : Array
            A set of phase factors that are useful in optical 
            calculations.
        """
        wavenumber = 2 * np.pi / wavelength
        radial_coordinates = np.hypot(x_coordinates, y_coordinates)
        return np.exp(0.5j * wavenumber * radial_coordinates ** 2 \
            / distance)
        

    def _propagate(self : Propagator, wavefront : Wavefront) -> Array:
        """
        Propagte the wavefront to the point specified by the pair 
        of parameters self._focal_length and self._focal_shift.
        
        Parameters
        ----------
        wavefront : Wavefront 
            The wavefront to propagate.

        Returns
        -------
        electric_field : Array
            The complex electric field in the output plane.
        """
        # See gihub issue #52
        offsets = self.get_pixel_offsets(wavefront)
    
        input_positions = wavefront.get_pixel_coordinates()
        output_positions = dLux.utils.coordinates.get_pixel_coordinates(
                                self.get_pixels_out(),
                                self.get_pixel_scale_out())

        propagation_distance = self.get_focal_length() + self.get_focal_shift()

        field = wavefront.get_complex_form()
        field *= self.quadratic_phase(*input_positions,
            wavefront.get_wavelength(), -self.get_focal_length())
        field *= self.quadratic_phase(*input_positions, 
            wavefront.get_wavelength(), propagation_distance)

        wavefront = wavefront.update_phasor(np.abs(field), np.angle(field))
        
        field = super()._propagate(wavefront) 
        field *= wavefront.transfer_function(propagation_distance)
        field *= self.quadratic_phase(*output_positions, 
            wavefront.get_wavelength(), propagation_distance)
        return field 
    
    
# TODO: Implement eventually

# class AngularFresnel(Propagator):
#     """
#     Propagates an AngularWavefront in the Fresnel approximation.

#     Attributes
#     ----------
#     """
#     pass

