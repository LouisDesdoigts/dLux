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
and FixedSamolingPropagtor. 


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
option for far field diffraction as well as a VairableSampling
near field algorithm. There is also a special intermediate 
plane Fresnel algorithm based on the GaussianWavefront class.


There is a total of seven non-abstract propagators defined in 
this file that are listed below:
 - PhysicalVariableFraunhofer
 - PhysicalFixedFraunhofer
 - PhysicalVariableFresnel
 - AngularVariableFraunhofer
 - AngularFixedFraunhofer
 - AngularVariableFresnel
 - VariableGaussianFresnel


Propagator
    Concrete Methods
    - _normalise
    - get_pixel_positions
    - generate_twiddle_factors

    Abstract Methods
    - __init__
    - __call__
    - get_pixel_offsets()
    - _fourier_transform()
    - _inverse_fourier_transform()

    VariableSampling (mft)
        PhysicalMFT
        PhysicalFresnel
        AngularMFT
        AngularFresnel
    FixedSampling (fft)
        PhysicalFFT
        AngularFFT
    GaussianPropagator
"""
__author__ = "Louis Desdoigts"
__author__ = "Jordan Dennis"
__date__ = "01/07/2022"

import jax.numpy as np
import equinox as eqx
import typing
import abc 
from wavefronts import *


Propagator = typing.NewType("Propagator", eqx.Module)


class Propagator(eqx.Module, abc.ABC):
    """
    An abstract class indicating a spatial transfromation of the
    `Wavefront`. This is a separate class because it allows 
    us to take gradients with respect to the fields of the 
    propagator and hence optimise distances ect.     
    """
    def _get_pixel_positions(self : Propagator, 
            pixel_offset : float, pixel_scale : float, 
            number_of_pixels : int) -> Array:
        """
        Generate the pixel coordinates for the plane of propagation.

        Parameters
        ----------
        pixel_offset : float
            The displacement of the center from the center of the 
            pixel array in pixels.
        pixel_scale : float
            The dimension of the pixel in meters/radians per pixel 
            in the plane of propagation.
        number_of_pixels : int
            The number of pixels in the plane of propagation. 

        Returns
        -------
        : Array
            The pixel positions along one dimension in meters.
        """
        # TODO: Review the npix // 2
        unscaled = np.arange(number_of_pixels) - number_of_pixels // 2
        return (unscaled + pixel_offset) * pixel_scale


    def _generate_twiddle_factors(self : Propagator,            
            pixel_offset : float, pixel_scales : tuple, 
            pixels : tuple, sign : int) -> Array:
        """
        The twiddle factors for the fourier transforms.

        Parameters
        ----------
        pixel_offset : float
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
        : Array
            The twiddle factors.
        """
        input_scale, output_scale = pixel_scales
        pixels_input, pixels_output = pixels

        input_coordinates = self.get_scaled_coordinates(
            wavefront, pixel_offset, input_scale, pixels_input)

        output_coordinates = self._get_scaled_coordinates(
            wavefront, pixel_offset, output_scale, pixels_output)

        input_to_output = np.outer(
            input_coordinates, output_coordinates)

        return np.exp(-2. * sign * np.pi * 1j * input_to_output)
         

    def _normalise(self : Propagator, complex_wavefront : Array, 
            number_of_fringes : int, pixels_input : int, 
            pixels_output : int) -> Wavefront:
        """
        Normalise the `Wavefront` according to the `poppy` convention. 

        Parameters
        ----------
        complex_wavefront : Array
            The `Wavefront` to normalise in the complex representation.
        number_of_fringes : int 
            The number of diffraction fringes at the detector layer.
        pixels_input : int 
            The number of pixels in the input image.
        pixels_output : int 
            The number of pixels in the output image.  

        Returns
        -------
        : Wavefront
            The normalised `Wavefront`.
        """
        normalising_factor = np.exp(np.log(number_of_fringes) - \
            (np.log(pixels_input) + np.log(pixels_output)))
        return complex_wavefront * normalising_factor


    @abc.abstractmethod
    def _fourier_transform(self : Propagtor, wavefront : Wavefront) -> Array:
        """
        The implementation of the Fourier transform that is to 
        be used for the optical propagation. This is for propagation 
        from the input plane to the plane of propagation. 

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront that is to be propagted.

        Returns 
        -------
        : Array
            The complex electric field amplitude following the 
            propagation in SI units of electric field. 
        """
        pass  


    @abc.abstractmethod
    def _inverse_fourier_transform(self : Propagator,
            wavefront : Wavefront) -> Array:
        """
        The implementation of the inverse Fourier transform that is 
        to be used for the optical propagation. The inverse represents
        propagtion from the plane of propagation to the input plane.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront that is getting propagated. 

        Returns 
        -------
        : Array
            The complex electric field amplitude following the 
            propagation in SI units of electric field. 
        """  
        pass


    @abc.abstractmethod
    def __init__(self : Propagator) -> Propagator:
        """
        Abstract method that must be implemented by the subclasses.

        Throws 
        ------
        : AbstractClassError
            This is an abstract class and should not be directly 
            instansiated.
        """
        pass         


    @abc.abstractmethod
    def __call__(self : Propagator, parameters : dict) -> dict:
        """
        Propagate light through some optical path length.

        Parameters
        ----------
        parameters : dict 
            A dictionary of parameters containing a "Wavefront" 
            key.
        """
        pass 


    # TODO: Review the location of this method
    @abc.abstractmethod
    def pixel_offests(self : Propagator) -> float:
        """
        Determine the offset of the wavefront from the plane 
        of propagation in units of pixels. 

        Returns
        -------
        : float
            The number of pixels in the x and y direction that the
            wavefront is offset from the plane of propagation.
        """              
        pass  


class VariableSamplingPropagator(Propagator, abc.ABC):
    """
    A propagator that users the Soummer et. al. 2007 MFT algorithm 
    to implement variable sampling in the plane of propagation 
    rather than the enforced; one pixel in the input plane = one 
    diffraction fringe in the output plane, fast fourier transform 
    algorithm.

    Attributes
    ----------
    pixel_scale_out : float 
        The pixel scale in the plane of propagation measured in 
        meters (radians) per pixel.
    pixels_out : int
        The number of pixels in the plane of propagation. 
    """
    pixel_scale_out : float
    pixels_out : float = eqx.static_field()


    def get_pixel_scale_out(self : Propagator) -> float:
        """
        Accessor for the pixel scale in the output plane. 

        Returns
        -------
        : float
            The pixel scale in the output plane in meters (radians)
            per pixel.
        """
        return self.pixel_scale_out


    def get_pixels_out(self : Propagator) -> int:
        """
        Accessor for the `pixels_out` parameter.

        Returns
        -------
        : int
            The number of pixels in the plane of propagation.
        """
        return self.pixels_out


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
        : Wavefront
            The complex un-normalised electric field after the 
            propagation.
        """
        complex_wavefront = wavefront.complex_form()
        pixels_input = 
 
        input_scale = 1.0 / wavefront.number_of_pixels()
        output_scale = number_of_fringes / self.get_pixels_out()
        
        x_offset, y_offset = self.get_pixel_offsets()
        
        x_twiddle_factors = self._get_twiddle_factors(
            x_offset, (input_scale, output_scale), 
            (pixels_input, pixels_output), sign)

        y_twiddle_factors = self._get_twiddle_factors(
            y_offset, (input_scale, output_scale), 
            (pixels_input, pixels_output), sign)
        
        complex_wavefront = (y_twiddle_factors @ complex_wavefront) \
            @ x_twiddle_factors

        return complex_wavefront


    def _fourier_transform(self : Propagator, wavefront : Wavefront) -> Array:
        """
        Compute the fourier transform of the electric field of the 
        input wavefront. This represents propagation from the input
        plane to the plane of propagation.

        Parameters
        ----------
        """
        return self._matrix_fourier_transform(wavefront, sign = 1)


    def _inverse_fourier_transform(self :)

    @abc.abstractmethod
    def number_of_fringes(self : Propagator, wavefront : Wavefront) -> float:
        """
        Calculates the number of diffraction fringes in the plane of
        propagation. 

        Parameters
        ----------
        wavefront : Wavefront 
            The wavefront that is getting propagated. Ussually 
            required for access to the `wavelength` parameter.

        Returns 
        -------
        : float
            The number of diffraction fringes in the plane of 
            propagation.
        """
        pass



class FixedSamplingPropagator(Propagator):
    def _inverse_fast_fourier_transform(self : Propagator,
            wavefront : Wavefront) -> Array
        """
        Perfrom an inverse fourier transform of a wavefront.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to be propagated.

        Returns
        -------
        : Array
            The complex electric field in SI units following propagation.
        """
        return wavefront.number_of_pixels() * \
            np.fft.fftshift(np.fft.ifft2(wavefront.get_complex_form()))
        

    def _fast_fourier_transform(self : Propagator, 
            wavefront : Wavefront) -> Array:
        """
        Perfrom a fast fourier transform on the wavefront.

        Parameters
        ----------
        wavefront : Wavefront 
            The wavefront to propagate.

        Returns
        -------
        : Array
            The complex electric field in SI units following propagation.
        """
        return 1. / wavefront.number_of_pixels() * \
            np.fft.fft2(np.fft.ifftshift(wavefront.get_complex_form()))



class FocalPropagtor(): 


class PhysicalMFT(Propagator):
    """
    Fraunhofer propagation based on the a matrix fourier transfrom 
    with adjustable scaling. 

    Attributes 
    ----------
    focal_length : float
        The focal length of the propagation distance.
    pixel_scale_out : float 
        The dimensions of an output pixel in meters.    
    """
    focal_length : float
    pixel_scale_out : float 
    pixels_out : int


    # TODO: Talk to @LouisDesdoigts about how the FFT can be run 
    # using the MFT by passing pixels_out == pixels_in and 
    # pixel_scale_out = wavel * self.focal_length / (pixelscale * npix_in)
    def __init__(focal_length : float, pixels_out : int, 
            pixel_scale_out : float, inverse : bool) -> PhysicalMFT:
        """
        Parameters
        ----------
        focal_length : float
            The focal length of the mirror or lens that the Wavefront
            is propagating away from.
        pixels_out : int
            The number of pixels in the output image. 
        pixel_scale_out : float 
        iverse : bool
        """
        self._fourier_transform = functools.partial(
            self._matrix_fourier_transform, sign = 1 if inverse else -1)

        self.pixel_scale_out = pixel_scale_out
        self.focal_length = focal_length
        self.pixels_out = pixels_out


    def get_focal_length(self : Propagator) -> float:
        """
        Accessor for the focal_length.

        Returns 
        -------
        : float
            The focal length in meters. 
        """
        return self.focal_length




    def get_number_of_fringes(self : Propagator, 
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
        size_in = wavefront.get_pixel_scale() * \
            wavefront.number_of_pixels()        
        size_out = self.get_pixel_scale() * \
            self.get_pixels_out()
        
        return size_in * size_out / self.get_focal_length() /\
            wavefront.get_wavelength()


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
        # TODO: Confirm that if the wavefront.get_offset != 0. then 
        # we will always want to use that offset.
        return wavefront.get_offset() * self.get_focal_length() / \
            self.get_pixel_scale()


    # TODO: Still not convinced that params_dict is nessecary
    # need to confirm this be examining some of the higher level 
    # code.
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
        : dict 
            A dictionary with the updated "Wavefront" key; value
        """
        wavefront = parameters["Wavefront"]

#        number_of_fringes = self.get_number_of_fringes(wavefront)
#        pixel_offsets = self.get_pixel_offsets(wavefront)
        complex_wavefront = self._fourier_transform(wavefront,
            number_of_fringes, pixel_offsets, self.get_pixels_out())

        normalised_wavefront = self._normalise(complex_wavefront)
        amplitude = np.abs(normlised_wavefront)
        phase = np.angle(normalised_wavefront)

        new_wavefront wavefront\
            .update_phasor(amplitude, phase)\
            .set_pixel_scale(self.get_pixel_scale_out())

        parameters["Wavefront"] = new_wavefront
        return parameters
        

class PhysicalFFT(Propagator):
    """
    Perfrom an FFT propagation on a `PhysicalWavefront`. This is the 
    same as the MFT however it samples 1 diffraction fringe per pixel.

    Attributes
    ----------
    focal_length : float 
        The focal length of the lens or mirror in meters.
    """
    focal_length : float


    def __init__(self : Propagator, focal_length : float, 
            inverse : bool) -> Propagtor:
        # TODO: Confirm documentation for inverse this is not currently 
        # correct.
        """
        Parameters
        ----------
        focal_length : float
            The focal length of the lens or mirror that is getting 
            propagated away from in meters.
        inverse : bool
            Propagation direction. True for forwards and False for 
            backwards. 

        Returns
        -------
        : Propagator
            A `Propagator` object representing the optical path from 
            the mirror to the focal plane. 
        """
        self._fourier_transform = self._fast_fourier_transform if \
            inverse else self._inverse_fast_fourier_transform

        self.focal_length = focal_length


    # TODO: Consider clever ways of writing automatic getters and 
    # setters. 
    def get_focal_length(self : Propagtor) -> float:
        """
        Accessor for the focal length in meters.

        Returns
        -------
        : float
            The focal length of the lens or mirror in meters.
        """
        return self.focal_length


    def get_pixel_scale_out(self : Propagator, 
            wavefront : Wavefront) -> float:
        """
        The pixel scale in the focal plane in meters per pixel

        Parameters
        ----------
        wavefront : Wavefront
            The `Wavefront` that is propagting.

        Returns
        -------
        : float
            The pixel scale in meters per pixel.
        """
        return wavefront.get_wavelength() * \
            self.get_focal_length() / (wavefront.get_pixel_scale() \
                * wavefront.number_of_pixels())


    # TODO: Talk to @LouisDesdoigts about this. 
    def __call__(self : Propagtor, parameters : dict) -> dict:
        """
        Propagate the wavefront to or from the focal plane. 

        Parameters
        ----------
        parameters : dict
            A dictionary containing the field "Wavefront"

        Returns 
        -------
        : dict
            The dictionary of parameters with the updated "Wavefront"
            key; value.
        """
        wavefront = parameters.get("Wavefront")

        complex_wavefront = self._fourier_transform(wavefront)
        normalised_wavefront = self._normalise(complex_wavefront)

        amplitude = np.abs(normlised_wavefront)
        phase = np.angle(normalised_wavefront)

        # TODO: Update the debugging information regarding the plane 
        # of the propagation.
        new_wavefront wavefront\
            .update_phasor(amplitude, phase)\
            .set_pixel_scale(self.get_pixel_scale_out())

        parameters["Wavefront"] = new_wavefront
        return parameters
  

class PhysicalFresnel(Propagator):
    """
    Near field diffraction based on the Frensel approximation.
    This implementation does not conserve flux because the 
    normalisation happens in the far field. 

    Attributes
    ----------
    focal_length : float
        The focal length of the lens or mirror in meters. This 
        is a differentiable parameter.
    focal_shift : float
        The displacement of the plane of propagation from the 
        the focal_length in meters. This is a differentiable
        parameter. The focal shift is positive if the plane of 
        propagation is beyond the focal length and negative 
        if the plane of propagation is inside the focal length.
    pixels_out : int
        The number of pixels in the plane of propagation. This 
        is not a differentiable parameter.
    pixel_scale_out : float
        The pixel scale in the plane of propagation in meters per 
        pixel. This is a differentiable parameter. 
    """
    focal_length : float
    focal_shift : float
    pixels_out : int = eqx.static_field()
    pixel_scale_out : float


    def __init__(self : Propagator, focal_length : float, 
            focal_shift : float, pixels_out : float, 
            pixel_scale_out : float, inverse : bool) -> Propagator:
        """
        Parameters
        ----------
        focal_length : float
            The focal length of the mirror or the lens in meters.
        focal_shift : float
            The distance away from the focal plane to be propagated
            to in meters. This quantity 
        pixels_out : int
            The number if pixels in the plane of propagation.
        pixel_scale_out : float
            The scale of a pixel in the plane of propagation in 
            units of meters per pixel. 
        inverse : bool
            True if propagating from the plane of propagation and
            False if propagating to the plane of propagation. 
        """
        self.focal_shift = numpy.asarray(focal_shift).astype(float)
        self.focal_length = numpy.asarray(focal_length).astype(float)
        self.pixels_out = int(pixels_out)
        self.pixel_scale_out = numpy.asarry(pixel_scale_out).astype(float)

        self._fourier_transform = functools.partial(
            self._matrix_fourier_transform, sign = 1 if inverse else -1)


    def get_focal_length(self : Propagator) -> float:
        """
        Accessor for the focal length.

        Returns
        -------
        : float
            The focal length of the mirror or lens in meters.
        """
        return self.focal_length
    

    def get_pixels_out(self : Propagator) -> int:
        """
        Accessor for the number of pixels in the plane of propagation.

        Returns
        : int
            The number of pixels in the plane of propagation.
        """
        return self.pixels_out


    def get_pixel_scale_out(self : Propagator) -> float:
        """
        Accessor for the pixel scale in the plane of propagation.

        Returns
        -------
        : float
            The pixel scale in the plane of propgation in meters per 
            pixel.
        """
        return self.pixel_scale_out


    def number_of_fringes(self : Propagtor, 
            wavefront : Wavefront) -> float:
        """
        The number of diffraction fringes in the plane of propagation.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront that is getting propagated. 

        Returns
        -------
        : float
            The number of diffraction fringes visible in the plane of 
            propagation.
        """
        size_in = wavefront.number_of_pixels() * wavefront.get_pixel_scale()
        size_out = self.get_pixels_out() * self.get_pixel_scale_out()
        propagtion_distance = self.get_focal_length() + self.get_focal_shift()
        focal_ratio = self.get_focal_length() / propagation_distance
        return focal_ratio * size_in * size_out / \
            (wavefront.get_wavelength() * self.get_focal_length()) 
               
        
    def quadratic_phase(self : Propagator, x_coordinates : Array, 
            y_coordinates : Array, wavelength : float) -> Array:
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
        wavelength : float
            The wavelength of the wavefront in meters.
        """
        wavenumber = 2 * np.pi / wavelength
        radial_coordinates = np.hypot(x_coordinates, y_coordinates)
        return np.exp(-0.5j * wavenumber * radial_coordinates ** 2 \
            / self.get_focal_length())

    
    # TODO: This duplication seems to be inevitable unless we were
    # to build a more complex class heirachy.
    def get_pixel_positions(self : Propagator) -> Array:
        """
        The pixel positions in meters in the plane of propagation.

        Returns 
        -------
        : Array
            The pixel positions in meters.
        """
        # TODO: confirm (npix - 1) / 2 or npix // 2
        pixel_coordinates = self._get_scaled_coordinates()
        return np.meshgrid(pixel_coordinates, pixel_coordinates)
        

    def thin_lens(self : Propagator, wavefront : Wavefront) -> Array:
        # TODO: Review this documentation 
        """
        A thin lens focusing the wavefront into the plane of 
        propagation instead of the focal plane.

        Parameters
        ----------
        wavefront : Wavefront 
            The wavefront that is getting focussed.

        Returns
        -------
        : Array
            The complex electric field in SI units of electric field. 
        """
        x_coordinates, y_coordinates = wavefront.get_pixel_positions()
        radial_coordinates = np.hypot(x_coordinates, y_coordinates)
        wavenumber = 2 * np.pi / self.get_wavelength() # Wavenumber
        return np.exp(-0.5j * wavenumber * radial_coordinates ** 2 \
            / self.get_focal_length())


    def __call__(self : Propagator, parameters : dict) -> dict:
        """
        Propagate the wavefront within the Fresnel approximation.

        Parameters
        ----------
        parameters : dict
            A dictionary of parameters that contains a key "Wavefront"

        Returns
        -------
        parameters : dict
            The `parameters` with an updated key "wavefront"; 
            value. 
        """
        # Get relevant parameters
        wavefront = parameters["Wavefront"]

        complex_wavefront = self.thin_lens(wavefront)
        input_positions = wavefront.get_pixel_positions()
        output_positions = self.get_pixel_positions()
        number_of_fringes = self.number_of_fringes(wavefront)
        first_quadratic_phase = self.quadractic_phase(
            *input_positions, wavefront.get_wavelength())
        transfer = wavefront.transfer_function(
            self.get_focal_length() + self.get_focal_shift())
        second_quadratic_phase = self.quadratic_phase(
            *output_positions, wavefront.get_wavelength())

        complex_wavefront *= first_quadratic_phase
        complex_wavefront = self._fourier_transform(complex_wavefront, 
            number_of_fringes, [0., 0.], self.get_pixels_out()) 
        complex_wavefront = self._normalise(complex_wavefront,
            number_of_fringes, wavefront.number_of_pixels(), 
            self.get_pixels_out)
        complex_wavefront *= wavefront.get_pixel_scale() ** 2
        complex_wavefront *= transfer * second_quadratic_phase
        
        amplitude = np.abs(wavefront_out)
        phase = np.angle(wavefront_out)

        # Update Wavefront Object
        new_wavefront = wavefront\
            .update_phasor(amplitude, phase)\
            .set_pixel_scale(self.get_pixel_scale_out())

        parameters["Wavefront"] = new_wavefront
        return parameters


class AngularMFT(Propagator):
    """
    Propagation of an AngularWavefront by a paraxial matrix fourier
    transform. 

    Attributes
    ----------

    """
    def __init__():
    def get_number_of_fringes():
    def get_pixel_offsets():
    def __call__():


class AngularFFT(Propagator):


class AngularFresnel(Propagator):


class GaussianPropagator(eqx.Module):
    """
    An intermediate plane fresnel algorithm for propagating the
    `GaussianWavefront` class between the planes. The propagator 
    is separate from the `Wavefront` to emphasise the similarity 
    of these algorithms to the layers in machine learning algorithms.

    Attributes
    ----------
    distance : float 
       The distance to propagate in meters. 
    """
    distance : float 


    def __init__(self : GaussianPropagator, 
            distance : float) -> GaussianPropagator:
        """
        Constructor for the GaussianPropagator.

        Parameters
        ----------
        distance : float
            The distance to propagate the wavefront in meters.
        """
        self.distance = distance


    def planar_to_planar(self : GaussianPropagator, 
            wavefront: GaussianWavefront, 
            distance: float) -> GaussianWavefront:
        """
        Modifies the state of the wavefront by propagating a planar 
        wavefront to a planar wavefront. 

        Parameters
        ----------
        wavefront : GaussianWavefront
            The wavefront to propagate. Must be `GaussianWavefront`
            or a subclass. 
        distance : float
            The distance of the propagation in metres.

        Returns
        -------
        : GaussianWavefront
            The new `Wavefront` propagated by `distance`. 
        """
        complex_wavefront = wavefront.get_amplitude() * \
            np.exp(1j * self.get_phase())

        new_complex_wavefront = np.fft.ifft2(
            wavefront.transfer_function(distance) * \
            np.fft.fft2(complex_wavefront))

        new_amplitude = np.abs(new_complex_wavefront)
        new_phase = np.angle(new_complex_wavefront)
        
        return wavefront\
            .set_position(wavefront.get_position() + distance)\
            .update_phasor(new_amplitude, new_phase)        


    def waist_to_spherical(self : GaussianPropagator, 
            wavefront: GaussianWavefront, 
            distance: float) -> GaussianWavefront:
        """
        Modifies the state of the wavefront by propagating it from 
        the waist of the gaussian beam to a spherical wavefront. 

        Parameters
        ----------
        wavefront : GaussianWavefront
            The `Wavefront` that is getting propagated. Must be either 
            a `GaussianWavefront` or a valid subclass.
        distance : float 
            The distance of the propagation in metres.

        Returns 
        -------
        : GaussianWavefront
            `wavefront` propgated by `distance`.
        """
        coefficient = 1 / 1j / wavefront.get_wavelength() / distance
        complex_wavefront = wavefront.get_amplitude() * \
            np.exp(1j * self.get_phase())

        fourier_transform = jax.lax.cond(np.sign(distance) > 0, 
            lambda wavefront, distance: \
                quadratic_phase_factor(distance) * \
                np.fft.fft2(wavefront), 
            lambda wavefront, distance: \
                quadratic_phase_factor(distance) * \
                np.fft.ifft2(wavefront),
            complex_wavefront, distance)

        new_complex_wavefront = coefficient * fourier_transform
        new_phase = np.angle(new_complex_wavefront)
        new_amplitude = np.abs(new_complex_wavefront)

        return wavefront\
            .update_phasor(new_amplitude, new_phase)\
            .set_position(wavefront.get_position() + distance)


    def spherical_to_waist(self : GaussianPropagator, 
            wavefront: GaussianWavefront, distance: float) -> GaussianWavefront:
        """
        Modifies the state of the wavefront by propagating it from 
        a spherical wavefront to the waist of the Gaussian beam. 

        Parameters
        ----------
        wavefront : GaussianWavefront 
            The `Wavefront` that is getting propagated. Must be either 
            a `GaussianWavefront` or a direct subclass.
        distance : float
            The distance of propagation in metres.

        Returns
        -------
        : GaussianWavefront
            The `wavefront` propagated by distance. 
        """
        # TODO: consider adding get_complex_wavefront() as this is 
        # used in a number of locations. This could be implemented
        # in the PhysicalWavefront and inherrited. 
        coefficient = 1 / 1j / wavefront.get_wavelength() / \
            distance * wavefront.quadratic_phase_factor(distance)
        complex_wavefront = wavefront.get_amplitude() * \
            np.exp(1j * wavefront.get_phase())

        fourier_transform = jax.lax.cond(np.sign(distance) > 0, 
            lambda wavefront: np.fft.fft2(wavefront), 
            lambda wavefront: np.fft.ifft2(wavefront),
            complex_wavefront)

        new_wavefront = coefficient * fourier_transform
        new_phase = np.angle(new_wavefront)
        new_amplitude = np.abs(new_wavefront)
        return wavefront\
            .update_phasor(new_amplitude, new_phase)\
            .set_position(wavefront.get_position() + distance)


    def outside_to_outside(self : GaussianWavefront, 
            wavefront : GaussianWavefront, 
            distance: float) -> GaussianWavefront:
        """
        Propagation from outside the Rayleigh range to another 
        position outside the Rayleigh range. 

        Parameters
        ----------
        wavefront : GaussianWavefront
            The wavefront to propagate. Assumed to be either a 
            `GaussianWavefront` or a direct subclass.
        distance : float
            The distance to propagate in metres.
    
        Returns
        -------
        : GaussianWavefront 
            The new `Wavefront` propgated by distance. 
        """
        # TODO: Make sure that the displacements are called by the 
        # correct function.
        from_waist_displacement = wavefront.get_position() \
            + distance - wavefront.location_of_waist()
        to_waist_displacement = wavefront.location_of_waist() \
            - wavefront.get_position()

        wavefront_at_waist = self.spherical_to_waist(
            wavefront, to_waist_displacement)
        wavefront_at_distance = self.spherical_to_waist(
            wavefront_at_waist, from_waist_displacement)

        return wavefront_at_distance


    def outside_to_inside(self : GaussianPropagator,
            wavefront: GaussianWavefront, distance: float) -> None:
        """
        Propagation from outside the Rayleigh range to inside the 
        rayleigh range. 

        Parameters
        ----------
        wavefront : GaussianWavefront
            The wavefront to propagate. Must be either 
            `GaussianWavefront` or a direct subclass.
        distance : float
            The distance to propagate in metres.

        Returns
        -------
        : GaussianWavefront
            The `Wavefront` propagated by `distance` 
        """
        from_waist_displacement = wavefront.position + distance - \
            wavefront.location_of_waist()
        to_waist_position = wavefront.location_of_waist() - \
            wavefront.get_position()

        wavefront_at_waist = self.planar_to_planar(
            wavefront, to_waist_displacement)
        wavefront_at_distance = self.spherical_to_waist(
            wavefront_at_waist, from_waist_displacement)

        return wavefront_at_distance


    def inside_to_inside(self : GaussianPropagator, 
            wavefront : GaussianWavefront, 
            distance: float) -> GaussianWavefront:
        """
        Propagation from inside the Rayleigh range to inside the 
        rayleigh range. 

        Parameters
        ----------
        wavefront : GaussianWavefront
            The `Wavefront` to propagate. This must be either a 
            `GaussianWavefront` or a direct subclass.
        distance : float
            The distance to propagate in metres.


        Returns
        -------
        : GaussianWavefront
            The `Wavefront` propagated by `distance`
        """
        # TODO: Consider removing after checking the Jaxpr for speed. 
        # This is just an alias for another function. 
        return self.planar_to_planar(wavefront, distance)


    def inside_to_outside(self : GaussianPropagator, 
            wavefront : GaussianWavefront, 
            distance: float) -> GaussianWavefront:
        """
        Propagation from inside the Rayleigh range to outside the 
        rayleigh range. 

        Parameters
        ----------
        wavefront : GaussianWavefront
            The `Wavefront` to propgate. Must be either a 
            `GaussianWavefront` or a direct subclass.
        distance : float
            The distance to propagate in metres.

        Returns
        -------
        : GaussianWavefront 
            The `Wavefront` propagated `distance`.
        """
        from_waist_displacement = wavefront.get_position() + \
            distance - wavefront.location_of_waist()
        to_waist_displacement = wavefront.location_of_waist() - \
            wavefront.get_position()

        wavefront_at_waist = self.planar_to_planar(
            wavefront, to_waist_displacement)
        wavefront_at_distance = self.waist_to_spherical(
            wavefront_at_waist, from_waist_displacement)

        return wavefront_at_distance


    def __call__(self : GaussianPropagator, 
            wavefront : GaussianWavefront, 
            distance: float) -> GaussianWavefront:
        """
        Propagates the wavefront approximated by a Gaussian beam 
        the amount specified by distance. Note that distance can 
        be negative.

        Parameters 
        ----------
        wavefront : GaussianWavefront
            The `Wavefront` to propagate. Must be a `GaussianWavefront`
            or a direct subclass.
        distance : float
            The distance to propagate in metres.

        Returns
        -------
        : GaussianWavefront 
            The `Wavefront` propagated `distance`.
        """
        # This works by considering the current position and distnace 
        # as a boolean array. The INDEX_GENERATOR converts this to 
        # and index according to the following table.
        #
        # sum((0, 0) * (1, 2)) == 0
        # sum((1, 0) * (1, 2)) == 1
        # sum((0, 1) * (1, 2)) == 2
        # sum((1, 1) * (1, 2)) == 3
        #
        # TODO: Test if a simple lookup is faster. 
        # Constants
        INDEX_GENERATOR = np.array([1, 2])

        decision_vector = wavefront.is_inside([
            wavefront.get_position(), 
            wavefront.get_position() + distance])
        decision_index = np.sum(
            self.INDEX_GENERATOR * descision_vector)
 
        # Enters the correct function differentiably depending on 
        # the descision.
        new_wavefront = jax.lax.switch(decision_index, 
            [self.inside_to_inside, self.inside_to_outside,
            self.outside_to_inside, self.outside_to_outside],
            wavefront, distance) 

        return new_wavefront
