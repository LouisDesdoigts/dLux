import jax.numpy as np
import equinox as eqx
import typing

from wavefronts import GaussianWavefront, AngularWavefront, 
    PhysicalWavefront


Propagator = typing.NewType("Propagator", eqx.Module)
GaussianPropagator = typing.NewType("GaussianPropgator", object)


class Propagator(eqx.Module):
    """
    An abstract class indicating a spatial transfromation of the
    `Wavefront`. This is a separate class because it allows 
    us to take gradients with respect to the fields of the 
    propagator and hence optimise distances ect.     
    """
    def _get_scaled_coordinates(self : Propagator, 
            wavefront : Wavefront, pixel_offset : float,
            pixel_scale : float, number_of_pixels : int) -> Array:
        """
        Generate the pixel coordinates for the 
        """
        return (wavefront.get_pixel_coordinates(number_of_pixels) + \
            pixel_offset) * pixel_scale


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

        input_coordinates = self._get_scaled_coordinates(
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


    def _matrix_fourier_transform(self : Propagator, 
            wavefront : Wavefront, number_of_fringes : float, 
            pixel_offsets : tuple, pixels_out : int, 
            sign : int) -> Array:
        """
        Take the paraxial fourier transform of the wavefront in the 
        complex representation.

        Parameters
        ----------
        wavefront : Wavefront
            The `Wavefront` object that we want to Fourier transform.
        number_of_fringes : float
            The size of the output region in wavelength / distance 
            units. i.e. The number of diffraction fringes. 
        pixel_offsets : Array
            The displacement of the centre of the transform from the 
            centre of the wavefront in pixels in the input plane. 
        pixels_out : int
            The number of pixels following the transform in the 
            detector layer. 
        sign : int
            1. if forward Fourier transform else -1.

        Returns
        -------
        : Wavefront
            The complex un-normalised electric field after the 
            propagation.
        """
        # TODO: This is actually doing the inverse and the Fourier 
        # transform at once and as a result I think it needs to be 
        # split, so that they happen separately. 
        complex_wavefront = wavefront.complex_form()
        pixels_input = wavefront.number_of_pixels()
 
        input_scale = 1.0 / pixels_input
        output_scale = number_of_fringes / pixels_output
        
        x_offset, y_offset = pixel_offsets
        
        x_twiddle_factors = self._get_twiddle_factors(
            x_offset, (input_scale, output_scale), 
            (pixels_input, pixels_output), sign)

        y_twiddle_factors = self._get_twiddle_factors(
            y_offset, (input_scale, output_scale), 
            (pixels_input, pixels_output), sign)
        
        complex_wavefront = (y_twiddle_factors @ complex_wavefront) \
            @ x_twiddle_factors

        return complex_wavefront


    # TODO: I don't like the way that we have mutliple types of 
    # transform. 
    # TODO: Discuss with @LouisDesdoigts the alternative, where 
    # the only functions that are required to be written for 
    # the propagator are the get_number_of_fringes and get_pixel_offsets
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
        

    def __init__(self : Propagator) -> Propagator:
        """
        Abstract method that must be implemented by the subclasses.

        Throws 
        ------
        : AbstractClassError
            This is an abstract class and should not be directly 
            instansiated.
        """
        raise TypeError("This is an Abstract Class and should " +\
            "never be instantiated directly")


    def __call__(self : Propagator, parameters : dict) -> dict:
        """
        Propagate light through some optical path length.

        Parameters
        ----------
        parameters : dict 
            A dictionary of parameters containing a "Wavefront" 
            key.
        """
        raise TypeError("This is an Abstract Class and should " + \
            "never be instantiated directly.")


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


    def get_pixel_scale_out(self : Propagator) -> float:
        """
        Accessor fro the `pixel_scale_out` parameter.

        Returns
        -------
        : float
            The pixel scale in the plane of propagation in meters per 
            pixel.
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
        wavefront = parameters.get("Wavefront")

        number_of_fringes = self.get_number_of_fringes(wavefront)
        pixel_offsets = self.get_pixel_offsets(wavefront)
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
class AngularMFT(Propagator):
class AngularFFT(Propagator):
class AngularFresnel(Propagator):


class FresnelProp(eqx.Module):
    """
    Layer for Fresnel propagation
    
    Note this algorithm is completely not intensity conservative and will
    give different answers for each wavelength too
    
    Note this probably gives wrong answers for tilted input wavefronts becuase
    I believe its based on the paraxial wave equation and tilted wavefronts
    are not paraxial
    
    -> Do something clever with MFTs in the final TF with the offset term to 
    get propper off axis behaviour?
    """
    npix_out: int
    focal_length: float
    focal_shift: float
    pixel_scale_out: float
    
    def __init__(self, npix_out, focal_length, focal_shift, pixel_scale_out):
        """
        Initialisation
        pixelscale must be in m/pixel, ie aperture/npix
        
        Aperture pixelscale is the pixelscale at the ORIGINAL lens -> for multiple
        non-conjuagte lens systems is it slightly more complex (EFL, EFZ?)
        
        Do we need 'effective focal length' and 'effective final distance' in order to 
        track pixelscales correctly? Ie in an RC telescope after M2 the EFL is changed
        and the effective propagated distance is different too
          -> What about free space propagation?
        """
        self.npix_out = int(npix_out)
        self.focal_length =   np.array(focal_length).astype(float)
        self.focal_shift = np.array(focal_shift).astype(float)
        self.pixel_scale_out = np.array(pixel_scale_out).astype(float)
        
    def __call__(self, params_dict):
        """
        Propagates Fresnel
        
        Note: Normalisation calcuated in the Far-Field
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        wavefront = WF.amplitude * np.exp(1j * WF.phase)
        wavel = WF.wavelength
        pixelscale = WF.pixel_scale

        # Coords
        x_coords, y_coords = WF.get_xycoords()
        r_coords = np.hypot(x_coords, y_coords)

        # Apply Thin Lens Equation
        k = 2*np.pi / wavel # Wavenumber
        wavefront *= np.exp(-0.5j * k * r_coords**2 * 1/self.focal_length)
        
        # Calc prop parameters
        npix = wavefront.shape[0]
        wf_size = npix * pixelscale
        det_size = self.npix_out * self.pixel_scale_out
        z_prop = self.focal_length + self.focal_shift
        focal_ratio = self.focal_length / z_prop
        num_fringe_out = focal_ratio * wf_size * det_size / (wavel * self.focal_length)

        # First Phase Operation
        rho1 = np.exp(1.0j * k * (x_coords ** 2 + y_coords ** 2) / (2 * z_prop))
        wavefront *= rho1
        wavefront = self.mft(wavefront, num_fringe_out, self.npix_out)
        wavefront *= pixelscale ** 2

        # Coords
        # NOTE: This needs to be able to match the cenetering convtion
        # of the input wavefront
        xs = np.arange(self.npix_out) - self.npix_out//2
        YY, XX = np.meshgrid(xs, xs)
        x_coords, y_coords = self.pixel_scale_out * np.array([XX, YY])

        # Second Phase Operation
        rho2 = np.exp(1.0j * k * z_prop) / (1.0j * wavel * z_prop) * np.exp(1.0j * k * 
                                (x_coords ** 2 + y_coords ** 2) / (2 * z_prop))
        wavefront_out = rho2 * wavefront

        # Update Wavefront Object
        WF = WF.update_phasor(np.abs(wavefront_out), np.angle(wavefront_out))
        WF = eqx.tree_at(lambda WF: WF.pixel_scale, WF, self.pixel_scale_out)
        WF = eqx.tree_at(lambda WF: WF.plane_type,  WF, None)
        params_dict["Wavefront"] = WF
        return params_dict
    
    def mft(self, wavefront, num_fringe, npix):
        """
        Minimal on-axis MFT function
        """

        # npix = wavefront.shape[0]
        npup = wavefront.shape[0]

        # Calulate Arrays
        dX = 1.0 / float(npup)
        dU = num_fringe / npix

        Xs = (np.arange(npup, dtype=float) - float(npup) / 2.0) * dX
        Us = (np.arange(npix, dtype=float) - float(npix) / 2.0) * dU
        XU = np.outer(Xs, Us)
        expXU = np.exp(-2.0 * np.pi * 1j * XU)

        # Note: Can casue overflow issues on 32-bit
        # norm_coeff = np.sqrt((num_fringe**2) / (npup**2 * npix**2))
        norm_coeff = np.exp(np.log(num_fringe) - (np.log(npup) + np.log(npix)))

        # Perform MFT
        t1 = np.dot(expXU.T, wavefront)
        t2 = np.dot(t1, expXU)
        wavefront_out = norm_coeff * t2

        return wavefront_out


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
