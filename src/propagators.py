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


    def _fourier_transform(self : Propagator, 
            wavefront : Wavefront, number_of_fringes : float, 
            pixel_offsets : tuple, pixels_out : int, 
            sign : int) -> Wavefront:
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
            The wavefront with the updated amplitude and phase after 
            the Fourier transform.
        """
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
            @ x_twiddle_factors)

        normalised_wavefront = self._normalise(complex_wavefront)
        amplitude = np.abs(normlised_wavefront)
        phase = np.angle(normalised_wavefront)
        
        return wavefront.update_phasor(amplitude, phase)


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
    pixels_out : int
        The number of pixels out. 
    pixel_scale_out : float 
        The dimensions of an output pixel in meters.    
    """
    focal_length : float
    pixel_scale_out : float = eqx.static_field()


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
        # TODO: So I could partial lock all of the parameters that 
        # I do not want to trace. Not sure how useful or why I want
        # to do it this way. 
        self._fourier_transform = functools.partial(
            self._fourier_transform, sign = 1 if inverse else -1,
            pixels_out = pixels_out)

        self.pixel_scale_out = pixel_scale_out
        self.focal_length = focal_length


    def get_focal_length():
    def get_pixel_scale():
    def get_number_of_fringes():
    def get_pixel_offsets():

    def __call__():


class PhysicalFFT(Propagator):
class PhysicalFresnel(Propagator):
class AngularMFT(Propagator):
class AngularFFT(Propagator):
class AngularFresnel(Propagator):


class MFT(eqx.Module):
    """
    Matches poppy but assumes square
    """
    npix_out: int = eqx.static_field()
    tilt_wf: bool
    inverse: bool
    focal_length:   float
    pixel_scale_out: float
    
    def __init__(self, npix_out, focal_length, pixel_scale_out, tilt_wf=False, inverse=False):
        self.npix_out = int(npix_out)
        self.tilt_wf = tilt_wf
        self.inverse = inverse
        self.focal_length =   np.array(focal_length).astype(float)
        self.pixel_scale_out = np.array(pixel_scale_out).astype(float)
    
    def __call__(self, params_dict):
        """
        
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        
        # Convert 
        wavefront = WF.amplitude * np.exp(1j * WF.phase)
        wavel = WF.wavelength
        pixelscale = WF.pixel_scale
        offset = WF.offset if self.tilt_wf else np.array([0., 0.])
        
        # Calculate NlamD parameter (Do in __init__?)
        npup = wavefront.shape[0] 
        npix = self.npix_out
        
        wf_size_in = pixelscale * npup # Wavefront size 'd'        
        det_size = self.pixel_scale_out * npix # detector size
        
        # wavel_scale =  wf_size_in * npix * pixel_rad
        wavel_scale = det_size * wf_size_in / self.focal_length
        num_fringe = wavel_scale / wavel
        
        # Calulate values
        offsetX, offsetY = offset * self.focal_length / self.pixel_scale_out
        dX = 1.0 / float(npup)
        dY = 1.0 / float(npup)
        dU = num_fringe / float(npix)
        dV = num_fringe / float(npix)
        
        # Generate arrays
        Xs = (np.arange(npup, dtype=float) - float(npup)/2 + offsetX + 0.5) * dX
        Ys = (np.arange(npup, dtype=float) - float(npup)/2 + offsetY + 0.5) * dY
        Us = (np.arange(npix, dtype=float) - float(npix)/2 + offsetX + 0.5) * dU
        Vs = (np.arange(npix, dtype=float) - float(npix)/2 + offsetY + 0.5) * dV
        XU = np.outer(Xs, Us)
        YV = np.outer(Ys, Vs)

        # Propagate wavefront
        sign = -1 if self.inverse else +1
        expXU = np.exp(sign * -2.0 * np.pi * 1j * XU)
        expYV = np.exp(sign * -2.0 * np.pi * 1j * YV).T
        t1 = np.dot(expYV, wavefront)
        wavefront = np.dot(t1, expXU)

        # Normalise wavefront
        # norm_coeff = np.sqrt((num_fringe**2) / (npup**2 * npix**2))
        norm_coeff = np.exp(np.log(num_fringe) - (np.log(npup) + np.log(npix)))
        wavefront_out = wavefront * norm_coeff
        
        # Update Wavefront Object
        WF = WF.update_phasor(np.abs(wavefront_out), np.angle(wavefront_out))
        WF = eqx.tree_at(lambda WF: WF.pixel_scale, WF, self.pixel_scale_out)
        WF = eqx.tree_at(lambda WF: WF.plane_type,  WF, "Focal")
        params_dict["Wavefront"] = WF
        return params_dict

class FFT(eqx.Module):
    """
    Note: FFT's natively center the zero frequency on a pixel center, basically
    fuck FFTs MFT 4 lyf
    """
    focal_length: float
    inverse: bool
    
    def __init__(self, focal_length, inverse=False):
        self.focal_length = np.array(focal_length).astype(float)
        self.inverse = inverse
        
    def __call__(self, params_dict):
        """
        Performs normalisation matching poppy
        """
        # Get relevant parameters
        WF = params_dict["Wavefront"]
        wavefront = WF.amplitude * np.exp(1j * WF.phase)
        wavel = WF.wavelength
        pixelscale = WF.pixel_scale

        # Calculate Wavefront & Pixelscale
        npix_in = wavefront.shape[0]
        if not self.inverse: # Forwards
            wavefront_out = npix_in * np.fft.fftshift(np.fft.ifft2(wavefront))
        else: # Inverse
            wavefront_out = 1./npix_in * np.fft.fft2(np.fft.ifftshift(wavefront))
        
        # Calculate Pixelscale
        pixelscale_out = wavel * self.focal_length / (pixelscale * npix_in)

        # Update Wavefront Object
        WF = WF.update_phasor(np.abs(wavefront_out), np.angle(wavefront_out))
        WF = eqx.tree_at(lambda WF: WF.pixel_scale, WF, pixelscale_out)
        WF = eqx.tree_at(lambda WF: WF.plane_type,  WF, "Focal")
        params_dict["Wavefront"] = WF
        return params_dict

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
