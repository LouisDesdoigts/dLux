from __future__ import annotations
import jax.numpy as np
from equinox import tree_at
from abc import ABC, abstractmethod
import dLux


__all__ = ["CartesianMFT", "AngularMFT", "CartesianFFT", "AngularFFT",
           "CartesianFresnel"]


Array = np.ndarray


########################
### Abstract Classes ###
########################
class Propagator(dLux.layers.OpticalLayer, ABC):
    """
    An abstract class to store the various properties of the propagation of
    some wavefront.

    Attributes
    ----------
    inverse : bool
        Is this an 'inverse' propagation. Non-inverse propagations represents
        propagation from a pupil to a focal plane, and inverse represents
        propagation from a focal to a pupil plane.
    """
    inverse : bool


    def __init__(self    : Propagator,
                 inverse : bool = False,
                 **kwargs) -> Propagator:
        """
        Constructor for the Propagator.

        Parameters
        ----------
        inverse : bool = False
            Is this an 'inverse' propagation. Non-inverse propagations
            represents propagation from a pupil to a focal plane, and inverse
            represents propagation from a focal to a pupil plane.
        """
        super().__init__(**kwargs)
        self.inverse = bool(inverse)


    def is_inverse(self : Propagator) -> bool:
        """
        Accessor for the inverse attribute.

        Returns
        -------
        inverse : bool
            Is this an 'inverse' propagation. Non-inverse propagations represent
            propagation from a pupil to a focal plane, and inverse represent
            propagation from a focal to a pupil plane.
        """
        return self.inverse


    def set_inverse(self : Propagator, inverse : bool) -> Propagator:
        """
        Mutator for the inverse attribute.

        Parameters
        ----------
        inverse : bool
            The new value for the inverse attribute.

        Returns
        -------
        propagator : Propagator
            The Propagator with the updated inverse attribute.
        """
        assert isinstance(inverse, bool), "inverse must be a boolen."
        return tree_at(lambda propagator: propagator.inverse, self, tilt)

    @abstractmethod
    def propagate(self : Propagator, wavefront : Wavefront) -> Array:
        """
        Performs the propagation as a directional wrapper to the fourier methods
        of the class.

        Parameters
        ----------
        wavefront : Wavefront
            The `Wavefront to propagate.

        Returns
        -------
        field : Array
            The normalised electric field of the wavefront after propagation.
        """
        return


class VariableSamplingPropagator(Propagator, ABC):
    """
    A propagator that implements the Soummer et. al. 2007 MFT algorithm
    allowing variable sampling in the outuput plane rather than the fixed
    sampling enforced by Fast Fourier Transforms(FFTs).

    Attributes
    ----------
    npixels_out : int
        The number of pixels in the output plane.
    pixel_scale_out : Array, meters/pixel or radians/pixel
        The pixel scale in the output plane, measured in meters or radians per
        pixel for Cartesian or Angular Wavefront respectively.
    shift : Array
        The (x, y) shift to apply to the wavefront in the output plane.
    pixel_shift : bool
        Should the shift value be considered in units of pixels, or in the
        physical units of the output plane (ie pixels or meters, radians). True
        interprets the shift value in pixel units.
    """
    npixels_out     : int
    pixel_scale_out : Array
    shift           : Array
    pixel_shift     : bool

    def __init__(self            : Propagator,
                 pixel_scale_out : Array,
                 npixels_out     : int,
                 shift           : Array = np.array([0., 0.]),
                 pixel_shift     : bool  = False,
                 **kwargs) -> Propagator:
        """
        Constructor for VariableSampling propagators.

        Parameters
        ----------
        npixels_out : int
            The number of pixels in the output plane.
        pixel_scale_out : Array, meters/pixel or radians/pixel
            The pixel scale in the output plane, measured in meters or radians
            per pixel for Cartesian or Angular Wavefront respectively.
        shift : Array = np.array([0., 0.])
            The (x, y) shift to apply to the wavefront in the output plane.
        pixel_shift : bool
            Should the shift value be considered in units of pixel, or in the
            physical units of the output plane (ie pixels or meters, radians). =
            True interprets the shift value in pixel units.
        """
        super().__init__(**kwargs)
        self.pixel_scale_out = np.asarray(pixel_scale_out, dtype=float)
        self.npixels_out     = int(npixels_out)
        self.shift           = np.asarray(shift, dtype=float)
        self.pixel_shift     = bool(pixel_shift)
        assert self.pixel_scale_out.ndim == 0, \
        ("pixel_scale_out must be a scalar.")
        assert self.shift.shape == (2,), \
        ("shift must be an array of shape (2,) ie (x, y).")


    def is_pixel_shifted(self : Propagator) -> bool:
        """
        Accessor for the pixel_shift parameter.

        Returns
        -------
        pixel_shift : bool

        """
        return self.pixel_shift


    def get_shift(self : Propagator) -> Array:
        """
        Accessor for the shift parameter.

        Returns
        -------
        shift : Array
            The (x, y) shift to apply to the wavefront throughout the
            propagation.
        """
        return self.shift


    def get_pixel_scale_out(self : Propagator) -> Array:
        """
        Accessor for the pixel_scale_out parameter.

        Returns
        -------
        pixel_scale_out : Array, meters/pixel or radians/pixel
            The pixel scale in the output plane, measured in meters or radians
            per pixel for Cartesian or Angular Wavefront respectively.
        """
        return self.pixel_scale_out


    def get_npixels_out(self : Propagator) -> int:
        """
        Accessor for the npixels_out parameter.

        Returns
        -------
        npixels_out : int
            The number of pixels in the output plane.
        """
        return self.npixels_out


    def set_shift(self : Propagator,
                  shift : bool) -> Propagator:
        """
        Mutator for the shift attribute.

        Parameters
        ----------
        shift : Array
            The new (x, y) value for the shift attribute.

        Returns
        -------
        propagator : Propagator
            The Propagator with the updated shift attribute.
        """
        assert isinstance(shift, Array) and shift.shape == (2,), \
        ("shift must be an array of shape (2,) ie (x, y).")
        return tree_at(lambda propagator: propagator.shift, self, shift)


    def set_pixel_shifted(self : Propagator, pixel_shift : bool) -> Propagator:
        """
        Mutator for the pixel_shift attribute.

        Parameters
        ----------
        pixel_shift : bool
            The new value for the pixel_shift attribute.

        Returns
        -------
        propagator : Propagator
            The Propagator with the updated pixel_shift attribute.
        """
        assert isinstance(pixel_shift, bool), "pixel_shift must be a boolean."
        return tree_at(lambda propagator: propagator.pixel_shift, self,
                                                                   pixel_shift)


    def set_pixel_scale_out(self            : Propagator,
                            pixel_scale_out : bool) -> Propagator:
        """
        Mutator for the pixel_scale_out attribute.

        Parameters
        ----------
        pixel_scale_out : Array
            The new value for the pixel_scale_out attribute.

        Returns
        -------
        propagator : Propagator
            The Propagator with the updated pixel_scale_out attribute.
        """
        assert isinstance(pixel_scale_out, Array) and \
        pixel_scale_out.ndim == 0, ("pixel_scale_out must be scalar array.")
        return tree_at(lambda propagator: propagator.pixel_scale_out, self,
                                                            pixel_scale_out)


    def set_npixels_out(self : Propagator, npixels_out : int) -> Propagator:
        """
        Mutator for the npixels_out attribute.

        Parameters
        ----------
        npixels_out : int
            The new value for the npixels_out attribute.

        Returns
        -------
        propagator : Propagator
            The Propagator with the updated npixels_out attribute.
        """
        assert isinstance(npixels_out, int), "npixels_out must be a integer."
        return tree_at(lambda propagator: propagator.npixels_out, self,
                                                                   npixels_out)


    @abstractmethod
    def get_nfringes(self : Propagator, wavefront : Wavefront) -> Array:
        """
        The number of diffraction fringes in the output plane.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront being propagated.

        Returns
        -------
        fringes : Array
            The number of diffraction fringes in the output plane.
        """
        return


    def _generate_twiddle_factors(self         : Propagator,
                                  pixel_offset : Array,
                                  pixel_scales : tuple,
                                  npixels      : tuple) -> Array:
        """
        The twiddle factors for the fourier transforms.

        Parameters
        ----------
        pixel_offset : Array, pixels
            The offset in units of pixels.
        pixel_scales : tuple
            The pixel_scale values at the input and output planes respectively.
        npixels : tuple
            The number of pixels at the input and output planes respectively.

        Returns
        -------
        twiddle_factors : Array
            The twiddle factors.
        """
        input_scale, output_scale = pixel_scales
        pixels_input, npixels_out = npixels
        sign = -1 if self.is_inverse() else 1

        input_coordinates = dLux.utils.coordinates.get_coordinates_vector(
                                                pixels_input, input_scale, 
                                                pixel_offset/input_scale)

        output_coordinates = dLux.utils.coordinates.get_coordinates_vector(
                                                npixels_out, output_scale, 
                                                pixel_offset/output_scale)

        input_to_output = np.outer(input_coordinates, output_coordinates)

        return np.exp(-2. * sign * np.pi * 1j * input_to_output)


    def propagate(self       : Propagator,
                   wavefront : Wavefront) -> Array:
        """
        Propagates the wavefront from the input plane to the output plane using
        a Matrix Fourier Transform.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront being propagated.

        Returns
        -------
        field : Array
            The normalised electric field phasor after the propagation.
        """
        field = wavefront.get_phasor()
        nfields = field.shape[0]

        input_scale = 1.0 / wavefront.get_npixels()
        output_scale = self.get_nfringes(wavefront) / self.get_npixels_out()
        npixels_in = wavefront.get_npixels()
        npixels_out = self.get_npixels_out()
        x_offset, y_offset = self.get_shift()

        # TODO: This can be vmapped
        x_twiddle_factors = np.tile(self._generate_twiddle_factors(
            x_offset, (input_scale, output_scale), 
            (npixels_in, npixels_out)), (nfields, 1, 1))

        y_twiddle_factors = np.tile(self._generate_twiddle_factors(
            y_offset, (input_scale, output_scale), 
            (npixels_in, npixels_out)).T, (nfields, 1, 1))

        output_field = (y_twiddle_factors @ field) @ x_twiddle_factors

        normalising_factor = np.exp(np.log(self.get_nfringes(wavefront)) - \
               (np.log(npixels_in) + np.log(npixels_out)))

        return output_field * normalising_factor


    def __call__(self : Propagator, wavefront : Wavefront) -> Wavefront:
        """
        Propagates the `Wavefront`.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate.

        Returns
        -------
        wavefront : Wavefront
            The wavefront with the optical layer applied.
        """
        # print(wavefront)
        phasor = self.propagate(wavefront)

        new_amplitude = np.abs(phasor)
        new_phase = np.angle(phasor)
        new_plane_type = dLux.PlaneType.Pupil if self.inverse else \
                         dLux.PlaneType.Focal

        return tree_at(lambda wavefront: \
                       (wavefront.amplitude,
                        wavefront.phase,
                        wavefront.plane_type,
                        wavefront.pixel_scale),
                        wavefront,
                       (new_amplitude,
                        new_phase,
                        new_plane_type,
                        self.get_pixel_scale_out()))


class FixedSamplingPropagator(Propagator, ABC):
    """
    A propagator that implements the Fast Fourier Transform algorithm. This
    algorith has a fixed sampling in the output plane, at one fringe per pixel.
    Note the size of the 'fringe' in this context is similar to an optical
    fringe in that its angular size is calcualted via wavelength/wavefront
    diameter.

    These propagators are implemented using the jax.numpy.fft package, with the
    appropriate normalisations and pixel sizes tracked for optical propagation.
    """


    def __init__(self : Propagator, **kwargs) -> Propagator:
        """
        Constructor for FixedSampling propagators.
        """
        super().__init__(**kwargs)


    @abstractmethod
    def get_pixel_scale_out(self     : Propagator,
                           wavefront : Wavefront) -> Array:
        """
        Calculates the pixel scale in the output plane.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront that is being propagated.

        Returns
        -------
        pixel_scale : Array, meters/pixel or radians/pixel
            The pixel scale in the output plane, measured in meters or radians
            per pixel for Cartesian or Angular Wavefront respectively.
        """
        return


    def propagate(self       : Propagator,
                   wavefront : Wavefront) -> Array:
        """
        Propagates the wavefront by perfroming a Fast Fourier Transform.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate.

        Returns
        -------
        field : Array
            The normalised electric field phasor after the propagation.
        """
        if self.is_inverse():
            output_field = np.fft.fft2(np.fft.ifftshift(wavefront.get_phasor()))
        else:
            output_field = np.fft.fftshift(np.fft.ifft2(wavefront.get_phasor()))

        normalising_factor = self.is_inverse() / wavefront.get_npixels() + \
                             (1 - self.inverse) * wavefront.get_npixels()

        return output_field * normalising_factor


    def __call__(self : Propagator, wavefront : Wavefront) -> Wavefront:
        """
        Propagates the `Wavefront`.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate.

        Returns
        -------
        wavefront : Wavefront
            The wavefront with the optical layer applied.
        """
        phasor = self.propagate(wavefront)

        new_amplitude = np.abs(phasor)
        new_phase = np.angle(phasor)
        new_plane_type = dLux.PlaneType.Pupil if self.inverse else \
                         dLux.PlaneType.Focal

        return tree_at(lambda wavefront: \
                       (wavefront.amplitude,
                        wavefront.phase,
                        wavefront.plane_type,
                        wavefront.pixel_scale),
                        wavefront,
                       (new_amplitude,
                        new_phase,
                        new_plane_type,
                        self.get_pixel_scale_out(wavefront)))


class CartesianPropagator(Propagator, ABC):
    """
    A propagator class to store the focal_length parameter for cartesian
    propagations defined by a physical propagation distance defined as
    focal_length.

    Attributes
    ----------
    focal_length : Array, meters
        The focal_length of the lens/mirror this propagator represents.
    """
    focal_length : Array


    def __init__(self         : Propagator,
                 focal_length : Array,
                 **kwargs) -> Propagator:
        """
        Constructor for Cartesian propagators.

        Parameters
        ----------
        focal_length : Array, meters
            The focal_length of the lens/mirror this propagator represents.
        """
        super().__init__(**kwargs)
        self.focal_length = np.asarray(focal_length, dtype=float)
        assert self.focal_length.ndim == 0, ("focal_length must a scalar.")


    def get_focal_length(self : Propagator) -> Array:
        """
        Accessor for the focal_length parameter.

        Returns
        -------
        focal_length: Array, meters
            The focal length.
        """
        return self.focal_length


    def set_focal_length(self         : Propagator,
                         focal_length : bool) -> Propagator:
        """
        Mutator for the focal_length attribute.

        Parameters
        ----------
        focal_length : Array
            The new value for the focal_length attribute.

        Returns
        -------
        propagator : Propagator
            The Propagator with the updated focal_length attribute.
        """
        assert isinstance(focal_length, Array) and \
        focal_length.ndim == 0, ("focal_length must be scalar array.")
        return tree_at(lambda propagator: propagator.focal_length, self,
                                                                focal_length)


class AngularPropagator(Propagator, ABC):
    """
    A simple propagator class designed to be inhereited by propagators that
    operate on wavefronts defined in angular units in focal planes.
    """


    def __init__(self : Propagator, **kwargs) -> Propagator:
        """
        Constructor for Angular propagators.
        """
        super().__init__(**kwargs)


class FarFieldFresnel(Propagator, ABC):
    """
    A propagator class to store the propagation_shift parameter required for
    Far-Field fresnel propagations. These classes implement algorithms that use
    quadratic phase factors to better represent out-of-plane behaviour of
    wavefronts, close to the focal plane.

    Attributes
    ----------
    propagation_shift : Array, meters
        The shift in the propagation distance of the wavefront.
    """
    propagation_shift : Array


    def __init__(self, propagation_shift, **kwargs):
        """
        Constructor for FarFieldFresnel propagators.

        Parameters
        ----------
        propagation_shift : Array, meters
            The shift in the propagation distance of the wavefront.
        """
        super().__init__(**kwargs)
        self.propagation_shift  = np.asarray(propagation_shift,  dtype=float)
        assert propagation_shift.ndim == 0, ("propagation_shift must be scalar "
                                             "array.")


    def get_propagation_shift(self : Propagator) -> Array:
        """
        Accessor for the propagation_shift attribute.

        Returns
        -------
        propagation_shift : Array, meters
            The shift in the propagation distance of the wavefront.
        """
        return self.propagation_shift


    def set_propagation_shift(self        : Propagator,
                        propagation_shift : bool) -> Propagator:
        """
        Mutator for the propagation_shift attribute.

        Parameters
        ----------
        propagation_shift : Array
            The new value for the propagation_shift attribute.

        Returns
        -------
        propagator : Propagator
            The Propagator with the updated propagation_shift attribute.
        """
        assert isinstance(propagation_shift, Array) and \
        propagation_shift.ndim == 0, ("propagation_shift must be scalar array.")
        return tree_at(lambda propagator: propagator.propagation_shift,
                                                       self, propagation_shift)


########################
### Concrete Classes ###
########################
class CartesianMFT(CartesianPropagator, VariableSamplingPropagator):
    """
    A Propagator class designed to propagate a wavefront to a plane that is
    defined in cartesian units (ie meters/pixel), with a variable output
    sampling in that plane.
    """


    def __init__(self            : Propagator,
                 npixels_out     : int,
                 pixel_scale_out : Array,
                 focal_length    : Array,
                 inverse         : bool  = False,
                 shift           : Array = np.array([0., 0.]),
                 pixel_shift     : bool  = False,
                 name            : str   = 'CartesianMFT') -> Propagator:
        """
        Parameters
        ----------
        pixel_scale_out : Array, meters/pixel
            The pixel scale in the output plane, measured in meters per pixel.
        npixels_out : int
            The number of pixels in the output plane.
        focal_length : Array, meters
            The focal_length of the lens/mirror this propagator represents.
        inverse : bool = False
            Is this an 'inverse' propagation. Non-inverse propagations
            represents propagation from a pupil to a focal plane, and inverse
            represents propagation from a focal to a pupil plane.
        shift : Array = np.array([0., 0.])
            The (x, y) shift to apply to the wavefront in the output plane.
        pixel_shift : bool
            Should the shift value be considered in units of pixel, or in the
            physical units of the output plane (ie pixels or meters, radians).
        name : str = 'CartesianMFT'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(name            = name,
                         inverse         = inverse,
                         shift           = shift,
                         pixel_shift     = pixel_shift,
                         focal_length    = focal_length,
                         pixel_scale_out = pixel_scale_out,
                         npixels_out     = npixels_out)


    def get_nfringes(self      : Propagator,
                     wavefront : Wavefront) -> Array:
        """
        The number of diffraction fringes in the output plane.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront being propagated.

        Returns
        -------
        fringes : Array
            The number of diffraction fringes in the output plane.
        """
        size_in = wavefront.get_pixel_scale() * wavefront.get_npixels()
        size_out = self.get_pixel_scale_out() * self.get_npixels_out()
        return size_in * size_out / self.get_focal_length() / \
                                                    wavefront.get_wavelength()


    def get_shift(self : Propagator) -> Array:
        """
        Accessor for the shift parameter. Converts to units of pixels if the
        pixel_shift parameter is True.

        Returns
        -------
        shift : Array
            The (x, y) shift to apply to the wavefront throughout the
            propagation.
        """
        return super().get_shift() if self.is_pixel_shifted() else \
               super().get_shift() * self.focal_length / self.pixel_scale_out


class AngularMFT(AngularPropagator, VariableSamplingPropagator):
    """
    A Propagator class designed to propagate wavefronts, with pixel scale units
    defined in meters per pixel in pupil planes and radians/pixel in focal
    planes, with a variable output sampling in the output plane.
    """
    def __init__(self            : Propagator,
                 npixels_out     : int,
                 pixel_scale_out : Array,
                 inverse         : bool  = False,
                 shift           : Array = np.array([0., 0.]),
                 pixel_shift     : bool  = False,
                 name            : str   = 'AngularMFT') -> Propagator:
        """
        Parameters
        ----------
        pixel_scale_out : Array, radians/pixel, meters/pixel
            The pixel scale in the output plane, measured in meters per pixel in
            pupil planes and radians per pixel in focal planes.
        npixels_out : int
            The number of pixels in the output plane.
        inverse : bool = False
            Is this an 'inverse' propagation. Non-inverse propagations
            represents propagation from a pupil to a focal plane, and inverse
            represents propagation from a focal to a pupil plane.
        shift : Array = np.array([0., 0.])
            The (x, y) shift to apply to the wavefront in the output plane.
        pixel_shift : bool
            Should the shift value be considered in units of pixel, or in the
            physical units of the output plane (ie pixels or meters, radians).
        name : str = 'AngularMFT'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(name            = name,
                         inverse         = inverse,
                         shift           = shift,
                         pixel_shift     = pixel_shift,
                         pixel_scale_out = pixel_scale_out,
                         npixels_out     = npixels_out)


    def get_nfringes(self      : Propagator,
                     wavefront : Wavefront) -> Array:
        """
        The number of diffraction fringes in the output plane.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront being propagated.

        Returns
        -------
        fringes : Array
            The number of diffraction fringes in the output plane.
        """
        diameter = wavefront.get_npixels() * wavefront.get_pixel_scale()
        fringe_size = wavefront.get_wavelength() / diameter
        detector_size = self.npixels_out * self.pixel_scale_out
        nfringe = detector_size / fringe_size
        return nfringe


    def get_shift(self : Propagator) -> Array:
        """
        Accessor for the shift parameter. Converts to units of pixels if the
        pixel_shift parameter is True.

        Returns
        -------
        shift : Array
            The (x, y) shift to apply to the wavefront throughout the
            propagation.
        """
        return super().get_shift() if self.is_pixel_shifted() else \
               super().get_shift() / self.pixel_scale_out


class CartesianFFT(CartesianPropagator, FixedSamplingPropagator):
    """
    A Propagator class designed to propagate a wavefront to a plane using a
    Fast Fourier Transfrom, with the pixel scale units defined in meters/pixel.
    """


    def __init__(self         : Propagator,
                 focal_length : Array,
                 inverse      : bool = False,
                 name         : str  = 'CartesianFFT') -> Propagator:
        """
        Parameters
        ----------
        focal_length : Array, meters
            The focal_length of the lens/mirror this propagator represents.
        inverse : bool = False
            Is this an 'inverse' propagation. Non-inverse propagations
            represents propagation from a pupil to a focal plane, and inverse
            represents propagation from a focal to a pupil plane.
        name : str = 'CartesianFFT'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(name         = name,
                         inverse      = inverse,
                         focal_length = focal_length)


    def get_pixel_scale_out(self : Propagator, wavefront : Wavefront) -> Array:
        """
        The pixel scale in the output plane.

        Parameters
        ----------
        wavefront : Wavefront
            The `Wavefront` that is being propagted.

        Returns
        -------
        pixel_scale_out : Array, meters/pixel
            The pixel scale in the output plane.
        """
        return wavefront.get_wavelength() * self.get_focal_length() / \
               (wavefront.get_pixel_scale() * wavefront.get_npixels())


class AngularFFT(AngularPropagator, FixedSamplingPropagator):
    """
    A Propagator class designed to propagate a wavefront to a plane using a
    Fast Fourier Transfrom, with the pixel scale units defined in meters/pixel
    in pupil planes and radians/pixel in focal planes.
    """


    def __init__(self    : Propagator,
                 inverse : bool = False,
                 name    : str = 'AngularFFT') -> Propagator:
        """
        Constructor for the AngularFFT propagator.

        Parameters
        ----------
        inverse : bool = False
            Is this an 'inverse' propagation. Non-inverse propagations
            represents propagation from a pupil to a focal plane, and inverse
            represents propagation from a focal to a pupil plane.
        name : str = 'AngularFFT'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(name = name, inverse = inverse)


    def get_pixel_scale_out(self : Propagator, wavefront : Wavefront) -> Array:
        """
        Calculate the pixel scale in the output plane in units of radians per
        pixel.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront that is being propagated.

        Returns
        -------
        pixel_scale_out : Array, radians/pixel
            The pixel scale in the output plane.
        """
        return wavefront.get_wavelength() / wavefront.get_diameter()


class CartesianFresnel(FarFieldFresnel, CartesianMFT):
    """
    A propagator class to for Far-Field fresnel propagations. This classes
    implements algorithms that use quadratic phase factors to better represent
    out-of-plane behaviour of wavefronts, close to the focal plane. This class
    is designed to work on Cartesian wavefronts, ie pixel units are in
    meters/pixel in the output plane.
    """


    def __init__(self              : Propagator,
                 npixels_out       : Array,
                 pixel_scale_out   : Array,
                 focal_length      : Array,
                 propagation_shift : Array,
                 inverse           : bool  = False,
                 shift             : Array = np.array([0., 0.]),
                 pixel_shift       : bool  = False,
                 name              : str   = 'CartesianFresnel') -> Propagator:
        """
        Constructor for the CartesianFresnel propagator

        Parameters
        ----------
        pixel_scale_out : Array, meters/pixel
            The pixel scale in the output plane, measured in meters per pixel.
        npixels_out : int
            The number of pixels in the output plane.
        focal_length : Array, meters
            The focal_length of the lens/mirror this propagator represents.
        propagation_shift : Array, meters
            The shift in the propagation distance of the wavefront.
        inverse : bool = False
            Is this an 'inverse' propagation. Non-inverse propagations
            represents propagation from a pupil to a focal plane, and inverse
            represents propagation from a focal to a pupil plane.
        shift : Array = np.array([0., 0.])
            The (x, y) shift to apply to the wavefront in the output plane.
        pixel_shift : bool
            Should the shift value be considered in units of pixel, or in the
            physical units of the output plane (ie pixels or meters, radians).
        name : str = 'CartesianFresnel'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(name               = name,
                         inverse            = inverse,
                         shift              = shift,
                         pixel_shift        = pixel_shift,
                         focal_length       = focal_length,
                         pixel_scale_out    = pixel_scale_out,
                         npixels_out        = npixels_out,
                         propagation_shift  = propagation_shift)


    def get_nfringes(self      : Propagator,
                     wavefront : Wavefront) -> Array:
        """
        The number of diffraction fringes in the output plane.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront being propagated.

        Returns
        -------
        fringes : Array
            The number of diffraction fringes in the output plane.
        """
        propagation_distance = self.get_focal_length() + self.get_propagation_shift()
        focal_ratio = self.get_focal_length() / propagation_distance

        size_in = wavefront.get_pixel_scale() * \
            wavefront.get_npixels()
        size_out = self.get_pixel_scale_out() * \
            self.get_npixels_out()

        return size_in * size_out / self.focal_length / \
                   wavefront.wavelength * focal_ratio


    def quadratic_phase(self          : Propagator,
                        x_coordinates : Array,
                        y_coordinates : Array,
                        wavelength    : Array,
                        distance      : Array) -> Array:
        """
        A convinience function for calculating quadratic phase factors.

        Parameters
        ----------
        x_coordinates : Array
            The x coordinates of the pixels in meters. This will be different
            in the plane of propagation and the initial plane.
        y_coordinates : Array
            The y coordinates of the pixels in meters. This will be different
            in the plane of propagation and the initial plane.
        wavelength : Array, meters
            The wavelength of the wavefront.
        distance : Array, meters
            The distance that is to be propagated in meters.

        Returns
        -------
        quadratic_phase : Array
            A set of phase factors that are useful in optical calculations.
        """
        wavenumber = 2 * np.pi / wavelength
        radial_coordinates = np.hypot(x_coordinates, y_coordinates)
        return np.exp(0.5j * wavenumber * radial_coordinates ** 2 / distance)


    def propagate(self : Propagator, wavefront : Wavefront) -> Array:
        """
        Propagates the wavefront from the input plane to the output plane using
        a Matrix Fourier Transform.

        TODO: Set plane type to intermediate

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate.

        Returns
        -------
        field : Array
            The normalised electric field phasor after the propagation.
        """
        # See gihub issue #52
        offsets = self.get_shift()

        input_positions = wavefront.get_pixel_coordinates()
        output_positions = dLux.utils.coordinates.get_pixel_coordinates(
                                self.get_npixels_out(),
                                self.get_pixel_scale_out())

        propagation_distance = self.get_focal_length() + self.get_propagation_shift()

        field = wavefront.get_phasor()
        field *= self.quadratic_phase(*input_positions,
            wavefront.get_wavelength(), - self.get_focal_length())
        field *= self.quadratic_phase(*input_positions, 
            wavefront.get_wavelength(), propagation_distance)

        wavefront = wavefront.update_phasor(np.abs(field), np.angle(field))

        field = super().propagate(wavefront)
        field *= wavefront.transfer_function(propagation_distance)
        field *= self.quadratic_phase(*output_positions,
            wavefront.get_wavelength(), propagation_distance)
        return field


# TODO: Implement eventually

# class AngularFresnel(FarFieldFresnel, AngularMFT):
#     """
#     Propagates an AngularWavefront in the Fresnel approximation.

#     Attributes
#     ----------
#     """
#     pass

