from __future__ import annotations
import jax.numpy as np
from jax import Array
from equinox import tree_at
from abc import ABC, abstractmethod
# from dLux.utils.coordinates import get_pixel_positions
import dLux


__all__ = ["CartesianMFT", "AngularMFT", "ShiftedCartesianMFT", 
    "ShiftedAngularMFT", "CartesianFFT", "AngularFFT", "CartesianFresnel",
    "PropagatorFactory"]


OpticalLayer = lambda : dLux.optical_layers.OpticalLayer

########################
### Abstract Classes ###
########################
class Propagator(OpticalLayer()):
    """
    An abstract class to store the various properties of the propagation of
    some wavefront.

    Attributes
    ----------
    inverse : bool
        Should the propagation be performed in the inverse direction.
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
            Should the propagation be performed in the inverse direction.
        """
        self.inverse = bool(inverse)
        super().__init__(**kwargs)


class VariableSamplingPropagator(Propagator):
    """
    A propagator that implements the Soummer et. al. 2007 MFT algorithm
    allowing variable sampling in the outuput plane rather than the fixed
    sampling enforced by Fast Fourier Transforms(FFTs).

    Attributes
    ----------
    npixels : int
        The number of pixels in the output plane.
    pixel_scale : Array, meters/pixel or radians/pixel
        The pixel scale in the output plane, measured in meters or radians per
        pixel for Cartesian or Angular Wavefront respectively.
    inverse : bool
        Should the propagation be performed in the inverse direction.
    """
    npixels     : int
    pixel_scale : Array


    def __init__(self        : Propagator,
                 pixel_scale : Array,
                 npixels     : int,
                 inverse     : bool = False,
                 **kwargs) -> Propagator:
        """
        Constructor for VariableSampling propagators.

        Parameters
        ----------
        npixels : int
            The number of pixels in the output plane.
        pixel_scale : Array, meters/pixel or radians/pixel
            The pixel scale in the output plane, measured in meters or radians
            per pixel for Cartesian or Angular Wavefront respectively.
        inverse : bool = False
            Should the propagation be performed in the inverse direction.
        """
        super().__init__(inverse=inverse, **kwargs)
        self.pixel_scale = np.asarray(pixel_scale, dtype=float)
        self.npixels     = int(npixels)
        if self.pixel_scale.ndim != 0:
            raise TypeError('pixel_scale must be a scalar.')


class ShiftedPropagator(VariableSamplingPropagator):
    """
    A propagator that implements the Soummer et. al. 2007 MFT algorithm
    allowing variable sampling in the outuput plane rather than the fixed
    sampling enforced by Fast Fourier Transforms(FFTs), as well as a shift
    in the center of the output plane.

    Attributes
    ----------
    npixels : int
        The number of pixels in the output plane.
    pixel_scale : Array, meters/pixel or radians/pixel
        The pixel scale in the output plane, measured in meters or radians per
        pixel for Cartesian or Angular Wavefront respectively.
    shift : Array
        The (x, y) shift to apply to the wavefront in the output plane.
    pixel : bool
        Should the shift value be considered in units of pixels, or in the
        physical units of the output plane (ie pixels or meters, radians). True
        interprets the shift value in pixel units.
    inverse : bool
        Should the propagation be performed in the inverse direction.
    """
    shift       : Array
    pixel       : bool


    def __init__(self        : Propagator,
                 pixel_scale : Array,
                 npixels     : int,
                 shift       : Array = np.zeros(2),
                 pixel       : bool  = False,
                 inverse     : bool = False,
                 **kwargs) -> Propagator:
        """
        Constructor for VariableSampling propagators.

        Parameters
        ----------
        npixels : int
            The number of pixels in the output plane.
        pixel_scale : Array, meters/pixel or radians/pixel
            The pixel scale in the output plane, measured in meters or radians
            per pixel for Cartesian or Angular Wavefront respectively.
        shift : Array = np.array([0., 0.])
            The (x, y) shift to apply to the wavefront in the output plane.
        pixel : bool = False
            Should the shift value be considered in units of pixel, or in the
            physical units of the output plane (ie pixels or meters, radians). =
            True interprets the shift value in pixel units.
        inverse : bool = False
            Should the propagation be performed in the inverse direction.
        """
        self.shift = np.asarray(shift, dtype=float)
        self.pixel = bool(pixel)
        if self.shift.shape != (2,):
            raise TypeError('shift must be an array of shape (2,).')

        super().__init__(pixel_scale=pixel_scale, 
                         npixels=npixels, 
                         inverse=inverse, 
                         **kwargs)


class FixedSamplingPropagator(Propagator):
    """
    A propagator that implements the Fast Fourier Transform algorithm. This
    algorith has a fixed sampling in the output plane, at one fringe per pixel.
    Note the size of the 'fringe' in this context is similar to an optical
    fringe in that its angular size is calcualted via wavelength/wavefront
    diameter.

    These propagators are implemented using the jax.numpy.fft package, with the
    appropriate normalisations and pixel sizes tracked for optical propagation.

    Attributes
    ----------
    pad : int
        The padding factory to apply to the input wavefront before performing
        the FFT.
    inverse : bool
        Should the propagation be performed in the inverse direction.
    """
    pad : int


    def __init__(self    : Propagator, 
                 pad     : int = 2, 
                 inverse : bool = False,
                 **kwargs) -> Propagator:
        """
        Constructor for FixedSampling propagators.
        """
        super().__init__(inverse=inverse, **kwargs)
        self.pad = int(pad)


class CartesianPropagator(Propagator):
    """
    A propagator class to store the focal_length parameter for cartesian
    propagations defined by a physical propagation distance defined as
    focal_length.

    Attributes
    ----------
    focal_length : Array, meters
        The focal_length of the lens/mirror this propagator represents.
    inverse : bool
        Should the propagation be performed in the inverse direction.
    """
    focal_length : Array


    def __init__(self         : Propagator,
                 focal_length : Array,
                 inverse      : bool = False,
                 **kwargs) -> Propagator:
        """
        Constructor for Cartesian propagators.

        Parameters
        ----------
        focal_length : Array, meters
            The focal_length of the lens/mirror this propagator represents.
        inverse : bool = False
            Should the propagation be performed in the inverse direction.
        """
        super().__init__(inverse=inverse, **kwargs)
        self.focal_length = np.asarray(focal_length, dtype=float)
        if self.focal_length.ndim != 0:
            raise TypeError('focal_length must be a scalar.')


class AngularPropagator(Propagator):
    """
    A simple propagator class designed to be inhereited by propagators that
    operate on wavefronts defined in angular units in focal planes.

    Attributes
    ----------
    inverse : bool
        Should the propagation be performed in the inverse direction.
    """


    def __init__(self    : Propagator, 
                 inverse : bool = False,
                 **kwargs) -> Propagator:
        """
        Constructor for Angular propagators.

        Parameters
        ----------
        inverse : bool = False
            Should the propagation be performed in the inverse direction.
        """
        super().__init__(inverse=inverse, **kwargs)


class FarFieldFresnel(ShiftedPropagator):
    """
    A propagator class to store the focal_shift parameter required for
    Far-Field fresnel propagations. These classes implement algorithms that use
    quadratic phase factors to better represent out-of-plane behaviour of
    wavefronts, close to the focal plane.

    Attributes
    ----------
    focal_shift : Array, meters
        The shift in the propagation distance of the wavefront.
    inverse : bool
        Should the propagation be performed in the inverse direction.
    """
    focal_shift : Array


    def __init__(self        : Propagator, 
                 focal_shift : Array, 
                 **kwargs) -> Propagator:
        """
        Constructor for FarFieldFresnel propagators.

        Parameters
        ----------
        focal_shift : Array, meters
            The shift in the propagation distance of the wavefront.
        """
        super().__init__(**kwargs)
        self.focal_shift  = np.asarray(focal_shift, dtype=float)
        if self.focal_shift.ndim != 0:
            raise TypeError('focal_shift must be a scalar.')


########################
### Concrete Classes ###
########################
class ShiftedCartesianMFT(CartesianPropagator, ShiftedPropagator):
    """
    A Propagator class designed to propagate a wavefront to a plane that is
    defined in cartesian units (ie meters/pixel), with a variable output
    sampling in that plane.

    Attributes
    ----------
    npixels : int
        The number of pixels in the output plane.
    pixel_scale : Array, meters/pixel
        The pixel scale in the output plane, measured in meters per pixel.
    focal_length : Array, meters
        The focal_length of the lens/mirror this propagator represents.
    shift : Array
        The (x, y) shift to apply to the wavefront in the output plane.
    pixel : bool
        Should the shift value be considered in units of pixels, or in the
        physical units of the output plane (ie pixels or meters, radians). True
        interprets the shift value in pixel units.
    inverse : bool
        Should the propagation be performed in the inverse direction.
    """


    def __init__(self         : Propagator,
                 npixels      : int,
                 pixel_scale  : Array,
                 focal_length : Array,
                 shift        : Array = np.array([0., 0.]),
                 pixel        : bool  = False,
                 inverse      : bool  = False) -> Propagator:
        """
        Parameters
        ----------
        npixels : int
            The number of pixels in the output plane.
        pixel_scale : Array, meters/pixel
            The pixel scale in the output plane, measured in meters per pixel.
        focal_length : Array, meters
            The focal_length of the lens/mirror this propagator represents.
        shift : Array = np.array([0., 0.])
            The (x, y) shift to apply to the wavefront in the output plane.
        pixel : bool = False
            Should the shift value be considered in units of pixel, or in the
            physical units of the output plane (ie pixels or meters, radians).
        inverse : bool = False
            Should the propagation be performed in the inverse direction.
        """
        super().__init__(pixel        = pixel,
                         shift        = shift,
                         focal_length = focal_length,
                         pixel_scale  = pixel_scale,
                         npixels      = npixels,
                         inverse      = inverse)
        

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
        if self.inverse:
            return wavefront.shifted_IMFT(self.npixels, self.pixel_scale,
                self.shift, self.focal_length, self.pixel)
        else:
            return wavefront.shifted_MFT(self.npixels, self.pixel_scale,
                self.shift, self.focal_length, self.pixel)


class CartesianMFT(CartesianPropagator, VariableSamplingPropagator):
    """
    A Propagator class designed to propagate a wavefront to a plane that is
    defined in cartesian units (ie meters/pixel).

    Attributes
    ----------
    npixels : int
        The number of pixels in the output plane.
    pixel_scale : Array, meters/pixel
        The pixel scale in the output plane, measured in meters per pixel.
    focal_length : Array, meters
        The focal_length of the lens/mirror this propagator represents.
    inverse : bool
        Should the propagation be performed in the inverse direction.
    """


    def __init__(self         : Propagator,
                 npixels      : int,
                 pixel_scale  : Array,
                 focal_length : Array,
                 inverse      : bool  = False) -> Propagator:
        """
        Parameters
        ----------
        npixels : int
            The number of pixels in the output plane.
        pixel_scale : Array, meters/pixel
            The pixel scale in the output plane, measured in meters per pixel.
        focal_length : Array, meters
            The focal_length of the lens/mirror this propagator represents.
        inverse : bool = False
            Should the propagation be performed in the inverse direction.
        """
        super().__init__(focal_length = focal_length,
                         pixel_scale  = pixel_scale,
                         npixels      = npixels,
                         inverse      = inverse)
        

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
        if self.inverse:
            return wavefront.IMFT(self.npixels, self.pixel_scale,
                focal_length=self.focal_length)
        else:
            return wavefront.MFT(self.npixels, self.pixel_scale,
                focal_length=self.focal_length)
      

class ShiftedAngularMFT(AngularPropagator, ShiftedPropagator):
    """
    A Propagator class designed to propagate wavefronts, with pixel scale units
    defined in meters per pixel in pupil planes and radians/pixel in focal
    planes, with a variable output sampling in the output plane with a shift
    in the center of the output plane.

    Attributes
    ----------
    npixels : int
        The number of pixels in the output plane.
    pixel_scale : Array, meters/pixel or radians/pixel
        The pixel scale in the output plane, measured in meters per pixel in
        pupil plane and radians per pixel in focal planes.
    shift : Array
        The (x, y) shift to apply to the wavefront in the output plane.
    pixel : bool
        Should the shift value be considered in units of pixels, or in the
        physical units of the output plane (ie pixels or meters, radians). True
        interprets the shift value in pixel units.
    inverse : bool
        Should the propagation be performed in the inverse direction.
    """


    def __init__(self        : Propagator,
                 npixels     : int,
                 pixel_scale : Array,
                 shift       : Array = np.zeros(2),
                 pixel       : bool  = False,
                 inverse     : bool  = False) -> Propagator:
        """
        Parameters
        ----------
        npixels : int
            The number of pixels in the output plane.
        pixel_scale : Array, radians/pixel, meters/pixel
            The pixel scale in the output plane, measured in meters per pixel in
            pupil planes and radians per pixel in focal planes.
        shift : Array = np.array([0., 0.])
            The (x, y) shift to apply to the wavefront in the output plane.
        pixel : bool = False
            Should the shift value be considered in units of pixel, or in the
            physical units of the output plane (ie pixels or meters, radians).
        inverse : bool = False
            Should the propagation be performed in the inverse direction.
        """
        super().__init__(shift       = shift,
                         pixel       = pixel,
                         pixel_scale = pixel_scale,
                         npixels     = npixels,
                         inverse     = inverse)


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
        if self.inverse:
            return wavefront.shifted_IMFT(self.npixels, self.pixel_scale,
                self.shift, pixel=self.pixel)
        else:
            return wavefront.shifted_MFT(self.npixels, self.pixel_scale,
                self.shift, pixel=self.pixel)


class AngularMFT(AngularPropagator, VariableSamplingPropagator):
    """
    A Propagator class designed to propagate wavefronts, with pixel scale units
    defined in meters per pixel in pupil planes and radians/pixel in focal
    planes, with a variable output sampling in the output plane.

    Attributes
    ----------
    npixels : int
        The number of pixels in the output plane.
    pixel_scale : Array, meters/pixel or radians/pixel
        The pixel scale in the output plane, measured in meters per pixel in
        pupil plane and radians per pixel in focal planes.
    inverse : bool
        Should the propagation be performed in the inverse direction.
    """


    def __init__(self        : Propagator,
                 npixels     : int,
                 pixel_scale : Array,
                 inverse     : bool  = False) -> Propagator:
        """
        Parameters
        ----------
        npixels : int
            The number of pixels in the output plane.
        pixel_scale : Array, radians/pixel, meters/pixel
            The pixel scale in the output plane, measured in meters per pixel in
            pupil planes and radians per pixel in focal planes.
        inverse : bool = False
            Should the propagation be performed in the inverse direction.
        """
        super().__init__(pixel_scale = pixel_scale,
                         npixels     = npixels,
                         inverse     = inverse)


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
        if self.inverse:
            return wavefront.IMFT(self.npixels, self.pixel_scale)
        else:
            return wavefront.MFT(self.npixels, self.pixel_scale)


class CartesianFFT(CartesianPropagator, FixedSamplingPropagator):
    """
    A Propagator class designed to propagate a wavefront to a plane using a
    Fast Fourier Transfrom, with the pixel scale units defined in meters/pixel.

    Attributes
    ----------
    focal_length : Array, meters
        The focal_length of the lens/mirror this propagator represents.
    pad : int
        The amount of padding to apply to the wavefront before propagating.
    inverse : bool
        Should the propagation be performed in the inverse direction.
    """


    def __init__(self         : Propagator,
                 focal_length : Array,
                 pad          : int = 2,
                 inverse      : bool = False) -> Propagator:
        """
        Parameters
        ----------
        focal_length : Array, meters
            The focal_length of the lens/mirror this propagator represents.
        pad : int = 2
            The amount of padding to apply to the wavefront before propagating.
        inverse : bool = False
            Should the propagation be performed in the inverse direction.
        """
        super().__init__(pad          = pad,
                         focal_length = focal_length,
                         inverse      = inverse)
    

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
        if self.inverse:
            return wavefront.IFFT(self.pad, self.focal_length)
        else:
            return wavefront.FFT(self.pad, self.focal_length)


class AngularFFT(AngularPropagator, FixedSamplingPropagator):
    """
    A Propagator class designed to propagate a wavefront to a plane using a
    Fast Fourier Transfrom, with the pixel scale units defined in meters/pixel
    in pupil planes and radians/pixel in focal planes.

    Attributes
    ----------
    pad : int
        The amount of padding to apply to the wavefront before propagating.
    inverse : bool
        Should the propagation be performed in the inverse direction.
    """


    def __init__(self : Propagator,
                 pad  : int = 2,
                 inverse : bool = False) -> Propagator:
        """
        Constructor for the AngularFFT propagator.

        Parameters
        ----------
        pad : int = 2
            The amount of padding to apply to the wavefront before propagating.
        inverse : bool = False
            Should the propagation be performed in the inverse direction.
        """
        super().__init__(pad=pad, inverse=inverse)


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
        if self.inverse:
            return wavefront.IFFT(self.pad)
        else:
            return wavefront.FFT(self.pad)


class CartesianFresnel(FarFieldFresnel, CartesianMFT):
    """
    A propagator class to for Far-Field fresnel propagations. This classes
    implements algorithms that use quadratic phase factors to better represent
    out-of-plane behaviour of wavefronts, close to the focal plane. This class
    is designed to work on Cartesian wavefronts, ie pixel units are in
    meters/pixel in the output plane.

    Attributes
    ----------
    npixels : int
        The number of pixels in the output plane.
    pixel_scale : Array, meters/pixel
        The pixel scale in the output plane, measured in meters per pixel.
    focal_length : Array, meters
        The focal_length of the lens/mirror this propagator represents.
    focal_shift : Array, meters
        The shift in the propagation distance of the wavefront.
    shift : Array
        The (x, y) shift to apply to the wavefront in the output plane.
    pixel : bool
        Should the shift value be considered in units of pixels, or in the
        physical units of the output plane (ie pixels or meters, radians). True
        interprets the shift value in pixel units.
    inverse : bool
        Should the propagation be performed in the inverse direction.
    """


    def __init__(self         : Propagator,
                 npixels      : Array,
                 pixel_scale  : Array,
                 focal_length : Array,
                 focal_shift  : Array,
                 shift        : Array = np.zeros(2),
                 pixel        : bool  = False,
                 inverse      : bool  = False) -> Propagator:
        """
        Constructor for the CartesianFresnel propagator

        Parameters
        ----------
        pixel_scale : Array, meters/pixel
            The pixel scale in the output plane, measured in meters per pixel.
        npixels : int
            The number of pixels in the output plane.
        focal_length : Array, meters
            The focal_length of the lens/mirror this propagator represents.
        focal_shift : Array, meters
            The shift in the propagation distance of the wavefront.
        shift : Array = np.array([0., 0.])
            The (x, y) shift to apply to the wavefront in the output plane.
        pixel : bool = False
            Should the shift value be considered in units of pixel, or in the
            physical units of the output plane (ie pixels or meters, radians).
        inverse : bool = False
            Should the propagation be performed in the inverse direction.
        """
        if inverse:
            raise NotImplementedError('Inverse propagation not implemented '
                'for CartesianFresnel.')
        super().__init__(shift        = shift,
                         pixel        = pixel,
                         focal_length = focal_length,
                         pixel_scale  = pixel_scale,
                         npixels      = npixels,
                         focal_shift  = focal_shift,
                         inverse      = inverse)


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
        return wavefront.shifted_fresnel_prop(self.npixels, self.pixel_scale,
            self.shift, self.focal_length, self.focal_shift, self.pixel)


# TODO: Implement eventually
# class AngularFresnel(FarFieldFresnel, AngularMFT):
#     """
#     Propagates an AngularWavefront in the Fresnel approximation.

#     Attributes
#     ----------
#     """
#     pass


###############
### Factory ###
###############
class PropagatorFactory():
    """
    This class is not actually ever instatiated, but is rather a class used to 
    give a simple constructor interface that is used to construct the most
    commonly used propagators. The constructor is used to determine which
    propagator to construct, and then the constructor for that propagator is
    called with the remaining arguments.
    """
    def __new__(cls              : ApertureFactory, 
                npixels          : int,
                pixel_scale      : Array,
                inverse          : bool = False,
                shift            : Array = np.zeros(2),
                pixel            : bool = False,
                focal_length     : float = None,
                focal_shift      : Array = 0.):
        """
        Constructs a new Propagator object.

        Parameters
        ----------
        npixels : int
            The number of pixels in the output plane.
        pixel_scale : Array, meters/pixel
            The pixel scale in the output plane, measured in radians per pixel
            if focal_length is None, else meters per pixel.
        inverse : bool = False
            Should the propagation be performed in the inverse direction.
        shift : Array = np.zeros(2)
            The (x, y) shift to apply to the wavefront in the output plane.
        pixel : bool = False
            Should the shift value be considered in units of pixel, or in the
            physical units of the output plane. If True the shift is taken in
            pixels, otherwise it is taken in the native units of the output 
            plane.
        focal_length : Array = None, meters
            The focal_length of the lens/mirror this propagator represents.
        focal_shift : Array = 0, meters
            The shift in the propagation distance of the wavefront.
        """
        # Type checking
        if not isinstance(npixels, int):
            raise TypeError('npixels must be an integer.')
        if not isinstance(inverse, bool):
            raise TypeError('inverse must be a boolean.')
        if not isinstance(pixel, bool):
            raise TypeError('pixel must be a boolean.')
        
        # Fresnel Propagators
        if focal_shift != 0.:
            if focal_length is None:
                raise ValueError('A focal length must be supplied if '
                    'focal_shift is non-zero.')
            
            return CartesianFresnel(npixels, pixel_scale, focal_length, 
                focal_shift, shift, pixel, inverse)
        
        # Angular Propagators
        if focal_length is None:
            if (shift == np.zeros(2)).all():
                return AngularMFT(npixels, pixel_scale, inverse)
            else:
                return ShiftedAngularMFT(npixels, pixel_scale, shift, pixel,
                    inverse)

        # Cartesian Propagators
        else:
            if (shift == np.zeros(2)).all():
                return CartesianMFT(npixels, pixel_scale, focal_length, 
                    inverse)
            else:
                return ShiftedCartesianMFT(npixels, pixel_scale, focal_length,
                    shift, pixel, inverse)


# TODO: Asserts