from __future__ import annotations
import jax.numpy as np
from jax import vmap, Array
from zodiax import Base
from dLux.utils.coordinates import get_pixel_positions
from dLux.utils.interpolation import interpolate_field, rotate_field
import dLux


__all__ = ["Wavefront", "FresnelWavefront"]


Aberration = lambda : dLux.optics.AberrationLayer
Aperture = lambda : dLux.apertures.ApertureLayer
TransmissiveLayer = lambda : dLux.optics.TransmissiveLayer
AberrationLayer = lambda : dLux.optics.AberrationLayer
Propagator = lambda : dLux.propagators.Propagator

class Wavefront(Base):
    """
    A class representing some wavefront, designed to track the various
    parameters such as wavelength, pixel_scale, amplitude and phase, as well as
    a helper parmeter, plane_type.

    All wavefronts currently only support square amplitude and phase arrays.

    Attributes
    ----------
    wavelength : float, meters
        The wavelength of the `Wavefront`.
    amplitude : Array, power
        The electric field amplitude of the `Wavefront`.
    phase : Array, radians
        The electric field phase of the `Wavefront`.
    pixel_scale : float, meters/pixel or radians/pixel
        The physical dimensions of the pixels representing the wavefront. This
        can be in units of either meters per pixel or radians per pixel
        depending on both the plane type and the wavfront type (Cartesian or
        Angular).
    plane_type : enum.IntEnum.PlaneType
        The current plane type of wavefront, can be Pupil, Focal or
        Intermediate.
    """
    wavelength  : Array
    pixel_scale : Array
    amplitude   : Array
    phase       : Array
    plane       : str
    units       : str


    def __init__(self       : Wavefront,
                 npixels    : int,
                 diameter   : Array,
                 wavelength : Array) -> Wavefront:
        """
        Constructor for the wavefront.

        Parameters
        ----------
        wavelength : float, meters
            The wavelength of the `Wavefront`.
        pixel_scale : float, meters/pixel
            The physical dimensions of each square pixel.
        amplitude : Array, power
            The electric field amplitude of the `Wavefront`.
        phase : Array, radians
            The electric field phase of the `Wavefront`.
        plane_type : enum.IntEnum.PlaneType
            The current plane of wavefront, can be Pupil, Focal.
        """
        self.wavelength  = np.asarray(wavelength,  dtype=float)
        self.pixel_scale = np.asarray(diameter/npixels, dtype=float)
        self.amplitude   = np.ones((npixels, npixels), dtype=float)
        self.phase       = np.zeros((npixels, npixels), dtype=float)
        
        # Input checks
        assert self.wavelength.shape == (), \
        ("wavelength must be a scalar Array.")
        assert self.pixel_scale.shape == (), \
        ("pixel_scale must be a scalar Array.")

        # Always initialised in Pupil plane with Cartesian Coords
        self.plane = 'Pupil'
        self.units = 'Cartesian'
        

    ########################
    ### Getter Functions ###
    ########################
    @property
    def diameter(self : Wavefront) -> Array:
        """
        Returns the current wavefront diameter calulated using the pixel scale
        and number of pixels.

        Returns
        -------
        diameter : Array, meters or radians
            The current diameter of the wavefront.
        """
        return self.npixels * self.pixel_scale


    @property
    def npixels(self : Wavefront) -> int:
        """
        Returns the side length of the arrays currently representing the
        wavefront. Taken from the amplitude array.

        Returns
        -------
        pixels : int
            The number of pixels that represent the `Wavefront`.
        """
        return self.amplitude.shape[-1]


    @property
    def real(self : Wavefront) -> Array:
        """
        Returns the real component of the `Wavefront`.

        Returns
        -------
        wavefront : Array
            The real component of the `Wavefront` phasor.
        """
        return self.amplitude * np.cos(self.phase)


    @property
    def imaginary(self : Wavefront) -> Array:
        """
        Returns the imaginary component of the `Wavefront`.

        Returns
        -------
        wavefront : Array
            The imaginary component of the `Wavefront` phasor.
        """
        return self.amplitude * np.sin(self.phase)


    @property
    def phasor(self : Wavefront) -> Array:
        """
        The electric field phasor described by this Wavefront in complex form.

        Returns
        -------
        field : Array
            The electric field phasor of the wavefront.
        """
        return self.amplitude * np.exp(1j * self.phase)


    @property
    def psf(self : Wavefront) -> Array:
        """
        Calculates the Point Spread Function (PSF), ie the squared modulus
        of the complex wavefront.

        Returns
        -------
        psf : Array
            The PSF of the wavefront.
        """
        return self.amplitude ** 2


    @property
    def pixel_coordinates(self : Wavefront) -> Array:
        """
        Returns the physical positions of the wavefront pixels in meters.

        Returns
        -------
        pixel_positions : Array
            The coordinates of the centers of each pixel representing the
            wavefront.
        """
        return get_pixel_positions((self.npixels, self.npixels), 
                                   (self.pixel_scale, self.pixel_scale))


    @property
    def wavenumber(self : Wavefront) -> Array:
        """
        Returns the wavenumber of the wavefront (2 * pi / wavelength).

        Returns
        -------
        wavenumber : Array, 1/meters
            The wavenumber of the wavefront.
        """
        return 2 * np.pi / self.wavelength


    #####################
    ### Magic Methods ###
    #####################
    def __add__(self, other):
        """
        Adds an OPD to the wavefront.
        """        
        # Array based inputs
        if isinstance(other, Array):
            return self.add('phase', other * self.wavenumber)
        
        # Layer inputs
        elif isinstance(other, AberrationLayer()):
            return other(self)
        
        # Other
        else:
            raise TypeError("Can only add an array or AberrationLayer to "
            f"Wavefront. Got: {type(other)}.")
    

    def __iadd__(self, other):
        """

        """
        return self.__add__(other)
    

    def __mul__(self, other):
        """
        Multiplies a wavefront by an array ApertureLayer or Proapgator.
        """
        # Array based inputs
        if isinstance(other, Array):
            return self.multiply('amplitude', other)

        # Aperture Layer inputs
        elif isinstance(other, Aperture()):
            return other(self)
        
        # Aberration Layer inputs
        elif isinstance(other, AberrationLayer()):
            return other(self)
        
        # Propagators
        elif isinstance(other, Propagator()):
            return other(self)
        
        # Wavefronts
        elif isinstance(other, Wavefront):
            transmission = other.amplitude * self.amplitude
            phase = other.phase + self.phase
            return self.set(['amplitude', 'phase'], [transmission, phase])
        
        # Other
        else:
            raise TypeError("Can only multiply an array or ApertureLayer to "
            f"Wavefront. Got: {type(other)}.")


    def __imul__(self, other):
        return self.__mul__(other)
    

    #############################
    ### Propagation Functions ###
    #############################
    def _FFT_output(self : Wavefront, focal_length : Array = None) -> tuple:
        """
        
        """
        if focal_length is None:
            units = 'Angular'
            pixel_scale =  self.wavelength / self.diameter
        else:
            units = 'Cartesian'
            pixel_scale = focal_length * self.wavelength / self.diameter

            # Check for invalid propagation
            if self.units == 'Angular':
                raise ValueError("focal_length can not be specific when"
                    "propagating from a Focal plane with angular units.")
        
        if self.plane == 'Focal':
            plane = 'Pupil'
            propagator = lambda phasor: \
                np.fft.fft2(np.fft.ifftshift(phasor)) / phasor.shape[-1]
            units = 'Cartesian' # Always cartesian in pupil plane
        elif self.plane == 'Pupil':
            plane = 'Focal'
            propagator = lambda phasor: \
                np.fft.fftshift(np.fft.ifft2(phasor)) * phasor.shape[-1]
        else:

            # Check for invalid propagation
            raise ValueError("plane must be either 'Pupil' or 'Focal'. "
                f"Got {self.plane}")
        
        return plane, units, pixel_scale, propagator


    def FFT(self : Wavefront, 
            pad : int = 2,
            focal_length : Array = None) -> Wavefront:
        """
        Propagates the wavefront by perfroming a Fast Fourier Transform.

        Parameters
        ----------
        focal_length : float, meters
            The focal length of the propagation. If None, the propagation is
            assumed to be angular.

        Returns
        -------
        wavefront : Wavefront
            The propagated wavefront.
        """
        # Calculate
        plane, units, pixel_scale, propagator = self._FFT_output(focal_length)

        # Pad must be int
        npixels = (self.npixels * (pad - 1)) // 2
        ampltide = np.pad(self.amplitude, npixels)
        phase = np.pad(self.phase, npixels)
        phasor = propagator(ampltide * np.exp(1j * phase))

        # Return new wavefront
        return self.set(['amplitude', 'phase', 'pixel_scale', 'plane', 'units'],
            [np.abs(phasor), np.angle(phasor), pixel_scale, plane, units])


    def _MFT_output(self : Wavefront, focal_length : Array = None) -> tuple:
        """
        
        """
        # Get units
        if focal_length is None:
            units = 'Angular'
        else:
            units = 'Cartesian'

            # Check for invalid propagation
            if self.units == 'Angular':
                raise ValueError("focal_length can not be specific when"
                    "propagating from a Focal plane with angular units.")
        
        # Get Plane
        if self.plane == 'Focal':
            plane = 'Pupil'
            units = 'Cartesian' # Always cartesian in pupil plane
        elif self.plane == 'Pupil':
            plane = 'Focal'
        else:

            # Check for invalid propagation
            raise ValueError("plane must be either 'Pupil' or 'Focal'. "
                f"Got {self.plane}")
        
        return plane, units


    def _nfringes(self : Wavefront, 
                  npixels : int,
                  pixel_scale : Array,
                  focal_length : Array = None) -> Array:
        """

        """
        output_size = npixels * pixel_scale
        fringe_size = self.wavelength / self.diameter

        # Angular
        if focal_length is None:
            return output_size / fringe_size
        
        # Cartesian
        else:
            return output_size / (fringe_size * focal_length)

    
    def _transfer_matrix(self : Wavefront, 
                         npixels : int,
                         pixel_scale : Array,
                         shift : Array = 0.,
                         focal_length : Array = None) -> Array:
        """
        
        """
        scale_in = 1.0 / self.npixels
        scale_out = self._nfringes(npixels, pixel_scale, focal_length) / npixels
        in_vec = get_pixel_positions(self.npixels, scale_in, shift * scale_in)
        out_vec = get_pixel_positions(npixels, scale_out, shift * scale_out)

        if self.plane == 'Pupil':
            return np.exp(2j * np.pi * np.outer(in_vec, out_vec))
        elif self.plane == 'Focal':
            return np.exp(-2j * np.pi * np.outer(in_vec, out_vec))
        else:
            raise ValueError("plane must be either 'Pupil' or 'Focal'. "
                f"Got {self.plane}")


    def _MFT(self : Wavefront, 
             npixels : int, 
             pixel_scale : Array,
             focal_length : Array = None,
             shift : Array = np.zeros(2)) -> Array:
        """
        
        """
        # Transfer Matrices
        x_matrix = self._transfer_matrix(npixels, pixel_scale, shift[0], 
            focal_length)
        y_matrix = self._transfer_matrix(npixels, pixel_scale, shift[1], 
            focal_length).T

        # Propagation
        phasor = (y_matrix @ self.phasor) @ x_matrix
        nfringes = self._nfringes(npixels, pixel_scale, focal_length)
        phasor *= np.exp(np.log(nfringes) - \
            (np.log(self.npixels) + np.log(npixels)))
        return phasor


    def MFT(self : Wavefront, 
            npixels : int, 
            pixel_scale : Array,
            focal_length : Array = None) -> Wavefront:
        """
        
        """
        # Calculate
        plane, units = self._MFT_output(focal_length)
        phasor = self._MFT(npixels, pixel_scale, focal_length)

        # Return new wavefront
        pixel_scale = np.array(pixel_scale) # Allow float input
        return self.set(['amplitude', 'phase', 'pixel_scale', 'plane', 'units'],
            [np.abs(phasor), np.angle(phasor), pixel_scale, plane, units])


    def shifted_MFT(self : Wavefront, 
                    npixels : int, 
                    pixel_scale : Array,
                    shift : Array,
                    focal_length : Array = None,
                    pixel : bool = True) -> Wavefront:
        """
        TODO: Add ShiftedWavefront to track shift?
        """
        # Calculate
        plane, units = self._MFT_output(focal_length)
        shift = shift if pixel else shift / pixel_scale
        phasor = self._MFT(npixels, pixel_scale, focal_length, shift)

        # Return new wavefront
        return self.set(['amplitude', 'phase', 'pixel_scale', 'plane', 'units'],
            [np.abs(phasor), np.angle(phasor), pixel_scale, plane, units])


    #######################
    ### Other Functions ###
    #######################
    def tilt_wavefront(self : Wavefront, tilt_angles : Array) -> Wavefront:
        """
        Tilts the wavefront by the tilt_angles.

        Parameters
        ----------
        tilt_angles : Array, radians
            The (x, y) angles by which to tilt the wavefront.

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with the (x, y) tilts applied.
        """
        assert isinstance(tilt_angles, Array) and tilt_angles.shape == (2,), \
        ("tilt_angles must be an array with shape (2,) ie. (x, y).")

        opds = tilt_angles[:, None, None] * self.pixel_coordinates
        return self.add('phase', - self.wavenumber * opds.sum(0))


    def add_opd(self: Wavefront, path_difference : Array) -> Wavefront:
        """
        Applies the wavelength-dependent phase based on the supplied optical
        path difference.

        Parameters
        ----------
        path_difference : Array, meters
            The physical optical path difference of either the entire wavefront
            or each pixel individually.

        Returns
        -------
        wavefront : Wavefront
            The new wavefront with the phases updated according to the supplied
            path_difference
        """
        phase_difference = self.wavenumber * path_difference
        return self.add('phase', phase_difference)


    def normalise(self : Wavefront) -> Wavefront:
        """
        Normalises the total power of the wavefront to 1.

        Returns
        -------
        wavefront : Wavefront
            The new wavefront with the normalised electric field amplitudes.
        """
        return self.multiply('amplitude', (1 / np.linalg.norm(self.amplitude)))


    def invert_x_and_y(self : Wavefront) -> Wavefront:
        """
        Reflects the wavefront about both axes.

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with the phase and amplitude arrays reversed
            along both axes.
        """
        amplitude = np.flip(self.amplitude, axis=(-1, -2))
        phase = np.flip(self.phase, axis=(-1, -2))
        return self.set(['amplitude', 'phase'], [amplitude, phase])


    def invert_x(self : Wavefront) -> Wavefront:
        """
        Reflects the wavefront about the x axis.

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with the phase and amplitude arrays reversed
            along the x axis.
        """
        amplitude = np.flip(self.amplitude, axis=-1)
        phase = np.flip(self.phase, axis=-1)
        return self.set(['amplitude', 'phase'], [amplitude, phase])


    def invert_y(self : Wavefront) -> Wavefront:
        """
        Reflects the wavefront about the y axis.

        Returns
        -------
        wavefront : Wavefront
            The new wavefront with the phase and amplitude arrays reversed
            along the y axis.
        """
        amplitude = np.flip(self.amplitude, axis=-2)
        phase = np.flip(self.phase, axis=-2)
        return self.set(['amplitude', 'phase'], [amplitude, phase])


    def interpolate(self            : Wavefront,
                    npixels_out     : int,
                    pixel_scale_out : Array,
                    real_imaginary  : bool = False) -> Wavefront:
        """
        Performs a paraxial interpolation on the wavefront, determined by the
        the pixel_scale_out and npixels_out parameters. By default the
        interpolation is done on the amplitude and phase arrays, however by
        passing `real_imgainary=True` the interpolation is done on the real and
        imaginary components. This option allows for consistent interpolation
        behaviour when the phase array has a large amount of wrapping.
        Automatically conserves energy though the interpolation.

        Parameters
        ----------
        npixels_out : int
            The number of pixels representing the wavefront after the
            interpolation.
        pixel_scale_out : Array
            The pixel scale of the array after the interpolation.
        real_imaginary : bool = False
            Whether to interpolate the real and imaginary representation of the
            wavefront as opposed to the the amplitude and phase representation.

        Returns
        -------
        wavefront : Wavefront
            The new wavefront interpolated to the size and shape determined by
            npixels_out and pixel_scale_out, with the updated pixel_scale.
        """
        sampling_ratio = pixel_scale_out / self.pixel_scale
        if real_imaginary:
            field = np.array([self.real, self.imaginary])
        else:
            field = np.array([self.amplitude, self.phase])

        amplitude, phase = interpolate_field(field, npixels_out, 
            sampling_ratio, real_imaginary=real_imaginary)

        return self.set(['amplitude', 'phase', 'pixel_scale'], 
            [amplitude, phase, pixel_scale_out])


    def rotate(self           : Wavefront,
               angle          : Array,
               real_imaginary : bool = False,
               fourier        : bool = False,
               order          : int  = 1,
               padding        : int  = 2) -> Wavefront:
        """
        Performs a paraxial rotation on the wavefront, determined by the
        the angle parameter. By default the rotation is performed using a
        simple linear interpolation, but an information perserving rotation
        using fourier methods can be done by setting `fourier = True`. By
        default rotation is done on the amplitude and phase arrays, however by
        passing `real_imgainary=True` the rotation is done on the real and
        imaginary components.

        Parameters
        ----------
        angle : Array, radians
            The angle by which to rotate the wavefront in a clockwise direction.
        real_imaginary : bool = False
            Whether to rotate the real and imaginary representation of the
            wavefront as opposed to the the amplitude and phase representation.
        fourier : bool = False
            Should the fourier rotation method be used (True), or regular
            interpolation method be used (False).
        order : int = 2
            The interpolation order to use. Must be 0, 1, or 3.
        padding : int = 2
            The amount of fourier padding to use. Only applies if fourier is
            True.


        Returns
        -------
        wavefront : Wavefront
            The new wavefront rotated by angle in the clockwise direction.
        """
        # Get Field
        if real_imaginary:
            field = np.array([self.real, self.imaginary])
        else:
            field = np.array([self.amplitude, self.phase])

        # Rotate
        amplitude, phase = rotate_field(field, angle, fourier=fourier,
            real_imaginary=real_imaginary, order=order)
        return self.set(['amplitude', 'phase'], [amplitude, phase])


    def pad_to(self : Wavefront, npixels_out : int) -> Wavefront:
        """
        Paraxially zero-pads the `Wavefront` to the size determined by
        npixels_out. Note this only supports padding arrays of even dimension
        to even dimension, and odd dimension to to odd dimension, ie 2 -> 4 or
        3 -> 5.

        Parameters
        ----------
        npixels_out : int
            The size of the array to pad to the wavefront to.

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` zero-padded to the size npixels_out.
        """
        npixels_in  = self.npixels
        assert npixels_in  % 2 == npixels_out % 2, \
        ("Only supports even -> even or odd -> odd padding")
        assert npixels_out > npixels_in, ("npixels_out must be larger than the"
        " current array size: {}".format(npixels_in))

        new_centre = npixels_out // 2
        centre = npixels_in  // 2
        remainder = npixels_in  % 2
        padded = np.zeros([npixels_out, npixels_out])

        amplitude = padded.at[
                new_centre - centre : centre + new_centre + remainder,
                new_centre - centre : centre + new_centre + remainder
            ].set(self.amplitude)
        phase = padded.at[
                new_centre - centre : centre + new_centre + remainder,
                new_centre - centre : centre + new_centre + remainder
            ].set(self.phase)
        return self.set(['amplitude', 'phase'], [amplitude, phase])


    def crop_to(self : Wavefront, npixels_out : int) -> Wavefront:
        """
        Paraxially crops the `Wavefront` to the size determined by npixels_out.
        Note this only supports padding arrays of even dimension to even
        dimension, and odd dimension to to odd dimension, ie 4 -> 2 or 5 -> 3.

        Parameters
        ----------
        npixels_out : int
            The size of the array to crop to the wavefront to.

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` cropped to the size npixels_out.
        """
        npixels_in  = self.npixels

        assert npixels_in%2 == npixels_out%2, \
        ("Only supports even -> even or odd -> odd cropping")
        assert npixels_out < npixels_in, ("npixels_out must be smaller than the"
        " current array size: {}".format(npixels_in))

        new_centre = npixels_in  // 2
        centre = npixels_out // 2

        amplitude = self.amplitude[
            new_centre - centre : new_centre + centre,
            new_centre - centre : new_centre + centre]
        phase = self.phase[
            new_centre - centre : new_centre + centre,
            new_centre - centre : new_centre + centre]

        return self.set(['amplitude', 'phase'], [amplitude, phase])



class FresnelWavefront(Wavefront):
    """

    """


    def __init__(self       : Wavefront,
                 npixels    : int,
                 diameter   : Array,
                 wavelength : Array,
                 plane      : str = 'Pupil',
                 units      : str = 'Cartesian') -> Wavefront:
        """
        Constructor for the wavefront.

        Parameters
        ----------
        wavelength : float, meters
            The wavelength of the `Wavefront`.
        pixel_scale : float, meters/pixel
            The physical dimensions of each square pixel.
        amplitude : Array, power
            The electric field amplitude of the `Wavefront`.
        phase : Array, radians
            The electric field phase of the `Wavefront`.
        plane_type : enum.IntEnum.PlaneType
            The current plane of wavefront, can be Pupil, Focal.
        """

        super().__init__(
            npixels    = npixels,
            wavelength = wavelength,
            diameter   = diameter,
        )
    

    # Overwrite
    def _nfringes(self, npixels, pixel_scale, focal_shift, focal_length):

        propagation_distance = focal_length + focal_shift
        output_size = npixels * pixel_scale
        fringe_size = self.wavelength / self.diameter

        # # Angular - Not Implemented
        # if focal_length is None:
        #     return output_size / fringe_size
        
        # Cartesian
        return output_size / (fringe_size * propagation_distance)


    # Move to utils as thinlens?
    def quadratic_phase(self          : Wavefront,
                        x_coordinates : Array,
                        y_coordinates : Array,
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
        r_coordinates = np.hypot(x_coordinates, y_coordinates)
        return np.exp(0.5j * self.wavenumber * r_coordinates ** 2 / distance)


    def transfer_function(self      : Wavefront,
                          distance  : Array) -> Array:
        """
        The Optical Transfer Function defining the phase evolution of the
        wavefront when propagating to a non-conjugate plane.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate.
        distance : Array, meters
            The distance that is being propagated in meters.

        Returns
        -------
        field : Array
            The field that represents the optical transfer.
        """
        return np.exp(1.0j * self.wavenumber * distance)


    # Overwritten
    def _transfer_matrix(self : Wavefront, 
                         npixels : int,
                         pixel_scale : Array,
                         focal_shift : Array,
                         focal_length : Array = None,
                         shift : Array = 0.) -> Array:
        """
        
        """
        scale_in = 1.0 / self.npixels
        scale_out = self._nfringes(npixels, pixel_scale, focal_shift, 
            focal_length) / npixels
        in_vec = get_pixel_positions(self.npixels, scale_in, shift * scale_in)
        out_vec = get_pixel_positions(npixels, scale_out, shift * scale_out)

        if self.plane == 'Pupil':
            return np.exp(2j * np.pi * np.outer(in_vec, out_vec))
        elif self.plane == 'Focal':
            return np.exp(-2j * np.pi * np.outer(in_vec, out_vec))
        else:
            raise ValueError("plane must be either 'Pupil' or 'Focal'. "
                f"Got {self.plane}")
    

    # Overwritten
    def _MFT(self : Wavefront,
             phasor : Array,
             npixels : int, 
             pixel_scale : Array,
             focal_length : Array,
             focal_shift : Array,
             shift : Array = np.zeros(2)) -> Array:
        """
        
        """
        # Set up
        nfringes = self._nfringes(npixels, pixel_scale, focal_shift, 
            focal_length)
        x_matrix = self._transfer_matrix(npixels, pixel_scale, focal_shift, 
            focal_length, shift[0])
        y_matrix = self._transfer_matrix(npixels, pixel_scale, focal_shift, 
            focal_length, shift[1]).T

        # Perform and normalise
        phasor = (y_matrix @ phasor) @ x_matrix
        phasor *= np.exp(np.log(nfringes) - \
            (np.log(self.npixels) + np.log(npixels)))
        return phasor


    def fresnel_prop(self : Wavefront,
                     npixels : int, 
                     pixel_scale : Array,
                     focal_length : Array,
                     focal_shift) -> Array:
        """
        Propagates the wavefront from the input plane to the output plane using
        a Matrix Fourier Transform.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate.

        Returns
        -------
        field : Array
            The normalised electric field phasor after the propagation.
        """
        # Calculate phase values
        first, second, third, fourth = self._phase_factors(npixels, pixel_scale, 
            focal_length, focal_shift)

        # Apply phases
        phasor = self.phasor
        phasor *= first
        phasor *= second

        # Propagate
        phasor = self._MFT(phasor, npixels, pixel_scale, focal_length, 
            focal_shift)

        # Apply phases
        phasor *= third
        phasor *= fourth

        # Update
        pixel_scale = np.array(pixel_scale)
        return self.set(['amplitude', 'phase', 'pixel_scale', 'plane', 'units'], 
            [np.abs(phasor), np.angle(phasor), pixel_scale, 'Intermediate', 
            'Cartesian'])


    def _phase_factors(self, npixels, pixel_scale, focal_length, focal_shift):
        """

        """
        # Coordaintes
        prop_dist = focal_length + focal_shift
        input_positions = self.pixel_coordinates
        output_positions = get_pixel_positions(
            (npixels, npixels), (pixel_scale, pixel_scale))

        # Calculate phase values
        first_factor = self.quadratic_phase(*input_positions, -focal_length)
        second_factor = self.quadratic_phase(*input_positions, prop_dist)
        third_factor = self.transfer_function(prop_dist)
        fourth_factor = self.quadratic_phase(*output_positions, prop_dist)
        return first_factor, second_factor, third_factor, fourth_factor


    def shifted_fresnel_prop(self : Wavefront,
                             npixels : int, 
                             pixel_scale : Array,
                             shift : Array,
                             focal_length : Array,
                             focal_shift : Array,
                             pixel : bool = True) -> Array:
        """
        Propagates the wavefront from the input plane to the output plane using
        a Matrix Fourier Transform.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to propagate.

        Returns
        -------
        field : Array
            The normalised electric field phasor after the propagation.
        """
        # Get shift
        shift = shift if pixel else shift / pixel_scale

        # Calculate phase values
        first, second, third, fourth = self._phase_factors(npixels, pixel_scale, 
            focal_length, focal_shift)

        # Apply phases
        phasor = self.phasor
        phasor *= first
        phasor *= second

        # Propagate
        phasor = self._MFT(phasor, npixels, pixel_scale, focal_length, 
            focal_shift, shift)

        # Apply phases
        phasor *= third
        phasor *= fourth

        # Update
        pixel_scale = np.array(pixel_scale)
        return self.set(['amplitude', 'phase', 'pixel_scale', 'plane', 'units'], 
            [np.abs(phasor), np.angle(phasor), pixel_scale, 'Intermediate', 
            'Cartesian'])