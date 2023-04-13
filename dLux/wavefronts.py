from __future__ import annotations
import jax.numpy as np
from jax import vmap
from zodiax import Base
from dLux.utils.coordinates import get_pixel_positions
from dLux.utils.interpolation import interpolate_field, rotate_field
import dLux


__all__ = ["Wavefront"]


Array = np.ndarray


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
    plane_type  : str
    units       : str


    def __init__(self        : Wavefront,
                 wavelength  : Array,
                 pixel_scale : Array,
                 amplitude   : Array,
                 phase       : Array,
                 plane       : str,
                 units       : str) -> Wavefront:
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
        self.pixel_scale = np.asarray(pixel_scale, dtype=float)
        self.amplitude   = np.asarray(amplitude,   dtype=float)
        self.phase       = np.asarray(phase,       dtype=float)

        # Input checks
        assert self.wavelength.shape == (), \
        ("wavelength must be a scalar Array.")
        assert self.pixel_scale.shape == (), \
        ("pixel_scale must be a scalar Array.")
        assert self.amplitude.ndim == 2, \
        ("amplitude must a 2d array (npix, npix).")
        assert self.phase.ndim == 2, \
        ("phase must a 2d array (npix, npix).")
        assert self.amplitude.shape == self.phase.shape, \
        ("The amplitude and phase arrays must have the same shape.")

        # Set by Propagators
        # Can eventually be 'Fresnel', 'Intermediate' 
        if plane not in ("Pupil", "Focal"):
            raise ValueError("plane must be either 'Pupil' or 'Focal'.")
        self.plane = str(plane)

        # Set by Propagators
        if units not in ("Cartesian", "Angular"):
            raise ValueError("unit must be either 'Cartesian' or 'Angular'.")
        self.units = str(units)


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
        return np.sum(self.amplitude ** 2, axis=0)


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


    def wavefront_to_psf(self: Wavefront) -> Array:
        """
        Calculates the Point Spread Function (PSF), ie the squared modulus
        of the complex wavefront.

        TODO: Take in the parameters dictionary and use the parameters in that
        to determine the way to output the wavefront.

        Returns
        -------
        psf : Array
            The PSF of the wavefront.
        """
        return self.amplitude ** 2


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

        amplitude, phase = interpolate_field(field[:, 0], npixels_out, 
            sampling_ratio, real_imaginary=real_imaginary)[:, None, :, :]

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
        amplitude, phase = rotate_field(field[:, 0], angle, fourier=fourier,
            real_imaginary=real_imaginary, order=order)[:, None, :, :]
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
        padded = np.zeros([self.nfields, npixels_out, npixels_out])

        amplitude = padded.at[:,
                new_centre - centre : centre + new_centre + remainder,
                new_centre - centre : centre + new_centre + remainder
            ].set(self.amplitude)
        phase = padded.at[:,
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

        amplitude = self.amplitude[:,
            new_centre - centre : new_centre + centre,
            new_centre - centre : new_centre + centre]
        phase = self.phase[:,
            new_centre - centre : new_centre + centre,
            new_centre - centre : new_centre + centre]

        return self.set(['amplitude', 'phase'], [amplitude, phase])