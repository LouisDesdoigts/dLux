from __future__ import annotations
import jax.numpy as np
from jax import vmap
from equinox import tree_at
from enum import IntEnum
from abc import ABC
from dLux.utils.coordinates import get_pixel_coordinates
from dLux.utils.interpolation import interpolate_field, rotate_field
import dLux

__all__ = ["PlaneType", "CartesianWavefront", "AngularWavefront",
           'FarFieldFresnelWavefront']


Array = np.ndarray


class PlaneType(IntEnum):
    """
    Enumeration object to keep track of plane types. This may prove to be
    redundant.

    NOTE: Propagtors are not currently set up to ever set the PlaneType to
    Intermediate. This will be done with the Near-Field Fresnel implementation.
    """
    Pupil = 1
    Focal = 2
    Intermediate = 3


class Wavefront(dLux.base.ExtendedBase, ABC):
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
        can be in units of eitehr meters per pixel or radians per pixel
        depending on both the plane type and the wavfront type (Cartesian or
        Angular).
    plane_type : enum.IntEnum.PlaneType
        The current plane type of wavefront, can be Pupil, Focal or
        Intermediate.
    """
    wavelength  : Array
    pixel_scale : Array
    plane_type  : PlaneType
    amplitude   : Array
    phase       : Array


    def __init__(self        : Wavefront,
                 wavelength  : Array,
                 pixel_scale : Array,
                 amplitude   : Array,
                 phase       : Array,
                 plane_type  : PlaneType) -> Wavefront:
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
        self.plane_type  = plane_type

        # Input checks
        assert self.wavelength.shape == (), \
        ("wavelength must be a scalar Array.")
        assert self.pixel_scale.shape == (), \
        ("pixel_scale must be a scalar Array.")
        assert self.amplitude.ndim == 3, \
        ("amplitude must a 3d array (nfields, npix, npix).")
        assert self.phase.ndim == 3, \
        ("phase must a 3d array (nfields, npix, npix).")
        assert self.amplitude.shape == self.phase.shape, \
        ("The amplitude and phase arrays must have the same shape.")
        assert isinstance(plane_type, PlaneType), \
        ("plane_type must a PlaneType object.")


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
    def nfields(self : Wavefront) -> int:
        """
        Returns the number of polarisation fields currently representing the
        wavefront. Taken from the amplitude array first dimension.

        Returns
        -------
        pixels : int
            The number of polarisation fields that represent the `Wavefront`.
        """
        return self.amplitude.shape[0]


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
        return get_pixel_coordinates(self.npixels, self.pixel_scale)


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


    ########################
    ### Setter Functions ###
    ########################
    def set_amplitude(self : Wavefront, amplitude : Array) -> Wavefront:
        """
        Mutator for the amplitude attribute.

        Parameters
        ---------
        amplitude : Array, power
            The electric field amplitude of the `Wavefront`.

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with the updated amplitude.
        """
        assert isinstance(amplitude, Array) and amplitude.ndim == 3, \
        ("amplitude must be a 3d array.")
        return tree_at(
            lambda wavefront : wavefront.amplitude, self, amplitude)


    def set_phase(self : Wavefront, phase : Array) -> Wavefront:
        """
        Mutator for the phase attribute.

        Parameters
        ----------
        phase : Array, radians
            The phases of each pixel on the `Wavefront`.

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with the updated phase.
        """
        assert isinstance(phase, Array) and phase.ndim == 3, \
        ("phase must be a 3d array.")
        return tree_at(
            lambda wavefront : wavefront.phase, self, phase)


    def set_pixel_scale(self : Wavefront, pixel_scale : Array) -> Wavefront:
        """
        Mutator for the pixel_scale attribute.

        Parameters
        ----------
        pixel_scale : Array
            The new pixel_scale associated with the wavefront.

        Returns
        -------
        wavefront : Wavefront
            The new Wavefront object with the updated pixel_scale.
        """
        assert isinstance(pixel_scale, Array) and pixel_scale.ndim == 0, \
        ("pixel_scale must be a scalar array.")
        return tree_at(
            lambda wavefront : wavefront.pixel_scale, self, pixel_scale)


    def set_plane_type(self : Wavefront, plane : PlaneType) -> Wavefront:
        """
        Mutator for the PlaneType attribute.

        Parameters
        ----------
        plane : PlaneType
            A PlaneType object describing the plane that the `Wavefront` is
            currently in.

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with the update plate_type information.
        """
        assert isinstance(plane, PlaneType), \
        ("plane must be a PlaneType object.")
        return tree_at(
            lambda wavefront : wavefront.plane_type, self, plane)


    def set_phasor(self      : Wavefront,
                      amplitude : Array,
                      phase     : Array) -> Wavefront:
        """
        Updates the phasor of the wavefront (ie both the amplitude and the
        phase).

        Parameters
        ----------
        amplitude : Array, power
            The new electric field amplitude of the wavefront.
        phase : Array, radians
            The new electric field phase of the wavefront.

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with updated amplitude and phase.
        """
        assert isinstance(amplitude, Array) and amplitude.ndim == 3, \
        ("amplitude must be a 3d array.")
        assert isinstance(phase, Array) and phase.ndim == 3, \
        ("phase must be a 3d array.")
        assert amplitude.shape == phase.shape, \
        ("amplitude and phase arrays must have the same shape.")
        return tree_at(
            lambda wavefront : (wavefront.amplitude, wavefront.phase), self,
                               (amplitude, phase))


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
        return self.add_phase(- self.wavenumber * opds.sum(0))


    def multiply_amplitude(self : Wavefront, array_like : Array) -> Wavefront:
        """
        Multiply the amplitude of the `Wavefront` by either a float or array.

        Parameters
        ----------
        array_like : Array
            An array or float that has the same dimensions as amplitude that is
            multipled by the current ampltide.

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with the updated amplitude.
        """
        amplitude = self.amplitude
        assert isinstance(array_like, Array) and array_like.ndim in (0, 2, 3), \
        ("array_like must be either a scalar array or array with 2 or 3 "
         "dimensions.")
        if array_like.ndim in (2, 3):
            assert array_like.shape[-2:] == amplitude.shape[-2:], \
            ("array_like shape must be equal to the current ampltude array.")
        return self.set_amplitude(amplitude * array_like)


    def add_phase(self : Wavefront, array_like : Array) -> Wavefront:
        """
        Add to the phase of the `Wavefront` by either a float or array.

        Parameters
        ----------
        array_like : Array
            An array or float that has the same dimensions as phase that is
            added to the current phase.

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with the updated phase.
        """
        phase = self.phase
        assert isinstance(array_like, Array) and array_like.ndim in (0, 2, 3), \
        ("array_like must be either a scalar array or array with 2 or 3 "
         "dimensions.")
        if array_like.ndim in (2, 3):
            assert array_like.shape[-2:] == phase.shape[-2:], \
            ("array_like shape must be equal to the current phase array.")
        return self.set_phase(phase + array_like)


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
        assert isinstance(path_difference, Array) and \
        path_difference.ndim in (0, 2, 3), ("path_difference must be either a "
        "scalar array or array with 2 or 3 dimensions.")
        phase_difference = self.wavenumber * path_difference
        return self.add_phase(phase_difference)


    def normalise(self : Wavefront) -> Wavefront:
        """
        Normalises the total power of the wavefront to 1.

        Returns
        -------
        wavefront : Wavefront
            The new wavefront with the normalised electric field amplitudes.
        """
        return self.multiply_amplitude(1 / np.linalg.norm(self.amplitude))


    def wavefront_to_psf(self             : Wavefront,
                         return_polarised : bool = False) -> Array:
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
        return np.sum(self.amplitude ** 2, axis=0)


    def invert_x_and_y(self : Wavefront) -> Wavefront:
        """
        Reflects the wavefront about both axes.

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with the phase and amplitude arrays reversed
            along both axes.
        """
        new_amplitude = np.flip(self.amplitude, axis=(-1, -2))
        new_phase = np.flip(self.phase, axis=(-1, -2))
        return self.set_phasor(new_amplitude, new_phase)


    def invert_x(self : Wavefront) -> Wavefront:
        """
        Reflects the wavefront about the x axis.

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with the phase and amplitude arrays reversed
            along the x axis.
        """
        new_amplitude = np.flip(self.amplitude, axis=-1)
        new_phase = np.flip(self.phase, axis=-1)
        return self.set_phasor(new_amplitude, new_phase)


    def invert_y(self : Wavefront) -> Wavefront:
        """
        Reflects the wavefront about the y axis.

        Returns
        -------
        wavefront : Wavefront
            The new wavefront with the phase and amplitude arrays reversed
            along the y axis.
        """
        new_amplitude = np.flip(self.amplitude, axis=-2)
        new_phase = np.flip(self.phase, axis=-2)
        return self.set_phasor(new_amplitude, new_phase)


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

        if field.shape[1] == 1:
            new_amplitude, new_phase = \
            interpolate_field(field[:, 0], npixels_out, sampling_ratio,
                              real_imaginary=real_imaginary)[:, None, :, :]
        else:
            interpolator = vmap(interpolate_field, in_axes=(1, None, None))
            new_amplitude, new_phase = interpolator(field, npixels_out,
                                sampling_ratio, real_imaginary=real_imaginary)

        # Update parameters
        return tree_at(lambda wavefront:
                 (wavefront.amplitude, wavefront.phase, wavefront.pixel_scale),
                        self, (new_amplitude, new_phase, pixel_scale_out))


    def rotate(self           : Wavefront,
               angle          : Array,
               real_imaginary : bool = False,
               fourier        : bool = False,
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
        if field.shape[1] == 1:
            new_amplitude, new_phase = \
            rotate_field(field[:, 0], angle, fourier=fourier,
                         real_imaginary=real_imaginary)[:, None, :, :]
        else:
            rotator = vmap(rotate_field, in_axes=(1, None))
            new_amplitude, new_phase = rotator(field, angle, fourier=fourier,
                                               real_imaginary=real_imaginary)

        # Update parameters
        return tree_at(lambda wavefront: (wavefront.amplitude, wavefront.phase),
                                   self, (new_amplitude, new_phase))


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

        new_amplitude = padded.at[:,
                new_centre - centre : centre + new_centre + remainder,
                new_centre - centre : centre + new_centre + remainder
            ].set(self.amplitude)
        new_phase = padded.at[:,
                new_centre - centre : centre + new_centre + remainder,
                new_centre - centre : centre + new_centre + remainder
            ].set(self.phase)
        return self.set_phasor(new_amplitude, new_phase)


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

        new_amplitude = self.amplitude[:,
            new_centre - centre : new_centre + centre,
            new_centre - centre : new_centre + centre]
        new_phase = self.phase[:,
            new_centre - centre : new_centre + centre,
            new_centre - centre : new_centre + centre]

        return self.set_phasor(new_amplitude, new_phase)


class CartesianWavefront(Wavefront):
    """
    A class representing some wavefront, designed to track the various
    parameters such as wavelength, pixel_scale, amplitude and phase, as well as
    a helper parmeter, plane_type. CartesianWavefronts have pixel scales in
    units of meters per pixel in all planes.
    """


    def __init__(self        : Wavefront,
                 wavelength  : Array,
                 pixel_scale : Array,
                 amplitude   : Array,
                 phase       : Array,
                 plane_type  : PlaneType) -> Wavefront:
        """
        Constructor for Cartesian wavefronts.

        Parameters
        ----------
        wavelength : float, meters
            The wavelength of the `Wavefront`.
        pixel_scale : float, meters/pixel
            The physical dimensions of each pixel.
        amplitude : Array, power
            The electric field amplitude of the `Wavefront`.
        phase : Array, radians
            The electric field phase of the `Wavefront`.
        plane_type : enum.IntEnum.PlaneType
            The current plane of wavefront, can be Pupil, Focal.
        """
        super().__init__(wavelength, pixel_scale,
                         amplitude, phase, plane_type)


class AngularWavefront(Wavefront):
    """
    A class representing some wavefront, designed to track the various
    parameters such as wavelength, pixel_scale, amplitude and phase, as well as
    a helper parmeter, plane_type. AngularWavefronts have pixel scales in
    units of meters per pixel in Pupil planes and radians per pixel in Focal
    planes.
    """


    def __init__(self        : Wavefront,
                 wavelength  : Array,
                 pixel_scale : Array,
                 amplitude   : Array,
                 phase       : Array,
                 plane_type  : PlaneType) -> Wavefront:
        """
        Constructor for Angular wavefronts.

        Parameters
        ----------
        wavelength : float, meters
            The wavelength of the `Wavefront`.
        pixel_scale : float, meters/pixel or radians/pixel
            The physical dimensions of each pixel. Units are in meters
            per pixel in Pupil planes and radians per pixel in Focal planes.
        amplitude : Array, power
            The electric field amplitude of the `Wavefront`.
        phase : Array, radians
            The electric field phase of the `Wavefront`.
        plane_type : enum.IntEnum.PlaneType
            The current plane of wavefront, can be Pupil, Focal.
        """
        super().__init__(wavelength, pixel_scale,
                         amplitude, phase, plane_type)


class FarFieldFresnelWavefront(Wavefront):
    """
    A class representing some wavefront, designed to track the various
    parameters such as wavelength, pixel_scale, amplitude and phase, as well as
    a helper parmeter, plane_type. FarFieldFresnelWavefronts are designed to
    work with FarFieldFresnel Propagators, and are better able to represent the
    behaviour of wavefronts outside of the focal planes, in the far-field
    approximation.
    """


    def __init__(self        : Wavefront,
                 wavelength  : Array,
                 pixel_scale : Array,
                 amplitude   : Array,
                 phase       : Array,
                 plane_type  : PlaneType) -> Wavefront:
        """
        Constructor for FarFieldFresnel wavefronts.

        Parameters
        ----------
        wavelength : float, meters
            The wavelength of the `Wavefront`.
        pixel_scale : float, meters/pixel or radians.pixl
            The physical dimensions of each pixel. Units are in meters
            per pixel in Pupil planes and meters per pixel or radians per pixel
            in Focal planes depending on if Cartesian or Angular Propagators
            are used respectively.
        amplitude : Array, power
            The electric field amplitude of the `Wavefront`.
        phase : Array, radians
            The electric field phase of the `Wavefront`.
        plane_type : enum.IntEnum.PlaneType
            The current plane of wavefront, can be Pupil, Focal.
        """
        super().__init__(wavelength, pixel_scale,
                         amplitude, phase, plane_type)