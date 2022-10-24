from __future__ import annotations
import jax.numpy as np
from equinox import tree_at
from enum import IntEnum
from abc import ABC
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


class Wavefront(dLux.base.Base, ABC):
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
        assert wavelength.shape == (), ("wavelength must be a scalar Array.")
        assert pixel_scale.shape == (), ("pixel_scale must be a scalar Array.")
        assert self.amplitude.shape == self.phase.shape, \
        ("The amplitude and phase arrays must have the same shape.")
        assert isinstance(plane_type, PlaneType), \
        ("plane_type must a PlaneType object.")


    ########################
    ### Getter Functions ###
    ########################
    def get_wavelength(self : Wavefront) -> Array:
        """
        Getter method for the wavelength attribute.

        Returns
        -------
        wavelength : float, meters
            The wavelength of the `Wavefront`.
        """
        return self.wavelength


    def get_amplitude(self : Wavefront) -> Array:
        """
        Getter method for the amplitude attribute.

        Returns
        -------
        amplitude : Array, power
            The electric field amplitude of the `Wavefront`.
        """
        return self.amplitude


    def get_phase(self : Wavefront) -> Array:
        """
        Getter method for the phase attribute.

        Returns
        -------
        phase : Array, radians
            The phases of each pixel on the `Wavefront`.
        """
        return self.phase


    def get_diameter(self : Wavefront) -> Array:
        """
        Returns the current wavefront diameter calulated using the pixel scale
        and number of pixels.

        Returns
        -------
        diameter : Array, meters or radians
           The current diameter of the wavefront.
        """
        return self.get_npixels() * self.get_pixel_scale()


    def get_plane_type(self : Wavefront) -> PlaneType:
        """
        Getter method for the plane_type attribute.

        Returns
        -------
        plane : PlaneType
            The plane that the `Wavefront` is currently in. The options are
            currently "Pupil", "Focal" and "Intermediate".
        """
        return self.plane_type


    def get_npixels(self : Wavefront) -> int:
        """
        Returns the side length of the arrays currently representing the
        wavefront. Taken from the amplitude array.

        Returns
        -------
        pixels : int
            The number of pixels that represent the `Wavefront`.
        """
        return self.get_amplitude().shape[-1]


    def get_pixel_scale(self : Wavefront) -> Array:
        """
        Getter method for the pixel_scale attribute.

        Returns
        -------
        pixel scale : float, meters/pixel or radians/pixel
            The current pixel scale associated with the wavefront.
        """
        return self.pixel_scale


    def get_real(self : Wavefront) -> Array:
        """
        Returns the real component of the `Wavefront`.

        Returns
        -------
        wavefront : Array
            The real component of the `Wavefront` phasor.
        """
        return self.get_amplitude() * np.cos(self.get_phase())


    def get_imaginary(self : Wavefront) -> Array:
        """
        Returns the imaginary component of the `Wavefront`.

        Returns
        -------
        wavefront : Array
           The imaginary component of the `Wavefront` phasor.
        """
        return self.get_amplitude() * np.sin(self.get_phase())


    def get_phasor(self : Wavefront) -> Array:
        """
        The electric field phasor described by this Wavefront in complex form.

        Returns
        -------
        field : Array
            The electric field phasor of the wavefront.
        """
        return self.get_amplitude() * np.exp(1j * self.get_phase())


    ########################
    ### Setter Functions ###
    #########################
    def set_wavelength(self : Wavefront, wavelength : Array) -> Wavefront:
        """
        Mutator for the wavelength attribute.

        Parameters
        ----------
        wavelength : float, meters
            The wavelength of the `Wavefront`.

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with the updated wavelength.
        """
        assert isinstance(wavelength, Array) and wavelength.ndim == 0, \
        ("wavelength must be a scalar array.")
        return tree_at(
            lambda wavefront : wavefront.wavelength, self, wavelength)


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


    def update_phasor(self      : Wavefront,
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
        assert isinstance(amplitude, Array) and amplitude.ndim == 2, \
        ("amplitude must be a 2d array.")
        assert isinstance(phase, Array) and phase.ndim == 2, \
        ("phase must be a 2d array.")
        assert amplitude.shape == phase.shape, \
        ("amplitude and phase arrays must have the same shape.")
        return tree_at(
            lambda wavefront : (wavefront.amplitude, wavefront.phase), self,
                               (amplitude, phase))


    #################################
    ### Mutator / Other Functions ###
    #################################
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
        assert isinstance(tilt_angles, Array) and tilt_angles.shape == (2,) \
        ("tilt_angles must be an array with shape (2,) ie. (x, y).")

        x_angle, y_angle = tilt_angle
        x_positions, y_positions = wavefront.get_pixel_coordinates()
        wavenumber = 2 * np.pi / wavefront.get_wavelength()
        phase = - wavenumber * (x_positions * x_angle + y_positions * y_angle)

        return wavefront.add_phase(phase)


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
            The new `Wavefront` with the updated ampltiude.
        """
        amplitude = self.get_amplitude()
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
        phase = self.get_phase()
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
        phase_difference = 2 * np.pi * path_difference / self.get_wavelength()
        return self.add_phase(phase_difference)


    def normalise(self : Wavefront) -> Wavefront:
        """
        Normalises the total power of the wavefront to 1.

        Returns
        -------
        wavefront : Wavefront
            The new wavefront with the normalised electric field amplitudes.
        """
        total_intensity = np.linalg.norm(self.get_amplitude())
        return self.multiply_amplitude(1 / total_intensity)


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
        # Sums the first axis for empty polarisation array
        return np.sum(self.get_amplitude() ** 2, axis=0)


    def get_pixel_coordinates(self : Wavefront) -> Array:
        """
        Returns the physical positions of the wavefront pixels in meters.

        Returns
        -------
        pixel_positions : Array
            The coordinates of the centers of each pixel representing the
            wavefront.
        """
        return dLux.utils.coordinates.get_pixel_coordinates( \
                    self.get_npixels(), self.get_pixel_scale())


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
        return self.update_phasor(new_amplitude, new_phase)


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
        return self.update_phasor(new_amplitude, new_phase)


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
        return self.update_phasor(new_amplitude, new_phase)


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
        sampling_ratio = pixel_scale_out / self.get_pixel_scale()
        if real_imaginary:
            field = np.array([self.get_real(), self.get_imaginary()])
        else:
            field = np.array([self.get_amplitude(), self.get_phase()])
        new_ampltiude, new_phase = dLux.utils.interpolation.interpolate_field( \
            field, npixels_out, sampling_ratio, real_imaginary=real_imaginary)

        # Update parameters
        return tree_at(lambda wavefront:
                 (wavefront.amplitude, wavefront.phase, wavefront.pixel_scale),
                        self, (new_ampltiude, new_phase, pixel_scale_out))


    def pad_to(self : Wavefront, npixels_out : int) -> Wavefront:
        """
        Paraxially zero-pads the `Wavefront` to the size determined by
        npixles_out. Note this only supports padding arrays of even dimension
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
        npixels_in  = self.get_npixels()
        assert npixels_in  % 2 == npixels_out % 2, \
        ("Only supports even -> even or odd -> odd padding")
        assert npixles_out > npixels_in, ("npixles_out must be larger than the"
        " current array size: {}".format(npixels_in))

        new_centre = npixels_out // 2
        centre = npixels_in  // 2
        remainder = npixels_in  % 2
        padded = np.zeros([npixels_out, npixels_out])

        new_amplitude = padded.at[
                new_centre - centre : centre + new_centre + remainder,
                new_centre - centre : centre + new_centre + remainder
            ].set(self.amplitude)
        new_phase = padded.at[
                new_centre - centre : centre + new_centre + remainder,
                new_centre - centre : centre + new_centre + remainder
            ].set(self.phase)
        return self.update_phasor(new_amplitude, new_phase)


    def crop_to(self : Wavefront, npixels_out : int) -> Wavefront:
        """
        Paraxially crops the `Wavefront` to the size determined by npixles_out.
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
        npixels_in  = self.get_npixels()

        assert npixels_in %2 == npixels_out%2, \
        ("Only supports even -> even or 0dd -> odd cropping")
        assert npixles_out < npixels_in, ("npixles_out must be smaller than the"
        " current array size: {}".format(npixels_in))

        new_centre = npixels_in  // 2
        centre = npixels_out // 2

        new_amplitude = self.amplitude[
            new_centre - centre : new_centre + centre,
            new_centre - centre : new_centre + centre]
        new_phase = self.phase[
            new_centre - centre : new_centre + centre,
            new_centre - centre : new_centre + centre]

        return self.update_phasor(new_amplitude, new_phase)


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
                 plane_type  : PlaneType,
                 amplitude   : Array,
                 phase       : Array) -> Wavefront:
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


    def transfer_function(self : Wavefront, distance : Array) -> Array:
        """
        The Optical Transfer Function defining the phase evolution of the
        wavefront when propagating to a non-conjugate plane.

        Parameters
        ----------
        distance : Array
            The distance that is being propagated in meters.

        Returns
        -------
        phase : Array, radians
            The phase that represents the optical transfer.
        """
        wavenumber = 2. * np.pi / self.get_wavelength()
        return np.exp(1.0j * wavenumber * distance)