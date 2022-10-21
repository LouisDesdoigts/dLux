from __future__ import annotations
import dLux
import jax
import equinox as eqx 
import jax.numpy as np
import typing
import enum
import abc


__all__ = ["PlaneType", "CartesianWavefront", "AngularWavefront",
           'FarFieldFresnelWavefront']
Array = np.ndarray


class PlaneType(enum.IntEnum):
    """
    Enumeration object to keep track of plane types. This may prove to be
    redundant.

    NOTE: Propagtors are not currently set up to ever set the PlaneType to
    Intermediate. This will be done with the Near-Field Fresnel implementation.
    """
    Pupil = 1
    Focal = 2
    Intermediate = 3


class Wavefront(dLux.base.Base, abc.ABC):
    """
    A class representing some wavefront, designed to track the various
    parameters such as wavelength, pixel_scale, amplitude and phase, as well as
    a helper parmeter, plane_type.

    All wavefront currently only support sqaure amplitude and phase arrays.

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
                 plane_type  : PlaneType,
                 amplitude   : Array, 
                 phase       : Array) -> Wavefront:
        """
        Constructor for the base wavefront.

        Parameters
        ----------
        wavelength : float, meters
            The wavelength of the `Wavefront`.
        amplitude : Array, power
            The electric field amplitude of the `Wavefront`. 
        phase : Array, radians
            The electric field phase of the `Wavefront`.
        pixel_scale : float, meters/pixel
            The physical dimensions of each square pixel.
        plane_type : enum.IntEnum.PlaneType
            The current plane of wavefront, can be Pupil, Focal
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
        Returns
        -------
        wavelength : float, meters
            The wavelength of the `Wavefront`.
        """
        return self.wavelength


    # def get_offset(self : Wavefront) -> Array:
    #     """
    #     Returns
    #     -------
    #     offset : Array, radians
    #         The (x, y) angular offset of the wavefront from the optical 
    #         axis.
    #     """
    #     return self.offset


    def get_amplitude(self : Wavefront) -> Array:
        """
        Returns
        -------
        amplitude : Array, power
            The electric field amplitude of the `Wavefront`. 
        """
        return self.amplitude


    def get_phase(self : Wavefront) -> Array:
        """
        Returns
        -------
        phase : Array, radians
            The phases of each pixel on the `Wavefront`.
        """
        return self.phase


    def get_diameter(self : Wavefront) -> Array:
        """
        Returns the current Wavefront diameter
        TODO: Add unit-tests for this function

        Returns
        -------
        diameter : float
           The current diameter of the wavefront
        """
        return self.get_npixels() * self.get_pixel_scale()


    def get_plane_type(self : Wavefront) -> PlaneType:
        """
        Returns
        -------
        plane : str
            The plane that the `Wavefront` is currently in. The 
            options are currently "Pupil", "Focal" and "None".
        """
        return self.plane_type


    def get_npixels(self : Wavefront) -> int:
        """
        The side length of the pixel array that represents the 
        electric field of the `Wavefront`. Calcualtes the `pixels`
        value from the shape of the amplitude array.

        Returns 
        -------
        pixels : int
            The number of pixels that represent the `Wavefront` along 
            one side.
        """
        return self.get_amplitude().shape[-1]


    def get_pixel_scale(self : Wavefront) -> Array:
        """
         Returns
        -------
        pixel scale : float, meters/pixel
            The current pixel scale associated with the wavefront.
        """
        return self.pixel_scale


    def get_real(self : Wavefront) -> Array:
        """
        The real component of the `Wavefront`. 

        Returns 
        -------
        wavefront : Array
            The real component of the complex `Wavefront`.
        """
        return self.get_amplitude() * np.cos(self.get_phase())


    def get_imaginary(self : Wavefront) -> Array:
        """
        The imaginary component of the `Wavefront`.

        Returns
        -------
        wavefront : Array
           The imaginary component of the complex `Wavefront`.
        """
        return self.get_amplitude() * np.sin(self.get_phase())


    def get_phasor(self : Wavefront) -> Array:
        """
        The electric field phasor described by this Wavefront in complex form.

        Returns
        -------
        field : Array[complex]
            The complex electric field phasor of the wavefront.
        """
        return self.get_amplitude() * np.exp(1j * self.get_phase()) 

    ########################
    ### Setter Functions ###
    #########################
    def set_wavelength(self : Wavefront, wavelength : Array) -> Wavefront:
        """
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
        return eqx.tree_at(
            lambda wavefront : wavefront.wavelength, self, wavelength)


#     def set_offset(self : Wavefront, offset : Array) -> Wavefront:
#         """
#         Parameters
#         ----------
#         offset : Array, radians
#             The (x, y) angular offset of the wavefront from the optical 
#             axis.

#         Returns
#         -------
#         wavefront : Wavefront
#             The new `Wavefront` with the updated offset. 
#         """
#         assert isinstance(offset, Array) and offset.shape == (2,), \
#         ("offset must be a array of shape (2,), ie (x, y).")
#         return eqx.tree_at(
#             lambda wavefront : wavefront.offset, self, offset)


    def set_amplitude(self : Wavefront, amplitude : Array) -> Wavefront:
        """
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
        return eqx.tree_at(
            lambda wavefront : wavefront.amplitude, self, amplitude)


    def set_phase(self : Wavefront, phase : Array) -> Wavefront:
        """
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
        return eqx.tree_at(
            lambda wavefront : wavefront.phase, self, phase)


    def set_pixel_scale(self : Wavefront, pixel_scale : Array) -> Wavefront:
        """
        Mutator for the pixel scale.

        Parameters
        ----------
        pixel_scale : Array
            The new pixel scale associated with the wavefront.

        Returns
        -------
        wavefront : Wavefront
            The new Wavefront object with the updated pixel scale.
        """
        assert isinstance(pixel_scale, Array) and pixel_scale.ndim == 0, \
        ("pixel_scale must be a scalar array.")
        return eqx.tree_at(
            lambda wavefront : wavefront.pixel_scale, self, pixel_scale)


    def set_plane_type(self : Wavefront, plane : PlaneType) -> Wavefront:
        """
        Parameters
        ----------
        plane : PlaneType
            A PlaneType object describing the plane that the `Wavefront` is 
            currently in.

        Returns 
        -------
        wavefront : Wavefront 
            The new `Wavefront` with the update plane information.
        """
        assert isinstance(plane, PlaneType), \
        ("plane must be a PlaneType object.")
        return eqx.tree_at(
            lambda wavefront : wavefront.plane_type, self, plane)


    def update_phasor(self      : Wavefront,
                      amplitude : Array,
                      phase     : Array) -> Wavefront:
        """
        Used to update the state of the wavefront. This should typically
        only be called from within a propagator layers in order to ensure
        that values such as pixelscale are updates appropriately. It is 
        assumed that `amplitude` and `phase` have the same shape 
        i.e. `amplitude.shape == phase.shape`. It is not assumed that the 
        shape of the wavefront is  maintained 
        i.e. `self.amplitude.shape == amplitude.shape` is __not__ required. 

        Parameters
        ----------
        amplitude : Array, power
            The electric field amplitudes of the wavefront.
        phase : Array, radians
            The phases of each pixel in the new wavefront.

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` specified by `amplitude` and `phase`
        """
        assert isinstance(amplitude, Array) and amplitude.ndim == 2, \
        ("amplitude must be a 2d array.")
        assert isinstance(phase, Array) and phase.ndim == 2, \
        ("phase must be a 2d array.")
        assert amplitude.shape == phase.shape, \
        ("amplitude and phase arrays must have the same shape.")
        return eqx.tree_at(
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
        Multiply the amplitude of the `Wavefront` by either a float or
        array.

        Parameters
        ----------
        array_like : Array
            An array that has the same dimensions as self.amplitude 
            by which elementwise multiply each pixel. 
            A float to apply to the entire array at once.

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with the applied changes to the 
            amplitude array. 
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
        Add either a float or array of phase to `Wavefront`.

        Parameters
        ----------
        array_like : Array
            The amount of phase to add to the current phase value of 
            each pixel. A scalar modifies the global phase of the 
            wavefront. 

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with the updated array of phases. 
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
        Applies the wavelength-dependent phase based on the supplied 
        optical path difference.

        Parameters
        ----------
        path_difference : Union[float, Array], meters
            The physical optical path difference of either the 
            entire wavefront or each pixel individually. 

        Returns
        -------
        wavefront : Wavefront
            The new wavefront with the phases updated according to 
            `path_difference`
        """
        phase_difference = 2 * np.pi * path_difference / self.get_wavelength()
        return self.add_phase(phase_difference)


    def normalise(self : Wavefront) -> Wavefront:
        """
        Normalises the total power of the wavefront to 1.

        Returns
        -------
        wavefront : Wavefront
            The new wavefront with the normalised electric field 
            amplitudes.
        """
        total_intensity = np.linalg.norm(self.get_amplitude())
        return self.multiply_amplitude(1 / total_intensity)


    def wavefront_to_psf(self             : Wavefront,
                         return_polarised : bool = False) -> Array:
        """
        Calculates the Point Spread Function (PSF), ie the squared modulus
        of the complex wavefront.

        Parameters
        ----------
        return_polarised : bool = False
            returns the raw polarisation matrix if True, else sums the
            polaristaion matrix into a single psf

        Throws
        ------
        error : TypeError
            If `self.amplitude` has not been externally initialised.

        Returns
        -------
        psf : Array
            The PSF of the wavefront.
        """
        # TODO: Make this work properly for polarisation matrices
        # and allows for the returning of the individual polarised wavefront
        # Note this might need to wait until OpticalSystem is divorced from 
        # Optics, since telescope detector do not respond to polarisation
        # effects, and this currently breaks the array formatting.
        # Also note we may not be able to pass a boolean value to return 
        # different array shapes, as a limitation within jax.lax.cond()
        # The solution MAY be add a CombinePolarisation layer, or 
        # some other layer to deal with this
        
        # Sums the first axis for empty polarisation array
        return np.sum(self.get_amplitude() ** 2, axis=0)


    def get_pixel_coordinates(self : Wavefront) -> Array:
        """
        Returns the physical positions of the wavefront pixels in meters

        Returns
        -------
        pixel_positions : Array
            The physical positions of the optical disturbance. 
            Guarantees that `self.get_coordinates().shape == 
            self.amplitude.shape`.
        """
        return dLux.utils.coordinates.get_pixel_coordinates( \
                    self.get_npixels(), self.get_pixel_scale())


    def invert_x_and_y(self : Wavefront) -> Wavefront:
        """
        Reflects the wavefront about both axes. 

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with the phase and amplitude arrays
            reversed along both axes.
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
            The new `Wavefront` with the phase and the amplitude arrays
            reversed along the x axis. 
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
            The new wavefront with the phase and the amplitude arrays 
            reversed along the y axis.
        """
        new_amplitude = np.flip(self.amplitude, axis=-2)
        new_phase = np.flip(self.phase, axis=-2)
        return self.update_phasor(new_amplitude, new_phase)


    def interpolate(self           : Wavefront,
                    coordinates    : Array,
                    real_imaginary : bool = False) -> tuple:
        """
        Interpolates the `Wavefront` at the points specified by 
        coordinates. The default interpolation uses the amplitude 
        and phase, however by passing `real_imgainary=True` 
        the interpolation can be based on the real and imaginary 
        information. The main use of this function is as a helper 
        method to `self.paraxial_interpolate`.

        Parameters
        ----------
        coordinates : Array
            The coordinates to interpolate the optical disturbance
            at. Assumed to have the units meters. 
        real_imaginary : bool
            Whether to use the amplitude-phase or real-imaginary
            representation for the interpolation.

        Returns
        -------
        field : tuple[Array, Array]
            The amplitude and phase of the wavefront at `coordinates`
            based on a linear interpolation.
        """
        if not real_imaginary:
            new_amplitude = map_coordinates(
                self.amplitude, coordinates, order=1)
            new_phase = map_coordinates(
                self.phase, coordinates, order=1)
        else:
            real = map_coordinates(
                self.get_real(), coordinates, order=1)
            imaginary = map_coordinates(
                self.get_imaginary(), coordinates, order=1)
            new_amplitude = np.hypot(real, imaginary)
            new_phase = np.arctan2(imaginary, real)
        return new_amplitude, new_phase


    def paraxial_interpolate(self            : Wavefront,
                             pixel_scale_out : float,
                             npixels         : int,
                             real_imaginary  : bool = False) -> Wavefront: 
        """
        Interpolates the `Wavefront` so that it remains centered on 
        the optical axis. Calculation can be performed using 
        either the real-imaginary or amplitude-phase representations 
        of the wavefront. The default is amplitude-phase. 

        Parameters
        ----------
        pixel_scale_out : float
            The dimensions of a single square pixel after the 
            interpolation.
        npixels : int
            The number of pixels along one side of the square
            `Wavefront` after the interpolation. 
        real_imaginary : bool = False
            Whether to use the real-imaginary representation of the 
            wavefront for the interpolation. 

        Returns
        -------
        wavefront : Wavefront
            The new wavefront with the updated optical disturbance. 
        """
        # Get coords arrays
        npixels_in = self.get_npixels()
        ratio = pixel_scale_out / self.get_pixel_scale()

        centre = (npixels_in  - 1) / 2
        new_centre = (npixels_out - 1) / 2
        pixels = ratio * (-new_centre, new_centre, npixels_out) + centre
        x_pixels, y_pixels = np.meshgrid(pixels, pixels)
        coordinates = np.array([y_pixels, x_pixels])
        new_amplitude, new_phase = self.interpolate(
            coordinates, real_imaginary=real_imaginary)

        # Conserve energy
        new_ampltiude_norm = new_amplitude * ratio

        # Update parameters
        return eqx.tree_at(lambda wavefront:
                 (wavefront.amplitude, wavefront.phase, wavefront.pixel_scale),
                        self, (new_ampltiude_norm, new_phase, pixel_scale_out))


    def pad_to(self : Wavefront, npixels_out : int) -> Wavefront:
        """
        Pads the `Wavefront` with zeros. Assumes that 
        `npixels_out > self.amplitude.shape[-1]`. 
        Note that `Wavefronts` with even pixel dimensions can 
        only be padded (without interpolation) to even pixel 
        dimensions and vice-versa. 

        Parameters
        ----------
        npixels_out : int
            The square side length of the array after it has been 
            zero padded. 

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with the optical disturbance zero 
            padded to the new dimensions.
        """
        npixels_in  = self.get_npixels()
        assert npixels_in  % 2 == npixels_out % 2, \
        ("Only supports even -> even or odd -> odd padding")

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
        Crops the `Wavefront`. Assumes that 
        `npixels_out < self.amplitude.shape[-1]`. 
        `Wavefront`s with an even number of pixels can only 
        be cropped to an even number of pixels without interpolation
        and vice-versa.

        Parameters
        ----------
        npixels_out : int
            The square side length of the array after it has been 
            zero padded. 

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` with the optical disturbance zero 
            cropped to the new dimensions.
        """
        npixels_in  = self.get_npixels()

        assert npixels_in %2 == npixels_out%2, \
        ("Only supports even -> even or 0dd -> odd cropping")

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
    A plane wave extending the abstract `Wavefront` class. 
    Stores phase and amplitude arrays. pixel scale has units of 
    meters/pixel. Assumes the wavefront is square. This is Cartesian 
    as opposed to Angular, because there are no infinite distances.  
    """


    def __init__(self        : Wavefront, 
                 wavelength  : Array, 
                 offset      : Array,
                 diameter    : Array,
                 plane_type  : PlaneType,
                 amplitude   : Array, 
                 phase       : Array) -> Wavefront:
        """
        Initialises a minimal `Wavefront` specified only by the 
        wavelength and offset.

        Parameters
        ----------
        wavelength : float, meters
            The wavelength of the `Wavefront`.
        offset : Array, radians
            The (x, y) angular offset of the `Wavefront` from 
            the optical axis.
        amplitude : Array, power
            The electric field amplitude of the `Wavefront`. 
        phase : Array, radians
            The electric field phase of the `Wavefront`.
        diameter : float, meters
            The physical dimensions of each square pixel.
        plane_type : enum.IntEnum.PlaneType
            The current plane of wavefront, can be Pupil, Focal
        """
        super().__init__(wavelength, offset, diameter, 
                         plane_type, amplitude, phase)


class AngularWavefront(Wavefront):
    """
    A plane wave extending the abstract `Wavefront` class. 
    Stores phase and amplitude arrays. pixel scale has units of 
    meters per pixel in pupil planes and radians per pixel in 
    focal planes. Assumes the wavefront is square.
    """


    def __init__(self        : Wavefront, 
                 wavelength  : Array, 
                 offset      : Array,
                 diameter    : Array,
                 plane_type  : PlaneType,
                 amplitude   : Array, 
                 phase       : Array) -> Wavefront:
        """
        Initialises a minimal `Wavefront` specified only by the 
        wavelength and offset.

        Parameters
        ----------
        wavelength : float, meters
            The wavelength of the `Wavefront`.
        offset : Array, radians
            The (x, y) angular offset of the `Wavefront` from 
            the optical axis.
        amplitude : Array, power
            The electric field amplitude of the `Wavefront`. 
        phase : Array, radians
            The electric field phase of the `Wavefront`.
        diameter : float, meters or radians
            The physical dimensions of each square pixel. Units 
            are meters in Pupil planes and radians
            in Focal planes.
        plane_type : enum.IntEnum.PlaneType
            The current plane of wavefront, can be Pupil, Focal
        """
        super().__init__(wavelength, offset, diameter, 
                         plane_type, amplitude, phase)


class FarFieldFresnelWavefront(Wavefront):
    """
    A plane wave extending the abstract `Wavefront` class. 
    Stores phase and amplitude arrays. pixel scale has units of 
    meters/pixel. Assumes the wavefront is square. This is FarFieldFresnel 
    as it is can be only represented in the far-field Fresnel approximation.
    """


    def __init__(self        : Wavefront, 
                 wavelength  : Array, 
                 offset      : Array,
                 diameter    : Array,
                 plane_type  : PlaneType,
                 amplitude   : Array, 
                 phase       : Array) -> Wavefront:
        """
        Initialises a minimal `Wavefront` specified only by the 
        wavelength and offset.

        Parameters
        ----------
        wavelength : float, meters
            The wavelength of the `Wavefront`.
        offset : Array, radians
            The (x, y) angular offset of the `Wavefront` from 
            the optical axis.
        amplitude : Array, power
            The electric field amplitude of the `Wavefront`. 
        phase : Array, radians
            The electric field phase of the `Wavefront`.
        diameter : float, meters
            The physical dimensions of each square pixel.
        plane_type : enum.IntEnum.PlaneType
            The current plane of wavefront, can be Pupil, Focal
        """
        super().__init__(wavelength, offset, diameter, 
                         plane_type, amplitude, phase)


    def transfer_function(self : Wavefront, 
            distance : float) -> float:
        """
        The Optical Transfer Function corresponding to the 
        evolution of the wavefront when propagating a distance.

        Parameters
        ----------
        distance : float
            The distance that is getting propagated in meters.
        Returns
        -------
        phase : Array
            The phase that represents the optical transfer. 
        """
        wavenumber = 2. * np.pi / self.get_wavelength()
        return np.exp(1.0j * wavenumber * distance)