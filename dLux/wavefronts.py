from __future__ import annotations
import jax.numpy as np
from typing import Any
from jax import vmap, Array
from zodiax import Base
import dLux.utils as dlu
import dLux

# Optical layers require Wavefront to init, so we alias it here to avoid MRO issues
OpticalLayer = lambda: dLux.layers.optical_layers.OpticalLayer


__all__ = ["Wavefront"]


class Wavefront(Base):
    """
    A simple class to hold the state of some wavefront as it is transformed and
    propagated throughout an optical system. All wavefronts assume square arrays.

    Attributes
    ----------
    wavelength : float, meters
        The wavelength of the `Wavefront`.
    amplitude : Array, power
        The electric field amplitude of the `Wavefront`.
    phase : Array, radians
        The electric field phase of the `Wavefront`.
    pixel_scale : float, meters/pixel or radians/pixel
        The pixel scale of the phase and amplitude arrays. If `units='Cartesian'` then
        the pixel scale is in meters/pixel, else if `units='Angular'` then the pixel
        scale is in radians/pixel.
    plane : str
        The current plane type of wavefront, can be 'Pupil', 'Focal' or 'Intermediate'.
    units : str
        The current units of the wavefront, can be 'Cartesian' or 'Angular'.
    """

    wavelength: float
    pixel_scale: float
    amplitude: Array
    phase: Array
    plane: str
    units: str

    def __init__(
        self: Wavefront, npixels: int, diameter: float, wavelength: float
    ):
        """
        Parameters
        ----------
        npixels : int
            The number of pixels that represent the `Wavefront`.
        diameter : float, meters
            The total diameter of the `Wavefront`.
        wavelength : float, meters
            The wavelength of the `Wavefront`.
        """
        self.wavelength = np.asarray(wavelength, float)
        self.pixel_scale = np.asarray(diameter / npixels, float)
        self.amplitude = (
            np.ones((npixels, npixels), dtype=float) / npixels**2
        )
        self.phase = np.zeros((npixels, npixels), dtype=float)

        # Always initialised in Pupil plane with Cartesian Coords
        self.plane = "Pupil"
        self.units = "Cartesian"

    ####################
    # Getter Functions #
    ####################
    @property
    def diameter(self: Wavefront) -> Array:
        """
        Returns the current wavefront diameter calculated using the pixel scale and
        number of pixels.

        Returns
        -------
        diameter : Array, meters or radians
            The current diameter of the wavefront.
        """
        return self.npixels * self.pixel_scale

    @property
    def npixels(self: Wavefront) -> int:
        """
        Returns the side length of the arrays currently representing the wavefront.
        Taken from the last axis of the amplitude array.

        Returns
        -------
        pixels : int
            The number of pixels that represent the `Wavefront`.
        """
        return self.amplitude.shape[-1]

    @property
    def real(self: Wavefront) -> Array:
        """
        Returns the real component of the `Wavefront`.

        Returns
        -------
        wavefront : Array
            The real component of the `Wavefront` phasor.
        """
        return self.amplitude * np.cos(self.phase)

    @property
    def imaginary(self: Wavefront) -> Array:
        """
        Returns the imaginary component of the `Wavefront`.

        Returns
        -------
        wavefront : Array
            The imaginary component of the `Wavefront` phasor.
        """
        return self.amplitude * np.sin(self.phase)

    @property
    def phasor(self: Wavefront) -> Array:
        """
        The electric field phasor described by this Wavefront in complex form.

        Returns
        -------
        field : Array
            The electric field phasor of the wavefront.
        """
        return self.amplitude * np.exp(1j * self.phase)

    @property
    def psf(self: Wavefront) -> Array:
        """
        Calculates the Point Spread Function (PSF), i.e. the squared modulus
        of the complex wavefront.

        Returns
        -------
        psf : Array
            The PSF of the wavefront.
        """
        return self.amplitude**2

    @property
    def coordinates(self: Wavefront) -> Array:
        """
        Returns the physical positions of the wavefront pixels in meters.

        Returns
        -------
        coordinates : Array
            The coordinates of the centers of each pixel representing the
            wavefront.
        """
        return dlu.pixel_coords(self.npixels, self.diameter)

    @property
    def wavenumber(self: Wavefront) -> Array:
        """
        Returns the wavenumber of the wavefront (2 * pi / wavelength).

        Returns
        -------
        wavenumber : Array, 1/meters
            The wavenumber of the wavefront.
        """
        return 2 * np.pi / self.wavelength

    @property
    def fringe_size(self: Wavefront) -> Array:
        """
        Returns the size of the fringes in angular units.

        TODO Units check from focal plane
        Returns
        -------
        fringe_size : Array, radians
            The size of the linear diffraction fringe of the wavefront.
        """
        return self.wavelength / self.diameter

    @property
    def ndim(self: Wavefront) -> int:
        """
        Returns the number of 'dimensions' of the wavefront. This is used to track the
        vectorised version of the wavefront returned from vmapping.

        Returns
        -------
        ndim : int
            The 'dimensionality' of dimensions of the wavefront.
        """
        return self.pixel_scale.ndim

    #################
    # Magic Methods #
    #################
    def __add__(self: Wavefront, other: Any) -> Wavefront:
        """
        Adds the input 'other' to the wavefront. If the input is a numeric type, it is
        treated as an OPD, else if it is an optical layer, it will be applied to the
        wavefront.

        Parameters
        ----------
        other : Any
            The input to add to the wavefront.

        Returns
        -------
        wavefront : Wavefront
            The output wavefront.
        """
        # None Type
        if other is None:
            return self

        # Some Optical Layer
        if isinstance(other, OpticalLayer()):
            return other.apply(self)

        # Array based inputs - Defaults to OPD
        if isinstance(other, (Array, float, int)):
            return self.add_opd(other)

        # Other
        else:
            raise TypeError(
                "Can only add an array or OpticalLayer to "
                f"Wavefront. Got: {type(other)}."
            )

    def __iadd__(self: Wavefront, other: Any) -> Wavefront:
        """
        Provides the += operator for the wavefront, calling the __add__ method.

        Parameters
        ----------
        other : Any
            The input to add to the wavefront.

        Returns
        -------
        wavefront : Wavefront
            The output wavefront.
        """
        return self.__add__(other)

    def __mul__(self: Wavefront, other: Any) -> Wavefront:
        """
        Multiplies the input 'other' to the wavefront. If the input is a numeric type,
        it is treated as an array of transmission values and is multiplied by the
        wavefront amplitude, unless it is a complex number, in which case it will be
        multiplied with the wavefront phasor. If it is an optical layer, it will be
        applied to the wavefront.

        Parameters
        ----------
        other : Any
            The input to multiply with the wavefront.

        Returns
        -------
        wavefront : Wavefront
            The output wavefront.
        """
        # None Type, return None
        if other is None:
            return self

        # Some Optical Layer, apply it
        if isinstance(other, OpticalLayer()):
            return other.apply(self)

        # Array based inputs
        if isinstance(other, (Array, float, int)):
            # Complex array - Multiply the phasors
            if isinstance(other, Array) and other.dtype.kind == "c":
                phasor = self.phasor * other
                return self.set(
                    ["amplitude", "phase"], [np.abs(phasor), np.angle(phasor)]
                )

            # Scalar array - Multiply amplitude
            else:
                return self.multiply("amplitude", other)

        # Other
        else:
            raise TypeError(
                "Can only multiply Wavefront by array or "
                f"OpticalLayer. Got: {type(other)}."
            )

    def __imul__(self: Wavefront, other: Any) -> Wavefront:
        """
        Provides the *= operator for the wavefront, calling the __mul__ method.

        Parameters
        ----------
        other : Any
            The input to multiply with the wavefront.

        Returns
        -------
        wavefront : Wavefront
            The output wavefront.
        """
        return self.__mul__(other)

    ###################
    # Adder Functions #
    ###################
    def add_opd(self: Wavefront, opd: Array) -> Wavefront:
        """
        Adds an optical path difference (OPD) to the wavefront.

        Parameters
        ----------
        opd : Array, meters
            The opd to add to the wavefront.

        Returns
        -------
        wavefront : Wavefront
            The new wavefront with the phases updated according to the supplied opd.
        """
        return self.add("phase", self.wavenumber * opd)

    def add_phase(self: Wavefront, phase: Array) -> Wavefront:
        """
        Adds a phase to the wavefront.

        Parameters
        ----------
        phase : Array, radians
            The phase to be added to the wavefront.

        Returns
        -------
        wavefront : Wavefront
            The new wavefront with updated phases.
        """
        # Add this extra None check to allow PhaseOptics to have a None phase
        # and still be able to be 'added' to it, making this the phase
        # equivalent of `wf += opd` -> `wf = wf.add_phase(phase)`
        if phase is not None:
            return self.add("phase", phase)
        return self

    ###################
    # Other Functions #
    ###################
    def tilt(self: Wavefront, angles: Array) -> Wavefront:
        """
        Tilts the wavefront by the (x, y) angles.

        Parameters
        ----------
        angles : Array, radians
            The (x, y) angles by which to tilt the wavefront.

        Returns
        -------
        wavefront : Wavefront
            The tilted wavefront.
        """
        if not isinstance(angles, Array) or angles.shape != (2,):
            raise ValueError("angles must be an array of shape (2,).")
        opd = -(angles[:, None, None] * self.coordinates).sum(0)
        return self.add_opd(opd)

    def normalise(self: Wavefront) -> Wavefront:
        """
        Normalises the total power of the wavefront to 1.

        Returns
        -------
        wavefront : Wavefront
            The normalised wavefront.
        """
        return self.divide("amplitude", np.linalg.norm(self.amplitude))

    def _to_field(self: Wavefront, complex: bool = False) -> Array:
        """
        Returns the wavefront in either (amplitude, phase) or (real, imaginary) form.

        Parameters
        ----------
        complex : bool = False
            Whether to return the wavefront in (real, imaginary) form.

        Returns
        -------
        field : Array
            The wavefront in either (amplitude, phase) or (real, imaginary) form.
        """
        if complex:
            return np.array([self.real, self.imaginary])
        return np.array([self.amplitude, self.phase])

    def _to_amplitude_phase(self: Wavefront, field: Array) -> Array:
        """
        Transforms the input field in (real, imaginary) to (amplitude, phase) form.

        Parameters
        ----------
        field : Array
            The wavefront field in (real, imaginary) form.

        Returns
        -------
        field : Array
            The wavefront field in (amplitude, phase) form.
        """
        amplitude = np.hypot(field[0], field[1])
        phase = np.arctan2(field[1], field[0])
        return np.array([amplitude, phase])

    def flip(self: Wavefront, axis: tuple) -> Wavefront:
        """
        Flips the wavefront along the specified axes. Note we use 'ij' indexing, so
        axis 0 is the y-axis and axis 1 is the x-axis.

        Parameters
        ----------
        axis : tuple
            The axes along which to flip the wavefront.

        Returns
        -------
        wavefront : Wavefront
            The new flipped wavefront.
        """
        field = self._to_field()
        flipper = vmap(np.flip, (0, None))
        amplitude, phase = flipper(field, axis)
        return self.set(["amplitude", "phase"], [amplitude, phase])

    def scale_to(
        self: Wavefront,
        npixels: int,
        pixel_scale: Array,
        complex: bool = False,
    ) -> Wavefront:
        """
        Interpolated the wavefront to a given npixels and pixel_scale. Can be done on
        the real and imaginary components by passing in complex=True.

        Parameters
        ----------
        npixels : int
            The number of pixels  to interpolate to.
        pixel_scale: Array
            The pixel scale to interpolate to.
        complex : bool = False
            Whether to rotate the real and imaginary representation of the wavefront as
            opposed to the amplitude and phase representation.

        Returns
        -------
        wavefront : Wavefront
            The new interpolated wavefront.
        """
        # Get field in either (amplitude, phase) or (real, imaginary)
        field = self._to_field(complex=complex)

        # Scale the field
        scale_fn = vmap(dlu.scale, (0, None, None))
        field = scale_fn(field, npixels, pixel_scale / self.pixel_scale)

        # Cast back to (amplitude, phase) if needed
        if complex:
            field = self._to_amplitude_phase(field)

        # Return new wavefront
        return self.set(
            ["amplitude", "phase", "pixel_scale"],
            [field[0], field[1], pixel_scale],
        )

    def rotate(
        self: Wavefront, angle: Array, order: int = 1, complex: bool = False
    ) -> Wavefront:
        """
        Rotates the wavefront by a given angle via interpolation. Can be done on the
        real and imaginary components by passing in complex=True.

        Parameters
        ----------
        angle : Array, radians
            The angle by which to rotate the wavefront in a clockwise
            direction.
        order : int = 1
            The interpolation order to use.
        complex : bool = False
            Whether to rotate the real and imaginary representation of the wavefront as
            opposed to the amplitude and phase representation.

        Returns
        -------
        wavefront : Wavefront
            The new wavefront rotated by angle in the clockwise direction.
        """
        # Get field in either (amplitude, phase) or (real, imaginary)
        field = self._to_field(complex=complex)

        # Rotate the field
        rotator = vmap(dlu.rotate, (0, None, None))
        field = rotator(field, angle, order)

        # Cast back to (amplitude, phase) if needed
        if complex:
            field = self._to_amplitude_phase(field)

        # Return new wavefront
        return self.set(["amplitude", "phase"], [field[0], field[1]])

    def resize(self: Wavefront, npixels: int) -> Wavefront:
        """
        Resizes the wavefront via a zero-padding or cropping operation.

        Parameters
        ----------
        npixels : int
            The size to resize the wavefront to.

        Returns
        -------
        wavefront : Wavefront
            The resized wavefront.
        """
        field = self._to_field()
        amplitude, phase = vmap(dlu.resize, (0, None))(field, npixels)
        return self.set(["amplitude", "phase"], [amplitude, phase])

    #########################
    # Propagation Functions #
    #########################
    def _prep_prop(self: Wavefront, focal_length) -> tuple:
        """
        Determines the propagation direction, output plane and output units.

        Parameters
        ----------
        focal_length : Union[float, None]
            The focal length of the propagation.

        Returns
        -------
        inverse : bool
            Whether the propagation is inverse or not.
        plane : str
            The output plane of the propagation.
        units : str
            The output units of the propagation.
        """
        # Determine propagation direction, output plane and output units
        if self.plane == "Pupil":
            inverse = False
            plane = "Focal"
            if focal_length is None:
                units = "Angular"
            else:
                units = "Cartesian"
        else:
            if focal_length is not None and self.units == "Angular":
                raise ValueError(
                    "focal_length can not be specific when"
                    "propagating from a Focal plane with angular units."
                )
            inverse = True
            plane = "Pupil"
            units = "Cartesian"

        return inverse, plane, units

    def propagate_FFT(
        self: Wavefront,
        focal_length: float = None,
        pad: int = 2,
    ) -> Wavefront:
        """
        Propagates the wavefront by performing a Fast Fourier Transform.

        Parameters
        ----------
        focal_length : float = None
            The focal length of the propagation. If None, the output pixel scale has
            units of radians, else meters.
        pad : int = 2
            The padding factory to apply to the input wavefront before the FFT.

        Returns
        -------
        wavefront : Wavefront
            The propagated wavefront.
        """
        inverse, plane, units = self._prep_prop(focal_length)

        # Calculate
        phasor, pixel_scale = dlu.FFT(
            self.phasor,
            self.wavelength,
            self.pixel_scale,
            focal_length,
            pad,
            inverse,
        )

        # Return new wavefront
        return self.set(
            ["amplitude", "phase", "pixel_scale", "plane", "units"],
            [np.abs(phasor), np.angle(phasor), pixel_scale, plane, units],
        )

    # TODO: Class method this?
    def _MFT(
        self: Wavefront,
        phasor: Array,
        wavelength: float,
        pixel_scale: float,
        *args: tuple,
    ) -> Array:
        """
        Simple alias for the MFT function to allow for vectorisation over phasors,
        wavelengths, pixel_scales, etc.

        Parameters
        ----------
        phasor : Array
            The phasor to propagate.
        wavelength : float
            The wavelength of the wavefront.
        pixel_scale : float
            The pixel scale of the wavefront.
        args : tuple
            The propagation arguments.

        Returns
        -------
        phasor : Array
            The propagated phasor.
        """
        return dlu.MFT(phasor, wavelength, pixel_scale, *args)

    def propagate(
        self: Wavefront,
        npixels: int,
        pixel_scale: float,
        focal_length: float = None,
        shift: Array = np.zeros(2),
        pixel: bool = True,
    ) -> Wavefront:
        """
        Propagates the wavefront by performing an MFT, allowing for the output pixel
        scale and npixels to be specified.

        Parameters
        ----------
        npixels : int
            The number of pixels in the output plane.
        pixel_scale : float, meters/pixel or radians/pixel
            The pixel scale of the output plane.
        focal_length : float = None
            The focal length of the propagation. If None, the propagation is angular
            and pixel_scale_out is taken in as radians/pixel, else meters/pixel.
        shift : Array = np.zeros(2)
            The shift in the center of the output plane.
        pixel : bool = True
            Should the shift be taken in units of pixels, or pixel scale.

        Returns
        -------
        wavefront : Wavefront
            The propagated wavefront.
        """
        inverse, plane, units = self._prep_prop(focal_length)

        # Enforce array so output can be vectorised by vmap
        pixel_scale = np.asarray(pixel_scale, float)

        # Calculate
        # Using a self._MFT here allows for broadband wavefronts to define
        # vectorised propagation fn over phasors, wavels, px_scales, etc.
        # It also makes the code muuuuch nicer to read
        args = (npixels, pixel_scale, focal_length, shift, pixel, inverse)
        phasor = self._MFT(
            self.phasor, self.wavelength, self.pixel_scale, *args
        )

        # Update
        return self.set(
            ["amplitude", "phase", "pixel_scale", "plane", "units"],
            [np.abs(phasor), np.angle(phasor), pixel_scale, plane, units],
        )

    # # TODO: Class method this?
    # def _fresnel(self, phasor, wavelength, pixel_scale, focal_shift, *args):
    #     return dlu.fresnel_MFT(phasor, wavelength, pixel_scale, *args)

    def propagate_fresnel(
        self: Wavefront,
        npixels: int,
        pixel_scale: float,
        focal_length: float,
        focal_shift: float = 0.0,
        shift: Array = np.zeros(2),
        pixel: bool = True,
    ) -> Wavefront:
        """
        Propagates the phasor using a Far-Field Fresnel propagation. This allows for
        psfs to be better modelled a few wavelengths from the focal plane.

        Parameters
        ----------
        npixels : int
            The number of pixels in the output plane.
        pixel_scale : float, meters/pixel or radians/pixel
            The pixel scale of the output plane.
        focal_length : float
            The focal length of the propagation.
        focal_shift: float, meters
            The shift from focus to propagate to.
        shift : Array = np.zeros(2)
            The shift in the center of the output plane.
        pixel : bool = True
            Should the shift be taken in units of pixels, or pixel scale.
        inverse: bool = False
            Is this a forward or inverse propagation.

        Returns
        -------
        wavefront : Wavefront
            The propagated wavefront.
        """
        # TODO: Try inverse propagation to see if it works, it probably will
        if self.plane == "Pupil":
            inverse = False
        else:
            inverse = True
        plane = "Intermediate"
        units = "Cartesian"

        # We can't fresnel from a focal plane
        if self.plane != "Pupil":
            raise ValueError(
                "Can only do an fresnel propagation from a Pupil plane, "
                f"current plane is {self.plane}."
            )

        # Calculate
        phasor = dlu.fresnel_MFT(
            self.phasor,
            self.wavelength,
            self.pixel_scale,
            npixels,
            pixel_scale,
            focal_length,
            focal_shift,
            shift,
            pixel,
            inverse,
        )

        # Update
        return self.set(
            ["amplitude", "phase", "pixel_scale", "plane", "units"],
            [np.abs(phasor), np.angle(phasor), pixel_scale, plane, units],
        )
