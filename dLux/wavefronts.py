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
    A class representing some wavefront, designed to track the various
    parameters such as wavelength, pixel_scale, amplitude and phase, as well as
    two helper parameters, plane and units.

    All wavefronts currently only support square amplitude and phase arrays.

    Attributes
    ----------
    wavelength : float, metres
        The wavelength of the `Wavefront`.
    amplitude : Array, power
        The electric field amplitude of the `Wavefront`.
    phase : Array, radians
        The electric field phase of the `Wavefront`.
    pixel_scale : float, metres/pixel or radians/pixel
        The physical dimensions of the pixels representing the wavefront. This
        can be in units of either metres per pixel or radians per pixel
        depending on if 'unit' is 'Cartesian' or 'Angular'.
    plane : str
        The current plane type of wavefront, can be 'Pupil', 'Focal' or
        'Intermediate'.
    units : str
        The current units of the wavefront, can be 'Cartesian' or 'Angular'.
    """

    wavelength: Array
    pixel_scale: Array
    amplitude: Array
    phase: Array
    plane: str
    units: str

    def __init__(
        self: Wavefront, npixels: int, diameter: Array, wavelength: Array
    ):
        """
        Constructor for the wavefront.

        Parameters
        ----------
        npixels : int
            The number of pixels that represent the `Wavefront`.
        diameter : float, metres
            The physical dimensions of each square pixel.
        wavelength : float, metres
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
        Returns the current wavefront diameter calculated using the pixel scale
        and number of pixels.

        Returns
        -------
        diameter : Array, metres or radians
            The current diameter of the wavefront.
        """
        return self.npixels * self.pixel_scale

    @property
    def npixels(self: Wavefront) -> int:
        """
        Returns the side length of the arrays currently representing the
        wavefront. Taken from the last axis of the amplitude array.

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
        Returns the physical positions of the wavefront pixels in metres.

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

        # TODO Units check
        Returns
        -------
        fringe_size : Array, radians
            The wavenumber of the wavefront.
        """
        return self.wavelength / self.diameter

    @property
    def ndim(self: Wavefront) -> int:
        """
        Returns the number of dimensions of the wavefront.
        """
        return self.pixel_scale.ndim

    #################
    # Magic Methods #
    #################
    def __add__(self: Wavefront, other: Any) -> Wavefront:
        """
        Magic method used to give a simple API for interaction with different
        layer types and arrays. If the input 'other' in an array it is treated
        as an array of OPD values and is added to the wavefront. If it is an
        Aberration, the wavefront is passed to the layer and the output
        wavefront is returned.

        Parameters
        ----------
        other : Array or Aberration
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
        Magic method used to give a simple API for interaction with different
        layer types and arrays. If the input 'other' in an array it is treated
        as an array of OPD values and is added to the wavefront. If it is an
        Aberration, the wavefront is passed to the layer and the output
        wavefront is returned.

        Parameters
        ----------
        other : Array or Aberration
            The input to add to the wavefront.

        Returns
        -------
        wavefront : Wavefront
            The output wavefront.
        """
        return self.__add__(other)

    def __mul__(self: Wavefront, other: Any) -> Wavefront:
        """
        Magic method used to give a simple API for interaction with different
        layer types and arrays. If the input 'other' in an array it is treated
        as an array of transmission values and is multiplied by the wavefront
        amplitude. If it is an Aperture, Aberration, or Propagator, the
        wavefront is passed to the layer and the output wavefront is returned.

        Parameters
        ----------
        other : Array or Aberration or Aperture or Propagator
            The input to add to the wavefront.

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
        Magic method used to give a simple API for interaction with different
        layer types and arrays. If the input 'other' in an array it is treated
        as an array of transmission values and is multiplied by the wavefront
        amplitude. If it is an Aperture, Aberration, or Propagator, the
        wavefront is passed to the layer and the output wavefront is returned.

        Parameters
        ----------
        other : Array or Aberration or Aperture or Propagator
            The input to add to the wavefront.

        Returns
        -------
        wavefront : Wavefront
            The output wavefront.
        """
        return self.__mul__(other)

    ###################
    # Adder Functions #
    ###################
    def add_opd(self: Wavefront, path_difference: Array) -> Wavefront:
        """
        Applies the wavelength-dependent phase based on the supplied optical
        path difference.

        Parameters
        ----------
        path_difference : Array, metres
            The physical optical path difference of either the entire wavefront
            or each pixel individually.

        Returns
        -------
        wavefront : Wavefront
            The new wavefront with the phases updated according to the supplied
            path_difference
        """
        phase_difference = self.wavenumber * path_difference
        return self.add("phase", phase_difference)

    def add_phase(self: Wavefront, phase: Array) -> Wavefront:
        """
        Applies input array to the phase of the wavefront.

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
        Tilts the wavefront by the angles in the (x, y) by modifying the
        phase arrays.

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
            The new wavefront with the normalised electric field amplitudes.
        """
        return self.divide("amplitude", np.linalg.norm(self.amplitude))

    def _to_field(self: Wavefront, complex: bool = False) -> Array:
        """
        Returns the wavefront in either (amplitude, phase) or (real, imaginary)
        form.

        Parameters
        ----------
        complex : bool = False
            Whether to return the wavefront in (real, imaginary) form.

        Returns
        -------
        field : Array
            The wavefront in either (amplitude, phase) or (real, imaginary)
            form.
        """
        if complex:
            return np.array([self.real, self.imaginary])
        return np.array([self.amplitude, self.phase])

    def _to_amplitude_phase(self: Wavefront, field: Array) -> Array:
        """
        Returns the input field in (real, imaginary) (amplitude, phase) form.

        Parameters
        ----------
        field : Array
            The wavefront field in (amplitude, phase) form.

        Returns
        -------
        field : Array
            The wavefront field in (real, imaginary) form.
        """
        amplitude = np.hypot(field[0], field[1])
        phase = np.arctan2(field[1], field[0])
        return np.array([amplitude, phase])

    def flip(self: Wavefront, axis: tuple) -> Wavefront:
        """
        Flips the amplitude and phase of the wavefront along the specified
        axes.

        Parameters
        ----------
        axis : tuple
            The axes along which to flip the wavefront.

        Returns
        -------
        wavefront : Wavefront
            The new wavefront with the flipped amplitude and phase.
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
        Performs a paraxial interpolation on the wavefront, determined by the
        pixel_scale_out and npixels parameters. The transformation is done
        on the amplitude and phase arrays, but can be done on the real and
        imaginary components by passing `complex=True`.

        Parameters
        ----------
        npixels : int
            The number of pixels representing the wavefront after the
            interpolation.
        pixel_scale: Array
            The pixel scale of the array after the interpolation.
        complex : bool = False
            Whether to rotate the real and imaginary representation of the
            wavefront as opposed to the amplitude and phase representation.

        Returns
        -------
        wavefront : Wavefront
            The new wavefront interpolated to the size and shape determined by
            npixels and pixel_scale_out, with the updated pixel_scale.
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
        Performs a paraxial rotation on the wavefront, determined by the
        angle parameter, using interpolation. The transformation is done
        on the amplitude and phase arrays, but can be done on the real and
        imaginary components by passing `complex=True`.

        Parameters
        ----------
        angle : Array, radians
            The angle by which to rotate the wavefront in a clockwise
            direction.
        order : int = 1
            The interpolation order to use.
        complex : bool = False
            Whether to rotate the real and imaginary representation of the
            wavefront as opposed to the amplitude and phase representation.

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
        Paraxially resizes the `Wavefront` to the size determined by
        npixels. To ensure that no output arrays are non-paraxial even shaped
        arrays can only be resized to even shapes, and odd shaped arrays can
        only be resized to odd shapes. i.e. 4 -> 2 or 5 -> 3.

        Parameters
        ----------
        npixels : int
            The size to output the array.

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
    def _prep_prop(self, focal_length: float):
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
        pad : int = 2
            The padding factory to apply to the input wavefront before
            performing the FFT.
        focal_length : Array = None
            The focal length of the propagation. If None, the propagation is
            treated as an 'angular' propagation, else it is treated as a
            'Cartesian' propagation.

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
    def _MFT(self, phasor, wavelength, pixel_scale, *args):
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

        Parameters
        ----------
        npixels : int
            The number of pixels in the output wavefront.
        pixel_scale : Array
            The pixel scale of the output wavefront.
        focal_length : Array = None
            The focal length of the propagation. If None, the propagation is
            treated as an 'angular' propagation, else it is treated as a
            'Cartesian' propagation.
        shift : Array = np.zeros(2)
            The shift in the center of the output plane.
        pixel : bool = True
            Whether the shift is in pixels or the units of pixel_scale.
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

        Parameters
        ----------
        npixels : int
            The number of pixels in the output wavefront.
        pixel_scale : Array
            The pixel scale of the output wavefront.
        focal_length : Array = None
            The focal length of the propagation. If None, the propagation is
            treated as an 'angular' propagation, else it is treated as a
            'Cartesian' propagation.
        shift : Array = np.zeros(2)
            The shift in the center of the output plane.
        pixel : bool = True
            Whether the shift is in pixels or the units of pixel_scale.
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
