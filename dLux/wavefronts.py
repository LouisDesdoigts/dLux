from __future__ import annotations
import jax.numpy as np
from typing import Any
from jax import vmap, Array
from zodiax import Base
import dLux.utils as dlu
import dLux

__all__ = ["Wavefront", "FresnelWavefront"]

Aberration = lambda: dLux.optical_layers.AberrationLayer
Aperture = lambda: dLux.apertures.ApertureLayer
Propagator = lambda: dLux.propagators.Propagator
OpticalLayer = lambda: dLux.optical_layers.OpticalLayer


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
        self.wavelength = np.asarray(wavelength, dtype=float)
        self.pixel_scale = np.asarray(diameter / npixels, dtype=float)
        self.amplitude = np.ones((npixels, npixels), dtype=float)
        self.phase = np.zeros((npixels, npixels), dtype=float)

        # Input checks
        if self.wavelength.shape != ():
            raise ValueError("wavelength must have shape ().")
        if self.diameter.shape != ():
            raise ValueError("diameter must have shape ().")

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
        return dlu.pixel_coords(self.npixels, self.pixel_scale)

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
        # TODO Possibly output in unit based on units attribute
        # TODO make methods use this
        Returns
        -------
        fringe_size : Array, radians
            The wavenumber of the wavefront.
        """
        return self.wavelength / self.diameter

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
            return other(self)

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
            return other(self)

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
        scaler = vmap(dlu.scale, (0, None, None))
        field = scaler(field, npixels, pixel_scale / self.pixel_scale)

        # Cast back to (amplitude, phase) if needed
        if complex:
            field = self._to_amplitude_phase(field)

        # Return new wavefront
        return self.set(["amplitude", "phase"], [field[0], field[1]])

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

    ########################
    # Padding and Cropping #
    ########################
    def pad_to(self: Wavefront, npixels: int) -> Wavefront:
        """
        Paraxially zero-pads the `Wavefront` to the size determined by
        npixels. Note this only supports padding arrays of even dimension
        to even dimension, and odd dimension to odd dimension, i.e. 2 -> 4 or
        3 -> 5.

        Parameters
        ----------
        npixels : int
            The size of the array to pad to the wavefront to.

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` zero-padded to the size npixels.
        """
        npixels_in = self.npixels
        if npixels_in % 2 != npixels % 2:
            raise ValueError("Only supports even -> even or odd -> odd input.")
        if npixels < npixels_in:
            raise ValueError(
                "npixels must be larger than the current array "
                "size: {}".format(npixels_in)
            )

        new_centre = npixels // 2
        centre = npixels_in // 2
        remainder = npixels_in % 2
        padded = np.zeros([npixels, npixels])

        amplitude = padded.at[
            new_centre - centre : centre + new_centre + remainder,
            new_centre - centre : centre + new_centre + remainder,
        ].set(self.amplitude)
        phase = padded.at[
            new_centre - centre : centre + new_centre + remainder,
            new_centre - centre : centre + new_centre + remainder,
        ].set(self.phase)
        return self.set(["amplitude", "phase"], [amplitude, phase])

    def crop_to(self: Wavefront, npixels: int) -> Wavefront:
        """
        Paraxially crops the `Wavefront` to the size determined by npixels.
        Note this only supports padding arrays of even dimension to even
        dimension, and odd dimension to odd dimension, i.e. 4 -> 2 or 5 -> 3.

        Parameters
        ----------
        npixels : int
            The size of the array to crop to the wavefront to.

        Returns
        -------
        wavefront : Wavefront
            The new `Wavefront` cropped to the size npixels.
        """
        npixels_in = self.npixels

        if npixels_in % 2 != npixels % 2:
            raise ValueError("Only supports even -> even or odd -> odd input.")
        if npixels > npixels_in:
            raise ValueError(
                "npixels must be smaller than the current array "
                "size: {}".format(npixels_in)
            )

        new_centre = npixels_in // 2
        centre = npixels // 2

        amplitude = self.amplitude[
            new_centre - centre : new_centre + centre,
            new_centre - centre : new_centre + centre,
        ]
        phase = self.phase[
            new_centre - centre : new_centre + centre,
            new_centre - centre : new_centre + centre,
        ]

        return self.set(["amplitude", "phase"], [amplitude, phase])

    #########################
    # Propagation Functions #
    #########################
    def _FFT_output(
        self: Wavefront,
        pad_factor: int = 1,
        focal_length: Array = None,
        inverse: bool = False,
    ) -> tuple:
        """
        Calculates the output plane, unit, and pixel scale.

        Parameters
        ----------
        focal_length : Array = None
            The focal length of the propagation. If None, the propagation is
            treated as an 'angular' propagation, else it is treated as a
            'Cartesian' propagation.
        inverse : bool = False
            If True, the propagation is treated as an inverse FFT.

        Returns
        -------
        plane : str
            The output plane of the propagation.
        units : str
            The units of the output plane.
        pixel_scale : Array
            The pixel scale of the output plane.
        """
        pixel_scale = self.fringe_size / pad_factor
        if focal_length is None:
            units = "Angular"
        else:
            units = "Cartesian"
            pixel_scale *= focal_length

            # Check for invalid propagation
            if self.units == "Angular":
                raise ValueError(
                    "focal_length can not be specific when"
                    "propagating from a Focal plane with angular units."
                )

        # Check planes
        if inverse:
            if self.plane != "Focal":
                raise ValueError(
                    "Can only do an IFFT from a Focal plane, "
                    f"current plane is {self.plane}."
                )
            plane = "Pupil"
            units = "Cartesian"
        else:
            if self.plane != "Pupil":
                raise ValueError(
                    "Can only do an FFT from a Pupil plane, "
                    f"current plane is {self.plane}."
                )
            plane = "Focal"

        return plane, units, pixel_scale

    # TODO: focal_length not used?
    def _FFT(
        self: Wavefront,
        phasor: Array,
        focal_length: Array = None,
        inverse: bool = False,
    ) -> tuple:
        """
        Calculates the output plane, unit, pixel scale, and returns the
        appropriate propagation function

        Parameters
        ----------
        inverse : bool = False
            If True, the inverse FFT is used.

        Returns
        -------
        phasor : Array
            The propagated phasor.
        """
        if inverse:
            return np.fft.fft2(np.fft.ifftshift(phasor)) / phasor.shape[-1]
        else:
            return np.fft.fftshift(np.fft.ifft2(phasor)) * phasor.shape[-1]

    def FFT(
        self: Wavefront, pad: int = 2, focal_length: Array = None
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
        # Calculate
        plane, units, pixel_scale = self._FFT_output(pad, focal_length)

        # Pad must be int
        npixels = (self.npixels * (pad - 1)) // 2
        amplitude = np.pad(self.amplitude, npixels)
        phase = np.pad(self.phase, npixels)
        phasor = amplitude * np.exp(1j * phase)
        phasor = self._FFT(phasor, focal_length)

        # Return new wavefront
        return self.set(
            ["amplitude", "phase", "pixel_scale", "plane", "units"],
            [np.abs(phasor), np.angle(phasor), pixel_scale, plane, units],
        )

    def IFFT(
        self: Wavefront, pad: int = 2, focal_length: Array = None
    ) -> Wavefront:
        """
        Propagates the wavefront by performing an Inverse Fast Fourier
        Transform.

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
        # Calculate
        plane, units, pixel_scale = self._FFT_output(
            pad, focal_length, inverse=True
        )

        # Pad must be int
        npixels = (self.npixels * (pad - 1)) // 2
        amplitude = np.pad(self.amplitude, npixels)
        phase = np.pad(self.phase, npixels)
        phasor = amplitude * np.exp(1j * phase)
        phasor = self._FFT(phasor, focal_length, inverse=True)

        # Return new wavefront
        return self.set(
            ["amplitude", "phase", "pixel_scale", "plane", "units"],
            [np.abs(phasor), np.angle(phasor), pixel_scale, plane, units],
        )

    def _MFT_output(
        self: Wavefront, focal_length: Array = None, inverse: bool = False
    ) -> tuple:
        """
        Calculates the output plane and unit for the MFT

        Parameters
        ----------
        focal_length : Array = None
            The focal length of the propagation. If None, the propagation is
            treated as an 'angular' propagation, else it is treated as a
            'Cartesian' propagation.
        inverse : bool = False
            If True, the inverse MFT is used.

        Returns
        -------
        plane : str
            The output plane of the propagation.
        units : str
            The units of the output plane.
        """
        # Get units
        if focal_length is None:
            units = "Angular"
        else:
            units = "Cartesian"

            # Check for invalid propagation
            if self.units == "Angular":
                raise ValueError(
                    "focal_length can not be specific when"
                    "propagating from a Focal plane with angular units."
                )

        # Check planes
        if inverse:
            if self.plane != "Focal":
                raise ValueError(
                    "Can only do an IMFT from a Focal plane, "
                    f"current plane is {self.plane}."
                )
            plane = "Pupil"
            units = "Cartesian"
        else:
            if self.plane != "Pupil":
                raise ValueError(
                    "Can only do an MFT from a Pupil plane, "
                    f"current plane is {self.plane}."
                )
            plane = "Focal"

        return plane, units

    def _nfringes(
        self: Wavefront,
        npixels: int,
        pixel_scale: Array,
        focal_length: Array = None,
    ) -> Array:
        """
        Calculates the number of fringes in the output plane.

        Parameters
        ----------
        npixels : int
            The number of pixels in the output plane.
        pixel_scale : Array
            The pixel scale of the output plane.
        focal_length : Array = None
            The focal length of the propagation. If None, the propagation is
            treated as an 'angular' propagation, else it is treated as a
            'Cartesian' propagation.

        Returns
        -------
        nfringes : Array
            The number of fringes in the output plane.
        """
        output_size = npixels * pixel_scale

        # Angular
        if focal_length is None:
            return output_size / self.fringe_size

        # Cartesian
        else:
            return output_size / (self.fringe_size * focal_length)

    def _transfer_matrix(
        self: Wavefront,
        npixels: int,
        pixel_scale: Array,
        shift: Array = 0.0,
        focal_length: Array = None,
        inverse: bool = False,
    ) -> Array:
        """
        Calculates the transfer matrix for the MFT.

        Parameters
        ----------
        npixels : int
            The number of pixels in the output plane.
        pixel_scale : Array
            The pixel scale of the output plane.
        shift : Array = 0.
            The shift to apply to the output plane.
        focal_length : Array = None
            The focal length of the propagation. If None, the propagation is
            treated as an 'angular' propagation, else it is treated as a
            'Cartesian' propagation.
        inverse : bool = False
            Is this a forward or inverse MFT.

        Returns
        -------
        transfer_matrix : Array
            The transfer matrix for the MFT.
        """
        scale_in = 1.0 / self.npixels
        scale_out = (
            self._nfringes(npixels, pixel_scale, focal_length) / npixels
        )
        in_vec = dlu.pixel_coordinates(
            self.npixels, scale_in, shift * scale_in
        )
        out_vec = dlu.pixel_coordinates(npixels, scale_out, shift * scale_out)

        if not inverse:
            return np.exp(2j * np.pi * np.outer(in_vec, out_vec))
        else:
            return np.exp(-2j * np.pi * np.outer(in_vec, out_vec))

    def _MFT(
        self: Wavefront,
        npixels: int,
        pixel_scale: Array,
        focal_length: Array = None,
        shift: Array = np.zeros(2),
        inverse: bool = False,
    ) -> Array:
        """
        Performs the actual phasor propagation and normalises the output

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
        inverse : bool = False
            Is this a forward or inverse MFT.

        Returns
        -------
        phasor : Array
            The propagated phasor.
        """
        # Transfer Matrices
        x_matrix = self._transfer_matrix(
            npixels, pixel_scale, shift[0], focal_length, inverse=inverse
        )
        y_matrix = self._transfer_matrix(
            npixels, pixel_scale, shift[1], focal_length, inverse=inverse
        ).T

        # Propagation
        phasor = (y_matrix @ self.phasor) @ x_matrix
        nfringes = self._nfringes(npixels, pixel_scale, focal_length)
        phasor *= np.exp(
            np.log(nfringes) - (np.log(self.npixels) + np.log(npixels))
        )
        return phasor

    def MFT(
        self: Wavefront,
        npixels: int,
        pixel_scale: Array,
        focal_length: Array = None,
    ) -> Wavefront:
        """
        Propagates the wavefront by performing a 2-sided Matrix Fourier
        Transform. TODO: Add link to Soumer et al. 2007(?).

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

        Returns
        -------
        wavefront : Wavefront
            The propagated wavefront.
        """
        # Calculate
        plane, units = self._MFT_output(focal_length)
        phasor = self._MFT(npixels, pixel_scale, focal_length)

        # Return new wavefront
        pixel_scale = np.array(pixel_scale)  # Allow float input
        return self.set(
            ["amplitude", "phase", "pixel_scale", "plane", "units"],
            [np.abs(phasor), np.angle(phasor), pixel_scale, plane, units],
        )

    def IMFT(
        self: Wavefront,
        npixels: int,
        pixel_scale: Array,
        focal_length: Array = None,
    ) -> Wavefront:
        """
        Propagates the wavefront by performing a, inverse 2-sided Matrix
        Fourier Transform. TODO: Add link to Soumer et al. 2007(?).

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

        Returns
        -------
        wavefront : Wavefront
            The propagated wavefront.
        """
        # Calculate
        plane, units = self._MFT_output(focal_length, inverse=True)
        phasor = self._MFT(npixels, pixel_scale, focal_length, inverse=True)

        # Return new wavefront
        pixel_scale = np.array(pixel_scale)  # Allow float input
        return self.set(
            ["amplitude", "phase", "pixel_scale", "plane", "units"],
            [np.abs(phasor), np.angle(phasor), pixel_scale, plane, units],
        )

    def shifted_MFT(
        self: Wavefront,
        npixels: int,
        pixel_scale: Array,
        shift: Array,
        focal_length: Array = None,
        pixel: bool = True,
    ) -> Wavefront:
        """
        Propagates the wavefront by performing a 2-sided Matrix Fourier
        Transform with a shift in the center of the output plane.
        TODO: Add link to Soumer et al. 2007(?),

        Parameters
        ----------
        npixels : int
            The number of pixels in the output wavefront.
        pixel_scale : Array
            The pixel scale of the output wavefront.
        shift : Array
            The shift in the center of the output plane.
        focal_length : Array = None
            The focal length of the propagation. If None, the propagation is
            treated as an 'angular' propagation, else it is treated as a
            'Cartesian' propagation.
        pixel : bool = True
            Whether the shift is in pixels or the units of pixel_scale.

        Returns
        -------
        wavefront : Wavefront
            The propagated wavefront.
        """
        # Calculate
        plane, units = self._MFT_output(focal_length)
        shift = shift if pixel else shift / pixel_scale
        phasor = self._MFT(npixels, pixel_scale, focal_length, shift)

        # Return new wavefront
        return self.set(
            ["amplitude", "phase", "pixel_scale", "plane", "units"],
            [np.abs(phasor), np.angle(phasor), pixel_scale, plane, units],
        )

    def shifted_IMFT(
        self: Wavefront,
        npixels: int,
        pixel_scale: Array,
        shift: Array,
        focal_length: Array = None,
        pixel: bool = True,
    ) -> Wavefront:
        """
        Propagates the wavefront by performing a, Inverse 2-sided Matrix
        Fourier Transform with a shift in the center of the output plane.
        TODO: Add link to Soumer et al. 2007(?),

        Parameters
        ----------
        npixels : int
            The number of pixels in the output wavefront.
        pixel_scale : Array
            The pixel scale of the output wavefront.
        shift : Array
            The shift in the center of the output plane.
        focal_length : Array = None
            The focal length of the propagation. If None, the propagation is
            treated as an 'angular' propagation, else it is treated as a
            'Cartesian' propagation.
        pixel : bool = True
            Whether the shift is in pixels or the units of pixel_scale.

        Returns
        -------
        wavefront : Wavefront
            The propagated wavefront.
        """
        # Calculate
        plane, units = self._MFT_output(focal_length, inverse=True)
        shift = shift if pixel else shift / pixel_scale
        phasor = self._MFT(
            npixels, pixel_scale, focal_length, shift, inverse=True
        )

        # Return new wavefront
        return self.set(
            ["amplitude", "phase", "pixel_scale", "plane", "units"],
            [np.abs(phasor), np.angle(phasor), pixel_scale, plane, units],
        )


class FresnelWavefront(Wavefront):
    """
    A class to represent a wavefront that can be propagated to a Far Field
    Fresnel plane.

    Parameters
    ----------
    wavelength : float, metres
        The wavelength of the `Wavefront`.
    pixel_scale : float, metres/pixel
        The physical dimensions of each square pixel.
    amplitude : Array, power
        The electric field amplitude of the `Wavefront`.
    phase : Array, radians
        The electric field phase of the `Wavefront`.
    plane : str
        The current plane of wavefront, can be Pupil, Focal, or Intermediate.
    units : str
        The units of the wavefront, can be 'Cartesian' or 'Angular'.
    """

    def __init__(
        self: Wavefront, npixels: int, diameter: Array, wavelength: Array
    ) -> Wavefront:
        """
        Constructor for the wavefront.

        Parameters
        ----------
        npixels : int
            The number of pixels that represent the `Wavefront`.
        pixel_scale : float, metres/pixel
            The physical dimensions of each square pixel.
        wavelength : float, metres
            The wavelength of the `Wavefront`.
        """
        super().__init__(
            npixels=npixels,
            wavelength=wavelength,
            diameter=diameter,
        )

    def _nfringes(
        self: Wavefront,
        npixels: int,
        pixel_scale: Array,
        focal_shift: Array,
        focal_length: Array,
    ) -> Array:
        """
        Calculates the number of fringes in the output plane.

        Parameters
        ----------
        npixels : int
            The number of pixels that represent the `Wavefront`.
        pixel_scale : Array, metres/pixel
            The physical dimensions of each square pixel.
        focal_shift : Array, metres
            The distance the focal plane is shifted from the focal length.
        focal_length : Array, metres
            The focal length of the lens.

        Returns
        -------
        nfringes : Array
            The number of fringes in the output plane.
        """
        propagation_distance = focal_length + focal_shift
        output_size = npixels * pixel_scale

        # # Angular - Not Implemented
        # if focal_length is None:
        #     return output_size / self.fringe_size

        # Cartesian
        return output_size / (self.fringe_size * propagation_distance)

    # Move to utils as thinlens?
    def quadratic_phase(
        self: Wavefront,
        x_coordinates: Array,
        y_coordinates: Array,
        distance: Array,
    ) -> Array:
        """
        A convenience function for calculating quadratic phase factors.

        Parameters
        ----------
        x_coordinates : Array, metres
            The x coordinates of the pixels. This will be different
            in the plane of propagation and the initial plane.
        y_coordinates : Array, metres
            The y coordinates of the pixels. This will be different
            in the plane of propagation and the initial plane.
        distance : Array, metres
            The distance that is to be propagated.

        Returns
        -------
        quadratic_phase : Array
            A set of phase factors that are useful in optical calculations.
        """
        r_coordinates = np.hypot(x_coordinates, y_coordinates)
        return np.exp(0.5j * self.wavenumber * r_coordinates**2 / distance)

    def transfer_function(self: Wavefront, distance: Array) -> Array:
        """
        The Optical Transfer Function defining the phase evolution of the
        wavefront when propagating to a non-conjugate plane.

        Parameters
        ----------
        distance : Array, metres
            The distance that is being propagated in metres.

        Returns
        -------
        field : Array
            The field that represents the optical transfer.
        """
        return np.exp(1.0j * self.wavenumber * distance)

    def _transfer_matrix(
        self: Wavefront,
        npixels: int,
        pixel_scale: Array,
        focal_shift: Array,
        focal_length: Array,
        shift: Array = 0.0,
    ) -> Array:
        """
        Calculates the transfer matrix for the MFT.

        Parameters
        ----------
        npixels : int
            The number of pixels that represent the `Wavefront`.
        pixel_scale : Array, metres/pixel
            The physical dimensions of each square pixel.
        focal_shift : Array, metres
            The distance the focal plane is shifted from the focal length.
        focal_length : Array, metres
            The focal length of the lens.
        shift : Array = 0., metres
            The shift to apply to the output plane.

        Returns
        -------
        transfer_matrix : Array
            The transfer matrix for the MFT.
        """
        scale_in = 1.0 / self.npixels
        scale_out = (
            self._nfringes(npixels, pixel_scale, focal_shift, focal_length)
            / npixels
        )
        in_vec = dlu.pixel_coordinates(
            self.npixels, scale_in, shift * scale_in
        )
        out_vec = dlu.pixel_coordinates(npixels, scale_out, shift * scale_out)

        if self.plane == "Pupil":
            return np.exp(2j * np.pi * np.outer(in_vec, out_vec))
        else:
            raise ValueError(f"plane must be 'Pupil' Got {self.plane}")

        # elif self.plane == 'Focal':
        #     return np.exp(-2j * np.pi * np.outer(in_vec, out_vec))

    def _MFT(
        self: Wavefront,
        phasor: Array,
        npixels: int,
        pixel_scale: Array,
        focal_length: Array,
        focal_shift: Array,
        shift: Array = np.zeros(2),
    ) -> Array:
        """
        Performs the MFT.

        Parameters
        ----------
        phasor : Array
            The phasor to be propagated.
        npixels : int
            The number of pixels that represent the `Wavefront`.
        pixel_scale : Array, metres/pixel
            The physical dimensions of each square pixel.
        focal_length : Array, metres
            The focal length of the lens.
        focal_shift : Array, metres
            The distance the focal plane is shifted from the focal length.
        shift : Array = np.zeros(2), metres
            The shift to apply to the output plane.

        Returns
        -------
        phasor : Array
            The propagated phasor.
        """
        # Set up
        nfringes = self._nfringes(
            npixels, pixel_scale, focal_shift, focal_length
        )
        x_matrix = self._transfer_matrix(
            npixels, pixel_scale, focal_shift, focal_length, shift[0]
        )
        y_matrix = self._transfer_matrix(
            npixels, pixel_scale, focal_shift, focal_length, shift[1]
        ).T

        # Perform and normalise
        phasor = (y_matrix @ phasor) @ x_matrix
        phasor *= np.exp(
            np.log(nfringes) - (np.log(self.npixels) + np.log(npixels))
        )
        return phasor

    def _phase_factors(
        self: Wavefront,
        npixels: int,
        pixel_scale: Array,
        focal_length: Array,
        focal_shift: Array,
    ) -> tuple:
        """
        Calculates the phase factors for the Fresnel propagation.

        Parameters
        ----------
        npixels : int
            The number of pixels in the output plane.
        pixel_scale : Array, metres/pixel
            The physical dimensions of each square pixel.
        focal_length : Array, metres
            The focal length of the lens.
        focal_shift : Array, metres
            The distance the focal plane is shifted from the focal length.

        Returns
        -------
        first_factor : Array
            The first factor in the Fresnel propagation.
        second_factor : Array
            The second factor in the Fresnel propagation.
        third_factor : Array
            The third factor in the Fresnel propagation.
        fourth_factor : Array
            The fourth factor in the Fresnel propagation.
        """
        # Coordinates
        prop_dist = focal_length + focal_shift
        input_positions = self.coordinates
        output_positions = dlu.pixel_coords(npixels, pixel_scale)

        # Calculate phase values
        first_factor = self.quadratic_phase(*input_positions, -focal_length)
        second_factor = self.quadratic_phase(*input_positions, prop_dist)
        third_factor = self.transfer_function(prop_dist)
        fourth_factor = self.quadratic_phase(*output_positions, prop_dist)
        return first_factor, second_factor, third_factor, fourth_factor

    def fresnel_prop(
        self: Wavefront,
        npixels: int,
        pixel_scale: Array,
        focal_length: Array,
        focal_shift: Array,
    ) -> Array:
        """
        Propagates the wavefront from the input plane to the output plane using
        a Fresnel Transform using a Matrix Fourier Transform.

        Parameters
        ----------
        npixels : int
            The number of pixels in the output plane.
        pixel_scale : Array, metres/pixel
            The physical dimensions of each square pixel.
        focal_length : Array, metres
            The focal length of the lens.
        focal_shift : Array, metres
            The distance the focal plane is shifted from the focal length.

        Returns
        -------
        wavefront : Wavefront
            The propagated wavefront.
        """
        # Calculate phase values
        first, second, third, fourth = self._phase_factors(
            npixels, pixel_scale, focal_length, focal_shift
        )

        # Apply phases
        phasor = self.phasor
        phasor *= first
        phasor *= second

        # Propagate
        phasor = self._MFT(
            phasor, npixels, pixel_scale, focal_length, focal_shift
        )

        # Apply phases
        phasor *= third
        phasor *= fourth

        # Update
        pixel_scale = np.array(pixel_scale)
        return self.set(
            ["amplitude", "phase", "pixel_scale", "plane", "units"],
            [
                np.abs(phasor),
                np.angle(phasor),
                pixel_scale,
                "Intermediate",
                "Cartesian",
            ],
        )

    def shifted_fresnel_prop(
        self: Wavefront,
        npixels: int,
        pixel_scale: Array,
        shift: Array,
        focal_length: Array,
        focal_shift: Array,
        pixel: bool = True,
    ) -> Array:
        """
        Propagates the wavefront from the input plane to the output plane using
        a Fresnel Transform using a Matrix Fourier Transform with a shift in
        the center of the output plane.
        TODO: Add link to Soumer et al. 2007(?),
        """
        # Get shift
        shift = shift if pixel else shift / pixel_scale

        # Calculate phase values
        first, second, third, fourth = self._phase_factors(
            npixels, pixel_scale, focal_length, focal_shift
        )

        # Apply phases
        phasor = self.phasor
        phasor *= first
        phasor *= second

        # Propagate
        phasor = self._MFT(
            phasor, npixels, pixel_scale, focal_length, focal_shift, shift
        )

        # Apply phases
        phasor *= third
        phasor *= fourth

        # Update
        pixel_scale = np.array(pixel_scale)
        return self.set(
            ["amplitude", "phase", "pixel_scale", "plane", "units"],
            [
                np.abs(phasor),
                np.angle(phasor),
                pixel_scale,
                "Intermediate",
                "Cartesian",
            ],
        )
