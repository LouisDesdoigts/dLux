from __future__ import annotations
import jax.numpy as np
from jax import vmap, Array
import zodiax as zdx
from typing import Any
import dLux.utils as dlu
import dLux

# Optical layers require Wavefront to init, so we alias it here to avoid MRO issues
OpticalLayer = lambda: dLux.layers.optical_layers.OpticalLayer


"""
High level notes:

 - Should we allow Nones to the magic methods?
 - Should we allow .apply, rahter than enforcing call?
    - The question here is whether we need other inputs to the call (ie normalise 
    or other meta-parameters)
"""

__all__ = ["Wavefront"]


class Wavefront(zdx.Base):
    """
    A simple class to hold the state of some wavefront as it is transformed and
    propagated throughout an optical system. All wavefronts assume square arrays.

    Attributes
    ----------
    wavelength : float, meters
        The wavelength of the `Wavefront`.
    phasor : Array[complex]
        The electric field of the `Wavefront`.
    pixel_scale : float, meters/pixel or radians/pixel
        The pixel scale of the phase and amplitude arrays. If `units='Cartesian'` then
        the pixel scale is in meters/pixel, else if `units='Angular'` then the pixel
        scale is in radians/pixel.
    plane : str
        The current plane type of wavefront, can be 'Pupil', 'Focal' or 'Intermediate'.
    units : str
        The current units of the wavefront, can be 'Cartesian' or 'Angular'.

    Properties
    ----------
    TODO
    """

    wavelength: float
    pixel_scale: float
    phasor: Array[complex]
    plane: str
    units: str

    # TODO: Allow a phasor input, or add a `from_phasor` class method like prysm
    def __init__(
        self: Wavefront,
        npixels: int,
        diameter: float,
        wavelength: float,
        plane: str = "Pupil",
        units: str = "Cartesian",
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
        amplitude = np.ones((npixels, npixels), dtype=float) / npixels**2
        phase = np.zeros((npixels, npixels), dtype=float)
        self.phasor = amplitude * np.exp(1j * phase)

        # Always initialised in Pupil plane with Cartesian Coords
        self.plane = str(plane)
        self.units = str(units)

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
        return self.phasor.shape[-1]

    @property
    def real(self: Wavefront) -> Array:
        """
        Returns the real component of the `Wavefront`.

        Returns
        -------
        wavefront : Array
            The real component of the `Wavefront` phasor.
        """
        return self.phasor.real

    @property
    def imaginary(self: Wavefront) -> Array:
        """
        Returns the imaginary component of the `Wavefront`.

        Returns
        -------
        wavefront : Array
            The imaginary component of the `Wavefront` phasor.
        """
        return self.phasor.imag

    @property
    def amplitude(self: Wavefront) -> Array:
        """
        Returns the amplitude component of the `Wavefront`.

        Returns
        -------
        wavefront : Array
            The amplitude component of the `Wavefront` phasor.
        """
        return np.abs(self.phasor)

    @property
    def phase(self: Wavefront) -> Array:
        """
        Returns the phase component of the `Wavefront`.

        Returns
        -------
        wavefront : Array
            The phase component of the `Wavefront` phasor.
        """
        return np.angle(self.phasor)

    @property
    def complex(self: Wavefront) -> Array:
        """
        Returns the complex phasor of the `Wavefront`.

        Returns
        -------
        wavefront : Array
            The complex phasor of the `Wavefront`.
        """
        return np.stack([self.phasor.real, self.phasor.imag], axis=0)

    @property
    def polar(self: Wavefront) -> Array:
        """
        Returns the polar representation (amplitude, phase) of the `Wavefront`.

        Returns
        -------
        wavefront : Array
            The polar representation of the `Wavefront` as a stack of amplitude and
            phase.
        """
        return np.stack([self.amplitude, self.phase], axis=0)

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
        return np.abs(self.phasor) ** 2

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

        Returns
        -------
        fringe_size : Array, radians
            The size of the linear diffraction fringe of the wavefront.
        """
        if self.plane != "Pupil":
            raise ValueError(
                "Fringe size is only defined for wavefronts in the Pupil plane."
            )
        return self.wavelength / self.diameter

    @property
    def ndim(self: Wavefront) -> int:
        """
        Returns the number of 'dimensions' of the wavefront. This is used to track the
        vectorised version of the wavefront returned from vmapping.

        NOTE: May clash with future polarised wavefront

        Returns
        -------
        ndim : int
            The 'dimensionality' of dimensions of the wavefront.
        """
        return self.pixel_scale.ndim

    @property
    def power(self: Wavefront) -> Array:
        """
        Returns the total power of the wavefront (sum of |E|^2 over pixels).

        Returns
        -------
        power : Array
            The total power of the wavefront.
        """
        return np.sum(np.abs(self.phasor) ** 2)

    def add_phase(self: Wavefront, phase: Array) -> Wavefront:
        """
        Applies a phase (in radians) to the wavefront by multiplying the phasor
        by exp(1j * phase). Supports broadcasting.

        Parameters
        ----------
        phase : Array, radians
            The phase to be added to the wavefront.

        Returns
        -------
        wavefront : Wavefront
            New wavefront whose phasor is self.phasor * exp(1j * phase).
        """
        if phase is None:
            return self
        return self.multiply("phasor", np.exp(1j * phase))

    def add_opd(self: Wavefront, opd: Array) -> Wavefront:
        """
        Applies an optical path difference (in meters) by multiplying the phasor
        by exp(1j * k * opd), where k = 2*pi / wavelength. Supports broadcasting.

        Parameters
        ----------
        opd : Array, meters
            The optical path difference to apply.

        Returns
        -------
        wavefront : Wavefront
            New wavefront with phasor multiplied by exp(1j * k * opd).
        """
        if opd is None:
            return self
        return self.add_phase(self.wavenumber * np.asarray(opd))

    def coordinates(
        self: Wavefront,
        scale=1.0,
        polar: bool = False,
        fft_style: bool = False,
    ) -> Array:
        """
        Returns the physical positions of the wavefront pixels in meters, with an
        optional scaling factor for numerical stability.

        Parameters
        ----------
        scale : float = 1.0
            Optional scaling factor applied to the diameter for numerical stability.
        polar : bool = False
            Output the coordinates in polar (r, phi) coordinates.
        fft_style : bool = False
            If True, use FFT-style centering. For even npixels this produces integer
            centered coordinates. For odd npixels this is identical to the default.

        Returns
        -------
        coordinates : Array
            The coordinates of the centers of each pixel representing the wavefront.
        """
        return dlu.pixel_coords(self.npixels, self.diameter * scale, polar, fft_style)

    def tilt(self: "Wavefront", angles: Array, unit: str = "rad") -> "Wavefront":
        """
        Tilts the wavefront by the (x, y) angles.

        Parameters
        ----------
        angles : Array
            The (x, y) angles by which to tilt the wavefront, in `unit`.
        unit : str
            The units of the angles, e.g. "rad", "deg", "arcmin", "arcsec", and
            prefixed forms like "mrad", "mas", etc (as supported by utils/units.py).

        Returns
        -------
        wavefront : Wavefront
            The tilted wavefront.
        """
        if getattr(angles, "shape", None) != (2,):
            raise ValueError("angles must be an array of shape (2,).")

        # factor such that angle_rad = angle_unit * factor
        scaling = dlu.unit_factor_to_rad(unit)

        # Calculate scales coordinates
        coords = self.coordinates(scale=scaling)

        # Tilt the wavefront
        return self.add_opd(-(angles[:, None, None] * coords).sum(0))

    def normalise(
        self: Wavefront,
        mode: str = "power",
        value: float = 1.0,
    ) -> Wavefront:
        """
        Normalise the wavefront.

        Parameters
        ----------
        mode : {"power","peak"} = "power"
            - "power": scales so sum(|E|^2) == value (discrete sum over pixels).
            - "peak" : scales so max(|E|^2) == value.
        value : float = 1.0
            Target value for the selected mode.

        Returns
        -------
        wavefront : Wavefront
            New wavefront with phasor scaled to achieve the normalisation.
        """
        if mode == "power":
            scale = np.sqrt(value / self.power.sum())
        elif mode == "peak":
            scale = np.sqrt(value / self.power.max())
        else:
            raise ValueError("mode must be 'power' or 'peak'")
        return self.multiply("phasor", scale)

    def flip(self: Wavefront, axis: tuple[int] | int) -> Wavefront:
        """
        Flip the complex phasor along one or more axes (ij indexing: 0=y, 1=x).

        Parameters
        ----------
        axis : int or tuple of ints
            Axes to flip.

        Returns
        -------
        wavefront : Wavefront
            New wavefront with phasor flipped.
        """
        return self.set(phasor=np.flip(self.phasor, axis))

    def scale_to(
        self: Wavefront,
        npixels: int,
        pixel_scale: Array,
        complex: bool = True,
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
        complex : bool = True
            If True, interpolate the real and imaginary components. If False,
            interpolate the amplitude and phase components.

        Returns
        -------
        wavefront : Wavefront
            The new interpolated wavefront.
        """
        # Get field in either (amplitude, phase) or (real, imaginary)
        fields = self.complex if complex else self.polar

        # Scale the field
        scale_fn = vmap(dlu.scale, (0, None, None))
        fields = scale_fn(fields, npixels, pixel_scale / self.pixel_scale)

        # Convert back to complex form
        if complex:
            phasor = fields[0] + 1j * fields[1]
        else:
            phasor = fields[0] * np.exp(1j * fields[1])

        # Return new wavefront
        return self.set(phasor=phasor, pixel_scale=pixel_scale)

    def rotate(
        self: Wavefront,
        angle: Array,
        method: str = "linear",
        complex: bool = True,
    ) -> Wavefront:
        """
        Rotates the wavefront by a given angle via interpolation. Can be done on the
        real and imaginary components by passing in complex=True.

        Parameters
        ----------
        angle : Array, radians
            The angle by which to rotate the wavefront in a clockwise
            direction.
        method : str = "linear"
            The interpolation method.
        complex : bool = False
            If True, rotate the real and imaginary components. If False, rotate the
            amplitude and phase components.

        Returns
        -------
        wavefront : Wavefront
            The new wavefront rotated by angle in the clockwise direction.
        """
        # Get field in either (amplitude, phase) or (real, imaginary)
        fields = self.complex if complex else self.polar

        # Rotate the field
        rotator = vmap(dlu.rotate, (0, None, None))
        fields = rotator(fields, angle, method)

        # Convert back to complex form
        if complex:
            phasor = fields[0] + 1j * fields[1]
        else:
            phasor = fields[0] * np.exp(1j * fields[1])

        # Return new wavefront
        return self.set(phasor=phasor)

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
        phasor = vmap(dlu.resize, (0, None, None))(self.phasor, npixels, 0j)
        return self.set(phasor=phasor)

    def _prep_prop(self: Wavefront, focal_length) -> tuple:
        """
        Determine propagation direction and output metadata.

        Parameters
        ----------
        focal_length : float | None
            Focal length for a pupil→focal propagation. If None, focal plane sampling
            is angular (radians/pixel). If provided, focal plane sampling is Cartesian
            (meters/pixel). For focal→pupil inverse propagation, must be None when
            current units are Angular.

        Returns
        -------
        inverse : bool
            False for forward pupil→focal; True for focal→pupil inverse.
        plane : str
            'Focal' if starting in a Pupil plane, else 'Pupil'.
        units : str
            'Angular' if forward propagation with focal_length=None, else 'Cartesian'.
        """
        if self.plane == "Pupil":
            inverse = False
            plane = "Focal"
            units = "Angular" if focal_length is None else "Cartesian"
        else:
            if focal_length is not None and self.units == "Angular":
                raise ValueError(
                    "Cannot specify focal_length when propagating from an angular "
                    "Focal plane."
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
        Fraunhofer (FFT) propagation between conjugate pupil and focal planes.

        Parameters
        ----------
        focal_length : float | None
            If None, output sampling is angular (radians/pixel).
            If float, output sampling is Cartesian at that focal length.
        pad : int
            Zero-padding factor applied before the FFT to control sampling / aliasing.

        Returns
        -------
        wavefront : Wavefront
            New wavefront with propagated phasor, updated pixel_scale, plane and units.

        Notes
        -----
        - Phasor is transformed directly; amplitude/phase are derived.
        - Energy conservation depends on padding conventions in dlu.FFT.
        """
        inverse, plane, units = self._prep_prop(focal_length)
        phasor, pixel_scale = dlu.FFT(
            self.phasor,
            self.wavelength,
            self.pixel_scale,
            focal_length,
            pad,
            inverse,
        )
        return self.set(
            ["phasor", "pixel_scale", "plane", "units"],
            [phasor, pixel_scale, plane, units],
        )

    def _MFT(
        self: Wavefront,
        phasor: Array,
        wavelength: float,
        pixel_scale: float,
        *args: tuple,
    ) -> Array:
        """
        Internal alias wrapper for dlu.MFT to support vmapped / broadband propagation.

        Parameters
        ----------
        phasor : Array[complex]
            Input complex field.
        wavelength : float
            Wavelength associated with the field.
        pixel_scale : float
            Input sampling (meters/pixel or radians/pixel).
        args : tuple
            Additional arguments passed through to dlu.MFT (npixels_out,
            pixel_scale_out, focal_length, shift, pixel_units, inverse_flag).

        Returns
        -------
        phasor : Array[complex]
            Propagated complex field.
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
        Flexible MFT propagation allowing explicit output sampling.

        Parameters
        ----------
        npixels : int
            Output array size (square).
        pixel_scale : float
            Desired output pixel scale (meters/pixel or radians/pixel depending on
            units).
        focal_length : float | None
            Focal length for Cartesian focal sampling; None for angular focal sampling.
        shift : Array, shape (2,)
            Offset of the output plane center (x, y). Units = pixels if pixel=True,
            else physical units matching input pixel_scale.
        pixel : bool
            Interpret shift in pixel units if True; else in pixel_scale units.

        Returns
        -------
        wavefront : Wavefront
            Propagated wavefront with new phasor and sampling metadata.

        Notes
        -----
        - Ideal for generating PSFs at arbitrary sampling.
        - For broadband propagation, vmap this function over wavelength and pixel_scale.
        """
        inverse, plane, units = self._prep_prop(focal_length)
        pixel_scale = np.asarray(pixel_scale, float)
        args = (npixels, pixel_scale, focal_length, shift, pixel, inverse)
        phasor = self._MFT(self.phasor, self.wavelength, self.pixel_scale, *args)
        return self.set(
            ["phasor", "pixel_scale", "plane", "units"],
            [phasor, pixel_scale, plane, units],
        )

    def propagate_fresnel(
        self: Wavefront,
        npixels: int,
        pixel_scale: float,
        focal_length: float,
        focal_shift: float = 0.0,
        shift: Array = np.zeros(2),
        pixel: bool = True,
        inverse: bool = False,
    ) -> Wavefront:
        """
        Far-field Fresnel propagation near focus for intermediate planes.

        Parameters
        ----------
        npixels : int
            Output array size.
        pixel_scale : float
            Output sampling (meters/pixel).
        focal_length : float
            System focal length.
        focal_shift : float
            Axial distance from best focus (meters).
        shift : Array, shape (2,)
            Lateral shift of output plane center.
        pixel : bool
            Interpret shift as pixels if True; else physical units.
        inverse : bool
            If True, perform inverse Fresnel (rare; leave False for forward).

        Returns
        -------
        wavefront : Wavefront
            New wavefront in an 'Intermediate' Cartesian plane.

        Raises
        ------
        ValueError
            If current plane is not 'Pupil'.

        Notes
        -----
        - Models defocus regions a few wavelengths from best focus.
        - Assumes forward propagation from pupil; inverse mode retained for
        experimentation.
        """
        if self.plane != "Pupil":
            raise ValueError(
                "Fresnel propagation requires starting in Pupil plane (got "
                f"{self.plane})."
            )
        pixel_scale = np.asarray(pixel_scale, float)
        plane = "Intermediate"
        units = "Cartesian"
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
        return self.set(
            ["phasor", "pixel_scale", "plane", "units"],
            [phasor, pixel_scale, plane, units],
        )

    def _magic_unified_op(self, other: Any, op: str) -> Wavefront:
        """
        Internal helper function to unify the logic of the magic methods for addition,
        subtraction, multiplication and division.

        Parameters
        ----------
        other : Any
            The object to operate with. Can be a complex array, a Wavefront, or None.
        op : str
            The operation to perform: 'add', 'subtract', 'multiply', or 'divide'.

        Returns
        -------
        wavefront : Wavefront
            The resulting wavefront after applying the operation.
        """
        # Nones always return unchanged
        if other is None:
            return self

        # Check for supported types
        if not isinstance(other, (Wavefront, Array, float, int, complex)):
            raise TypeError(
                f"Unsupported type for {op}: {type(other)}. Must be an array, "
                "Wavefront, or None."
            )

        # Extract phasor if other is a Wavefront
        if isinstance(other, Wavefront):
            other = other.phasor

        # Apply the operation
        if op == "add":
            return self.add("phasor", other)
        elif op == "subtract":
            return self.add("phasor", -other)
        elif op == "multiply":
            return self.multiply("phasor", other)
        elif op == "divide":
            return self.multiply("phasor", 1 / other)
        else:
            raise ValueError(f"Unsupported operation '{op}'.")

    def __add__(self: Wavefront, other: Any) -> Wavefront:
        """
        Allows complex phasors or Wavefronts to be added together. Nones are ignored.
        """
        return self._magic_unified_op(other, "add")

    def __sub__(self: Wavefront, other: Any) -> Wavefront:
        """
        Allows complex phasors or Wavefronts to be subtracted. Nones are ignored.
        """
        return self._magic_unified_op(other, "subtract")

    def __mul__(self: Wavefront, other: Any) -> Wavefront:
        """
        Allows complex phasors or Wavefronts to be multiplied. Nones are ignored.
        """
        return self._magic_unified_op(other, "multiply")

    def __truediv__(self: Wavefront, other: Any) -> Wavefront:
        """
        Allows complex phasors or Wavefronts to be divided. Nones are ignored.
        """
        return self._magic_unified_op(other, "divide")

    def __iadd__(self: Wavefront, other: Any) -> Wavefront:
        """In-place addition."""
        return self.__add__(other)

    def __isub__(self: Wavefront, other: Any) -> Wavefront:
        """In-place subtraction."""
        return self.__sub__(other)

    def __imul__(self: Wavefront, other: Any) -> Wavefront:
        """In-place multiplication."""
        return self.__mul__(other)

    def __itruediv__(self: Wavefront, other: Any) -> Wavefront:
        """In-place division."""
        return self.__truediv__(other)
