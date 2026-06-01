"""Wavefront state and propagation utilities used by optical systems."""

from __future__ import annotations
import jax.numpy as np
from jax import vmap, Array
import zodiax as zdx
import dLux.utils as dlu

from .psfs import PSF
from .coordinates import CoordSpec

__all__ = ["Wavefront", "PolarisedWavefront"]

# TODO: Make coord specs 2d compatible


class Wavefront(zdx.Base):
    """
    Holds the state of a wavefront as it is transformed and propagated
    through an optical system. All wavefronts assume square arrays.

    ??? abstract "UML"
        ![UML](../../assets/uml/Wavefront.png)

    Attributes
    ----------
    wavelength : float, meters
        The wavelength of the `Wavefront`.
    phasor : Array[complex]
        The electric field of the `Wavefront`.
    pixel_scale : float, meters/pixel
        The pixel scale of the phase and amplitude arrays.
    center : Array
        The centre coordinate of the wavefront grid.
    diameter : Array, property
        Derived property from `pixel_scale` and `npixels`; wavefront diameter.
    npixels : int, property
        Derived property from `phasor`; side length of wavefront arrays.
    real : Array, property
        Derived property from `phasor`; real component of the electric field.
    imaginary : Array, property
        Derived property from `phasor`; imaginary component of the electric field.
    amplitude : Array, property
        Derived property from `phasor`; field amplitude `abs(phasor)`.
    phase : Array, property
        Derived property from `phasor`; field phase angle.
    complex : tuple[Array, Array], property
        Derived property from `phasor`; `(real, imaginary)` representation.
    polar : tuple[Array, Array], property
        Derived property from `phasor`; `(amplitude, phase)` representation.
    psf : Array, property
        Derived property from `phasor`; intensity image `abs(phasor) ** 2`.
    wavenumber : Array, property
        Derived property from `wavelength`; scalar `2 * pi / wavelength`.
    ndim : int, property
        Derived property from `pixel_scale`; vectorisation rank of wavefront state.
    power : Array, property
        Derived property from `amplitude`; total wavefront power.
    """

    phasor: Array[complex]
    wavelength: float
    pixel_scale: float
    center: float

    def __init__(
        self: Wavefront,
        wavelength: float,
        npixels: int,
        diameter: float = None,
        pixel_scale: float = None,
        center: Array = None,
    ):
        """
        Parameters
        ----------
        wavelength : float, meters
            The wavelength of the `Wavefront`.
        npixels : int
            The number of pixels that represent the `Wavefront`.
        diameter : float = None, meters
            The total diameter of the `Wavefront`. Either `diameter` or `pixel_scale`
            must be provided.
        pixel_scale : float = None, meters/pixel
            The pixel scale of the `Wavefront`. Either `diameter` or `pixel_scale`
            must be provided.
        center : Array = None
            The centre coordinate of the wavefront grid, in metres. Defaults to zero.
        """
        # Handle diameter vs pixel_scale
        if diameter is None and pixel_scale is None:
            raise ValueError("Provide one: diameter or pixel_scale.")
        if diameter is not None and pixel_scale is not None:
            raise ValueError(
                "Cannot specify both 'diameter' and 'pixel_scale' - they are "
                "interdependent (diameter = pixel_scale × npixels). Choose one: "
                "use 'diameter' for wavefront diameter, or 'pixel_scale' for "
                "wavefront sampling."
            )

        self.wavelength = np.asarray(wavelength, float)
        if diameter is not None:
            self.pixel_scale = np.asarray(diameter / npixels, float)
        else:
            self.pixel_scale = np.asarray(pixel_scale, float)

        amplitude = np.ones((npixels, npixels), dtype=float) / npixels**2
        phase = np.zeros((npixels, npixels), dtype=float)
        self.phasor = amplitude * np.exp(1j * phase)

        if center is not None:
            self.center = np.asarray(center, float)

            # NOTE: only 1d offsets are presently supported
            if self.center.shape != (1,):
                raise ValueError("center must have shape (1,).")
        else:
            self.center = np.zeros(1, float)

    @classmethod
    def from_phasor(
        cls,
        phasor: Array[complex],
        wavelength: float,
        pixel_scale: float = None,
        diameter: float = None,
        center: Array = None,
    ) -> Wavefront:
        """
        Create a Wavefront from an existing phasor array.

        Parameters
        ----------
        phasor : Array[complex]
            The complex electric field array.
        wavelength : float, meters
            The wavelength of the wavefront.
        pixel_scale : float = None, meters/pixel
            The pixel scale of the phasor array. Either `pixel_scale` or
            `diameter` must be provided.
        diameter : float = None, meters
            The diameter of the phasor array. Either `pixel_scale` or
            `diameter` must be provided.
        center : Array = None
            The centre coordinate of the wavefront grid, in metres. Defaults to zero.

        Returns
        -------
        wavefront : Wavefront
            A new Wavefront object with the specified phasor.
        """
        # Infer npixels from phasor shape
        phasor_arr = np.asarray(phasor, complex)
        npixels = phasor_arr.shape[-1]

        # Create instance with appropriate parameters and set the phasor
        return cls(
            npixels=npixels,
            wavelength=wavelength,
            diameter=diameter,
            pixel_scale=pixel_scale,
            center=center,
        ).set(phasor=phasor_arr)

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

    def to_psf(self: Wavefront) -> PSF:
        """
        Converts the wavefront to a dLux PSF object.

        Returns
        -------
        psf : PSF
            A PSF object containing the current wavefront intensity and
            pixel scale.
        """
        return PSF(self.psf, self.pixel_scale)

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
    def ndim(self: Wavefront) -> int:
        """
        Returns the number of 'dimensions' of the wavefront. This is used to track the
        vectorised version of the wavefront returned from vmapping.

        NOTE: May clash with future polarised wavefront.

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

    def tilt(self: Wavefront, angles: Array, unit: str = "rad") -> Wavefront:
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
        angles = np.asarray(angles, dtype=float)
        if angles.shape != (2,):
            raise ValueError("angles must be a 1d array of shape (2,).")

        # Calculate scaled coordinates
        coords = self.coordinates(scale=dlu.unit_factor_to_rad(unit))

        # Tilt the wavefront
        return self.add_opd(np.sum(angles[:, None, None] * coords, axis=0))

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
            scale = np.sqrt(value / self.power)
        elif mode == "peak":
            scale = np.sqrt(value / self.psf.max())
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
        Interpolates the wavefront to a given npixels and pixel_scale. Can be done on
        the real and imaginary components by passing in complex=True.

        Parameters
        ----------
        npixels : int
            The number of pixels to interpolate to.
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
        complex : bool = True
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
        return self.set(phasor=dlu.resize(self.phasor, npixels, 0j))

    def coordinates(
        self: Wavefront,
        scale=1.0,
        polar: bool = False,
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

        Returns
        -------
        coordinates : Array
            The coordinates of the centers of each pixel representing the wavefront.
        """
        xs = self.xs * scale
        coords = np.array(np.meshgrid(xs, xs))
        if polar:
            return dlu.cart2polar(coords)
        return coords

    @property
    def spec(self):
        """
        Returns the current wavefront sampling as a `CoordSpec`.

        Returns
        -------
        spec : CoordSpec
            Coordinate specification with `n`, `d`, and `c` set from the
            current wavefront state.
        """
        return CoordSpec(self.npixels, self.pixel_scale, self.center)

    @property
    def xs(self):
        """
        1D array of pixel centre coordinates along one axis.

        Returns
        -------
        xs : Array
            Coordinates of pixel centres, in metres.
        """
        return self.spec.xs

    def set_spec(self, spec: CoordSpec):
        """
        Updates the wavefront pixel scale and centre from a `CoordSpec`.

        Parameters
        ----------
        spec : CoordSpec
            The coordinate specification to apply.

        Returns
        -------
        wavefront : Wavefront
            New wavefront with updated `pixel_scale` and `center`.
        """
        return self.set(pixel_scale=spec.d, center=spec.c)

    def propagate_FFT(
        self,
        pad=2,
        focal_length=None,
        spec_out: CoordSpec = None,
        inverse=False,
    ):
        """
        Propagates the wavefront using an FFT-based method.

        Parameters
        ----------
        pad : int = 2
            Zero-padding factor applied before the FFT.
        focal_length : float | None = None
            Focal length for Cartesian focal sampling. Pass `None` for
            angular (far-field) sampling.
        spec_out : CoordSpec | None = None
            Output coordinate specification. If provided, only `c` (centre)
            may be set; `n` and `d` are determined by the propagation.
        inverse : bool = False
            If False, propagate forward through the system. If True, propagate
            backward through the system.

        Returns
        -------
        wavefront : Wavefront
            Propagated wavefront with updated phasor and sampling metadata.
        """
        # Input spec
        spec_in = self.spec
        wl = self.wavelength

        # Default FFT output center
        n_out = spec_in.n * pad
        d_fft, c_fft = dlu.fft_spec(n_out, spec_in.d, wl, focal_length)

        # Get the phase ramp and the output center
        if spec_out is not None:
            if spec_out.d is not None:
                raise ValueError("Output spec cannot specify d; FFT output d is fixed.")
            if spec_out.n is not None:
                raise ValueError(
                    "Output spec cannot specify n; FFT output n is determined by the ",
                    "pad parameter.",
                )

            # Calculate the input phase ramp for the FFT propagation
            shift = c_fft - spec_out.c
            in_ramp = dlu.fft_phase_ramp(spec_in.xs, wl, shift, focal_length, inverse)

            # Calculate the output phase ramp correction
            spec_out = spec_out.set(n=n_out, d=d_fft)
            shift = dlu.fft_spec(spec_out.n, spec_out.d, wl, focal_length)[1]
            out_ramp = dlu.fft_phase_ramp(spec_out.xs, wl, shift, focal_length, inverse)

        else:
            in_ramp, out_ramp = 1.0, 1.0
            spec_out = CoordSpec(n=n_out, d=d_fft, c=c_fft)

        # Apply ramp and FFT
        phasor, pixel_scale = dlu.FFT(
            phasor=self.phasor * in_ramp,
            wavelength=self.wavelength,
            pixel_scale=self.pixel_scale,
            focal_length=focal_length,
            inverse=inverse,
            pad=pad,
        )

        # Update the values
        return self.set(
            phasor=phasor * out_ramp, pixel_scale=pixel_scale, center=spec_out.c
        )

    def propagate(
        self: Wavefront,
        npixels: int,
        pixel_scale: float,
        focal_length: float = None,
        inverse: bool = False,
    ) -> Wavefront:
        """
        Legacy MFT propagation function without CoordSpec.

        Parameters
        ----------
        npixels : int
            Output array size (square).
        pixel_scale : float
            Desired output pixel scale (meters/pixel or radians/pixel depending on
            units).
        focal_length : float | None
            Focal length for Cartesian focal sampling; None for angular focal sampling.
        inverse : bool = False
            If False, propagate forward through the system. If True, propagate
            backward through the system.

        Returns
        -------
        wavefront : Wavefront
            Propagated wavefront with new phasor and sampling metadata.

        Notes
        -----
        - Ideal for generating PSFs at arbitrary sampling.
        - For broadband propagation, vmap this function over wavelength and pixel_scale.
        """
        # Propagate
        phasor = dlu.MFT(
            phasor=self.phasor,
            wavelength=self.wavelength,
            pixel_scale_in=self.pixel_scale,
            npixels_out=npixels,
            pixel_scale_out=pixel_scale,
            focal_length=focal_length,
            inverse=inverse,
        )
        return self.set(phasor=phasor, pixel_scale=np.array(pixel_scale, float))

    def propagate_MFT(self, spec_out, focal_length=None, inverse=None):
        """
        Propagates the wavefront using an MFT-based method with a `CoordSpec`.

        Parameters
        ----------
        spec_out : CoordSpec
            Output coordinate specification defining the number of pixels
            and pixel scale of the propagated field.
        focal_length : float | None = None
            Focal length for Cartesian focal sampling. Pass `None` for
            angular (far-field) sampling.
        inverse : bool | None = None
            If False or None, propagate forward through the system. If True,
            propagate backward through the system.

        Returns
        -------
        wavefront : Wavefront
            Propagated wavefront with updated phasor and pixel scale.
        """
        # Propagate
        phasor = dlu.MFT(
            phasor=self.phasor,
            wavelength=self.wavelength,
            pixel_scale_in=self.pixel_scale,
            npixels_out=spec_out.n,
            pixel_scale_out=spec_out.d,
            focal_length=focal_length,
            inverse=inverse,
        )
        return self.set(phasor=phasor, pixel_scale=np.array(spec_out.d, float))

    #######################
    ### New Propagators ###
    #######################
    def propagate_ASM(self):
        """Angular spectrum free-space propagation"""
        raise NotImplementedError()

    def propagate_fresnel(self):
        """LCT-based MFT Fresnel propagation"""
        raise NotImplementedError()

    def propagate_fresnel_fft(self):
        """LCT-based FFT Fresnel propagation"""
        raise NotImplementedError()

    def propagate_fraunhofer(self):
        """
        Fraunhofer propagation via MFT (same as propagate MFT, but with abcdLux backend)
        """
        raise NotImplementedError()

    def propagate_fraunhofer_fft(self):
        """
        Fraunhofer propagation via FFT (same as propagate FFT, but with abcdLux backend)
        """
        raise NotImplementedError()

    def _magic_unified_op(
        self: Wavefront, other: Wavefront | Array | None, op: str
    ) -> Wavefront:
        """
        Internal helper function to unify the logic of the magic methods for addition,
        subtraction, multiplication and division.

        Parameters
        ----------
        other : Wavefront | Array | None
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

    def __add__(self: Wavefront, other: Wavefront | Array | None) -> Wavefront:
        """
        Allows complex phasors or Wavefront objects to be added together. None values
        are ignored.
        """
        return self._magic_unified_op(other, "add")

    def __sub__(self: Wavefront, other: Wavefront | Array | None) -> Wavefront:
        """
        Allows complex phasors or Wavefront objects to be subtracted. None values are
        ignored.
        """
        return self._magic_unified_op(other, "subtract")

    def __mul__(self: Wavefront, other: Wavefront | Array | None) -> Wavefront:
        """
        Allows complex phasors or Wavefront objects to be multiplied. None values are
        ignored.
        """
        return self._magic_unified_op(other, "multiply")

    def __truediv__(self: Wavefront, other: Wavefront | Array | None) -> Wavefront:
        """
        Allows complex phasors or Wavefront objects to be divided. None values are
        ignored.
        """
        return self._magic_unified_op(other, "divide")

    def __iadd__(self: Wavefront, other: Wavefront | Array | None) -> Wavefront:
        """In-place addition."""
        return self.__add__(other)

    def __isub__(self: Wavefront, other: Wavefront | Array | None) -> Wavefront:
        """In-place subtraction."""
        return self.__sub__(other)

    def __imul__(self: Wavefront, other: Wavefront | Array | None) -> Wavefront:
        """In-place multiplication."""
        return self.__mul__(other)

    def __itruediv__(self: Wavefront, other: Wavefront | Array | None) -> Wavefront:
        """In-place division."""
        return self.__truediv__(other)


class PolarisedWavefront(Wavefront):
    """
    A polarisation wavefront, supporting partial polarisation.
    The internal represation uses Jones calculus in the general case,
    i.e. tracking a 2x2 complex coherence matrix for the state.

    If, for whatever reason, you need a strictly polarised wavefront, add a PR.
    """

    initial_stokes: Array

    def __init__(
        self: Wavefront,
        wavelength: float,
        npixels: int,
        diameter: float = None,
        pixel_scale: float = None,
        center: Array = None,
        initial_stokes: Array = np.array([1.0, 0.0, 0.0, 0.0]),
    ):
        super().__init__(wavelength, npixels, diameter, pixel_scale, center)

        # stack to (2,2, npixels, npixels)
        self.phasor = np.stack(
            [
                np.stack([self.phasor, np.zeros_like(self.phasor)], axis=0),
                np.stack([np.zeros_like(self.phasor), self.phasor], axis=0),
            ],
            axis=0,
        )

        self.initial_stokes = initial_stokes

    @staticmethod
    def from_wavefront(
        wavefront: Wavefront, initial_stokes: Array = None
    ) -> PolarisedWavefront:
        """
        Promotes a regular Wavefront to a PolarisedWavefront by multiplying the scalar phasor
        by eye(2)
        Parameters
        ----------
        wavefront : Wavefront
            The input wavefront to promote.
        initial_stokes : Array = None
            The initial Stokes parameters to set for the polarised wavefront. If None, defaults to [1, 0, 0, 0] (fully unpolarised).

        Returns
        -------
        polarised_wavefront : PolarisedWavefront
            A new PolarisedWavefront with the same wavelength, pixel scale, and center as the input wavefront, and the phasor promoted

        """
        if initial_stokes is None:
            initial_stokes = np.array([1.0, 0.0, 0.0, 0.0])
        return PolarisedWavefront(
            wavelength=wavefront.wavelength,
            npixels=wavefront.npixels,
            diameter=wavefront.diameter,
            center=wavefront.center,
            initial_stokes=initial_stokes,
        )

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
        return np.sum(np.abs(self.phasor) ** 2, axis=0)

    @property
    def I(self: Wavefront) -> Array:
        """Stokes I parameter."""
        pass

    @property
    def Q(self: Wavefront) -> Array:
        """Stokes Q parameter."""
        pass

    @property
    def U(self: Wavefront) -> Array:
        """Stokes U parameter."""
        pass

    @property
    def V(self: Wavefront) -> Array:
        """Stokes V parameter."""
        pass

    @property
    def stokes(self: Wavefront) -> Array:
        """Returns the Stokes parameters as an array."""
        return np.stack([self.I, self.Q, self.U, self.V], axis=0)
