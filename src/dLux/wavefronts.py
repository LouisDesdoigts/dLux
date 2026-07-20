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
        return np.sum(self.psf)

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
        Flip the complex phasor along one or more axes (ij indexing: -2=y, -1=x). Note
        that if the wavefront is polarised, the polarisation dimensions are leading.

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
        method: str = "linear",
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
        method : str = "linear"
            The interpolation method.
        complex : bool = True
            If True, interpolate the real and imaginary components. If False,
            interpolate the amplitude and phase components.

        Returns
        -------
        wavefront : Wavefront
            The new interpolated wavefront.
        """
        if isinstance(method, bool):
            complex, method = method, "linear"

        ratio = pixel_scale / self.pixel_scale
        scale = np.vectorize(
            lambda phasor, ratio: dlu.scale(phasor, npixels, ratio, method, complex),
            signature="(n,n),()->(m,m)",
        )
        return self.set(phasor=scale(self.phasor, ratio), pixel_scale=pixel_scale)

    def interpolate(
        self: Wavefront,
        knot_coords: Array,
        sample_coords: Array,
        method: str = "linear",
        fill: float = 0.0,
        complex: bool = True,
    ) -> Wavefront:
        """
        Interpolates the wavefront onto a set of sample coordinates.

        Parameters
        ----------
        knot_coords : Array
            The coordinates of the sampled points in the wavefront.
        sample_coords : Array
            The coordinates to interpolate onto.
        method : str = "linear"
            The interpolation method.
        fill : float = 0.0
            Fill value used outside `knot_coords`.
        complex : bool = True
            If True, interpolate the real and imaginary components. If False,
            interpolate the amplitude and phase components.

        Returns
        -------
        wavefront : Wavefront
            The new interpolated wavefront.
        """
        interp = np.vectorize(
            lambda phasor: dlu.interp(
                phasor, knot_coords, sample_coords, method, fill, complex
            ),
            signature="(n,n)->(m,m)",
        )
        return self.set(phasor=interp(self.phasor))

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
        rotate = np.vectorize(
            lambda phasor, angle: dlu.rotate(phasor, angle, method, complex),
            signature="(n,n),()->(n,n)",
        )
        return self.set(phasor=rotate(self.phasor, angle))

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

    def downsample(self: Wavefront, n: int, mean: bool = True) -> Wavefront:
        """
        Downsamples the wavefront by a factor of n.

        Parameters
        ----------
        n : int
            The factor by which to downsample the wavefront.
        mean : bool = True
            Whether to downsample by taking the mean or sum of the phasor.

        Returns
        -------
        wavefront : Wavefront
            The downsampled wavefront.
        """
        phasor = dlu.downsample(self.phasor, n, mean)
        return self.set(phasor=phasor, pixel_scale=self.pixel_scale * n)

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
        if spec_out is not None:
            if spec_out.d is not None:
                raise ValueError("Output spec cannot specify d; FFT output d is fixed.")
            if spec_out.n is not None:
                raise ValueError(
                    "Output spec cannot specify n; FFT output n is determined by the "
                    "pad parameter."
                )

        output_center = None if spec_out is None else spec_out.c

        def prop_fn(phasor, wavelength, pixel_scale):
            return dlu.FFT(
                phasor,
                wavelength,
                pixel_scale,
                focal_length=focal_length,
                pad=pad,
                inverse=inverse,
                center=self.center,
                output_center=output_center,
            )

        prop_fn = np.vectorize(prop_fn, signature="(n,n),(),()->(m,m),(),()")
        phasor, pixel_scale, center = prop_fn(
            self.phasor, self.wavelength, self.pixel_scale
        )
        return self.set(phasor=phasor, pixel_scale=pixel_scale, center=center)

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
        - Supports broadband propagation via vectorised wavelength and pixel_scale.
        """
        fn = np.vectorize(
            lambda phasor, wavelength, pixel_scale_in, pixel_scale_out: dlu.MFT(
                phasor,
                wavelength,
                pixel_scale_in,
                npixels,
                pixel_scale_out,
                focal_length,
                inverse=inverse,
            ),
            signature="(n,n),(),(),()->(m,m)",
        )

        phasor = fn(self.phasor, self.wavelength, self.pixel_scale, pixel_scale)
        return self.set(phasor=phasor, pixel_scale=pixel_scale)

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
        fn = np.vectorize(
            lambda phasor, wavelength, pixel_scale_in, pixel_scale_out: dlu.MFT(
                phasor,
                wavelength,
                pixel_scale_in,
                spec_out.n,
                pixel_scale_out,
                focal_length,
                inverse=inverse,
            ),
            signature="(n,n),(),(),()->(m,m)",
        )

        phasor = fn(self.phasor, self.wavelength, self.pixel_scale, spec_out.d)
        return self.set(phasor=phasor, pixel_scale=spec_out.d)

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

    def apply_jones(self, jones):
        """
        Applies a Jones matrix by promoting it to a PolarisedWavefront and applying the
        Jones matrix.
        """
        return PolarisedWavefront.from_wavefront(self).apply_jones(jones)

    def psf_from_stokes(self, stokes: Array | None = None) -> Array:
        """
        Calculates the PSF from the input Stokes vector. Since the Wavefront presently
        has no polarisation state, the PSF is simply the total intensity (first Stokes
        parameter) multiplied by the PSF of the wavefront.
        """
        if stokes is None:
            return self.psf

        # For a polarisation-insensitive system, only total input intensity matters.
        return stokes[0] * self.psf


class PolarisedWavefront(Wavefront):
    """
    A polarisation wavefront, supporting partial polarisation.
    The internal representation uses Jones calculus in the general case,
    i.e. tracking a 2x2 complex coherence matrix for the state.

    If, for whatever reason, you need a strictly polarised wavefront, add a PR.
    """

    def __init__(
        self: Wavefront,
        wavelength: float,
        npixels: int,
        diameter: float = None,
        pixel_scale: float = None,
        center: Array = None,
    ):
        super().__init__(wavelength, npixels, diameter, pixel_scale, center)

        self.phasor = self.phasor[None, None] * np.eye(2)[:, :, None, None]

    def from_phasor(self):
        """
        Needed to handle input phasors that are already in Jones form or from a
        regular wavefront phasor
        """

        raise NotImplementedError(
            "PolarisedWavefront.from_phasor is not implemented yet."
        )

    @staticmethod
    def from_wavefront(wavefront: Wavefront) -> PolarisedWavefront:
        """
        Promotes a regular Wavefront to a PolarisedWavefront by multiplying the scalar
        phasor by eye(2)
        Parameters
        ----------
        wavefront : Wavefront
            The input wavefront to promote.

        Returns
        -------
        polarised_wavefront : PolarisedWavefront
            A new PolarisedWavefront with the same wavelength, pixel scale, and center
            as the input wavefront, and the phasor promoted

        """
        pwf = PolarisedWavefront(
            wavelength=wavefront.wavelength,
            npixels=wavefront.npixels,
            diameter=wavefront.diameter,
            center=wavefront.center,
        )

        jones_phasor = wavefront.phasor[None, None] * np.eye(2)[:, :, None, None]
        return pwf.set("phasor", jones_phasor)

    @property
    def psf(self: Wavefront) -> Array:
        """Assumes an unpolarised input Stokes vector of [1, 0, 0, 0]"""
        return self.psf_from_stokes()

    def psf_from_stokes(self: Wavefront, input_stokes: Array | None = None) -> Array:
        """Produces the PSF from the input Stokes vector"""
        stokes = self.stokes(input_stokes)
        if self.ndim > 0:
            return stokes[:, 0]
        return stokes[0]

    def stokes(self: Wavefront, input_stokes: Array | None = None) -> Array:
        """
        Returns the Stokes parameters as an array. Note that we have to explicitly
        handle the broadband/vectorised wavefront case here by checking the
        dimensionality of the wavefront. This is necessary since the jones_to_stokes
        function vectorises over the trailing dimensions of the phasor, so we can't
        also vectorise it over arbitrary leading dimensions. We _could_ get around this
        by swapping the jones dimensions from being leading to being trailing, but we
        have implicitly adopted the convention that the trailing axes are the spatial
        ones. The other option would be to manually change the output axes for the
        phasor when returning the wavefront from filter_vmap, but that requires manual
        tuning and wont work for non-advanced users. Therefore the simplest solution is
        to just check the dimensionality of the wavefront and handle the vectorisation
        manually here.
        """
        # Manually handle potentially chromatic wavefront
        if self.ndim > 0:
            return vmap(lambda x: dlu.jones_to_stokes(x, input_stokes))(self.phasor)
        return dlu.jones_to_stokes(self.phasor, input_stokes)

    def apply_jones(self, jones):
        """Applies a Jones matrix to the polarised wavefront."""
        return self.set(phasor=dlu.apply_jones(jones, self.phasor))
