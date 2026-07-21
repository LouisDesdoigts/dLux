"""Wavefront state and propagation utilities used by optical systems."""

from __future__ import annotations
import jax.numpy as np
from jax import Array
import zodiax as zdx
import dLux.utils as dlu

from .psfs import PSF
from .coordinates import CoordSpec

__all__ = ["Wavefront", "PolarisedWavefront"]

# TODO: Make coord specs 2d compatible


class Wavefront(zdx.Base):
    """
    Holds the state of a wavefront as it is transformed and propagated through an
    optical system. The final two phasor axes are the square spatial wavefront; any
    preceding axes are treated as vectorisation axes. Passing a vector of wavelengths
    creates a chromatic wavefront with matching leading phasor dimensions.

    ??? abstract "UML"
        ![UML](../assets/uml/Wavefront.png)

    Attributes
    ----------
    wavelength : float or Array, meters
        The wavelength of the `Wavefront`. Vector-valued wavelengths define a chromatic
        wavefront.
    phasor : Array[complex]
        The electric field of the `Wavefront`, with shape `(..., npixels, npixels)`.
        Leading dimensions are vectorisation dimensions.
    pixel_scale : float or Array, meters/pixel
        The pixel scale of the phase and amplitude arrays. This may be scalar or
        vectorised over the leading phasor dimensions.
    center : float
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
    complex : Array, property
        Derived property from `phasor`; `(real, imaginary)` representation.
    polar : Array, property
        Derived property from `phasor`; `(amplitude, phase)` representation.
    wavenumber : float or Array, property
        Derived property from `wavelength`; scalar `2 * pi / wavelength`.
    ndim : int, property
        Derived property from `phasor`; vectorisation rank of wavefront state.
    is_chromatic : bool, property
        Derived property from `wavelength`; whether wavelength is vector-valued.
    power : Array, property
        Derived property from `amplitude`; total wavefront power.
    spec : CoordSpec, property
        Derived coordinate specification for the current wavefront sampling.
    xs : Array, property
        Derived pixel-centre coordinates along one axis, in metres.
    """

    phasor: Array[complex]
    wavelength: Array
    pixel_scale: Array
    center: Array

    def __init__(
        self: Wavefront,
        wavelength: float | Array,
        npixels: int,
        diameter: float | Array = None,
        pixel_scale: float | Array = None,
        center: float | Array = None,
    ):
        """
        Parameters
        ----------
        wavelength : float or Array, meters
            The wavelength of the `Wavefront`. Passing an array creates a chromatic
            wavefront with phasor shape `wavelength.shape + (npixels, npixels)`.
        npixels : int
            The number of pixels that represent the `Wavefront`.
        diameter : float = None, meters
            The total diameter of the `Wavefront`. Either `diameter` or `pixel_scale`
            must be provided.
        pixel_scale : float or Array = None, meters/pixel
            The pixel scale of the `Wavefront`. Either `diameter` or `pixel_scale`
            must be provided. Scalar values are broadcast across chromatic axes.
        center : float = None
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

        shape = self.wavelength.shape + (npixels, npixels)
        amplitude = np.ones(shape, dtype=float) / npixels**2
        phase = np.zeros(shape, dtype=float)
        self.phasor = amplitude * np.exp(1j * phase)

        self.center = np.asarray(0.0 if center is None else center, float)
        if self.center.ndim != 0:
            raise ValueError(
                f"center must be scalar, got shape {self.center.shape}. "
                "For the current square-array wavefront convention, the same "
                "centre coordinate is used for both axes."
            )

    @classmethod
    def from_phasor(
        cls,
        phasor: Array[complex],
        wavelength: float | Array,
        pixel_scale: float | Array = None,
        diameter: float | Array = None,
        center: float | Array = None,
    ) -> Wavefront:
        """
        Create a Wavefront from an existing phasor array.

        Parameters
        ----------
        phasor : Array[complex]
            The complex electric field array. The final two axes are spatial; leading
            axes are vectorisation axes. If a 2D phasor is passed with vector
            wavelengths, it is broadcast over the wavelength axes.
        wavelength : float or Array, meters
            The wavelength of the wavefront. Vector-valued wavelengths define a
            chromatic wavefront.
        pixel_scale : float or Array = None, meters/pixel
            The pixel scale of the phasor array. Either `pixel_scale` or
            `diameter` must be provided. Scalar values are broadcast across
            chromatic axes.
        diameter : float = None, meters
            The diameter of the phasor array. Either `pixel_scale` or
            `diameter` must be provided.
        center : float = None
            The centre coordinate of the wavefront grid, in metres. Defaults to zero.

        Returns
        -------
        wavefront : Wavefront
            A new Wavefront object with the specified phasor.
        """
        phasor_arr = np.asarray(phasor, complex)
        wavelength = np.asarray(wavelength, float)
        if phasor_arr.ndim == 2 and wavelength.ndim > 0:
            phasor_arr = phasor_arr * np.ones(wavelength.shape + (1, 1))
        npixels = phasor_arr.shape[-1]

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
        number of pixels. If the pixel scale is vectorised, the diameter is also
        vectorised.

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
        Taken from the final phasor axis.

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
        Returns the real and imaginary components of the `Wavefront`.

        Returns
        -------
        wavefront : Array
            The complex representation with component axis first.
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
            A PSF object containing the current wavefront intensity and pixel scale,
            including any leading chromatic dimensions.
        """
        return PSF(self.psf, self.pixel_scale)

    @property
    def wavenumber(self: Wavefront) -> Array:
        """
        Returns the wavenumber of the wavefront (2 * pi / wavelength), with the same
        shape as `wavelength`.

        Returns
        -------
        wavenumber : Array, 1/meters
            The wavenumber of the wavefront.
        """
        return 2 * np.pi / self.wavelength

    @property
    def ndim(self: Wavefront) -> int:
        """
        Returns the number of leading vectorisation dimensions on the phasor.

        Returns
        -------
        ndim : int
            The number of non-spatial dimensions before the final two phasor axes.
        """
        return self.phasor.ndim - 2

    @property
    def is_chromatic(self: Wavefront) -> bool:
        """
        Returns whether the wavefront carries a leading wavelength dimension.

        Returns
        -------
        is_chromatic : bool
            True if the wavefront wavelength is vectorised.
        """
        return self.wavelength.ndim > 0

    @property
    def _mapped_axis(self: Wavefront) -> Wavefront | None:
        """
        Returns the input-axis specification for mapping over wavelength.

        Dimensional sampling metadata is mapped over its leading axis, while scalar
        metadata is shared. The phasor is mapped only when the wavefront carries a
        vectorisation dimension, ensuring intrinsic axes such as the Jones axes of a
        `PolarisedWavefront` are never mistaken for a wavelength axis.

        Returns
        -------
        mapped_axis : Wavefront | None
            A Wavefront-shaped pytree containing ``0`` for mapped leaves and ``None``
            for shared leaves. Returns ``None`` for a monochromatic wavefront.
        """
        if not self.is_chromatic:
            return None

        get_axis = lambda array: 0 if array.ndim > 0 else None
        return self.set(
            phasor=0 if self.ndim > 0 else None,
            wavelength=0,
            pixel_scale=get_axis(self.pixel_scale),
            center=get_axis(self.center),
        )

    @property
    def power(self: Wavefront) -> Array:
        """
        Returns the total power of the wavefront, summed over all phasor entries.

        Returns
        -------
        power : Array
            The total power of the wavefront.
        """
        return np.sum(self.psf)

    def _to_phasor_shape(self: Wavefront, array: Array) -> Array:
        """
        Reshape scalar or spatial arrays to broadcast against the phasor, preserving
        chromatic and other leading vectorisation axes.

        Parameters
        ----------
        array : Array
            Input scalar, vectorised scalar, spatial, or vectorised spatial array.

        Returns
        -------
        array : Array
            The input reshaped to broadcast over the phasor axes.
        """
        array = np.asarray(array)
        chromatic_ndim = self.wavelength.ndim
        extra_ndim = self.phasor.ndim - chromatic_ndim - 2

        if array.ndim == chromatic_ndim:
            return array.reshape(array.shape + (1,) * (extra_ndim + 2))
        if array.ndim == chromatic_ndim + 2:
            return array.reshape(
                array.shape[:chromatic_ndim] + (1,) * extra_ndim + array.shape[-2:]
            )
        return array

    def _to_vec(self: Wavefront, array: Array) -> Array:
        """
        Reshape metadata before passing it into a vectorised scalar function.

        Metadata lives on the physical vector axes counted by `self.ndim`. If a subclass
        stores extra axes between those vector axes and the final spatial axes, append
        singleton dimensions so NumPy's right-aligned broadcasting keeps metadata on the
        physical axes.
        """
        array = np.asarray(array)
        extra_ndim = self.phasor.ndim - self.ndim - 2
        if array.ndim == self.ndim:
            return array.reshape(array.shape + (1,) * extra_ndim)
        return array

    def _from_vec(self: Wavefront, array: Array) -> Array:
        """
        Remove redundant non-physical axes from metadata returned by `np.vectorize`.
        """
        array = np.asarray(array)
        extra_ndim = self.phasor.ndim - self.ndim - 2
        if extra_ndim and array.ndim == self.ndim + extra_ndim:
            return array[(...,) + (0,) * extra_ndim]
        return array

    def add_phase(self: Wavefront, phase: float | Array) -> Wavefront:
        """
        Applies a phase (in radians) to the wavefront by multiplying the phasor by
        exp(1j * phase). Scalar, spatial, chromatic, and vectorised phases are broadcast
        to the phasor shape.

        Parameters
        ----------
        phase : float or Array, radians
            The phase to be added to the wavefront.

        Returns
        -------
        wavefront : Wavefront
            New wavefront whose phasor is self.phasor * exp(1j * phase).
        """
        if phase is None:
            return self
        return self.multiply("phasor", np.exp(1j * self._to_phasor_shape(phase)))

    def add_opd(self: Wavefront, opd: float | Array) -> Wavefront:
        """
        Applies an optical path difference (in meters) by multiplying the phasor by
        exp(1j * k * opd), where k = 2*pi / wavelength. Scalar, spatial, chromatic, and
        vectorised OPDs are broadcast to the phasor shape.

        Parameters
        ----------
        opd : float or Array, meters
            The optical path difference to apply.

        Returns
        -------
        wavefront : Wavefront
            New wavefront with phasor multiplied by exp(1j * k * opd).
        """
        if opd is None:
            return self
        return self.add_phase(self.wavenumber[..., None, None] * np.asarray(opd))

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
        return self.add_opd(np.einsum("i,...ijk->...jk", angles, coords))

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
            - "power": scales so sum(|E|^2) == value across the full phasor.
            - "peak" : scales so max(|E|^2) == value across the full phasor.
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
        Flip the complex phasor along one or more axes. The final two axes are spatial
        (`-2=y`, `-1=x`); leading axes are vectorisation dimensions.

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
        pixel_scale: float | Array,
        method: str = "linear",
        complex: bool = True,
    ) -> Wavefront:
        """
        Interpolates the wavefront to a given npixels and pixel_scale. Leading phasor
        dimensions are vectorised over directly.

        Parameters
        ----------
        npixels : int
            The number of pixels to interpolate to.
        pixel_scale: float or Array
            The pixel scale to interpolate to. Scalar values are broadcast over leading
            phasor dimensions.
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
        pixel_scale = np.asarray(pixel_scale, float)
        ratio = pixel_scale / self.pixel_scale

        # Interpolate the phasor to the new pixel scale and size
        fn = lambda phasor, ratio: dlu.scale(phasor, npixels, ratio, method, complex)
        phasor = np.vectorize(fn, signature="(n,n),()->(m,m)")(self.phasor, ratio)
        return self.set(phasor=phasor, pixel_scale=pixel_scale)

    def interpolate(
        self: Wavefront,
        knot_coords: Array,
        sample_coords: Array,
        method: str = "linear",
        fill: float = 0.0,
        complex: bool = True,
    ) -> Wavefront:
        """
        Interpolates the wavefront onto a set of sample coordinates. Leading phasor
        dimensions are vectorised over directly.

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
        angle: float | Array,
        method: str = "linear",
        complex: bool = True,
    ) -> Wavefront:
        """
        Rotates the wavefront by a given angle via interpolation. Leading phasor
        dimensions are vectorised over directly.

        Parameters
        ----------
        angle : float or Array, radians
            The angle by which to rotate the wavefront in a clockwise direction. Scalar
            values are broadcast over leading phasor dimensions.
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
        Resizes the spatial axes of the wavefront via zero-padding or cropping,
        preserving leading vectorisation dimensions.

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
        Downsamples the spatial axes of the wavefront by a factor of n, preserving
        leading vectorisation dimensions.

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
        optional scaling factor for numerical stability. If the coordinate spec is
        vectorised, coordinates include matching leading dimensions.

        Parameters
        ----------
        scale : float = 1.0
            Optional scaling factor applied to the diameter for numerical stability.
        polar : bool = False
            Output the coordinates in polar (r, phi) coordinates.

        Returns
        -------
        coordinates : Array
            The coordinates of each pixel centre, with shape `(..., 2, n, n)`.
        """
        xs = self.xs * scale
        coords_fn = lambda xs: np.array(np.meshgrid(xs, xs))
        coords = np.vectorize(coords_fn, signature="(n)->(c,n,n)")(xs)

        if polar:
            return np.vectorize(dlu.cart2polar, signature="(c,n,n)->(c,n,n)")(coords)
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
        1D array of pixel centre coordinates along one axis. If the coordinate spec is
        vectorised, the returned array has shape `(..., n)`.

        Returns
        -------
        xs : Array
            Coordinates of pixel centres, in metres.
        """
        return self.spec.xs

    def set_spec(self, spec: CoordSpec):
        """
        Updates the wavefront pixel scale and centre from a `CoordSpec`, preserving
        the current phasor.

        Parameters
        ----------
        spec : CoordSpec
            The coordinate specification to apply.

        Returns
        -------
        wavefront : Wavefront
            New wavefront with updated `pixel_scale` and `center`.
        """
        pixel_scale = None if spec.d is None else np.asarray(spec.d, float)
        center = None if spec.c is None else np.asarray(spec.c, float)
        return self.set(pixel_scale=pixel_scale, center=center)

    def propagate_FFT(
        self,
        pad=2,
        focal_length=None,
        spec_out: CoordSpec = None,
        inverse=False,
    ):
        """
        Propagates the wavefront using an FFT-based method. Leading phasor dimensions
        are vectorised over directly, so chromatic wavefronts propagate once per
        wavelength.

        Parameters
        ----------
        pad : int = 2
            Zero-padding factor applied before the FFT.
        focal_length : float | None = None
            Focal length for Cartesian focal sampling. Pass `None` for
            angular (far-field) sampling.
        spec_out : CoordSpec | None = None
            Output coordinate specification. If provided, only `c` (centre) may be set;
            `n` and `d` are determined by the propagation. Scalar centres are broadcast
            over leading phasor dimensions.
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

        def prop_fn(phasor, wavelength, pixel_scale, center):
            return dlu.FFT(
                phasor,
                wavelength,
                pixel_scale,
                focal_length=focal_length,
                pad=pad,
                inverse=inverse,
                center=center,
                output_center=output_center,
            )

        prop_fn = np.vectorize(prop_fn, signature="(n,n),(),(),()->(m,m),(),()")
        phasor, pixel_scale, center = prop_fn(
            self.phasor,
            self._to_vec(self.wavelength),
            self._to_vec(self.pixel_scale),
            self._to_vec(self.center),
        )
        return self.set(
            phasor=phasor,
            pixel_scale=self._from_vec(pixel_scale),
            center=self._from_vec(center),
        )

    def propagate(
        self: Wavefront,
        npixels: int,
        pixel_scale: float | Array,
        focal_length: float = None,
        inverse: bool = False,
    ) -> Wavefront:
        """
        Legacy MFT propagation function without CoordSpec. Leading phasor dimensions
        are vectorised over directly.

        Parameters
        ----------
        npixels : int
            Output array size (square).
        pixel_scale : float or Array
            Desired output pixel scale (meters/pixel or radians/pixel depending on
            units). Scalar values are broadcast over leading phasor dimensions.
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
        - Supports chromatic propagation via leading phasor and wavelength dimensions.
        """
        pixel_scale = np.asarray(pixel_scale, float)
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

        phasor = fn(
            self.phasor,
            self._to_vec(self.wavelength),
            self._to_vec(self.pixel_scale),
            self._to_vec(pixel_scale),
        )
        return self.set(phasor=phasor, pixel_scale=pixel_scale)

    def propagate_MFT(self, spec_out, focal_length=None, inverse=None):
        """
        Propagates the wavefront using an MFT-based method with a `CoordSpec`. Leading
        phasor dimensions are vectorised over directly.

        Parameters
        ----------
        spec_out : CoordSpec
            Output coordinate specification defining the number of pixels and pixel
            scale of the propagated field. Scalar pixel scales are broadcast over
            leading phasor dimensions.
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

        pixel_scale = np.asarray(spec_out.d, float)
        phasor = fn(
            self.phasor,
            self._to_vec(self.wavelength),
            self._to_vec(self.pixel_scale),
            self._to_vec(pixel_scale),
        )
        return self.set(phasor=phasor, pixel_scale=pixel_scale)

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

        if op not in ("add", "subtract", "multiply", "divide"):
            raise ValueError(f"Unsupported operation '{op}'.")

        # Division between two wavefront states has no well-defined optical meaning.
        if op == "divide" and isinstance(other, Wavefront):
            raise TypeError(
                "dLux has detected an attempt to perform dark optics. Your wavefront "
                "privileges have been temporarily suspended and the authorities have "
                "been notified."
            )

        # Align the operand for chromatic and polarisation broadcasting.
        self, other = self._prepare_operand(other)

        # Apply the operation
        if op == "add":
            output = self.add("phasor", other)
        elif op == "subtract":
            output = self.add("phasor", -other)
        elif op == "multiply":
            output = self.multiply("phasor", other)
        elif op == "divide":
            output = self.multiply("phasor", 1 / other)
        return output

    def _prepare_operand(
        self: Wavefront, other: Wavefront | Array | float | int | complex
    ) -> tuple[Wavefront, Array | float | int | complex]:
        """Dispatches operand preparation according to the operand type."""
        if isinstance(other, Wavefront):
            return self._prepare_wavefront_operand(other)
        if isinstance(other, Array):
            return self._prepare_array_operand(other)
        return self, other

    def _promote_for_arithmetic(self: Wavefront) -> PolarisedWavefront:
        """Promotes the base type while retaining scalar Jones broadcasting."""
        phasor = self.phasor[..., None, None, :, :]
        return PolarisedWavefront.from_wavefront(self).set(phasor=phasor)

    def _prepare_wavefront_operand(
        self: Wavefront, other: Wavefront
    ) -> tuple[Wavefront, Array]:
        """
        Prepares an operand for elementwise wavefront arithmetic.

        The left operand is treated as the base wavefront and therefore supplies the
        output wavelength and sampling metadata. A monochromatic right operand can
        broadcast over a chromatic base, but a chromatic right operand must match the
        base wavelength shape. For mixed polarisation, singleton Jones axes are added
        to the regular phasor so it broadcasts across every polarisation component.

        Parameters
        ----------
        other : Wavefront | Array
            The operand to align with the base wavefront.

        Returns
        -------
        wavefront : Wavefront
            The base wavefront, promoted to `PolarisedWavefront` if required.
        operand : Array
            The array operand aligned for elementwise arithmetic.
        """
        if self.npixels != other.npixels:
            raise ValueError("Wavefront operands must have matching spatial shapes.")

        # A chromatic right operand cannot be represented by a monochromatic base,
        # and two chromatic operands require matching wavelength dimensions.
        if other.is_chromatic and (
            not self.is_chromatic or self.wavelength.shape != other.wavelength.shape
        ):
            raise ValueError(
                "A chromatic Wavefront operand requires a chromatic base with the "
                "same wavelength shape."
            )

        self_polarised = isinstance(self, PolarisedWavefront)
        other_polarised = isinstance(other, PolarisedWavefront)

        # Matching polarisation types already have compatible intrinsic dimensions.
        if self_polarised == other_polarised:
            return self, other.phasor

        # Regular phasors act as scalar Jones modulation, so insert singleton Jones
        # axes immediately before their spatial dimensions.
        if self_polarised:
            return self, other.phasor[..., None, None, :, :]

        return self._promote_for_arithmetic(), other.phasor

    def _prepare_array_operand(
        self: Wavefront, other: Array
    ) -> tuple[Wavefront, Array]:
        """
        Classifies and aligns an array operand by its semantic dimensions.

        Supported array layouts are scalar, spectral, spatial, spectral-spatial,
        Jones, spectral-Jones, Jones-spatial, and spectral-Jones-spatial. Ambiguous
        layouts are rejected rather than assigned an implicit interpretation.

        Parameters
        ----------
        other : Array
            The array operand to align with the base wavefront.

        Returns
        -------
        wavefront : Wavefront
            The base wavefront, promoted to `PolarisedWavefront` if required.
        operand : Array
            The operand reshaped for elementwise arithmetic.
        """
        # Layouts use the canonical wavelength, Jones, then spatial axis order.
        layouts = (
            (),
            ("w",),
            ("x", "y"),
            ("w", "x", "y"),
            ("j0", "j1"),
            ("w", "j0", "j1"),
            ("j0", "j1", "x", "y"),
            ("w", "j0", "j1", "x", "y"),
        )

        def matches(layout):
            """Checks whether the operand shape matches a semantic layout."""
            if len(layout) != other.ndim or ("w" in layout and not self.is_chromatic):
                return False

            axes = dict(zip(layout, other.shape))
            wavelength_matches = (
                axes.get("w", self.wavelength.size) == self.wavelength.size
            )
            jones_matches = all(axes.get(axis, 2) == 2 for axis in ("j0", "j1"))
            spatial_matches = axes.get("x") == axes.get("y")

            # A two-pixel spatial axis is only spatial when the base agrees. This
            # leaves (2, 2) arrays unambiguously Jones-valued for larger wavefronts.
            if "x" in axes and axes["x"] == 2 and self.npixels != 2:
                spatial_matches = False
            return wavelength_matches and jones_matches and spatial_matches

        matches = [layout for layout in layouts if matches(layout)]
        if len(matches) > 1:
            if other.ndim == 2:
                raise ValueError("Array shape (2, 2) is ambiguous for npixels=2.")
            raise ValueError(
                "Array shape is ambiguous between spectral-spatial and "
                "spectral-Jones layouts."
            )
        if not matches:
            if other.ndim == 1:
                raise ValueError(
                    "A vector operand must match the base wavelength shape."
                )
            raise ValueError(
                f"Unsupported array shape {other.shape} for a Wavefront with phasor "
                f"shape {self.phasor.shape}."
            )

        layout = matches[0]
        if "j0" in layout and not isinstance(self, PolarisedWavefront):
            self = self._promote_for_arithmetic()

        # Insert singleton dimensions for semantic axes absent from the operand.
        axes = ("w",) if self.is_chromatic else ()
        if isinstance(self, PolarisedWavefront):
            axes += ("j0", "j1")
        axes += ("x", "y")
        shape = tuple(
            other.shape[layout.index(axis)] if axis in layout else 1 for axis in axes
        )
        return self, other.reshape(shape)

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
        Allows the wavefront phasor to be divided by a scalar or array. Division by
        another Wavefront is undefined. None values are ignored.
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
    i.e. tracking a 2x2 complex coherence matrix for the state. Phasors use shape
    `(..., 2, 2, n, n)`, where leading axes are vectorisation axes, the next two axes
    are Jones axes, and the final two axes are spatial.

    If, for whatever reason, you need a strictly polarised wavefront, add a PR.
    """

    def __init__(
        self: Wavefront,
        wavelength: float | Array,
        npixels: int,
        diameter: float | Array = None,
        pixel_scale: float | Array = None,
        center: Array = None,
    ):
        super().__init__(wavelength, npixels, diameter, pixel_scale, center)
        self.phasor = self._promote_phasor(self.phasor)

    @staticmethod
    def _promote_phasor(phasor: Array) -> Array:
        """
        Promote a scalar phasor into an unpolarised Jones phasor.

        Input phasors have shape `(..., n, n)`. The leading `...` axes are preserved,
        and identity Jones axes are inserted immediately before the spatial axes:
        `(..., n, n) -> (..., 2, 2, n, n)`.
        """
        vector_shape = phasor.shape[:-2]
        eye = np.eye(2, dtype=complex)

        # Give the identity matrix singleton vector and spatial axes so it broadcasts
        # cleanly against `phasor[..., None, None, :, :]`.
        eye = eye.reshape((1,) * len(vector_shape) + (2, 2, 1, 1))
        return phasor[..., None, None, :, :] * eye

    @classmethod
    def from_phasor(
        cls,
        phasor: Array[complex],
        wavelength: float | Array,
        pixel_scale: float | Array = None,
        diameter: float | Array = None,
        center: float = None,
    ) -> PolarisedWavefront:
        """
        Create a PolarisedWavefront from a regular or Jones phasor.

        Parameters
        ----------
        phasor : Array[complex]
            Regular phasor with shape `(..., n, n)` or Jones phasor with shape
            `(..., 2, 2, n, n)`.
        wavelength : float or Array, meters
            The wavelength of the wavefront. If a 2D phasor is passed with vector
            wavelengths, it is broadcast over the wavelength axes.
        pixel_scale : float or Array = None, meters/pixel
            The pixel scale of the phasor array. Either `pixel_scale` or `diameter`
            must be provided.
        diameter : float or Array = None, meters
            The diameter of the phasor array. Either `pixel_scale` or `diameter`
            must be provided.
        center : float = None
            The centre coordinate of the wavefront grid, in metres. Defaults to zero.

        Returns
        -------
        wavefront : PolarisedWavefront
            A new polarised wavefront with phasor shape `(..., 2, 2, n, n)`.
        """
        phasor = np.asarray(phasor, complex)
        wavelength = np.asarray(wavelength, float)

        # Jones phasors already have the Jones axes immediately before the final two
        # spatial axes: `(..., 2, 2, n, n)`.
        is_jones = phasor.ndim >= 4 and phasor.shape[-4:-2] == (2, 2)

        # A single spatial phasor with vector wavelengths represents the same spatial
        # field at each wavelength, so add the wavelength axes before promotion.
        if phasor.ndim == 2 and wavelength.ndim > 0:
            phasor = phasor * np.ones(wavelength.shape + (1, 1))

        # A single Jones phasor with vector wavelengths is broadcast in the same way,
        # preserving the Jones axes before the spatial axes.
        elif is_jones and phasor.ndim == 4 and wavelength.ndim > 0:
            phasor = phasor * np.ones(wavelength.shape + (1, 1, 1, 1))

        # Regular phasors are converted to an unpolarised Jones representation.
        # Jones phasors are assumed to already be in the desired axis order.
        if not is_jones:
            phasor = cls._promote_phasor(phasor)

        # Construct the object to initialise wavelength/sampling metadata, then replace
        # the default identity phasor with the supplied phasor.
        return cls(
            wavelength=wavelength,
            npixels=phasor.shape[-1],
            diameter=diameter,
            pixel_scale=pixel_scale,
            center=center,
        ).set(phasor=phasor)

    @property
    def ndim(self: PolarisedWavefront) -> int:
        """
        Returns the number of leading vectorisation dimensions, excluding the Jones
        and spatial axes.
        """
        return self.phasor.ndim - 4

    @staticmethod
    def from_wavefront(wavefront: Wavefront) -> PolarisedWavefront:
        """
        Promotes a regular Wavefront to a PolarisedWavefront.

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
            pixel_scale=wavefront.pixel_scale,
            center=wavefront.center,
        )

        return pwf.set(phasor=PolarisedWavefront._promote_phasor(wavefront.phasor))

    @property
    def psf(self: Wavefront) -> Array:
        """Assumes an unpolarised input Stokes vector of [1, 0, 0, 0]"""
        return self.psf_from_stokes()

    def psf_from_stokes(self: Wavefront, input_stokes: Array | None = None) -> Array:
        """Produces the PSF from the input Stokes vector"""
        if input_stokes is None:
            return 0.5 * np.sum(np.abs(self.phasor) ** 2, axis=(-4, -3))
        stokes = self.stokes(input_stokes)
        return stokes[..., 0, :, :]

    def stokes(self: Wavefront, input_stokes: Array | None = None) -> Array:
        """
        Returns the Stokes parameters as an array.

        The polarised wavefront stores phasors as `(..., 2, 2, n, n)`, while the
        polarisation utilities operate on `(2, 2, ...)`. We move the Jones axes to the
        front, call the utility function, then move the Stokes axis back behind any
        leading wavefront dimensions.
        """
        phasor = np.moveaxis(self.phasor, (-4, -3), (0, 1))
        stokes = dlu.jones_to_stokes(phasor, input_stokes)
        return np.moveaxis(stokes, 0, -3)

    def apply_jones(self, jones):
        """
        Applies a Jones matrix to the polarised wavefront.

        The Jones matrix follows the utility convention `(2, 2, ...)`. The wavefront
        Jones axes are moved to the front before applying the utility function, then
        moved back to preserve `(..., 2, 2, n, n)` ordering.
        """
        phasor = np.moveaxis(self.phasor, (-4, -3), (0, 1))
        phasor = dlu.apply_jones(jones, phasor)
        return self.set(phasor=np.moveaxis(phasor, (0, 1), (-4, -3)))
