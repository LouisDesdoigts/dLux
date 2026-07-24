"""Wavefront state and propagation utilities used by optical systems."""

from __future__ import annotations
from math import prod

import jax.numpy as np
import zodiax as zdx
from jax import Array
import dLux.utils as dlu

from .coordinates import CoordSpec
from .coordinates import CoordTransform

__all__ = ["BaseSpatial", "Wavefront", "PolarisedWavefront"]


class BaseSpatial(zdx.Base):
    """Base class for regularly sampled real or complex spatial objects."""

    spec: CoordSpec

    def __init__(self, spec: CoordSpec):
        if not isinstance(spec, CoordSpec):
            raise TypeError("spec must be a CoordSpec.")
        self.spec = spec

    def __getattr__(self, key):
        """Forward coordinate attributes to the stored specification."""
        if key == "spec":
            raise AttributeError(key)
        try:
            spec = object.__getattribute__(self, "spec")
        except AttributeError:
            raise AttributeError(key) from None
        if hasattr(spec, key):
            return getattr(spec, key)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {key!r}.")

    @property
    def _field_name(self) -> str:
        """Return the name of the stored sampled array."""
        raise NotImplementedError()

    @property
    def field(self) -> Array:
        """Return the stored sampled array."""
        return getattr(self, self._field_name)

    def set_field(self, field: Array) -> BaseSpatial:
        """Return a copy with an updated sampled array."""
        return self.set(**{self._field_name: field})

    def _apply_field_op(self, other, op: str) -> BaseSpatial:
        """Apply one arithmetic operation to the stored sampled array."""
        if op == "add":
            field = self.field + other
        elif op == "subtract":
            field = self.field - other
        elif op == "multiply":
            field = self.field * other
        elif op == "divide":
            field = self.field / other
        else:
            raise ValueError(f"Unsupported operation '{op}'.")
        return self.set_field(field)

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        """Return the spatial array shape."""
        return self.field.shape[-2:]

    @property
    def axes(self) -> tuple[Array, ...]:
        """Return coordinate axes using the field's static spatial shape."""
        return self.spec.axes_for(self.spatial_shape[::-1])

    @property
    def coordinates(self) -> Array:
        """Return coordinates using the field's static spatial shape."""
        return self.spec.coordinates_for(self.spatial_shape[::-1])

    @property
    def xs(self) -> Array:
        """Return stacked coordinate axes using the field's static spatial shape."""
        return self.spec.xs_for(self.spatial_shape[::-1])

    @property
    def npixels(self) -> int:
        """Return the final spatial-axis size for square-grid compatibility."""
        return self.field.shape[-1]

    @property
    def pixel_scale(self) -> Array:
        """Return x-axis sampling in canonical SI units."""
        if self.d is None:
            raise ValueError("spec.d is not defined.")
        return self.d[0] * self.scale

    @property
    def center(self) -> Array:
        """Return the x-axis grid center in canonical SI units."""
        return 0.0 if self.c is None else self.c[0] * self.scale

    @property
    def diameter(self) -> Array:
        """Return the x-axis field width for square-grid compatibility."""
        return self.fov[0]

    def scale_to(
        self,
        npixels: int,
        pixel_scale: float | Array,
        method: str = "linear",
        complex: bool = True,
    ) -> BaseSpatial:
        """Interpolate to a square size and physical pixel scale.

        ``complex`` selects Cartesian or polar decomposition for complex fields and
        has no effect on real fields such as PSFs.
        """
        pixel_scale = np.asarray(pixel_scale, float)
        ratio = pixel_scale / self.pixel_scale
        scale = np.vectorize(
            lambda field, value: dlu.scale(field, npixels, value, method, complex),
            signature="(n,n),()->(m,m)",
        )
        field = scale(self.field, ratio)
        spacing = np.broadcast_to(pixel_scale / self.spec.scale, (2,))
        n = (int(npixels),) * 2
        return self.set_field(field).set(spec=self.spec.set(n=n, d=spacing))

    def interpolate(
        self,
        transformation: CoordTransform,
        method: str = "linear",
        complex: bool = True,
        fill: float = 0.0,
    ) -> BaseSpatial:
        """Interpolate through a coordinate transformation.

        ``complex`` has no effect when the stored sampled array is real.
        """
        if not isinstance(transformation, CoordTransform):
            raise TypeError("transformation must be a CoordTransform.")
        knots = self.coordinates
        samples = transformation(knots)
        interpolate = np.vectorize(
            lambda field: dlu.interp(field, knots, samples, method, fill, complex),
            signature="(n,m)->(n,m)",
        )
        return self.set_field(interpolate(self.field))

    def rotate(
        self,
        angle: float | Array,
        method: str = "linear",
        complex: bool = True,
    ) -> BaseSpatial:
        """Rotate the sampled array clockwise through interpolation.

        ``complex`` has no effect when the stored sampled array is real.
        """
        rotate = np.vectorize(
            lambda field, value: dlu.rotate(field, value, method, complex),
            signature="(n,n),()->(n,n)",
        )
        return self.set_field(rotate(self.field, angle))

    def resize(self, npixels: int) -> BaseSpatial:
        """Resize spatial axes by centered zero-padding or cropping."""
        fill = 0j if np.iscomplexobj(self.field) else 0.0
        field = dlu.resize(self.field, npixels, fill)
        n = (int(npixels),) * 2
        return self.set_field(field).set(spec=self.spec.set(n=n))

    def downsample(self, n: int, mean: bool | None = None) -> BaseSpatial:
        """Downsample spatial axes and update their sampling."""
        if mean is None:
            mean = bool(np.iscomplexobj(self.field))
        field = dlu.downsample(self.field, n, mean)
        size = tuple(value // n for value in self.n)
        spec = self.spec.set(n=size, d=self.d * n)
        return self.set_field(field).set(spec=spec)

    def flip(self, axis: tuple[int, ...] | int) -> BaseSpatial:
        """Flip the sampled array about one or more array axes."""
        return self.set_field(np.flip(self.field, axis))

    def __add__(self, other) -> BaseSpatial:
        return self._magic_unified_op(other, "add")

    def __sub__(self, other) -> BaseSpatial:
        return self._magic_unified_op(other, "subtract")

    def __mul__(self, other) -> BaseSpatial:
        return self._magic_unified_op(other, "multiply")

    def __truediv__(self, other) -> BaseSpatial:
        return self._magic_unified_op(other, "divide")

    def __iadd__(self, other) -> BaseSpatial:
        return self.__add__(other)

    def __isub__(self, other) -> BaseSpatial:
        return self.__sub__(other)

    def __imul__(self, other) -> BaseSpatial:
        return self.__mul__(other)

    def __itruediv__(self, other) -> BaseSpatial:
        return self.__truediv__(other)


class Wavefront(BaseSpatial):
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

    @property
    def _field_name(self) -> str:
        return "phasor"

    def __init__(
        self: Wavefront,
        wavelength: float | Array,
        spec: CoordSpec,
        phasor: Array | None = None,
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
        spec = spec.broadcast(2)
        self.wavelength = np.asarray(wavelength, float)
        if phasor is None:
            if spec.n is None:
                raise ValueError("spec.n is required when phasor is not provided.")
            if spec.ndim != 2:
                raise ValueError("Wavefront requires a two-dimensional CoordSpec.")
            shape = self.wavelength.shape + spec.shape
            self.phasor = np.ones(shape, dtype=complex) / prod(spec.n)
        else:
            phasor = np.asarray(phasor, complex)
            if phasor.ndim < 2:
                raise ValueError("phasor must have at least two spatial dimensions.")
            inferred_n = np.asarray(phasor.shape[-2:][::-1], int)
            if spec.n is None:
                spec = spec.set(n=inferred_n)
            elif tuple(int(value) for value in spec.n) != tuple(inferred_n):
                raise ValueError("phasor spatial shape must match spec.n.")
            if spec.ndim != 2:
                raise ValueError("Wavefront requires a two-dimensional CoordSpec.")
            if phasor.ndim == 2 and self.wavelength.ndim > 0:
                phasor = phasor * np.ones(self.wavelength.shape + (1, 1))
            self.phasor = phasor
        BaseSpatial.__init__(self, spec)

    @classmethod
    def from_phasor(
        cls,
        phasor: Array[complex],
        wavelength: float | Array,
        spec: CoordSpec,
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
        return cls(wavelength=wavelength, spec=spec, phasor=phasor)

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
        return 2 * np.pi / np.asarray(self.wavelength)

    @property
    def batch_ndim(self: Wavefront) -> int:
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

        # get_axis = lambda array: 0 if array.ndim > 0 else None
        return self.set(
            phasor=0 if self.batch_ndim > 0 else None,
            wavelength=0,
            spec=None,
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
        chromatic_ndim = np.asarray(self.wavelength).ndim
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

        Metadata lives on the physical vector axes counted by `self.batch_ndim`. If a
        subclass stores extra axes between those vector axes and the final spatial
        axes, append singleton dimensions so NumPy's right-aligned broadcasting keeps
        metadata on the physical axes.
        """
        array = np.asarray(array)
        extra_ndim = self.phasor.ndim - self.batch_ndim - 2
        if array.ndim == self.batch_ndim:
            return array.reshape(array.shape + (1,) * extra_ndim)
        return array

    def _from_vec(self: Wavefront, array: Array) -> Array:
        """
        Remove redundant non-physical axes from metadata returned by `np.vectorize`.
        """
        array = np.asarray(array)
        extra_ndim = self.phasor.ndim - self.batch_ndim - 2
        if extra_ndim and array.ndim == self.batch_ndim + extra_ndim:
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
        return self.add_opd(dlu.tilt_opd(self.coordinates, angles, unit))

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

    def interpolate(
        self: Wavefront,
        transformation: CoordTransform,
        method: str = "linear",
        complex: bool = True,
        fill: float = 0.0,
    ) -> Wavefront:
        """Interpolate through a coordinate transformation.

        Leading phasor dimensions, including wavelength and Jones matrix axes, are
        vectorised over directly.

        Parameters
        ----------
        transformation : CoordTransform
            Transformation applied to the wavefront sampling coordinates.
        method : str = "linear"
            Interpolation method passed to ``interpax``.
        complex : bool = True
            If True, interpolate the real and imaginary components. If False,
            interpolate the amplitude and phase components.
        fill : float = 0.0
            Value used when sampling outside the input grid.

        Returns
        -------
        wavefront : Wavefront
            The interpolated wavefront.
        """
        if not isinstance(transformation, CoordTransform):
            raise TypeError("transformation must be a CoordTransform.")
        knot_coords = self.coordinates
        transform = np.vectorize(
            transformation,
            signature="(c,n,n)->(c,n,n)",
        )
        sample_coords = transform(knot_coords)

        # Per-wavelength coordinate grids need singleton axes inserted for intrinsic
        # leading dimensions such as the Jones matrix axes of PolarisedWavefront.
        chromatic_ndim = self.wavelength.ndim
        extra_ndim = self.phasor.ndim - chromatic_ndim - 2
        if knot_coords.ndim == chromatic_ndim + 3:
            shape = (
                knot_coords.shape[:chromatic_ndim]
                + (1,) * extra_ndim
                + knot_coords.shape[-3:]
            )
            knot_coords = knot_coords.reshape(shape)
            sample_coords = sample_coords.reshape(shape)

        interp = np.vectorize(
            lambda phasor, knots, samples: dlu.interp(
                phasor, knots, samples, method, fill, complex
            ),
            signature="(n,n),(c,n,n),(c,m,m)->(m,m)",
        )
        return self.set(phasor=interp(self.phasor, knot_coords, sample_coords))

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
        output_spacing = self._from_vec(pixel_scale) / self.spec.scale
        output_center = self._from_vec(center) / self.spec.scale
        spec = self.spec.set(d=output_spacing, c=output_center)
        return self.set(phasor=phasor, spec=spec)

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
        spec = self.spec.set(
            n=(int(npixels),) * 2,
            d=np.broadcast_to(pixel_scale / self.spec.scale, (2,)),
        )
        return self.set(phasor=phasor, spec=spec)

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
        if spec_out.n is None or spec_out.d is None:
            raise ValueError("spec_out.n and spec_out.d must be specified.")
        if not np.all(spec_out.n == spec_out.n[0]):
            raise ValueError("Legacy MFT propagation requires a square output spec.")
        npixels_out = int(spec_out.n[0])
        fn = np.vectorize(
            lambda phasor, wavelength, pixel_scale_in, pixel_scale_out: dlu.MFT(
                phasor,
                wavelength,
                pixel_scale_in,
                npixels_out,
                pixel_scale_out,
                focal_length,
                inverse=inverse,
            ),
            signature="(n,n),(),(),()->(m,m)",
        )

        pixel_scale = np.asarray(spec_out.d * spec_out.scale, float)
        phasor = fn(
            self.phasor,
            self._to_vec(self.wavelength),
            self._to_vec(self.pixel_scale),
            self._to_vec(pixel_scale),
        )
        return self.set(phasor=phasor, spec=spec_out)

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

        return self._apply_field_op(other, op)

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
        if self.spatial_shape != other.spatial_shape:
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
            if "x" in axes and axes["x"] == 2 and self.spatial_shape[-1] != 2:
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
        spec: CoordSpec,
        phasor: Array | None = None,
    ):
        if phasor is None:
            super().__init__(wavelength, spec)
            self.phasor = self._promote_phasor(self.phasor)
            return

        phasor = np.asarray(phasor, complex)
        is_jones = phasor.ndim >= 4 and phasor.shape[-4:-2] == (2, 2)
        wavelength = np.asarray(wavelength, float)
        if phasor.ndim == 2 and wavelength.ndim > 0:
            phasor = phasor * np.ones(wavelength.shape + (1, 1))
        elif is_jones and phasor.ndim == 4 and wavelength.ndim > 0:
            phasor = phasor * np.ones(wavelength.shape + (1, 1, 1, 1))
        if not is_jones:
            phasor = self._promote_phasor(phasor)
        super().__init__(wavelength, spec, phasor)

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
        spec: CoordSpec,
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
        return cls(wavelength=wavelength, spec=spec, phasor=phasor)

    @property
    def batch_ndim(self: PolarisedWavefront) -> int:
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
        return PolarisedWavefront(
            wavelength=wavefront.wavelength,
            spec=wavefront.spec,
            phasor=wavefront.phasor,
        )

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
