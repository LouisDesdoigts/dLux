"""Regularly sampled optical fields and operations."""

from __future__ import annotations

from abc import abstractmethod
from math import prod
import operator

import equinox as eqx
import jax.numpy as np
import jax.random as jr
from jax import Array, vmap
from jax.scipy.signal import convolve

import dLux.utils as dlu

from .base import Base
from .grids import BaseCoordTransform, GridSpec

__all__ = [
    "BaseField",
    "ContinuousField",
    "DiscreteField",
    "Wavefront",
    "PolarisedWavefront",
    "Intensity",
    "Image",
]

_ops = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
}


def __getattr__(name):
    """Resolve field names retained by the compatibility layer."""
    if name == "PSF":
        # Keep the legacy module lazy: compatibility imports Intensity.
        from . import compatibility

        return compatibility.PSF
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _scale_field(field, npixels, pixel_scale, method, complex):
    """Interpolate a field to new dimensions and physical sampling."""
    # Resolve the requested sampling and scale ratios
    n = dlu.as_size(npixels, 2, "npixels")
    spacing = dlu.as_axis(pixel_scale, 2, "pixel_scale") / field.grid.scale
    ratio = spacing / field.d

    # Vectorise resampling over leading field dimensions
    scale = np.vectorize(
        lambda data, value: dlu.scale(data, n, value, method, complex),
        signature="(n,m),(c)->(p,q)",
    )
    data = scale(field.field, ratio)

    # Update the sampled field and coordinate specification
    return field.set(field=data, grid=field.grid.resample(n, spacing))


def _interpolate_field(field, transformation, method, complex, fill):
    """Interpolate a field through a coordinate transformation."""
    # Validate and transform the sampled coordinate grid
    if not isinstance(transformation, BaseCoordTransform):
        raise TypeError("transformation must be a BaseCoordTransform.")

    knots = field.coordinates
    transform = np.vectorize(transformation, signature="(c,n,m)->(c,n,m)")
    samples = transform(knots)

    # Align coordinate batches before intrinsic field dimensions
    n_batch = field.field.ndim - 2
    c_batch = knots.ndim - 3
    if c_batch > n_batch:
        raise ValueError("Coordinate batch dimensions exceed field dimensions.")
    shape = knots.shape[:c_batch] + (1,) * (n_batch - c_batch) + knots.shape[-3:]
    knots, samples = knots.reshape(shape), samples.reshape(shape)

    # Vectorise interpolation over every leading field dimension
    interpolate = np.vectorize(
        lambda data, x, y: dlu.interp(data, x, y, method, fill, complex),
        signature="(n,m),(c,n,m),(c,p,q)->(p,q)",
    )
    return field.set(field=interpolate(field.field, knots, samples))


def _rotate_field(field, angle, method, complex):
    """Rotate a field clockwise through interpolation."""
    rotate = np.vectorize(
        lambda data, value: dlu.rotate(data, value, method, complex),
        signature="(n,m),()->(n,m)",
    )
    return field.set(field=rotate(field.field, angle))


class BaseField(Base):
    """Base class for regularly sampled real or complex fields."""

    grid: GridSpec

    def __getattr__(self, key):
        """Forward unknown attributes to the coordinate specification."""
        return dlu.resolve_attr(self, key, self.grid)

    @property
    @abstractmethod
    def field(self) -> Array:
        """Return the stored sampled array.

        Concrete fields define whether this is a real data array or complex phasor.
        Leading axes are vectorisation axes and the final two axes are spatial.
        """

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        """Return the final ``(ny, nx)`` sampled array shape."""
        return self.field.shape[-2:]

    @property
    def axes(self) -> tuple[Array, ...]:
        """Return SI coordinate axes using the field's realised spatial shape.

        Axes are returned in physical ``(x, y)`` order and may contain leading grid
        batch dimensions.
        """
        return self.xs

    @property
    def coordinates(self) -> Array:
        """Return SI coordinates using the field's realised spatial shape.

        The result has shape ``(..., 2, ny, nx)`` with physical ``(x, y)`` component
        order and NumPy ``(y, x)`` spatial-axis order.
        """
        return self.grid._coordinates_for(self.spatial_shape[::-1])

    @property
    def xs(self) -> tuple[Array, ...]:
        """Alias `axes` using the retained compact coordinate terminology."""
        return self.grid._xs_for(self.spatial_shape[::-1])

    @property
    def npixels(self) -> int:
        """Return ``nx`` for compatibility with square-grid APIs.

        This does not assert that ``ny == nx``; use `spatial_shape` for rectangular
        fields.
        """
        return self.field.shape[-1]

    @property
    def pixel_scale(self) -> Array:
        """Return per-axis pixel scales in canonical SI units.

        Values follow physical ``(x, y)`` order and preserve leading grid batch axes.
        """
        if self.d is None:
            raise ValueError("grid.d is not defined.")
        return self.d * self.scale

    @property
    def center(self) -> Array:
        """Return the grid centre in canonical SI units and physical-axis order."""
        return np.zeros(len(self.n)) if self.c is None else self.c * self.scale

    @property
    def diameter(self) -> Array:
        """Return the sampled field width along each physical axis.

        This aliases the grid field of view and uses the grid's declared unit rather
        than canonical SI units.
        """
        return self.fov

    def normalise(self, mode: str = "power", value: float = 1.0) -> BaseField:
        """Return a copy normalised by total sum or peak value.

        ``mode="power"`` sets ``field.sum()`` to ``value`` and ``mode="peak"``
        sets ``field.max()`` to it. Normalisation spans every array axis, including
        leading axes; use explicit arithmetic for independent batch normalisation.

        Parameters
        ----------
        mode : {"power", "peak"}
            Select total-sum or maximum-value normalisation.
        value : float
            Target total or peak in the field's current value unit.

        Returns
        -------
        field : BaseField
            Normalised immutable copy retaining the concrete type and grid.
        """
        if mode == "power":
            scale = value / self.field.sum()
        elif mode == "peak":
            scale = value / self.field.max()
        else:
            raise ValueError("mode must be 'power' or 'peak'")
        return self.set(field=self.field * scale)

    def convolve(
        self, other: Array, mode: str = "same", method: str = "auto"
    ) -> BaseField:
        """Convolve the final two spatial axes with a broadcastable kernel.

        Parameters
        ----------
        other : Array
            Kernel with final ``(ny, nx)`` axes and broadcastable leading axes.
        mode : str
            JAX convolution output mode; spatial grid size follows the result.
        method : str
            JAX convolution method such as ``"auto"``, ``"direct"``, or ``"fft"``.
        """
        # Broadcast the field and kernel batch dimensions
        other = np.asarray(other)
        batch = np.broadcast_shapes(self.field.shape[:-2], other.shape[:-2])
        field = np.broadcast_to(self.field, batch + self.field.shape[-2:])
        kernel = np.broadcast_to(other, batch + other.shape[-2:])

        # Flatten and convolve every pair of spatial arrays
        apply = lambda x, y: convolve(x, y, mode=mode, method=method)
        fields = field.reshape((-1,) + field.shape[-2:])
        kernels = kernel.reshape((-1,) + kernel.shape[-2:])
        field = vmap(apply)(fields, kernels)

        # Restore the batch dimensions and realised sampling
        field = field.reshape(batch + field.shape[-2:])
        grid = self.grid.resize(field.shape[-2:][::-1])
        return self.set(field=field, grid=grid)

    def _binary_op(self, other, op: str) -> BaseField:
        """Apply arithmetic to another compatible sampled field or array."""
        if other is None:
            return self
        if not isinstance(other, (BaseField, Array, float, int, complex)):
            raise TypeError(
                f"Unsupported type for {op}: {type(other)}. Must be an array, "
                "field, or None."
            )
        self, other = self._prepare_operand(other)
        return self.set(field=_ops[op](self.field, other))

    def _prepare_operand(self, other) -> tuple[BaseField, Array]:
        """Return the field and array used for arithmetic."""
        if isinstance(other, BaseField):
            return self, other.field
        return self, other

    def resize(self, npixels: int | tuple[int, int]) -> BaseField:
        """Resize spatial axes by centred zero-padding or cropping.

        ``npixels`` follows physical ``(x, y)`` order. Pixel scales and centre are
        retained, so changing the sample count changes the field of view.

        Returns a new field of the same concrete type; leading axes are preserved.
        """
        fill = 0j if np.iscomplexobj(self.field) else 0.0
        field = dlu.resize(self.field, npixels, fill)
        return self.set(field=field, grid=self.grid.resize(npixels))

    def downsample(
        self, n: int | tuple[int, int], mean: bool | None = None
    ) -> BaseField:
        """Downsample spatial axes by integer factors.

        ``n`` follows physical ``(x, y)`` order. Complex fields average each block by
        default; real fields sum it. Set ``mean`` explicitly to override this choice.
        Pixel scales are increased to preserve the field of view.

        Returns a new field of the same concrete type with leading axes preserved.
        """
        if mean is None:
            mean = bool(np.iscomplexobj(self.field))
        field = dlu.downsample(self.field, n, mean)
        return self.set(field=field, grid=self.grid.downsample(n))

    def flip(self, axis: tuple[int, ...] | int) -> BaseField:
        """Return a copy flipped about one or more NumPy array axes.

        ``axis`` follows array-axis indexing and may include leading batch axes. Grid
        metadata are retained unchanged.
        """
        return self.set(field=np.flip(self.field, axis))

    def __add__(self, other) -> BaseField:
        return self._binary_op(other, "add")

    def __sub__(self, other) -> BaseField:
        return self._binary_op(other, "subtract")

    def __mul__(self, other) -> BaseField:
        return self._binary_op(other, "multiply")

    def __truediv__(self, other) -> BaseField:
        return self._binary_op(other, "divide")

    def __iadd__(self, other) -> BaseField:
        return self._binary_op(other, "add")

    def __isub__(self, other) -> BaseField:
        return self._binary_op(other, "subtract")

    def __imul__(self, other) -> BaseField:
        return self._binary_op(other, "multiply")

    def __itruediv__(self, other) -> BaseField:
        return self._binary_op(other, "divide")


class ContinuousField(BaseField):
    """Base class for fields representing a continuously sampled quantity."""

    grid: GridSpec

    def scale_to(
        self,
        npixels: int | tuple[int, int],
        pixel_scale: float | Array,
        method: str = "linear",
        complex: bool = True,
    ) -> ContinuousField:
        """Interpolate to a size and physical per-axis pixel scale.

        ``complex`` selects Cartesian or polar decomposition for complex fields and
        has no effect on real fields such as intensities.

        Parameters
        ----------
        npixels : int or tuple[int, int]
            Output sizes in physical ``(x, y)`` order.
        pixel_scale : float or Array
            Output physical pixel scales in the field grid unit.
        method : str
            Interpolation method.
        complex : bool
            Select complex-aware interpolation for complex fields.
        """
        return _scale_field(self, npixels, pixel_scale, method, complex)

    def interpolate(
        self,
        transformation: BaseCoordTransform,
        method: str = "linear",
        complex: bool = True,
        fill: float = 0.0,
    ) -> ContinuousField:
        """Interpolate through a coordinate transformation.

        Parameters
        ----------
        transformation : BaseCoordTransform
            Map from output coordinates into the sampled input frame.
        method : str
            Interpolation method.
        complex : bool
            Select complex-aware interpolation for complex fields.
        fill : float
            Value outside the sampled support.
        """
        return _interpolate_field(self, transformation, method, complex, fill)

    def rotate(
        self, angle: float | Array, method: str = "linear", complex: bool = True
    ) -> ContinuousField:
        """Rotate the sampled array clockwise through interpolation.

        ``complex`` has no effect when the stored sampled array is real.

        Parameters
        ----------
        angle : float or Array
            Clockwise rotation in radians.
        method : str
            Interpolation method.
        complex : bool
            Select complex-aware interpolation for complex fields.
        """
        return _rotate_field(self, angle, method, complex)


class DiscreteField(BaseField):
    """Base class for real-valued fields sampled on a discrete grid."""


class Wavefront(ContinuousField):
    """Represent a scalar complex wavefront on a regular coordinate grid.

    The final two phasor axes are spatial. Any preceding axes are vectorisation
    axes. Vector-valued wavelengths generate matching leading phasor dimensions.

    Parameters
    ----------
    wavelength : float or Array, meters
        Scalar wavelength or array of wavelengths.
    grid : GridSpec
        Two-dimensional spatial sampling specification.
    phasor : Array or None
        Complex field with shape ``(..., ny, nx)``. When omitted, a uniform
        unit-power field is generated from ``grid``.
    """

    grid: GridSpec
    phasor: Array[complex]
    wavelength: Array

    def __init__(
        self: Wavefront,
        wavelength: float | Array,
        grid: GridSpec,
        phasor: Array | None = None,
    ):
        """Initialise a complex wavefront sampled on a physical grid.

        Parameters
        ----------
        wavelength : float or Array, metres
            Scalar wavelength or array whose shape defines leading wavelength axes.
        grid : GridSpec
            Two-dimensional spatial grid. It must define ``n`` when ``phasor`` is
            omitted; otherwise its spatial shape is matched to ``phasor``.
        phasor : Array or None
            Complex field with shape ``(..., ny, nx)``. A two-dimensional field is
            broadcast over array-valued wavelengths. If omitted, a uniform
            unit-power field is constructed.
        """
        # Validate the grid and resolve the input wavelengths
        if not isinstance(grid, GridSpec):
            raise TypeError("grid must be a GridSpec.")
        self.wavelength = dlu.to_value(wavelength)

        # Initialise a uniform field when no phasor is supplied
        if phasor is None:
            grid = grid.broadcast(2)
            if grid.n is None:
                raise ValueError("grid.n is required when phasor is not provided.")
            shape = self.wavelength.shape + grid.shape
            self.phasor = np.ones(shape, dtype=complex) / prod(grid.n)

        # Validate and align an explicit phasor with wavelengths
        else:
            phasor = dlu.to_value(phasor, complex)
            if phasor.ndim < 2:
                raise ValueError("phasor must have at least two spatial dimensions.")
            grid = grid.match_shape(phasor.shape[-2:])
            if phasor.ndim == 2 and self.wavelength.ndim > 0:
                phasor = phasor * np.ones(self.wavelength.shape + (1, 1))
            self.phasor = phasor

        # Store the realised spatial specification
        self.grid = grid

    @property
    def field(self) -> Array:
        """Return the complex phasor with final ``(ny, nx)`` spatial axes."""
        return self.phasor

    @classmethod
    def from_phasor(
        cls,
        phasor: Array[complex],
        wavelength: float | Array,
        grid: GridSpec,
    ) -> Wavefront:
        """Create a wavefront from an existing phasor array.

        Deprecated compatibility constructor. Pass ``phasor`` directly to
        `Wavefront` in new code.

        Parameters
        ----------
        phasor : Array[complex]
            The complex electric field array. The final two axes are spatial; leading
            axes are vectorisation axes. If a 2D phasor is passed with vector
            wavelengths, it is broadcast over the wavelength axes.
        wavelength : float or Array, meters
            The wavelength of the wavefront. Vector-valued wavelengths define a
            chromatic wavefront.
        grid : GridSpec
            Sampling and coordinate definition for the wavefront.

        Returns
        -------
        wavefront : Wavefront
            A new Wavefront object with the specified phasor.
        """
        from .compatibility import warn_deprecated

        migration = "`dl.Wavefront.from_phasor(p, w, g)` -> `dl.Wavefront(w, g, p)`"
        warn_deprecated("Wavefront.from_phasor", "Wavefront", migration, stacklevel=3)
        return cls(wavelength, grid, phasor)

    @property
    def real(self: Wavefront) -> Array:
        """Return the real phasor component with the original array shape."""
        return self.phasor.real

    @property
    def imaginary(self: Wavefront) -> Array:
        """Return the imaginary phasor component with the original array shape."""
        return self.phasor.imag

    @property
    def amplitude(self: Wavefront) -> Array:
        """Return the non-negative modulus of the complex phasor."""
        return np.abs(self.phasor)

    @property
    def phase(self: Wavefront) -> Array:
        """Return the wrapped phasor phase in radians over ``[-π, π]``."""
        return np.angle(self.phasor)

    @property
    def complex(self: Wavefront) -> Array:
        """Return real and imaginary components stacked on a new leading axis.

        The result has shape ``(2, *phasor.shape)``.
        """
        return np.stack([self.phasor.real, self.phasor.imag], axis=0)

    @property
    def polar(self: Wavefront) -> Array:
        """Return amplitude and radian phase stacked on a new leading axis.

        The result has shape ``(2, *phasor.shape)``.
        """
        return np.stack([self.amplitude, self.phase], axis=0)

    @property
    def intensity(self: Wavefront) -> Array:
        """Return ``abs(phasor)**2`` with all phasor axes preserved."""
        return np.abs(self.phasor) ** 2

    @property
    def psf(self: Wavefront) -> Array:
        """Alias `intensity` using retained point-spread-function terminology."""
        return self.intensity

    def to_intensity(self, stokes=None) -> Intensity:
        """Convert the wavefront into deterministic sampled intensity.

        ``stokes`` is an optional input Stokes vector with final component axis of
        length four. For scalar wavefronts only its total-intensity component is used;
        polarised wavefronts propagate the complete vector. The returned `Intensity`
        retains the wavefront grid and leading output axes.
        """
        return Intensity(self.intensity_from_stokes(stokes), self.grid)

    @property
    def wavenumber(self: Wavefront) -> Array:
        """Return ``2π / wavelength`` in inverse metres.

        The result has the same shape as ``wavelength`` and no spatial axes.
        """
        return 2 * np.pi / np.asarray(self.wavelength)

    @property
    def batch_ndim(self: Wavefront) -> int:
        """Return the number of phasor axes preceding the final spatial axes."""
        return self.phasor.ndim - 2

    @property
    def is_chromatic(self: Wavefront) -> bool:
        """Return whether ``wavelength`` has one or more array dimensions."""
        return self.wavelength.ndim > 0

    @property
    def is_polarised(self: Wavefront) -> bool:
        """Return ``False`` for a scalar wavefront without Jones-matrix axes."""
        return False

    @property
    def _mapped_axis(self: Wavefront) -> tuple[int | None, ...] | None:
        """Returns the input-axis specification for mapping over the leading batch axis.

        Vectorised wavelength and sampling metadata are mapped when they share the
        leading phasor axis; scalar metadata is shared. Intrinsic axes such as the
        Jones axes of a `PolarisedWavefront` are never mapped.

        Returns
        -------
        mapped_axis : tuple or None
            Mapping axes for the phasor, wavelength, spacing, and centre. Returns
            ``None`` for an unbatched wavefront.
        """
        if self.batch_ndim == 0:
            return None

        size = self.phasor.shape[0]
        axis = lambda x, ndim=0: (
            0 if x is not None and x.ndim > ndim and x.shape[0] == size else None
        )
        return 0, axis(self.wavelength), axis(self.grid.d, 1), axis(self.grid.c, 1)

    @property
    def power(self: Wavefront) -> Array:
        """Return intensity summed independently over the final spatial axes.

        All wavelength, batch, and Jones-derived leading axes are preserved.
        """
        return np.sum(self.intensity, axis=(-2, -1))

    def _to_phasor_shape(self: Wavefront, array: Array) -> Array:
        """Reshape scalar or spatial arrays to broadcast against the phasor, preserving
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

    def add_phase(self: Wavefront, phase: float | Array) -> Wavefront:
        """Return a copy with a scalar, spatial, or vectorised phase applied.

        ``phase`` is measured in radians and must broadcast against the wavelength,
        Jones, and final spatial phasor axes. ``None`` leaves the wavefront unchanged.
        """
        if phase is None:
            return self
        return self.multiply("phasor", np.exp(1j * self._to_phasor_shape(phase)))

    def add_opd(self: Wavefront, opd: float | Array) -> Wavefront:
        """Return a copy with an optical-path difference applied.

        ``opd`` is measured in metres and may be scalar, spatial, or wavelength-
        vectorised. It is converted to phase using each wavefront wavelength.
        ``None`` leaves the wavefront unchanged.
        """
        if opd is None:
            return self
        return self.add_phase(self.wavenumber[..., None, None] * np.asarray(opd))

    def tilt(self: Wavefront, angles: Array, unit: str = "rad") -> Wavefront:
        """Return a copy with an angular source tilt applied.

        ``angles`` has final shape ``(2,)`` in physical ``(x, y)`` order and uses
        ``unit`` (radians by default). Leading angle axes follow utility broadcasting
        conventions.
        """
        return self.add_opd(dlu.tilt_opd(self.coordinates, angles, unit))

    def normalise(
        self: Wavefront, mode: str = "power", value: float = 1.0
    ) -> Wavefront:
        """Normalise the wavefront.

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
            scale = np.sqrt(value / self.intensity.max(axis=(-2, -1)))
        else:
            raise ValueError("mode must be 'power' or 'peak'")
        return self.set(phasor=self.phasor * self._to_phasor_shape(scale))

    def _binary_op(
        self: Wavefront, other: Wavefront | Array | None, op: str
    ) -> Wavefront:
        """Apply arithmetic after aligning wavelength and Jones axes."""
        if op == "divide" and isinstance(other, Wavefront):
            raise TypeError(
                "dLux has detected an attempt to perform dark optics. Your wavefront "
                "privileges have been temporarily suspended and the authorities have "
                "been notified."
            )
        return super()._binary_op(other, op)

    def _prepare_operand(
        self: Wavefront, other: Wavefront | Array | float | int | complex
    ) -> tuple[Wavefront, Array | float | int | complex]:
        """Align wavelength and Jones axes for wavefront arithmetic."""
        if isinstance(other, Wavefront):
            return self._prepare_wavefront_operand(other)
        if isinstance(other, Array):
            return self._prepare_array_operand(other)
        if isinstance(other, BaseField):
            raise TypeError("Wavefront arithmetic requires another Wavefront or array.")
        return self, other

    def _promote_for_arithmetic(self: Wavefront) -> PolarisedWavefront:
        """Promotes the base type while retaining scalar Jones broadcasting."""
        phasor = self.phasor[..., None, None, :, :]
        return PolarisedWavefront.from_wavefront(self).set(phasor=phasor)

    def _prepare_wavefront_operand(
        self: Wavefront, other: Wavefront
    ) -> tuple[Wavefront, Array]:
        """Prepares an operand for elementwise wavefront arithmetic.

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

        self_polarised = self.is_polarised
        other_polarised = other.is_polarised

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
        """Align standard field modulation and otherwise use JAX broadcasting."""
        other = self._to_phasor_shape(other)
        try:
            shape = np.broadcast_shapes(self.phasor.shape, other.shape)
        except ValueError as error:
            raise ValueError(
                f"Array shape {other.shape} cannot broadcast to phasor shape "
                f"{self.phasor.shape}."
            ) from error
        if shape != self.phasor.shape:
            raise ValueError("Array operands cannot add dimensions to a Wavefront.")
        return self, other

    def apply_jones(self, jones):
        """Promote the field and apply a Jones matrix.

        ``jones`` follows the utility convention ``(2, 2, ...)``. The returned
        `PolarisedWavefront` retains wavelength and grid metadata.
        """
        return PolarisedWavefront.from_wavefront(self).apply_jones(jones)

    def intensity_from_stokes(self, stokes: Array | None = None) -> Array:
        """Evaluate intensity for an optional input Stokes vector.

        For a scalar wavefront, only ``stokes[0]`` contributes. With no vector the
        unit-input intensity is returned. Spatial axes remain final.
        """
        if stokes is None:
            return self.intensity

        # For a polarisation-insensitive system, only total input intensity matters.
        return stokes[0] * self.intensity

    def psf_from_stokes(self, stokes: Array | None = None) -> Array:
        """Alias `intensity_from_stokes` using retained PSF terminology.

        ``stokes`` and the returned array follow the same contract as
        `intensity_from_stokes`.
        """
        return self.intensity_from_stokes(stokes)


class PolarisedWavefront(Wavefront):
    """Represent a partially polarised wavefront using Jones calculus.

    The internal representation uses Jones calculus in the general case,
    tracking a 2x2 complex coherence matrix. Phasors have shape
    ``(..., 2, 2, ny, nx)`` with leading vectorisation dimensions, Jones axes,
    and final spatial axes.
    """

    grid: GridSpec
    phasor: Array[complex]
    wavelength: Array

    def __init__(
        self: Wavefront,
        wavelength: float | Array,
        grid: GridSpec,
        phasor: Array | None = None,
    ):
        """Initialise a scalar or Jones-matrix wavefront.

        Parameters
        ----------
        wavelength : float or Array, metres
            Scalar wavelength or array whose shape defines leading wavelength axes.
        grid : GridSpec
            Two-dimensional physical spatial grid.
        phasor : Array or None
            Scalar field ``(..., ny, nx)`` or Jones field
            ``(..., 2, 2, ny, nx)``. Scalar inputs and generated uniform fields are
            promoted to an unpolarised Jones representation.
        """
        if phasor is None:
            super().__init__(wavelength, grid)
            self.phasor = self._promote_phasor(self.phasor)
            return

        phasor = dlu.to_value(phasor, complex)
        is_jones = phasor.ndim >= 4 and phasor.shape[-4:-2] == (2, 2)
        wavelength = dlu.to_value(wavelength)
        if phasor.ndim == 2 and wavelength.ndim > 0:
            phasor = phasor * np.ones(wavelength.shape + (1, 1))
        elif is_jones and phasor.ndim == 4 and wavelength.ndim > 0:
            phasor = phasor * np.ones(wavelength.shape + (1, 1, 1, 1))
        if not is_jones:
            phasor = self._promote_phasor(phasor)
        super().__init__(wavelength, grid, phasor)

    @property
    def is_polarised(self: PolarisedWavefront) -> bool:
        """Return ``True`` for a wavefront carrying explicit Jones-matrix axes."""
        return True

    @staticmethod
    def _promote_phasor(phasor: Array) -> Array:
        """Promote a scalar phasor into an unpolarised Jones phasor.

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
        grid: GridSpec,
    ) -> PolarisedWavefront:
        """Create a polarised wavefront from a regular or Jones phasor.

        Deprecated compatibility constructor. Pass ``phasor`` directly to
        `PolarisedWavefront` in new code.

        Parameters
        ----------
        phasor : Array[complex]
            Regular phasor with shape `(..., n, n)` or Jones phasor with shape
            `(..., 2, 2, n, n)`.
        wavelength : float or Array, meters
            The wavelength of the wavefront. If a 2D phasor is passed with vector
            wavelengths, it is broadcast over the wavelength axes.
        grid : GridSpec
            Sampling and coordinate definition for the wavefront.

        Returns
        -------
        wavefront : PolarisedWavefront
            A new polarised wavefront with phasor shape `(..., 2, 2, n, n)`.
        """
        from .compatibility import warn_deprecated

        migration = (
            "`dl.PolarisedWavefront.from_phasor(p, w, g)` -> "
            "`dl.PolarisedWavefront(w, g, p)`"
        )
        warn_deprecated(
            "PolarisedWavefront.from_phasor",
            "PolarisedWavefront",
            migration,
            stacklevel=3,
        )
        return cls(wavelength, grid, phasor)

    @property
    def batch_ndim(self: PolarisedWavefront) -> int:
        """Return leading dimensions excluding two Jones and two spatial axes."""
        return self.phasor.ndim - 4

    @staticmethod
    def from_wavefront(wavefront: Wavefront) -> PolarisedWavefront:
        """Promotes a regular Wavefront to a PolarisedWavefront.

        Parameters
        ----------
        wavefront : Wavefront
            The input wavefront to promote.

        Returns
        -------
        polarised_wavefront : PolarisedWavefront
            A new PolarisedWavefront with the same wavelength, pixel scale, and centre
            as the input wavefront, and the phasor promoted
        """
        return PolarisedWavefront(
            wavelength=wavefront.wavelength,
            grid=wavefront.grid,
            phasor=PolarisedWavefront._promote_phasor(wavefront.phasor),
        )

    @property
    def intensity(self: Wavefront) -> Array:
        """Return output intensity for an unpolarised unit input.

        Jones axes are consumed and wavelength, batch, and spatial axes are retained.
        """
        return self.intensity_from_stokes()

    @property
    def psf(self: Wavefront) -> Array:
        """Alias `intensity` using retained point-spread-function terminology."""
        return self.intensity

    def intensity_from_stokes(
        self: Wavefront, input_stokes: Array | None = None
    ) -> Array:
        """Evaluate output intensity for an optional input Stokes vector.

        ``input_stokes`` has final component axis of length four. When omitted, an
        unpolarised unit input is used. The result preserves leading wavelength and
        batch axes followed by the final spatial axes.
        """
        if input_stokes is None:
            return 0.5 * np.sum(np.abs(self.phasor) ** 2, axis=(-4, -3))
        stokes = self.stokes(input_stokes)
        return stokes[..., 0, :, :]

    def psf_from_stokes(self: Wavefront, input_stokes: Array | None = None) -> Array:
        """Alias `intensity_from_stokes` using retained PSF terminology.

        ``input_stokes`` and the returned array follow the same contract as
        `intensity_from_stokes`.
        """
        return self.intensity_from_stokes(input_stokes)

    def stokes(self: Wavefront, input_stokes: Array | None = None) -> Array:
        """Return output Stokes parameters for an optional input state.

        The polarised wavefront stores phasors as `(..., 2, 2, n, n)`, while the
        polarisation utilities operate on `(2, 2, ...)`. We move the Jones axes to the
        front, call the utility function, then move the Stokes axis back behind any
        leading wavefront dimensions. ``input_stokes`` has final component axis four;
        when omitted an unpolarised unit input is used. The result has shape
        ``(..., 4, ny, nx)``.
        """
        phasor = np.moveaxis(self.phasor, (-4, -3), (0, 1))
        stokes = dlu.jones_to_stokes(phasor, input_stokes)
        return np.moveaxis(stokes, 0, -3)

    def apply_jones(self, jones):
        """Apply a Jones matrix to the polarised wavefront.

        The Jones matrix follows the utility convention `(2, 2, ...)`. The wavefront
        Jones axes are moved to the front before applying the utility function, then
        moved back to preserve `(..., 2, 2, n, n)` ordering. ``jones`` follows the
        utility convention ``(2, 2, ...)`` and the returned wavefront is a new object.
        """
        phasor = np.moveaxis(self.phasor, (-4, -3), (0, 1))
        phasor = dlu.apply_jones(jones, phasor)
        return self.set(phasor=np.moveaxis(phasor, (0, 1), (-4, -3)))


class Intensity(DiscreteField):
    """A deterministic sampled optical intensity or expected detector signal.

    The stored values may represent a normalised photon distribution, expected
    counts, or another deterministic detector-domain signal. Their interpretation
    is set by the preceding model rather than enforced by this container.

    Parameters
    ----------
    data : Array
        Real sampled values with shape ``(..., ny, nx)``. Leading axes are preserved
        as vectorisation axes.
    grid : GridSpec
        Spatial sampling whose physical ``(x, y)`` sizes must match the final array
        axes in reversed ``(nx, ny)`` order.

    Examples
    --------
    Convert a propagated wavefront, then explicitly begin detector-image modelling:

    ```python
    intensity = wavefront.to_intensity()
    image = intensity.to_image(read_noise=3.0)
    ```
    """

    data: Array
    grid: GridSpec

    def __init__(self: Intensity, data: Array, grid: GridSpec):
        """Initialise deterministic intensity data on a physical grid.

        Parameters
        ----------
        data : Array
            Real values with shape ``(..., ny, nx)``. Leading axes are retained.
        grid : GridSpec
            Spatial grid matched against the final ``(ny, nx)`` array axes.
        """
        self.data = dlu.to_value(data)
        if self.data.ndim < 2:
            raise ValueError("data must have at least two spatial dimensions.")
        if not isinstance(grid, GridSpec):
            raise TypeError("grid must be a GridSpec.")
        self.grid = grid.match_shape(self.data.shape[-2:])

    @property
    def field(self) -> Array:
        """Return deterministic intensity data with final ``(ny, nx)`` axes."""
        return self.data

    @property
    def batch_ndim(self: Intensity) -> int:
        """Return the number of intensity axes preceding the spatial axes."""
        return self.data.ndim - 2

    @classmethod
    def from_wavefront(cls, wavefront, stokes=None) -> Intensity:
        """Construct deterministic intensity from a wavefront.

        ``stokes`` optionally defines the input polarisation with final component
        axis four. The wavefront grid is retained and the intensity has shape
        ``(..., ny, nx)``.
        """
        if not isinstance(wavefront, Wavefront):
            raise TypeError("wavefront must be a Wavefront.")
        return cls(wavefront.psf_from_stokes(stokes), wavefront.grid)

    def scale_to(self, npixels, pixel_scale, method="linear", complex=True):
        """Interpolate to new spatial sizes and physical pixel scales.

        Parameters
        ----------
        npixels : int or tuple[int, int]
            Output sizes in physical ``(x, y)`` order.
        pixel_scale : float or Array
            Output scales in the intensity grid unit.
        method : str
            Interpolation method.
        complex : bool
            Accepted for field API consistency; real intensity data remain real.
        """
        return _scale_field(self, npixels, pixel_scale, method, complex)

    def interpolate(self, transformation, method="linear", complex=True, fill=0.0):
        """Interpolate through a coordinate transformation.

        Parameters
        ----------
        transformation : BaseCoordTransform
            Map from output coordinates into the sampled input frame.
        method : str
            Interpolation method.
        complex : bool
            Accepted for field API consistency; real intensity data remain real.
        fill : float
            Value outside the sampled support.
        """
        return _interpolate_field(self, transformation, method, complex, fill)

    def rotate(self, angle, method="linear", complex=True):
        """Rotate the sampled intensity clockwise through interpolation.

        Parameters
        ----------
        angle : float or Array, radians
            Clockwise rotation angle.
        method : str
            Interpolation method.
        complex : bool
            Accepted for field API consistency; real intensity data remain real.
        """
        return _rotate_field(self, angle, method, complex)

    def to_image(self, std=None, read_noise=0.0) -> Image:
        """Create a realised-image container from this deterministic intensity.

        ``std`` is optional standard deviation broadcastable to the intensity shape;
        ``read_noise`` is the Gaussian standard deviation used by later simulation.
        No noise is generated by this conversion.
        """
        return Image(self, std=std, read_noise=read_noise)


class Image(DiscreteField):
    """A discrete detector image and its uncertainty metadata.

    Parameters
    ----------
    data : Array or Intensity
        Detector data, or an Intensity to convert directly into an image.
    grid : GridSpec or None
        Coordinate specification for array inputs. This is inherited from an
        Intensity when one is supplied.
    std : Array or None
        Known standard deviation of the observed data.
    read_noise : float or Array
        Gaussian read-noise standard deviation associated with the image.

    Notes
    -----
    ``Image`` stores realised or simulated data. Deterministic optical and detector
    models should retain ``Intensity`` until an image is explicitly constructed.
    Uncertainty is optional because deterministic detector transformations generally
    do not define a complete propagation rule for it.
    """

    data: Array
    grid: GridSpec
    std: Array | None
    read_noise: Array

    def __init__(
        self,
        data: Array | Intensity,
        grid: GridSpec | None = None,
        std: Array | None = None,
        read_noise: float | Array = 0.0,
    ):
        """Initialise realised detector data and optional uncertainty.

        Parameters
        ----------
        data : Array or Intensity
            Detector values with shape ``(..., ny, nx)``, or an intensity whose data
            and grid are inherited.
        grid : GridSpec or None
            Spatial grid required for array input and forbidden for `Intensity` input.
        std : Array or None
            Standard deviation broadcast to the detector-data shape.
        read_noise : float or Array
            Gaussian read-noise standard deviation retained for simulation.
        """
        # Unpack a sampled intensity directly into detector image data
        if isinstance(data, Intensity):
            if grid is not None:
                raise ValueError("grid must not be supplied with an Intensity.")
            data, grid = data.data, data.grid

        # Validate and store the detector image
        data = dlu.to_value(data)
        if data.ndim < 2:
            raise ValueError("data must have at least two spatial dimensions.")
        if not isinstance(grid, GridSpec):
            raise TypeError("grid must be a GridSpec.")

        # Resolve and store optional uncertainty information
        grid = grid.match_shape(data.shape[-2:])
        if std is not None:
            std = np.broadcast_to(dlu.to_value(std), data.shape)
        self.data = data
        self.std = std
        self.read_noise = dlu.to_value(read_noise)
        self.grid = grid

    @property
    def field(self) -> Array:
        """Return realised detector data with final ``(ny, nx)`` spatial axes."""
        return self.data

    @classmethod
    def from_intensity(cls, intensity, std=None, read_noise=0.0) -> Image:
        """Construct an image from deterministic sampled intensity.

        Parameters
        ----------
        intensity : Intensity
            Deterministic data and grid to retain.
        std : Array or None
            Optional standard deviation broadcast to the data shape.
        read_noise : float or Array
            Gaussian read-noise standard deviation retained for simulation.
        """
        if not isinstance(intensity, Intensity):
            raise TypeError("intensity must be an Intensity.")
        return cls(intensity, std=std, read_noise=read_noise)

    @property
    def variance(self) -> Array | None:
        """Return ``std**2``, or ``None`` when no uncertainty is stored."""
        return None if self.std is None else self.std**2

    @property
    def fourier_transform(self) -> Array:
        """Return the centred Fourier transform over the final two spatial axes.

        Leading image axes are transformed independently.
        """
        transformed = np.fft.fft2(self.field, axes=(-2, -1))
        return np.fft.fftshift(transformed, axes=(-2, -1))

    @property
    def amplitude_spectrum(self) -> Array:
        """Return the non-negative modulus of `fourier_transform`."""
        return np.abs(self.fourier_transform)

    @property
    def power_spectrum(self) -> Array:
        """Return ``amplitude_spectrum**2`` with image leading axes preserved."""
        return self.amplitude_spectrum**2

    def add_poisson_noise(self, key: Array) -> Image:
        """Return a Poisson realisation and its expected uncertainty.

        ``key`` is a JAX random key. Stored data are interpreted as non-negative
        expected counts, and any existing variance is added to the Poisson variance.
        """
        expectation = self.field
        data = jr.poisson(key, expectation).astype(self.field.dtype)
        variance = expectation
        if self.std is not None:
            variance = variance + self.std**2
        return self.set(field=data, std=np.sqrt(variance))

    def add_read_noise(self, key: Array, sigma: float | Array) -> Image:
        """Return a Gaussian read-noise realisation with updated uncertainty.

        ``key`` is a JAX random key and ``sigma`` is a standard deviation in the same
        units as the image data. It must broadcast against the data shape. Existing
        variance and read-noise metadata are combined in quadrature.
        """
        sigma = np.asarray(sigma, dtype=self.field.dtype)
        noise = jr.normal(key, self.field.shape, self.field.dtype) * sigma
        variance = sigma**2
        if self.std is not None:
            variance = variance + self.std**2
        read_noise = np.sqrt(self.read_noise**2 + sigma**2)
        return self.set(
            field=self.field + noise,
            std=np.broadcast_to(np.sqrt(variance), self.field.shape),
            read_noise=read_noise,
        )

    def simulate(self, key: Array, n_frames: int = 1) -> Image:
        """Simulate and average independent noisy images.

        Each image receives Poisson noise followed by zero-mean Gaussian read noise.
        The returned image stores the expected variance of the mean, including any
        variance already attached to the input image.

        Parameters
        ----------
        key : Array
            JAX random key used to generate every noise realisation.
        n_frames : int
            Number of independent images to average.

        Returns
        -------
        image : Image
            Mean realised image. ``std`` stores the expected standard deviation of
            the mean and ``read_noise`` stores its effective read-noise contribution.
        """
        if n_frames < 1:
            raise ValueError("n_frames must be a positive integer.")

        # Simulate independent photon and read-noise realisations
        read_noise = self.read_noise
        expectation = self.set(read_noise=np.zeros_like(read_noise))

        def simulate(key):
            photon_key, read_key = jr.split(key)
            image = expectation.add_poisson_noise(photon_key)
            return image.add_read_noise(read_key, read_noise)

        keys = jr.split(key, n_frames)
        images = eqx.filter_vmap(simulate)(keys)

        # Average the images and propagate the variance of their mean
        data = images.field.mean(0)
        variance = (images.std**2).mean(0) / n_frames
        read_noise = images.read_noise[0] / np.sqrt(n_frames)
        return self.set(field=data, std=np.sqrt(variance), read_noise=read_noise)
