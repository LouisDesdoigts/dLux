"""Regularly sampled optical fields and operations."""

from __future__ import annotations
from abc import abstractmethod
from math import prod
import operator

import jax.numpy as np
import jax.random as jr
import jax.scipy as jsp
import zodiax as zdx
from jax import Array, vmap
from jax.scipy.signal import convolve

import dLux.utils as dlu

from .grids import GridSpec
from .grids import CoordTransform

__all__ = [
    "BaseField",
    "ContinuousField",
    "DiscreteField",
    "Wavefront",
    "PolarisedWavefront",
    "PSF",
    "Image",
]

_ops = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
}


def _field_spec(spec, shape):
    """Validate a field specification against its two spatial axes."""
    if not isinstance(spec, GridSpec):
        raise TypeError("spec must be a GridSpec.")
    spec = spec.broadcast(2)
    n = shape[-2:][::-1]
    if spec.n is None:
        return spec.set(n=n)
    if spec.n != n:
        raise ValueError("Field spatial shape must match spec.n.")
    return spec


class BaseField(zdx.Base):
    """Base class for regularly sampled real or complex fields."""

    spec: GridSpec

    def __getattr__(self, key):
        """Forward unknown attributes to the coordinate specification."""
        if hasattr(self.spec, key):
            return getattr(self.spec, key)
        raise AttributeError(f"{type(self).__name__} has no attribute {key!r}.")

    @property
    @abstractmethod
    def field(self) -> Array:
        """Return the stored sampled array."""
        raise NotImplementedError()  # pragma: no cover

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        """Return the spatial array shape."""
        return self.field.shape[-2:]

    @property
    def axes(self) -> tuple[Array, ...]:
        """Return coordinate axes using the field's static spatial shape."""
        return self.xs

    @property
    def coordinates(self) -> Array:
        """Return coordinates using the field's static spatial shape."""
        return self.spec.coordinates_for(self.spatial_shape[::-1])

    @property
    def xs(self) -> tuple[Array, ...]:
        """Return coordinate axes using the field's static spatial shape."""
        return self.spec.xs_for(self.spatial_shape[::-1])

    @property
    def npixels(self) -> int:
        """Return the final spatial-axis size for square-grid compatibility."""
        return self.field.shape[-1]

    @property
    def pixel_scale(self) -> Array:
        """Return per-axis sampling in canonical SI units."""
        if self.d is None:
            raise ValueError("spec.d is not defined.")
        return self.d * self.scale

    @property
    def center(self) -> Array:
        """Return the per-axis grid centre in canonical SI units."""
        return np.zeros(len(self.n)) if self.c is None else self.c * self.scale

    @property
    def diameter(self) -> Array:
        """Return the physical field width along every axis."""
        return self.fov

    def normalise(self, mode: str = "power", value: float = 1.0) -> BaseField:
        """Return a field normalised by total power or peak value."""
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
        """Convolve the sampled field with an input array."""
        other = np.asarray(other)
        batch = np.broadcast_shapes(self.field.shape[:-2], other.shape[:-2])
        field = np.broadcast_to(self.field, batch + self.field.shape[-2:])
        kernel = np.broadcast_to(other, batch + other.shape[-2:])
        apply = lambda x, y: convolve(x, y, mode=mode, method=method)
        field = vmap(apply)(
            field.reshape((-1,) + field.shape[-2:]),
            kernel.reshape((-1,) + kernel.shape[-2:]),
        )
        field = field.reshape(batch + field.shape[-2:])
        return self.set(field=field, spec=self.spec.resize(field.shape[-2:][::-1]))

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
        """Resize spatial axes by centered zero-padding or cropping."""
        fill = 0j if np.iscomplexobj(self.field) else 0.0
        field = dlu.resize(self.field, npixels, fill)
        return self.set(field=field, spec=self.spec.resize(npixels))

    def downsample(
        self, n: int | tuple[int, int], mean: bool | None = None
    ) -> BaseField:
        """Downsample spatial axes and update their sampling."""
        if mean is None:
            mean = bool(np.iscomplexobj(self.field))
        field = dlu.downsample(self.field, n, mean)
        return self.set(field=field, spec=self.spec.downsample(n))

    def flip(self, axis: tuple[int, ...] | int) -> BaseField:
        """Flip the sampled array about one or more array axes."""
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

    spec: GridSpec

    def scale_to(
        self,
        npixels: int | tuple[int, int],
        pixel_scale: float | Array,
        method: str = "linear",
        complex: bool = True,
    ) -> ContinuousField:
        """Interpolate to a size and physical per-axis pixel scale.

        ``complex`` selects Cartesian or polar decomposition for complex fields and
        has no effect on real fields such as PSFs.
        """
        n = dlu.as_size(npixels, 2, "npixels")
        spacing = dlu.as_axis(pixel_scale, 2, "pixel_scale") / self.spec.scale
        ratio = spacing / self.d
        scale = np.vectorize(
            lambda field, value: dlu.scale(field, npixels, value, method, complex),
            signature="(n,m),(c)->(p,q)",
        )
        field = scale(self.field, ratio)
        return self.set(field=field, spec=self.spec.resample(n, spacing))

    def interpolate(
        self,
        transformation: CoordTransform,
        method: str = "linear",
        complex: bool = True,
        fill: float = 0.0,
    ) -> ContinuousField:
        """Interpolate every sampled field through a coordinate transformation."""
        if not isinstance(transformation, CoordTransform):
            raise TypeError("transformation must be a CoordTransform.")
        knots = self.coordinates
        transform = np.vectorize(transformation, signature="(c,n,m)->(c,n,m)")
        samples = transform(knots)

        # Coordinate batch axes precede intrinsic field axes such as Jones matrices.
        n_batch = self.field.ndim - 2
        c_batch = knots.ndim - 3
        if c_batch > n_batch:
            raise ValueError("Coordinate batch dimensions exceed field dimensions.")
        shape = knots.shape[:c_batch] + (1,) * (n_batch - c_batch) + knots.shape[-3:]
        knots, samples = knots.reshape(shape), samples.reshape(shape)

        interpolate = np.vectorize(
            lambda field, x, y: dlu.interp(field, x, y, method, fill, complex),
            signature="(n,m),(c,n,m),(c,p,q)->(p,q)",
        )
        return self.set(field=interpolate(self.field, knots, samples))

    def rotate(
        self, angle: float | Array, method: str = "linear", complex: bool = True
    ) -> ContinuousField:
        """Rotate the sampled array clockwise through interpolation.

        ``complex`` has no effect when the stored sampled array is real.
        """
        rotate = np.vectorize(
            lambda field, value: dlu.rotate(field, value, method, complex),
            signature="(n,m),()->(n,m)",
        )
        return self.set(field=rotate(self.field, angle))


class DiscreteField(BaseField):
    """Base class for discrete detector-sampled fields."""

    spec: GridSpec
    variance: Array | None
    read_noise: Array

    @property
    def error(self) -> Array | None:
        """Return the standard deviation implied by ``variance``."""
        return None if self.variance is None else np.sqrt(self.variance)

    @property
    def fourier_transform(self) -> Array:
        """Return the centered two-dimensional Fourier transform."""
        transformed = np.fft.fft2(self.field, axes=(-2, -1))
        return np.fft.fftshift(transformed, axes=(-2, -1))

    @property
    def amplitude_spectrum(self) -> Array:
        """Return the amplitude of the centered Fourier transform."""
        return np.abs(self.fourier_transform)

    @property
    def power_spectrum(self) -> Array:
        """Return the squared amplitude of the centered Fourier transform."""
        return self.amplitude_spectrum**2

    def add_poisson_noise(self, key: Array) -> DiscreteField:
        """Add a Poisson realization and its expected variance."""
        expectation = self.field
        data = jr.poisson(key, expectation).astype(self.field.dtype)
        variance = expectation
        if self.variance is not None:
            variance = variance + self.variance
        return self.set(field=data, variance=variance)

    def add_read_noise(self, key: Array, sigma: float | Array) -> DiscreteField:
        """Add zero-mean Gaussian read noise and update its variance."""
        sigma = np.asarray(sigma, dtype=self.field.dtype)
        noise = jr.normal(key, self.field.shape, self.field.dtype) * sigma
        variance = sigma**2
        if self.variance is not None:
            variance = variance + self.variance
        read_noise = np.sqrt(self.read_noise**2 + sigma**2)
        return self.set(field=self.field + noise).set(
            variance=np.broadcast_to(variance, self.field.shape), read_noise=read_noise
        )

    def log_likelihood(
        self, model: BaseField | Array, distribution: str = "gaussian"
    ) -> Array:
        """Return a summed Gaussian or Poisson log likelihood.

        The Gaussian likelihood uses the stored ``variance``. The Poisson
        likelihood is exact for count data without additive read noise.
        """
        model = model.field if isinstance(model, BaseField) else np.asarray(model)
        if model.shape != self.field.shape:
            raise ValueError("model and data must have matching shapes.")
        if distribution == "gaussian":
            if self.variance is None:
                raise ValueError("variance is required for a Gaussian likelihood.")
            residual = self.field - model
            terms = residual**2 / self.variance + np.log(2 * np.pi * self.variance)
            return -0.5 * terms.sum()
        if distribution == "poisson":
            return (
                jsp.special.xlogy(self.field, model)
                - model
                - jsp.special.gammaln(self.field + 1)
            ).sum()
        raise ValueError("distribution must be 'gaussian' or 'poisson'.")


class Wavefront(ContinuousField):
    """
    Holds the state of a wavefront as it is transformed and propagated through an
    optical system. The final two phasor axes are the square spatial wavefront; any
    preceding axes are treated as vectorisation axes. Passing a vector of wavelengths
    creates a chromatic wavefront with matching leading phasor dimensions.

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
    spec : GridSpec, property
        Derived coordinate specification for the current wavefront sampling.
    xs : Array, property
        Derived pixel-centre coordinates along one axis, in metres.
    """

    spec: GridSpec
    phasor: Array[complex]
    wavelength: Array

    @property
    def field(self) -> Array:
        """Return the complex phasor."""
        return self.phasor

    def __init__(
        self: Wavefront,
        wavelength: float | Array,
        spec: GridSpec,
        phasor: Array | None = None,
    ):
        """
        Parameters
        ----------
        wavelength : float or Array, meters
            The wavelength of the `Wavefront`. Passing an array creates a chromatic
            wavefront with phasor shape `wavelength.shape + (npixels, npixels)`.
        spec : GridSpec
            Sampling and coordinate definition for the wavefront.
        phasor : Array or None
            Optional complex electric field. If omitted, a uniform field is generated
            from ``spec.n``.
        """
        self.wavelength = np.asarray(wavelength, float)
        if phasor is None:
            if not isinstance(spec, GridSpec):
                raise TypeError("spec must be a GridSpec.")
            spec = spec.broadcast(2)
            if spec.n is None:
                raise ValueError("spec.n is required when phasor is not provided.")
            shape = self.wavelength.shape + spec.shape
            self.phasor = np.ones(shape, dtype=complex) / prod(spec.n)
        else:
            phasor = np.asarray(phasor, complex)
            if phasor.ndim < 2:
                raise ValueError("phasor must have at least two spatial dimensions.")
            spec = _field_spec(spec, phasor.shape)
            if phasor.ndim == 2 and self.wavelength.ndim > 0:
                phasor = phasor * np.ones(self.wavelength.shape + (1, 1))
            self.phasor = phasor
        self.spec = spec

    @classmethod
    def from_phasor(
        cls, phasor: Array[complex], wavelength: float | Array, spec: GridSpec
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
        spec : GridSpec
            Sampling and coordinate definition for the wavefront.

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
    def is_polarised(self: Wavefront) -> bool:
        """Return whether this wavefront carries Jones-matrix axes."""
        return False

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
            phasor=0 if self.batch_ndim > 0 else None, wavelength=0, spec=None
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
        self: Wavefront, mode: str = "power", value: float = 1.0
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

    spec: GridSpec
    phasor: Array[complex]
    wavelength: Array

    @property
    def is_polarised(self: PolarisedWavefront) -> bool:
        """Return whether this wavefront carries Jones-matrix axes."""
        return True

    def __init__(
        self: Wavefront,
        wavelength: float | Array,
        spec: GridSpec,
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
        cls, phasor: Array[complex], wavelength: float | Array, spec: GridSpec
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
        spec : GridSpec
            Sampling and coordinate definition for the wavefront.

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
            phasor=PolarisedWavefront._promote_phasor(wavefront.phasor),
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


class PSF(ContinuousField):
    """A real-valued point-spread function sampled on a coordinate grid."""

    data: Array
    spec: GridSpec

    @property
    def field(self) -> Array:
        """Return the sampled intensity."""
        return self.data

    def __init__(self: PSF, data: Array, spec: GridSpec):
        self.data = np.asarray(data, dtype=float)
        if self.data.ndim < 2:
            raise ValueError("data must have at least two spatial dimensions.")
        self.spec = _field_spec(spec, self.data.shape)

    @classmethod
    def from_wavefront(cls, wavefront) -> PSF:
        """Construct a PSF from a wavefront's intensity and specification."""
        return cls(wavefront.psf, wavefront.spec)

    @property
    def batch_ndim(self: PSF) -> int:
        """Return the number of leading vectorisation dimensions."""
        return self.data.ndim - 2


class Image(DiscreteField):
    """A discrete detector image and its uncertainty metadata.

    Parameters
    ----------
    data : Array
        Detector-sampled image data.
    spec : GridSpec
        Coordinate specification tracking the detector pixel grid.
    variance : Array or None
        Known variance of the observed data. This is populated by the noise
        simulation methods and may also be supplied directly.
    read_noise : float or Array
        Gaussian read-noise standard deviation associated with the image.
    """

    data: Array
    spec: GridSpec
    variance: Array | None
    read_noise: Array

    @property
    def field(self) -> Array:
        """Return the detector data."""
        return self.data

    def __init__(
        self,
        data: Array,
        spec: GridSpec,
        variance: Array | None = None,
        read_noise: float | Array = 0.0,
    ):
        data = np.asarray(data, dtype=float)
        if data.ndim < 2:
            raise ValueError("data must have at least two spatial dimensions.")
        spec = _field_spec(spec, data.shape)
        if variance is not None:
            variance = np.broadcast_to(np.asarray(variance, dtype=float), data.shape)
        self.data = data
        self.variance = variance
        self.read_noise = np.asarray(read_noise, dtype=float)
        self.spec = spec
