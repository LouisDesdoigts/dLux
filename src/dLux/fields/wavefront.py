"""Scalar, chromatic and polarised optical fields."""

from __future__ import annotations

from math import prod

import equinox as eqx
import jax.numpy as np
import jax.tree as jtu
import zodiax as zdx
from jax import Array, core, vmap

import dLux.utils as dlu
from . import psf, intensity

__all__ = ["Wavefront"]


class Wavefront(dlu.Field):
    """A monochromatic or chromatic scalar/Jones optical field.

    Ordinary batch axes precede wavelength axes; the final two axes are spatial.
    With ``jones=True``, two Jones axes precede the spatial axes, giving
    (..., 2, 2, ny, nx). These samples represent the optical transfer matrix,
    evaluated for unpolarised unit input unless an input Stokes vector is supplied.

    Wavefront contains optical samples only. Sources and imaging systems own
    spectral weights, fluxes, filters and detector responses.

    A spatially uniform ``jones`` matrix remains separate from the scalar array.
    ``array``, ``field`` and ``phasor`` expose this stored coherent factor; call
    ``materialise()`` for the complete spatial Jones field. Intensity and Stokes
    calculations always include the uniform matrix without expanding it spatially.
    """

    # Inherited fields
    array: Array
    grid: dlu.UnitSpec

    wavelength: Array
    jones: Array | bool | None = None

    def __init__(
        self,
        wavelength: float | Array,
        grid: dlu.Grid | None = None,
        phasor: Array | None = None,
        *,
        jones: Array | bool | None = None,
        npixels=None,
        diameter=None,
        pixel_scale=None,
        center=None,
        alias=None,
    ):
        """Create sampled complex amplitudes on a two-dimensional grid.

        Parameters
        ----------
        wavelength : float or Array, metres
            Scalar or array defining wavelength axes after the ordinary batch axes.
            Population-only sampling and Jones matrices need singleton wavelength axes.
        grid : Grid
            Spatial sampling. Its n is required when phasor is omitted.
        phasor : Array or None
            Scalar (..., ny, nx) or polarised (..., 2, 2, ny, nx) samples.
            An unbatched field broadcasts across wavelength axes. None creates
            uniform unit-power samples. With jones=True, a 2D input is
            promoted to a scalar field times the identity Jones matrix.
        jones : Array, True or None
            None denotes a scalar field. Literal True marks complete spatial
            Jones samples. Otherwise this is a spatially uniform matrix of shape
            ``(*batch, 2, 2)``. Leading axes broadcast against the field batches.
            For a field shaped (S, W, ny, nx), wavelength-only matrices use
            (W, 2, 2), while source-only matrices use (S, 1, 2, 2).
        """
        # Released sampling arguments construct this same grid-based field.
        if not isinstance(grid, dlu.Grid) or any(
            value is not None for value in (npixels, diameter, pixel_scale, center)
        ):
            from ..compat.sampling import wavefront_sampling

            if (
                grid is not None
                and not isinstance(grid, dlu.Grid)
                and phasor is not None
                and diameter is None
            ):
                diameter, phasor = phasor, None
            grid, phasor = wavefront_sampling(
                grid, npixels, diameter, pixel_scale, center, phasor
            )

        self.wavelength = zdx.as_array(wavelength, dtype=float)
        self.jones = (
            jones
            if jones is None or jones is True
            else zdx.as_array(jones, dtype=complex)
        )
        if not isinstance(grid, dlu.Grid):
            raise TypeError("grid must be a Grid.")
        grid = grid.resolve(wavelength=self.wavelength)

        # Generate a unit-power scalar field before optional Jones promotion.
        if phasor is None:
            if grid.n is None:
                raise ValueError("grid.n is required when phasor is not provided.")
            metadata = [v.shape[:-1] for v in (grid.d, grid.c) if v is not None]
            if self.jones is not None and self.jones is not True:
                metadata.append(self.jones.shape[:-2])
            batch = np.broadcast_shapes(*metadata) if metadata else ()
            batch = np.broadcast_shapes(batch, self.wavelength.shape)
            shape = batch + grid.shape
            phasor = np.ones(shape, dtype=complex) / np.sqrt(prod(grid.n))
            if self.jones is True:
                phasor = self._promote_phasor(phasor)
        else:
            phasor = zdx.as_array(phasor, dtype=complex)
            if self.jones is True and phasor.ndim == 2:
                phasor = self._promote_phasor(phasor)
            trailing = 4 if self.jones is True else 2
            if phasor.ndim < trailing:
                raise ValueError("phasor is missing its spatial or Jones axes.")
            if self.jones is True and phasor.shape[-4:-2] != (2, 2):
                raise ValueError("A polarised phasor must have Jones axes (2, 2).")
            if phasor.ndim == trailing and self.wavelength.ndim:
                phasor = np.broadcast_to(phasor, self.wavelength.shape + phasor.shape)
            end = phasor.ndim - trailing
            start = end - self.wavelength.ndim
            if start < 0 or phasor.shape[start:end] != self.wavelength.shape:
                raise ValueError("phasor wavelength axes must match wavelength.shape.")

        super().__init__(phasor, grid, alias=alias)
        self._grid_batch(self.grid.d)
        self._grid_batch(self.grid.c)
        if self.jones is not None and self.jones is not True:
            self._uniform_jones()

    @property
    def phasor(self) -> Array:
        """Return the stored coherent factor, excluding the uniform Jones leaf."""
        return self.array

    @classmethod
    def from_phasor(
        cls,
        phasor: Array[complex],
        wavelength: float | Array,
        grid: dlu.Grid | None = None,
        *,
        pixel_scale=None,
        diameter=None,
        center=None,
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
        grid : Grid
            Sampling and coordinate definition for the wavefront.

        Returns
        -------
        wavefront : Wavefront
            A new Wavefront object with the specified phasor.
        """
        from ..compat import warn_deprecated

        migration = "`dl.Wavefront.from_phasor(p, w, g)` -> `dl.Wavefront(w, g, p)`"
        warn_deprecated("Wavefront.from_phasor", "Wavefront", migration, stacklevel=3)
        if grid is None:
            return cls(
                wavelength,
                phasor=phasor,
                npixels=phasor.shape[-2:][::-1],
                pixel_scale=pixel_scale,
                diameter=diameter,
                center=center,
            )
        if any(value is not None for value in (pixel_scale, diameter, center)):
            raise TypeError("Supply grid or released sampling arguments, not both.")
        return cls(wavelength, grid, phasor)

    @property
    def intensity(self: Wavefront) -> Array:
        """Return unit-input intensity, consuming Jones axes when present."""
        intensity = np.abs(self.phasor) ** 2
        if self.jones is True:
            intensity = 0.5 * intensity.sum(axis=(-4, -3))
        elif self.jones is not None:
            power = 0.5 * np.sum(np.abs(self._uniform_jones()) ** 2, axis=(-2, -1))
            intensity = intensity * power[..., None, None]
        return intensity

    @property
    def psf(self: Wavefront) -> Array:
        """Alias the unit-input intensity."""
        return self.intensity

    def to_intensity(self, stokes=None) -> intensity.Intensity:
        """Convert optical samples to intensity, retaining wavelength axes."""
        return intensity.Intensity(self.intensity_from_stokes(stokes), self.grid)

    def to_psf(self, stokes=None) -> psf.PSF:
        """Convert to point responses without weighting or summing wavelengths."""
        return psf.PSF.from_wavefront(self, stokes)

    @property
    def wavenumber(self: Wavefront) -> Array:
        """Return ``2π / wavelength`` in inverse metres.

        The result has the same shape as ``wavelength`` and no spatial axes.
        """
        return dlu.wavenumber(self.wavelength)

    @property
    def batch_ndim(self: Wavefront) -> int:
        """Return batch dimensions, excluding Jones and spatial axes."""
        return self.phasor.ndim - (4 if self.jones is True else 2)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Leading population/wavelength axes, excluding Jones and spatial axes."""
        return self.phasor.shape[: self.batch_ndim]

    @property
    def batch_wavelength(self) -> Array:
        """Wavelengths broadcast over the explicit field batch, without pixels."""
        return dlu.broadcast_batch(self.wavelength, self.batch_shape, ndim=0)

    @property
    def is_chromatic(self: Wavefront) -> bool:
        """Return whether ``wavelength`` has one or more array dimensions."""
        return self.wavelength.ndim > 0

    @property
    def is_polarised(self: Wavefront) -> bool:
        """Return whether a uniform or spatial Jones response is present."""
        return self.jones is not None

    def _grid_batch(self, value):
        """Broadcast explicit sampling batches against wavelength and population.

        All sampling aligns from the right. Population-only grids therefore use
        a singleton wavelength axis, such as (S, 1, 2) for a (S, W) field batch.
        """
        if value is None:
            return None
        leading = self.phasor.shape[: self.batch_ndim]
        return dlu.broadcast_batch(value, leading, ndim=1)

    def _batch(self, value, ndim):
        """Broadcast metadata to the field batch and flatten it for one vmap."""
        if value is None:
            return None
        value = dlu.broadcast_batch(value, self.batch_shape, ndim)
        return dlu.flatten_batch(value, ndim)

    def _map_wavefronts(self, function, **context):
        """Call a layer once per wavelength/population sample.

        Each call receives scalar wavelength, one grid and a complete scalar or
        Jones field. Layer parameters are shared. Returned arrays, including grid
        and compact Jones metadata, regain the same leading batch axes.
        """
        if not self.batch_ndim:
            return function(self, **context)

        field_ndim = self.phasor.ndim - self.batch_ndim
        phasor = dlu.flatten_batch(self.phasor, field_ndim)
        wavelength = self._batch(self.batch_wavelength, ndim=0)
        d = self._batch(self.grid.d, ndim=1)
        c = self._batch(self.grid.c, ndim=1)
        matrices = None
        if self.jones is not None and self.jones is not True:
            matrices = self._batch(self.jones, ndim=2)

        def apply(phasor, wavelength, d, c, matrix):
            grid = self.grid.set(d=d, c=c)
            local = self.set(
                phasor=phasor,
                wavelength=wavelength,
                grid=grid,
                jones=self.jones if matrix is None else matrix,
            )
            result = function(local, **context)

            # A layer may introduce a batch, such as sparse apertures. Give its
            # metadata those axes before vmap adds the outer wavelength batches.
            grid = result.grid.set(
                d=result._grid_batch(result.grid.d), c=result._grid_batch(result.grid.c)
            )
            jones = result.jones
            if jones is not None and jones is not True:
                jones = result._uniform_jones()
            # Keep wavelength labels outside the mapped data; they are restored below.
            return result.set(grid=grid, jones=jones, wavelength=None)

        output = eqx.filter_vmap(apply)(phasor, wavelength, d, c, matrices)

        # vmap puts new axes after the flattened input batch. Restore that batch,
        # then move its wavelengths past any new population axes (e.g. apertures).
        added = output.batch_ndim - 1
        populations = self.batch_ndim - self.wavelength.ndim
        spectral_axes = tuple(range(populations, self.batch_ndim))
        destinations = tuple(axis + added for axis in spectral_axes)

        def restore(value):
            if not eqx.is_array(value):
                return value
            value = value.reshape(self.batch_shape + value.shape[1:])
            if added and spectral_axes:
                value = np.moveaxis(value, spectral_axes, destinations)
            return value

        return jtu.map(restore, output).set(wavelength=self.wavelength)

    @property
    def power(self: Wavefront) -> Array:
        """Return intensity summed independently over the final spatial axes.

        All wavelength, batch, and Jones-derived leading axes are preserved.
        """
        return np.sum(self.intensity, axis=(-2, -1))

    def _to_phasor_shape(self, array):
        """Align scalar modulation with the field, inserting only Jones axes.

        Non-scalars end in two spatial axes, which may be singleton. A value per
        batch therefore uses (*batch, 1, 1). Leading axes broadcast normally;
        their physical meaning is never inferred from matching lengths.
        """
        array = np.asarray(array)
        if array.ndim == 0:
            return array
        if array.ndim < 2:
            raise ValueError("Batched modulation needs spatial axes: (*batch, 1, 1).")
        target = self.batch_shape + self.spatial_shape
        if np.broadcast_shapes(array.shape, target) != target:
            raise ValueError("Modulation must broadcast to the existing field shape.")
        if self.jones is True:
            return array[..., None, None, :, :]
        return array

    def add_phase(self: Wavefront, phase: float | Array) -> Wavefront:
        """Return a copy with a scalar, spatial, or vectorised phase applied.

        ``phase`` is measured in radians and must broadcast against the wavelength,
        Jones, and final spatial phasor axes. ``None`` leaves the wavefront unchanged.
        """
        if phase is None:
            return self
        return self.multiply("phasor", np.exp(1j * self._to_phasor_shape(phase)))

    def phase_from_opd(self, opd) -> Array:
        """Convert scalar/spatial OPD to phase with explicit batch and Jones axes."""
        trailing = self.phasor.ndim - self.batch_ndim
        wavelength = self.batch_wavelength.reshape(self.batch_shape + (1,) * trailing)
        return dlu.opd2phase(self._to_phasor_shape(opd), wavelength)

    def add_opd(self: Wavefront, opd: float | Array) -> Wavefront:
        """Return a copy with an optical-path difference applied.

        ``opd`` is measured in metres and is scalar or ends in spatial axes.
        Batched constants use (*population, *wavelength.shape, 1, 1).
        It is converted to phase using each wavefront wavelength.
        ``None`` leaves the wavefront unchanged.
        """
        if opd is None:
            return self
        return self.multiply("phasor", np.exp(1j * self.phase_from_opd(opd)))

    def tilt(self: Wavefront, angles: Array, unit: str = "rad") -> Wavefront:
        """Return a copy with an angular source tilt applied.

        ``angles`` has final shape ``(2,)`` in physical ``(x, y)`` order and uses
        ``unit`` (radians by default). Leading angle axes follow utility broadcasting
        conventions.
        """
        return self.add_opd(dlu.tilt_opd(self.coordinates, angles, unit))

    def propagate(self, propagator: dlu.BasePropagator, **context) -> Wavefront:
        """Apply a numerical propagator while preserving physical batch axes.

        Population axes precede wavelength axes. One prepared operation is shared
        across Jones entries, and its realised grid becomes the output sampling.
        Numerical propagator parameters must broadcast to the existing field batch.
        """
        if not isinstance(propagator, dlu.BasePropagator):
            raise TypeError("propagator must be a numerical BasePropagator.")
        # The field owns wavelength; local layer context may also contain it.
        context.pop("wavelength", None)
        operation, grid_out = propagator.prepare(
            self.grid, self.batch_wavelength, return_grid=True, **context
        )
        field = self._map_components(operation, self.phasor)
        trailing = 4 if self.jones is True else 2
        if field.shape[:-trailing] != self.phasor.shape[:-trailing]:
            raise ValueError(
                "Propagator parameters must broadcast to the Wavefront batch; "
                "use explicit singleton axes for wavelength and population."
            )
        return self.set(phasor=field, grid=grid_out)

    def coherent_sum(self, axis: int = -1) -> Wavefront:
        """Add fields on one common grid before forming intensity.

        ``axis`` selects a population axis, with negative indices counted before
        wavelength, Jones and spatial axes. Wavelengths cannot be summed coherently.
        For sparse pupils, propagate patches to a common grid before summing.
        """
        populations = self.batch_ndim - self.wavelength.ndim
        axis = axis + populations if axis < 0 else axis
        if not 0 <= axis < populations:
            raise ValueError("coherent_sum requires a non-wavelength batch axis.")

        def collapse(value):
            if value is None or value.ndim == 1:
                return value
            samples = self._grid_batch(value)
            first = np.take(samples, 0, axis=axis)
            equal = np.all(
                np.isclose(samples, np.expand_dims(first, axis), rtol=1e-6, atol=0.0)
            )
            if not isinstance(equal, core.Tracer) and not bool(equal):
                raise ValueError("Coherent summation requires a common spatial grid.")
            return first

        grid = self.grid.set(d=collapse(self.grid.d), c=collapse(self.grid.c))
        if self.jones is not None and self.jones is not True:
            matrix_axis = axis - (self.batch_ndim - (self.jones.ndim - 2))
            if matrix_axis >= 0 and self.jones.shape[matrix_axis] != 1:
                self = self.materialise()
            elif matrix_axis >= 0:
                self = self.set(jones=np.squeeze(self.jones, axis=matrix_axis))
        return self.set(phasor=self.phasor.sum(axis), grid=grid)

    def normalise(
        self: Wavefront, mode: str = "power", value: float = 1.0
    ) -> Wavefront:
        """Normalise the wavefront.

        Parameters
        ----------
        mode : {"power","peak"} = "power"
            - "power": scales so summed spatial intensity equals value per batch.
            - "peak" : scales so maximum spatial intensity equals value per batch.
        value : float = 1.0
            Target value for the selected mode.

        Returns
        -------
        wavefront : Wavefront
            New wavefront with phasor scaled to achieve the normalisation.
        """
        if mode not in ("power", "peak"):
            raise ValueError("mode must be 'power' or 'peak'")
        reduction = "sum" if mode == "power" else "peak"
        measure = dlu.norm(self.intensity, reduction, axis=(-2, -1))
        scale = np.sqrt(value / measure)
        trailing = self.phasor.ndim - scale.ndim
        scale = scale.reshape(scale.shape + (1,) * trailing)
        return self.set(phasor=self.phasor * scale)

    def _binary_op(
        self: Wavefront, other: Wavefront | Array | None, op: str
    ) -> Wavefront:
        """Apply arithmetic after aligning wavelength and Jones axes."""
        if other is None:
            return self
        if op == "divide" and isinstance(other, Wavefront):
            raise TypeError(
                "dLux has detected an attempt to perform dark optics. Your wavefront "
                "privileges have been temporarily suspended and the authorities have "
                "been notified."
            )
        if isinstance(other, Wavefront):
            if op in ("add", "subtract") and (self.is_polarised or other.is_polarised):
                self, other = self.materialise(), other.materialise()
            elif (
                op == "multiply" and other.jones is not None and other.jones is not True
            ):
                if not self.is_polarised:
                    matrix = other.jones
                    if self.is_chromatic and not other.is_chromatic:
                        shape = other.batch_shape + (1,) * self.wavelength.ndim + (2, 2)
                        matrix = other._uniform_jones().reshape(shape)
                    self, other = self.set(jones=matrix), other.set(jones=None)
                else:
                    self, other = self.materialise(), other.materialise()
            elif op == "multiply" and self.jones is not None and other.jones is True:
                self = self.materialise()
        elif op in ("add", "subtract") and self.jones is not None:
            self = self.materialise()
        return super()._binary_op(other, op)

    def _prepare_operand(
        self: Wavefront, other: Wavefront | Array | float | int | complex
    ) -> tuple[Wavefront, Array | float | int | complex]:
        """Align wavelength and Jones axes for wavefront arithmetic."""
        if isinstance(other, Wavefront):
            return self._prepare_wavefront_operand(other)
        if isinstance(other, Array):
            return self._prepare_array_operand(other)
        if isinstance(other, dlu.Field):
            raise TypeError("Wavefront arithmetic requires another Wavefront or array.")
        return self, other

    def _promote_for_arithmetic(self: Wavefront) -> Wavefront:
        """Promotes the base type while retaining scalar Jones broadcasting."""
        phasor = self.phasor[..., None, None, :, :]
        return self.to_polarised().set(phasor=phasor)

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
            The base wavefront, promoted to Jones samples if required.
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

        self_polarised = self.jones is True
        other_polarised = other.jones is True
        phasor = other.phasor
        if self.is_chromatic and not other.is_chromatic:
            # A monochromatic population applies to every wavelength of the base.
            trailing = phasor.shape[other.batch_ndim :]
            shape = other.batch_shape + (1,) * self.wavelength.ndim + trailing
            phasor = phasor.reshape(shape)

        # Matching polarisation types already have compatible intrinsic dimensions.
        if self_polarised == other_polarised:
            return self, phasor

        # Regular phasors act as scalar Jones modulation, so insert singleton Jones
        # axes immediately before their spatial dimensions.
        if self_polarised:
            return self, phasor[..., None, None, :, :]

        return self._promote_for_arithmetic(), phasor

    def _prepare_array_operand(
        self: Wavefront, other: Array
    ) -> tuple[Wavefront, Array]:
        """Align standard field modulation and otherwise use JAX broadcasting."""
        return self, self._to_phasor_shape(other)

    @staticmethod
    def _promote_phasor(phasor):
        """Insert identity Jones axes before the two spatial dimensions."""
        return phasor[..., None, None, :, :] * np.eye(2)[..., None, None]

    def _uniform_jones(self, matrix=None) -> Array:
        """Broadcast compact matrices over explicit field batches, without pixels."""
        matrix = self.jones if matrix is None else matrix
        matrix = np.eye(2, dtype=complex) if matrix is None else np.asarray(matrix)
        if matrix.shape[-2:] != (2, 2):
            raise ValueError("Jones matrices must end in (2, 2).")
        return dlu.broadcast_batch(matrix, self.batch_shape, ndim=2)

    def materialise(self) -> Wavefront:
        """Return the complete spatial Jones transfer, marked by ``jones=True``.

        This allocates intrinsic Jones axes only when needed. A scalar field with
        no Jones response becomes that scalar times the identity matrix.
        """
        if self.jones is True:
            return self
        matrix = self._uniform_jones()
        phasor = self.phasor[..., None, None, :, :] * matrix[..., None, None]
        return self.set(phasor=phasor, jones=True)

    def to_polarised(self) -> Wavefront:
        """Return the complete spatial Jones transfer."""
        return self.materialise()

    def apply_jones(self, jones, *, spatial=False) -> Wavefront:
        """Apply compact (*batch, 2, 2) or explicitly spatial Jones matrices.

        spatial=True accepts (*batch, ny, nx, 2, 2), including singleton spatial
        axes. Both forms use ordinary leading batch broadcasting. No array shape
        is interpreted as a wavelength or spatial axis implicitly.
        """
        jones = np.asarray(jones, dtype=complex)
        if jones.shape[-2:] != (2, 2):
            raise ValueError("Jones matrices must end in (2, 2).")
        if not spatial and self.jones is not True:
            self._uniform_jones(jones)
            previous = np.eye(2, dtype=complex) if self.jones is None else self.jones
            return self.set(jones=jones @ previous)

        wavefront = self.materialise()
        # Numerical matrices keep their intrinsic axes last; fields keep pixels last.
        phasor = np.moveaxis(wavefront.phasor, (-4, -3), (-2, -1))
        if not spatial:
            jones = self._uniform_jones(jones)[..., None, None, :, :]
        if np.broadcast_shapes(jones.shape, phasor.shape) != phasor.shape:
            raise ValueError("Jones matrices must broadcast to the existing field.")
        phasor = jones @ phasor
        return wavefront.set(phasor=np.moveaxis(phasor, (-2, -1), (-4, -3)))

    def stokes(self, input_stokes=None) -> Array:
        """Return output Stokes samples with shape (..., 4, ny, nx).

        Input Stokes vectors have a final axis of length four. None denotes an
        unpolarised unit input. Leading axes broadcast against the wavefront.
        """
        if input_stokes is not None:
            input_stokes = np.asarray(input_stokes)
            if input_stokes.ndim == 0 or input_stokes.shape[-1] != 4:
                raise ValueError("Input Stokes vectors need a final axis of four.")
            input_stokes = dlu.broadcast_batch(input_stokes, self.batch_shape, ndim=1)
        if self.jones is not True:
            stokes = dlu.JonesMatrix(self._uniform_jones()).to_stokes(input_stokes)
            return stokes[..., None, None] * (np.abs(self.phasor) ** 2)[..., None, :, :]

        matrix = np.moveaxis(self.phasor, (-4, -3), (-2, -1))
        if input_stokes is not None:
            input_stokes = input_stokes[..., None, None, :]
        stokes = dlu.JonesMatrix(matrix).to_stokes(input_stokes)
        return np.moveaxis(stokes, -1, -3)

    def intensity_from_stokes(self, stokes=None) -> Array:
        """Return intensity for an optional input state (..., 4)."""
        if stokes is None:
            return self.intensity
        if self.is_polarised:
            return self.stokes(stokes)[..., 0, :, :]
        stokes = np.asarray(stokes)
        if stokes.ndim == 0 or stokes.shape[-1] != 4:
            raise ValueError("Input Stokes vectors need a final axis of four.")
        stokes = dlu.broadcast_batch(stokes, self.batch_shape, ndim=1)
        return stokes[..., 0, None, None] * self.intensity

    def psf_from_stokes(self, stokes=None) -> Array:
        """Alias intensity_from_stokes; no photometric weighting is applied."""
        return self.intensity_from_stokes(stokes)

    def _map_components(self, function, array, sample_ndim=2):
        """Apply a scalar spatial kernel independently to complete Jones samples."""
        if self.jones is not True:
            return function(array)
        array = np.moveaxis(array, (-4, -3), (0, 1))
        output = vmap(vmap(function))(array)
        jones_axes = (-sample_ndim - 2, -sample_ndim - 1)
        return np.moveaxis(output, (0, 1), jones_axes)

    def _map_arrays(self, function, *others):
        """Share Field's array kernels across complete Jones components."""

        def apply(array):
            field = self.set(phasor=array)
            return dlu.Field._map_arrays(field, function, *others)

        return self._map_components(apply, self.phasor)

    def _sample(self, knots, coordinates, method, complex, fill=None, **kwargs):
        """Interpolate field batches, sharing the sampling across Jones entries."""

        def sample(array):
            operation = dlu.Interp(
                array, knots, ndim=2, method=method, complex=complex, fill=fill
            )
            return operation.sample(coordinates)

        sample_ndim = 2
        if isinstance(coordinates, tuple):
            shapes = [np.shape(value) for value in coordinates]
            query_shape = np.broadcast_shapes(*shapes)
            sample_ndim = len(query_shape)
        array = self._map_components(sample, self.phasor, sample_ndim)
        return self.set(phasor=array, grid=kwargs.get("grid_out") or self.grid)

    def _resample(self, operation, **context):
        """Apply Interp's ordinary batches to complete field sampling."""
        context = dict(
            context, wavelength=self.batch_wavelength, wavelengths=self.batch_wavelength
        )
        grid = self.grid.resolve(**context)
        output = operation.output_grid(self.spatial_shape, grid, **context)
        operation = operation.set(grid_out=output)

        def sample(array):
            return operation(array, grid_in=grid, **context)

        array = self._map_components(sample, self.phasor)
        return self.set(phasor=array, grid=output)

    def sample(self, coordinates, method="cubic2", *, fill=None) -> Array:
        """Sample the complete Jones response; None extrapolates, else use fill."""
        wavefront = self.materialise() if self.jones is not None else self
        return dlu.Field.sample(wavefront, coordinates, method, fill=fill)

    def flip(self, axis) -> Wavefront:
        """Flip samples and any compact Jones metadata on the same batch axes."""
        phasor = np.flip(self.phasor, axis)
        jones = self.jones
        if jones is not None and jones is not True:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            axes = tuple(value % self.phasor.ndim for value in axes)
            batch_axes = tuple(value for value in axes if value < self.batch_ndim)
            if batch_axes:
                jones = np.flip(self._uniform_jones(), batch_axes)
        return self.set(phasor=phasor, jones=jones)

    def resample_wavelengths(self, wavelengths, method="cubic2") -> Wavefront:
        """Interpolate complex samples onto a new wavelength vector.

        The current wavelength vector must contain at least two increasing
        samples. Real and imaginary parts are interpolated separately, without
        intensity normalisation or spectral weighting. Every wavelength must
        share spatial sampling; resample to a common grid before using this
        method. Queries outside the wavelength range return zero.
        """
        wavelengths = zdx.as_array(wavelengths, dtype=float)
        if self.wavelength.ndim != 1 or self.wavelength.size < 2:
            raise ValueError("Spectral interpolation requires a wavelength vector.")
        if wavelengths.ndim != 1:
            raise ValueError("wavelengths must be a one-dimensional vector.")
        # Only explicit singleton wavelength axes may be shared by all samples.
        for value in (self.grid.d, self.grid.c):
            if value is not None and value.ndim > 1 and value.shape[-2] != 1:
                raise ValueError("Resample onto a common spatial grid first.")

        # A wavelength-dependent matrix must be interpolated as part of the field.
        if self.jones is not None and self.jones is not True:
            if self.jones.ndim > 2 and self.jones.shape[-3] != 1:
                self = self.materialise()

        spectral_axis = self.batch_ndim - 1
        interpolation = dlu.Interp(
            ndim=1,
            array=np.moveaxis(self.phasor, spectral_axis, -1),
            grid_in=self.wavelength,
            method=method,
            fill=0.0,
        )
        phasor = np.moveaxis(interpolation.sample(wavelengths), -1, spectral_axis)
        return self.set(phasor=phasor, wavelength=wavelengths)
