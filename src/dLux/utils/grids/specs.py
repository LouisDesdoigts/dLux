"""Axis specifications, unit-aware sampling and one-dimensional coordinate axes."""

from numbers import Integral

import equinox as eqx
import jax.numpy as np
import zodiax as zdx
from jax import Array, core

import dLux.utils as dlu
from ..numerical.units import Unitful

__all__ = ["Spec", "UnitSpec", "Axis"]


def _check_parity(sizes_in, sizes_out, operation):
    """Validate centre-preserving parity across every spatial axis."""
    for axis, (size_in, size_out) in enumerate(zip(sizes_in, sizes_out)):
        if size_in % 2 != size_out % 2:
            raise ValueError(
                f"Center-preserving {operation} requires matching parity on axis "
                f"{axis}: {size_in} cannot be changed to {size_out}."
            )


class _PixelSpacing(zdx.Expression):
    """Resolve a diameter expression without freezing its numerical parameters."""

    diameter: zdx.Expression
    n: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, diameter, n):
        self.diameter = diameter
        self.n = n
        self.alias = None

    def evaluate(self, **context) -> Array:
        diameter = zdx.resolve(self.diameter, **context)
        diameter = dlu.as_axis(diameter, len(self.n), "diam")
        return diameter / np.asarray(self.n)


class Spec(zdx.Module):
    """Axis specifications with physical counts and array-order output shapes.

    Subclasses define ``ndim`` and ``n``. Counts are in physical-axis order and
    may be absent until an operation supplies them. ``shape`` requires explicit
    counts and reverses them into array-axis order, excluding any leading batches.
    Coordinate spacing and units belong to UnitSpec, not to every specification.
    """

    # Subclasses may provide these as stored fields or calculated properties.
    n: eqx.AbstractVar[int | tuple[int, ...] | None]
    ndim: eqx.AbstractVar[int]

    @property
    def shape(self) -> tuple[int, ...]:
        """Sample counts in array-axis order."""
        if self.n is None:
            raise ValueError("n must be specified to calculate shape.")
        counts = (self.n,) if isinstance(self.n, Integral) else self.n
        return counts[::-1]


class UnitSpec(Spec, Unitful):
    """Regular coordinate sampling shared by grids and individual axes.

    ``n`` gives static sample counts. ``d`` and ``c`` are differentiable spacing
    and centre values in ``unit``; coordinates and axis vectors use SI units.
    Concrete classes own these leaves or derive them from their physical inputs.
    There is no stored coordinate array to become stale when parameters change.
    """

    # Inherited fields
    unit: str | None = eqx.field(kw_only=True, default=None)

    d: eqx.AbstractVar[Array | zdx.Expression | None]
    c: eqx.AbstractVar[Array | zdx.Expression | None]

    def __init__(
        self, n=None, d=None, c=None, unit=None, diam=None, *, ndim=None, alias=None
    ):
        if d is not None and diam is not None:
            raise ValueError("Provide only one of d or diam.")
        if diam is not None and n is None:
            raise ValueError("n must be provided with diam.")

        # Concrete sampling types fix their dimensionality. Other UnitSpec subclasses
        # can still infer it from explicitly supplied coordinate metadata.
        if ndim is None:
            values = [
                value
                for value in (n, d, c, diam)
                if value is not None and not isinstance(value, zdx.Expression)
            ]
            lengths = [
                np.asarray(value).shape[-1]
                for value in values
                if np.asarray(value).ndim >= 1
            ]
            ndim = max(lengths, default=1 if values else 0)

        # Convert counts and diameter to spacing, keeping expression diameters deferred.
        self.n = None if n is None else dlu.as_size(n, ndim, "n")
        if diam is not None:
            if isinstance(diam, zdx.Expression):
                d = _PixelSpacing(diam, self.n)
            else:
                d = dlu.as_axis(diam, ndim, "diam") / np.asarray(self.n)

        # Store per-axis metadata; only concrete spacing values are checked here.
        self.d = d if isinstance(d, zdx.Expression) else dlu.as_axis(d, ndim, "d")
        self.c = c if isinstance(c, zdx.Expression) else dlu.as_axis(c, ndim, "c")
        if self.d is not None and not isinstance(self.d, zdx.Expression):
            invalid = np.any(~np.isfinite(self.d) | (self.d <= 0))
            if not isinstance(invalid, core.Tracer) and bool(invalid):
                raise ValueError("d must contain finite positive values.")
        self.unit = self.validate_unit(unit, name="sampling unit")
        self.alias = alias

    @property
    def ndim(self) -> int:
        """Physical dimensionality fixed by counts, or inferred from d and c."""
        if self.n is not None:
            return 1 if isinstance(self.n, Integral) else len(self.n)
        ndim = 0
        for value in (self.d, self.c):
            if value is not None and not isinstance(value, zdx.Expression):
                ndim = max(ndim, 1 if value.ndim == 0 else value.shape[-1])
        return ndim

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Broadcast leading shape of resolved spacing and centre arrays."""
        self = self.resolve()
        shapes = [
            dlu.as_axis(value, self.ndim).shape[:-1]
            for value in (self.d, self.c)
            if value is not None
        ]
        return np.broadcast_shapes(*shapes)

    @property
    def d_si(self) -> Array:
        """Spacing in SI units, retaining leading sampling batches."""
        self = self.resolve()
        if self.d is None:
            raise ValueError("d must be specified to calculate spacing.")
        return dlu.as_axis(zdx.resolve(self.d), self.ndim, "d") * dlu.unit_factor(
            self.unit
        )

    @property
    def c_si(self) -> Array:
        """Centre in SI units, with an omitted centre interpreted as zero."""
        self = self.resolve()
        if self.c is None:
            return np.zeros(self.ndim)
        return dlu.as_axis(zdx.resolve(self.c), self.ndim, "c") * dlu.unit_factor(
            self.unit
        )

    @property
    def axes(self) -> tuple[Array, ...]:
        """SI sample-centre vectors in physical-axis order."""

        # Require counts and spacing, except for a single sample located at its centre.
        self = self.resolve()
        if self.n is None:
            raise ValueError("n must be specified to calculate axes.")
        counts = (self.n,) if isinstance(self.n, Integral) else self.n
        if self.d is None:
            if all(size == 1 for size in counts):
                return tuple(self.c_si[..., i, None] for i in range(self.ndim))
            raise ValueError("d must be specified to generate coordinates.")

        # Broadcast centres and spacing, then generate physical coordinate vectors.
        spacing, centre = np.broadcast_arrays(self.d_si, self.c_si)
        return tuple(
            centre[..., i, None]
            + (np.arange(size) - (size - 1) / 2) * spacing[..., i, None]
            for i, size in enumerate(counts)
        )

    @property
    def coordinates(self) -> Array:
        """SI coordinates with shape ``(..., ndim, *shape)``."""
        return dlu.coordinate_grid(self.axes)

    @property
    def cell_volume(self) -> Array:
        """Regular coordinate-cell measure in SI units."""
        return np.prod(self.d_si, axis=-1)

    @property
    def fov(self) -> Array:
        """Cell extent n*d in declared units, distinct from endpoint intervals."""
        self = self.resolve()
        if self.n is None or self.d is None:
            raise ValueError("n and d must be specified to calculate fov.")
        return np.asarray(self.n) * dlu.as_axis(self.d, self.ndim, "d")

    def match_shape(self, shape) -> "UnitSpec":
        """Supply missing counts, or retain sampling matching an array shape."""
        shape = dlu.as_size(shape, name="shape")
        if len(shape) != self.ndim:
            raise ValueError(
                "Sampled array dimension must match its sampling UnitSpec."
            )
        if self.n is None:
            counts = shape[::-1]
            return self.set(n=counts)
        if shape != self.shape:
            raise ValueError("Sampled array shape must match its sampling UnitSpec.")
        return self

    def resample(self, n, d) -> "UnitSpec":
        """Set counts and spacing in the stored unit, retaining the centre.

        Values follow physical-axis order. Unlike padding or cropping, this
        describes new sampling; it does not preserve the original sample points.
        """
        self = self.resolve()
        n = dlu.as_size(n, self.ndim, "n")
        d = d if isinstance(d, zdx.Expression) else dlu.as_axis(d, self.ndim, "d")
        return self.set(n=n, d=d)

    def resize(self, n) -> "UnitSpec":
        """Change sample counts while keeping spacing and centre fixed.

        Counts follow physical-axis order. This defines new sampling geometry;
        changing odd/even parity moves its samples by half a cell. Use
        :meth:`resize_samples` to track padding or cropping of an existing array.
        Missing spacing remains undefined on Axis and Grid specifications.
        """
        self = self.resolve()
        n = dlu.as_size(n, self.ndim, "n")
        if self.n is not None and n == dlu.as_size(self.n, self.ndim, "n"):
            return self
        return self.resample(n, self.d)

    def resize_samples(self, n) -> "UnitSpec":
        """Track central array padding or cropping without moving retained samples.

        Spacing and centre stay fixed; spacing may remain undefined. Each axis
        must retain its odd/even parity, allowing equal padding or cropping at
        both ends. Resize the sample array separately using ``dlu.resize`` with
        the same physical-axis counts.
        """
        self = self.resolve()
        if self.n is None:
            raise ValueError("n must be specified to resize sampled coordinates.")
        counts = dlu.as_size(self.n, self.ndim, "n")
        n = dlu.as_size(n, self.ndim, "n")
        _check_parity(counts, n, "resizing")
        return self.resize(n)

    def downsample(self, factors) -> "UnitSpec":
        """Reduce counts by integer factors, preserving cell extent and centre.

        Factors follow physical-axis order and must divide each count exactly.
        Spacing grows by the same factors, matching block downsampling of an
        array without cropping leftover samples.
        """
        self = self.resolve()
        if self.n is None or self.d is None:
            raise ValueError("n and d are required to downsample sampling.")
        counts = dlu.as_size(self.n, self.ndim, "n")
        factors = dlu.as_size(factors, self.ndim, "factors")
        if any(n % factor for n, factor in zip(counts, factors)):
            raise ValueError("Downsample factors must divide sample counts exactly.")
        n = tuple(n // factor for n, factor in zip(counts, factors))
        spacing = dlu.as_axis(self.d, self.ndim, "d")
        return self.resample(n, spacing * np.asarray(factors))

    def oversample(self, factors) -> "UnitSpec":
        """Increase counts by integer factors, preserving cell extent and centre.

        Factors follow physical-axis order. Spacing decreases by the same factors;
        this changes sampling metadata without interpolating any array values.
        """
        self = self.resolve()
        if self.n is None or self.d is None:
            raise ValueError("n and d are required to oversample sampling.")
        counts = dlu.as_size(self.n, self.ndim, "n")
        factors = dlu.as_size(factors, self.ndim, "factors")
        n = tuple(n * factor for n, factor in zip(counts, factors))
        spacing = dlu.as_axis(self.d, self.ndim, "d")
        return self.resample(n, spacing / np.asarray(factors))


class Axis(UnitSpec):
    """One regularly sampled coordinate axis.

    Supply n, d and c, or give regular values from which those parameters are
    recovered. Values use unit; axes and coordinates expose SI coordinates.
    Counts are fixed topology and values are always calculated from n, d and c.
    """

    # Inherited fields
    unit: str | None = eqx.field(kw_only=True, default=None)

    n: tuple[int, ...] | None = None
    d: Array | zdx.Expression | None = None
    c: Array | zdx.Expression | None = None

    def __init__(self, values=None, unit="m", *, n=None, d=None, c=None, alias=None):
        if values is not None:
            if any(value is not None for value in (n, d, c)):
                raise ValueError("Supply values or n, d and c.")
            values = zdx.as_array(values, dtype=float)
            if values.ndim != 1 or values.size == 0:
                raise ValueError("Axis values must be a non-empty vector.")

            # Recover the regular sampling from the supplied endpoints.
            n = values.size
            c = (values[0] + values[-1]) / 2
            invalid = np.any(~np.isfinite(values))
            if n > 1:
                d = (values[-1] - values[0]) / (n - 1)
                expected = c + (np.arange(n) - (n - 1) / 2) * d
                tolerance = 8 * np.finfo(values.dtype).eps * np.max(np.abs(values))
                invalid |= np.any(np.abs(values - expected) > tolerance)
            if not isinstance(invalid, core.Tracer) and bool(invalid):
                raise ValueError("Axis values must be finite and regularly spaced.")

        # Axis parameters each describe one coordinate dimension.
        if n is not None and len(dlu.as_size(n, name="n")) != 1:
            raise ValueError("Axis has exactly one coordinate dimension.")
        for value in (d, c):
            if value is not None and not isinstance(value, zdx.Expression):
                value = np.asarray(value)
                if value.ndim > 0 and value.shape[-1] != 1:
                    raise ValueError(
                        "Axis spacing and centre must have one axis value."
                    )
        super().__init__(n=n, d=d, c=c, unit=unit, ndim=1, alias=alias)

    @property
    def ndim(self) -> int:
        return 1

    @property
    def values(self) -> Array:
        """Coordinate values in the declared unit."""
        return self.axes[0] / dlu.unit_factor(self.unit)

    @property
    def integration_weights(self) -> Array | None:
        """SI trapezoidal widths; one point alone defines no integration interval."""
        values = self.axes[0]
        if values.shape[-1] < 2:
            return None
        weights = np.broadcast_to(self.d_si, values.shape)
        return weights.at[..., 0].multiply(0.5).at[..., -1].multiply(0.5)
