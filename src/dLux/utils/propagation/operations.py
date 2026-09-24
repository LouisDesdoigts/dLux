"""Small prepared operations shared by propagation evaluations."""

from __future__ import annotations

from jax import Array
import jax.numpy as np

from ..grids.array_ops import resize
from ..grids.array_ops import as_size
from ..batching import broadcast_batch
from ..sequential import BaseLayer, Series

__all__ = ["Resize", "Chirp"]


def _flatten_steps(steps) -> tuple:
    """Flatten numerical construction sequences while retaining evaluation order."""
    flat = []
    for step in steps:
        # Specialised chains may override apply; flatten only the generic sequence.
        if type(step) is Series:
            flat.extend(_flatten_steps(step.steps.values()))
        else:
            flat.append(step)
    return tuple(flat)


def _series(*steps) -> BaseLayer:
    """Return one numerical operation directly or a flattened series."""
    steps = _flatten_steps(steps)
    if not steps:
        raise ValueError("steps must contain at least one operation.")
    if any(not isinstance(step, BaseLayer) for step in steps):
        raise TypeError("Every step must be a BaseLayer.")

    # Fold adjacent sampled phases during preparation, never during evaluation.
    fused = []
    for step in steps:
        if fused and isinstance(fused[-1], Chirp) and isinstance(step, Chirp):
            fused[-1] = fused[-1].then(step)
        else:
            fused.append(step)
    return fused[0] if len(fused) == 1 else Series(fused)


class Resize(BaseLayer):
    """Centrally resize the final field axes to fixed physical-order counts.

    Input and output counts must have matching odd/even parity on every axis.
    Padding and cropping are symmetric, retaining the physical centre.
    """

    n: tuple[int, int]

    def __init__(self, n, *, alias=None):
        """Initialise a fixed central field resize.

        Parameters
        ----------
        n : int or tuple[int, int]
            Positive static output counts in (x, y) order; a scalar defines a square.
        alias : mapping or sequence of pairs, optional
            Zodiax parameter aliases.
        """
        self.n = as_size(n, 2, "n")

        self.alias = alias

    def apply(self, field, **context) -> Array:
        """Resize ``field`` while preserving every leading axis."""
        return resize(field, self.n)


class Chirp(BaseLayer):
    """Apply separable sampled chirps without constructing a two-dimensional mesh.

    Each sampled chirp has shape ``chirp_batch + (n_axis,)`` and chirps follow
    physical (x, y) order. Leading dimensions use ordinary JAX broadcasting with
    the field batch, including expansion through explicit singleton axes.
    """

    factors: tuple[Array, Array]

    def __init__(self, factors, *, alias=None):
        """Initialise separable sampled chirps.

        Parameters
        ----------
        factors : sequence[Array]
            Two non-empty chirp vectors with optional leading batch dimensions.
        alias : mapping or sequence of pairs, optional
            Zodiax parameter aliases.
        """
        factors = tuple(np.asarray(factor) for factor in factors)
        empty = any(factor.ndim < 1 or factor.shape[-1] < 1 for factor in factors)
        if len(factors) != 2 or empty:
            raise ValueError(
                "factors must contain two non-empty vectors in (x, y) order."
            )
        self.factors = factors
        self.alias = alias

    @property
    def n(self) -> tuple[int, int]:
        """Return chirp lengths in physical-axis order."""
        return tuple(factor.shape[-1] for factor in self.factors)

    @classmethod
    def from_axes(cls, axes, quadratic=None, linear=None, constant=None) -> Chirp:
        """Sample ``exp(i * (quadratic*x**2 + linear*x + constant))`` per axis.

        ``axes`` contains two coordinate vectors in (x, y) order. Coefficients
        are shared scalars or broadcast arrays; a tuple supplies one per axis.
        Leading axes broadcast normally during preparation and application.
        ``None`` omits a phase term entirely; at least one term is required.
        """
        if quadratic is None and linear is None and constant is None:
            raise ValueError("Provide at least one phase coefficient.")

        # Share scalar coefficients across axes; tuples specify each axis independently.
        def coefficients(value):
            if value is None:
                return (None,) * len(axes)
            if not isinstance(value, tuple):
                value = (value,) * len(axes)
            if len(value) != len(axes):
                raise ValueError("coefficients must contain one value per axis.")
            return tuple(np.asarray(item)[..., None] for item in value)

        q, l, c = map(coefficients, (quadratic, linear, constant))
        # Omitted terms create no sampled arrays, particularly on linear FFT ramps.
        factors = []
        for x, q_i, l_i, c_i in zip(axes, q, l, c):
            phase = 0.0
            if q_i is not None:
                phase = phase + q_i * x**2
            if l_i is not None:
                phase = phase + l_i * x
            if c_i is not None:
                phase = phase + c_i
            factor = np.exp(1j * phase)
            shape = np.broadcast_shapes(np.shape(phase), np.shape(x))
            factors.append(np.broadcast_to(factor, shape))
        return cls(tuple(factors))

    def broadcast(self, batch_shape) -> Chirp:
        """Broadcast both factors to an explicit leading batch shape."""
        factors = tuple(broadcast_batch(x, batch_shape, 1) for x in self.factors)
        return Chirp(factors, alias=self.alias)

    def then(self, other) -> Chirp:
        """Fuse matching chirps with ordinary leading batch broadcasting."""
        if not isinstance(other, Chirp) or other.n != self.n:
            raise ValueError("other must be a Chirp with matching sample counts.")
        factors = tuple(
            first * second for first, second in zip(self.factors, other.factors)
        )
        return Chirp(factors, alias=self.alias)

    def apply(self, field, **context) -> Array:
        """Apply physical-order factors with ordinary leading batch broadcasting."""
        factors = self.factors
        field = np.asarray(field)
        if field.shape[-2:] != self.n[::-1]:
            raise ValueError("field spatial shape must match the separable factors.")

        factor_x, factor_y = factors
        return (field * factor_x[..., None, :]) * factor_y[..., :, None]
