"""Small prepared operations shared by propagation evaluations."""

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jax import Array, lax
import jax.numpy as np
import zodiax as zdx

from ..array_ops import resize_samples
from ..helpers import as_size
from ..sequential import Sequential

__all__ = ["BaseOperation", "Resize", "Chirp"]


def _prepare_factors(factors, name="factors") -> tuple[Array, ...]:
    """Convert separable factors, retaining their preparation batch shape."""
    factors = tuple(np.asarray(factor) for factor in factors)
    if not factors or any(
        factor.ndim < 1 or factor.shape[-1] < 1 for factor in factors
    ):
        raise ValueError(f"{name} must contain at least one non-empty vector.")

    return factors


def _factor_sizes(factors) -> tuple[int, ...]:
    """Return separable factor lengths in physical-axis order."""
    return tuple(factor.shape[-1] for factor in factors)


def _resize_array(array, n) -> Array:
    """Centrally resize final array axes to physical-order sample counts."""
    return resize_samples(array, n)


def _flatten_steps(steps) -> tuple:
    """Flatten numerical construction sequences while retaining evaluation order."""
    flat = []
    for step in steps:
        if isinstance(step, Sequential):
            flat.extend(_flatten_steps(step.steps.values()))
        else:
            flat.append(step)
    return tuple(flat)


def _series(*steps):
    """Return one numerical operation directly or a flattened series."""
    steps = _flatten_steps(steps)
    if not steps:
        raise ValueError("steps must contain at least one operation.")
    if any(not isinstance(step, BaseOperation) for step in steps):
        raise TypeError("Every step must be a BaseOperation.")

    # Fold adjacent sampled phases during preparation, never during evaluation.
    fused = []
    for step in steps:
        if fused and isinstance(fused[-1], Chirp) and isinstance(step, Chirp):
            fused[-1] = fused[-1].then(step)
        else:
            fused.append(step)
    return fused[0] if len(fused) == 1 else Sequential(fused)


class BaseOperation(zdx.Module):
    """Extension contract for callable array-to-array numerical operations.

    Implement ``apply(array, **context)``. Operations store their numerical
    parameters and accept the incident array explicitly on every call.
    """

    @abstractmethod
    def apply(self, array, **context):
        """Return the transformed array, preserving compatible leading axes."""

    def __call__(self, array, **context):
        """Apply the numerical operation."""
        return self.apply(array, **context)


class Resize(BaseOperation):
    """Centrally resize the final field axes to fixed physical-order counts.

    Parameters
    ----------
    n : int or tuple[int, ...]
        Positive output counts in physical-axis order.
    """

    n: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, n, *, alias=None):
        """Initialise a fixed central field resize.

        Parameters
        ----------
        n : int or tuple[int, ...]
            Positive static output counts in physical-axis order.
        alias : mapping or sequence of pairs, optional
            Zodiax parameter aliases.
        """
        self.n = as_size(n, name="n")

        self.alias = alias

    def apply(self, field, **context) -> Array:
        """Resize ``field`` while preserving every leading axis."""
        del context
        return _resize_array(field, self.n)


class Chirp(BaseOperation):
    """Apply separable sampled chirps without constructing an N-D mesh.

    Each sampled chirp has shape ``chirp_batch + (n_axis,)`` and chirps follow
    physical-axis order. Leading dimensions align with the same leading prefix of
    the field; remaining leading field axes, such as Jones axes, are left free.

    Parameters
    ----------
    factors : sequence[Array]
        One sampled complex chirp per physical axis.
    """

    factors: tuple[Array, ...]

    def __init__(self, factors, *, alias=None):
        """Initialise separable sampled chirps.

        Parameters
        ----------
        factors : sequence[Array]
            Non-empty chirp vectors with optional leading prefix dimensions.
        alias : mapping or sequence of pairs, optional
            Zodiax parameter aliases.
        """
        self.factors = _prepare_factors(factors)

        self.alias = alias

    @property
    def n(self) -> tuple[int, ...]:
        """Return chirp lengths in physical-axis order."""
        return _factor_sizes(self.factors)

    @classmethod
    def from_axes(cls, axes, quadratic=None, linear=None, constant=None) -> Chirp:
        """Sample ``exp(i * (quadratic*x**2 + linear*x + constant))`` per axis.

        ``axes`` is a tuple of coordinate vectors in physical order. Coefficients
        are shared scalars or broadcast arrays; a tuple supplies one per axis.
        Leading axes broadcast normally during preparation. Use :meth:`broadcast`
        to align wavelength-independent chirps to a larger field batch prefix.
        ``None`` omits a phase term entirely; at least one term is required.
        """
        if quadratic is None and linear is None and constant is None:
            raise ValueError("Provide at least one phase coefficient.")

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
        """Align preparation batches to a complete field prefix without copying."""
        batch_shape = tuple(batch_shape)
        factors = []
        for factor in self.factors:
            if np.broadcast_shapes(factor.shape[:-1], batch_shape) != batch_shape:
                raise ValueError("chirp batches must broadcast to batch_shape.")
            extra = len(batch_shape) - (factor.ndim - 1)
            factors.append(factor.reshape((1,) * extra + factor.shape))
        return Chirp(tuple(factors), alias=self.alias)

    def then(self, other) -> Chirp:
        """Fuse matching chirps using the same left-aligned batches as application."""
        if not isinstance(other, Chirp) or other.n != self.n:
            raise ValueError("other must be a Chirp with matching sample counts.")
        factors = []
        for first, second in zip(self.factors, other.factors):
            # Missing batch axes sit after the existing prefix, before samples.
            rank = max(first.ndim, second.ndim)
            first_shape = first.shape[:-1] + (1,) * (rank - first.ndim)
            second_shape = second.shape[:-1] + (1,) * (rank - second.ndim)
            first = first.reshape(first_shape + first.shape[-1:])
            second = second.reshape(second_shape + second.shape[-1:])
            factors.append(first * second)
        return Chirp(tuple(factors), alias=self.alias)

    def apply(self, field, **context) -> Array:
        """Apply physical-order factors using left-aligned prefix batching."""
        factors = self.factors
        field = np.asarray(field)
        n = _factor_sizes(factors)
        ndim = len(factors)
        if field.shape[-ndim:] != n[::-1]:
            raise ValueError("field spatial shape must match the separable factors.")

        # Align each factor batch to a leading field prefix and its spatial axis
        leading = field.ndim - ndim
        for physical_axis, factor in enumerate(factors):
            q = factor.ndim - 1
            if q > leading:
                raise ValueError("factor batch rank exceeds the field leading rank.")
            target = field.shape[:q]
            if lax.broadcast_shapes(target, factor.shape[:-1]) != target:
                raise ValueError("factor batches may not expand the field prefix.")
            factor = np.broadcast_to(factor, target + (factor.shape[-1],))

            shape = [1] * field.ndim
            shape[:q] = target
            shape[field.ndim - physical_axis - 1] = factor.shape[-1]
            field = field * factor.reshape(shape)
        return field
