"""Separable matrix Fourier kernels, preparation and contraction."""

from __future__ import annotations

import jax.numpy as np
from jax import Array, core, lax
from .operations import BaseOperation

__all__ = [
    "MFT",
    "mft_kernel",
    "mft_kernels",
    "mft_phase_parameter",
    "mft_sampling_parameter",
]


def _as_arrays(values, name, ndim) -> tuple[Array, ...]:
    """Convert a bare array or physical-axis tuple into validated arrays."""
    values = values if isinstance(values, tuple) else (values,)
    arrays = tuple(np.asarray(value) for value in values)
    if not arrays or any(array.ndim < ndim for array in arrays):
        suffix = "dimension" if ndim == 1 else "dimensions"
        raise ValueError(f"{name} must contain arrays with at least {ndim} {suffix}.")
    return arrays


def _per_axis(value, ndim, name):
    """Expand a shared value or validate one value per physical axis."""
    if not isinstance(value, tuple):
        return (value,) * ndim
    if len(value) != ndim:
        raise ValueError(f"{name} must contain {ndim} values.")
    return value


def _prefix_matmul(left, right) -> Array:
    """Multiply matrices whose leading batches align from the left."""
    left = np.asarray(left)
    right = np.asarray(right)
    ndim = max(left.ndim, right.ndim) - 2
    left_batch = left.shape[:-2] + (1,) * (ndim - left.ndim + 2)
    right_batch = right.shape[:-2] + (1,) * (ndim - right.ndim + 2)
    lax.broadcast_shapes(left_batch, right_batch)

    left = left.reshape(left_batch + left.shape[-2:])
    right = right.reshape(right_batch + right.shape[-2:])
    return np.matmul(left, right)


def _prefix_factor(kernel, factor, *, output) -> Array:
    """Fuse a vector into one kernel using left-aligned prefix batches."""
    kernel = np.asarray(kernel)
    factor = np.asarray(factor)
    if factor.ndim < 1:
        raise ValueError("factors must have at least one dimension.")

    size = kernel.shape[-2 if output else -1]
    if factor.shape[-1] != size:
        plane = "output" if output else "input"
        raise ValueError(f"{plane} factor length must match the kernel {plane} size.")

    ndim = max(kernel.ndim - 2, factor.ndim - 1)
    kernel_batch = kernel.shape[:-2] + (1,) * (ndim - kernel.ndim + 2)
    factor_batch = factor.shape[:-1] + (1,) * (ndim - factor.ndim + 1)
    lax.broadcast_shapes(kernel_batch, factor_batch)

    kernel = kernel.reshape(kernel_batch + kernel.shape[-2:])
    factor = factor.reshape(factor_batch + factor.shape[-1:])
    return factor[..., :, None] * kernel if output else kernel * factor[..., None, :]


def _prefix_scale(kernel, factor) -> Array:
    """Fuse a scalar factor into one kernel using left-aligned batches."""
    kernel = np.asarray(kernel)
    factor = np.asarray(factor)
    ndim = max(kernel.ndim - 2, factor.ndim)
    kernel_batch = kernel.shape[:-2] + (1,) * (ndim - kernel.ndim + 2)
    factor_batch = factor.shape + (1,) * (ndim - factor.ndim)
    lax.broadcast_shapes(kernel_batch, factor_batch)

    kernel = kernel.reshape(kernel_batch + kernel.shape[-2:])
    factor = factor.reshape(factor_batch + (1, 1))
    return factor * kernel


def mft_kernel(x_in, x_out, alpha, weight=1.0) -> Array:
    """Construct one matrix Fourier-transform kernel.

    The kernel is defined by

    ``K[j, k] = weight * exp(1j * alpha * x_out[j] * x_in[k])``.

    Parameters
    ----------
    x_in, x_out : Array
        Input and output coordinate vectors with shapes ``batch + (n_in,)`` and
        ``batch + (n_out,)``. Leading dimensions use ordinary JAX broadcasting.
    alpha : ArrayLike
        Phase coefficient, optionally carrying broadcast leading dimensions.
    weight : ArrayLike
        Input quadrature weight. It defaults to one and may carry broadcast leading
        dimensions.

    Returns
    -------
    kernel : Array
        Complex kernel with shape ``batch + (n_out, n_in)``.
    """
    x_in = np.asarray(x_in)
    x_out = np.asarray(x_out)
    if x_in.ndim < 1 or x_out.ndim < 1:
        raise ValueError("coordinate axes must have at least one dimension.")
    if x_in.shape[-1] < 1 or x_out.shape[-1] < 1:
        raise ValueError("coordinate axes must contain at least one sample.")

    # Broadcast only through the outer product; do not tile coordinate vectors.
    alpha, weight = np.asarray(alpha), np.asarray(weight)
    phase = alpha[..., None, None] * x_out[..., :, None] * x_in[..., None, :]
    return weight[..., None, None] * np.exp(1j * phase)


def mft_kernels(axes_in, axes_out, alpha, weight=1.0) -> tuple[Array, ...]:
    """Construct separable MFT kernels for one or more physical axes.

    Parameters
    ----------
    axes_in, axes_out : Array or tuple[Array, ...]
        A bare array defines a one-dimensional transform. Tuples define multiple
        axes in physical ``(x, y, z, ...)`` order. Every coordinate vector may carry
        broadcast leading dimensions.
    alpha : ArrayLike or tuple[ArrayLike, ...]
        Shared phase coefficient or one coefficient per physical axis.
    weight : ArrayLike or tuple[ArrayLike, ...]
        Shared input quadrature weight or one weight per physical axis.

    Returns
    -------
    kernels : tuple[Array, ...]
        Kernels in physical-axis order. Each ends in ``(n_out, n_in)`` and carries
        the broadcast prefix of its coordinates, coefficient, and weight.
    """
    axes_in = _as_arrays(axes_in, "axes_in", 1)
    axes_out = _as_arrays(axes_out, "axes_out", 1)
    if len(axes_in) != len(axes_out):
        raise ValueError("axes_in and axes_out must contain the same number of axes.")

    alpha = _per_axis(alpha, len(axes_in), "alpha")
    weight = _per_axis(weight, len(axes_in), "weight")
    return tuple(
        mft_kernel(x_in, x_out, coefficient, quadrature)
        for x_in, x_out, coefficient, quadrature in zip(
            axes_in,
            axes_out,
            alpha,
            weight,
        )
    )


def mft_phase_parameter(x_in, x_out, alpha, eps=1e-30) -> tuple[Array, Array]:
    """Return input and output Nyquist margins for one MFT kernel.

    ``p_in`` measures the largest phase step along the input coordinates and
    ``p_out`` measures the corresponding output-coordinate step. Values greater
    than or equal to one are Nyquist sampled. Leading coordinate and coefficient
    dimensions use ordinary JAX broadcasting.

    Parameters
    ----------
    x_in, x_out : Array
        Regular coordinate vectors ending in at least two samples.
    alpha : ArrayLike
        Phase coefficient in ``exp(1j * alpha * x_out * x_in)``.
    eps : float
        Non-negative denominator stabiliser used away from exactly flat phases.

    Returns
    -------
    p_in, p_out : Array
        Broadcast input and output phase-sampling margins.
    """
    eps = np.asarray(eps)
    if eps.ndim != 0 or np.issubdtype(eps.dtype, np.complexfloating):
        raise ValueError("eps must be a real scalar.")
    if not isinstance(eps, core.Tracer) and (not np.isfinite(eps) or eps < 0):
        raise ValueError("eps must be finite and non-negative.")

    x_in = np.asarray(x_in)
    x_out = np.asarray(x_out)
    if x_in.ndim < 1 or x_out.ndim < 1:
        raise ValueError("coordinate axes must have at least one dimension.")
    if x_in.shape[-1] < 2 or x_out.shape[-1] < 2:
        raise ValueError("sampling parameters require at least two samples per axis.")

    alpha = np.asarray(alpha)

    dx_in = np.abs(x_in[..., 1] - x_in[..., 0])
    dx_out = np.abs(x_out[..., 1] - x_out[..., 0])
    xmax_in = np.max(np.abs(x_in), axis=-1)
    xmax_out = np.max(np.abs(x_out), axis=-1)

    dphi_in = np.abs(alpha) * xmax_out * dx_in
    dphi_out = np.abs(alpha) * xmax_in * dx_out
    p_in = np.where(dphi_in > 0, np.pi / (dphi_in + eps), np.inf)
    p_out = np.where(dphi_out > 0, np.pi / (dphi_out + eps), np.inf)
    return p_in, p_out


def mft_sampling_parameter(x_in, x_out, alpha, eps=1e-30) -> Array:
    """Return the minimum input/output Nyquist margin for one MFT kernel."""
    p_in, p_out = mft_phase_parameter(x_in, x_out, alpha, eps)
    return np.minimum(p_in, p_out)


class MFT(BaseOperation):
    """A prepared separable matrix Fourier transform.

    Kernels follow physical ``(x, y, z, ...)`` order and each has shape
    ``kernel_batch + (n_out, n_in)``. A kernel batch aligns with the same leading
    prefix of the field, allowing wavelength-dependent kernels to act on fields whose
    later leading axes contain polarisation or other independent structure. Kernel
    arrays are the only leaves; sizes and field shapes are derived from them.

    Parameters
    ----------
    kernels : Array or tuple[Array, ...]
        A bare array defines a one-dimensional transform. A non-empty tuple defines
        a separable transform in physical-axis order. Every kernel must end in
        positive ``(n_out, n_in)`` dimensions.

    Examples
    --------
    Construct and apply a two-dimensional transform from coordinate axes:

    ```python
    transform = MFT.from_axes((x_in, y_in), (x_out, y_out), alpha=-2 * np.pi)
    output = transform(field)
    ```
    """

    kernels: tuple[Array, ...]

    def __init__(self, kernels, *, alias=None):
        """Initialise a prepared separable transform.

        Parameters
        ----------
        kernels : Array or tuple[Array, ...]
            One kernel per physical axis. Leading dimensions use the field-prefix
            batching contract and final dimensions are ``(n_out, n_in)``.
        alias : mapping or sequence of pairs, optional
            Zodiax parameter aliases.
        """
        kernels = _as_arrays(kernels, "kernels", 2)
        if any(0 in kernel.shape[-2:] for kernel in kernels):
            raise ValueError("kernel input and output dimensions must be positive.")
        self.kernels = kernels

        self.alias = alias

    @property
    def ndim(self) -> int:
        """Return the number of transformed physical axes."""
        return len(self.kernels)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Broadcast preparation shape of the stored kernel prefixes."""
        return np.broadcast_shapes(*(kernel.shape[:-2] for kernel in self.kernels))

    def broadcast(self, batch_shape) -> MFT:
        """Align preparation batches to a common field prefix without copying.

        Missing leading batch axes become singleton dimensions. Existing axes
        must broadcast to ``batch_shape`` by ordinary JAX rules. Field application
        then aligns this complete prefix from the left, before polarisation axes.
        """
        batch_shape = tuple(batch_shape)
        kernels = []
        for kernel in self.kernels:
            if np.broadcast_shapes(kernel.shape[:-2], batch_shape) != batch_shape:
                raise ValueError("kernel batches must broadcast to batch_shape.")
            extra = len(batch_shape) - (kernel.ndim - 2)
            kernels.append(kernel.reshape((1,) * extra + kernel.shape))
        return MFT(tuple(kernels), alias=self.alias)

    @property
    def n_in(self) -> tuple[int, ...]:
        """Return input counts in physical-axis order."""
        return tuple(kernel.shape[-1] for kernel in self.kernels)

    @property
    def n_out(self) -> tuple[int, ...]:
        """Return output counts in physical-axis order."""
        return tuple(kernel.shape[-2] for kernel in self.kernels)

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Return the final input field shape in array-axis order."""
        return self.n_in[::-1]

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Return the final output field shape in array-axis order."""
        return self.n_out[::-1]

    def apply(self, field, **context) -> Array:
        """Apply physical-order separable kernels to reversed final field axes.

        Kernels end in (n_out, n_in). Their batch axes align with the existing
        leading field prefix; they may not add field axes. Spatial rank is arbitrary.
        """
        kernels = tuple(self._squeeze_batch(k) for k in self.kernels)
        ndim = len(kernels)
        n_in = tuple(kernel.shape[-1] for kernel in kernels)
        n_out = tuple(kernel.shape[-2] for kernel in kernels)
        field = np.asarray(field)
        if field.ndim < ndim or field.shape[-ndim:] != n_in[::-1]:
            raise ValueError("field spatial shape must match the MFT input shape.")

        # Promote once because dot_general requires identical operand dtypes
        dtype = np.result_type(field, *kernels)
        field = field.astype(dtype)
        kernels = tuple(kernel.astype(dtype) for kernel in kernels)

        # Contract shrinking axes first and expanding axes last to minimise MACs
        order = sorted(
            range(ndim),
            key=lambda i: 1 / n_in[i] - 1 / n_out[i],
        )
        wanted = list(range(ndim - 1, -1, -1))
        labels = wanted.copy()
        leading = field.ndim - ndim

        for physical_axis in order:
            # Pair kernel batches with the existing leading field prefix.
            kernel = kernels[physical_axis]
            batch_ndim = kernel.ndim - 2
            if batch_ndim > leading:
                raise ValueError("kernel batch rank exceeds the field leading rank.")

            target = field.shape[:batch_ndim]
            if lax.broadcast_shapes(target, kernel.shape[:-2]) != target:
                raise ValueError("kernel batches may not expand the field prefix.")
            kernel = np.broadcast_to(kernel, target + kernel.shape[-2:])

            # Contract one physical axis and track its new array-axis position.
            field_axis = leading + labels.index(physical_axis)
            batch_axes = tuple(range(batch_ndim))
            dimensions = (((field_axis,), (batch_ndim + 1,)), (batch_axes, batch_axes))
            field = lax.dot_general(field, kernel, dimensions)

            labels.remove(physical_axis)
            labels.append(physical_axis)

        # Restore reversed physical-axis order with at most one final transpose
        permutation = list(range(leading))
        permutation += [leading + labels.index(axis) for axis in wanted]
        if permutation != list(range(field.ndim)):
            field = field.transpose(permutation)
        return field

    def then(self, other) -> MFT:
        """Compose this transform with a compatible following MFT."""
        if not isinstance(other, MFT):
            raise TypeError("other must be an MFT.")
        if self.n_out != other.n_in:
            raise ValueError("this output counts must match other input counts.")
        kernels = tuple(
            _prefix_matmul(second, first)
            for first, second in zip(self.kernels, other.kernels)
        )
        return MFT(kernels, alias=self.alias)

    def fuse_input(self, factors) -> MFT:
        """Return a transform with separable input factors fused into its kernels.

        ``factors`` is a bare vector for a one-dimensional transform or a tuple in
        physical-axis order. Each factor ends in the matching input count and its
        batch aligns from the left with the corresponding kernel batch.
        """
        factors = _as_arrays(getattr(factors, "factors", factors), "factors", 1)
        if len(factors) != self.ndim:
            raise ValueError(f"factors must contain {self.ndim} arrays.")
        kernels = tuple(
            _prefix_factor(kernel, factor, output=False)
            for kernel, factor in zip(self.kernels, factors)
        )
        return MFT(kernels, alias=self.alias)

    def fuse_output(self, factors) -> MFT:
        """Return a transform with separable output factors fused into its kernels.

        ``factors`` is a bare vector for a one-dimensional transform or a tuple in
        physical-axis order. Each factor ends in the matching output count and its
        batch aligns from the left with the corresponding kernel batch.
        """
        factors = _as_arrays(getattr(factors, "factors", factors), "factors", 1)
        if len(factors) != self.ndim:
            raise ValueError(f"factors must contain {self.ndim} arrays.")
        kernels = tuple(
            _prefix_factor(kernel, factor, output=True)
            for kernel, factor in zip(self.kernels, factors)
        )
        return MFT(kernels, alias=self.alias)

    def fuse_scale(self, factor) -> MFT:
        """Return a transform with one overall prefix-batched factor fused in."""
        kernels = (_prefix_scale(self.kernels[0], factor), *self.kernels[1:])
        return MFT(kernels, alias=self.alias)

    @classmethod
    def from_axes(cls, axes_in, axes_out, alpha, weight=1.0) -> MFT:
        """Construct a prepared transform from explicit coordinate axes.

        Parameters
        ----------
        axes_in, axes_out : tuple[Array, ...]
            Input and output coordinate vectors in physical-axis order. Each vector
            ends in its sample count; leading dimensions broadcast during kernel
            construction. Both tuples must contain the same number of axes.
        alpha : Array or tuple[Array, ...]
            Phase coefficient in ``exp(1j * alpha * x_out * x_in)``. A tuple sets
            one coefficient per axis; a shared scalar or array applies to all axes.
            Units must cancel the product of the two coordinate units.
        weight : Array or tuple[Array, ...]
            Per-axis kernel multiplier, defaulting to one. A tuple sets independent
            axis weights; shared arrays broadcast with the coordinate batches.

        Returns
        -------
        MFT
            Callable transform storing the constructed kernel arrays. Coefficients
            remain differentiable through construction.
        """
        return cls(mft_kernels(axes_in, axes_out, alpha, weight))

    @classmethod
    def from_grids(cls, grid_in, grid_out, alpha, weight=1.0) -> MFT:
        """Construct separable kernels on regular Cartesian sampling.

        Parameters
        ----------
        grid_in, grid_out : GridSpec
            Complete grids with matching dimensionality and indexing='xy'. Their
            axes supply SI-valued coordinate vectors in physical-axis order.
        alpha : Array or tuple[Array, ...]
            Coefficient of ``x_out * x_in`` in the complex exponential, with units
            cancelling those of the coordinates. Arrays broadcast with grid batches;
            a tuple supplies a different coefficient for each physical axis.
        weight : Array or tuple[Array, ...]
            Kernel multiplier per axis, defaulting to one. A tuple supplies separate
            weights; shared scalar or array weights apply to each physical axis.

        Returns
        -------
        MFT
            Callable transform with coordinate phases and weights stored in kernels.
        """
        if grid_in.indexing != "xy" or grid_out.indexing != "xy":
            raise ValueError("MFT grids must use indexing='xy'.")
        return cls.from_axes(grid_in.axes, grid_out.axes, alpha, weight)

    @staticmethod
    def _squeeze_batch(kernel):
        """Remove trailing singleton prefix axes that inhibit shared GEMMs."""
        while kernel.ndim > 2 and kernel.shape[-3] == 1:
            kernel = np.squeeze(kernel, -3)
        return kernel
