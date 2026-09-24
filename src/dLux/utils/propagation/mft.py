"""Separable matrix Fourier kernels, preparation and contraction."""

from __future__ import annotations

import jax.numpy as np
from jax import Array, core
from ..batching import broadcast_batch
from ..sequential import BaseLayer

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
    # Pair input and output axes, expanding shared kernel coefficients per axis.
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
    # Require a scalar tolerance and coordinate vectors long enough to infer spacing.
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

    # Measure the largest phase step in each direction and convert to Nyquist margins.
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


class MFT(BaseLayer):
    """A prepared two-dimensional separable matrix Fourier transform.

    Kernels follow physical ``(x, y)`` order and each has shape
    ``kernel_batch + (n_out, n_in)``. Kernel and field batches use ordinary JAX
    broadcasting, including expansion through explicit singleton axes. Kernel arrays
    are the only leaves; sizes and field shapes are derived from them. The final two
    field axes are spatial; all preceding axes are numerical batches.

    Examples
    --------
    ```python
    import dLux as dl
    import jax.numpy as np

    # Supply physical-order axes and an explicit Fourier phase coefficient.
    axis = np.linspace(-0.5, 0.5, 8)
    operation = dl.MFT.from_axes((axis, axis), (axis, axis), alpha=-2 * np.pi)
    output = operation(np.ones((8, 8), dtype=complex))
    print(output.shape)
    ```
    """

    kernels: tuple[Array, Array]

    def __init__(self, kernels, *, alias=None):
        """Initialise a prepared separable transform.

        Parameters
        ----------
        kernels : tuple[Array, Array]
            Two kernels in physical ``(x, y)`` order. Leading dimensions broadcast
            with the field batch and final dimensions are ``(n_out, n_in)``.
        alias : mapping or sequence of pairs, optional
            Zodiax parameter aliases.
        """
        kernels = _as_arrays(kernels, "kernels", 2)
        if len(kernels) != 2:
            raise ValueError("kernels must contain two arrays in (x, y) order.")
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
        """Return the ordinary broadcast shape of the kernel batches."""
        batches = [kernel.shape[:-2] for kernel in self.kernels]
        return np.broadcast_shapes(*batches)

    def broadcast(self, batch_shape) -> MFT:
        """Broadcast both kernels to an explicit leading batch shape."""
        kernels = tuple(broadcast_batch(x, batch_shape, 2) for x in self.kernels)
        return MFT(kernels, alias=self.alias)

    @property
    def n_in(self) -> tuple[int, int]:
        """Return input counts in physical-axis order."""
        return tuple(kernel.shape[-1] for kernel in self.kernels)

    @property
    def n_out(self) -> tuple[int, int]:
        """Return output counts in physical-axis order."""
        return tuple(kernel.shape[-2] for kernel in self.kernels)

    @property
    def input_shape(self) -> tuple[int, int]:
        """Return the final input field shape in array-axis order."""
        return self.n_in[::-1]

    @property
    def output_shape(self) -> tuple[int, int]:
        """Return the final output field shape in array-axis order."""
        return self.n_out[::-1]

    def apply(self, field, **context) -> Array:
        """Apply physical-order separable kernels to reversed final field axes.

        Kernels end in (n_out, n_in); the field ends in (ny, nx). All leading
        dimensions broadcast normally. For example, operator batch (B, 1) and
        field batch (1, C) produce a (B, C) batch of transformed fields.
        """
        field = np.asarray(field)
        if field.ndim < 2 or field.shape[-2:] != self.input_shape:
            raise ValueError("field spatial shape must match the MFT input shape.")

        dtype = np.result_type(field, *self.kernels)
        field = field.astype(dtype)
        kernel_x, kernel_y = (kernel.astype(dtype) for kernel in self.kernels)
        kernel_x = np.swapaxes(kernel_x, -1, -2)

        # Evaluate Ky @ field @ Kx.T, contracting the smaller intermediate first.
        # The static counts choose the order; matmul handles shared/paired batches.
        nx_in, ny_in = self.n_in
        nx_out, ny_out = self.n_out
        if 1 / nx_in - 1 / nx_out <= 1 / ny_in - 1 / ny_out:
            return kernel_y @ (field @ kernel_x)
        return (kernel_y @ field) @ kernel_x

    def then(self, other) -> MFT:
        """Compose this transform with a compatible following MFT."""
        if not isinstance(other, MFT):
            raise TypeError("other must be an MFT.")
        if self.n_out != other.n_in:
            raise ValueError("this output counts must match other input counts.")
        kernels = tuple(
            second @ first for first, second in zip(self.kernels, other.kernels)
        )
        return MFT(kernels, alias=self.alias)

    def fuse_input(self, factors) -> MFT:
        """Return a transform with separable input factors fused into its kernels.

        ``factors`` is a tuple in physical ``(x, y)`` order. Each factor ends in the
        matching input count; its batch broadcasts with the kernel batch.
        """
        factors = _as_arrays(getattr(factors, "factors", factors), "factors", 1)
        if len(factors) != self.ndim:
            raise ValueError(f"factors must contain {self.ndim} arrays.")
        kernels = tuple(
            self._fuse_factor(kernel, factor, output=False)
            for kernel, factor in zip(self.kernels, factors)
        )
        return MFT(kernels, alias=self.alias)

    def fuse_output(self, factors) -> MFT:
        """Return a transform with separable output factors fused into its kernels.

        ``factors`` is a tuple in physical ``(x, y)`` order. Each factor ends in the
        matching output count; its batch broadcasts with the kernel batch.
        """
        factors = _as_arrays(getattr(factors, "factors", factors), "factors", 1)
        if len(factors) != self.ndim:
            raise ValueError(f"factors must contain {self.ndim} arrays.")
        kernels = tuple(
            self._fuse_factor(kernel, factor, output=True)
            for kernel, factor in zip(self.kernels, factors)
        )
        return MFT(kernels, alias=self.alias)

    def fuse_scale(self, factor) -> MFT:
        """Return a transform with a broadcast overall scale fused into one kernel."""
        factor = np.asarray(factor)
        kernels = (self.kernels[0] * factor[..., None, None], *self.kernels[1:])
        return MFT(kernels, alias=self.alias)

    @classmethod
    def from_axes(cls, axes_in, axes_out, alpha, weight=1.0) -> MFT:
        """Construct a prepared transform from explicit coordinate axes.

        Parameters
        ----------
        axes_in, axes_out : tuple[Array, Array]
            Input and output coordinate vectors in physical ``(x, y)`` order. Each
            vector ends in its sample count; leading dimensions broadcast during
            kernel construction. Both tuples must contain two axes.
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
        grid_in, grid_out : Grid
            Complete two-dimensional grids with indexing='xy'. Their
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
        return cls.from_axes(grid_in.axes, grid_out.axes, alpha, weight)

    @staticmethod
    def _fuse_factor(kernel, factor, *, output) -> Array:
        """Fuse one sampled vector into a kernel with ordinary batch broadcasting."""
        kernel = np.asarray(kernel)
        factor = np.asarray(factor)
        size = kernel.shape[-2 if output else -1]
        if factor.shape[-1] != size:
            plane = "output" if output else "input"
            raise ValueError(
                f"{plane} factor length must match the kernel {plane} size."
            )

        if output:
            return factor[..., :, None] * kernel
        return kernel * factor[..., None, :]
