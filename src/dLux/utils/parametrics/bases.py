"""Explicit, generated, and compact optical basis expansions."""

from __future__ import annotations

from abc import abstractmethod
from operator import index
from typing import Any

import equinox as eqx
import jax
import jax.numpy as np
import zodiax as zdx
from jax import Array

import dLux.utils as dlu

from ..grids import PasteSpec
from .parametrics import Parametric, SpatialExpression, _spatial_context

__all__ = [
    "ParametricBasis",
    "Basis",
    "PastedBasis",
    "ImplicitBasis",
    "SpatialBasis",
    "CLIMBBasis",
    "FourierBasis",
]


def _coefficients(coeffs, shape):
    """Initialise coefficients, deferring expression shapes until evaluation."""
    coeffs = np.zeros(shape) if coeffs is None else zdx.as_array(coeffs)
    if isinstance(coeffs, Array) and coeffs.shape[-len(shape) :] != tuple(shape):
        raise ValueError(f"coeffs must end in the shape {shape}.")
    return coeffs


class ParametricBasis(Parametric):
    """A physical expansion with stored coefficients and a coefficient solve.

    Concrete families implement ``evaluate`` and ``solve`` for their representation.
    Evaluation resolves nested coefficient expressions. Solving returns physical
    coefficients without reversing any transform stored in ``coeffs``.
    """

    coeffs: Array | zdx.Expression | None

    @abstractmethod
    def solve(self, samples, **context):
        """Recover physical coefficients; each family defines its solve contract."""


class Basis(ParametricBasis):
    """Expand coefficients over an explicit or context-generated sampled basis.

    ``basis`` has leading coefficient axes followed by arbitrary output axes.
    ``coeffs`` ends in the same coefficient shape; preceding axes are independent
    batches. Use ``jax.vmap`` for paired basis and coefficient batches. Evaluation
    does not normalise basis vectors or their output. Local ``solve`` returns a
    minimum-norm least-squares representative, including for rank-deficient bases.
    ``solve`` returns physical coefficients, leaving stored transforms unchanged.

    Coefficients are stored directly in ``coeffs``.
    Either the coefficients or basis can be a Zodiax expression. Numerical leaves
    remain differentiable through evaluation, resolution, and the linear solve.

    Examples
    --------
    ```python
    import dLux as dl
    import jax.numpy as jnp

    basis = dl.Basis(jnp.eye(3), coeffs=[1., 2., 3.])
    values = basis.resolve()
    coefficients = basis.solve(values)
    ```
    """

    coeffs: Array | zdx.Expression | None
    basis: Array | zdx.Expression
    axes: int = eqx.field(static=True)

    def __init__(self, basis, coeffs=None, *, axes=1, alias=None):
        """Initialise a sampled linear expansion.

        Parameters
        ----------
        basis : Array or zodiax.Expression
            Non-empty tensor whose first ``axes`` dimensions are coefficient axes.
            Remaining dimensions are output axes, with no assumed physical units.
        coeffs : Array, zodiax.Expression, or None
            Coefficients ending in the leading basis shape. Leading coefficient
            batches are retained. None gives zero coefficients, initialised at
            construction for concrete vectors or when generated vectors resolve.
        axes : int
            Positive number of coefficient axes, fixed at construction; default one.
        alias : mapping or None
            Optional parameter aliases.
        """
        if isinstance(axes, bool):
            raise TypeError("axes must be a positive integer.")
        self.axes = index(axes)
        if self.axes < 1:
            raise ValueError("axes must be a positive integer.")
        self.basis = zdx.as_array(basis)
        self.coeffs = zdx.as_array(coeffs)
        self.alias = alias

        # Check sampled arrays now. A generator declares its context in evaluate;
        # its output dimensions are checked by the contraction after resolution.
        if isinstance(self.basis, Array):
            if self.basis.ndim < self.axes or 0 in self.basis.shape:
                raise ValueError("basis must contain non-empty coefficient axes.")
            if self.coeffs is None:
                self.coeffs = np.zeros(self.basis.shape[: self.axes])
            if isinstance(self.coeffs, Array):
                basis_shape = self.basis.shape[: self.axes]
                coeff_shape = self.coeffs.shape[-self.axes :]
                if coeff_shape != basis_shape:
                    raise ValueError(f"coeffs must end in the shape {basis_shape}.")

    def evaluate(self, **context: Any) -> Array:
        """Expand stored coefficients, retaining independent leading batches."""
        basis = zdx.resolve(self.basis, **context)
        coeffs = zdx.resolve(self.coeffs, **context)
        if coeffs is None:
            coeffs = np.zeros(basis.shape[: self.axes])
        return dlu.eval_basis(basis, coeffs, self.axes)

    def solve(self, y: Array, **context: Any) -> Array:
        """Resolve the stored modes and recover physical coefficients.

        The sampled output y has independent leading batch axes followed by the
        basis output axes. The result has those batch axes followed by coefficient
        axes. This is a minimum-norm least-squares representative, including for
        rank-deficient modes. It does not reverse transforms stored in coeffs.
        Context is used where mode generation or stored modes require it.
        """
        basis = zdx.resolve(self.basis, **context)
        return dlu.solve_basis(y, basis, self.axes)


class PastedBasis(ParametricBasis):
    """Parameterise compact local bases pasted into a common output grid.

    This representation is normally produced by segmented aperture builders. It is
    public so specialised builders can retain dense local modes and avoid storing a
    global mostly-zero basis.

    Parameters
    ----------
    basis : Array
        Basis stamps with shape ``(n_stamps, *basis_shape, ny, nx)``.
    spec : PasteSpec
        Placement and sampling contract shared by every compact stamp.
    coeffs : Array, zodiax.Expression, or None
        Per-stamp coefficients with shape ``(n_stamps, *basis_shape)``. They default
        to zero.
    method : str
        ``"scan"`` for bounded-memory placement or ``"scatter"`` for parallel
        placement using flattened pixel indices.
    """

    coeffs: Array | zdx.Expression | None
    shape: tuple[int, ...] = eqx.field(static=True)
    basis: Array
    spec: PasteSpec
    method: str

    def __init__(
        self,
        basis,
        spec,
        coeffs=None,
        method="scan",
        *,
        alias=None,
    ):
        """Initialise compact basis stamps and their placement.

        Parameters
        ----------
        basis : Array
            Per-stamp basis values with final two local spatial axes.
        spec : PasteSpec
            Fixed placement geometry for the compact stamps.
        coeffs : Array, zodiax.Expression, or None
            Per-stamp coefficients of shape ``(n_stamps, *basis_shape)``.
        method : str
            ``"scan"`` for lower memory or ``"scatter"`` for parallel placement.
        alias : mapping or None
            Optional parameter aliases.
        """
        if not isinstance(spec, PasteSpec):
            raise TypeError("spec must be a PasteSpec.")

        basis = zdx.as_array(basis, dtype=float)
        if basis.ndim < 4:
            raise ValueError("basis must have shape (n_stamps, ..., ny, nx).")
        if basis.shape[0] != len(spec.starts):
            raise ValueError("basis and PasteSpec must contain the same stamp count.")
        if basis.shape[-2:] != spec.shape[::-1]:
            raise ValueError("basis spatial shape must match the PasteSpec stamps.")

        method = str(method).lower()
        if method not in ("scan", "scatter"):
            raise ValueError("method must be either 'scan' or 'scatter'.")

        shape = basis.shape[:-2]
        coeffs = np.zeros(shape) if coeffs is None else coeffs
        self.coeffs = _coefficients(coeffs, shape)
        self.shape = shape
        self.basis = basis
        self.spec = spec
        self.method = method

        self.alias = alias

    def evaluate(self, **context: Any) -> Array:
        """Expand per-stamp coefficients and add overlapping stamps globally."""
        coeffs = zdx.resolve(self.coeffs, **context)
        if coeffs.shape != self.shape:
            raise ValueError(f"coeffs must have shape {self.shape}.")
        expand = lambda b, c: dlu.eval_basis(b, c, axes=len(self.shape) - 1)
        local = jax.vmap(expand)(self.basis, coeffs)
        return self.spec.paste(local, self.method)

    def solve(self, value: Array, **kwargs: Any) -> Array:
        """Solve coefficients independently from every extracted local stamp.

        ``value`` must end with the global output shape. Returned coefficients have
        shape ``(n_stamps, *basis_shape)``. This is a local projection, not a
        global inverse: overlapping stamps contain contributions from their
        neighbours. This method is not a global least-squares solve.
        """
        values = self.spec.extract(value)
        solve = lambda value, basis: dlu.solve_basis(value, basis, len(self.shape) - 1)
        return jax.vmap(solve)(values, self.basis)


class ImplicitBasis(ParametricBasis):
    """Extension point for linear bases generated from runtime context.

    Subclasses store ``coeffs`` and their native contraction ``shape``,
    then implement ``evaluate`` and ``calculate_basis`` with explicit inputs.
    Basis arrays begin with ``shape``;
    remaining axes are sampled outputs. Leading coefficient axes produce independent
    batches, following ``Basis``. Use ``jax.vmap`` for paired coefficient and
    coordinate batches. Local ``solve`` returns minimum-norm least-squares
    coefficients; stored coefficient transforms are not reversed by ``solve``.
    """

    coeffs: Array | zdx.Expression | None
    shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, coeffs, shape, *, alias=None):
        """Initialise the coefficient input for a generated basis.

        Parameters
        ----------
        coeffs : Array, Expression, or None
            Coefficients ending in ``shape``; omission initialises zeros.
        shape : tuple[int, ...]
            Native coefficient axes, fixed at construction.
        alias : mapping or None
            Optional parameter aliases.
        """
        self.shape = tuple(shape)
        if not self.shape or any(n < 1 for n in self.shape):
            raise ValueError("shape must contain positive coefficient dimensions.")
        self.coeffs = _coefficients(coeffs, self.shape)
        self.alias = alias

    @abstractmethod
    def calculate_basis(self, **context: Any) -> Array:
        """Return basis vectors with coefficient axes followed by output axes."""

    def solve(self, y: Array, **context: Any) -> Array:
        """Generate modes and recover physical coefficients.

        The sampled output y has independent leading batch axes followed by the
        basis output axes. The result has those batch axes followed by coefficient
        axes. This is a minimum-norm least-squares representative, including for
        rank-deficient modes. It does not reverse transforms stored in coeffs.
        Context is used where mode generation or stored modes require it.
        """
        basis = self.calculate_basis(**context)
        return dlu.solve_basis(y, basis, len(self.shape))


class SpatialBasis(SpatialExpression, ImplicitBasis):
    """A generated basis evaluated on two-dimensional spatial coordinates.

    Subclasses implement ``evaluate`` and ``calculate_basis`` with their complete
    input signatures. Missing alternative inputs raise Unresolved from evaluation.
    Modes have leading coefficient axes matching ``shape``, followed by sampled
    output axes.
    Evaluation contracts stored coefficients; solving recovers minimum-norm
    physical coefficients. Coefficient batches remain independent of grid batches.

    The shared parent owns coefficient storage, spatial calls and solving.
    ``evaluate`` and ``solve`` accept a grid or coordinates and optional transform;
    ``calculate_basis`` generates the modes; concrete evaluation owns contraction.
    """

    coeffs: Array | zdx.Expression | None
    shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, coeffs, shape, *, alias=None):
        """Store coefficients and the static mode-axis shape.

        Parameters
        ----------
        coeffs : Array, zodiax.Expression, or None
            Coefficients ending in shape; leading axes are independent batches.
            None initialises zeros. Expressions resolve during evaluation.
        shape : tuple[int, ...]
            Non-empty tuple of positive mode-axis lengths.
        alias : mapping or None
            Optional user-defined parameter aliases.
        """
        ImplicitBasis.__init__(self, coeffs, shape, alias=alias)

    def solve(self, samples, *, grid=None, coordinates=None, transform=None, **context):
        """Recover minimum-norm physical coefficients on spatial sampling.

        Parameters
        ----------
        samples : Array
            Sampled output, with independent leading batches before spatial axes.
        grid : GridSpec or None
            Grid used to generate modes, mutually exclusive with coordinates.
        coordinates : Array or None
            Cartesian coordinates in mode-parameter units, shape (2, ny, nx).
        transform : BaseCoordTransform or None
            Mapping applied before generating modes. Source pixel scale is retained.
        **context
            Additional mode-generation inputs, such as an explicit pupil diameter.

        Returns
        -------
        Array
            Minimum-norm least-squares coefficients with sampled-output batch axes
            followed by mode shape. Stored coefficient transforms are not inverted.
            Required spatial inputs must be supplied; solve does not defer itself.
        """
        local = _spatial_context(
            grid, coordinates=coordinates, transform=transform, **context
        )
        return ImplicitBasis.solve(self, samples, **local)


class CLIMBBasis(Parametric):
    """Map a continuous explicit-basis field through CLIMB soft binarisation.

    Coefficients first generate an oversampled latent two-dimensional field. It is
    softly binarised, downsampled by `oversample`, and mapped between two configured
    output values for differentiable binary-mask design. The nonlinear result has
    no coefficient inverse; ``expansion.solve(...)`` only reverses
    the latent linear field.
    """

    expansion: Basis
    values: Array
    oversample: int = eqx.field(static=True)

    def __init__(
        self,
        basis,
        coeffs=None,
        shape=None,
        values=(0.0, 1.0),
        oversample=3,
        *,
        alias=None,
    ):
        """Initialise a CLIMB-softened explicit basis.

        Parameters
        ----------
        basis : Array
            Explicit sampled basis values.
        coeffs : Array, zodiax.Expression, or None
            Basis coefficients, defaulting to zeros.
        shape : tuple[int, ...] or None
            Coefficient-axis shape, inferred from concrete coefficients. Required
            if coefficients are omitted or have no known shape.
        values : tuple[float, float]
            Lower and upper values used by soft binarisation.
        oversample : int
            Internal soft-binarisation oversampling factor.
        alias : mapping or None
            Optional parameter aliases.
        """
        basis = zdx.as_array(basis, dtype=float)
        if coeffs is not None:
            coeffs = zdx.as_array(coeffs, dtype=float)
        if shape is None and isinstance(coeffs, Array):
            shape = coeffs.shape
        if shape is None:
            raise ValueError("shape is required when coeffs has no known shape.")
        shape = tuple(shape)
        self.expansion = Basis(basis, _coefficients(coeffs, shape), axes=len(shape))
        output_shape = basis.shape[len(shape) :]
        if len(output_shape) != 2 or output_shape[0] != output_shape[1]:
            raise ValueError("The CLIMB latent output must be a square 2D array.")
        values = zdx.as_array(values, dtype=float)
        if values.shape != (2,):
            raise ValueError("values must contain exactly two output values.")
        self.values = values
        self.oversample = int(oversample)
        if self.oversample < 1:
            raise ValueError("oversample must be a positive integer.")
        if output_shape[0] % self.oversample != 0:
            raise ValueError(
                "The CLIMB latent output size must be divisible by oversample."
            )

        self.alias = alias

    def evaluate(self, **kwargs: Any) -> Array:
        """Evaluate and softly binarise the latent field.

        The result is downsampled by ``oversample`` and mapped between the configured
        lower and upper values.
        """
        binary = dlu.soft_binarise(
            zdx.resolve(self.expansion, **kwargs), self.oversample
        )
        low, high = zdx.resolve(self.values, **kwargs)
        return low + (high - low) * binary


class FourierBasis(ParametricBasis):
    """A parameterisation over a separable real Fourier basis.

    Coefficients are stored in ``coeffs``. Evaluation returns
    dLux image axes ``(ny, nx)`` while coefficient axes remain in physical ``(x, y)``
    order. The forward operation retains
    compact kernels. Only ``calculate_basis`` and ``solve`` materialise the basis.

    Examples
    --------
    Construct and evaluate the basis without materialising its vectors:

    ```python
    import dLux as dl

    # Construct and evaluate an implicit Fourier basis
    basis = dl.FourierBasis(npix=128, n_modes=8)
    values = basis.evaluate()

    # Evaluate the same basis at a different resolution
    resized = basis.resize(npix=256)
    resized_values = resized.evaluate()
    ```
    """

    coeffs: Array | zdx.Expression | None
    shape: tuple[int, ...] = eqx.field(static=True)
    kernels: tuple[Array, Array]

    def __init__(
        self,
        npix,
        n_modes,
        coeffs=None,
        scale: float = 1.0,
        *,
        alias=None,
    ):
        """Initialise an implicit separable Fourier basis.

        Parameters
        ----------
        npix : int or tuple[int, int]
            Sampled output size.
        n_modes : int or tuple[int, int]
            Fourier-mode counts in physical-axis order.
        coeffs : Array, zodiax.Expression, or None
            Coefficient array matching the generated separable basis shape.
        scale : float
            Coordinate scale used to construct the Fourier kernels.
        alias : mapping or None
            Optional parameter aliases.
        """
        self.kernels = dlu.fourier_kernels(n_modes, npix, scale)
        shape = tuple(kernel.shape[1] for kernel in self.kernels)
        coeffs = np.zeros(shape) if coeffs is None else coeffs
        self.coeffs = _coefficients(coeffs, shape)
        self.shape = shape

        self.alias = alias

    def calculate_basis(self, **kwargs: Any) -> Array:
        """Materialise the complete separable real Fourier basis.

        The result has shape ``(*shape, ny, nx)``, matching dLux image conventions.
        Additional context is accepted but not consumed because the sampling
        kernels are fixed at construction.
        """
        Kx, Ky = self.kernels
        return np.einsum("xi,yj->ijyx", Kx, Ky)

    def evaluate(self, **kwargs: Any) -> Array:
        """Evaluate the separable Fourier expansion without materialising its basis.

        Returns a sampled ``(ny, nx)`` image with leading coefficient batches
        retained, without materialising the full basis tensor.
        """
        coeffs = zdx.resolve(self.coeffs, **kwargs)
        return np.swapaxes(dlu.eval_fourier_basis(coeffs, *self.kernels), -1, -2)

    def solve(self, y: Array, **context: Any) -> Array:
        """Materialise Fourier modes and recover physical coefficients.

        The sampled output y has independent leading batch axes followed by the
        basis output axes. The result has those batch axes followed by coefficient
        axes. This is a minimum-norm least-squares representative, including for
        rank-deficient modes. It does not reverse transforms stored in coeffs.
        Context is used where mode generation or stored modes require it.
        """
        basis = self.calculate_basis()
        return dlu.solve_basis(y, basis, axes=2)

    def resize(self, npix, scale: float = 1.0):
        """Return a copy sampled onto a resized Fourier grid.

        ``npix`` gives physical ``(x, y)`` sample counts and ``scale`` defines the
        dimensionless Fourier coordinate extent. Coefficients are retained while the
        separable kernels are regenerated.
        """
        kernels = dlu.fourier_kernels(self.shape, npix, scale)
        return self.set(kernels=kernels)
