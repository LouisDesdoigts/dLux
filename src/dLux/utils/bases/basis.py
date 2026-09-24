"""Explicit, generated, and compact optical basis expansions."""

from __future__ import annotations

from abc import abstractmethod
from math import prod
from operator import index
from typing import Any

import equinox as eqx
import jax
import jax.numpy as np
import zodiax as zdx
from jax import Array

from ..grids import PasteSpec
from ..numerical.units import Unitful, unit_factor

__all__ = [
    "ParametricBasis",
    "Basis",
    "PastedBasis",
    "initialise_coeffs",
    "eval_basis",
    "solve_basis",
]


def eval_basis(basis: Array, coeffs: Array, axes: int | None = None) -> Array:
    """Contract trailing coefficient axes with leading sampled basis axes.

    Parameters
    ----------
    basis: Array
        The basis to use.
    coeffs: Array
        Coefficients ending in ``basis.shape[:axes]``. Leading axes are independent
        batches; basis axes after the contraction are retained as output axes.
    axes : int or None
        Positive contraction-axis count. ``None`` contracts every coefficient axis,
        preserving the original unbatched utility convention.
    """
    # An explicitly empty expansion waits until coefficients are supplied.
    if coeffs is None:
        raise zdx.Unresolved("Basis coefficients have not been supplied.")

    # Require resolved arrays, then check the coefficient contraction shapes.
    if isinstance(basis, zdx.Expression) or isinstance(coeffs, zdx.Expression):
        raise zdx.Unresolved(
            "Basis modes and coefficients need their required context."
        )
    basis, coeffs = np.asarray(basis), np.asarray(coeffs)
    axes = coeffs.ndim if axes is None else index(axes)
    if axes < 1 or axes > basis.ndim or axes > coeffs.ndim:
        raise ValueError("axes must identify at least one coefficient and basis axis.")
    if basis.shape[:axes] != coeffs.shape[-axes:]:
        raise ValueError(
            "The leading basis dimensions must match the trailing coefficient "
            f"dimensions, received {basis.shape[:axes]} and {coeffs.shape[-axes:]}."
        )

    # Contract coefficient axes while retaining batches and all output axes.
    coeff_axes = tuple(range(coeffs.ndim - axes, coeffs.ndim))
    return np.tensordot(coeffs, basis, axes=(coeff_axes, tuple(range(axes))))


def solve_basis(array: Array, basis: Array, axes: int | None = None) -> Array:
    """Return minimum-norm least-squares coefficients over a sampled basis.

    Parameters
    ----------
    array : Array
        The array to solve for the basis coefficients of.
    basis : Array
        Basis with leading coefficient axes and trailing sampled output axes.
    axes : int or None
        Positive coefficient-axis count. ``None`` treats the complete ``array``
        shape as the output shape. An explicit count preserves leading value batches.

    Returns
    -------
    coeffs : Array
        Leading value batches followed by ``basis.shape[:axes]``. This is a
        representative solution, not an exact inverse for a rank-deficient basis.
    """
    if isinstance(array, zdx.Expression) or isinstance(basis, zdx.Expression):
        raise zdx.Unresolved("Basis modes and samples need their required context.")
    array, basis = np.asarray(array), np.asarray(basis)
    axes = basis.ndim - array.ndim if axes is None else index(axes)
    if axes < 1 or axes > basis.ndim or 0 in basis.shape:
        raise ValueError("axes must identify non-empty leading coefficient axes.")
    coefficient_shape, output_shape = basis.shape[:axes], basis.shape[axes:]
    output_ndim = len(output_shape)
    if output_ndim and array.shape[-output_ndim:] != output_shape:
        raise ValueError(f"array must end in the sampled output shape {output_shape}.")

    # Flatten only the coefficient and sampled output axes for the linear solve.
    batch_shape = array.shape[:-output_ndim] if output_ndim else array.shape
    dtype = np.result_type(array.dtype, basis.dtype, np.float32)
    matrix = basis.reshape((prod(coefficient_shape), prod(output_shape))).astype(dtype)
    values = array.reshape((-1, prod(output_shape))).astype(dtype)

    # The pseudoinverse also gives zero coefficients for a zero-rank basis.
    coeffs = values @ np.linalg.pinv(matrix)
    return coeffs.reshape(batch_shape + coefficient_shape)


def initialise_coeffs(coeffs, shape, *, empty=False):
    """Return editable coefficients for a non-empty intrinsic ``shape`` tuple.

    None creates zeros of that shape unless ``empty=True`` retains None.
    Array-like inputs become JAX arrays and must end in ``shape``; leading axes
    remain batches. Expressions are retained
    unchanged, with their shapes left for evaluation to check.
    """
    if coeffs is None and empty:
        return None
    coeffs = np.zeros(shape) if coeffs is None else zdx.as_array(coeffs)
    if isinstance(coeffs, Array) and coeffs.shape[-len(shape) :] != tuple(shape):
        raise ValueError(f"coeffs must end in the shape {shape}.")
    return coeffs


class ParametricBasis(zdx.Expression, Unitful):
    """A physical expansion with stored coefficients and a coefficient solve.

    Concrete families implement ``evaluate`` and ``solve`` for their representation.
    Evaluation resolves nested coefficient expressions. Solving returns physical
    coefficients without reversing any transform stored in ``coeffs``.
    ``coeffs_ndim`` counts the intrinsic trailing coefficient axes; preceding axes
    represent independent batches, where supported by the family.

    ``unit`` names the expansion's output unit. Its conversion to metres, radians
    or photons is applied after coefficient contraction; numerical modes remain
    unscaled. ``solve`` returns coefficients in the declared unit.
    """

    # Inherited fields
    unit: str | None = eqx.field(kw_only=True, default=None)

    coeffs: Array | zdx.Expression | None

    @property
    @abstractmethod
    def coeffs_ndim(self) -> int:
        """Number of intrinsic coefficient axes, excluding independent batches."""

    @abstractmethod
    def solve(self, samples, **context):
        """Recover physical coefficients; each family defines its solve contract."""


class Basis(ParametricBasis):
    """Expand coefficients over an explicit or context-generated sampled basis.

    ``basis`` has leading coefficient axes followed by arbitrary output axes.
    ``coeffs`` ends in the same coefficient shape; preceding axes are independent
    batches when axes is explicit or the generator advertises ``coeffs_shape``.
    Otherwise, all supplied coefficient dimensions are contracted by default.
    Use ``jax.vmap`` for paired basis and coefficient batches.
    Evaluation does not normalise basis vectors or their output. Local ``solve``
    returns minimum-norm least-squares coefficients, including for deficient bases.
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

    # Inherited fields
    unit: str | None = eqx.field(kw_only=True, default=None)

    coeffs: Array | zdx.Expression | None
    basis: Array | zdx.Expression
    axes: int = eqx.field(static=True, default=1)

    def __init__(
        self,
        basis,
        coeffs=None,
        *,
        axes=None,
        shape=None,
        empty=False,
        unit=None,
        alias=None,
    ):
        """Initialise a sampled linear expansion.

        Parameters
        ----------
        basis : Array or zodiax.Expression
            Non-empty tensor whose first ``axes`` dimensions are coefficient axes.
            Remaining dimensions are output axes, with no assumed physical units.
        coeffs : Array, zodiax.Expression, or None
            Coefficients ending in the leading basis shape. Leading coefficient
            batches are retained. None creates stored zero coefficients at
            construction, using the sampled basis dimensions or explicit shape.
        axes : int or None
            Number of coefficient axes, fixed at construction. None uses a generator
            coeffs_shape first, then coeffs.ndim or len(shape). With neither,
            omitted coefficients use one axis. Supply axes explicitly to retain
            leading coefficient batches, or axes/shape for expression coefficients.
        shape : tuple of int or None
            Coefficient shape for a generated basis when coeffs is omitted. It
            contains exactly axes positive dimensions. Sampled arrays supply this
            shape directly. A generator can instead expose ``coeffs_shape``, a tuple
            of intrinsic coefficient dimensions; it is never evaluated here.
        empty : bool
            Leave omitted coefficients as None instead of allocating zeros. Supplied
            coefficients are retained. Fill an empty expansion with ``set(coeffs=...)``.
        unit : str or None
            Output unit, converted after contraction. None leaves values unchanged.
        alias : mapping or None
            Optional parameter aliases.
        """
        # A generator may advertise its intrinsic coefficient shape without sampling.
        # This is an optional public contract, independent of the generator's class.
        basis = zdx.as_array(basis)
        generated_shape = getattr(basis, "coeffs_shape", None)
        if shape is None and generated_shape is not None:
            shape = generated_shape

        # Infer the coefficient domain from arrays or an explicit construction shape.
        coeffs = zdx.as_array(coeffs)
        if shape is not None:
            shape = tuple(index(size) for size in shape)
        if axes is None:
            if generated_shape is not None:
                axes = len(generated_shape)
            elif isinstance(coeffs, Array):
                axes = coeffs.ndim
            elif shape is not None:
                axes = len(shape)
            elif isinstance(coeffs, zdx.Expression):
                raise ValueError("Provide axes or shape for expression coefficients.")
            else:
                axes = 1

        # Store the contraction count and definitions without evaluating expressions.
        if isinstance(axes, bool):
            raise TypeError("axes must be a positive integer.")
        self.axes = index(axes)
        if self.axes < 1:
            raise ValueError("axes must be a positive integer.")
        self.basis = basis
        self.unit = self.validate_unit(unit)
        self.alias = alias

        if shape is not None:
            if len(shape) != self.axes or any(size < 1 for size in shape):
                raise ValueError("shape must contain axes positive dimensions.")

        if generated_shape is not None and tuple(shape) != tuple(generated_shape):
            raise ValueError(
                f"shape must match the generator's coeffs_shape {generated_shape}."
            )

        # Arrays tell us the coefficient dimensions immediately. A generated
        # basis needs them from its constructor, without sampling the generator.
        if isinstance(self.basis, Array):
            if self.basis.ndim < self.axes or 0 in self.basis.shape:
                raise ValueError("basis must contain non-empty coefficient axes.")
            basis_shape = self.basis.shape[: self.axes]
            if shape is not None and shape != basis_shape:
                raise ValueError(f"shape must match the basis shape {basis_shape}.")
            shape = basis_shape

        # Keep supplied coefficients, or initialise zeros for the known shape.
        if shape is None:
            if coeffs is None and not empty:
                raise ValueError(
                    "Provide coeffs or shape for a generated basis so its "
                    "coefficients can be initialised, or use empty=True."
                )
            self.coeffs = coeffs
        else:
            self.coeffs = initialise_coeffs(coeffs, shape, empty=empty)

    @property
    def coeffs_ndim(self) -> int:
        """Number of leading basis axes contracted with stored coefficients."""
        return self.axes

    def evaluate(self, **context: Any) -> Array:
        """Expand stored coefficients, retaining independent leading batches."""
        self = self.resolve_children(**context)
        return eval_basis(self.basis, self.coeffs, self.axes) * unit_factor(self.unit)

    def solve(self, y: Array, **context: Any) -> Array:
        """Resolve the stored modes and recover physical coefficients.

        The sampled output y has independent leading batch axes followed by the
        basis output axes. The result has those batch axes followed by coefficient
        axes. This is a minimum-norm least-squares representative, including for
        rank-deficient modes. It does not reverse transforms stored in coeffs.
        Context is used where mode generation or stored modes require it.
        """
        self = self.populate()
        basis = zdx.resolve(self.basis, **context)
        return solve_basis(y / unit_factor(self.unit), basis, self.axes)


class PastedBasis(ParametricBasis):
    """Parameterise compact local bases pasted into a common output grid.

    This representation is normally produced by segmented aperture builders. It is
    public so specialised builders can retain dense local modes and avoid storing a
    global mostly-zero basis.
    """

    # Inherited fields
    unit: str | None = eqx.field(kw_only=True, default=None)

    coeffs: Array | zdx.Expression | None
    shape: tuple[int, ...] = eqx.field(static=True)
    basis: Array
    spec: PasteSpec
    method: str = "scan"

    def __init__(
        self,
        basis,
        spec,
        coeffs=None,
        method="scan",
        *,
        empty=False,
        unit=None,
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
            Coefficients with shape ``(..., n_stamps, *basis_shape)``.
        empty : bool
            Retain omitted coefficients as None; otherwise initialise stored zeros.
        method : str
            ``"scan"`` for lower memory or ``"scatter"`` for parallel placement.
        unit : str or None
            Output unit, converted after contraction. None leaves values unchanged.
        alias : mapping or None
            Optional parameter aliases.
        """
        # Check that each sampled basis stamp matches its placement specification.
        if not isinstance(spec, PasteSpec):
            raise TypeError("spec must be a PasteSpec.")

        basis = zdx.as_array(basis, dtype=float)
        if basis.ndim < 4:
            raise ValueError("basis must have shape (n_stamps, ..., ny, nx).")
        if basis.shape[0] != len(spec.starts):
            raise ValueError("basis and PasteSpec must contain the same stamp count.")
        if basis.shape[-2:] != spec.stamp_shape[::-1]:
            raise ValueError("basis spatial shape must match the PasteSpec stamps.")

        method = str(method).lower()
        if method not in ("scan", "scatter"):
            raise ValueError("method must be either 'scan' or 'scatter'.")

        # The stamp index is part of the coefficient domain, not an independent batch.
        shape = basis.shape[:-2]
        self.coeffs = initialise_coeffs(coeffs, shape, empty=empty)
        self.shape = shape
        self.basis = basis
        self.spec = spec
        self.method = method
        self.unit = self.validate_unit(unit)

        self.alias = alias

    @property
    def coeffs_ndim(self) -> int:
        """Complete coefficient domain, including the intrinsic stamp axis."""
        return len(self.shape)

    def evaluate(self, **context: Any) -> Array:
        """Expand per-stamp coefficients and add overlapping stamps globally."""
        self = self.resolve_children(**context)
        coeffs = self.coeffs
        if coeffs is None or isinstance(coeffs, zdx.Expression):
            raise zdx.Unresolved("Basis coefficients need their required context.")
        ndim = self.coeffs_ndim
        if coeffs.shape[-ndim:] != self.shape:
            raise ValueError(f"coeffs must end in the shape {self.shape}.")

        # Placement maps over stamps; independent coefficient batches stay inside.
        coeffs = np.moveaxis(coeffs, -ndim, 0)
        expand = lambda b, c: eval_basis(b, c, axes=ndim - 1)
        local = jax.vmap(expand)(self.basis, coeffs)
        return self.spec.paste(local, self.method) * unit_factor(self.unit)

    def solve(self, value: Array, **kwargs: Any) -> Array:
        """Solve coefficients independently from every extracted local stamp.

        ``value`` must end with the global output shape. Its leading batches are
        retained before ``(n_stamps, *basis_shape)`` in the returned coefficients.
        This is a local projection, not a global inverse: overlapping stamps
        contain contributions from their neighbours. This method is not a global
        least-squares solve.
        """
        values = self.spec.extract(value / unit_factor(self.unit))
        solve = lambda value, basis: solve_basis(value, basis, len(self.shape) - 1)
        coeffs = jax.vmap(solve)(values, self.basis)
        return np.moveaxis(coeffs, 0, -self.coeffs_ndim)
