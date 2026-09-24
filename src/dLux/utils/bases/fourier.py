"""Compact separable Fourier expansions on Cartesian grids."""

from typing import Any

import equinox as eqx
import jax.numpy as np
import zodiax as zdx
from jax import Array

from ..grids.array_ops import as_size
from .basis import ParametricBasis, initialise_coeffs, solve_basis
from ..numerical.units import unit_factor

__all__ = ["FourierBasis", "fourier_kernel_1d", "fourier_kernels"]


def fourier_kernel_1d(n_modes: int, npix: int, scale: float = 1.0) -> Array:
    """Calculates a cached 1D Fourier basis evaluation kernel.

    A unit-amplitude mode with `scale=1` evaluates to values between `-1` and `+1`.

    Parameters
    ----------
    n_modes : int
        The number of Fourier modes.
    npix : int
        The number of output pixels.
    scale : float = 1.0
        The output amplitude scaling.

    Returns
    -------
    kernel : Array
        The cached Fourier kernel.
    """
    # Real modes are ordered DC, cos1, sin1, cos2, sin2, ... .
    modes = np.arange(n_modes)
    frequencies = (modes + 1) // 2
    coordinates = (np.arange(npix) - npix // 2) / npix
    phase = 2 * np.pi * coordinates[:, None] * frequencies
    sine = (modes != 0) & (modes % 2 == 0)
    return scale * np.where(sine, np.sin(phase), np.cos(phase))


def fourier_kernels(
    n_modes: int | tuple[int], npix: int | tuple[int], scale: float = 1.0
) -> tuple[Array, Array]:
    """Calculates the cached 2D Fourier basis evaluation kernels.

    The input `n_modes` and `npix` follow array-axis order `(y, x)`. If an
    integer is provided it is cast to `(value, value)`. The returned kernels satisfy

    `output = Ky @ coefficients @ Kx.T`

    where `coefficients` has shape `(n_modes_y, n_modes_x)` and `output` has shape
    `(npix_y, npix_x)`. This retains the released matrix contraction and ordering.

    Parameters
    ----------
    n_modes : int | tuple[int]
        The number of Fourier modes in `(y, x)`.
    npix : int | tuple[int]
        The output number of pixels in `(y, x)`.
    scale : float = 1.0
        The per-axis output amplitude scaling.

    Returns
    -------
    Ky : Array
        The y-axis Fourier kernel.
    Kx : Array
        The x-axis Fourier kernel.
    """
    # Pairs follow array order; a scalar supplies the same count for both axes.
    for value, name in ((n_modes, "n_modes"), (npix, "npix")):
        if isinstance(value, (tuple, list)) and len(value) != 2:
            raise ValueError(f"{name} must be a scalar or a (y, x) pair.")
    n_modes_y, n_modes_x = as_size(n_modes, 2, "n_modes")
    npix_y, npix_x = as_size(npix, 2, "npix")

    # Calculate the Fourier kernels for each dimension.
    Kx = fourier_kernel_1d(n_modes_x, npix_x, scale)
    Ky = fourier_kernel_1d(n_modes_y, npix_y, scale)

    return Ky, Kx


class FourierBasis(ParametricBasis):
    """A parameterisation over a separable real Fourier basis.

    Coefficients are stored in ``coeffs``. Evaluation returns
    dLux image axes ``(ny, nx)``; coefficient axes and size tuples use the same
    array-axis order. The forward operation retains
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

    # Inherited fields
    unit: str | None = eqx.field(kw_only=True, default=None)

    coeffs: Array | zdx.Expression | None
    shape: tuple[int, ...] = eqx.field(static=True)
    kernels: tuple[Array, Array]

    def __init__(
        self, npix, n_modes, coeffs=None, *, empty=False, unit=None, alias=None
    ):
        """Initialise an implicit separable Fourier basis.

        Parameters
        ----------
        npix : int or tuple[int, int]
            Sampled output size.
        n_modes : int or tuple[int, int]
            Fourier-mode counts in array-axis order (y, x).
        coeffs : Array, zodiax.Expression, or None
            Coefficient array matching the generated separable basis shape.
        empty : bool
            Retain omitted coefficients as None; otherwise initialise stored zeros.
        unit : str or None
            Output unit, converted after contraction. None leaves values unchanged.
        alias : mapping or None
            Optional parameter aliases.
        """
        self.kernels = fourier_kernels(n_modes, npix)
        shape = tuple(kernel.shape[1] for kernel in self.kernels)
        self.coeffs = initialise_coeffs(coeffs, shape, empty=empty)
        self.shape = shape
        self.unit = self.validate_unit(unit)

        self.alias = alias

    @property
    def coeffs_ndim(self) -> int:
        """Number of separable Fourier-mode axes in the coefficient array."""
        return len(self.shape)

    def calculate_basis(self, **kwargs: Any) -> Array:
        """Materialise the complete separable real Fourier basis.

        The result has shape ``(*shape, ny, nx)``, matching dLux image conventions.
        Additional context is accepted but not consumed because the sampling
        kernels are fixed at construction.
        """
        Ky, Kx = self.kernels
        return np.einsum("yi,xj->ijyx", Ky, Kx)

    def evaluate(self, **kwargs: Any) -> Array:
        """Evaluate the separable Fourier expansion without materialising its basis.

        Returns a sampled ``(ny, nx)`` image with leading coefficient batches
        retained, without materialising the full basis tensor.
        """
        self = self.resolve_children(**kwargs)
        if self.coeffs is None or isinstance(self.coeffs, zdx.Expression):
            raise zdx.Unresolved("Basis coefficients need their required context.")
        Ky, Kx = self.kernels
        return (Ky @ self.coeffs @ Kx.T) * unit_factor(self.unit)

    def solve(self, y: Array, **context: Any) -> Array:
        """Materialise Fourier modes and recover their coefficients.

        The sampled output y has independent leading batch axes followed by the
        basis output axes. The result has those batch axes followed by coefficient
        axes. This is a minimum-norm least-squares representative, including for
        rank-deficient modes. It does not reverse transforms stored in coeffs.
        Context is used where mode generation or stored modes require it.
        """
        basis = self.calculate_basis()
        return solve_basis(y / unit_factor(self.unit), basis, axes=2)

    def resize(self, npix):
        """Return a copy sampled onto a resized Fourier grid.

        ``npix`` gives array ``(y, x)`` sample counts. Each kernel's constant-mode
        amplitude, the output unit and coefficients are retained. Fourier kernels
        are regenerated, so arbitrary edits to individual modes are not preserved.
        """
        kernels = fourier_kernels(self.shape, npix)
        # The DC column stores the scale directly, including fitted updates.
        kernels = tuple(
            new * original[0, 0] for new, original in zip(kernels, self.kernels)
        )
        return self.set(kernels=kernels)
