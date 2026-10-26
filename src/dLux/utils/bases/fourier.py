"""Compact separable Fourier expansions on Cartesian grids."""

from typing import Any

import equinox as eqx
import jax.numpy as np
import zodiax as zdx
from jax import Array

from ..grids.array_ops import as_size
from ..numerical.units import unit_factor
from .basis import ParametricBasis, initialise_coeffs, solve_basis

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
    array-axis order. Without ``pupil_diameter``, the forward operation retains
    compact, fixed kernels. With a pupil diameter in metres, evaluation uses
    supplied local ``(x, y)`` coordinates in metres, including arbitrary
    coordinate transforms. For fixed, separable coordinates, ``prepare`` caches
    compact physical kernels for repeated evaluations. Prepared bases ignore
    runtime coordinates; re-prepare explicitly to change their geometry.

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

    # Cache physical modes when the local grid and pupil diameter are fixed
    grid = dl.Grid(n=128, diam=2.0, unit="m")
    physical = dl.FourierBasis(128, 8, pupil_diameter=1.0)
    prepared = physical.prepare(coordinates=grid.coordinates)
    physical_values = prepared.evaluate()
    ```
    """

    # Inherited fields
    unit: str | None = eqx.field(kw_only=True, default=None)

    coeffs: Array | zdx.Expression | None
    shape: tuple[int, ...] = eqx.field(static=True)
    kernels: tuple[Array, Array]
    pupil_diameter: Array | zdx.Expression | None = None
    base_kernels: tuple[Array, Array] | None = None

    def __init__(
        self,
        npix,
        n_modes,
        coeffs=None,
        *,
        empty=False,
        pupil_diameter=None,
        unit=None,
        alias=None,
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
        pupil_diameter : Array, zodiax.Expression, or None
            Physical pupil diameter in metres. When supplied, basis calculation,
            evaluation, and solving require local ``coordinates`` with shape
            ``(2, ny, nx)`` matching ``npix`` unless kernels have been prepared.
        unit : str or None
            Output unit, converted after contraction. None leaves values unchanged.
        alias : mapping or None
            Optional parameter aliases.
        """
        self.kernels = fourier_kernels(n_modes, npix)
        shape = tuple(kernel.shape[1] for kernel in self.kernels)
        self.coeffs = initialise_coeffs(coeffs, shape, empty=empty)
        self.shape = shape
        self.pupil_diameter = zdx.as_array(pupil_diameter, dtype=float)
        self.base_kernels = None
        self.unit = self.validate_unit(unit)

        self.alias = alias

    @property
    def coeffs_ndim(self) -> int:
        """Number of separable Fourier-mode axes in the coefficient array."""
        return len(self.shape)

    @property
    def prepared(self) -> bool:
        """Whether physical kernels have been cached for fixed coordinates."""
        return self.base_kernels is not None

    def calculate_basis(self, *, coordinates=None, **context: Any) -> Array:
        """Materialise the complete separable real Fourier basis.

        The result has shape ``(*shape, ny, nx)``, matching dLux image conventions.
        Physical pupil sampling requires local ``(x, y)`` coordinates in metres;
        fixed and prepared kernels ignore runtime coordinates.
        """
        Ky, Kx = self.kernels
        if self.prepared:
            return np.einsum("yi,xj->ijyx", Ky, Kx)
        if self.pupil_diameter is None:
            return np.einsum("yi,xj->ijyx", Ky, Kx)

        y_modes, x_modes = self._physical_modes(coordinates=coordinates, **context)
        return y_modes[:, None] * x_modes[None]

    def evaluate(self, **kwargs: Any) -> Array:
        """Evaluate the Fourier expansion on fixed or physical sampling.

        Returns a sampled ``(ny, nx)`` image with leading coefficient batches
        retained. Fixed sampling uses compact kernels; physical sampling requires
        local ``coordinates`` and contracts its mode families without materialising
        the full basis tensor. Prepared kernels ignore runtime coordinates.
        """
        self = self.resolve_children(**kwargs)
        if self.coeffs is None or isinstance(self.coeffs, zdx.Expression):
            raise zdx.Unresolved("Basis coefficients need their required context.")
        if self.prepared:
            Ky, Kx = self.kernels
            return (Ky @ self.coeffs @ Kx.T) * unit_factor(self.unit)
        if self.pupil_diameter is not None:
            y_modes, x_modes = self._physical_modes(**kwargs)
            values = np.einsum("...ij,iyx,jyx->...yx", self.coeffs, y_modes, x_modes)
            return values * unit_factor(self.unit)
        Ky, Kx = self.kernels
        return (Ky @ self.coeffs @ Kx.T) * unit_factor(self.unit)

    def solve(self, y: Array, **context: Any) -> Array:
        """Materialise Fourier modes and recover their coefficients.

        The sampled output y has independent leading batch axes followed by the
        basis output axes. The result has those batch axes followed by coefficient
        axes. This is a minimum-norm least-squares representative, including for
        rank-deficient modes. It does not reverse transforms stored in coeffs.
        Unprepared physical sampling requires local ``coordinates`` matching
        ``npix``; prepared kernels ignore runtime coordinates.
        """
        basis = self.calculate_basis(**context)
        return solve_basis(y / unit_factor(self.unit), basis, axes=2)

    def prepare(self, *, coordinates=None, **context: Any):
        """Cache physical kernels for repeated evaluation on fixed sampling.

        Coordinates must be separable: x may vary along image columns and y along
        image rows. Preparation is an eager operation. The returned object retains
        the diameter, coefficients, and output unit. It is a snapshot; call prepare
        again after changing geometry. Runtime coordinates do not alter the cache.
        """
        if self.pupil_diameter is None:
            return self
        if coordinates is None:
            raise zdx.Unresolved("Physical Fourier modes require coordinates.")
        coordinates = np.asarray(coordinates)
        y_modes, x_modes = self._physical_modes(coordinates=coordinates, **context)
        x_separable = np.all(coordinates[0] == coordinates[0, :1, :])
        y_separable = np.all(coordinates[1] == coordinates[1, :, :1])
        if not bool(x_separable & y_separable):
            raise ValueError("Prepared physical Fourier modes need separable coordinates.")

        base_kernels = self.base_kernels if self.prepared else self.kernels
        kernels = (y_modes[:, :, 0].T, x_modes[:, 0, :].T)
        return self.set(kernels=kernels, base_kernels=base_kernels)

    def resize(self, npix):
        """Return a copy sampled onto a resized Fourier grid.

        ``npix`` gives array ``(y, x)`` sample counts. Each kernel's constant-mode
        amplitude, pupil diameter, output unit, and coefficients are retained.
        Fourier kernels are regenerated, so arbitrary edits to individual modes
        are not preserved. Physical sampling needs matching new coordinates.
        """
        source_kernels = self.base_kernels if self.prepared else self.kernels
        kernels = fourier_kernels(self.shape, npix)
        # The DC column stores the scale directly, including fitted updates.
        kernels = tuple(
            new * original[0, 0] for new, original in zip(kernels, source_kernels)
        )
        return self.set(kernels=kernels, base_kernels=None)

    def _physical_modes(self, *, coordinates=None, **context):
        """Evaluate the two mode families on local physical coordinates."""
        if coordinates is None:
            raise zdx.Unresolved("Physical Fourier modes require coordinates.")
        coordinates = np.asarray(coordinates)
        Ky, Kx = self.base_kernels if self.prepared else self.kernels
        expected_shape = (2, Ky.shape[0], Kx.shape[0])
        if coordinates.shape != expected_shape:
            raise ValueError(f"coordinates must have shape {expected_shape}.")

        diameter = zdx.resolve(self.pupil_diameter, coordinates=coordinates, **context)
        if isinstance(diameter, zdx.Expression):
            raise zdx.Unresolved("Pupil diameter needs its required context.")

        # The mode axes follow array order (y, x), while coordinates use (x, y).
        def axis_modes(axis, count, scale):
            modes = np.arange(count)
            frequencies = (modes + 1) // 2
            phase = 2 * np.pi * frequencies[:, None, None] * axis[None] / diameter
            sine = (modes != 0) & (modes % 2 == 0)
            return scale * np.where(sine[:, None, None], np.sin(phase), np.cos(phase))

        y_modes = axis_modes(coordinates[1], self.shape[0], Ky[0, 0])
        x_modes = axis_modes(coordinates[0], self.shape[1], Kx[0, 0])
        return y_modes, x_modes
