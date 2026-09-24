"""Dimensionless circular and polygonal Zernike mode generation."""

from collections.abc import Callable
from math import isqrt
from typing import ClassVar

import equinox as eqx
import jax.numpy as np
import jax.scipy as jsp
import zodiax as zdx
from jax import Array, core, lax, vmap

from ..grids import Affine, cart2polar
from ..grids.specs import UnitSpec
from ..numerical.norms import normalise
from ..numerical.polynomials import polynomial_basis
from ..sampled import Sampled

__all__ = ["Zernike"]


class Zernike(Sampled):
    """Generate dimensionless Zernike modes on a circular or polygonal aperture.

    Store this Expression in ``Basis.basis`` to fit its coefficients. Its
    ``coeffs_shape`` property lets Basis allocate zeros without sampling any modes.
    Resolving Zernike itself returns modes shaped ``(n_modes, ny, nx)``, with an
    optional aperture axis before the mode axis when centres are supplied.

    ``n_sides=0`` selects the circular definition; values at least three select
    the polygonal (Polike) mapping. Support and normalisation apply to each mode,
    while coefficient amplitudes and units belong to the containing Basis.

    Examples
    --------
    ```python
    import dLux as dl

    # Keep mode geometry separate from fitted OPD coefficients.
    grid = dl.Grid(8, diam=1.0, unit="m")
    modes = dl.Zernike([4, 5, 6], diameter=1.0)
    opd = dl.Basis(modes, coeffs=[10.0, 0.0, 0.0], unit="nm")
    print(opd.resolve(grid=grid).shape)
    print(dl.Zernike.names[4])
    ```
    """

    # Inherited fields
    grid: UnitSpec | None = None
    transform: Callable | None = None
    strict: bool = eqx.field(static=True, default=False)

    # Shared metadata stays outside the PyTree. X denotes cosine and Y sine.
    names: ClassVar[dict[int, str]] = {
        # 0th Radial
        1: "Piston",
        # 1st Radial
        2: "Tilt X",
        3: "Tilt Y",
        # Second Radial
        4: "Defocus",
        5: "Astig Y",
        6: "Astig X",
        # Third Radial
        7: "Coma Y",
        8: "Coma X",
        9: "Trefoil Y",
        10: "Trefoil X",
        # Fourth Radial
        11: "Spherical",
        12: "2nd Astig X",
        13: "2nd Astig Y",
        14: "Quadrafoil X",
        15: "Quadrafoil Y",
        # Fifth Radial
        16: "2nd Coma X",
        17: "2nd Coma Y",
        18: "2nd Trefoil X",
        19: "2nd Trefoil Y",
        20: "Pentafoil X",
        21: "Pentafoil Y",
        # Sixth Radial
        22: "2nd Spherical",
        23: "3rd Astig Y",
        24: "3rd Astig X",
        25: "2nd Quadrafoil Y",
        26: "2nd Quadrafoil X",
        27: "Hexafoil Y",
        28: "Hexafoil X",
        # Seventh Radial
        29: "3rd Coma Y",
        30: "3rd Coma X",
        31: "3rd Trefoil Y",
        32: "3rd Trefoil X",
        33: "2nd Pentafoil Y",
        34: "2nd Pentafoil X",
        35: "Heptafoil Y",
        36: "Heptafoil X",
    }

    nolls: Array
    groups: tuple["Zernike._Group", ...]
    order: Array
    diameter: Array | zdx.Expression | None = None
    n_sides: int = eqx.field(default=0, static=True)
    oversize: Array | zdx.Expression
    norm: str | None = eqx.field(default=None, static=True)
    method: str = eqx.field(default="padded", static=True)
    support: Array | zdx.Expression | None = None
    centers: Array | None = None

    def __init__(
        self,
        nolls=None,
        order=None,
        orders=None,
        diameter=None,
        n_sides=0,
        oversize=0.0,
        norm=None,
        method="padded",
        *,
        grid=None,
        transform=None,
        support=None,
        centers=None,
        strict=False,
        alias=None,
    ):
        """Select modes and prepare polynomial factors without sampling a grid.

        ``method`` chooses padded or equal-width mapped radial groups. Geometry,
        support and transform may contain Expressions. Centres have a fixed shape
        ``(n_apertures, 2)``; their values remain differentiable.

        Parameters
        ----------
        nolls : sequence of ints or None
            One-based Noll indices. Supply exactly one of nolls, order or orders.
        order : int or None
            Maximum radial order, including lower orders; alternative to nolls and
            orders.
        orders : sequence of ints or None
            Selected radial orders; alternative to nolls and order.
        diameter : ArrayLike or Expression or None
            Aperture diameter in SI coordinate units; None uses 2.0 during
            evaluation.
        n_sides : int
            Zero selects circular modes; values of at least three select the
            polygonal mapping.
        oversize : ArrayLike or Expression
            Fractional diameter enlargement; zero leaves diameter unchanged.
        norm : str or None
            Optional per-mode support normalisation: l1, l2, max, rms or p2v. None
            retains standard mode amplitudes.
        method : str
            Radial evaluation strategy: padded (default) or mapped. Polynomial
            factors are prepared at construction; mode arrays are not.
        grid : Grid or None
            Optional stored sampling; caller context supplies sampling when it is
            absent.
        transform : callable or None
            Map evaluation coordinates into the input frame. None leaves coordinates
            unchanged.
        support : Array or Expression or None
            Optional mode-support mask on the evaluation grid. None uses the natural
            aperture support.
        centers : Array or None
            Aperture centres shaped (A, 2) in SI coordinate units; adds an aperture
            axis before the modes.
        strict : bool
            Reject conflicting runtime sampling or transforms when true; false
            allows runtime overrides.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        # Precompute radial groups and remember how to restore the requested mode order.
        selected = self.indices(nolls, order, orders)
        self.nolls = np.asarray(selected, dtype=int)
        method = str(method).lower()
        if method not in ("padded", "mapped"):
            raise ValueError("method must be either 'padded' or 'mapped'.")
        self.method = method
        self.groups = self._groups(selected, method)
        indices = np.concatenate(tuple(group.indices for group in self.groups))
        self.order = np.argsort(indices)

        # Store geometry and normalisation settings without sampling their expressions.
        self.diameter = zdx.as_array(diameter, dtype=float)
        if self.diameter is not None:
            if not isinstance(self.diameter, (zdx.Expression, core.Tracer)):
                if np.any(self.diameter <= 0):
                    raise ValueError("diameter must be greater than zero.")
        self.n_sides = int(n_sides)
        if self.n_sides != 0 and self.n_sides < 3:
            raise ValueError("n_sides must be zero or greater than two.")
        self.oversize = zdx.as_array(oversize, dtype=float)
        if norm is not None:
            norm = str(norm).lower()
            if norm not in ("l1", "l2", "max", "rms", "p2v"):
                raise ValueError("norm must be one of l1, l2, max, rms, p2v, or None.")
        self.norm = norm
        self.support = zdx.as_array(support, dtype=float)
        self.centers = zdx.as_array(centers, dtype=float)

        # Aperture count is structural; positions remain differentiable leaves.
        if self.centers is not None:
            if self.centers.ndim != 2 or self.centers.shape[-1] != 2:
                raise ValueError("centers must have shape (n_apertures, 2).")
        super().__init__(grid=grid, transform=transform, strict=strict, alias=alias)

    @property
    def coeffs_shape(self) -> tuple[int, ...]:
        """Intrinsic leading mode dimensions, available without a coordinate grid."""
        shape = (len(self.nolls),)
        if self.centers is not None:
            shape = (self.centers.shape[0],) + shape
        return shape

    def evaluate(
        self,
        *,
        grid=None,
        coordinates=None,
        transform=None,
        diameter=None,
        n_sides=None,
        support=None,
        centers=None,
        **context,
    ):
        """Sample modes using stored geometry or explicit runtime overrides.

        The generator owns its transformed coordinates. Sibling coefficient
        expressions in a containing Basis receive the parent's context separately.
        """
        self = self.populate()
        local = self.sampling_context(
            grid, coordinates=coordinates, transform=transform, ndim=2, **context
        )
        diameter = self.diameter if diameter is None else diameter
        local["diameter"] = zdx.resolve(2.0 if diameter is None else diameter, **local)
        local["n_sides"] = self.n_sides if n_sides is None else n_sides
        support = self.support if support is None else support
        centers = self.centers if centers is None else centers
        local["support"] = zdx.resolve(support, **local)
        local["centers"] = zdx.resolve(centers, **local)
        self = self.resolve_children(**local)

        coordinates = local["coordinates"]
        oversize = zdx.resolve(self.oversize, **local)
        diameter = np.asarray(local["diameter"]) * (1 + oversize)
        n_sides = local["n_sides"]
        support, centers = local["support"], local["centers"]

        # Evaluate rectangular groups, then restore the requested Noll order.
        def calculate(coords):
            groups = [
                group.calculate(coords, diameter, n_sides) for group in self.groups
            ]
            return np.concatenate(groups)[self.order]

        if centers is None:
            basis = calculate(coordinates)
        else:
            evaluate = lambda center: calculate(Affine(translation=center)(coordinates))
            basis = vmap(evaluate)(centers)
        if basis.shape[:-2] != self.coeffs_shape:
            raise ValueError(
                "Aperture centres must match the stored coefficient shape."
            )

        # Clip on the aperture support and normalise each dimensionless mode.
        if support is not None:
            support = np.atleast_2d(support)
            while support.ndim < basis.ndim:
                support = np.expand_dims(support, -3)
            basis *= support
        if self.norm is not None:
            basis = normalise(basis, self.norm, weights=support, axis=(-2, -1))
        return basis

    @classmethod
    def name(cls, j: int) -> str:
        """Gets the name of the jth Zernike polynomial.

        Parameters
        ----------
        j : int
            The Zernike (noll) index.

        Returns
        -------
        name : str
            The name of the Zernike polynomial.
        """
        return cls.names[int(j)] if j >= 1 and j <= 36 else f"Zernike {int(j)}"

    @staticmethod
    def noll_indices(j: int) -> tuple[int]:
        """Calculate the radial and azimuthal orders of the Zernike polynomial.

        Parameters
        ----------
        j : int
            The Zernike (noll) index.

        Returns
        -------
        n, m : tuple[int]
            The radial and azimuthal orders of the Zernike polynomial.
        """
        # The mode chooses fixed polynomial topology, so keep this calculation in
        # Python even when the surrounding model is being constructed under JIT.
        j = int(j)
        if j < 1:
            raise ValueError("The Zernike index must be greater than zero.")
        n = (isqrt(8 * j - 7) - 1) // 2
        smallest_j_in_row = n * (n + 1) // 2 + 1
        number_of_shifts = (j - smallest_j_in_row + ~(n & 1) + 2) // 2
        sign_of_shift = -(j & 1) + ~(j & 1) + 2
        base_case = n & 1
        m = sign_of_shift * (base_case + number_of_shifts * 2)
        return n, m

    @staticmethod
    def radial_orders_to_indices(radial_orders: list[int]) -> list[int]:
        """Convert radial Zernike orders to their complete Noll index sequence."""
        indices = []
        for order in radial_orders:
            order = int(order)
            if order < 0:
                raise ValueError("radial_orders must contain non-negative integers.")
            first = order * (order + 1) // 2 + 1
            indices.extend(range(first, first + order + 1))
        return indices

    @classmethod
    def indices(cls, nolls=None, order=None, orders=None) -> list[int]:
        """Resolve individual modes or complete radial orders."""

        # Convert explicit Noll selections or radial orders into one ordered mode list.
        if sum(value is not None for value in (nolls, order, orders)) != 1:
            raise ValueError("Provide exactly one of nolls, order, or orders.")

        if nolls is not None:
            if isinstance(nolls, int) or getattr(nolls, "ndim", None) == 0:
                nolls = (nolls,)
            if getattr(nolls, "ndim", 1) != 1:
                raise ValueError("Noll indices must be one-dimensional.")
            indices = [int(j) for j in nolls]
        else:
            orders = range(int(order) + 1) if order is not None else orders
            if isinstance(orders, int) or getattr(orders, "ndim", None) == 0:
                orders = (orders,)
            if getattr(orders, "ndim", 1) != 1:
                raise ValueError("Radial orders must be one-dimensional.")
            indices = cls.radial_orders_to_indices(orders)

        # Reject empty or invalid selections before precomputing radial factors.
        if not indices:
            raise ValueError("At least one Zernike mode must be selected.")
        if any(j < 1 for j in indices):
            raise ValueError("Zernike indices must be greater than zero.")
        return indices

    @classmethod
    def factors(cls, j: int) -> tuple[Array]:
        """Calculate radial coefficients and powers for one Noll index.

        Parameters
        ----------
        j : int
            The Zernike (noll) index.

        Returns
        -------
        c, k : tuple[Array]
            The normalisation coefficients and powers of the Zernike polynomial.
        """
        if j < 1:
            raise ValueError("The Zernike index must be greater than 0.")
        n, m = cls.noll_indices(j)
        # The sign selects sine/cosine only; the radial polynomial uses |m|.
        m = abs(m)

        # Calculate k
        k = np.arange(((n - m) // 2) + 1, dtype=float)

        # Calculate c
        sign = lax.pow(-1.0, k)
        _fact_1 = jsp.special.factorial(np.abs(n - k))
        _fact_2 = jsp.special.factorial(k)
        _fact_3 = jsp.special.factorial(((n + m) // 2) - k)
        _fact_4 = jsp.special.factorial(((n - m) // 2) - k)
        c = sign * _fact_1 / _fact_2 / _fact_3 / _fact_4
        return c, k

    @staticmethod
    @eqx.filter_jit
    def radial(rho: Array, n: int, c: Array, k: Array) -> Array:
        """Calculates the radial component of the Zernike polynomial.

        Parameters
        ----------
        rho : Array
            The radial coordinate of the Zernike polynomial.
        n : int
            The radial order of the Zernike polynomial.
        c : Array
            The normalisation coefficients of the Zernike polynomial.
        k : Array
            The powers of the Zernike polynomial.

        Returns
        -------
        radial : Array
            The radial component of the Zernike polynomial.
        """
        # Reuse the polynomial power rule so constant terms at rho=0 have finite
        # derivatives, including when differentiating a compiled calculation twice.
        powers = (np.abs(n) - 2 * k).astype(int)
        rads = polynomial_basis(rho[None, ...], powers[None, :])
        return (c[:, None, None] * rads).sum(axis=0)

    @staticmethod
    @eqx.filter_jit
    def azimuthal(theta: Array, n: int, m: int) -> Array:
        """Calculates the azimuthal component of the Zernike polynomial.

        Parameters
        ----------
        theta : Array
            The azimuthal coordinate of the Zernike polynomial.
        n : int
            The radial order of the Zernike polynomial.
        m : int
            The azimuthal order of the Zernike polynomial.

        Returns
        -------
        azimuthal : Array
            The azimuthal component of the Zernike polynomial.
        """
        # Get the radial and angular factors
        norm_coeff = np.sqrt(n + 1) * np.where(m == 0, 1.0, np.sqrt(2))
        angle = np.abs(m) * theta

        # Evaluate the positive or negative azimuthal convention
        azimuthal = np.where(m >= 0, np.cos(angle), np.sin(angle))
        return norm_coeff * azimuthal

    @staticmethod
    def mode(n, m, c, k, coordinates, diameter=2, n_sides=0):
        """Evaluate one prepared Zernike mode, optionally on a regular polygon.

        Circular modes use their Cartesian polynomial, which is smooth at the origin.
        The polygon mapping retains its angular boundary scaling; that mapping need
        not be differentiable at the origin or at changes of polygon sector.
        """
        # A scalar diameter gives circular support; an (x, y) pair gives an ellipse.
        radius = np.asarray(diameter) / 2
        if radius.ndim and radius.shape[-1] == coordinates.shape[-3]:
            radius = radius.reshape(
                radius.shape + (1,) * (coordinates.ndim - radius.ndim)
            )
        coordinates = coordinates / radius
        radius_scale = 1.0
        if n_sides:
            if n_sides < 3:
                raise ValueError(f"n_sides must be >= 3, not {n_sides}.")
            alpha = np.pi / n_sides
            phi = cart2polar(coordinates)[1] + alpha
            wedge = np.floor((phi + alpha) / (2 * alpha))
            radius_scale = np.cos(alpha) / np.cos(phi - wedge * (2 * alpha))
            coordinates = coordinates / radius_scale

        # Factor out rho**|m| from the radial polynomial, leaving powers of rho**2.
        # Zero-coefficient padding may give negative powers here; use a constant for
        # those unused terms so they remain finite at the origin.
        x, y = coordinates
        radius_squared = x * x + y * y
        magnitude = np.abs(m)
        powers = np.maximum((n - magnitude) // 2 - k, 0).astype(int)
        terms = polynomial_basis(radius_squared[None, ...], powers[None, :])
        radial = (c[:, None, None] * terms).sum(axis=0)

        # (x + iy)**|m| equals rho**|m| * (cos(|m| theta) + i sin(|m| theta)).
        # Its real/imaginary parts replace the singular polar calculation without
        # changing the Noll convention. Reuse the same polynomial power rule here.
        harmonic = polynomial_basis((x + 1j * y)[None, ...], magnitude.reshape(1, 1))[0]
        azimuthal = np.where(m >= 0, harmonic.real, harmonic.imag)
        norm = np.sqrt(n + 1) * np.where(m == 0, 1.0, np.sqrt(2))
        return (radius_squared <= 1.0) * norm * radial * azimuthal / radius_scale

    @classmethod
    def _groups(cls, nolls, method):
        """Precompute padded or equal-complexity Zernike evaluation groups."""
        # Generate native metadata for every requested mode
        metadata = []
        for index, j in enumerate(nolls):
            n, m = cls.noll_indices(j)
            c, k = cls.factors(j)
            metadata.append((index, n, m, c, k))

        # Partition modes by the requested evaluation strategy
        if method == "padded":
            widths = (max(len(values[3]) for values in metadata),)
        else:
            widths = tuple(sorted({len(values[3]) for values in metadata}))

        # Build rectangular arrays within each evaluation group
        groups = []
        for width in widths:
            if method == "padded":
                values = metadata
            else:
                values = [value for value in metadata if len(value[3]) == width]
            indices, ns, ms, coeffs, ks = zip(*values)

            # Pad coefficients with zeros; padded exponents stay finite at the origin.
            if method == "padded":
                pad_k = lambda a, n: np.pad(
                    a, (0, width - len(a)), constant_values=n / 2
                )
                ks = [pad_k(array, n) for array, n in zip(ks, ns)]
                coeffs = [np.pad(array, (0, width - len(array))) for array in coeffs]

            groups.append(cls._Group(indices, ns, ms, np.stack(coeffs), np.stack(ks)))
        return tuple(groups)

    class _Group(zdx.Module):
        """A rectangular collection of jointly vectorised Zernike modes."""

        indices: Array
        n: Array
        m: Array
        c: Array
        k: Array

        def __init__(self, indices, n, m, c, k, *, alias=None):
            self.indices = zdx.as_array(indices, dtype=int)
            self.n = zdx.as_array(n, dtype=int)
            self.m = zdx.as_array(m, dtype=int)
            self.c = zdx.as_array(c, dtype=float)
            self.k = zdx.as_array(k, dtype=float)

            self.alias = alias

        def calculate(self, coordinates, diameter, nsides=0):
            """Evaluate every mode in this equal-width group."""
            calculate = lambda n, m, c, p: Zernike.mode(
                n, m, c, p, coordinates, diameter, nsides
            )
            return vmap(calculate)(self.n, self.m, self.c, self.k)
