"""Construction-time generators for sampled optical components."""

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax.tree as jtu
import jax.numpy as np
import jax.random as jr
from jax import Array, vmap

import dLux.utils as dlu

from .base import Base
from .grids import CoordTransform, GridSpec
from .layers.optical import Optic
from .layers.sparse import SparseOptic
from .parametric import Basis, Shape
from .parametric.bases import _resolve_coeffs

__all__ = [
    "GridBuilder",
    "OPDDef",
    "Norm",
    "ZernikeDef",
    "ApertureBuilder",
    "SparseApertureBuilder",
]


def _initialise_coeffs(shape, coeffs=None, key=None, initial_shape=None):
    """Return explicit, random, or zero coefficients for a basis shape."""
    # Validate mutually exclusive initialisation inputs
    if coeffs is not None and key is not None:
        raise ValueError("Provide only one of coeffs or key.")

    # Initialise explicit or random coefficients
    if key is not None:
        initial_shape = shape if initial_shape is None else initial_shape
        coeffs = jr.normal(key, initial_shape)
    elif coeffs is None and initial_shape is not None:
        coeffs = np.zeros(initial_shape)
    return coeffs


def _explicit_basis(basis, coeffs=None, key=None, shape=None, initial_shape=None):
    """Materialise a sampled OPD basis with explicit or random coefficients."""
    # Resolve the native and initial coefficient shapes
    shape = basis.shape[:-2] if shape is None else tuple(shape)
    coeffs = _initialise_coeffs(shape, coeffs, key, initial_shape)

    # Materialise the sampled basis
    return Basis(basis, coeffs=coeffs, shape=shape)


def _zernike_groups(nolls, method):
    """Precompute padded or equal-complexity Zernike evaluation groups."""
    # Generate native metadata for every requested mode
    metadata = []
    for index, j in enumerate(nolls):
        n, m = dlu.noll_indices(j)
        coeffs, k = dlu.zernike_factors(j)
        metadata.append((index, n, m, coeffs, k))

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

        if method == "padded":
            pad_k = lambda a, n: np.pad(a, (0, width - len(a)), constant_values=n / 2)
            ks = [pad_k(array, n) for array, n in zip(ks, ns)]
            coeffs = [np.pad(array, (0, width - len(array))) for array in coeffs]

        groups.append(ZernikeGroup(indices, ns, ms, np.stack(coeffs), np.stack(ks)))
    return tuple(groups)


class ZernikeGroup(Base):
    """A rectangular collection of jointly vectorised Zernike modes."""

    indices: Array
    n: Array
    m: Array
    coeffs: Array
    k: Array

    def __init__(self, indices, n, m, coeffs, k):
        self.indices = dlu.to_value(indices, int)
        self.n = dlu.to_value(n, int)
        self.m = dlu.to_value(m, int)
        self.coeffs = dlu.to_value(coeffs)
        self.k = dlu.to_value(k)

    def calculate(self, coordinates, diameter):
        """Evaluate every mode in this equal-width group."""
        calculate = lambda n, m, c, p: dlu.zernike_fast(
            n, m, c, p, coordinates, diameter
        )
        return vmap(calculate)(self.n, self.m, self.coeffs, self.k)


class Norm(Base):
    """Normalise and scale sampled basis modes over their aperture support.

    Parameters
    ----------
    mode : str
        One of ``"l1"``, ``"l2"``, ``"max"``, ``"rms"``, or ``"p2v"``.
    scale : ArrayLike
        Physical scale applied after each mode is normalised.
    """

    mode: str = eqx.field(static=True)
    scale: Array

    def __init__(self, mode="rms", scale=1.0):
        mode = str(mode).lower()
        if mode not in ("l1", "l2", "max", "rms", "p2v"):
            raise ValueError("mode must be one of l1, l2, max, rms, or p2v.")
        self.mode = mode
        self.scale = dlu.to_value(scale)

    def __call__(self, basis, support):
        """Normalise every basis vector over its matching aperture support."""
        while support.ndim < basis.ndim:
            support = np.expand_dims(support, -3)
        norm = getattr(dlu, f"{self.mode}_norm")
        value = norm(basis, mask=support, axis=(-2, -1), keepdims=True)
        return self.scale * basis / value


class _ApertureData(Base):
    """Sampled geometry passed internally to an OPD definition.

    ``transmission`` is the final downsampled pupil. ``support`` describes the
    primary or individual sub-apertures used to clip and normalise OPD modes.
    ``diameter`` and optional ``(x, y)`` centres define the local OPD geometry.
    """

    transmission: Array
    support: Array
    diameter: Array
    centers: Array | None

    def __init__(self, transmission, support, diameter, centers=None):
        self.transmission = dlu.to_value(transmission, dtype=None)
        self.support = dlu.to_value(support, dtype=bool)
        self.diameter = dlu.to_value(diameter)
        self.centers = dlu.to_value(centers, optional=True)


class OPDDef(Base):
    """Define sampled OPD data from coordinates and aperture geometry."""

    @abstractmethod
    def calculate(self, coordinates, support, diameter, centers=None):
        """Return the sampled representation required by this definition."""


class ZernikeDef(OPDDef):
    """Configure a support-clipped, explicitly sampled Zernike basis.

    Parameters
    ----------
    nolls : ArrayLike or None
        One-dimensional collection of Noll indices.
    orders : ArrayLike or None
        Radial orders expanded into their complete Noll-index sequences. Exactly one
        of ``nolls`` or ``orders`` must be provided.
    oversize : float
        Fractional enlargement of the aperture diameter used to sample the modes.
    norm : Norm or None
        Optional support-aware normalisation and physical scaling.
    method : str
        ``"padded"`` evaluates one zero-padded vectorised group. ``"mapped"``
        evaluates separate vectorised groups for each radial-term count.

    Notes
    -----
    This definition returns sampled basis data. Calling an ``ApertureBuilder``
    performs the separate materialisation into a ``Basis`` parameterisation.
    """

    nolls: Array
    groups: tuple[ZernikeGroup, ...]
    order: Array
    oversize: Array
    norm: Norm | None
    method: str = eqx.field(static=True)

    def __init__(
        self, nolls=None, orders=None, oversize=0.01, norm=None, method="padded"
    ):
        # Validate and expand the requested Zernike indices
        if (nolls is None) == (orders is None):
            raise ValueError("Provide exactly one of nolls or orders.")
        if orders is not None:
            orders = np.atleast_1d(dlu.to_value(orders, int))
            if orders.ndim != 1 or orders.size == 0:
                raise ValueError("orders must contain at least one radial order.")
            nolls = dlu.radial_orders_to_indices(orders)

        self.nolls = np.atleast_1d(dlu.to_value(nolls, int))
        if self.nolls.ndim != 1 or self.nolls.size == 0:
            raise ValueError("nolls must contain at least one Noll index.")
        if np.any(self.nolls < 1):
            raise ValueError("nolls must contain positive Noll indices.")

        # Build the static evaluation topology and output ordering
        method = str(method).lower()
        if method not in ("padded", "mapped"):
            raise ValueError("method must be either 'padded' or 'mapped'.")
        self.method = method
        self.groups = _zernike_groups(map(int, self.nolls), method)
        indices = np.concatenate(tuple(group.indices for group in self.groups))
        self.order = np.argsort(indices)

        # Store the basis sampling and normalisation definitions
        self.oversize = dlu.to_value(oversize)
        if norm is not None and not isinstance(norm, Norm):
            raise TypeError("norm must be a Norm or None.")
        self.norm = norm

    def calculate(self, coordinates, support, diameter, centers=None):
        """Sample, normalise, and clip the configured Zernike basis."""
        # Prepare the enlarged basis diameter
        diameter = np.asarray(diameter) * (1 + self.oversize)

        # Evaluate and restore the configured Noll ordering
        def calculate_basis(coords):
            calculate = lambda group: group.calculate(coords, diameter)
            is_group = lambda value: isinstance(value, ZernikeGroup)
            bases = jtu.map(calculate, self.groups, is_leaf=is_group)
            return np.concatenate(bases)[self.order]

        # Generate the global or translated local bases
        if centers is not None:
            calculate = lambda c: calculate_basis(dlu.translate_coords(coordinates, c))
            basis = vmap(calculate)(centers)
        else:
            basis = calculate_basis(coordinates)

        # Promote and apply the aperture supports
        while support.ndim < basis.ndim:
            support = np.expand_dims(support, -3)
        basis = basis * support

        # Normalise the supported basis when requested
        return basis if self.norm is None else self.norm(basis, support)


class GridBuilder(Base):
    """Base class for construction-time objects evaluated on a ``GridSpec``."""

    def validate(self, grid, transform):
        """Validate the sampling grid and optional coordinate transformation."""
        if not isinstance(grid, GridSpec):
            raise TypeError("grid must be a GridSpec.")
        if grid.n is None or grid.d is None:
            raise ValueError("grid must define n and d.")
        if grid.ndim != 2:
            raise ValueError("grid must have two dimensions.")
        if grid.d.ndim != 1 or (grid.c is not None and grid.c.ndim != 1):
            raise ValueError("batched grids are not yet supported by GridBuilder.")
        if transform is not None and not isinstance(transform, CoordTransform):
            raise TypeError("transform must be a CoordTransform or None.")

    def build(self, grid, transform=None, jit=True):
        """Evaluate this builder on a grid.

        Parameters
        ----------
        grid : GridSpec
            Two-dimensional output sampling grid.
        transform : CoordTransform or None
            Optional map from grid coordinates into the builder's local frame.
        jit : bool
            Use the shared compiled construction path. Defaults to ``True``;
            repeated calls reuse compiled programs for matching static topology.

        Returns
        -------
        Any
            The concrete return contract is defined by the builder subclass.
        """
        self.validate(grid, transform)
        build_fn = eqx.filter_jit(self._build) if jit else self._build
        return build_fn(grid, transform)

    @abstractmethod
    def _build(self, grid, transform):
        """Evaluate this builder on an already validated grid."""


class ApertureBuilder(GridBuilder):
    """Compose a primary, obscurations, and optional OPD definition on a grid.

    This is the general construction API. Named telescope builders retain simple
    numeric constructors and act as templates which assemble these definitions.
    One-dimensional square-grid specifications are promoted to two spatial axes
    before validation; explicit two-dimensional grids are preserved unchanged.

    Parameters
    ----------
    primary : Shape
        Transmissive primary geometry.
    obscurations : list or tuple of Shape
        Geometry removed from the primary transmission.
    opd : OPDDef or None
        Optional OPD data definition evaluated over the primary support.
    oversample : int or tuple of int
        Sampling factor used before downsampling hard-edged geometry.

    Returns
    -------
    transmission : Array
        Returned by ``build`` when ``opd`` is ``None`` and support is not requested.
    transmission, support : tuple of Array
        Returned when ``opd`` is ``None`` and ``return_support=True``.
    transmission, opd_data : tuple of Array
        Returned when ``opd`` is present.
    transmission, opd_data, support : tuple of Array
        Returned when ``opd`` is present and ``return_support=True``.
    """

    primary: Shape
    obscurations: tuple
    opd: OPDDef | None
    oversample: tuple[int, int] = eqx.field(static=True)

    def __init__(self, primary, obscurations=(), opd=None, oversample=5):
        # Validate the primary and obscuration geometry
        if not isinstance(primary, Shape):
            raise TypeError("primary must be a Shape.")
        if not isinstance(obscurations, (list, tuple)):
            raise TypeError("obscurations must be a list or tuple of Shape objects.")
        obscurations = tuple(obscurations)
        if not all(isinstance(shape, Shape) for shape in obscurations):
            raise TypeError("obscurations must contain only Shape objects.")

        # Store the geometry and optional OPD definition
        self.primary = primary
        self.obscurations = obscurations
        if opd is not None and not isinstance(opd, OPDDef):
            raise TypeError("opd must be an OPDDef or None.")
        self.opd = opd
        self.oversample = dlu.as_size(oversample, 2, "oversample")

    @staticmethod
    def _promote_grid(grid):
        """Promote a scalar square-grid specification to two spatial axes."""
        if isinstance(grid, GridSpec) and grid.ndim == 1:
            return grid.broadcast(2)
        return grid

    def build(self, grid, transform=None, jit=True, return_support=False):
        """Build on a 2D grid, optionally returning the aperture support."""
        grid = self._promote_grid(grid)
        self.validate(grid, transform)
        build_fn = eqx.filter_jit(self._build) if jit else self._build
        return build_fn(grid, transform, return_support)

    @staticmethod
    def _evaluate(shape, grid, transform):
        """Evaluate one shape on a possibly transformed grid."""
        coordinates = grid.transformed(transform)
        return shape.evaluate(coordinates=coordinates, pixel_scale=grid.d * grid.scale)

    def aperture_data(self, grid, transform):
        """Sample the aperture and retain its native primary support."""
        # Generate the oversampled aperture components
        fine = grid.oversample(self.oversample)
        primary = self._evaluate(self.primary, fine, transform)
        eval_fn = lambda s: 1 - self._evaluate(s, fine, transform)
        obscurations = [eval_fn(s) for s in self.obscurations]

        # Combine and downsample the transmission and support
        transmissions = np.stack([primary, *obscurations])
        transmission = dlu.downsample(transmissions.prod(0), self.oversample)
        support = dlu.downsample(primary, self.oversample) > 0

        # Get the physical diameter required for OPD generation
        extent = self.primary.extent
        if extent is None and self.opd is not None:
            raise ValueError("primary must define an extent when opd is provided.")
        diameter = np.asarray(0.0) if extent is None else 2 * extent

        # Package the sampled aperture data
        return _ApertureData(transmission, support, diameter)

    def _build(self, grid, transform, return_support=False):
        """Build sampled transmission, OPD data, and optional support."""
        # Generate the aperture transmission and support
        aperture = self.aperture_data(grid, transform)

        # Return the transmission when no OPD is defined
        if self.opd is None:
            if return_support:
                return aperture.transmission, aperture.support
            return aperture.transmission

        # Generate the supported OPD data
        basis = self.opd.calculate(
            grid.transformed(transform),
            aperture.support,
            aperture.diameter,
            aperture.centers,
        )

        # Return the requested components
        if return_support:
            return aperture.transmission, basis, aperture.support
        return aperture.transmission, basis

    def __call__(
        self,
        grid,
        transform=None,
        coeffs=None,
        key=None,
        normalise=True,
        jit=True,
        coefficients=None,
    ):
        """Materialise this definition as a globally sampled ``Optic``.

        ``coeffs`` and ``key`` are mutually exclusive. With an OPD definition,
        explicit coefficients are used directly, a key draws standard-normal values,
        and omitting both initialises zero coefficients. ``normalise`` retains the
        existing wavefront-normalisation meaning of ``Optic.normalise``.
        """
        coeffs = _resolve_coeffs(coeffs, coefficients)
        components = self.build(grid, transform, jit)
        if self.opd is None:
            return Optic(transmission=components, normalise=normalise)
        transmission, basis = components[:2]
        opd = _explicit_basis(basis, coeffs, key)
        return Optic(transmission=transmission, opd=opd, normalise=normalise)


class SparseApertureBuilder(ApertureBuilder):
    """Repeat one local aperture shape over explicit ``(x, y)`` centres.

    Parameters
    ----------
    subaperture : Shape
        Geometry shared by every local aperture.
    centers : ArrayLike
        Centre coordinates with shape ``(n_apertures, 2)``.
    obscurations : list or tuple of Shape
        Local geometry repeated inside every sub-aperture.
    global_obscurations : list or tuple of Shape
        Geometry removed after assembling the full global pupil.
    opd, oversample
        As defined by ``ApertureBuilder``.

    Notes
    -----
    Calling the builder returns one globally sampled compound pupil by default.
    Passing ``sparse=True`` instead returns one local transmission plus explicit
    centres. Global obscurations cannot be represented by that shared local
    transmission.
    """

    centers: Array
    global_obscurations: tuple

    def __init__(
        self,
        subaperture,
        centers,
        obscurations=(),
        global_obscurations=(),
        opd=None,
        oversample=5,
    ):
        # Validate and store the sub-aperture centres
        centers = dlu.to_value(centers)
        if centers.ndim != 2 or centers.shape[-1] != 2:
            raise ValueError("centers must have shape (n_apertures, 2).")
        self.centers = centers

        # Validate and store the global obscuration geometry
        if not isinstance(global_obscurations, (list, tuple)):
            raise TypeError(
                "global_obscurations must be a list or tuple of Shape objects."
            )
        global_obscurations = tuple(global_obscurations)
        if not all(isinstance(shape, Shape) for shape in global_obscurations):
            raise TypeError("global_obscurations must contain only Shape objects.")
        self.global_obscurations = global_obscurations

        # Initialise the shared local aperture definition
        super().__init__(subaperture, obscurations, opd, oversample)

    def _component(self, center, grid, transform):
        """Evaluate one complete local aperture component."""
        # Generate coordinates in the local aperture frame
        coordinates = dlu.translate_coords(grid.transformed(transform), center)
        pixel_scale = grid.d * grid.scale

        # Evaluate the primary and local obscurations
        transmission = self.primary.evaluate(
            coordinates=coordinates, pixel_scale=pixel_scale
        )
        for shape in self.obscurations:
            transmission *= 1 - shape.evaluate(
                coordinates=coordinates, pixel_scale=pixel_scale
            )

        return transmission

    def _primary_component(self, center, grid, transform):
        """Evaluate one local primary aperture support."""
        coordinates = dlu.translate_coords(grid.transformed(transform), center)
        return self.primary.evaluate(
            coordinates=coordinates, pixel_scale=grid.d * grid.scale
        )

    def aperture_data(self, grid, transform):
        """Sample global component transmissions and non-redundant supports."""
        # Generate the oversampled components and primary supports
        fine = grid.oversample(self.oversample)
        comp_fn = lambda c: self._component(c, fine, transform)
        prim_fn = lambda c: self._primary_component(c, fine, transform)
        components = vmap(comp_fn)(self.centers)
        primaries = vmap(prim_fn)(self.centers)

        # Generate the combined global obscuration mask
        eval_fn = lambda s: 1 - self._evaluate(s, fine, transform)
        obscurations = [eval_fn(s) for s in self.global_obscurations]
        mask = np.stack([np.ones_like(components[0]), *obscurations]).prod(0)

        # Combine and downsample the aperture components
        aperture = np.clip(components.sum(0), 0.0, 1.0)
        transmission = dlu.downsample(aperture * mask, self.oversample)
        primaries = dlu.downsample(primaries, self.oversample)
        support = dlu.non_redundant_support(primaries)

        # Get the physical sub-aperture diameter
        extent = self.primary.extent
        if extent is None and self.opd is not None:
            raise ValueError("subaperture must define an extent when opd is used.")
        diameter = np.asarray(0.0) if extent is None else 2 * extent

        # Package the sampled aperture data
        return _ApertureData(transmission, support, diameter, self.centers)

    def _sparse_data(self, grid, transform):
        """Sample the shared transmission and optional local OPD basis."""
        # Sample the shared local transmission
        fine = grid.oversample(self.oversample)
        transmission = self._component(np.zeros(2), fine, transform)
        transmission = dlu.downsample(transmission, self.oversample)
        if self.opd is None:
            return transmission, None

        # Generate the supported local OPD basis
        coordinates = grid.transformed(transform)
        support = self._primary_component(np.zeros(2), grid, transform) > 0
        diameter = 2 * self.primary.extent
        basis = self.opd.calculate(coordinates, support, diameter)
        return transmission, basis

    def __call__(
        self,
        grid,
        transform=None,
        coeffs=None,
        key=None,
        normalise=True,
        jit=True,
        sparse=False,
        shared=False,
        coefficients=None,
    ):
        """Materialise this definition as an ``Optic`` or ``SparseOptic``.

        By default this uses the globally sampled ``Optic`` contract. With
        ``sparse=True``, ``shared=True`` keeps the native OPD coefficient shape for
        every aperture, while ``shared=False`` adds a leading aperture axis.
        Explicit coefficients may use either representation. A supplied random key
        follows the selected layout.
        """
        coeffs = _resolve_coeffs(coeffs, coefficients)

        # Materialise a global optic unless sparse output is requested
        if not sparse:
            return super().__call__(grid, transform, coeffs, key, normalise, jit)

        # Validate sparse construction requirements
        grid = self._promote_grid(grid)
        self.validate(grid, transform)
        if self.global_obscurations:
            raise ValueError(
                "Global obscurations cannot be represented by one shared local "
                "SparseOptic transmission; use sparse=False instead."
            )
        if transform is not None:
            raise ValueError(
                "transform cannot be represented by one shared local SparseOptic; "
                "use sparse=False instead."
            )

        # Sample the shared local transmission and OPD basis
        build_fn = eqx.filter_jit(self._sparse_data) if jit else self._sparse_data
        transmission, basis = build_fn(grid, transform)
        if self.opd is None:
            return SparseOptic(
                self.centers, transmission=transmission, normalise=normalise
            )

        # Materialise shared or aperture-dependent OPD coefficients
        shape = basis.shape[:-2]
        initial_shape = shape if shared else (len(self.centers),) + shape
        opd = _explicit_basis(
            basis,
            coeffs,
            key,
            shape=shape,
            initial_shape=initial_shape,
        )

        # Package the locally sampled sparse optic
        return SparseOptic(
            self.centers, transmission=transmission, opd=opd, normalise=normalise
        )
