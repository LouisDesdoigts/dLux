"""Construction-time generators for sampled optical components."""

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax.numpy as np
import jax.random as jr
import zodiax as zdx
from jax import Array

import dLux.utils as dlu

from .grids import CoordTransform, GridSpec
from .parametric import Shape

__all__ = [
    "GridBuilder",
    "OPDDef",
    "ApertureData",
    "Norm",
    "ZernikeDef",
    "ApertureBuilder",
    "SparseApertureBuilder",
]


class Norm(zdx.Base):
    """Normalize and scale sampled basis modes over their aperture support.

    Parameters
    ----------
    mode : str
        One of ``"l1"``, ``"l2"``, ``"max"``, ``"rms"``, or ``"p2v"``.
    scale : ArrayLike
        Physical scale applied after each mode is normalized.
    """

    mode: str = eqx.field(static=True)
    scale: Array = eqx.field(converter=dlu.as_float)

    def __init__(self, mode="rms", scale=1.0):
        mode = str(mode).lower()
        if mode not in ("l1", "l2", "max", "rms", "p2v"):
            raise ValueError("mode must be one of l1, l2, max, rms, or p2v.")
        self.mode = mode
        self.scale = scale

    def __call__(self, basis, support):
        """Normalize every basis vector over its matching aperture support."""
        while support.ndim < basis.ndim:
            support = np.expand_dims(support, -3)
        norm = getattr(dlu, f"{self.mode}_norm")
        value = norm(basis, mask=support, axis=(-2, -1), keepdims=True)
        return self.scale * basis / value


class ApertureData(zdx.Base):
    """Sampled geometry passed internally to an OPD definition.

    ``transmission`` is the final downsampled pupil. ``support`` describes the
    primary or individual sub-apertures used to clip and normalize OPD modes.
    ``diameter`` and optional ``(x, y)`` centres define the local OPD geometry.
    """

    transmission: Array
    support: Array
    diameter: Array = eqx.field(converter=dlu.as_float)
    centers: Array | None = eqx.field(converter=dlu.as_array)

    def __init__(self, transmission, support, diameter, centers=None):
        self.transmission = np.asarray(transmission)
        self.support = np.asarray(support, dtype=bool)
        self.diameter = diameter
        self.centers = centers


def _explicit_basis(
    basis,
    coefficients=None,
    key=None,
    coefficient_shape=None,
    initial_shape=None,
):
    """Materialize a sampled OPD basis with explicit or random coefficients."""
    from .parametric import Basis

    if coefficients is not None and key is not None:
        raise ValueError("Provide only one of coefficients or key.")
    coefficient_shape = (
        basis.shape[:-2] if coefficient_shape is None else tuple(coefficient_shape)
    )
    if key is not None:
        initial_shape = coefficient_shape if initial_shape is None else initial_shape
        coefficients = jr.normal(key, initial_shape)
    elif coefficients is None and initial_shape is not None:
        coefficients = np.zeros(initial_shape)
    return Basis(
        basis,
        coefficients=coefficients,
        coefficient_shape=coefficient_shape,
    )


class OPDDef(zdx.Base):
    """Define sampled OPD data from coordinates and aperture geometry."""

    @abstractmethod
    def calculate(self, coordinates, support, diameter, centers=None):
        """Return the sampled representation required by this definition."""


class ZernikeDef(OPDDef):
    """Configure a support-clipped, explicitly sampled Zernike basis.

    Parameters
    ----------
    nolls : ArrayLike
        One-dimensional collection of Noll indices.
    oversize : float
        Fractional enlargement of the aperture diameter used to sample the modes.
    norm : Norm or None
        Optional support-aware normalization and physical scaling.

    Notes
    -----
    This definition returns sampled basis data. ``ApertureBuilder.as_optic`` performs
    the separate materialization into a ``Basis`` parameterization.
    """

    nolls: Array
    oversize: Array = eqx.field(converter=dlu.as_float)
    norm: Norm | None

    def __init__(self, nolls, oversize=0.01, norm=None):
        self.nolls = np.atleast_1d(np.asarray(nolls, dtype=int))
        if self.nolls.ndim != 1 or self.nolls.size == 0:
            raise ValueError("nolls must contain at least one Noll index.")
        self.oversize = oversize
        if norm is not None and not isinstance(norm, Norm):
            raise TypeError("norm must be a Norm or None.")
        self.norm = norm

    def calculate(self, coordinates, support, diameter, centers=None):
        """Sample, normalize, and clip the configured Zernike basis."""
        diameter = np.asarray(diameter) * (1 + self.oversize)
        if centers is None:
            basis = dlu.zernike_basis(self.nolls, coordinates, diameter)
        else:
            bases = [
                dlu.zernike_basis(
                    self.nolls,
                    dlu.translate_coords(coordinates, center),
                    diameter,
                )
                for center in centers
            ]
            basis = np.stack(bases)
        while support.ndim < basis.ndim:
            support = np.expand_dims(support, -3)
        basis = basis * support
        return basis if self.norm is None else self.norm(basis, support)


class GridBuilder(zdx.Base):
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

    def build(self, grid, transform=None, jit=False):
        """Evaluate this builder on a grid.

        Parameters
        ----------
        grid : GridSpec
            Two-dimensional output sampling grid.
        transform : CoordTransform or None
            Optional map from grid coordinates into the builder's local frame.
        jit : bool
            Compile the internal build operation with ``eqx.filter_jit``.

        Returns
        -------
        Any
            The concrete return contract is defined by the builder subclass.
        """
        self.validate(grid, transform)
        if jit:
            return eqx.filter_jit(self._build)(grid, transform)
        return self._build(grid, transform)

    @abstractmethod
    def _build(self, grid, transform):  # pragma: no cover
        """Evaluate this builder on an already validated grid."""


class ApertureBuilder(GridBuilder):
    """Compose a primary, obscurations, and optional OPD definition on a grid.

    This is the general construction API. Named telescope builders retain simple
    numeric constructors and act as templates which assemble these definitions.

    Parameters
    ----------
    primary : Shape
        Transmissive primary geometry.
    obscurations : sequence or dict of Shape
        Geometry removed from the primary transmission.
    opd : OPDDef or None
        Optional OPD data definition evaluated over the primary support.
    oversample : int or tuple of int
        Sampling factor used before downsampling hard-edged geometry.
    return_support : bool
        Append the primary support to OPD-bearing ``build`` results.

    Returns
    -------
    transmission : Array
        Returned by ``build`` when ``opd`` is ``None``.
    transmission, opd_data : tuple of Array
        Returned when ``opd`` is present.
    transmission, opd_data, support : tuple of Array
        Returned when ``opd`` is present and ``return_support=True``.
    """

    primary: Shape
    obscurations: dict
    opd: OPDDef | None
    oversample: tuple[int, int] = eqx.field(static=True)
    return_support: bool = eqx.field(static=True)

    def __init__(
        self, primary, obscurations=(), opd=None, oversample=5, return_support=False
    ):
        if not isinstance(primary, Shape):
            raise TypeError("primary must be a Shape.")
        if isinstance(obscurations, dict):
            obscurations = list(obscurations.items())
        else:
            obscurations = list(obscurations)
        self.primary = primary
        self.obscurations = dlu.list2dictionary(obscurations, True, Shape)
        if opd is not None and not isinstance(opd, OPDDef):
            raise TypeError("opd must be an OPDDef or None.")
        self.opd = opd
        self.oversample = dlu.as_size(oversample, 2, "oversample")
        self.return_support = bool(return_support)

    @staticmethod
    def _evaluate(shape, grid, transform):
        coordinates = grid.transformed(transform)
        return shape.evaluate(
            coordinates=coordinates,
            pixel_scale=grid.d * grid.scale,
        )

    def aperture_data(self, grid, transform):
        """Sample the aperture and retain its native primary support."""
        fine = grid.oversample(self.oversample)
        primary = self._evaluate(self.primary, fine, transform)
        transmissions = [primary]
        transmissions.extend(
            1 - self._evaluate(shape, fine, transform)
            for shape in self.obscurations.values()
        )
        transmission = dlu.downsample(
            np.prod(np.stack(transmissions), 0), self.oversample
        )
        support = dlu.downsample(primary, self.oversample) > 0
        if self.primary.extent is None:
            if self.opd is not None:
                raise ValueError("primary must define an extent when opd is provided.")
            diameter = np.asarray(0.0)
        else:
            diameter = 2 * self.primary.extent
        return ApertureData(transmission, support, diameter)

    def _build(self, grid, transform):
        aperture = self.aperture_data(grid, transform)
        if self.opd is None:
            return aperture.transmission
        basis = self.opd.calculate(
            grid.transformed(transform),
            aperture.support,
            aperture.diameter,
            aperture.centers,
        )
        if self.return_support:
            return aperture.transmission, basis, aperture.support
        return aperture.transmission, basis

    def as_optic(
        self,
        grid,
        transform=None,
        coefficients=None,
        key=None,
        normalise=False,
        jit=False,
    ):
        """Materialize this definition as a globally sampled ``Optic``.

        ``coefficients`` and ``key`` are mutually exclusive. With an OPD definition,
        explicit coefficients are used directly, a key draws standard-normal values,
        and omitting both initializes zero coefficients. ``normalise`` retains the
        existing wavefront-normalization meaning of ``Optic.normalise``.
        """
        from .layers import Optic

        components = self.build(grid, transform=transform, jit=jit)
        if self.opd is None:
            return Optic(transmission=components, normalise=normalise)
        transmission, basis = components[:2]
        opd = _explicit_basis(basis, coefficients, key)
        return Optic(transmission=transmission, opd=opd, normalise=normalise)

    def as_sparse_optic(self, grid, **kwargs):
        """Materialize compatible geometry as a locally sampled ``SparseOptic``."""
        raise NotImplementedError  # pragma: no cover


class SparseApertureBuilder(ApertureBuilder):
    """Repeat one local aperture shape over explicit ``(x, y)`` centres.

    Parameters
    ----------
    subaperture : Shape
        Geometry shared by every local aperture.
    centers : ArrayLike
        Centre coordinates with shape ``(n_apertures, 2)``.
    obscurations : sequence or dict of Shape
        Local geometry repeated inside every sub-aperture.
    global_obscurations : sequence or dict of Shape
        Geometry removed after assembling the full global pupil.
    opd, oversample, return_support
        As defined by ``ApertureBuilder``.

    Notes
    -----
    ``as_optic`` returns one globally sampled compound pupil. ``as_sparse_optic``
    instead returns one local transmission plus explicit centres. Global
    obscurations cannot be represented by that shared local transmission.
    """

    centers: Array = eqx.field(converter=dlu.as_float)
    global_obscurations: dict

    def __init__(
        self,
        subaperture,
        centers,
        obscurations=(),
        global_obscurations=(),
        opd=None,
        oversample=5,
        return_support=False,
    ):
        centers = dlu.as_float(centers)
        if centers.ndim != 2 or centers.shape[-1] != 2:
            raise ValueError("centers must have shape (n_apertures, 2).")
        self.centers = centers
        if isinstance(global_obscurations, dict):
            global_obscurations = list(global_obscurations.items())
        else:
            global_obscurations = list(global_obscurations)
        self.global_obscurations = dlu.list2dictionary(
            global_obscurations, True, Shape
        )
        super().__init__(
            subaperture, obscurations, opd, oversample, return_support
        )

    def _component(self, center, grid, transform):
        coordinates = dlu.translate_coords(grid.transformed(transform), center)
        pixel_scale = grid.d * grid.scale
        transmission = self.primary.evaluate(
            coordinates=coordinates, pixel_scale=pixel_scale
        )
        for shape in self.obscurations.values():
            transmission *= 1 - shape.evaluate(
                coordinates=coordinates, pixel_scale=pixel_scale
            )
        return transmission

    def _primary_component(self, center, grid, transform):
        coordinates = dlu.translate_coords(grid.transformed(transform), center)
        return self.primary.evaluate(
            coordinates=coordinates, pixel_scale=grid.d * grid.scale
        )

    def aperture_data(self, grid, transform):
        """Sample global component transmissions and non-redundant supports."""
        fine = grid.oversample(self.oversample)
        components = np.stack(
            [self._component(center, fine, transform) for center in self.centers]
        )
        primaries = np.stack(
            [
                self._primary_component(center, fine, transform)
                for center in self.centers
            ]
        )
        global_transmission = np.prod(
            np.stack(
                [np.ones_like(components[0])]
                + [
                    1 - self._evaluate(shape, fine, transform)
                    for shape in self.global_obscurations.values()
                ]
            ),
            0,
        )
        transmission = dlu.downsample(
            np.clip(components.sum(0), 0.0, 1.0) * global_transmission,
            self.oversample,
        )
        primaries = dlu.downsample(primaries, self.oversample)
        support = dlu.non_redundant_support(primaries)
        if self.primary.extent is None:
            if self.opd is not None:
                raise ValueError("subaperture must define an extent when opd is used.")
            diameter = np.asarray(0.0)
        else:
            diameter = 2 * self.primary.extent
        return ApertureData(transmission, support, diameter, self.centers)

    def as_sparse_optic(
        self,
        grid,
        transform=None,
        coefficients=None,
        key=None,
        shared=False,
        normalise=False,
    ):
        """Materialize the definition as a locally sampled ``SparseOptic``.

        ``shared=True`` keeps the native OPD coefficient shape for every aperture;
        ``shared=False`` adds a leading aperture axis. Explicit coefficients may use
        either representation. A supplied random key follows the selected layout.
        """
        from .layers import SparseOptic

        self.validate(grid, transform)
        if self.global_obscurations:
            raise ValueError(
                "Global obscurations cannot be represented by one shared local "
                "SparseOptic transmission; use as_optic instead."
            )
        fine = grid.oversample(self.oversample)
        transmission = self._component(np.zeros(2), fine, transform)
        transmission = dlu.downsample(transmission, self.oversample)
        if self.opd is None:
            return SparseOptic(
                self.centers, transmission=transmission, normalise=normalise
            )
        coordinates = grid.transformed(transform)
        support = self._primary_component(np.zeros(2), grid, transform) > 0
        diameter = 2 * self.primary.extent
        basis = self.opd.calculate(coordinates, support, diameter)
        shape = basis.shape[:-2]
        initial_shape = shape if shared else (len(self.centers),) + shape
        opd = _explicit_basis(
            basis,
            coefficients,
            key,
            coefficient_shape=shape,
            initial_shape=initial_shape,
        )
        return SparseOptic(
            self.centers,
            transmission=transmission,
            opd=opd,
            normalise=normalise,
        )
