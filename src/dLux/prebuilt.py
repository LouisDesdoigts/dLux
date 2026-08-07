"""Prebuilt generic and representative aperture models."""

import equinox as eqx
import jax.numpy as np
from jax import vmap

import dLux.utils as dlu

from .builders import ApertureBuilder, SparseApertureBuilder, _initialise_coefficients
from .grids import Affine, PasteSpec
from .parametric import (
    Circle,
    Rectangle,
    RegularPolygon,
    PastedBasis,
    Shape,
    Spider,
    TransformedShape,
)

__all__ = [
    "SimpleCircular",
    "SegmentedHex",
    "NRMLike",
    "HSTLike",
    "JWSTLike",
    "JWSTNRMLike",
    "EuclidLike",
]


class SimpleCircular(ApertureBuilder):
    """Pedagogical circular pupil assembled with a compact numeric API.

    Parameters
    ----------
    diameter : float
        Primary diameter in the grid's physical length unit.
    secondary_diameter : float or None
        Optional central obscuration diameter.
    spider_width, spider_angles
        Optional radial support width and one-dimensional angles in degrees. Both
        must be supplied together.
    opd, oversample
        As defined by ``ApertureBuilder``.
    """

    def __init__(
        self,
        diameter,
        secondary_diameter=None,
        spider_width=None,
        spider_angles=None,
        opd=None,
        oversample=5,
    ):
        if (spider_width is None) != (spider_angles is None):
            raise ValueError("spider_width and spider_angles must both be provided.")
        obscurations = []
        if secondary_diameter is not None:
            obscurations.append(Circle(secondary_diameter))
        if spider_width is not None:
            obscurations.append(Spider(spider_width, spider_angles))
        super().__init__(Circle(diameter), obscurations, opd, oversample)


class SegmentedHex(SparseApertureBuilder):
    """Pedagogical hexagonal tiling assembled with a compact numeric API.

    ``segment_diameter`` is the point-to-point diameter of the circumscribed circle;
    ``segment_f2f`` is the flat-to-flat width. Exactly one must be provided.
    Obscurations are global structures applied across the assembled pupil.

    Parameters
    ----------
    nrings : int
        Number of hexagonal rings including the central segment.
    segment_diameter, segment_f2f : float or None
        Alternative segment-size conventions; exactly one is required.
    gap : float
        Edge-to-edge gap between adjacent segments.
    remove_center : bool
        Remove the central segment after generating the complete tiling.
    obscurations : list or tuple of Shape
        Global geometry removed from the assembled pupil.
    paste_method : str
        Compact-stamp assembly strategy. ``"scan"`` is sequential and
        memory-efficient; ``"scatter"`` uses per-pixel indices to expose more
        parallelism at higher memory cost. Benchmark both for large systems because
        performance depends on the hardware and pupil sampling.
    opd, oversample
        As defined by ``ApertureBuilder``.

    Notes
    -----
    Both methods are numerically equivalent. Changing ``paste_method`` through an
    immutable update may trigger a separate JAX compilation.
    """

    paste_method: str

    def __init__(
        self,
        nrings,
        segment_diameter=None,
        segment_f2f=None,
        gap=0.0,
        remove_center=False,
        obscurations=(),
        opd=None,
        oversample=5,
        paste_method="scan",
    ):
        if (segment_diameter is None) == (segment_f2f is None):
            raise ValueError("Provide exactly one of segment_diameter or segment_f2f.")
        paste_method = str(paste_method).lower()
        if paste_method not in ("scan", "scatter"):
            raise ValueError("paste_method must be either 'scan' or 'scatter'.")
        if segment_diameter is None:
            segment_diameter = 2 * segment_f2f / np.sqrt(3)
        centers = dlu.segmented_hex_cens(nrings, segment_diameter / 2, gap)
        if remove_center:
            centers = centers[1:]
        super().__init__(
            RegularPolygon(6, segment_diameter),
            centers,
            global_obscurations=obscurations,
            opd=opd,
            oversample=oversample,
        )
        self.paste_method = paste_method

    def _validate_ideal(self, grid, transform):
        """Validate an untransformed grid for ideal segmented construction."""
        super().validate(grid, transform)
        if transform is not None:
            raise ValueError("SegmentedHex does not support coordinate transforms.")

    def build(self, grid, transform=None, jit=False, return_support=False):
        """Build the ideal segmented aperture on an untransformed grid."""
        # Promote and validate the construction grid
        grid = self._promote_grid(grid)
        self._validate_ideal(grid, transform)

        # Compile the fixed-shape pasted calculation when requested
        if jit:
            if self.opd is not None or return_support:
                raise ValueError(
                    "jit is not yet supported for segmented OPD or support data."
                )
            fine, spec = self._stamp_data(grid)
            build_fn = eqx.filter_jit(self._assemble_transmission)
            return build_fn(fine, spec)

        # Use compact stamps for ordinary eager construction
        return self._build(grid, transform, return_support)

    def _stamp_data(self, grid):
        """Return the oversampled grid and compact placement specification."""
        # Generate the oversampled grid and fixed stamp topology
        fine = grid.oversample(self.oversample)
        spec = PasteSpec.from_grid(fine, self.centers, self.primary.extent)
        return fine, spec

    def _assemble_transmission(self, fine, spec):
        """Assemble a transmission from prepared segment stamps."""
        # Generate each compact segment transmission in parallel
        coordinates = spec.coordinates
        scale = fine.d * fine.scale
        eval_fn = lambda c: self.primary.evaluate(coordinates=c, pixel_scale=scale)
        components = vmap(eval_fn)(coordinates)

        for shape in self.obscurations:
            eval_fn = lambda c: shape.evaluate(coordinates=c, pixel_scale=scale)
            components *= 1 - vmap(eval_fn)(coordinates)

        # Paste the compact segments into the oversampled global pupil
        aperture = spec.paste(components, self.paste_method)
        aperture = np.clip(aperture, 0.0, 1.0)

        # Apply global obscurations on the complete pupil grid
        mask = np.ones_like(aperture)
        for shape in self.global_obscurations:
            mask *= 1 - self._evaluate(shape, fine, None)

        # Downsample the completed ideal pupil
        return dlu.downsample(aperture * mask, self.oversample)

    def _pasted_transmission(self, grid):
        """Prepare and assemble the compact segment stamps."""
        fine, spec = self._stamp_data(grid)
        return self._assemble_transmission(fine, spec)

    def _assemble_opd(self, spec):
        """Generate compact supported OPD basis stamps."""
        # Generate each compact segment support
        coordinates = spec.coordinates
        eval_fn = lambda c: self.primary.evaluate(coordinates=c, pixel_scale=spec.d)
        support = vmap(eval_fn)(coordinates) > 0

        # Generate the configured OPD basis over each local support
        diameter = 2 * self.primary.extent
        calc_fn = lambda c, s: self.opd.calculate(c, s, diameter)
        basis = vmap(calc_fn)(coordinates, support)
        return basis, support

    def _pasted_opd(self, grid):
        """Prepare and generate the compact OPD basis data."""
        spec = PasteSpec.from_grid(grid, self.centers, self.primary.extent)
        basis, support = self._assemble_opd(spec)
        return basis, support, spec

    def _build(self, grid, transform, return_support=False):
        """Build the ideal pupil using compact segment stamps where possible."""
        if self.opd is not None or return_support:
            return super()._build(grid, transform, return_support)
        return self._pasted_transmission(grid)

    def __call__(
        self,
        grid,
        transform=None,
        coefficients=None,
        key=None,
        normalise=True,
        jit=False,
        sparse=False,
        shared=False,
    ):
        """Materialise a global pasted optic or a genuinely sparse optic."""
        # Retain the local sparse-optic construction path
        if sparse:
            return SparseApertureBuilder.__call__(
                self,
                grid,
                transform=transform,
                coefficients=coefficients,
                key=key,
                normalise=normalise,
                jit=jit,
                sparse=True,
                shared=shared,
            )

        # Promote and validate the ideal construction grid
        from .layers import Optic

        grid = self._promote_grid(grid)
        self._validate_ideal(grid, transform)

        # Generate the compactly assembled pupil transmission
        fine, spec = self._stamp_data(grid)
        build_fn = self._assemble_transmission
        build_fn = eqx.filter_jit(build_fn) if jit else build_fn
        transmission = build_fn(fine, spec)
        if self.opd is None:
            return Optic(transmission=transmission, normalise=normalise)

        # Generate and materialise the compact per-segment OPD basis
        basis, _, spec = self._pasted_opd(grid)
        shape = basis.shape[:-2]
        coefficients = _initialise_coefficients(shape, coefficients, key, shape)
        opd = PastedBasis(basis, spec, coefficients, self.paste_method)
        return Optic(transmission=transmission, opd=opd, normalise=normalise)


class NRMLike(SparseApertureBuilder):
    """Simple NRM with ``(x, y)`` hole centres and one shared hole shape.

    This helper deliberately excludes global stops and obscurations. Sparse
    materialisation always gives every hole independent OPD coefficients.

    Parameters
    ----------
    centers : ArrayLike
        Hole centres with shape ``(n_holes, 2)`` in ``(x, y)`` order.
    hole : Shape
        The single local geometry shared by every hole.
    opd, oversample
        As defined by ``ApertureBuilder``.

    Returns
    -------
    Array or tuple of Array
        ``build`` follows the ``ApertureBuilder`` global-pupil contract.
    SparseOptic
        Calling with ``sparse=True`` creates one local pupil per centre with
        independent OPD coefficients.
    """

    def __init__(self, centers, hole, opd=None, oversample=5):
        if not isinstance(hole, Shape):
            raise TypeError("hole must be a Shape.")
        super().__init__(hole, centers, opd=opd, oversample=oversample)

    def __call__(
        self,
        grid,
        transform=None,
        coefficients=None,
        key=None,
        normalise=True,
        jit=False,
        sparse=False,
    ):
        """Materialise a global NRM, or independent holes with ``sparse=True``."""
        if sparse and coefficients is not None:
            coefficients = np.asarray(coefficients)
            if coefficients.ndim == 0 or coefficients.shape[0] != len(self.centers):
                raise ValueError(
                    "coefficients must have a leading axis matching the holes."
                )
        return super().__call__(
            grid,
            transform=transform,
            coefficients=coefficients,
            key=key,
            normalise=normalise,
            jit=jit,
            sparse=sparse,
            shared=False,
        )


class HSTLike(SimpleCircular):
    """Representative HST-like pupil preserving the legacy dLux geometry.

    This model omits detailed support-pad, baffle, and edge geometry and is intended
    for examples rather than observatory-grade reproduction. Its dimensions are
    carried forward from ``dLux.utils.hst_like`` and are not independently validated
    here against a contemporary observatory pupil model.

    The primary, secondary, and support dimensions may be overridden while retaining
    the HST-like circular-pupil topology.
    """

    def __init__(
        self,
        diameter=2.4,
        secondary_diameter=0.305,
        spider_width=0.038,
        spider_angles=(0, 90, 180, 270),
        opd=None,
        oversample=5,
    ):
        super().__init__(
            diameter=diameter,
            secondary_diameter=secondary_diameter,
            spider_width=spider_width,
            spider_angles=spider_angles,
            opd=opd,
            oversample=oversample,
        )


class JWSTLike(SegmentedHex):
    """Representative JWST-like pupil preserving the legacy dLux geometry.

    This model includes the 18 primary segments and three support arms, but omits
    features such as the secondary-mirror support hinges and detailed pupil edges.
    Its dimensions are carried forward from ``dLux.utils.jwst_like``; it is not an
    STPSF-equivalent or independently validated observatory pupil.

    Segment size, gap, and simplified support geometry may be overridden while the
    18-segment JWST-like topology remains fixed.
    """

    def __init__(
        self,
        segment_diameter=1.524,
        gap=0.007,
        spider_width=0.1,
        spider_angles=(30, 180, 330),
        opd=None,
        oversample=5,
        paste_method="scan",
    ):
        super().__init__(
            nrings=3,
            segment_diameter=segment_diameter,
            gap=gap,
            remove_center=True,
            obscurations=(Spider(spider_width, spider_angles),),
            opd=opd,
            oversample=oversample,
            paste_method=paste_method,
        )


class JWSTNRMLike(NRMLike):
    """Representative seven-hole JWST/NIRISS NRM geometry.

    Hole centres and the nominal 0.8 m flat-to-flat width are taken from the AMIGO
    model. Its fitted pupil-registration offset and detailed mask-edge effects are
    deliberately omitted.

    ``centers`` and ``hole_f2f`` may be overridden for calibrated or deliberately
    perturbed NRM geometries while retaining a shared hexagonal hole shape.
    """

    def __init__(self, centers=None, hole_f2f=0.8, opd=None, oversample=5):
        # Ideal mask coordinates used by AMIGO, excluding its fitted pupil offset.
        if centers is None:
            centers = np.asarray(
                (
                    (0.0, 2.64),
                    (2.28631, 0.0),
                    (-2.28631, 1.32),
                    (2.28631, -1.32),
                    (1.14315, -1.98),
                    (-2.28631, -1.32),
                    (-1.14315, -1.98),
                )
            )
        diameter = 2 * hole_f2f / np.sqrt(3)
        super().__init__(
            centers, RegularPolygon(6, diameter), opd=opd, oversample=oversample
        )


class EuclidLike(ApertureBuilder):
    """Approximate Euclid-like pupil preserving the legacy dLux geometry.

    The dimensions and simplified displaced arms are carried forward from
    ``dLux.utils.euclid_like``. This is suitable for examples, not a validated Euclid
    mission pupil model.

    Primary, secondary, and support dimensions may be overridden. Increasing
    ``spider_width`` is useful when a stronger asymmetric diffraction signature is
    desired for phase-retrieval experiments.
    """

    def __init__(
        self,
        diameter=1.21,
        secondary_diameter=0.395,
        spider_width=0.012,
        spider_angles=(0, 120, 240),
        opd=None,
        oversample=5,
    ):
        shift = np.asarray((secondary_diameter / 2 - spider_width / 2, diameter / 2))
        obscurations = [Circle(secondary_diameter)]
        for angle in spider_angles:
            transformation = Affine(
                translation=shift,
                rotation=dlu.deg2rad(angle + 30),
                order=("rotation", "translation"),
            )
            obscurations.append(
                TransformedShape(Rectangle(spider_width, diameter), transformation)
            )
        super().__init__(Circle(diameter), obscurations, opd=opd, oversample=oversample)
