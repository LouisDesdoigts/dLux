"""Prebuilt generic and representative aperture models."""

import jax.numpy as np

import dLux.utils as dlu

from .builders import ApertureBuilder, SparseApertureBuilder
from .grids import Affine
from .parametric import (
    Circle,
    Rectangle,
    RegularPolygon,
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
    opd, oversample, return_support
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
        return_support=False,
    ):
        if (spider_width is None) != (spider_angles is None):
            raise ValueError("spider_width and spider_angles must both be provided.")
        obscurations = []
        if secondary_diameter is not None:
            obscurations.append(Circle(secondary_diameter))
        if spider_width is not None:
            obscurations.append(Spider(spider_width, spider_angles))
        super().__init__(
            Circle(diameter), obscurations, opd, oversample, return_support
        )


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
    obscurations : sequence or dict of Shape
        Global geometry removed from the assembled pupil.
    opd, oversample, return_support
        As defined by ``ApertureBuilder``.
    """

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
        return_support=False,
    ):
        if (segment_diameter is None) == (segment_f2f is None):
            raise ValueError(
                "Provide exactly one of segment_diameter or segment_f2f."
            )
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
            return_support=return_support,
        )


class NRMLike(SparseApertureBuilder):
    """Simple NRM with ``(x, y)`` hole centres and one shared hole shape.

    This helper deliberately excludes global stops and obscurations. Sparse
    materialization always gives every hole independent OPD coefficients.

    Parameters
    ----------
    centers : ArrayLike
        Hole centres with shape ``(n_holes, 2)`` in ``(x, y)`` order.
    hole : Shape
        The single local geometry shared by every hole.
    opd, oversample, return_support
        As defined by ``ApertureBuilder``.

    Returns
    -------
    Array or tuple of Array
        ``build`` follows the ``ApertureBuilder`` global-pupil contract.
    SparseOptic
        ``as_sparse_optic`` creates one local pupil per centre with independent OPD
        coefficients.
    """

    def __init__(
        self,
        centers,
        hole,
        opd=None,
        oversample=5,
        return_support=False,
    ):
        if not isinstance(hole, Shape):
            raise TypeError("hole must be a Shape.")
        super().__init__(
            hole,
            centers,
            opd=opd,
            oversample=oversample,
            return_support=return_support,
        )

    def as_sparse_optic(
        self,
        grid,
        transform=None,
        coefficients=None,
        key=None,
        normalise=False,
    ):
        """Materialize an NRM with independent coefficients for every hole."""
        if coefficients is not None:
            coefficients = np.asarray(coefficients)
            if coefficients.ndim == 0 or coefficients.shape[0] != len(self.centers):
                raise ValueError(
                    "coefficients must have a leading axis matching the holes."
                )
        return super().as_sparse_optic(
            grid,
            transform=transform,
            coefficients=coefficients,
            key=key,
            shared=False,
            normalise=normalise,
        )


class HSTLike(SimpleCircular):
    """Representative HST-like pupil preserving the legacy dLux geometry.

    This model omits detailed support-pad, baffle, and edge geometry and is intended
    for examples rather than observatory-grade reproduction. Its dimensions are
    carried forward from ``dLux.utils.hst_like`` and are not independently validated
    here against a contemporary observatory pupil model.
    """

    def __init__(self, opd=None, oversample=5, return_support=False):
        super().__init__(
            diameter=2.4,
            secondary_diameter=0.305,
            spider_width=0.038,
            spider_angles=(0, 90, 180, 270),
            opd=opd,
            oversample=oversample,
            return_support=return_support,
        )


class JWSTLike(SegmentedHex):
    """Representative JWST-like pupil preserving the legacy dLux geometry.

    This model includes the 18 primary segments and three support arms, but omits
    features such as the secondary-mirror support hinges and detailed pupil edges.
    Its dimensions are carried forward from ``dLux.utils.jwst_like``; it is not an
    STPSF-equivalent or independently validated observatory pupil.
    """

    def __init__(self, opd=None, oversample=5, return_support=False):
        super().__init__(
            nrings=3,
            segment_diameter=1.524,
            gap=0.007,
            remove_center=True,
            obscurations=(Spider(0.1, (30, 180, 330)),),
            opd=opd,
            oversample=oversample,
            return_support=return_support,
        )


class JWSTNRMLike(NRMLike):
    """Representative seven-hole JWST/NIRISS NRM geometry.

    Hole centres and the nominal 0.8 m flat-to-flat width are taken from the AMIGO
    model. Its fitted pupil-registration offset and detailed mask-edge effects are
    deliberately omitted.
    """

    def __init__(self, opd=None, oversample=5, return_support=False):
        # Ideal mask coordinates used by AMIGO, excluding its fitted pupil offset.
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
        diameter = 2 * 0.8 / np.sqrt(3)
        super().__init__(
            centers,
            RegularPolygon(6, diameter),
            opd=opd,
            oversample=oversample,
            return_support=return_support,
        )


class EuclidLike(ApertureBuilder):
    """Approximate Euclid-like pupil preserving the legacy dLux geometry.

    The dimensions and simplified displaced arms are carried forward from
    ``dLux.utils.euclid_like``. This is suitable for examples, not a validated Euclid
    mission pupil model.
    """

    def __init__(self, opd=None, oversample=5, return_support=False):
        diameter = 1.21
        secondary_diameter = 0.395
        spider_width = 0.012
        shift = np.asarray(
            (secondary_diameter / 2 - spider_width / 2, diameter / 2)
        )
        obscurations = [Circle(secondary_diameter)]
        for angle in (0, 120, 240):
            transformation = Affine(
                translation=shift,
                rotation=dlu.deg2rad(angle + 30),
                order=("rotation", "translation"),
            )
            obscurations.append(
                TransformedShape(
                    Rectangle(spider_width, diameter), transformation
                )
            )
        super().__init__(
            Circle(diameter),
            obscurations,
            opd=opd,
            oversample=oversample,
            return_support=return_support,
        )
