"""Convenience aperture constructors; geometry lives in current shape expressions."""

import dLux.utils as dlu
from ...layers.optical import Optic, OpticalLayer
from ..warnings import warn_deprecated

__all__ = [
    "ApertureLayer",
    "BaseDynamicAperture",
    "DynamicAperture",
    "CircularAperture",
    "SquareAperture",
    "RectangularAperture",
    "RegPolyAperture",
    "Spider",
    "AberratedAperture",
    "CompositeAperture",
    "CompoundAperture",
    "MultiAperture",
]
ApertureLayer = OpticalLayer
BaseDynamicAperture = OpticalLayer
DynamicAperture = OpticalLayer


class _Aperture(Optic):
    """Expose released geometry names without copying the shape's parameters.

    These are read-only views. Immutable parameter updates use the current paths,
    such as ``transmission.diameter`` or ``transmission.edge.pixels``.
    """

    @property
    def nsides(self):
        shape = self.transmission
        if isinstance(shape, dlu.Circle):
            return 0
        if isinstance(shape, dlu.Rectangle):
            return 4
        if isinstance(shape, dlu.RegularPolygon):
            return shape.nsides
        raise TypeError("Spiders do not have a number of polygon sides.")

    @property
    def extent(self):
        extent = self.transmission.extent
        if extent is None:
            raise TypeError("Spiders do not have an extent.")
        return extent

    @property
    def radius(self):
        return self.transmission.diameter / 2

    @property
    def rmax(self):
        return self.radius

    @property
    def softness(self):
        edge = self.transmission.edge
        return None if edge is None else edge.pixels

    @property
    def occulting(self):
        return self.transmission.invert

    @property
    def transformation(self):
        return self.transmission.transform


def CircularAperture(
    radius, transformation=None, occulting=False, softening=1.0, normalise=False
):
    warn_deprecated(
        "CircularAperture", "Optic", "Optic(transmission=dlu.Circle(2*radius, ...))"
    )
    return _Aperture(
        transmission=dlu.Circle(
            2 * radius, edge=softening, invert=occulting, transform=transformation
        ),
        normalise=normalise,
    )


def SquareAperture(
    width, transformation=None, occulting=False, softening=1.0, normalise=False
):
    return RectangularAperture(
        width, width, transformation, occulting, softening, normalise
    )


def RectangularAperture(
    height, width, transformation=None, occulting=False, softening=1.0, normalise=False
):
    warn_deprecated(
        "RectangularAperture", "Optic", "Optic(transmission=dlu.Rectangle(...))"
    )
    return _Aperture(
        transmission=dlu.Rectangle(
            width, height, edge=softening, invert=occulting, transform=transformation
        ),
        normalise=normalise,
    )


def RegPolyAperture(
    nsides, rmax, transformation=None, occulting=False, softening=1.0, normalise=False
):
    warn_deprecated(
        "RegPolyAperture", "Optic", "Optic(transmission=dlu.RegularPolygon(...))"
    )
    return _Aperture(
        transmission=dlu.RegularPolygon(
            nsides, 2 * rmax, edge=softening, invert=occulting, transform=transformation
        ),
        normalise=normalise,
    )


def Spider(
    width, angles, transformation=None, occulting=True, softening=1.0, normalise=False
):
    warn_deprecated("Spider layer", "Optic", "Optic(transmission=dlu.Spider(...))")
    return _Aperture(
        transmission=dlu.Spider(
            width, angles, edge=softening, invert=occulting, transform=transformation
        ),
        normalise=normalise,
    )


def AberratedAperture(aperture, noll_inds, coefficients=None, as_phase=False):
    raise TypeError(
        "AberratedAperture needs explicit basis support. Use Optic(transmission=..., "
        "opd=Basis(Zernike(...), coeffs=...))."
    )


def CompositeAperture(apertures, transformation=None, normalise=False):
    raise TypeError(
        "Composite apertures require an explicit combination rule. Use "
        "Optic(transmission=Combine(shapes, ...)), or SparseOptic for separate "
        "pupils."
    )


def _transform_transmission(transmission, transformation):
    """Carry a containing aperture's coordinate map into its generated shapes."""
    if isinstance(transmission, dlu.Combine):
        children = {
            name: _transform_transmission(child, transformation)
            for name, child in transmission.parametrics.items()
        }
        return transmission.set(parametrics=children)

    # The parent's map runs first, then the individual shape's local map.
    transform = transformation
    if transmission.transform is not None:
        transform = dlu.TransformChain([transformation, transmission.transform])
    return transmission.set(transform=transform)


def _combine(apertures, transformation, normalise, operation):
    entries = list(apertures.items()) if isinstance(apertures, dict) else apertures
    apertures = dlu.list2dictionary(entries, True, Optic)
    shapes = {}
    for name, aperture in apertures.items():
        if any(
            value is not None
            for value in (aperture.opd, aperture.phase, aperture.grid_out)
        ):
            raise TypeError(
                "Combine transmissions explicitly; aberrated subapertures require "
                "SparseOptic or an explicit phase definition."
            )
        transmission = aperture.transmission
        if not isinstance(transmission, (dlu.Shape, dlu.Combine)):
            raise TypeError("Aperture combination requires shapes or Combine.")

        # Nested combinations remain Combine expressions. Move a child layer's
        # coordinate map into its shapes before removing that layer wrapper.
        if aperture.transform is not None:
            transmission = _transform_transmission(transmission, aperture.transform)
        shapes[name] = transmission
    warn_deprecated(
        "Compound/MultiAperture",
        "Optic",
        "Optic(transmission=Combine(shapes, operation=...))",
    )
    return Optic(
        transmission=dlu.Combine(shapes, operation),
        transform=transformation,
        normalise=normalise,
    )


def CompoundAperture(apertures, transformation=None, normalise=False):
    return _combine(apertures, transformation, normalise, "product")


def MultiAperture(apertures, transformation=None, normalise=False):
    return _combine(apertures, transformation, normalise, "sum")
