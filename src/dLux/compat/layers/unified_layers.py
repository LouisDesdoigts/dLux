"""Old unified-layer paths, backed by current field operations."""

import dLux.utils as dlu
from ...layers.unified import (
    UnifiedLayer,
    ResizeLayer as Resize,
    FlipLayer as Flip,
    LambdaLayer as Lambda,
    InterpolateLayer,
)
from ..warnings import warn_deprecated

__all__ = ["UnifiedLayer", "Resize", "Rotate", "Flip", "Lambda"]


class _Rotate(InterpolateLayer):
    """Expose the released angle as a view of the affine rotation."""

    @property
    def angle(self):
        return self.transformation.rotation


def Rotate(angle, method="linear", complex=False):
    warn_deprecated(
        "Rotate", "InterpolateLayer", "InterpolateLayer(dlu.Affine(rotation=angle))"
    )
    return _Rotate(dlu.Affine(rotation=angle), method=method, complex=complex, fill=0.0)
