"""Paraxial ABCD matrix objects, constructors, and diagnostics."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import jax.numpy as np
from jax import Array, core
import numpy as onp
import zodiax as zdx

from ..helpers import list2dictionary

__all__ = [
    "ABCD",
    "ABCDElement",
    "ABCDFreeSpace",
    "ABCDLens",
    "ABCDMirror",
    "ABCDFraunhofer",
    "ABCDChain",
    "abcd_surface_power",
    "abcd_lens",
    "abcd_mirror",
    "abcd_free_space",
    "abcd_fraunhofer",
    "compose_abcd",
    "is_surface",
    "is_free_space",
    "abcd_effective_focal_length",
    "abcd_paraxial_power",
    "abcd_geometric_magnification",
    "abcd_angular_magnification",
    "abcd_distance",
    "abcd_front_focal_length",
    "abcd_back_focal_length",
    "abcd_unimodularity",
]


def _validate_finite(value, name):
    """Check concrete inputs in Python; traced values add no compiled checks."""
    value = np.asarray(value)
    if isinstance(value, core.Tracer):
        return value
    if not onp.all(onp.isfinite(onp.asarray(value))):
        raise ValueError(f"{name} must contain finite values.")
    return value


def _validate_nonzero(value, name):
    """Check finite non-zero values outside JAX tracing."""
    value = np.asarray(value)
    if isinstance(value, core.Tracer):
        return value
    concrete = onp.asarray(value)
    if onp.any(~onp.isfinite(concrete) | (concrete == 0)):
        raise ValueError(f"{name} must be finite and non-zero.")
    return value


def _validate_nonnegative(value, name):
    """Check finite non-negative values outside JAX tracing."""
    value = np.asarray(value)
    if isinstance(value, core.Tracer):
        return value
    concrete = onp.asarray(value)
    if onp.any(~onp.isfinite(concrete) | (concrete < 0)):
        raise ValueError(f"{name} must contain finite non-negative values.")
    return value


def _stack_matrix(a, b, c, d) -> Array:
    """Stack broadcast coefficients into arrays ending in ``(2, 2)``."""
    a, b, c, d = np.broadcast_arrays(a, b, c, d)
    values = np.stack((a, b, c, d), axis=-1)
    return values.reshape(a.shape + (2, 2))


def _as_tolerance(tol, dtype) -> Array:
    """Return a finite tolerance, defaulting to a dtype-aware numerical margin."""
    if tol is None:
        tol = 100 * np.finfo(dtype).eps
    return _validate_nonnegative(np.asarray(tol, dtype=float), "tol")


class ABCD(zdx.Module):
    """A paraxial ray-transfer matrix stored as one direct array leaf.

    Leading dimensions batch independent matrices. Construction validates real,
    finite coefficients but deliberately does not require determinant one:
    non-unimodular matrices may represent intentional scaling or numerical drift.
    Physical LCT propagators apply their stricter unimodularity contract separately.

    Parameters
    ----------
    matrix : ArrayLike or object
        Real finite array ending in ``(2, 2)``, another :class:`ABCD`, or an object
        exposing a matrix through ``matrix`` or ``abcd``. Leading dimensions may
        batch matrices.
    """

    matrix: Array

    def __init__(self, matrix, *, alias=None):
        """Initialise a ray-transfer matrix.

        Parameters
        ----------
        matrix : ArrayLike or object
            Real finite array ending in shape ``(2, 2)``, another :class:`ABCD`, or
            an object exposing a matrix through ``matrix`` or ``abcd``.
            Unimodularity is not enforced.
        """
        if isinstance(matrix, ABCD):
            matrix = matrix.matrix
        else:
            candidate = getattr(matrix, "matrix", None)
            matrix = getattr(matrix, "abcd", matrix) if candidate is None else candidate
            matrix = getattr(matrix, "matrix", matrix)
        matrix = np.asarray(matrix)
        if matrix.shape[-2:] != (2, 2):
            raise ValueError("matrix must end in shape (2, 2).")
        if np.issubdtype(matrix.dtype, np.complexfloating):
            raise ValueError("matrix must be real-valued.")
        self.matrix = _validate_finite(matrix.astype(float), "matrix")
        self.alias = alias

    @property
    def a(self) -> Array:
        """Return the upper-left matrix coefficient."""
        return self.matrix[..., 0, 0]

    @property
    def b(self) -> Array:
        """Return the upper-right matrix coefficient."""
        return self.matrix[..., 0, 1]

    @property
    def c(self) -> Array:
        """Return the lower-left matrix coefficient."""
        return self.matrix[..., 1, 0]

    @property
    def d(self) -> Array:
        """Return the lower-right matrix coefficient."""
        return self.matrix[..., 1, 1]

    @property
    def determinant(self) -> Array:
        """Return ``A D - B C`` for every matrix in the leading batch."""
        return self.a * self.d - self.b * self.c

    @property
    def effective_focal_length(self) -> Array:
        """Return effective focal length ``-1 / C`` in metres."""
        return -1 / self.c

    @property
    def paraxial_power(self) -> Array:
        """Return paraxial optical power ``-C`` in reciprocal metres."""
        return -self.c

    @property
    def geometric_magnification(self) -> Array:
        """Return transverse geometric magnification ``A``."""
        return self.a

    @property
    def angular_magnification(self) -> Array:
        """Return angular magnification ``D``."""
        return self.d

    @property
    def distance(self) -> Array:
        """Return effective free-space distance ``B`` in metres."""
        return self.b

    @property
    def front_focal_length(self) -> Array:
        """Return the signed front focal length ``D / C`` in metres."""
        return self.d / self.c

    @property
    def back_focal_length(self) -> Array:
        """Return the signed back focal length ``-A / C`` in metres."""
        return -self.a / self.c

    @classmethod
    def identity(cls) -> ABCD:
        """Construct the identity transform."""
        return cls(np.eye(2))

    @classmethod
    def surface_power(cls, power) -> ABCD:
        """Construct a thin surface with optical power ``power`` in ``1 / m``."""
        power = np.asarray(power, dtype=float)
        one = np.ones_like(power)
        zero = np.zeros_like(power)
        return cls(_stack_matrix(one, zero, -power, one))

    @classmethod
    def free_space(cls, distance) -> ABCD:
        """Construct free-space propagation over ``distance`` metres."""
        distance = np.asarray(distance, dtype=float)
        one = np.ones_like(distance)
        zero = np.zeros_like(distance)
        return cls(_stack_matrix(one, distance, zero, one))

    @classmethod
    def lens(cls, focal_length) -> ABCD:
        """Construct a thin lens with ``focal_length`` metres."""
        focal_length = _validate_nonzero(
            np.asarray(focal_length, dtype=float),
            "focal_length",
        )
        return cls.surface_power(1 / focal_length)

    @classmethod
    def mirror(cls, radius) -> ABCD:
        """Construct a spherical mirror with curvature radius ``radius`` metres."""
        radius = _validate_nonzero(np.asarray(radius, dtype=float), "radius")
        return cls.surface_power(2 / radius)

    @classmethod
    def fraunhofer(cls, focal_length) -> ABCD:
        """Construct the focal Fourier transform of a thin lens."""
        focal_length = _validate_nonzero(
            np.asarray(focal_length, dtype=float),
            "focal_length",
        )
        zero = np.zeros_like(focal_length)
        return cls(_stack_matrix(zero, focal_length, -1 / focal_length, zero))

    @classmethod
    def compose(cls, matrices) -> ABCD:
        """Compose an ordered sequence as ``M_n @ ... @ M_2 @ M_1``.

        An empty sequence returns :meth:`identity`. Items may be :class:`ABCD`
        objects or real arrays ending in ``(2, 2)``.
        """
        matrix = np.eye(2)
        for item in matrices:
            item = item if isinstance(item, ABCD) else cls(item)
            matrix = item.matrix @ matrix
        return cls(matrix)

    def is_surface(self, tol=None) -> Array:
        """Return where ``A=D=1`` and ``B=0`` within ``tol``."""
        tol = _as_tolerance(tol, self.matrix.dtype)
        return (
            (np.abs(self.a - 1) <= tol)
            & (np.abs(self.b) <= tol)
            & (np.abs(self.d - 1) <= tol)
        )

    def is_free_space(self, tol=None) -> Array:
        """Return where the matrix is non-zero-distance free space within ``tol``."""
        tol = _as_tolerance(tol, self.matrix.dtype)
        return (
            (np.abs(self.a - 1) <= tol)
            & (np.abs(self.b) > tol)
            & (np.abs(self.c) <= tol)
            & (np.abs(self.d - 1) <= tol)
        )

    def is_unimodular(self, tol=None) -> Array:
        """Return where the determinant differs from one by no more than ``tol``."""
        tol = _as_tolerance(tol, self.matrix.dtype)
        return np.abs(self.determinant - 1) <= tol

    def inverse(self) -> ABCD:
        """Return the inverse ray-transfer matrix."""
        determinant = _validate_nonzero(self.determinant, "ABCD determinant")
        inverse = _stack_matrix(self.d, -self.b, -self.c, self.a)
        return ABCD(inverse / determinant[..., None, None])

    def then(self, other) -> ABCD:
        """Return the physical composition that applies ``self`` then ``other``."""
        other = other if isinstance(other, ABCD) else ABCD(other)
        return ABCD(other.matrix @ self.matrix)


class ABCDElement(zdx.Expression):
    """Deferred definition of one paraxial ray-transfer matrix.

    Element parameters may themselves be Zodiax expressions. Evaluation realises a
    concrete :class:`ABCD` only at the consuming propagation boundary.
    """

    @abstractmethod
    def evaluate(self, **context: Any) -> ABCD:
        """Evaluate this element into a concrete ray-transfer matrix."""

    @property
    def abcd(self) -> Array:
        """Return the context-free concrete matrix for compatibility."""
        return self().matrix


class ABCDFreeSpace(ABCDElement):
    """Deferred free-space translation over a signed distance in metres."""

    distance: Any

    def __init__(self, distance, *, alias=None):
        """Initialise a deferred free-space matrix definition."""
        self.distance = zdx.as_array(distance, dtype=float)
        self.alias = alias

    def evaluate(self, **context: Any) -> ABCD:
        """Resolve the distance and construct its free-space matrix."""
        return ABCD.free_space(zdx.resolve(self.distance, **context))


class ABCDLens(ABCDElement):
    """Deferred ideal thin-lens definition with focal length in metres."""

    focal_length: Any

    def __init__(self, focal_length, *, alias=None):
        """Initialise a deferred thin-lens matrix definition."""
        self.focal_length = zdx.as_array(focal_length, dtype=float)
        self.alias = alias

    def evaluate(self, **context: Any) -> ABCD:
        """Resolve the focal length and construct its thin-lens matrix."""
        return ABCD.lens(zdx.resolve(self.focal_length, **context))


class ABCDMirror(ABCDElement):
    """Deferred spherical-mirror definition with radius in metres."""

    radius: Any

    def __init__(self, radius, *, alias=None):
        """Initialise a deferred spherical-mirror matrix definition."""
        self.radius = zdx.as_array(radius, dtype=float)
        self.alias = alias

    def evaluate(self, **context: Any) -> ABCD:
        """Resolve the radius and construct its spherical-mirror matrix."""
        return ABCD.mirror(zdx.resolve(self.radius, **context))


class ABCDFraunhofer(ABCDElement):
    """Deferred focal Fourier transform with focal length in metres."""

    focal_length: Any

    def __init__(self, focal_length, *, alias=None):
        """Initialise a deferred focal Fourier matrix definition."""
        self.focal_length = zdx.as_array(focal_length, dtype=float)
        self.alias = alias

    def evaluate(self, **context: Any) -> ABCD:
        """Resolve the focal length and construct its focal Fourier matrix."""
        return ABCD.fraunhofer(zdx.resolve(self.focal_length, **context))


class ABCDChain(zdx.Expression):
    """Deferred ordered composition of concrete or expression ABCD definitions."""

    elements: dict[str, Any]

    def __init__(self, elements, *, alias=None):
        """Initialise a named ordered ABCD definition sequence."""
        elements = (
            list(elements.items()) if isinstance(elements, dict) else list(elements)
        )
        self.elements = list2dictionary(
            elements,
            ordered=True,
            allowed_types=(ABCD, zdx.Expression),
        )
        if not self.elements:
            raise ValueError("elements must contain at least one ABCD definition.")
        self.alias = alias

    def evaluate(self, **context: Any) -> ABCD:
        """Resolve and compose all elements in physical application order."""
        elements = [
            zdx.resolve(element, **context) for element in self.elements.values()
        ]
        return ABCD.compose(elements)

    @property
    def abcd(self) -> Array:
        """Return the context-free concrete composed matrix for compatibility."""
        return self().matrix


def abcd_surface_power(power) -> Array:
    """Return the matrix of a thin surface with power in reciprocal metres."""
    return ABCD.surface_power(power).matrix


def abcd_lens(focal_length) -> Array:
    """Return the matrix of a thin lens with focal length in metres."""
    return ABCD.lens(focal_length).matrix


def abcd_mirror(radius) -> Array:
    """Return the matrix of a spherical mirror with radius in metres."""
    return ABCD.mirror(radius).matrix


def abcd_free_space(distance) -> Array:
    """Return the matrix of free-space propagation over a distance in metres."""
    return ABCD.free_space(distance).matrix


def abcd_fraunhofer(focal_length) -> Array:
    """Return the matrix of a physical focal Fourier transform."""
    return ABCD.fraunhofer(focal_length).matrix


def compose_abcd(matrices) -> Array:
    """Compose an ordered matrix sequence as ``M_n @ ... @ M_1``."""
    return ABCD.compose(matrices).matrix


def is_surface(matrix, tol=1e-8) -> Array:
    """Return where a matrix is a same-plane thin powered surface."""
    return ABCD(matrix).is_surface(tol)


def is_free_space(matrix, tol=1e-8) -> Array:
    """Return where a matrix is non-zero-distance free space."""
    return ABCD(matrix).is_free_space(tol)


def abcd_effective_focal_length(matrix) -> Array:
    """Return effective focal length ``-1 / C`` in metres."""
    return ABCD(matrix).effective_focal_length


def abcd_paraxial_power(matrix) -> Array:
    """Return paraxial optical power ``-C`` in reciprocal metres."""
    return ABCD(matrix).paraxial_power


def abcd_geometric_magnification(matrix) -> Array:
    """Return transverse geometric magnification ``A``."""
    return ABCD(matrix).geometric_magnification


def abcd_angular_magnification(matrix) -> Array:
    """Return angular magnification ``D``."""
    return ABCD(matrix).angular_magnification


def abcd_distance(matrix) -> Array:
    """Return effective free-space distance ``B`` in metres."""
    return ABCD(matrix).distance


def abcd_front_focal_length(matrix) -> Array:
    """Return signed front focal length ``D / C`` in metres."""
    return ABCD(matrix).front_focal_length


def abcd_back_focal_length(matrix) -> Array:
    """Return signed back focal length ``-A / C`` in metres."""
    return ABCD(matrix).back_focal_length


def abcd_unimodularity(matrix) -> Array:
    """Return the determinant ``A D - B C``."""
    return ABCD(matrix).determinant
