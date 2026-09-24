"""Parametric paraxial matrices and their ordered composition."""

from __future__ import annotations

from collections import OrderedDict

import jax.numpy as np
from jax import Array, core
import zodiax as zdx

from ..sequential import list2dictionary

__all__ = ["ABCD", "ABCDChain", "abcd_determinant", "abcd_is_unimodular"]


def abcd_determinant(matrix) -> Array:
    """Return AD - BC for an array of matrices ending in (2, 2)."""
    matrix = np.asarray(matrix)
    a, b = matrix[..., 0, 0], matrix[..., 0, 1]
    c, d = matrix[..., 1, 0], matrix[..., 1, 1]
    return a * d - b * c


def abcd_is_unimodular(matrix, tol=None) -> Array:
    """Return a boolean per matrix whose determinant is within tol of one.

    The default absolute tolerance is 100 machine epsilons for the matrix dtype,
    matching ABCD's other classification methods. Integer matrices use float.
    """
    matrix = np.asarray(matrix)
    if not np.issubdtype(matrix.dtype, np.inexact):
        matrix = matrix.astype(float)
    determinant = abcd_determinant(matrix)
    return np.abs(determinant - 1) <= ABCD._tolerance(tol, matrix.dtype)


def _validate_value(value, name, *, positive=False, nonzero=False):
    """Check concrete parameters in Python, adding no compiled runtime checks."""
    if isinstance(value, (zdx.Expression, core.Tracer)):
        return value
    value = np.asarray(value)
    invalid = ~np.isfinite(value)
    if positive:
        invalid |= value <= 0
    if nonzero:
        invalid |= value == 0
    invalid = np.any(invalid)
    if not isinstance(invalid, core.Tracer) and invalid:
        domain = "finite"
        if positive:
            domain += " and positive"
        elif nonzero:
            domain += " and non-zero"
        raise ValueError(f"{name} must be {domain}.")
    return value


class ABCD(zdx.Expression):
    """A matrix supplied directly or generated from one physical parameter.

    Resolve to a real array with shape (..., 2, 2). Physical parameters remain
    editable leaves until resolution and may themselves be expressions. Distances,
    focal lengths and mirror radii are in metres; power is in reciprocal metres.
    Leading parameter dimensions batch independent matrices.

    Supply exactly one definition. An unused field is None and is omitted from
    the ordinary object representation. No derived matrix is cached alongside a
    physical parameter. Diagnostics evaluate context-free definitions; for local
    context, resolve first and inspect ABCD(matrix).

    Examples
    --------
    ```python
    import dLux.utils as dlu

    lens = dlu.ABCD(lens=0.5)
    print(lens)                       # ABCD(lens=f32[])
    matrix = lens.resolve()           # [[1, 0], [-2, 1]]
    system = dlu.ABCDChain({"space": dlu.ABCD(free_space=0.2), "lens": lens})
    matrix = system.resolve()
    ```
    """

    matrix: Array | zdx.Expression | None = None
    free_space: Array | zdx.Expression | None = None
    lens: Array | zdx.Expression | None = None
    power: Array | zdx.Expression | None = None
    radius: Array | zdx.Expression | None = None
    fraunhofer: Array | zdx.Expression | None = None

    def __init__(
        self,
        matrix=None,
        *,
        free_space=None,
        lens=None,
        power=None,
        radius=None,
        fraunhofer=None,
        alias=None,
    ):
        """Store one direct matrix or physical matrix definition.

        Parameters
        ----------
        matrix : Array or Expression, optional
            Real array ending in (2, 2), or an expression producing one.
        free_space : Array or Expression, optional
            Signed translation distance in metres, including zero.
        lens : Array or Expression, optional
            Nonzero thin-lens focal length in metres.
        power : Array or Expression, optional
            Thin-surface optical power in reciprocal metres, including zero.
        radius : Array or Expression, optional
            Nonzero spherical-mirror radius in metres; its power is 2 / radius.
        fraunhofer : Array or Expression, optional
            Nonzero focal Fourier-transform scale in metres.
        alias : mapping or sequence of pairs, optional
            User-defined Zodiax parameter aliases.
        """
        # Require exactly one matrix definition before storing physical parameters.
        definitions = (matrix, free_space, lens, power, radius, fraunhofer)
        if sum(value is not None for value in definitions) != 1:
            raise ValueError("Supply exactly one ABCD matrix or physical definition.")

        self.matrix = matrix
        if matrix is not None and not isinstance(matrix, zdx.Expression):
            self.matrix = ABCD.validate(matrix)
        self.free_space = zdx.as_array(free_space, dtype=float)
        self.lens = zdx.as_array(lens, dtype=float)
        self.power = zdx.as_array(power, dtype=float)
        self.radius = zdx.as_array(radius, dtype=float)
        self.fraunhofer = zdx.as_array(fraunhofer, dtype=float)
        self.alias = alias

        # Check finite values and the nonzero scales used as divisors.
        for name in ("free_space", "lens", "power", "radius", "fraunhofer"):
            value = getattr(self, name)
            if value is not None:
                _validate_value(
                    value, name, nonzero=name in ("lens", "radius", "fraunhofer")
                )

    @property
    def a(self) -> Array:
        """Upper-left coefficient of the realised matrix."""
        return self.evaluate()[..., 0, 0]

    @property
    def b(self) -> Array:
        """Upper-right coefficient of the realised matrix."""
        return self.evaluate()[..., 0, 1]

    @property
    def c(self) -> Array:
        """Lower-left coefficient of the realised matrix."""
        return self.evaluate()[..., 1, 0]

    @property
    def d(self) -> Array:
        """Lower-right coefficient of the realised matrix."""
        return self.evaluate()[..., 1, 1]

    @property
    def determinant(self) -> Array:
        """Return A D - B C for each matrix."""
        return abcd_determinant(self.evaluate())

    @property
    def effective_focal_length(self) -> Array:
        """Effective focal length -1 / C in metres."""
        return -1 / self.c

    @property
    def paraxial_power(self) -> Array:
        """Paraxial optical power -C in reciprocal metres."""
        return -self.c

    @property
    def geometric_magnification(self) -> Array:
        """Transverse geometric magnification A."""
        return self.a

    @property
    def angular_magnification(self) -> Array:
        """Angular magnification D."""
        return self.d

    @property
    def distance(self) -> Array:
        """Effective free-space distance B in metres."""
        return self.b

    @property
    def front_focal_length(self) -> Array:
        """Signed front focal length D / C in metres."""
        matrix = self.evaluate()
        return matrix[..., 1, 1] / matrix[..., 1, 0]

    @property
    def back_focal_length(self) -> Array:
        """Signed back focal length -A / C in metres."""
        matrix = self.evaluate()
        return -matrix[..., 0, 0] / matrix[..., 1, 0]

    def evaluate(self, **context) -> Array:
        """Resolve the selected physical parameter and form its matrix."""
        if self.matrix is not None:
            matrix = zdx.resolve(self.matrix, **context)
            return ABCD.validate(matrix)
        if self.free_space is not None:
            distance = zdx.resolve(self.free_space, **context)
            return ABCD._stack(1, distance, 0, 1)
        if self.fraunhofer is not None:
            focal_length = zdx.resolve(self.fraunhofer, **context)
            return ABCD._stack(0, focal_length, -1 / focal_length, 0)

        # Lens and mirror definitions share the thin-surface power matrix.
        if self.lens is not None:
            power = 1 / zdx.resolve(self.lens, **context)
        elif self.radius is not None:
            power = 2 / zdx.resolve(self.radius, **context)
        else:
            power = zdx.resolve(self.power, **context)
        return ABCD._stack(1, 0, -power, 1)

    def is_surface(self, tol=None) -> Array:
        """Return where A=D=1 and B=0 within the supplied tolerance."""
        matrix = self.evaluate()
        tol = ABCD._tolerance(tol, matrix.dtype)
        return (
            (np.abs(matrix[..., 0, 0] - 1) <= tol)
            & (np.abs(matrix[..., 0, 1]) <= tol)
            & (np.abs(matrix[..., 1, 1] - 1) <= tol)
        )

    def is_free_space(self, tol=None) -> Array:
        """Return where A=D=1, C=0 and the distance B is nonzero."""
        matrix = self.evaluate()
        tol = ABCD._tolerance(tol, matrix.dtype)
        return (
            (np.abs(matrix[..., 0, 0] - 1) <= tol)
            & (np.abs(matrix[..., 0, 1]) > tol)
            & (np.abs(matrix[..., 1, 0]) <= tol)
            & (np.abs(matrix[..., 1, 1] - 1) <= tol)
        )

    def is_unimodular(self, tol=None) -> Array:
        """Return where the determinant is one within tolerance."""
        return abcd_is_unimodular(self.evaluate(), tol)

    def inverse(self, **context) -> ABCD:
        """Return the optical system traversed from its other end.

        The longitudinal axis points along the reversed propagation direction.
        For a unit determinant, this swaps A and D while retaining B and C.
        Use ``np.linalg.inv(self.resolve(**context))`` for the matrix inverse.
        """
        matrix = self.evaluate(**context)
        a, b = matrix[..., 0, 0], matrix[..., 0, 1]
        c, d = matrix[..., 1, 0], matrix[..., 1, 1]
        determinant = abcd_determinant(matrix)
        determinant = _validate_value(determinant, "ABCD determinant", nonzero=True)
        return ABCD(ABCD._stack(d, b, c, a) / determinant[..., None, None])

    def then(self, other) -> ABCDChain:
        """Retain both definitions in physical application order."""
        return ABCDChain([self, other])

    @classmethod
    def identity(cls) -> ABCD:
        """Construct a direct identity matrix."""
        return cls(np.eye(2))

    @staticmethod
    def validate(matrix, *, unimodular=False) -> Array:
        """Return a real float matrix with trailing shape (2, 2).

        Concrete values must be finite. Set unimodular=True for physical LCT
        propagation, requiring determinant one within the shared dtype tolerance.
        Shape/type checks also work under JIT; value checks run only for concrete
        inputs and add no compiled callbacks. Resolve Expressions before calling.
        """
        matrix = np.asarray(matrix)
        if matrix.shape[-2:] != (2, 2):
            raise ValueError("matrix must end in shape (2, 2).")
        if np.issubdtype(matrix.dtype, np.complexfloating):
            raise ValueError("matrix must be real-valued.")
        matrix = _validate_value(matrix.astype(float), "matrix")

        # Arbitrary ABCD arrays are allowed; only physical propagation requires AD-BC=1.
        if unimodular and not isinstance(matrix, core.Tracer):
            invalid = ~np.all(abcd_is_unimodular(matrix))
            if not isinstance(invalid, core.Tracer) and invalid:
                raise ValueError(
                    "ABCD must be finite and unimodular for physical LCT propagation."
                )
        return matrix

    @staticmethod
    def _stack(a, b, c, d) -> Array:
        """Broadcast four coefficients into arrays ending in (2, 2)."""
        a, b, c, d = np.broadcast_arrays(a, b, c, d)
        return np.stack((a, b, c, d), axis=-1).reshape(a.shape + (2, 2))

    @staticmethod
    def _tolerance(tol, dtype):
        """Use a dtype-aware tolerance unless the caller supplies one."""
        if tol is None:
            return 100 * np.finfo(dtype).eps
        tol = np.asarray(tol, dtype=float)
        _validate_value(tol, "tol")
        invalid = np.any(tol < 0)
        if not isinstance(invalid, core.Tracer) and invalid:
            raise ValueError("tol must be non-negative.")
        return tol


class ABCDChain(zdx.Expression):
    """Named matrices resolved and multiplied in physical application order.

    For elements [first, second], the result is second @ first. Arrays and matrix
    expressions may be mixed; leading matrix dimensions broadcast normally.
    """

    elements: OrderedDict[str, Array | zdx.Expression]

    def __init__(self, elements, *, alias=None):
        """Store a nonempty mapping or sequence of optionally named matrices."""
        entries = (
            list(elements.items()) if isinstance(elements, dict) else list(elements)
        )
        elements = list2dictionary(entries, ordered=True)
        if not elements:
            raise ValueError("elements must contain at least one ABCD definition.")
        for name, value in elements.items():
            if not isinstance(value, zdx.Expression):
                elements[name] = ABCD.validate(value)
        self.elements = elements
        self.alias = alias

    def evaluate(self, **context) -> Array:
        """Resolve elements and accumulate M_n @ ... @ M_1."""
        elements = zdx.resolve(self.elements, **context)
        matrix = np.eye(2)
        for element in elements.values():
            matrix = element @ matrix
        return matrix

    def then(self, other) -> ABCDChain:
        """Append a definition while retaining this named chain."""
        elements = list(self.elements.items())
        elements.append(other)
        return ABCDChain(elements, alias=self.alias)

    def inverse(self, **context) -> ABCD:
        """Return this optical chain traversed from its other end."""
        return ABCD(self.evaluate(**context)).inverse()
