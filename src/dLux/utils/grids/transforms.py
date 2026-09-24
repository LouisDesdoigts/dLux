"""Callable affine and polynomial transformations of coordinates."""

from __future__ import annotations

import warnings
from abc import abstractmethod
from collections import OrderedDict
from operator import index

import jax.numpy as np
import zodiax as zdx
from jax import Array, core

import dLux.utils as dlu

from ..sequential import Series
from .specs import UnitSpec

__all__ = [
    "BaseCoordTransform",
    "Affine",
    "AffineMap",
    "TransformChain",
    "Distortion",
    "Transform",
]


class BaseCoordTransform(zdx.Module):
    """Extension contract for transforms of ``(..., 2, ny, nx)`` coordinates.

    Leading transform and coordinate dimensions use paired JAX broadcasting. An
    unbatched coordinate field may therefore be expanded by batched transform
    parameters, while matching batches are transformed element-by-element. Subclasses
    preserve the physical ``(x, y)`` component axis and final spatial axes. A
    transform changes where an object is evaluated; it does not resample an array.
    """

    @staticmethod
    def get_coordinates(coordinates) -> Array:
        """Validate and return a Cartesian coordinate array.

        Sampling specifications supply their SI coordinates. Array inputs must
        have shape ``(..., 2, ny, nx)`` with its physical
        ``(x, y)`` component axis immediately before the spatial axes. Values are not
        converted or copied beyond standard JAX array coercion.
        """
        if coordinates is None:
            raise ValueError("Provide coordinates when calling the transformation.")

        if isinstance(coordinates, UnitSpec):
            coordinates = coordinates.coordinates
        coordinates = zdx.as_array(coordinates, dtype=float)
        if coordinates.ndim < 3 or coordinates.shape[-3] != 2:
            raise ValueError("coordinates must have shape (..., 2, ny, nx).")

        return coordinates

    @abstractmethod
    def __call__(self, coordinates: Array, **context) -> Array:
        """Transform Cartesian coordinates with shape ``(..., 2, ny, nx)``.

        Subclasses return the same coordinate convention. Leading transform axes are
        paired with matching leading coordinate axes; singleton axes broadcast by
        ordinary JAX rules.
        """

    def apply(self, coordinates: Array) -> Array:
        """Apply the transform through its deprecated method alias.

        ``coordinates`` and the returned array follow ``(..., 2, ny, nx)``. Use
        ``transform(coordinates)`` in new code; this alias is removed in dLux 0.17.
        """
        warnings.warn(
            "The `.apply()` method is deprecated and will be removed in dLux "
            "0.17.0. Use `transform(coordinates)` instead: "
            "`transform.apply(coordinates)` -> `transform(coordinates)`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self(coordinates)


class TransformChain(Series, BaseCoordTransform):
    """A numerical series mapping coordinates through named transformations.

    Without a starting grid, ``resolve()`` retains a reusable coordinate operator.
    With a grid or coordinate array, it returns the transformed coordinates when
    its parameters are ready. An explicit call overrides that stored input.
    Coordinates follow ``(..., 2, ny, nx)`` in SI units, and leading transform
    and coordinate axes broadcast. An empty chain leaves coordinates unchanged.
    Store an unbound chain in a transform field: resolving a parent replaces a
    bound chain with its coordinate array.
    """

    # Inherited fields
    layers: OrderedDict

    grid: UnitSpec | Array | zdx.Expression | None = None

    def __init__(self, transformations=(), grid=None, *, alias=None):
        """Store named transforms and an optional starting grid.

        Parameters
        ----------
        transformations : mapping or sequence, optional
            Coordinate transforms, optionally named with ``(name, transform)``
            pairs. The first listed transform acts first.
        grid : UnitSpec, Array, Expression, or None, optional
            Starting sampling specification or SI coordinates with shape
            ``(..., 2, ny, nx)``. Stored as ``grid``;
            None retains an unbound operator until an explicit call.
        alias : mapping or sequence of pairs, optional
            User-defined Zodiax parameter aliases.
        """
        super().__init__(transformations, alias=alias)
        self.grid = (
            grid if isinstance(grid, UnitSpec) else zdx.as_array(grid, dtype=float)
        )

    def evaluate(self, *, grid=None, **context) -> Array:
        """Transform the supplied grid or the stored starting sampling."""
        grid = self.grid if grid is None else grid
        if grid is None:
            raise zdx.Unresolved("TransformChain requires a grid or coordinates.")
        return self.apply(grid, **context)

    def apply(self, target, **context) -> Array:
        """Resolve a grid or coordinate array and pass it through every transform."""
        target = zdx.resolve(target, **context)
        coordinates = self.get_coordinates(target)
        return super().apply(coordinates, **context)


class Distortion(BaseCoordTransform):
    """Apply a polynomial distortion to Cartesian coordinates.

    Polynomial coefficients have shape ``(2, n_terms)`` for output ``x`` and
    ``y``. Leading coefficient axes vectorise independent distortions; matching
    coordinate batches are transformed element-by-element.

    Examples
    --------
    Construct a quadratic and cubic distortion, leaving linear terms out:

    ```python
    import dLux as dl

    transform = dl.Distortion(orders=(2, 3))

    # Apply the distortion to a grid
    grid = dl.Grid(n=128, diam=1.0, unit="m")
    coordinates = grid.transformed(transform)

    # Inspect the polynomial terms and their coefficients
    powers = transform.powers
    coeffs = transform.distortion
    ```
    """

    powers: Array
    distortion: Array | zdx.Expression

    def __init__(
        self,
        order: int | None = None,
        distortion: Array | zdx.Expression | None = None,
        *,
        orders: tuple[int, ...] | list[int] | None = None,
        powers: Array | None = None,
        alias=None,
    ):
        """Initialise a polynomial coordinate distortion.

        Parameters
        ----------
        order : int or None
            Maximum total order, mutually exclusive with ``orders`` and ``powers``.
        distortion : Array, Expression, or None
            Coefficients with trailing shape ``(2, n_terms)``. Defaults to zeros.
        orders : sequence[int] or None
            Selected positive total orders. Use (2, 3) to omit linear terms.
        powers : Array or None
            Explicit exponents with shape ``(2, n_terms)``.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        if sum(value is not None for value in (order, orders, powers)) > 1:
            raise ValueError("Provide only one of order, orders, or powers.")

        # Choose polynomial terms before allocating their coordinate coefficients.
        if powers is None:
            if orders is None:
                order = 1 if order is None else index(order)
                orders = range(1, order + 1)
            if any(value < 1 for value in orders):
                raise ValueError("orders must contain positive integers.")
            self.powers = dlu.polynomial_powers(ndim=2, degrees=orders)
        else:
            powers = np.asarray(powers)
            if powers.ndim != 2 or powers.shape[0] != 2:
                raise ValueError("powers must have shape (2, n_terms).")
            self.powers = dlu.polynomial_powers(powers=powers)

        # Expressions remain deferred; supplied arrays must match the selected terms.
        if distortion is None:
            distortion = np.zeros_like(self.powers)
        distortion = zdx.as_array(distortion, dtype=float)
        if not isinstance(distortion, zdx.Expression):
            if distortion.shape[-2:] != self.powers.shape:
                raise ValueError(
                    "distortion trailing dimensions must match powers shape."
                )
        self.distortion = distortion

        self.alias = alias

    def __call__(self, coords: Array, **context) -> Array:
        """Apply the configured polynomial coordinate distortion."""
        coords = self.get_coordinates(coords)
        self = self.resolve(**{**context, "coordinates": coords})

        # No selected terms means an identity map, with the usual batch broadcasting.
        if self.powers.shape[-1] == 0:
            batch = np.broadcast_shapes(coords.shape[:-3], self.distortion.shape[:-2])
            return np.broadcast_to(coords, batch + coords.shape[-3:])

        # Replace the coordinate component axis with polynomial terms, then sum
        # those terms separately for x and y while broadcasting the batch axes.
        basis = dlu.polynomial_basis(coords, self.powers, axis=-3)
        distortion = np.einsum("...ik,...kyx->...iyx", self.distortion, basis)
        return coords + distortion


class AffineMap(BaseCoordTransform):
    """Apply a direct affine coordinate map ``x' = matrix @ x + offset``.

    Matrix, offset, and coordinate leading dimensions use paired JAX broadcasting.
    This is the low-level direct representation; use `Affine` for semantic
    translation, rotation, scale, and shear parameters.
    """

    matrix: Array | zdx.Expression
    offset: Array | zdx.Expression

    def __init__(self, matrix=None, offset=None, *, alias=None):
        """Initialise a direct affine map.

        Parameters
        ----------
        matrix : Array, Expression, or None
            Matrix with trailing shape ``(2, 2)``; defaults to identity.
        offset : Array, Expression, or None
            Translation with trailing shape ``(2,)``; defaults to zero.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        matrix = np.eye(2) if matrix is None else zdx.as_array(matrix, dtype=float)
        offset = np.zeros(2) if offset is None else zdx.as_array(offset, dtype=float)
        if not isinstance(matrix, zdx.Expression):
            if matrix.shape[-2:] != (2, 2):
                raise ValueError("matrix must have trailing shape (2, 2).")
        if not isinstance(offset, zdx.Expression):
            if offset.shape[-1:] != (2,):
                raise ValueError("offset must have trailing shape (2,).")
        self.matrix = matrix
        self.offset = offset

        self.alias = alias

    def __call__(self, coords: Array, **context) -> Array:
        """Apply the direct affine matrix and offset to coordinates."""
        coords = self.get_coordinates(coords)
        self = self.resolve(**{**context, "coordinates": coords})
        shift = self.offset[..., :, None, None]
        return np.einsum("...ij,...jxy->...ixy", self.matrix, coords) + shift


class Affine(BaseCoordTransform):
    """An affine coordinate transform with semantic parameters.

    Translation, rotation, scale, and shear map coordinates into a transformed
    object's local frame. Positive rotation turns coordinates anticlockwise in
    physical (x, y), so a sampled optical object turns clockwise. Operations
    on coordinates are composed in the order supplied by ``order``.
    Parameter and coordinate leading dimensions use paired JAX broadcasting.

    Examples
    --------
    Transform a grid using semantic affine parameters:

    ```python
    import jax.numpy as np

    import dLux as dl

    # Construct a semantic affine transformation
    transform = dl.Affine(
        translation=[0.1, -0.1],
        rotation=np.deg2rad(15),
        scale=[1.0, 0.8],
    )

    # Apply the transformation to grid coordinates
    grid = dl.Grid(n=128, diam=1.0, unit="m")
    coordinates = grid.transformed(transform)

    # Access the equivalent affine matrix and offset
    matrix, offset = transform.coeffs
    ```
    """

    translation: Array | zdx.Expression | None = None
    rotation: Array | zdx.Expression | None = None
    scale: Array | zdx.Expression | None = None
    shear: Array | zdx.Expression | None = None
    order: tuple[str, ...]

    def __init__(
        self,
        translation=None,
        rotation=None,
        scale=None,
        shear=None,
        order=("translation", "rotation", "scale", "shear"),
        *,
        alias=None,
    ):
        """Initialise a semantic affine coordinate transformation.

        Parameters
        ----------
        translation : ArrayLike, Expression, or None
            Physical ``(x, y)`` translation into the object's local frame.
        rotation : ArrayLike, Expression, or None
            Counter-clockwise coordinate angle in radians; any leading batch shape.
        scale : ArrayLike, Expression, or None
            Non-zero scalar or ``(x, y)`` coordinate scale.
        shear : ArrayLike, Expression, or None
            Two-component shear.
        order : tuple[str, ...]
            Unique composition order drawn from translation, rotation, scale, and
            shear.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        self.translation = self._vector(translation, "translation")
        self.rotation = zdx.as_array(rotation, dtype=float)

        self.scale = zdx.as_array(scale, dtype=float)
        if self.scale is not None and not isinstance(self.scale, zdx.Expression):
            self.scale = dlu.as_axis(self.scale, 2, "scale")
            # Check concrete setup values; traced calculations have no callback.
            if not isinstance(self.scale, core.Tracer):
                if np.any(self.scale == 0):
                    raise ValueError("scale values must be non-zero.")

        self.shear = self._vector(shear, "shear")

        valid = ("translation", "rotation", "scale", "shear")
        self.order = tuple(order)
        if len(set(self.order)) != len(self.order) or not set(self.order) <= set(valid):
            raise ValueError(f"order entries must be unique values from {valid}.")

        self.alias = alias

    @staticmethod
    def _vector(value, name):
        if value is None:
            return None
        value = zdx.as_array(value, dtype=float)
        if not isinstance(value, zdx.Expression):
            if value.shape[-1:] != (2,):
                raise ValueError(f"{name} must have trailing shape (2,).")
        return value

    @property
    def coeffs(self) -> tuple[Array, Array]:
        """Return the composed affine matrix and offset.

        Returns
        -------
        matrix : Array
            Forward coordinate matrix with shape ``(..., 2, 2)``.
        offset : Array
            Translation vector with shape ``(..., 2)``. Leading parameter axes are
            broadcast and preserved in both outputs.
        """
        self = self.resolve()

        # Request the operands before using their shapes. Missing context can
        # then defer a containing expression, such as a bound TransformChain.
        values = {
            name: zdx.resolve(getattr(self, name))
            for name in ("translation", "rotation", "scale", "shear")
        }

        # Preserve the shared parameter batch, including an empty operation order.
        shapes = []
        for name, value in values.items():
            if value is not None:
                shape = value.shape if name == "rotation" else value.shape[:-1]
                shapes.append(shape)
        batch = np.broadcast_shapes(*shapes)
        homogeneous = np.broadcast_to(np.eye(3), batch + (3, 3))

        # The order and optional fields are fixed while tracing. JIT unrolls this
        # short loop into matrix operations, with no loop over image pixels.
        for name in self.order:
            value = values[name]
            if value is None:
                continue

            # Each component carries only its own batch axes. Matrix products
            # broadcast these axes as the coordinate operations are composed.
            if name == "rotation":
                operation = np.broadcast_to(np.eye(3), value.shape + (3, 3))
                cosine, sine = np.cos(value), np.sin(value)
                operation = operation.at[..., 0, 0].set(cosine)
                operation = operation.at[..., 0, 1].set(-sine)
                operation = operation.at[..., 1, 0].set(sine)
                operation = operation.at[..., 1, 1].set(cosine)
            else:
                if name == "scale":
                    value = dlu.as_axis(value, 2, "scale")
                operation = np.broadcast_to(np.eye(3), value.shape[:-1] + (3, 3))
                if name == "translation":
                    operation = operation.at[..., :2, 2].set(-value)
                elif name == "scale":
                    operation = operation.at[..., 0, 0].set(1 / value[..., 0])
                    operation = operation.at[..., 1, 1].set(1 / value[..., 1])
                elif name == "shear":
                    operation = operation.at[..., 0, 1].set(value[..., 0])
                    operation = operation.at[..., 1, 0].set(value[..., 1])

            homogeneous = operation @ homogeneous

        return homogeneous[..., :2, :2], homogeneous[..., :2, 2]

    def __call__(self, coords: Array, **context) -> Array:
        """Apply the composed semantic affine transformation."""
        coords = self.get_coordinates(coords)
        self = self.resolve(**{**context, "coordinates": coords})
        matrix, offset = self.coeffs
        shift = offset[..., :, None, None]
        return np.einsum("...ij,...jxy->...ixy", matrix, coords) + shift


class Transform(BaseCoordTransform):
    """Map coordinates through an affine frame and then polynomial distortion.

    Both components are optional: an empty transform leaves coordinates unchanged.
    The polynomial displacement is evaluated in the affine-transformed frame.
    Coordinates retain their physical ``(x, y)`` axis and final spatial axes;
    component parameters follow the broadcasting rules of their own classes.

    This is a reusable coordinate mapping, including as a ``Sampled.transform``.
    It changes where an expression is sampled rather than resampling an array.
    """

    affine: Affine | AffineMap | None = None
    polynomial: Distortion | None = None

    def __init__(self, affine=None, polynomial=None, *, alias=None):
        """Initialise an affine mapping with optional polynomial displacement.

        Parameters
        ----------
        affine : Affine, AffineMap, or None
            Mapping applied first. None leaves the input frame unchanged.
        polynomial : Distortion or None
            Polynomial displacement applied to the affine-mapped coordinates.
            None leaves those coordinates unchanged.
        alias : mapping or sequence of pairs, optional
            User-defined parameter paths into either component.
        """
        self.affine = affine
        self.polynomial = polynomial
        self.alias = alias

    def __call__(self, coordinates: Array, **context) -> Array:
        """Resolve stored parameters and transform ``(..., 2, ny, nx)`` coordinates."""
        coordinates = self.get_coordinates(coordinates)
        # Populate shared links together, but let each operation resolve its
        # parameters on the coordinates produced by the preceding operation.
        self = self.populate()

        if self.affine is not None:
            coordinates = self.affine(coordinates, **context)
        if self.polynomial is not None:
            coordinates = self.polynomial(coordinates, **context)
        return coordinates
