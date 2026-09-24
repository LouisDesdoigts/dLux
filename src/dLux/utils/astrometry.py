"""Geometrical source positions, independent of brightness and optical models."""

from __future__ import annotations

import jax.numpy as np
import zodiax as zdx
from jax import Array

from .numerical.units import Unitful, unit_factor

__all__ = ["Positions", "Point", "Binary", "Companions"]


class Positions(zdx.Expression, Unitful):
    """Cartesian positions in angular units, resolving to ``(C, 2)`` radians.

    Supply one ``(x, y)`` pair or an array of pairs. Arcseconds are the default;
    use unit="mas", "deg" or "rad" for other angular conventions.
    """

    value: Array | zdx.Expression
    unit: str

    def __init__(self, value, unit="arcsec", *, alias=None):
        """Store angular positions for conversion to radians.

        Parameters
        ----------
        value : ArrayLike or Expression
            One (2,) position or non-empty (C, 2) array in unit.
        unit : str
            Angular unit; arcsec by default. Evaluation returns radians.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        self.value = zdx.as_array(value, dtype=float)
        if not isinstance(self.value, zdx.Expression):
            shape = self.value.shape
            if not (
                shape == (2,) or (len(shape) == 2 and shape[0] > 0 and shape[1] == 2)
            ):
                raise ValueError("value must have shape (2,) or non-empty (C, 2).")
        self.unit = self.validate_unit(unit, dimension="angle")
        self.alias = alias

    def evaluate(self, **context) -> Array:
        """Convert the supplied positions to radians, retaining a component axis."""
        value = zdx.resolve(self.value, **context)
        return np.atleast_2d(value) * unit_factor(self.unit)


class Point(zdx.Expression, Unitful):
    """One position, returned as an array of shape ``(1, 2)`` in radians.

    Coordinates follow physical ``(x, y)`` order. Omitting position selects the
    optical axis. Expression inputs may supply the position at evaluation.
    """

    position: Array | zdx.Expression
    unit: str = "rad"

    def __init__(self, position=None, *, unit="rad", alias=None):
        """Store one angular position, defaulting to the optical axis.

        Parameters
        ----------
        position : ArrayLike or Expression or None
            Physical (x, y) position; None is the optical axis.
        unit : str
            Angular coordinate unit; rad by default. Evaluation returns a (1, 2)
            radian array.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        self.position = zdx.as_array(
            np.zeros(2) if position is None else position, dtype=float
        )
        if not isinstance(self.position, zdx.Expression):
            if self.position.shape != (2,):
                raise ValueError("position must have shape (2,).")
        self.unit = self.validate_unit(unit, dimension="angle")
        self.alias = alias

    def evaluate(self, **context) -> Array:
        """Return a single component with its component axis retained."""
        position = zdx.resolve(self.position, **context)
        return position[None, :] * unit_factor(self.unit)


class Companions(Point):
    """A primary and companions, returned in primary-first ``(C, 2)`` order.

    ``position`` locates the primary. Every separation is relative to that same
    primary. Position and separation use unit (arcseconds by default); position
    angles run from physical ``+y`` towards ``+x`` in degrees.
    Scalar separation or angle broadcasts across its companion vector. Empty
    vectors describe the primary alone.

    Examples
    --------
    ```python
    import dLux as dl

    positions = dl.Companions(separation=[0.2, 0.4], pa=[0, 90], unit="arcsec")
    print(positions)
    print(positions.resolve())  # Primary followed by companions, in radians.
    ```
    """

    # Inherited fields
    position: Array | zdx.Expression

    separation: Array | zdx.Expression
    pa: Array | zdx.Expression
    unit: str

    def __init__(
        self, position=None, separation=(), pa=0.0, *, unit="arcsec", alias=None
    ):
        """Store a primary position and relative companion geometry.

        Parameters
        ----------
        position : ArrayLike or Expression or None
            Physical (x, y) position; None is the optical axis.
        separation : ArrayLike or Expression
            Primary-to-companion separation in unit. Scalars or companion vectors
            are accepted.
        pa : ArrayLike or Expression
            Position angle in degrees from +y towards +x; broadcasts against
            separation.
        unit : str
            Angular position/separation unit; arcsec by default. PA remains degrees.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        super().__init__(position, unit=unit, alias=alias)
        self.separation = zdx.as_array(separation, dtype=float)
        self.pa = zdx.as_array(pa, dtype=float)
        for name in ("separation", "pa"):
            value = getattr(self, name)
            if not isinstance(value, zdx.Expression) and value.ndim > 1:
                raise ValueError(f"{name} must be scalar or a companion vector.")

    def evaluate(self, **context) -> Array:
        """Offset companions from the primary, then prepend the primary."""
        position, separation, pa = zdx.resolve(
            (self.position, self.separation, self.pa), **context
        )
        separation, pa = np.broadcast_arrays(
            np.atleast_1d(separation), np.atleast_1d(pa)
        )
        pa = np.deg2rad(pa)
        directions = np.stack((np.sin(pa), np.cos(pa)), axis=-1)
        companions = position + separation[:, None] * directions
        positions = np.concatenate((position[None, :], companions), axis=0)
        return positions * unit_factor(self.unit)


class Binary(Companions):
    """A primary and one companion, with scalar separation and position angle."""

    # Inherited fields
    position: Array | zdx.Expression
    separation: Array | zdx.Expression
    pa: Array | zdx.Expression
    unit: str

    def __init__(
        self, position=None, separation=0.0, pa=0.0, *, unit="arcsec", alias=None
    ):
        """Store scalar binary geometry about the primary.

        Parameters
        ----------
        position : ArrayLike or Expression or None
            Physical (x, y) position; None is the optical axis.
        separation : float or Expression
            Single primary-to-companion separation in unit.
        pa : float or Expression
            Single position angle in degrees from +y towards +x.
        unit : str
            Angular position/separation unit; arcsec by default. PA remains degrees.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        super().__init__(position, separation, pa, unit=unit, alias=alias)
        for name in ("separation", "pa"):
            value = getattr(self, name)
            if not isinstance(value, zdx.Expression) and value.ndim != 0:
                raise ValueError(f"{name} must be a scalar for Binary.")
