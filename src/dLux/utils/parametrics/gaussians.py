"""Coordinate-dependent two-dimensional Gaussian profile parametrisations.

Every class evaluates a sampled shape proportional to a continuous Gaussian on
Cartesian coordinates. Each sampled profile is rescaled to have a peak of one,
leaving finite-support and pixel-integration normalisation to consumers such as
`Jitter`. Widths, means, and covariance components follow physical ``(x, y)`` order.

These classes own optical two-dimensional conventions for standard deviations,
correlation, ellipse parameters, and stable sampled-peak normalisation through the
dLux profile utilities. They are not continuously normalised probability densities.

Construct profiles with concrete arrays or expression definitions before JAX tracing.
Constructors check concrete values; callers supply valid expression results and
immutable updates. Evaluation remains differentiable and supports JIT and leading
batches. Derived properties require numerical fields: use
``profile.resolve_children(**context)`` to prepare them without sampling the profile.
"""

from __future__ import annotations

import jax.numpy as np
import zodiax as zdx
from jax import Array

import dLux.utils as dlu

from .parametrics import SpatialExpression, _spatial_context

__all__ = [
    "Gaussian",
    "CorrelatedGaussian",
    "EllipticalGaussian",
]


def _validate_finite(value, name):
    """Standardise an array parameter and require finite concrete values."""
    value = zdx.as_array(value, dtype=float)
    if isinstance(value, zdx.Expression):
        return value
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain finite values.")
    return value


def _covariance(xx, xy, yy):
    """Assemble physical ``(x, y)`` covariance entries."""
    row_x = np.stack((xx, xy), axis=-1)
    row_y = np.stack((xy, yy), axis=-1)
    return np.stack((row_x, row_y), axis=-2)


def _correlated_covariance(stds, corr):
    """Convert resolved physical widths and correlation to covariance."""
    batch = np.broadcast_shapes(stds.shape[:-1], corr.shape)
    stds = np.broadcast_to(stds, batch + (2,))
    corr = np.broadcast_to(corr, batch)
    cross = corr * stds.prod(-1)
    return _covariance(np.square(stds[..., 0]), cross, np.square(stds[..., 1]))


def _elliptical_covariance(major, ratio, angle):
    """Convert resolved ellipse parameters to physical x/y covariance."""
    major, minor, angle = np.broadcast_arrays(major, major * ratio, angle)
    cosine, sine = np.cos(angle), np.sin(angle)
    major_var, minor_var = np.square(major), np.square(minor)
    xx = major_var * cosine**2 + minor_var * sine**2
    yy = major_var * sine**2 + minor_var * cosine**2
    xy = (major_var - minor_var) * cosine * sine
    return _covariance(xx, xy, yy)


class Gaussian(SpatialExpression):
    """An axis-aligned two-dimensional Gaussian profile.

    ``std`` is measured in the unit of the evaluation coordinates. A scalar produces
    a circular Gaussian and physical ``(x, y)`` values produce independent widths.
    Leading parameter axes broadcast against leading coordinate axes. Zero widths are
    represented by a discrete delta on the samples nearest to ``mean`` along that
    axis, allowing an exact identity kernel when used by `Jitter`.

    The sampled profile is rescaled to have a peak of one and is not sum-normalised
    over the coordinates.

    Examples
    --------
    Evaluate an anisotropic Gaussian on a detector-pixel grid:

    ```python
    import dLux as dl

    grid = dl.GridSpec(n=(9, 7), d=0.25)
    gaussian = dl.Gaussian(std=(0.5, 1.0))
    profile = gaussian.evaluate(coordinates=grid.coordinates)
    ```
    """

    std: Array | zdx.Expression
    mean: Array | zdx.Expression

    def __init__(self, std, mean=0.0, *, alias=None):
        """Initialise an axis-aligned relative Gaussian profile.

        Parameters
        ----------
        std : Array or zodiax.Expression
            Non-negative scalar or trailing physical ``(x, y)`` widths in the
            coordinate unit. Concrete values must be finite; expression values
            remain deferred until evaluation.
        mean : Array or zodiax.Expression
            Finite scalar or trailing physical ``(x, y)`` centre in the coordinate
            unit; default zero. Leading axes broadcast with the profile and grid.
        alias : mapping or None
            Optional parameter aliases; default None.
        """
        self.std = _validate_finite(std, "std")
        self.mean = _validate_finite(mean, "mean")

        # Check physical widths and centres before numerical evaluation
        if not isinstance(self.std, zdx.Expression):
            if self.std.ndim and self.std.shape[-1] != 2:
                raise ValueError("std must be scalar or have trailing shape (2,).")
            if np.any(self.std < 0):
                raise ValueError("std must contain non-negative values.")

        if not isinstance(self.mean, zdx.Expression):
            if self.mean.ndim and self.mean.shape[-1] != 2:
                raise ValueError("mean must be scalar or have trailing shape (2,).")

        self.alias = alias

    def evaluate(self, *, grid=None, coordinates=None, transform=None, **context):
        """Evaluate the axis-aligned Gaussian on one spatial sampling.


        Parameters
        ----------
        grid : GridSpec or None
            Spatial sampling in SI units, or native unitless units. A one-dimensional
            grid is promoted to a square. Mutually exclusive with coordinates.
        coordinates : Array or None
            Samples with shape (..., 2, ny, nx), in physical (x, y) order and the same
            units as the geometric parameters. Supply exactly one spatial input.
        transform : BaseCoordTransform or None
            Optional coordinate mapping applied before this definition's calculation.
        **context
            Inputs for stored expressions. Explicit coordinates require pixel_scale
            for soft edges; a grid supplies its own scale. Coordinate transforms
            preserve that scale. Missing spatial inputs raise Unresolved and defer
            during resolution; ambiguous inputs raise ValueError.


        Returns
        -------
        Array
            Peak-one profile with broadcast leading axes and final shape (ny, nx).
            Zero widths select the nearest samples. There is no sum normalisation.
        """
        context = _spatial_context(
            grid, coordinates=coordinates, transform=transform, **context
        )
        coordinates = context.pop("coordinates")
        std, mean = zdx.resolve(
            (self.std, self.mean), coordinates=coordinates, **context
        )
        return dlu.gaussian_profile(coordinates, mean, std)


class CorrelatedGaussian(SpatialExpression):
    """A Gaussian described by x/y standard deviations and their correlation.

    ``stds`` has a trailing physical ``(x, y)`` axis and ``corr`` is the
    dimensionless Pearson coefficient :math:`\\rho`, giving cross-covariance
    :math:`\\rho\\,\\sigma_x\\sigma_y`. Correlations must lie strictly between -1 and
    1 so the covariance remains positive definite.

    The sampled profile is rescaled to have a peak of one and is not sum-normalised
    over its coordinates. Leading parameter axes broadcast normally.

    Examples
    --------
    Apply correlated detector jitter in pixel units:

    ```python
    import dLux as dl

    profile = dl.CorrelatedGaussian(
        stds=[0.5, 1.0],
        corr=0.4,
    )
    jitter = dl.Jitter(profile=profile, kernel_size=9, oversample=3)
    ```
    """

    stds: Array | zdx.Expression
    corr: Array | zdx.Expression
    mean: Array | zdx.Expression

    def __init__(self, stds, corr=0.0, mean=0.0, *, alias=None):
        """Initialise an x/y standard-deviation and correlation profile.

        Parameters
        ----------
        stds : Array or zodiax.Expression
            Finite positive widths with trailing physical ``(x, y)`` axis in the
            coordinate unit. Leading axes represent profile batches.
        corr : Array or zodiax.Expression
            Dimensionless Pearson correlation strictly inside ``(-1, 1)``. Its shape
            must broadcast against ``stds.shape[:-1]``; default zero.
        mean : Array or zodiax.Expression
            Finite scalar or trailing physical ``(x, y)`` centre in the coordinate
            unit; default zero. Leading axes broadcast with the profile and grid.
        alias : mapping or None
            Optional parameter aliases; default None.
        """
        self.stds = _validate_finite(stds, "stds")
        self.corr = _validate_finite(corr, "corr")
        self.mean = _validate_finite(mean, "mean")

        # Check widths, correlation and their batch compatibility
        if not isinstance(self.stds, zdx.Expression):
            if self.stds.ndim == 0 or self.stds.shape[-1] != 2:
                raise ValueError("stds must have trailing shape (2,).")
            if np.any(self.stds <= 0):
                raise ValueError("stds must contain positive values.")
        if not isinstance(self.corr, zdx.Expression) and np.any(np.abs(self.corr) >= 1):
            raise ValueError("corr values must lie strictly between -1 and 1.")
        if not any(
            isinstance(value, zdx.Expression) for value in (self.stds, self.corr)
        ):
            np.broadcast_shapes(self.stds.shape[:-1], self.corr.shape)

        if not isinstance(self.mean, zdx.Expression):
            if self.mean.ndim and self.mean.shape[-1] != 2:
                raise ValueError("mean must be scalar or have trailing shape (2,).")

        self.alias = alias

    @property
    def cov(self) -> Array:
        """Return x/y covariance in squared coordinate units from numerical fields."""
        return _correlated_covariance(self.stds, self.corr)

    def evaluate(self, *, grid=None, coordinates=None, transform=None, **context):
        """Evaluate the correlated Gaussian on one spatial sampling.


        Parameters
        ----------
        grid : GridSpec or None
            Spatial sampling in SI units, or native unitless units. A one-dimensional
            grid is promoted to a square. Mutually exclusive with coordinates.
        coordinates : Array or None
            Samples with shape (..., 2, ny, nx), in physical (x, y) order and the same
            units as the geometric parameters. Supply exactly one spatial input.
        transform : BaseCoordTransform or None
            Optional coordinate mapping applied before this definition's calculation.
        **context
            Inputs for stored expressions. Explicit coordinates require pixel_scale
            for soft edges; a grid supplies its own scale. Coordinate transforms
            preserve that scale. Missing spatial inputs raise Unresolved and defer
            during resolution; ambiguous inputs raise ValueError.


        Returns
        -------
        Array
            Peak-one profile with broadcast leading axes and final shape (ny, nx).
            There is no sum normalisation.
        """
        context = _spatial_context(
            grid, coordinates=coordinates, transform=transform, **context
        )
        coordinates = context.pop("coordinates")
        stds, corr, mean = zdx.resolve(
            (self.stds, self.corr, self.mean), coordinates=coordinates, **context
        )
        return dlu.mv_gaussian_profile(
            coordinates, mean, _correlated_covariance(stds, corr)
        )

    @classmethod
    def from_cov(cls, cov, mean=0.0):
        """Construct an equivalent profile from physical x/y covariance.

        Parameters
        ----------
        cov : Array
            Symmetric positive-definite covariance with trailing shape ``(2, 2)`` in
            squared coordinate units. Leading axes vectorise profiles.
        mean : Array or zodiax.Expression
            Scalar or trailing physical ``(x, y)`` centre in the coordinate unit.

        Returns
        -------
        profile : CorrelatedGaussian
            Profile storing the corresponding x/y standard deviations and Pearson
            correlation.
        """
        # Validate the supplied physical covariance
        cov = _validate_finite(cov, "cov")
        if cov.ndim < 2 or cov.shape[-2:] != (2, 2):
            raise ValueError("cov must have trailing shape (2, 2).")
        if not np.allclose(cov, np.swapaxes(cov, -1, -2)):
            raise ValueError("cov must be symmetric.")
        if np.any(np.linalg.eigvalsh(cov) <= 0):
            raise ValueError("cov must be positive definite.")

        # Recover widths and correlation from the covariance entries
        stds = np.sqrt(np.diagonal(cov, axis1=-2, axis2=-1))
        corr = cov[..., 0, 1] / stds.prod(-1)
        return cls(stds, corr, mean)


class EllipticalGaussian(SpatialExpression):
    """A rotated elliptical Gaussian described by major width, ratio, and angle.

    ``major`` is the major-axis standard deviation and ``ratio`` is minor divided by
    major width. ``angle`` is measured anticlockwise in radians from physical +x to
    the major axis. The resulting covariance follows physical ``(x, y)`` order.

    The sampled profile is rescaled to have a peak of one and is not sum-normalised
    over its coordinates. Construction validates positive widths and
    ``0 < ratio <= 1``; immutable updates are not revalidated during compiled
    evaluation.
    """

    major: Array | zdx.Expression
    ratio: Array | zdx.Expression
    angle: Array | zdx.Expression
    mean: Array | zdx.Expression

    def __init__(self, major, ratio=1.0, angle=0.0, mean=0.0, *, alias=None):
        """Initialise a rotated elliptical Gaussian profile.

        Parameters
        ----------
        major : Array or zodiax.Expression
            Finite positive major-axis standard deviation in the coordinate unit.
        ratio : Array or zodiax.Expression
            Finite minor-to-major width ratio in ``(0, 1]``; default one.
        angle : Array or zodiax.Expression
            Finite anticlockwise angle in radians from physical +x to the major
            axis; default zero. Major, ratio and angle shapes must broadcast.
        mean : Array or zodiax.Expression
            Finite scalar or trailing physical ``(x, y)`` centre in the coordinate
            unit; default zero. Leading axes broadcast with the profile and grid.
        alias : mapping or None
            Optional parameter aliases; default None.
        """
        self.major = _validate_finite(major, "major")
        self.ratio = _validate_finite(ratio, "ratio")
        self.angle = _validate_finite(angle, "angle")
        self.mean = _validate_finite(mean, "mean")

        # Check ellipse dimensions and their batch compatibility
        if not isinstance(self.major, zdx.Expression) and np.any(self.major <= 0):
            raise ValueError("major must contain positive values.")
        if not isinstance(self.ratio, zdx.Expression):
            if np.any(self.ratio <= 0) or np.any(self.ratio > 1):
                raise ValueError("ratio values must lie in the interval (0, 1].")
        if not any(
            isinstance(value, zdx.Expression)
            for value in (self.major, self.ratio, self.angle)
        ):
            np.broadcast_shapes(self.major.shape, self.ratio.shape, self.angle.shape)

        if not isinstance(self.mean, zdx.Expression):
            if self.mean.ndim and self.mean.shape[-1] != 2:
                raise ValueError("mean must be scalar or have trailing shape (2,).")

        self.alias = alias

    @property
    def minor(self) -> Array:
        """Return minor-axis width in coordinate units from numerical fields."""
        return self.major * self.ratio

    @property
    def cov(self) -> Array:
        """Return x/y covariance in squared coordinate units from numerical fields."""
        return _elliptical_covariance(self.major, self.ratio, self.angle)

    def evaluate(self, *, grid=None, coordinates=None, transform=None, **context):
        """Evaluate the elliptical Gaussian on one spatial sampling.


        Parameters
        ----------
        grid : GridSpec or None
            Spatial sampling in SI units, or native unitless units. A one-dimensional
            grid is promoted to a square. Mutually exclusive with coordinates.
        coordinates : Array or None
            Samples with shape (..., 2, ny, nx), in physical (x, y) order and the same
            units as the geometric parameters. Supply exactly one spatial input.
        transform : BaseCoordTransform or None
            Optional coordinate mapping applied before this definition's calculation.
        **context
            Inputs for stored expressions. Explicit coordinates require pixel_scale
            for soft edges; a grid supplies its own scale. Coordinate transforms
            preserve that scale. Missing spatial inputs raise Unresolved and defer
            during resolution; ambiguous inputs raise ValueError.


        Returns
        -------
        Array
            Peak-one profile with broadcast leading axes and final shape (ny, nx).
            There is no sum normalisation.
        """
        context = _spatial_context(
            grid, coordinates=coordinates, transform=transform, **context
        )
        coordinates = context.pop("coordinates")
        major, ratio, angle, mean = zdx.resolve(
            (self.major, self.ratio, self.angle, self.mean),
            coordinates=coordinates,
            **context,
        )
        return dlu.mv_gaussian_profile(
            coordinates, mean, _elliptical_covariance(major, ratio, angle)
        )
