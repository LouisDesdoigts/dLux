"""Gaussian calculations and coordinate-sampled parameterisations.

Every class evaluates a continuous Gaussian with amplitude one at its mean.
Values at a coordinate do not depend on the other coordinates being sampled.
Consumers such as `Jitter` own finite-support and pixel-integration normalisation,
using `log_profile` to retain narrow profiles without numerical underflow.
Widths, means, and covariance components follow physical ``(x, y)`` order.
These profiles are not continuously normalised probability densities.

Constructors convert arrays and check shapes. Callers supply valid widths,
correlations and covariance matrices. Evaluation supports leading batches.
Derived properties require numerical fields: use
``profile.resolve_children(**context)`` to prepare them without sampling the profile.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as np
import jax.scipy.linalg as jspl
import zodiax as zdx
from jax import Array

from ..batching import as_axis
from ..grids.specs import UnitSpec
from ..batching import broadcast_batches
from ..sampled import Sampled

__all__ = [
    "eval_gaussian",
    "eval_mv_gaussian",
    "log_gaussian",
    "log_mv_gaussian",
    "correlated_covariance",
    "elliptical_covariance",
    "Gaussian",
    "CorrelatedGaussian",
    "EllipticalGaussian",
]


def log_gaussian(coordinates, mean=0.0, std=1.0, ndim=2) -> Array:
    """Evaluate Gaussian log amplitudes without exponentiating small values.

    Coordinates have shape ``(..., ndim, *sample_shape)``, with ``ndim`` sampled
    axes (two by default). Means and non-negative widths are scalars or end in
    ``(ndim,)``, in coordinate units. Leading batches broadcast.
    The output drops the coordinate-component axis and is zero at the mean.
    A zero width has support only at its exact mean, with -inf elsewhere.
    These are relative amplitudes, not normalised log probability densities.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim < ndim + 1 or coordinates.shape[-ndim - 1] != ndim:
        raise ValueError("coordinates must have shape (..., ndim, *sample_shape).")

    component_axis = -ndim - 1
    shape = (1,) * ndim

    # Broadcast distribution parameters over the coordinate sampling
    mean = as_axis(mean, ndim, "mean")
    std = as_axis(std, ndim, "std")
    mean = mean.reshape(mean.shape + shape)
    std = std.reshape(std.shape + shape)
    offset = coordinates - mean

    # Avoid division by zero; a zero width has support only at its mean.
    safe = np.where(std == 0, 1.0, std)
    terms = np.where(std == 0, 0.0, np.square(offset / safe))
    exponent = 0.5 * terms.sum(component_axis)
    delta = offset == 0
    support = np.where(std == 0, delta, True).all(component_axis)
    return np.where(support, -exponent, -np.inf)


def log_mv_gaussian(coordinates, mean, cov, ndim=2) -> Array:
    """Evaluate relative Gaussian log amplitudes from a covariance matrix.

    Coordinates have shape ``(..., ndim, *sample_shape)``, with ``ndim`` sampled
    axes (two by default). Means are scalars or end in ``(ndim,)``. Covariance
    ends in ``(ndim, ndim)`` and must be positive definite, in squared coordinate
    units. Leading batches broadcast; the output drops the component axis.
    Values are zero at the mean. No density or discrete-sum normalisation is
    applied; logarithms avoid underflow before downstream normalisation.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if coordinates.ndim < ndim + 1 or coordinates.shape[-ndim - 1] != ndim:
        raise ValueError("coordinates must have shape (..., ndim, *sample_shape).")
    if cov.ndim < 2 or cov.shape[-2:] != (ndim, ndim):
        raise ValueError("cov must have trailing shape (ndim, ndim).")

    component_axis = -ndim - 1
    spatial_shape = (1,) * ndim

    # Align the distribution parameters with the coordinate component axis
    mean = as_axis(mean, ndim, "mean")
    mean = mean.reshape(mean.shape + spatial_shape)
    offset = coordinates - mean
    points = np.moveaxis(offset, component_axis, -1)

    # Broadcast and flatten the sampled coordinates as triangular-solve columns
    cholesky = np.linalg.cholesky(cov)
    spatial = points.shape[-ndim - 1 : -1]
    points, cholesky = broadcast_batches(points, cholesky, ndims=(ndim + 1, 2))
    leading = points.shape[: -ndim - 1]
    rhs = np.moveaxis(points, -1, -ndim - 1).reshape(leading + (ndim, -1))

    # The continuous amplitude is exp(-0.5 * squared Mahalanobis distance).
    whitened = jspl.solve_triangular(cholesky, rhs, lower=True)
    exponent = np.square(whitened).sum(-2).reshape(leading + spatial)
    return -0.5 * exponent


def eval_gaussian(
    coordinates: Array,
    mean: float | Array = 0.0,
    std: float | Array = 1.0,
) -> Array:
    """Evaluate a relative axis-aligned Gaussian on two-dimensional coordinates.

    Parameters
    ----------
    coordinates : Array
        Cartesian coordinates with shape ``(..., 2, ny, nx)``. Components follow
        physical ``(x, y)`` order and final axes follow array ``(y, x)`` order.
    mean : float or Array
        Scalar or physical ``(x, y)`` centre in the coordinate unit. Leading axes
        broadcast against coordinate and width batches.
    std : float or Array
        Non-negative scalar or physical ``(x, y)`` standard deviation in the
        coordinate unit. A zero width has support only at its exact mean component.

    Returns
    -------
    profile : Array
        Relative profile with final shape ``(ny, nx)`` and broadcast leading axes.
        Amplitude is one at the continuous mean; there is no grid normalisation.
    """
    return np.exp(log_gaussian(coordinates, mean, std))


def eval_mv_gaussian(coordinates: Array, mean: Array, cov: Array) -> Array:
    """Evaluate a relative covariance Gaussian on two-dimensional coordinates.

    Parameters
    ----------
    coordinates : Array
        Cartesian coordinates with shape ``(..., 2, ny, nx)``. Components follow
        physical ``(x, y)`` order and final axes follow array ``(y, x)`` order.
    mean : Array
        Scalar or physical ``(x, y)`` centre in the coordinate unit. Leading axes
        broadcast against coordinate and covariance batches.
    cov : Array
        Symmetric positive-definite covariance with trailing shape ``(2, 2)`` in
        physical ``(x, y)`` order. Leading axes vectorise covariance matrices.

    Returns
    -------
    profile : Array
        Relative profile with final shape ``(ny, nx)`` and broadcast leading axes.
        Amplitude is one at the continuous mean; there is no grid normalisation.
    """
    return np.exp(log_mv_gaussian(coordinates, mean, cov))


def _covariance(xx, xy, yy):
    """Assemble physical ``(x, y)`` covariance entries."""
    row_x = np.stack((xx, xy), axis=-1)
    row_y = np.stack((xy, yy), axis=-1)
    return np.stack((row_x, row_y), axis=-2)


def correlated_covariance(stds, corr):
    """Convert x/y widths and correlation to covariance with shape (..., 2, 2).

    ``stds`` ends in (2,) and ``corr`` broadcasts against its leading batch.
    Widths are positive and correlation is strictly between -1 and 1.
    Covariance uses squared width units, with axes in physical (x, y) order.
    """
    stds, corr = broadcast_batches(stds, corr, ndims=(1, 0))
    cross = corr * stds.prod(-1)
    return _covariance(np.square(stds[..., 0]), cross, np.square(stds[..., 1]))


def elliptical_covariance(major, ratio, angle):
    """Convert ellipse parameters to covariance with shape (..., 2, 2).

    ``major`` is the positive major-axis width, ``ratio`` is minor/major in
    (0, 1], and ``angle`` is anticlockwise from +x in radians. Inputs broadcast;
    covariance uses squared width units, with axes in physical (x, y) order.
    """
    major, ratio, angle = np.broadcast_arrays(major, ratio, angle)
    minor = major * ratio
    cosine, sine = np.cos(angle), np.sin(angle)
    major_var, minor_var = np.square(major), np.square(minor)
    xx = major_var * cosine**2 + minor_var * sine**2
    yy = major_var * sine**2 + minor_var * cosine**2
    xy = (major_var - minor_var) * cosine * sine
    return _covariance(xx, xy, yy)


class Gaussian(Sampled):
    """An axis-aligned two-dimensional Gaussian profile.

    ``std`` is measured in the unit of the evaluation coordinates. A scalar produces
    a circular Gaussian and physical ``(x, y)`` values produce independent widths.
    Leading parameter axes broadcast against leading coordinate axes. Zero widths
    have support only at ``mean`` along that axis. `Jitter` separately selects the
    nearest samples to retain an identity kernel on its discrete sampling.

    Amplitude is one at the continuous mean, without rescaling sampled values.

    Examples
    --------
    Evaluate an anisotropic Gaussian on a detector-pixel grid:

    ```python
    import dLux as dl

    grid = dl.Grid(n=(9, 7), d=0.25)
    gaussian = dl.Gaussian(std=(0.5, 1.0))
    profile = gaussian.evaluate(coordinates=grid.coordinates)
    ```
    """

    # Inherited fields
    grid: UnitSpec | None = None
    transform: Callable | None = None
    strict: bool = eqx.field(static=True, default=False)

    std: Array | zdx.Expression
    mean: Array | zdx.Expression

    def __init__(
        self, std, mean=0.0, *, grid=None, transform=None, strict=False, alias=None
    ):
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
        self.std = zdx.as_array(std, dtype=float)
        self.mean = zdx.as_array(mean, dtype=float)

        # Check physical widths and centres before numerical evaluation
        if not isinstance(self.std, zdx.Expression):
            if self.std.ndim and self.std.shape[-1] != 2:
                raise ValueError("std must be scalar or have trailing shape (2,).")

        if not isinstance(self.mean, zdx.Expression):
            if self.mean.ndim and self.mean.shape[-1] != 2:
                raise ValueError("mean must be scalar or have trailing shape (2,).")

        super().__init__(grid=grid, transform=transform, strict=strict, alias=alias)

    def evaluate(self, *, grid=None, coordinates=None, transform=None, **context):
        """Sample the axis-aligned Gaussian, with unit amplitude at its mean.

        Supply a Grid or coordinates of shape (..., 2, ny, nx), using the same
        units as the profile parameters. Leading axes broadcast independently.
        Values have no grid-dependent normalisation.
        """
        logarithms = self.log_profile(
            grid=grid, coordinates=coordinates, transform=transform, **context
        )
        return np.exp(logarithms)

    def log_profile(self, *, grid=None, coordinates=None, transform=None, **context):
        """Sample log amplitudes for stable downstream normalisation.

        Spatial inputs follow evaluate; logarithms retain narrow profile values
        that would underflow when exponentiated directly.
        Zero widths retain support only at the exact mean.
        """
        context = self.sampling_context(
            grid, coordinates=coordinates, transform=transform, ndim=2, **context
        )
        self = self.resolve_children(**context)
        return log_gaussian(context["coordinates"], self.mean, self.std, 2)


class CorrelatedGaussian(Sampled):
    """A Gaussian described by x/y standard deviations and their correlation.

    ``stds`` has a trailing physical ``(x, y)`` axis and ``corr`` is the
    dimensionless Pearson coefficient :math:`\\rho`, giving cross-covariance
    :math:`\\rho\\,\\sigma_x\\sigma_y`. Correlations must lie strictly between -1 and
    1 so the covariance remains positive definite.

    Amplitude is one at the continuous mean, without rescaling sampled values.
    Leading parameter axes broadcast normally.

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

    # Inherited fields
    grid: UnitSpec | None = None
    transform: Callable | None = None
    strict: bool = eqx.field(static=True, default=False)

    stds: Array | zdx.Expression
    corr: Array | zdx.Expression
    mean: Array | zdx.Expression

    def __init__(
        self,
        stds,
        corr=0.0,
        mean=0.0,
        *,
        grid=None,
        transform=None,
        strict=False,
        alias=None,
    ):
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
        self.stds = zdx.as_array(stds, dtype=float)
        self.corr = zdx.as_array(corr, dtype=float)
        self.mean = zdx.as_array(mean, dtype=float)

        # Check widths, correlation and their batch compatibility
        if not isinstance(self.stds, zdx.Expression):
            if self.stds.ndim == 0 or self.stds.shape[-1] != 2:
                raise ValueError("stds must have trailing shape (2,).")
        parameters = (self.stds, self.corr)
        if not any(isinstance(value, zdx.Expression) for value in parameters):
            np.broadcast_shapes(self.stds.shape[:-1], self.corr.shape)

        if not isinstance(self.mean, zdx.Expression):
            if self.mean.ndim and self.mean.shape[-1] != 2:
                raise ValueError("mean must be scalar or have trailing shape (2,).")

        super().__init__(grid=grid, transform=transform, strict=strict, alias=alias)

    @property
    def cov(self) -> Array:
        """Return x/y covariance in squared coordinate units from numerical fields."""
        return correlated_covariance(self.stds, self.corr)

    def evaluate(self, *, grid=None, coordinates=None, transform=None, **context):
        """Sample the correlated Gaussian, with unit amplitude at its mean.

        Supply a Grid or coordinates of shape (..., 2, ny, nx), using the same
        units as the profile parameters. Leading axes broadcast independently.
        Values have no grid-dependent normalisation.
        """
        logarithms = self.log_profile(
            grid=grid, coordinates=coordinates, transform=transform, **context
        )
        return np.exp(logarithms)

    def log_profile(self, *, grid=None, coordinates=None, transform=None, **context):
        """Sample log amplitudes for stable downstream normalisation.

        Spatial inputs follow evaluate; logarithms retain narrow profile values
        that would underflow when exponentiated directly.
        """
        context = self.sampling_context(
            grid, coordinates=coordinates, transform=transform, ndim=2, **context
        )
        self = self.resolve_children(**context)
        covariance = correlated_covariance(self.stds, self.corr)
        return log_mv_gaussian(context["coordinates"], self.mean, covariance, 2)

    @classmethod
    def from_cov(
        cls, cov, mean=0.0, *, grid=None, transform=None, strict=False, alias=None
    ):
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
        # Check the matrix shape; positive definiteness is a caller requirement.
        cov = zdx.as_array(cov, dtype=float)
        if cov.ndim < 2 or cov.shape[-2:] != (2, 2):
            raise ValueError("cov must have trailing shape (2, 2).")
        # Recover widths and correlation from the covariance entries
        stds = np.sqrt(np.diagonal(cov, axis1=-2, axis2=-1))
        corr = cov[..., 0, 1] / stds.prod(-1)
        return cls(
            stds, corr, mean, grid=grid, transform=transform, strict=strict, alias=alias
        )


class EllipticalGaussian(Sampled):
    """A rotated elliptical Gaussian described by major width, ratio, and angle.

    ``major`` is the major-axis standard deviation and ``ratio`` is minor divided by
    major width. ``angle`` is measured anticlockwise in radians from physical +x to
    the major axis. The resulting covariance follows physical ``(x, y)`` order.

    Amplitude is one at the continuous mean, without rescaling sampled values.
    Widths must be positive and ``0 < ratio <= 1``.
    """

    # Inherited fields
    grid: UnitSpec | None = None
    transform: Callable | None = None
    strict: bool = eqx.field(static=True, default=False)

    major: Array | zdx.Expression
    ratio: Array | zdx.Expression
    angle: Array | zdx.Expression
    mean: Array | zdx.Expression

    def __init__(
        self,
        major,
        ratio=1.0,
        angle=0.0,
        mean=0.0,
        *,
        grid=None,
        transform=None,
        strict=False,
        alias=None,
    ):
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
        self.major = zdx.as_array(major, dtype=float)
        self.ratio = zdx.as_array(ratio, dtype=float)
        self.angle = zdx.as_array(angle, dtype=float)
        self.mean = zdx.as_array(mean, dtype=float)

        # Check ellipse dimensions and their batch compatibility
        parameters = (self.major, self.ratio, self.angle)
        if not any(isinstance(value, zdx.Expression) for value in parameters):
            np.broadcast_shapes(self.major.shape, self.ratio.shape, self.angle.shape)

        if not isinstance(self.mean, zdx.Expression):
            if self.mean.ndim and self.mean.shape[-1] != 2:
                raise ValueError("mean must be scalar or have trailing shape (2,).")

        super().__init__(grid=grid, transform=transform, strict=strict, alias=alias)

    @property
    def minor(self) -> Array:
        """Return minor-axis width in coordinate units from numerical fields."""
        return self.major * self.ratio

    @property
    def cov(self) -> Array:
        """Return x/y covariance in squared coordinate units from numerical fields."""
        return elliptical_covariance(self.major, self.ratio, self.angle)

    def evaluate(self, *, grid=None, coordinates=None, transform=None, **context):
        """Sample the elliptical Gaussian, with unit amplitude at its mean.

        Supply a Grid or coordinates of shape (..., 2, ny, nx), using the same
        units as the profile parameters. Leading axes broadcast independently.
        Values have no grid-dependent normalisation.
        """
        logarithms = self.log_profile(
            grid=grid, coordinates=coordinates, transform=transform, **context
        )
        return np.exp(logarithms)

    def log_profile(self, *, grid=None, coordinates=None, transform=None, **context):
        """Sample log amplitudes for stable downstream normalisation.

        Spatial inputs follow evaluate; logarithms retain narrow profile values
        that would underflow when exponentiated directly.
        """
        context = self.sampling_context(
            grid, coordinates=coordinates, transform=transform, ndim=2, **context
        )
        self = self.resolve_children(**context)
        covariance = elliptical_covariance(self.major, self.ratio, self.angle)
        return log_mv_gaussian(context["coordinates"], self.mean, covariance, 2)
