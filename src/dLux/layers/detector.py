"""Composable deterministic detector transformations."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import jax.numpy as np
import zodiax as zdx
from jax import Array

import dLux.utils as dlu

from ..fields import Intensity
from ..parametric import CorrelatedGaussian, Gaussian
from .optical import BaseLayer

__all__ = [
    "BaseDetectorLayer",
    "DetectorLayer",
    "Sensitivity",
    "Convolve",
    "Jitter",
    "Bias",
    "Gain",
    "Saturation",
]


class BaseDetectorLayer(BaseLayer):
    """Base contract for deterministic transformations of an `Intensity`.

    Detector layers operate on expected sampled signals and return another
    `Intensity`; they do not realise noise or construct an `Image`. Parameter axes
    follow ordinary JAX broadcasting and differentiability follows the underlying
    numerical operation.
    """

    @abstractmethod
    def apply(self, intensity: Intensity, **context) -> Intensity:
        """Transform a complete deterministic intensity.

        Implementations must return a new `Intensity`, preserve unrelated leading
        axes, and maintain grid metadata unless the operation explicitly changes the
        spatial sampling. Detector layers do not generate noise or uncertainty.
        """


class DetectorLayer(BaseDetectorLayer):
    """Public contract for deterministic detector transformations.

    Implementations receive an ``Intensity`` and must return an ``Intensity`` while
    preserving unrelated leading axes and consistent grid metadata. Resolve
    Parametric leaves against :meth:`context` before applying the transformation.
    Custom layers should change sampling only when that behaviour is part of their
    explicit public contract.
    """

    @staticmethod
    def context(intensity: Intensity, **context) -> dict[str, Any]:
        """Return the standard parametric context for a detector layer.

        Spatial responses receive the detector grid, which supplies SI coordinates
        and pixel spacing. Explicit caller context, such as state, is forwarded.
        """
        return {**context, "grid": intensity.grid}


class Sensitivity(DetectorLayer):
    """Multiply an intensity by a scalar, spatial, or parametric response.

    The resolved response follows ordinary JAX broadcasting against
    ``intensity.data``.

    Parameters
    ----------
    response : Array or Parametric
        Multiplicative detector response. Parametric values are evaluated against
        the input intensity.

    Examples
    --------
    Apply a spatially varying pixel-response map:

    ```python
    import jax.random as jr

    import dLux as dl
    import dLux.utils as dlu

    # Construct a spatially varying detector sensitivity
    response = jr.uniform(jr.key(0), (64, 64), minval=0.95, maxval=1.0)
    sensitivity = dl.Sensitivity(response=response)

    # Construct an input detector intensity
    grid = dl.GridSpec(n=64, d=10, unit="um")
    data = 1e5 * dlu.gaussian(std=8, npixels=(64, 64), extent=32)
    intensity = dl.Intensity(data=data, grid=grid)

    # Apply the pixel sensitivity
    intensity = sensitivity(intensity)
    ```
    """

    response: Array | zdx.Expression

    def __init__(self, response, *, alias=None):
        """Initialise the multiplicative detector response.

        Parameters
        ----------
        response : Array or Parametric
            Scalar or spatial response broadcast against the intensity data, or a
            parametric resolved from the detector context.
        """
        self.response = zdx.as_array(response, dtype=float)

        self.alias = alias

    def apply(self, intensity: Intensity, **context) -> Intensity:
        """Multiply intensity data by the resolved response.

        The response follows ordinary JAX broadcasting against ``(..., ny, nx)``
        data. The returned `Intensity` retains the input grid.
        """
        self = self.resolve(**self.context(intensity, **context))
        return intensity * self.response


class Convolve(DetectorLayer):
    """Convolve an intensity with a two-dimensional kernel.

    ``kernel`` may be a fixed array or Parametric resolving to shape ``(ny, nx)``.
    It is applied independently over the final two intensity axes and is not
    normalised automatically. Leading intensity axes are preserved.

    Parameters
    ----------
    kernel : Array or Parametric
        Two-dimensional convolution kernel in array ``(y, x)`` order.

    Examples
    --------
    Convolve detector intensity with a compact Gaussian kernel:

    ```python
    import dLux as dl
    import dLux.utils as dlu

    # Construct a detector convolution kernel
    kernel = dlu.gaussian(std=1.0, npixels=(9, 9), extent=4)
    convolution = dl.Convolve(kernel=kernel)

    # Construct an input detector intensity
    grid = dl.GridSpec(n=64, d=10, unit="um")
    data = 1e5 * dlu.gaussian(std=8, npixels=(64, 64), extent=32)
    intensity = dl.Intensity(data=data, grid=grid)

    # Convolve the final two spatial axes
    intensity = convolution(intensity)
    ```
    """

    kernel: Array | zdx.Expression

    def __init__(self, kernel, *, alias=None):
        """Initialise a detector convolution.

        Parameters
        ----------
        kernel : Array or Parametric
            Fixed two-dimensional ``(y, x)`` kernel or a parametric resolving to one.
            The kernel is not normalised automatically.
        """
        self.kernel = zdx.as_array(kernel, dtype=float)

        if not isinstance(self.kernel, zdx.Expression) and self.kernel.ndim != 2:
            raise ValueError("kernel must be a 2d array.")

        self.alias = alias

    def apply(self, intensity: Intensity, **context) -> Intensity:
        """Convolve the final two intensity axes with the resolved kernel.

        Leading axes and the input grid are preserved. The kernel is not normalised.
        """
        self = self.resolve(**self.context(intensity, **context))
        if self.kernel.ndim != 2:
            raise ValueError("Resolved kernel must be a 2d array.")
        return intensity.convolve(self.kernel)


class Jitter(DetectorLayer):
    """Convolve an intensity with a sampled continuous displacement profile.

    ``profile`` is evaluated on a compact coordinate grid measured in detector
    pixels. `Jitter` owns that grid, sub-pixel integration, and independent unit-sum
    normalisation over each realised kernel. The parametric owns only the continuous
    relative profile, allowing Gaussian and future non-Gaussian motion models to use
    the same detector operation. Its leaves are raised through the layer, so Gaussian
    parameters retain concise paths such as ``std``, ``stds``, and ``corr``.

    ``kernel_size`` and ``oversample`` are scalar or physical ``(x, y)`` integer
    pairs. Kernels use final array ``(y, x)`` order and may have broadcast leading
    axes supplied by the profile. Resolved profiles must be non-negative with a
    finite, non-zero spatial sum; these conditions are not checked during compiled
    evaluation.

    Numeric ``profile`` values retain the Gaussian convenience route: a scalar is a
    circular standard deviation, physical ``(x, y)`` values are independent widths,
    and a ``(2, 2)`` value is converted from covariance. Diagonal zero widths produce
    discrete deltas and an entirely zero width produces the identity kernel.

    Parameters
    ----------
    profile : Parametric or Array
        Continuous relative profile evaluated in detector-pixel coordinates. Numeric
        values are promoted to `Gaussian` or converted with
        `CorrelatedGaussian.from_cov`.
    kernel_size : int or tuple[int, int]
        Odd detector-pixel kernel dimensions in physical ``(x, y)`` order.
    oversample : int or tuple[int, int]
        Sub-pixel integration factors in physical ``(x, y)`` order.

    Examples
    --------
    Apply correlated detector-pixel jitter and inspect its sampled kernel:

    ```python
    import dLux as dl
    import dLux.utils as dlu

    # Construct a continuous correlated Gaussian displacement profile
    profile = dl.CorrelatedGaussian(
        stds=[0.5, 1.0],
        corr=0.4,
    )

    # Sample and apply the profile as detector jitter
    jitter = dl.Jitter(
        profile=profile,
        kernel_size=9,
        oversample=3,
    )

    # Construct an input detector intensity
    grid = dl.GridSpec(n=64, d=10, unit="um")
    data = 1e5 * dlu.gaussian(std=8, npixels=(64, 64), extent=32)
    intensity = dl.Intensity(data=data, grid=grid)

    # Apply the normalised jitter kernel
    intensity = jitter(intensity)

    # Access the kernel directly
    kernel = jitter.kernel()
    ```
    """

    profile: zdx.Expression
    kernel_size: tuple[int, ...]
    oversample: tuple[int, ...]

    def __init__(self, profile, kernel_size=9, oversample=3, *, alias=None):
        """Initialise a continuous jitter profile and its numerical sampling.

        Parameters
        ----------
        profile : Parametric or Array
            Relative profile evaluated on local detector-pixel coordinates. Numeric
            scalar or ``(x, y)`` values construct `Gaussian`; a numeric ``(2, 2)``
            covariance is converted with `CorrelatedGaussian.from_cov`.
        kernel_size : int or tuple[int, int]
            Odd detector-pixel dimensions in physical ``(x, y)`` order.
        oversample : int or tuple[int, int]
            Positive sub-pixel integration factors in physical ``(x, y)`` order.
        """
        # Promote numeric widths and covariances to continuous Gaussian profiles
        if not isinstance(profile, zdx.Expression):
            value = zdx.as_array(profile, dtype=float)
            if value.ndim == 0 or value.shape == (2,):
                profile = Gaussian(value)
            elif value.shape == (2, 2):
                profile = CorrelatedGaussian.from_cov(value)
            else:
                raise ValueError(
                    "Numeric profile must be scalar or have shape (2,) or (2, 2)."
                )

        # Store the dynamic profile and fixed sampling topology
        self.profile = profile
        self.kernel_size = dlu.as_size(kernel_size, 2, "kernel_size")
        self.oversample = dlu.as_size(oversample, 2, "oversample")

        if any(size % 2 == 0 for size in self.kernel_size):
            raise ValueError("kernel_size must contain odd values.")

        self.alias = alias

    def kernel(self, intensity=None, **context) -> Array:
        """Return the integrated, unit-sum detector-pixel jitter kernel.

        Parameters
        ----------
        intensity : Intensity or None
            Optional detector input supplying ``data`` to a signal-dependent
            profile. Spatial inputs are local detector-pixel ``coordinates`` and
            ``pixel_scale``, independently of the detector's physical grid.

        Returns
        -------
        kernel : Array
            Unit-sum kernels with any profile leading axes followed by final array
            shape ``(kernel_y, kernel_x)``.
        """
        # Generate compact sub-pixel coordinates in detector-pixel units
        shape = tuple(n * o for n, o in zip(self.kernel_size, self.oversample))
        spacing = tuple(1 / o for o in self.oversample)
        coordinates = dlu.nd_coords(shape, spacing)

        # Resolve the profile on its own pixel sampling, with optional incident data.
        context = dict(context)
        if intensity is not None:
            context["data"] = intensity.data
        context.update(
            coordinates=coordinates,
            pixel_scale=np.asarray(spacing),
        )
        density = zdx.resolve(self.profile, **context)
        if density.ndim < 2 or density.shape[-2:] != shape[::-1]:
            raise ValueError(
                "Resolved profile must end in the oversampled kernel shape "
                f"{shape[::-1]}."
            )

        # Integrate to detector pixels and normalise every finite-support kernel
        kernel = dlu.downsample(density, self.oversample, mean=False)
        total = kernel.sum((-2, -1), keepdims=True)
        return kernel / total

    def apply(self, intensity: Intensity, **context) -> Intensity:
        """Convolve the final two intensity axes with the resolved jitter kernel.

        Leading axes and the input grid are preserved.
        """
        return intensity.convolve(self.kernel(intensity, **context))


class Bias(DetectorLayer):
    """Add a scalar, spatial, or parametric bias to an intensity.

    Parameters
    ----------
    bias : Array or Parametric
        Additive detector signal broadcast against the input intensity.

    Examples
    --------
    Add a spatially varying detector background:

    ```python
    import jax.random as jr

    import dLux as dl
    import dLux.utils as dlu

    # Construct a spatially varying detector bias
    bias = jr.uniform(jr.key(0), (64, 64), minval=5.0, maxval=10.0)
    bias_layer = dl.Bias(bias=bias)

    # Construct an input detector intensity
    grid = dl.GridSpec(n=64, d=10, unit="um")
    data = 1e5 * dlu.gaussian(std=8, npixels=(64, 64), extent=32)
    intensity = dl.Intensity(data=data, grid=grid)

    # Add the detector bias
    intensity = bias_layer(intensity)
    ```
    """

    bias: Array | zdx.Expression

    def __init__(self, bias, *, alias=None):
        """Initialise the additive detector signal.

        Parameters
        ----------
        bias : Array or Parametric
            Scalar or spatial signal broadcast against the intensity data, or a
            parametric resolved from the detector context.
        """
        self.bias = zdx.as_array(bias, dtype=float)

        self.alias = alias

    def apply(self, intensity: Intensity, **context) -> Intensity:
        """Add the resolved scalar or broadcastable bias to intensity data.

        Leading axes and grid metadata are preserved in the returned `Intensity`.
        """
        self = self.resolve(**self.context(intensity, **context))
        return intensity + self.bias


class Gain(DetectorLayer):
    """Multiply an intensity by a linear or parametrically generated gain.

    A Parametric receives intensity data as its ``coordinates`` before being
    multiplied into the data. This enables nonlinear responses while retaining the
    same broadcasting contract as scalar and spatial gains.

    Parameters
    ----------
    gain : Array or Parametric
        Multiplicative gain or context-dependent nonlinear response.

    Examples
    --------
    Apply fixed detector gain:

    ```python
    import dLux as dl
    import dLux.utils as dlu

    # Construct a fixed detector gain
    gain = dl.Gain(gain=2.0)

    # Construct an input detector intensity
    grid = dl.GridSpec(n=64, d=10, unit="um")
    data = 1e5 * dlu.gaussian(std=8, npixels=(64, 64), extent=32)
    intensity = dl.Intensity(data=data, grid=grid)

    # Apply the fixed detector gain
    intensity = gain(intensity)

    ```
    """

    gain: Array | zdx.Expression

    def __init__(self, gain, *, alias=None):
        """Initialise the linear or parametric detector gain.

        Parameters
        ----------
        gain : Array or Parametric
            Multiplicative response. Parametrics receive intensity data through
            ``coordinates`` and may therefore describe nonlinear gain.
        """
        self.gain = zdx.as_array(gain, dtype=float)

        self.alias = alias

    @staticmethod
    def context(intensity: Intensity, **context) -> dict[str, Any]:
        """Sample the gain as a function of incident intensity data."""
        return {**context, "coordinates": intensity.data}

    def apply(self, intensity: Intensity, **context) -> Intensity:
        """Multiply ``intensity.data`` by the resolved gain.

        Parametric gains may depend on the input data and therefore represent
        nonlinear response without a separate detector-layer type.
        """
        self = self.resolve(**self.context(intensity, **context))
        return intensity * self.gain


class Saturation(DetectorLayer):
    """Limit intensity values to a scalar, spatial, or parametric maximum.

    Parameters
    ----------
    limit : Array or Parametric
        Maximum retained intensity, broadcast over the input data.

    Examples
    --------
    Apply fixed and spatially varying detector limits:

    ```python
    import jax.random as jr

    import dLux as dl
    import dLux.utils as dlu

    # Construct a fixed detector saturation limit
    saturation = dl.Saturation(limit=5e3)

    # Construct an input detector intensity
    grid = dl.GridSpec(n=64, d=10, unit="um")
    data = 1e5 * dlu.gaussian(std=8, npixels=(64, 64), extent=32)
    intensity = dl.Intensity(data=data, grid=grid)

    # Apply the fixed saturation limit
    intensity = saturation(intensity)

    # Apply a spatially varying saturation limit
    limit = jr.uniform(jr.key(0), (64, 64), minval=4e3, maxval=6e3)
    saturation = dl.Saturation(limit=limit)
    intensity = saturation(intensity)
    ```
    """

    limit: Array | zdx.Expression

    def __init__(self, limit, *, alias=None):
        """Initialise the upper detector limit.

        Parameters
        ----------
        limit : Array or Parametric
            Scalar or spatial maximum broadcast against the intensity data, or a
            parametric resolved from the detector context.
        """
        self.limit = zdx.as_array(limit, dtype=float)

        self.alias = alias

    def apply(self, intensity: Intensity, **context) -> Intensity:
        """Clip intensity data at the resolved upper limit.

        The limit follows ordinary JAX broadcasting. Values below the limit and all
        grid metadata are preserved.
        """
        self = self.resolve(**self.context(intensity, **context))
        return intensity.set(data=np.minimum(intensity.data, self.limit))
