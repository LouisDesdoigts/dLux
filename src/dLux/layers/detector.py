"""Composable deterministic detector transformations."""

from __future__ import annotations
from abc import abstractmethod
from typing import Any

import jax.numpy as np
from jax import Array

import dLux.utils as dlu

from ..fields import Intensity
from ..parametric import Parametric
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
    """Base class for transformations from one deterministic intensity to another."""

    @abstractmethod
    def apply(self, intensity: Intensity) -> Intensity:
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
    """

    @staticmethod
    def context(intensity: Intensity) -> dict[str, Any]:
        """Return the standard parametric context for a detector layer.

        The mapping exposes ``intensity``, its sampled ``data``, a leading-variable
        view under ``variables``, and SI-valued ``coordinates``.
        """
        return {
            "intensity": intensity,
            "data": intensity.data,
            "variables": intensity.data[None],
            "coordinates": intensity.coordinates,
        }


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

    response: Array | Parametric

    def __init__(self, response):
        """Initialise the multiplicative detector response.

        Parameters
        ----------
        response : Array or Parametric
            Scalar or spatial response broadcast against the intensity data, or a
            parametric resolved from the detector context.
        """
        self.response = dlu.to_value(response, types=Parametric)

    def apply(self, intensity: Intensity) -> Intensity:
        """Multiply intensity data by the resolved response.

        The response follows ordinary JAX broadcasting against ``(..., ny, nx)``
        data. The returned `Intensity` retains the input grid.
        """
        self = self.resolve(**self.context(intensity))
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

    kernel: Array | Parametric

    def __init__(self, kernel):
        """Initialise a detector convolution.

        Parameters
        ----------
        kernel : Array or Parametric
            Fixed two-dimensional ``(y, x)`` kernel or a parametric resolving to one.
            The kernel is not normalised automatically.
        """
        self.kernel = dlu.to_value(kernel, types=Parametric)

        if not isinstance(self.kernel, Parametric) and self.kernel.ndim != 2:
            raise ValueError("kernel must be a 2d array.")

    def apply(self, intensity: Intensity) -> Intensity:
        """Convolve the final two intensity axes with the resolved kernel.

        Leading axes and the input grid are preserved. The kernel is not normalised.
        """
        self = self.resolve(**self.context(intensity))
        if self.kernel.ndim != 2:
            raise ValueError("Resolved kernel must be a 2d array.")
        return intensity.convolve(self.kernel)


class Jitter(DetectorLayer):
    """Convolve an intensity with a normalised Gaussian jitter kernel.

    ``sigma`` is expressed in detector pixels. It may be scalar for circular jitter,
    length two for independent ``(x, y)`` widths, or a ``(2, 2)`` covariance matrix.
    A zero width produces a delta function along that axis and an entirely zero input
    produces the identity kernel.

    ``kernel_size`` and ``oversample`` are scalar or physical ``(x, y)`` integer
    pairs. The returned kernel uses array ``(y, x)`` order, is integrated from the
    sub-pixel grid, and is normalised over its finite support. Covariance matrices
    must be symmetric positive definite; this is not checked during compiled
    evaluation.

    Parameters
    ----------
    sigma : Array or Parametric, pixels
        Scalar circular width, ``(x, y)`` widths, or an ``(x, y)`` covariance matrix.
    kernel_size : int or tuple[int, int]
        Odd detector-pixel kernel dimensions in physical ``(x, y)`` order.
    oversample : int or tuple[int, int]
        Sub-pixel integration factors in physical ``(x, y)`` order.

    Examples
    --------
    Apply anisotropic detector-pixel jitter and inspect its sampled kernel:

    ```python
    import dLux as dl
    import dLux.utils as dlu

    # Construct anisotropic jitter in physical (x, y) pixel order
    jitter = dl.Jitter(
        sigma=[0.5, 1.0],
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

    sigma: Array | Parametric
    kernel_size: tuple[int, ...]
    oversample: tuple[int, ...]

    def __init__(self, sigma, kernel_size=9, oversample=3):
        """Initialise the jitter distribution and numerical sampling.

        Parameters
        ----------
        sigma : Array or Parametric, pixels
            Scalar width, ``(x, y)`` widths, or ``(2, 2)`` covariance matrix.
        kernel_size : int or tuple[int, int]
            Odd detector-pixel dimensions in physical ``(x, y)`` order.
        oversample : int or tuple[int, int]
            Positive sub-pixel integration factors in physical ``(x, y)`` order.
        """
        self.sigma = dlu.to_value(sigma, types=Parametric)
        self.kernel_size = dlu.as_size(kernel_size, 2, "kernel_size")
        self.oversample = dlu.as_size(oversample, 2, "oversample")

        if any(size % 2 == 0 for size in self.kernel_size):
            raise ValueError("kernel_size must contain odd values.")
        if not isinstance(self.sigma, Parametric):
            valid = self.sigma.ndim == 0 or self.sigma.shape in ((2,), (2, 2))
            if not valid:
                raise ValueError("sigma must be scalar or have shape (2,) or (2, 2).")

    def kernel(self, intensity=None) -> Array:
        """Return the resolved, normalised detector-pixel jitter kernel.

        Supply ``intensity`` when ``sigma`` is parametric so it can be resolved from
        the detector context. The returned two-dimensional ``(y, x)`` kernel has unit
        sum and is expressed directly in detector-pixel coordinates.
        """
        if isinstance(self.sigma, Parametric) and intensity is None:
            raise ValueError("intensity is required when sigma is parametric.")

        if isinstance(self.sigma, Parametric):
            self = self.resolve(**self.context(intensity))
        sigma = self.sigma

        # Resolve the oversampled kernel dimensions and physical extent
        shape = tuple(n * o for n, o in zip(self.kernel_size, self.oversample))
        spacing = tuple(1 / o for o in self.oversample)
        extent = (np.asarray(shape) - 1) * np.asarray(spacing) / 2

        # Evaluate the independent or correlated Gaussian distribution
        if sigma.ndim < 2:
            widths = np.repeat(sigma, 2) if sigma.ndim == 0 else sigma
            density = dlu.gaussian(0, widths[::-1], shape[::-1], extent[::-1])
        else:
            covariance = sigma[::-1, ::-1]
            scales = np.sqrt(np.diag(covariance))
            limits = extent[::-1] / scales
            density = dlu.mv_gaussian(np.zeros(2), covariance, shape[::-1], limits)

        # Integrate the normalised distribution to detector pixels
        return dlu.downsample(density, self.oversample, mean=False)

    def apply(self, intensity: Intensity) -> Intensity:
        """Convolve the final two intensity axes with the resolved jitter kernel.

        Leading axes and the input grid are preserved.
        """
        return intensity.convolve(self.kernel(intensity))


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

    bias: Array | Parametric

    def __init__(self, bias):
        """Initialise the additive detector signal.

        Parameters
        ----------
        bias : Array or Parametric
            Scalar or spatial signal broadcast against the intensity data, or a
            parametric resolved from the detector context.
        """
        self.bias = dlu.to_value(bias, types=Parametric)

    def apply(self, intensity: Intensity) -> Intensity:
        """Add the resolved scalar or broadcastable bias to intensity data.

        Leading axes and grid metadata are preserved in the returned `Intensity`.
        """
        self = self.resolve(**self.context(intensity))
        return intensity + self.bias


class Gain(DetectorLayer):
    """Multiply an intensity by a linear or parametrically generated gain.

    A Parametric receives the intensity through ``variables`` with shape
    ``(1, ..., ny, nx)`` before being multiplied into the data. This enables nonlinear
    responses such as polynomial gain curves while retaining the same broadcasting
    contract as scalar and spatial gains.

    Parameters
    ----------
    gain : Array or Parametric
        Multiplicative gain or context-dependent nonlinear response.

    Examples
    --------
    Apply fixed and intensity-dependent detector gain:

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

    # Construct and apply a second-order intensity-dependent gain
    nonlinear_gain = dl.Gain(
        gain=dl.Polynomial(degrees=[1, 2], coeffs=[1.0, 1e-6])
    )
    intensity = nonlinear_gain(intensity)
    ```
    """

    gain: Array | Parametric

    def __init__(self, gain):
        """Initialise the linear or parametric detector gain.

        Parameters
        ----------
        gain : Array or Parametric
            Multiplicative response. Parametrics receive intensity data through
            ``variables`` and may therefore describe nonlinear gain.
        """
        self.gain = dlu.to_value(gain, types=Parametric)

    def apply(self, intensity: Intensity) -> Intensity:
        """Multiply ``intensity.data`` by the resolved gain.

        Parametric gains may depend on the input data and therefore represent
        nonlinear response without a separate detector-layer type.
        """
        self = self.resolve(**self.context(intensity))
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

    limit: Array | Parametric

    def __init__(self, limit):
        """Initialise the upper detector limit.

        Parameters
        ----------
        limit : Array or Parametric
            Scalar or spatial maximum broadcast against the intensity data, or a
            parametric resolved from the detector context.
        """
        self.limit = dlu.to_value(limit, types=Parametric)

    def apply(self, intensity: Intensity) -> Intensity:
        """Clip intensity data at the resolved upper limit.

        The limit follows ordinary JAX broadcasting. Values below the limit and all
        grid metadata are preserved.
        """
        self = self.resolve(**self.context(intensity))
        return intensity.set(data=np.minimum(intensity.data, self.limit))
