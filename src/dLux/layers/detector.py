"""Composable deterministic detector transformations."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import jax.numpy as np
import zodiax as zdx
from jax import Array

import dLux.utils as dlu

from ..fields import Intensity

__all__ = [
    "DetectorLayer",
    "SensitivityLayer",
    "ConvolveLayer",
    "BiasLayer",
    "GainLayer",
    "SaturationLayer",
]


class DetectorLayer(dlu.BaseLayer):
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

    @staticmethod
    def context(intensity: Intensity, **context) -> dict[str, Any]:
        """Return the standard parametric context for a detector layer.

        Spatial responses receive the detector grid, which supplies SI coordinates
        and pixel spacing. Explicit caller context, such as state, is forwarded.
        """
        return {**context, "grid": intensity.grid}


class SensitivityLayer(DetectorLayer):
    """Multiply an intensity by a scalar, spatial, or parametric response.

    The resolved response follows ordinary JAX broadcasting against
    ``intensity.data``.

    Examples
    --------
    Apply a spatially varying pixel-response map:

    ```python
    import jax.random as jr

    import dLux as dl
    import dLux.utils as dlu

    # Construct a spatially varying detector sensitivity
    response = jr.uniform(jr.key(0), (64, 64), minval=0.95, maxval=1.0)
    sensitivity = dl.SensitivityLayer(response=response)

    # Construct an input detector intensity
    grid = dl.Grid(n=64, d=10, unit="um")
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
        response : Array or Expression
            Scalar or spatial response broadcast against the intensity data, or a
            parametric resolved from the detector context.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
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


class ConvolveLayer(DetectorLayer):
    """Convolve an intensity with a two-dimensional kernel.

    ``kernel`` may be a fixed array or Expression resolving to shape ``(ny, nx)``.
    It is applied independently over the final two intensity axes and is not
    normalised automatically. Leading intensity axes are preserved.

    Examples
    --------
    Convolve detector intensity with a compact Gaussian kernel:

    ```python
    import dLux as dl
    import dLux.utils as dlu

    # Construct a detector convolution kernel
    kernel = dlu.gaussian(std=1.0, npixels=(9, 9), extent=4)
    convolution = dl.ConvolveLayer(kernel=kernel)

    # Construct an input detector intensity
    grid = dl.Grid(n=64, d=10, unit="um")
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
        kernel : Array or Expression
            Fixed two-dimensional ``(y, x)`` kernel or a parametric resolving to one.
            The kernel is not normalised automatically.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
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


class BiasLayer(DetectorLayer):
    """Add a scalar, spatial, or parametric bias to an intensity.

    Examples
    --------
    Add a spatially varying detector background:

    ```python
    import jax.random as jr

    import dLux as dl
    import dLux.utils as dlu

    # Construct a spatially varying detector bias
    bias = jr.uniform(jr.key(0), (64, 64), minval=5.0, maxval=10.0)
    bias_layer = dl.BiasLayer(bias=bias)

    # Construct an input detector intensity
    grid = dl.Grid(n=64, d=10, unit="um")
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
        bias : Array or Expression
            Scalar or spatial signal broadcast against the intensity data, or a
            parametric resolved from the detector context.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        self.bias = zdx.as_array(bias, dtype=float)

        self.alias = alias

    def apply(self, intensity: Intensity, **context) -> Intensity:
        """Add the resolved scalar or broadcastable bias to intensity data.

        Leading axes and grid metadata are preserved in the returned `Intensity`.
        """
        self = self.resolve(**self.context(intensity, **context))
        return intensity + self.bias


class GainLayer(DetectorLayer):
    """Multiply an intensity by a linear or parametrically generated gain.

    A Expression receives intensity data as its ``coordinates`` before being
    multiplied into the data. This enables nonlinear responses while retaining the
    same broadcasting contract as scalar and spatial gains.

    Examples
    --------
    Apply fixed detector gain:

    ```python
    import dLux as dl
    import dLux.utils as dlu

    # Construct a fixed detector gain
    gain = dl.GainLayer(gain=2.0)

    # Construct an input detector intensity
    grid = dl.Grid(n=64, d=10, unit="um")
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
        gain : Array or Expression
            Multiplicative response. Expressions receive intensity data through
            ``coordinates`` and may therefore describe nonlinear gain.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        self.gain = zdx.as_array(gain, dtype=float)

        self.alias = alias

    @staticmethod
    def context(intensity: Intensity, **context) -> dict[str, Any]:
        """Sample the gain as a function of incident intensity data."""
        return {**context, "coordinates": intensity.data}

    def apply(self, intensity: Intensity, **context) -> Intensity:
        """Multiply ``intensity.data`` by the resolved gain.

        Expression gains may depend on the input data and therefore represent
        nonlinear response without a separate detector-layer type.
        """
        self = self.resolve(**self.context(intensity, **context))
        return intensity * self.gain


class SaturationLayer(DetectorLayer):
    """Limit intensity values to a scalar, spatial, or parametric maximum.

    Examples
    --------
    Apply fixed and spatially varying detector limits:

    ```python
    import jax.random as jr

    import dLux as dl
    import dLux.utils as dlu

    # Construct a fixed detector saturation limit
    saturation = dl.SaturationLayer(limit=5e3)

    # Construct an input detector intensity
    grid = dl.Grid(n=64, d=10, unit="um")
    data = 1e5 * dlu.gaussian(std=8, npixels=(64, 64), extent=32)
    intensity = dl.Intensity(data=data, grid=grid)

    # Apply the fixed saturation limit
    intensity = saturation(intensity)

    # Apply a spatially varying saturation limit
    limit = jr.uniform(jr.key(0), (64, 64), minval=4e3, maxval=6e3)
    saturation = dl.SaturationLayer(limit=limit)
    intensity = saturation(intensity)
    ```
    """

    limit: Array | zdx.Expression

    def __init__(self, limit, *, alias=None):
        """Initialise the upper detector limit.

        Parameters
        ----------
        limit : Array or Expression
            Scalar or spatial maximum broadcast against the intensity data, or a
            parametric resolved from the detector context.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
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
