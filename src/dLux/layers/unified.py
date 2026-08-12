"""Layers that operate on both wavefronts and intensities."""

from __future__ import annotations

from jax import Array

import dLux.utils as dlu

from ..grids import BaseCoordTransform
from ..fields import Intensity, Wavefront
from .detector import DetectorLayer
from .optical import OpticalLayer

__all__ = [
    "UnifiedLayer",
    "Resize",
    "Downsample",
    "Flip",
    "Interpolate",
    "Normalise",
    "Lambda",
]


class UnifiedLayer(OpticalLayer, DetectorLayer):
    """Public contract for operations shared by wavefronts and intensities.

    Unified operations implement :meth:`apply_mono`. Optical-layer dispatch maps that
    operation over wavefront axes, while natively batched intensity methods consume
    the complete intensity in one call.
    """


class Resize(UnifiedLayer):
    """Resize a wavefront or intensity by padding or cropping."""

    npixels: tuple[int, ...]

    def __init__(self, npixels: int | tuple[int, ...]):
        """Initialise an explicit spatial resize.

        Parameters
        ----------
        npixels : int or tuple[int, ...]
            Output sizes in physical-axis order.
        """
        self.npixels = dlu.as_size(npixels, name="npixels")

    def apply_mono(self, target: Wavefront | Intensity) -> Wavefront | Intensity:
        """Resize target spatial axes to the configured sample counts.

        Centred padding or cropping preserves pixel scales and leading axes. The
        returned object retains the concrete input field type.
        """
        return target.resize(self.npixels)


class Downsample(UnifiedLayer):
    """Downsample a wavefront or intensity by an integer factor."""

    n: tuple[int, ...]

    def __init__(self, n: int | tuple[int, ...]):
        """Initialise integer spatial downsampling.

        Parameters
        ----------
        n : int or tuple[int, ...]
            Downsampling factors in physical-axis order.
        """
        self.n = dlu.as_size(n, name="n")

    def apply_mono(self, target: Wavefront | Intensity) -> Wavefront | Intensity:
        """Downsample target spatial axes by the configured integer factors.

        Field-specific sum or mean semantics are delegated to `BaseField.downsample`,
        and sampling metadata are updated to preserve the field of view.
        """
        return target.downsample(self.n)


class Flip(UnifiedLayer):
    """Flip a wavefront or intensity about one or more array axes."""

    axes: tuple[int, ...] | int

    def __init__(self, axes: tuple[int, ...] | int):
        """Initialise spatial-axis reversal.

        Parameters
        ----------
        axes : int or tuple[int, ...]
            Array axes to reverse.
        """
        self.axes = axes
        axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
        if not all(isinstance(axis, int) for axis in axes):
            raise ValueError("axes must be an int or tuple of ints.")

    def apply_mono(self, target: Wavefront | Intensity) -> Wavefront | Intensity:
        """Flip the target about the configured NumPy array axes.

        Sampling metadata are retained and the concrete field type is preserved.
        """
        return target.flip(self.axes)


class Interpolate(UnifiedLayer):
    """Interpolate a wavefront or intensity through a coordinate transformation."""

    transformation: BaseCoordTransform
    method: str
    complex: bool
    fill: Array

    def __init__(self, transformation, method="linear", complex=True, fill=0.0):
        """Initialise coordinate-based field interpolation.

        Parameters
        ----------
        transformation : BaseCoordTransform
            Map from output coordinates into the sampled input frame.
        method : str
            Interpolation method accepted by the interpolation utility.
        complex : bool
            Interpolate complex values through their complex representation.
        fill : float
            Value used outside the sampled support.
        """
        if not isinstance(transformation, BaseCoordTransform):
            raise TypeError("transformation must be a BaseCoordTransform.")
        self.transformation = transformation
        self.method = str(method)
        self.complex = bool(complex)
        self.fill = dlu.to_value(fill)

    def apply_mono(self, target: Wavefront | Intensity) -> Wavefront | Intensity:
        """Interpolate the target through the configured coordinate transformation.

        The transformation maps output SI coordinates into the sampled input frame.
        Output size and grid metadata remain unchanged.
        """
        return target.interpolate(
            self.transformation,
            method=self.method,
            complex=self.complex,
            fill=self.fill,
        )


class Normalise(UnifiedLayer):
    """Normalise a wavefront or intensity to unit total power."""

    mode: str
    value: Array

    def __init__(self, mode="power", value=1.0):
        """Initialise field normalisation.

        Parameters
        ----------
        mode : str
            Supported field-normalisation mode.
        value : float or Array
            Target normalisation value.
        """
        self.mode = str(mode)
        self.value = dlu.to_value(value)

    def apply_mono(self, target: Wavefront | Intensity) -> Wavefront | Intensity:
        """Normalise the target using the configured mode and value.

        Wavefronts use optical power while real fields use their sampled values. The
        concrete field type and grid are preserved.
        """
        return target.normalise(self.mode, self.value)


class Lambda(UnifiedLayer):
    """Return a wavefront or intensity unchanged."""

    def apply_mono(self, target: Wavefront | Intensity) -> Wavefront | Intensity:
        """Return the input wavefront or intensity unchanged.

        This identity layer is useful as a named placeholder in immutable systems.
        """
        return target
