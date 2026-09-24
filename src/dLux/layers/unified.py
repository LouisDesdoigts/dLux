"""Layers that operate on both wavefronts and intensities."""

from __future__ import annotations

import zodiax as zdx
from jax import Array

import dLux.utils as dlu

from ..fields import Intensity, Wavefront
from .detector import DetectorLayer
from .optical import OpticalLayer

__all__ = [
    "UnifiedLayer",
    "ResizeLayer",
    "DownsampleLayer",
    "FlipLayer",
    "InterpolateLayer",
    "NormaliseLayer",
    "LambdaLayer",
]


class UnifiedLayer(OpticalLayer, DetectorLayer):
    """Public contract for operations shared by wavefronts and intensities.

    Unified operations implement :meth:`apply_mono`. Optical-layer dispatch maps that
    operation over wavefront axes, while natively batched intensity methods consume
    the complete intensity in one call.

    Custom unified layers must preserve the concrete input field type and keep grid
    metadata consistent with any change to spatial sampling.
    """


class ResizeLayer(UnifiedLayer):
    """Centrally pad or crop a wavefront or intensity to explicit sample counts.

    Final spatial axes are resized while leading axes are preserved. Pixel scales do
    not change, so padding expands and cropping reduces the represented field of view.
    """

    npixels: tuple[int, ...]

    def __init__(self, npixels: int | tuple[int, ...], *, alias=None):
        """Initialise an explicit spatial resize.

        Parameters
        ----------
        npixels : int or tuple[int, ...]
            Output sizes in physical-axis order.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        self.npixels = dlu.as_size(npixels, name="npixels")

        self.alias = alias

    def apply_mono(
        self, target: Wavefront | Intensity, **context
    ) -> Wavefront | Intensity:
        """Resize target spatial axes to the configured sample counts.

        Centred padding or cropping preserves pixel scales and leading axes. The
        returned object retains the concrete input field type.
        """
        return target.resize(self.npixels)


class DownsampleLayer(UnifiedLayer):
    """Downsample spatial axes by integer factors in physical-axis order.

    Wavefront and intensity reduction semantics are delegated to their field methods,
    and pixel scales increase so the represented field of view remains unchanged.
    """

    n: tuple[int, ...]

    def __init__(self, n: int | tuple[int, ...], *, alias=None):
        """Initialise integer spatial downsampling.

        Parameters
        ----------
        n : int or tuple[int, ...]
            Downsampling factors in physical-axis order.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        self.n = dlu.as_size(n, name="n")

        self.alias = alias

    def apply_mono(
        self, target: Wavefront | Intensity, **context
    ) -> Wavefront | Intensity:
        """Downsample target spatial axes by the configured integer factors.

        Field-specific sum or mean semantics are delegated to `dlu.Field.downsample`,
        and sampling metadata are updated to preserve the field of view.
        """
        return target.downsample(self.n)


class FlipLayer(UnifiedLayer):
    """Reverse a wavefront or intensity along selected NumPy array axes.

    Values are flipped without changing their `Grid`; use a coordinate transform
    and `InterpolateLayer` when a physical resampling operation is required instead.
    """

    axes: tuple[int, ...] | int

    def __init__(self, axes: tuple[int, ...] | int, *, alias=None):
        """Initialise spatial-axis reversal.

        Parameters
        ----------
        axes : int or tuple[int, ...]
            Array axes to reverse.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        self.axes = axes
        axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
        if not all(isinstance(axis, int) for axis in axes):
            raise ValueError("axes must be an int or tuple of ints.")

        self.alias = alias

    def apply_mono(
        self, target: Wavefront | Intensity, **context
    ) -> Wavefront | Intensity:
        """Flip the target about the configured NumPy array axes.

        Sampling metadata are retained and the concrete field type is preserved.
        """
        return target.flip(self.axes)


class InterpolateLayer(UnifiedLayer):
    """Resample a wavefront or intensity through a coordinate transformation.

    The transform maps output-grid coordinates into the sampled input frame. Output
    spatial size and sampling remain fixed while values are interpolated, with explicit
    handling for complex fields and samples outside the original support.
    """

    transformation: dlu.BaseCoordTransform
    method: str
    complex: bool = True
    fill: Array | zdx.Expression | None = None

    def __init__(
        self,
        transformation,
        method="cubic2",
        complex=True,
        fill=None,
        *,
        alias=None,
    ):
        """Initialise coordinate-based field interpolation.

        Parameters
        ----------
        transformation : BaseCoordTransform
            Map from output coordinates into the sampled input frame.
        method : str
            Interpolation method accepted by the interpolation utility.
        complex : bool
            Interpolate complex values through their complex representation.
        fill : float, Array, zodiax.Expression, or None
            Outside value, resolved from the target context. None extrapolates.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        if not isinstance(transformation, dlu.BaseCoordTransform):
            raise TypeError("transformation must be a BaseCoordTransform.")
        self.transformation = transformation
        self.method = str(method)
        self.complex = bool(complex)
        self.fill = zdx.as_array(fill)

        self.alias = alias

    def apply_mono(
        self, target: Wavefront | Intensity, **context
    ) -> Wavefront | Intensity:
        """Interpolate the target through the configured coordinate transformation.

        The transformation maps output SI coordinates into the sampled input frame.
        Spatial size and physical sampling are preserved; transforms may add batch axes.
        """
        if isinstance(target, Wavefront):
            context = OpticalLayer.context(target, **context)
        else:
            context = DetectorLayer.context(target, **context)
        self = self.resolve(**context)

        return target.interpolate(
            self.transformation,
            method=self.method,
            complex=self.complex,
            fill=self.fill,
        )


class NormaliseLayer(UnifiedLayer):
    """Normalise a wavefront or intensity to a configured total or peak value.

    The field owns the meaning of each mode: wavefront power is computed from the
    complex phasor, while real intensities operate on sampled values. Sampling and
    concrete field type are preserved. An expression-valued target resolves using
    the incident field's optical or detector context.
    """

    mode: str
    value: Array | zdx.Expression

    def __init__(self, mode="power", value=1.0, *, alias=None):
        """Initialise field normalisation.

        Parameters
        ----------
        mode : {"power", "peak"}
            Normalise the total power or maximum intensity.
        value : float, Array, or zodiax.Expression
            Target normalisation value.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        self.mode = str(mode)
        self.value = zdx.as_array(value, dtype=float)

        self.alias = alias

    def apply_mono(
        self, target: Wavefront | Intensity, **context
    ) -> Wavefront | Intensity:
        """Normalise the target using the configured mode and value.

        Wavefronts use optical power while real fields use their sampled values. The
        concrete field type and grid are preserved.
        """
        if isinstance(target, Wavefront):
            context = OpticalLayer.context(target, **context)
        else:
            context = DetectorLayer.context(target, **context)
        value = zdx.resolve(self.value, **context)
        return target.normalise(self.mode, value)


class LambdaLayer(UnifiedLayer):
    """Identity layer returning a wavefront or intensity unchanged.

    This provides a named no-op inside immutable system topologies, for example when
    swapping optional branches without changing the layer structure used by JIT.
    """

    def apply_mono(
        self, target: Wavefront | Intensity, **context
    ) -> Wavefront | Intensity:
        """Return the input wavefront or intensity unchanged.

        This identity layer is useful as a named placeholder in immutable systems.
        """
        return target
