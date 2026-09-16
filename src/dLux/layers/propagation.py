"""Wavefront adapters for numerical propagation."""

from __future__ import annotations
from typing import Any
import jax.numpy as np
from jax import Array, core, tree
import zodiax as zdx

import dLux.utils as dlu
from ..utils.grids import GridSpec, ResizeSpec, FFTSpec
from ..utils import propagation as _prop
from ..utils.propagation.propagators import _resolved
from .optical import OpticalLayer

__all__ = [
    "BasePropagator",
    "Propagator",
    "FFTPropagator",
    "SpectralPropagator",
    "FreeSpace",
]


def _align_propagation_batch(value, wavefront, tail=0, *, sampling=False):
    """Align numerical metadata to wavelength/batch axes, excluding Jones axes.

    Sampling that matches only the non-spectral batch is shared across wavelengths,
    as in Wavefront._mapped_axis. Scalar optical parameters prefer wavelength axes,
    as in Wavefront._broadcast_array. Explicit full prefixes remove any ambiguity.
    """
    if value is None:
        return None
    value = np.asarray(value)
    prefix = value.shape[:-tail] if tail else value.shape
    suffix = value.shape[-tail:] if tail else ()
    if not prefix:
        return value

    # Match the established semantic axes before considering ordinary broadcasting.
    leading = wavefront.phasor.shape[: wavefront.batch_ndim]
    chromatic = wavefront.wavelength.shape
    batch = leading[len(chromatic) :]
    on_wavelength = prefix == chromatic and bool(chromatic)
    on_batch = prefix == batch and bool(batch)
    if on_batch and (sampling or not on_wavelength):
        prefix = (1,) * len(chromatic) + prefix
    elif on_wavelength:
        prefix = prefix + (1,) * len(batch)

    # Explicit singleton dimensions remain views, never tiled metadata arrays.
    if np.broadcast_shapes(prefix, leading) != leading:
        raise ValueError(
            "Propagation parameters must broadcast to the wavefront batch."
        )
    prefix = (1,) * (len(leading) - len(prefix)) + prefix
    return value.reshape(prefix + suffix)


def _align_propagation_model(model, wavefront):
    """Align resolved physical parameters, keeping coordinate and matrix axes intact."""

    def align(value):
        if isinstance(value, GridSpec):
            value = value.sampled().broadcast(wavefront.grid.ndim)
            return value.set(
                d=_align_propagation_batch(value.d, wavefront, 1, sampling=True),
                c=_align_propagation_batch(value.c, wavefront, 1, sampling=True),
            )
        if isinstance(value, ResizeSpec):
            center = dlu.as_axis(value.c, wavefront.grid.ndim, "c")
            return value.set(
                c=_align_propagation_batch(center, wavefront, 1, sampling=True)
            )
        if isinstance(value, _prop.ABCD):
            return value.set(
                matrix=_align_propagation_batch(value.matrix, wavefront, 2)
            )
        if isinstance(value, (Array, core.Tracer)):
            return _align_propagation_batch(value, wavefront)
        return value

    metadata = lambda value: isinstance(
        value, (GridSpec, ResizeSpec, _prop.ABCD, zdx.Expression)
    )
    return tree.map(align, model, is_leaf=metadata)


class BasePropagator(_prop.BasePropagator, OpticalLayer):
    """Adapt inherited numerical propagation to the Wavefront interface.

    Concrete classes inherit their constructors and physical leaves directly from
    the numerical class. This adapter aligns wavelength and independent batch axes,
    preserves intrinsic Jones axes, and replaces only the phasor and output grid.
    Numerical propagation owns parameter resolution, units and the calculation.
    To adapt another numerical propagator, inherit BasePropagator before that
    numerical class. Its _prepare implementation supplies the calculation; the
    shared adapter supplies the Wavefront interface.
    """

    def apply(self, wavefront, **context):
        """Propagate a complete Wavefront, preserving spectral and Jones metadata.

        Parameters
        ----------
        wavefront : Wavefront
            Incident field, wavelength and physical sampling. Wavelength and grid
            populations follow Wavefront broadcasting semantics.
        **context
            External expression inputs, including optional state.

        Returns
        -------
        Wavefront
            The same concrete field type with updated phasor and realised grid.
        """
        # Resolve physical leaves on the incident plane, then align semantic axes.
        definition = _resolved(self, **self.context(wavefront, **context))
        if isinstance(definition, _prop.LCT):
            abcd = definition.abcd
            if abcd is not None and not isinstance(abcd, (dlu.ABCD, zdx.Expression)):
                definition = definition.set(abcd=dlu.ABCD(abcd))
        definition = _align_propagation_model(definition, wavefront)
        grid_in = _align_propagation_model(wavefront.grid, wavefront)
        wavelength = _align_propagation_batch(wavefront.wavelength, wavefront)

        # Delegate to the numerical parent on the aligned immutable definition.
        field, grid_out = super(BasePropagator, definition).__call__(
            wavefront.phasor, grid_in, wavelength, grid_out=True, **context
        )
        return wavefront.set(phasor=field, grid=grid_out)

    def __call__(self, wavefront, **context):
        """Apply propagation using the Wavefront calling convention."""
        return self.apply(wavefront, **context)

    def apply_mono(self, wavefront, **context):
        """Apply the same propagation to one monochromatic Wavefront."""
        return self.apply(wavefront, **context)


class Propagator(BasePropagator, _prop.Propagator):
    """Explicit-grid conjugate or custom ABCD propagation of a Wavefront.

    The inherited numerical constructor accepts grid_out, optional focal_length
    and defocus, or a custom abcd. Angular focal units require focal_length=None;
    a supplied focal length requires Cartesian units. Calling returns a Wavefront.

    Examples
    --------
    ```python
    import dLux as dl

    pupil = dl.GridSpec(n=32, diam=1.0, unit="m")
    focus = dl.GridSpec(n=24, d=20.0, unit="mas")
    wavefront = dl.Wavefront(600e-9, pupil)
    propagator = dl.Propagator(focus)
    output = propagator(wavefront)
    print(propagator)
    print(output.phasor.shape)
    ```
    """

    abcd: dlu.ABCD | zdx.Expression | None = None
    grid_out: GridSpec
    focal_length: Any = None
    defocus: Any = None
    inverse: bool | None = False


class FFTPropagator(BasePropagator, _prop.FFTPropagator):
    """Conjugate or custom ABCD Wavefront propagation on native FFT sampling.

    The inherited constructor accepts an FFTSpec or ResizeSpec and the same
    physical options as Propagator. Output spacing is derived at application.
    """

    abcd: dlu.ABCD | zdx.Expression | None = None
    grid_out: FFTSpec
    focal_length: Any = None
    defocus: Any = None
    inverse: bool | None = False


class SpectralPropagator(BasePropagator, _prop.SpectralPropagator):
    """ABCD Wavefront propagation through an intermediate frequency grid.

    The inherited constructor accepts abcd, grid_freq and grid_out. The spectral
    factorisation requires nonzero A, with Cartesian incident and output grids.
    """

    abcd: dlu.ABCD | zdx.Expression
    grid_freq: GridSpec | FFTSpec
    grid_out: GridSpec


class FreeSpace(BasePropagator, _prop.FreeSpace):
    """Signed-distance paraxial Wavefront propagation by FFT spectral transfer.

    The inherited constructor accepts distance in metres and optional resizing.
    Output spacing is inherited; padding and cropping change the sampled extent.
    Zero distance is supported. The axial carrier is omitted.
    """

    distance: Any
    resize: ResizeSpec
