"""Released propagation constructors returning current numerical propagators."""

import dLux.utils as dlu
import jax.numpy as np
from ...layers.optical import OpticalLayer
from ..warnings import warn_deprecated

Propagator = dlu.BasePropagator
__all__ = ["Propagator", "FFT", "MFT"]


class _Propagation(OpticalLayer):
    """Adapt any current numerical propagator to the released Wavefront call."""

    operation: dlu.BasePropagator
    phase: complex = 1.0

    def __init__(self, operation, phase=1.0):
        self.operation = operation
        self.phase = phase

    @property
    def n(self):
        sampling = getattr(self.operation, "grid_out", None)
        if sampling is None:
            sampling = self.operation.resize
        return None if sampling.n is None else sampling.n[0]

    @property
    def pixel_scale(self):
        return self.operation.grid_out.d[0]

    def apply(self, wavefront, **context):
        field = wavefront.propagate(self.operation, **context)
        return field.set(phasor=field.phasor * self.phase)

    def apply_mono(self, wavefront, **context):
        return self.apply(wavefront, **context)


class _ReleasedFFT(dlu.FFTPropagator):
    """Use released output phases with the current FFT preparation machinery."""

    def _output_ramp(self, working, grid_freq, n_out, inverse):
        # Released uncentred FFTs applied neither an origin ramp nor Collins phase.
        if self.grid_out.c is None:
            return dlu.Chirp.from_axes(grid_freq.resize_samples(n_out).axes, constant=0)

        # Centred FFTs applied only the half-pixel ramp on centred output samples.
        axes = grid_freq.set(c=np.zeros(2)).resize_samples(n_out).axes
        offset = np.where(np.asarray(working.n) % 2 == 0, working.d_si / 2, 0)
        direction = 1 if inverse else -1
        linear = tuple(direction * 2 * np.pi * offset)
        return dlu.Chirp.from_axes(axes, linear=linear)


def FFT(focal_length=None, inverse=False, pad=1, crop=1, center=True):
    warn_deprecated(
        "compat.FFT",
        "dl.FFTPropagator",
        "FFTPropagator(FFTSpec(pad=..., crop=...))",
    )
    return _Propagation(
        _ReleasedFFT(
            dlu.FFTSpec(pad=pad, crop=crop, c=0 if center else None),
            focal_length=focal_length,
            inverse=inverse,
        )
    )


def MFT(npixels, pixel_scale, focal_length=None, inverse=False):
    warn_deprecated(
        "compat.MFT", "dl.Propagator", "Propagator(Grid(...), focal_length=...)"
    )
    unit = "rad" if focal_length is None and not inverse else "m"
    # Released MFT omitted the Collins global phase; retain its complex values.
    return _Propagation(
        dlu.Propagator(
            dlu.Grid(npixels, pixel_scale, unit=unit),
            focal_length=focal_length,
            inverse=inverse,
        ),
        phase=-1j if inverse else 1j,
    )
