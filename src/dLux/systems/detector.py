"""Detector response, sampling and readout."""

from __future__ import annotations

from collections import OrderedDict

import jax.numpy as np
import zodiax as zdx
from jax import Array, core

import dLux.utils as dlu
from ..fields import Image, Intensity, Wavefront
from ..layers.detector import DetectorLayer

__all__ = ["Detector", "DetectorSystem"]


class DetectorSystem(dlu.Sequential):
    """A standalone sequence mapping Intensity to Intensity.

    This is the detector-layer version of Sequential. It owns no photometry,
    sampling recipe or noise model and is not an Imager's Detector.
    """

    # Inherited fields
    layers: OrderedDict

    def __init__(self, layers=(), *, alias=None):
        """Store named detector layers or nested compatible sequences."""
        super().__init__(layers, DetectorLayer, alias=alias)

    def apply(self, intensity, **context):
        """Apply the ordered layers to an existing Intensity."""
        if not isinstance(intensity, Intensity):
            raise TypeError("DetectorSystem requires an Intensity.")
        return super().apply(intensity, **context)

    def insert_layer(self, layer, index, layer_type=DetectorLayer):
        """Insert a compatible detector layer at the requested position."""
        return super().insert_layer(layer, index, layer_type)


class Detector(zdx.Module):
    """Sample and detect an optical image without storing a second grid.

    Sampling comes from the incoming field. ``resample`` acts on the complex
    wavefront; ``layers`` acts on expected photoelectrons before pixel integration.
    Simulation then draws Poisson counts, adds read noise in electrons, and applies
    gain in output units per electron. Imager owns spectral weighting, including QE.
    """

    layers: OrderedDict
    qe: Array | zdx.Expression | dlu.Filter | None = None
    read_noise: Array | zdx.Expression | None = None
    gain: Array | zdx.Expression | None = None
    resample: dlu.Interp | None = None
    downsample: tuple[int, int] = (1, 1)

    def __init__(
        self,
        layers=(),
        *,
        qe=None,
        read_noise=None,
        gain=None,
        resample=None,
        downsample=1,
        alias=None,
    ):
        """Store a photon response and optional sampling/readout operations.

        Parameters
        ----------
        layers : mapping or sequence
            Named detector layers, such as jitter and sensitivity, applied before
            pixel integration and photon noise. Readout gain has its own leaf.
        qe : Array, Expression, Filter, or None
            Spectral conversion fraction sampled by the photometric model. None
            means unit efficiency. Spatial sensitivity belongs in layers.
        read_noise : Array, Expression, or None
            Gaussian standard deviation in electrons, added before gain.
        gain : Array, Expression, or None
            Output units per electron; None means unit gain.
        resample : Interp or None
            Optional rotation, scaling, shift or explicit resampling of the incoming
            complex field. Its input grid is supplied by that field. Shifts therefore
            use SI units. Interpolation samples the complex values directly,
            without amplitude rescaling. None keeps the incoming samples.
        downsample : int or pair of ints
            Positive pixel-integration factors in physical (x, y) order. Defaults
            to one. The output grid follows the integrated input sampling.
        alias : sequence or None
            User-defined parameter aliases.
        """
        # Store detector response settings and require complex-field spatial resampling.
        self.layers = DetectorSystem(layers).layers
        self.qe = qe if isinstance(qe, dlu.Filter) else zdx.as_array(qe, dtype=float)
        self.read_noise = zdx.as_array(read_noise, dtype=float)
        self.gain = zdx.as_array(gain, dtype=float)
        if resample is not None and not isinstance(resample, dlu.Interp):
            raise TypeError("resample must be an Interp or None.")
        if resample is not None and resample.ndim != 2:
            raise ValueError("Detector resampling requires ndim=2.")
        if resample is not None and not resample.complex:
            raise ValueError("Detector resampling requires complex=True.")
        self.resample = resample
        self.downsample = dlu.as_size(downsample, 2, "downsample")
        self.alias = alias

    def sample(self, wavefront: Wavefront, *, oversample=1, **context) -> Wavefront:
        """Resample physical amplitudes before forming intensity."""
        if self.resample is None:
            return wavefront
        operation = self.resample
        if operation.grid_out is not None:
            # An explicit target describes native pixels, just like image_grid.
            context = dict(
                context,
                wavelength=wavefront.batch_wavelength,
                wavelengths=wavefront.batch_wavelength,
            )
            output = operation.grid_out.resolve(**context)
            if output.n is None:
                native_shape = wavefront.grid.downsample(oversample).shape
                output = output.match_shape(native_shape)
            operation = operation.set(grid_out=output.oversample(oversample))
        return wavefront.resample(operation, **context)

    def prepare(self, intensity: Intensity, *, oversample=1, **context) -> Intensity:
        """Apply the photon-response layers, then integrate neighbouring pixels."""
        for layer in self.populate().layers.values():
            intensity = layer(intensity, oversample=oversample, **context)
        # Integrate optical subpixels and optional detector binning in one sum.
        optical = dlu.as_size(oversample, 2, "oversample")
        factors = tuple(a * b for a, b in zip(optical, self.downsample))
        return intensity.downsample(factors, mean=False)

    def readout(self, intensity: Intensity, **context) -> Intensity:
        """Apply gain after detection and any read-noise draw."""
        return intensity * self._gain(intensity, **context)

    def _gain(self, intensity, **context):
        """Resolve a linear output-per-electron factor on the final pixel grid."""
        if self.gain is None:
            return 1.0
        context = DetectorLayer.context(intensity, **context)
        return zdx.resolve(self.gain, **context)

    def noise(self, intensity, *, output=False, **context):
        """Resolve read-noise sigma in electrons, or after gain when output=True."""
        if self.read_noise is None:
            return None
        context = DetectorLayer.context(intensity, **context)
        sigma = zdx.resolve(self.read_noise, **context)
        invalid = np.any(~np.isfinite(sigma) | (sigma < 0))
        if not isinstance(invalid, core.Tracer) and bool(invalid):
            raise ValueError("read_noise must be finite and non-negative.")
        return sigma * np.abs(self._gain(intensity, **context)) if output else sigma

    def image(
        self,
        intensity: Intensity,
        *,
        oversample=1,
        key=None,
        n_frames=1,
        **context,
    ) -> Image:
        """Detect expected photoelectrons, returning their sampled Image.

        Response layers run before optical subpixels and detector pixels are
        integrated. Without a key, return the expectation after gain. With a key,
        draw Poisson counts, add read noise in electrons, then apply gain to each
        exposure. Multiple frames return their mean on the same detector grid.
        QE has already been applied by photometry, before wavelength integration.
        """
        if not isinstance(intensity, Intensity):
            raise TypeError("Detector.image requires an Intensity.")
        if not isinstance(n_frames, int) or n_frames < 1:
            raise ValueError("n_frames must be a positive integer.")
        self = self.populate()
        counts = self.prepare(intensity, oversample=oversample, **context)
        if key is None:
            return Image(self.readout(counts, **context))

        samples = Image(counts).simulate(
            key, n_frames, read_noise=self.noise(counts, **context)
        )
        # The same linear gain applies to every exposure, so it also scales
        # their mean. Image owns the shared Poisson/read-noise calculation.
        counts = counts.set(data=samples.data)
        return Image(self.readout(counts, **context))

    def apply(self, intensity: Intensity, **context) -> Image:
        """Return a detector Image from expected photoelectrons."""
        return self.image(intensity, **context)

    def __call__(self, intensity, **context):
        """Return the expected detector Image, or simulate when supplied a key."""
        return self.apply(intensity, **context)
