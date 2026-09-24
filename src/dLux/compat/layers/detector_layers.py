"""Released detector constructors using current deterministic layers."""

import jax.numpy as np
import zodiax as zdx
import dLux.utils as dlu
from ...layers.detector import (
    DetectorLayer,
    SensitivityLayer,
    ConvolveLayer,
    SaturationLayer,
    BiasLayer,
)
from ...layers.unified import DownsampleLayer as Downsample
from ..warnings import warn_deprecated

__all__ = [
    "DetectorLayer",
    "ApplyPixelResponse",
    "ApplyJitter",
    "ApplySaturation",
    "AddConstant",
    "Downsample",
]


class _JitterKernel(dlu.Jitter):
    """Keep released sigma values while evaluating with the current pixel grid."""

    def evaluate(self, *, grid=None, oversample=1, **context):
        # Released samples span [-5, 5], rather than kernel_size native pixels.
        # Convert at evaluation so immutable width and sampling updates stay live.
        sigma = zdx.resolve(self.profile.std, **context)
        size = np.asarray(self.kernel_size)
        samples = np.asarray(self.oversample)
        sigma_pixels = sigma * (size * samples - 1) / (10 * samples)
        converted = self.set("profile.std", sigma_pixels)

        # Reuse the canonical calculation without changing the stored sigma.
        return dlu.Jitter.evaluate(
            converted, grid=grid, oversample=oversample, **context
        )


class _Jitter(ConvolveLayer):
    """Read the released Gaussian width from the current kernel definition."""

    @property
    def sigma(self):
        return self.kernel.profile.std


class _Saturation(SaturationLayer):
    """Read the released threshold name from the current limit."""

    @property
    def threshold(self):
        return self.limit


class _Bias(BiasLayer):
    """Read the released constant name from the current bias."""

    @property
    def value(self):
        return self.bias


def ApplyPixelResponse(pixel_response):
    warn_deprecated(
        "ApplyPixelResponse", "SensitivityLayer", "SensitivityLayer(response=...)"
    )
    pixel_response = np.asarray(pixel_response, dtype=float)
    if pixel_response.ndim != 2:
        raise ValueError("pixel_response must be a two-dimensional array.")
    return SensitivityLayer(pixel_response)


def ApplyJitter(sigma, kernel_size=9, oversample=3):
    """Preserve the released width convention using the current blur calculation.

    The old fixed [-5, 5] kernel sampling corresponds to a pixel width of
    ``sigma * (kernel_size * oversample - 1) / (10 * oversample)``.
    Stored sigma retains its released meaning; dlu.Jitter uses actual pixels.
    """
    warn_deprecated(
        "ApplyJitter",
        "ConvolveLayer",
        "ConvolveLayer(dlu.Jitter(sigma_pixels, ...)); "
        "sigma_pixels = sigma * (kernel_size * oversample - 1) / (10 * oversample)",
    )
    return _Jitter(_JitterKernel(sigma, kernel_size, oversample))


def ApplySaturation(threshold):
    warn_deprecated("ApplySaturation", "SaturationLayer", "SaturationLayer(threshold)")
    return _Saturation(threshold)


def AddConstant(value):
    warn_deprecated("AddConstant", "BiasLayer", "BiasLayer(value)")
    return _Bias(value)
