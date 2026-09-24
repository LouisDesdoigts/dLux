"""Released system constructors built from current optical sequences and Imagers."""

from math import prod
import jax.numpy as np
import dLux.utils as dlu
from ..fields import Wavefront
from ..systems.optical import OpticalSystem
from ..systems.imagers import Imager
from ..layers.optical import Optic
from .modelling import result, model_source, spectral_weights
from .warnings import warn_deprecated
from .layers.propagators import _Propagation

BaseOpticalSystem = OpticalSystem
__all__ = [
    "OpticalSystem",
    "BaseOpticalSystem",
    "LayeredOpticalSystem",
    "ParametricOpticalSystem",
    "AngularOpticalSystem",
    "CartesianOpticalSystem",
]


class _SampledSequence(OpticalSystem):
    """Current optical sequence with input sampling and released return conveniences."""

    grid: dlu.Grid

    def __init__(self, grid, layers):
        super().__init__(layers, grid)

    @property
    def wf_npixels(self):
        return self.grid.n[0]

    @property
    def diameter(self):
        return self.grid.fov[0]

    def propagate_mono(self, wavelength, offset=None, return_wf=False):
        wavefront = Wavefront(
            wavelength, npixels=self.wf_npixels, diameter=self.diameter
        )
        if offset is not None:
            wavefront = wavefront.tilt(offset)
        return result(self.apply(wavefront), return_wf=return_wf)

    def propagate(
        self, wavelengths, offset=None, weights=None, return_wf=False, return_psf=False
    ):
        wavelengths = np.atleast_1d(np.asarray(wavelengths))
        weights = (
            np.ones_like(wavelengths) / wavelengths.size
            if weights is None
            else np.asarray(weights)
        )
        field = self.propagate_mono(wavelengths, offset, return_wf=True)
        return result(field, weights, return_wf=return_wf, return_psf=return_psf)

    def initialise_wavefront(self, wavelength, offset=None, **context):
        field = super().initialise_wavefront(wavelength, offset, **context)
        return field * (1 / np.sqrt(prod(self.grid.n)))

    def debug_propagate_mono(self, wavelength, offset=None):
        wavefront = Wavefront(
            wavelength, npixels=self.wf_npixels, diameter=self.diameter
        )
        wavefront = wavefront if offset is None else wavefront.tilt(offset)
        output, steps = self.debug(wavefront)
        return output, {"initial_wavefront": wavefront, **steps}


class _PointWeightsImager(Imager):
    """Use the current imaging stages with released discrete spectral weights.

    Only the photometry boundary differs: old samples already represent photon
    masses and must not be interpolated or multiplied by wavelength-bin widths.
    Current density spectra still use the ordinary BandPass integration route.
    """

    def _photometry(self, source, bandpass, filter=None, qe=None, **context):
        """Keep released point weights at this compatibility boundary only."""
        sampling = source.spectrum.bandpass if bandpass is None else bandpass
        if isinstance(sampling, dlu.BandPass):
            return super()._photometry(source, sampling, filter, qe, **context)
        return spectral_weights(source, bandpass, filter, qe, **context)


class _Imager(_PointWeightsImager):
    """Current Imager with released array/PSF return flags."""

    @property
    def wf_npixels(self):
        return self.pupil_grid.n[0]

    @property
    def diameter(self):
        return self.pupil_grid.fov[0]

    @property
    def psf_npixels(self):
        return self.image_grid.n[0]

    @property
    def psf_pixel_scale(self):
        return self.image_grid.d[0]

    @property
    def fov(self):
        return self.psf_npixels * self.psf_pixel_scale

    @property
    def layers(self):
        return self.pupil.layers

    def initialise_wavefront(self, wavelength, offset=None, **context):
        field = super().initialise_wavefront(wavelength, offset, **context)
        return field.set(phasor=field.phasor / np.sqrt(prod(self.pupil_grid.n)))

    def _propagate(self, wavefront, offset=None, *, unit_input=False, **context):
        field = super()._propagate(wavefront, offset, unit_input=unit_input, **context)
        # The released Fourier relay omitted the two-dimensional Collins factor
        # exp(-i*pi/2). Remove that global phase on this released interface only.
        return field.set(phasor=1j * field.phasor)

    def propagate_mono(self, wavelength, offset=None, return_wf=False):
        return result(self._point_wavefront(wavelength, offset), return_wf=return_wf)

    def propagate(
        self, wavelengths, offset=None, weights=None, return_wf=False, return_psf=False
    ):
        wavelengths = np.atleast_1d(np.asarray(wavelengths))
        weights = (
            np.ones_like(wavelengths) / wavelengths.size
            if weights is None
            else np.asarray(weights)
        )
        return result(
            self._point_wavefront(wavelengths, offset),
            weights,
            return_wf=return_wf,
            return_psf=return_psf,
        )

    def model(self, source=None, return_wf=False, return_psf=False):
        source = self.source if source is None else source
        return model_source(self, source, return_wf, return_psf)

    def debug_propagate_mono(self, wavelength, offset=None):
        field = self.initialise_wavefront(wavelength, offset)
        field, outputs = self.pupil.debug(field)
        outputs = {
            "initial_wavefront": self.initialise_wavefront(wavelength, offset),
            **outputs,
        }
        outputs["output"] = self.apply(self.initialise_wavefront(wavelength, offset))
        return outputs["output"], outputs


def LayeredOpticalSystem(wf_npixels, diameter, layers):
    warn_deprecated(
        "LayeredOpticalSystem",
        "OpticalSystem",
        "OpticalSystem(layers, grid=Grid(wf_npixels, diam=diameter, unit='m'))",
    )
    return _SampledSequence(dlu.Grid(wf_npixels, diam=diameter, unit="m"), layers)


def AngularOpticalSystem(
    wf_npixels, diameter, layers, psf_npixels, psf_pixel_scale, oversample=1
):
    return _imager(
        wf_npixels, diameter, layers, psf_npixels, psf_pixel_scale, oversample, "arcsec"
    )


def CartesianOpticalSystem(
    wf_npixels,
    diameter,
    layers,
    focal_length,
    psf_npixels,
    psf_pixel_scale,
    oversample=1,
):
    return _imager(
        wf_npixels,
        diameter,
        layers,
        psf_npixels,
        psf_pixel_scale,
        oversample,
        "um",
        focal_length,
    )


def _imager(
    n, diameter, layers, output_n, spacing, oversample, unit, focal_length=None
):
    warn_deprecated(
        "Angular/CartesianOpticalSystem",
        "Imager",
        "Imager(pupil_grid, image_grid, pupil=dlu.Sequential(layers))",
    )
    sequence = OpticalSystem(layers)
    for step in sequence.steps.values():
        propagates = (
            step.grid_out is not None
            if isinstance(step, Optic)
            else isinstance(step, (dlu.BasePropagator, _Propagation))
        )
        if propagates:
            raise ValueError(
                "Intermediate propagation requires LayeredOpticalSystem with an "
                "explicit final propagator; this constructor builds an Imager pupil."
            )
    if focal_length is not None:
        sequence = sequence.insert_layer(
            ("relay", Optic(focal_length=focal_length)), len(sequence.layers)
        )
    pupil = dlu.Sequential(sequence.layers)
    return _Imager(
        dlu.Grid(n, diam=diameter, unit="m"),
        dlu.Grid(output_n, spacing, unit=unit),
        pupil=pupil,
        oversample=oversample,
    )


def ParametricOpticalSystem(*args, **kwargs):
    raise TypeError(
        "Construct Imager with explicit pupil and image grids instead of subclassing "
        "ParametricOpticalSystem."
    )
