"""Ordered layer systems for optical and detector modelling."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import jax.numpy as np

import dLux.utils as dlu

from .base import Base
from .grids import GridSpec
from .layers.detector import BaseDetectorLayer
from .layers.optical import BaseLayer, BaseOpticalLayer
from .fields import Image, PSF, Wavefront
from .sources import Spectrum

__all__ = ["LayeredSystem", "OpticalSystem", "DetectorSystem", "Detector"]


class LayeredSystem(Base):
    """Apply an ordered collection of layers to a compatible dLux object."""

    layers: OrderedDict

    def __init__(self, layers, layer_type=BaseLayer):
        self.layers = dlu.list2dictionary(layers, True, layer_type)

    def __getattr__(self, key: str) -> Any:
        """Resolve attributes from named or contained layers."""
        return dlu.resolve_attr(self, key, self.layers)

    def __call__(self, target):
        """Apply every layer to a target in insertion order."""
        for layer in self.layers.values():
            target = layer(target)
        return target

    def debug(self, target):
        """Apply every layer and return the intermediate states."""
        outputs = {"input": target}
        for name, layer in self.layers.items():
            target = layer(target)
            outputs[name] = target
        return target, outputs

    def insert_layer(self, layer, index: int, layer_type=BaseLayer):
        """Return a copy with a layer inserted at the requested index."""
        layers = dlu.insert_layer(self.layers, layer, index, layer_type)
        return self.set("layers", layers)

    def remove_layer(self, key: str):
        """Return a copy with the named layer removed."""
        return self.set("layers", dlu.remove_layer(self.layers, key))


class OpticalSystem(LayeredSystem, BaseOpticalLayer):
    """Model an optical train from a physical input coordinate specification.

    The input grid may use any supported length unit. Its coordinates are converted
    to canonical SI values when fields are evaluated; angular input grids are not
    accepted.
    """

    layers: OrderedDict
    grid: GridSpec

    def __init__(self, layers, grid: GridSpec):
        if not isinstance(grid, GridSpec):
            raise TypeError("grid must be a GridSpec.")
        grid = grid.broadcast(2)
        if grid.n is None or grid.d is None:
            raise ValueError("grid must define n and d.")
        dlu.canonical_unit(grid.unit, dimension="length", name="input grid unit")
        self.grid = grid
        super().__init__(layers, BaseOpticalLayer)

    @staticmethod
    def _to_psf(wavefront: Wavefront, stokes=None) -> PSF:
        """Convert a propagated wavefront into a sampled PSF."""
        data = wavefront.psf_from_stokes(stokes)
        mapped_sampling = wavefront.d.ndim > 1 or (
            wavefront.c is not None and wavefront.c.ndim > 1
        )
        if wavefront.is_chromatic and not mapped_sampling:
            data = data.sum(0)
        return PSF(data, wavefront.grid)

    def apply_mono(self, wavefront: Wavefront):
        """Propagate one monochromatic wavefront through every optical layer."""
        if not isinstance(wavefront, Wavefront):
            raise TypeError("wavefront must be a Wavefront instance.")
        return LayeredSystem.__call__(self, wavefront)

    def apply(self, wavefront: Wavefront):
        """Propagate the complete wavefront through every optical layer."""
        return LayeredSystem.__call__(self, wavefront)

    def __call__(self, wavefront: Wavefront):
        """Call :meth:`apply` using concise system syntax."""
        return self.apply(wavefront)

    def initialise_wavefront(self, wavelength, offset=None) -> Wavefront:
        """Construct an input Wavefront and apply an optional angular offset."""
        offset = np.zeros(2) if offset is None else np.asarray(offset)
        if offset.shape != (2,):
            raise ValueError("offset must have shape (2,).")
        return Wavefront(wavelength, self.grid).tilt(offset)

    def propagate_mono(
        self, wavelength, offset=None, return_wf=False, return_all=False, stokes=None
    ):
        """Propagate a monochromatic point source through the system.

        Returns the sampled PSF array by default, or the final wavefront when
        ``return_wf`` is true. ``return_all`` returns all output containers.
        """
        # Validate the requested output
        if return_wf and return_all:
            raise ValueError("return_wf and return_all are mutually exclusive.")

        # Initialize and propagate the monochromatic wavefront
        wavefront = self(self.initialise_wavefront(wavelength, offset))
        psf = self._to_psf(wavefront, stokes)

        # Return the requested output container
        if return_all:
            return {"Wavefront": wavefront, "PSF": psf, "psf": psf.data}
        if return_wf:
            return wavefront
        return psf.data

    def propagate(
        self,
        wavelengths,
        offset=None,
        weights=None,
        return_wf=False,
        return_all=False,
        stokes=None,
    ):
        """Propagate a weighted polychromatic point source through the system.

        Returns the sampled PSF array by default, or the final wavefront when
        ``return_wf`` is true. ``return_all`` returns all output containers.
        """
        # Validate the requested output
        if return_wf and return_all:
            raise ValueError("return_wf and return_all are mutually exclusive.")

        # Standardize and validate the spectrum
        wavelengths = np.atleast_1d(wavelengths)
        if weights is None:
            weights = np.ones_like(wavelengths) / wavelengths.size
        else:
            weights = np.atleast_1d(weights)

        if weights.shape != wavelengths.shape:
            raise ValueError("wavelengths and weights must have matching shapes.")

        # Initialize and spectrally weight the wavefront
        wavefront = self(self.initialise_wavefront(wavelengths, offset))
        ndim = wavefront.phasor.ndim - weights.ndim
        shape = weights.shape + (1,) * ndim
        scale = np.sqrt(weights).reshape(shape)
        wavefront = wavefront.set(phasor=wavefront.phasor * scale)

        # Convert the propagated wavefront into a PSF
        psf = self._to_psf(wavefront, stokes)

        # Return the requested output container
        if return_all:
            return {"Wavefront": wavefront, "PSF": psf}
        if return_wf:
            return wavefront
        return psf.data

    def model(self, source, return_all=False):
        """Model a spectral source, returning its PSF by default."""
        if not isinstance(source, Spectrum):
            raise TypeError("source must be a Spectrum.")
        return source.model(self, return_all)

    def debug_propagate_mono(self, wavelength, offset=None):
        """Propagate once and return all intermediate system states."""
        wavefront = self.initialise_wavefront(wavelength, offset)
        output, states = self.debug(wavefront)
        states = {
            "initial_wavefront": states["input"],
            **{name: value for name, value in states.items() if name != "input"},
        }
        return output, states


class DetectorSystem(LayeredSystem):
    """Transform a PSF through detector layers and produce an Image."""

    layers: OrderedDict

    def __init__(self, layers):
        super().__init__(layers, BaseDetectorLayer)

    def __call__(self, psf: PSF) -> PSF:
        """Apply every detector layer to a PSF."""
        if not isinstance(psf, PSF):
            raise TypeError("psf must be a PSF instance.")
        return super().__call__(psf)

    def model(self, psf: PSF, return_all=False):
        """Apply the detector model and return an Image."""
        output = self(psf)
        image = Image(output.data, output.grid)
        if return_all:
            return {"PSF": output, "Image": image}
        return image


# Backwards-compatible public name.
Detector = DetectorSystem
