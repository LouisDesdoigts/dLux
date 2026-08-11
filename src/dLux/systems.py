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
from .fields import Intensity, Wavefront
from .sources import Spectrum

__all__ = ["LayeredSystem", "OpticalSystem", "DetectorSystem"]


class LayeredSystem(Base):
    """Apply an ordered collection of layers to a compatible dLux object."""

    layers: OrderedDict

    def __init__(self, layers, layer_type=BaseLayer):
        self.layers = dlu.list2dictionary(layers, True, layer_type)

    def __getattr__(self, key: str) -> Any:
        """Resolve attributes from named or contained layers."""
        return dlu.resolve_attr(self, key, self.layers)

    def apply(self, target):
        """Apply every layer to a target in insertion order."""
        for layer in self.layers.values():
            target = layer(target)
        return target

    def __call__(self, target):
        """Call :meth:`apply` using concise system syntax."""
        return self.apply(target)

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
    def _to_intensity(wavefront: Wavefront, stokes=None) -> Intensity:
        """Convert a propagated wavefront into sampled intensity."""
        intensity = wavefront.to_intensity(stokes)
        mapped_sampling = wavefront.d.ndim > 1 or (
            wavefront.c is not None and wavefront.c.ndim > 1
        )
        if wavefront.is_chromatic and not mapped_sampling:
            intensity = intensity.set(data=intensity.data.sum(0))
        return intensity

    def apply_mono(self, wavefront: Wavefront):
        """Propagate one monochromatic wavefront through every optical layer."""
        return self.apply(wavefront)

    def apply(self, wavefront: Wavefront):
        """Propagate the complete wavefront through every optical layer."""
        if not isinstance(wavefront, Wavefront):
            raise TypeError("wavefront must be a Wavefront instance.")
        return super().apply(wavefront)

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

        Returns the sampled intensity array by default, or the final wavefront when
        ``return_wf`` is true. ``return_all`` returns all output containers.
        """
        # Validate the requested output
        if return_wf and return_all:
            raise ValueError("return_wf and return_all are mutually exclusive.")

        # Initialize and propagate the monochromatic wavefront
        wavefront = self(self.initialise_wavefront(wavelength, offset))
        intensity = self._to_intensity(wavefront, stokes)

        # Return the requested output container
        if return_all:
            return {
                "Wavefront": wavefront,
                "Intensity": intensity,
                "intensity": intensity.data,
                "PSF": intensity,
                "psf": intensity.data,
            }
        if return_wf:
            return wavefront
        return intensity.data

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

        Returns the sampled intensity array by default, or the final wavefront when
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

        # Convert the propagated wavefront into sampled intensity
        intensity = self._to_intensity(wavefront, stokes)

        # Return the requested output container
        if return_all:
            return {
                "Wavefront": wavefront,
                "Intensity": intensity,
                "PSF": intensity,
            }
        if return_wf:
            return wavefront
        return intensity.data

    def model(self, source, return_all=False):
        """Model a spectral source, returning its intensity by default."""
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
    """Apply deterministic detector transformations to an intensity."""

    layers: OrderedDict

    def __init__(self, layers):
        super().__init__(layers, BaseDetectorLayer)

    def apply(self, intensity: Intensity) -> Intensity:
        """Apply every detector layer to an intensity."""
        if not isinstance(intensity, Intensity):
            raise TypeError("intensity must be an Intensity instance.")
        return super().apply(intensity)

    def model(self, intensity: Intensity, return_all=False):
        """Apply the detector model to an intensity."""
        output = self(intensity)
        return {"Intensity": output} if return_all else output
