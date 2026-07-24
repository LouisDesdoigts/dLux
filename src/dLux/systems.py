"""Ordered layer systems for optical and detector modelling."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import equinox as eqx
import jax.numpy as np
import zodiax as zdx

import dLux.utils as dlu
from .coordinates import CoordSpec
from .layers.detector_layers import BaseDetectorLayer
from .layers.optical_layers import BaseLayer, BaseOpticalLayer
from .states import PSF, Wavefront
from .sources import Spectrum

__all__ = ["LayeredSystem", "OpticalSystem", "Detector"]


class LayeredSystem(zdx.Base):
    """Apply an ordered collection of layers to a compatible dLux object."""

    layers: OrderedDict

    def __init__(self, layers, layer_type=BaseLayer):
        self.layers = dlu.list2dictionary(layers, True, layer_type)

    def __getattr__(self, key: str) -> Any:
        if key in self.layers:
            return self.layers[key]
        for layer in self.layers.values():
            if hasattr(layer, key):
                return getattr(layer, key)
        raise dlu.missing_attribute_error(self, key, list(self.layers))

    def __call__(self, target):
        for layer in self.layers.values():
            target = layer(target)
        return target

    def apply(self, target):
        """Backwards-compatible alias for calling the system."""
        return self(target)

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


class OpticalSystem(LayeredSystem):
    """Model an optical train from an input coordinate specification."""

    spec: CoordSpec

    def __init__(self, layers, spec: CoordSpec):
        if not isinstance(spec, CoordSpec):
            raise TypeError("spec must be a CoordSpec.")
        spec = spec.broadcast(2)
        if spec.n is None or spec.d is None:
            raise ValueError("spec must define n and d.")
        if spec.unit != "m":
            raise ValueError("OpticalSystem input coordinates must use metres.")
        self.spec = spec
        super().__init__(layers, BaseOpticalLayer)

    @staticmethod
    def _to_psf(wavefront: Wavefront, stokes=None) -> PSF:
        data = wavefront.psf_from_stokes(stokes)
        if wavefront.is_chromatic:
            data = data.sum(0)
        return PSF(data, wavefront.spec)

    def __call__(self, wavefront: Wavefront):
        if not isinstance(wavefront, Wavefront):
            raise TypeError("wavefront must be a Wavefront instance.")

        def apply(value):
            output = LayeredSystem.__call__(self, value)
            return output.set(spec=None), output.spec

        if not wavefront.is_chromatic:
            return LayeredSystem.__call__(self, wavefront)
        mapped = eqx.filter_vmap(
            apply,
            in_axes=(wavefront._mapped_axis,),
            out_axes=(eqx.if_array(0), None),
        )
        output, spec = mapped(wavefront)
        return output.set(spec=spec)

    def initialise_wavefront(self, wavelength, offset=None) -> Wavefront:
        """Construct an input Wavefront and apply an optional angular offset."""
        offset = np.zeros(2) if offset is None else np.asarray(offset)
        if offset.shape != (2,):
            raise ValueError("offset must have shape (2,).")
        return Wavefront(wavelength, self.spec).tilt(offset)

    def propagate_mono(self, wavelength, offset=None, return_all=False, stokes=None):
        """Propagate a monochromatic point source through the system."""
        wavefront = self(self.initialise_wavefront(wavelength, offset))
        psf = self._to_psf(wavefront, stokes)
        if return_all:
            return {"Wavefront": wavefront, "PSF": psf, "psf": psf.data}
        return psf.data

    def propagate(
        self,
        wavelengths,
        offset=None,
        weights=None,
        return_all=False,
        stokes=None,
    ):
        """Propagate a weighted polychromatic point source through the system."""
        wavelengths = np.atleast_1d(wavelengths)
        weights = (
            np.ones_like(wavelengths) / wavelengths.size
            if weights is None
            else np.atleast_1d(weights)
        )
        if weights.shape != wavelengths.shape:
            raise ValueError("wavelengths and weights must have matching shapes.")

        wavefront = self.initialise_wavefront(wavelengths, offset)
        weights = weights.reshape(weights.shape + (1, 1))
        wavefront = wavefront.set(phasor=wavefront.phasor * np.sqrt(weights))
        wavefront = self(wavefront)
        psf = self._to_psf(wavefront, stokes)
        if return_all:
            return {"Wavefront": wavefront, "PSF": psf, "psf": psf.data}
        return psf.data

    def model(
        self,
        source,
        return_all=False,
    ):
        """Model a spectral source through the optical system."""
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


class Detector(LayeredSystem):
    """Apply detector and unified layers to a PSF."""

    def __init__(self, layers):
        super().__init__(layers, BaseDetectorLayer)

    def __call__(self, psf: PSF, return_all=False):
        if not isinstance(psf, PSF):
            raise TypeError("psf must be a PSF instance.")
        output = super().__call__(psf)
        if return_all:
            return {"PSF": output, "psf": output.data}
        return output.data

    def model(self, psf: PSF, return_all=False):
        """Apply this detector to a PSF."""
        return self(psf, return_all)
