"""User-facing optical elements and direct wavefront operations."""

from __future__ import annotations

from typing import Any

import jax.numpy as np
from jax import Array

import dLux.utils as dlu

from ..coordinates import BaseCoordTransform
from ..parametric import BaseParametric, Shape
from ..wavefronts import Wavefront
from .polarised_layers import BasePolarisingOptic
from .unified_layers import BaseOpticalLayer

__all__ = [
    "TransmissiveLayer",
    "AberratedLayer",
    "Normalise",
    "Optic",
    "DynamicOptic",
    "Lens",
    "Wedge",
    "Interpolate",
    "Tilt",
]


class TransmissiveLayer(BaseOpticalLayer):
    """Apply a transmission, with optional output normalisation."""

    transmission: Array | BaseParametric | None
    normalise: bool

    def __init__(self, transmission=None, normalise=False):
        self.transmission = self.as_parametric(transmission)
        self.normalise = bool(normalise)

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        transmission = self.resolve(self.transmission, wavefront=wavefront)
        if transmission is not None:
            transmission = wavefront._to_phasor_shape(transmission)
            wavefront = wavefront.set(phasor=wavefront.phasor * transmission)
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront


class AberratedLayer(BaseOpticalLayer):
    """Apply optical-path and phase aberrations to a wavefront."""

    opd: Array | BaseParametric | None
    phase: Array | BaseParametric | None

    def __init__(self, opd=None, phase=None):
        self.opd = self.as_parametric(opd)
        self.phase = self.as_parametric(phase)

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        opd = self.resolve(self.opd, wavefront=wavefront)
        phase = self.resolve(self.phase, wavefront=wavefront)
        wavefront = wavefront.add_opd(opd)
        return wavefront.add_phase(phase)


class Normalise(BaseOpticalLayer):
    """Normalise a wavefront to unit power."""

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        return wavefront.normalise()


class Optic(TransmissiveLayer, AberratedLayer, Normalise):
    """A physical optic evaluated at one plane, with optional onward propagation."""

    transmission: Array | BaseParametric | None
    opd: Array | BaseParametric | None
    phase: Array | BaseParametric | None
    polarisation: dict | None
    normalise: bool
    propagator: BaseOpticalLayer | None

    def __init__(
        self,
        transmission=None,
        opd=None,
        phase=None,
        polarisation=None,
        normalise=False,
        propagator=None,
    ):
        TransmissiveLayer.__init__(self, transmission, normalise)
        AberratedLayer.__init__(self, opd, phase)
        self.polarisation = self._parse_polarisation(polarisation)
        if propagator is not None and not isinstance(propagator, BaseOpticalLayer):
            raise TypeError("propagator must be a BaseOpticalLayer or None.")
        self.propagator = propagator

    @staticmethod
    def _parse_polarisation(polarisation) -> dict | None:
        if polarisation is None:
            return None
        items = (
            list(polarisation)
            if isinstance(polarisation, (list, tuple))
            else [polarisation]
        )
        return dlu.list2dictionary(items, True, BasePolarisingOptic)

    def context(self, wavefront: Wavefront) -> dict[str, Any]:
        """Return the parameter context shared by this optic's properties."""
        return {"wavefront": wavefront}

    def params(self, wavefront: Wavefront) -> dict[str, Any]:
        """Resolve all physical parameters exactly once for one application."""
        context = self.context(wavefront)
        return {
            "transmission": self.resolve(self.transmission, **context),
            "opd": self.resolve(self.opd, **context),
            "phase": self.resolve(self.phase, **context),
            "polarisation": self.polarisation,
        }

    def phasor(self, wavefront: Wavefront, params: dict = None) -> Array:
        """Return the cumulative complex scalar field for this optical plane."""
        params = self.params(wavefront) if params is None else params
        transmission = params["transmission"]
        opd = params["opd"]
        phase = params["phase"]

        transmission = 1.0 if transmission is None else transmission
        opd = 0.0 if opd is None else opd
        phase = 0.0 if phase is None else phase

        wavenumber = wavefront._to_phasor_shape(wavefront.wavenumber)
        opd = wavefront._to_phasor_shape(opd)
        phase = wavefront._to_phasor_shape(phase)
        transmission = wavefront._to_phasor_shape(transmission)
        return transmission * np.exp(1j * (wavenumber * opd + phase))

    @staticmethod
    def polarisation_matrix(polarisation, wavefront: Wavefront) -> Array | None:
        """Evaluate and compose polarising optics in their listed physical order."""
        if polarisation is None:
            return None
        matrix = np.eye(2, dtype=complex)
        for optic in polarisation.values():
            if hasattr(optic, "evaluate_jones"):
                jones = optic.evaluate_jones(wavefront)
            elif hasattr(optic, "orientation"):
                jones = dlu.rotate_jones(optic.jones, optic.orientation)
            else:
                jones = optic.jones
            matrix = np.einsum("ij...,jk...->ik...", jones, matrix)
        return matrix

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        params = self.params(wavefront)
        phasor = wavefront.phasor * self.phasor(wavefront, params)
        wavefront = wavefront.set(phasor=phasor)
        polarisation = self.polarisation_matrix(params["polarisation"], wavefront)
        if polarisation is not None:
            wavefront = wavefront.apply_jones(polarisation)
        if self.normalise:
            wavefront = wavefront.normalise()
        if self.propagator is not None:
            wavefront = self.propagator(wavefront)
        return wavefront


class DynamicOptic(Optic):
    """An optic evaluated in one shared transformed coordinate frame."""

    aperture: Shape
    transformation: BaseCoordTransform | None

    def __init__(
        self,
        aperture,
        transformation=None,
        opd=None,
        phase=None,
        polarisation=None,
        normalise=False,
        propagator=None,
    ):
        if not isinstance(aperture, Shape):
            raise TypeError("aperture must be a Shape.")
        if transformation is not None and not isinstance(
            transformation, BaseCoordTransform
        ):
            raise TypeError("transformation must be a BaseCoordTransform or None.")
        self.aperture = aperture
        self.transformation = transformation
        super().__init__(
            opd=opd,
            phase=phase,
            polarisation=polarisation,
            normalise=normalise,
            propagator=propagator,
        )

    def context(self, wavefront: Wavefront) -> dict[str, Any]:
        """Return a shared coordinate context for every dynamic property."""
        coordinates = wavefront.coordinates()
        if self.transformation is not None:
            coordinates = self.transformation(coordinates)
        extent = self.aperture.extent
        return {
            "wavefront": wavefront,
            "coordinates": coordinates,
            "pixel_scale": wavefront.pixel_scale,
            "diameter": wavefront.diameter if extent is None else 2 * extent,
            "aperture": self.aperture,
        }

    def params(self, wavefront: Wavefront) -> dict[str, Any]:
        """Resolve the aperture and remaining properties in one shared context."""
        context = self.context(wavefront)
        return {
            "transmission": self.aperture.evaluate(**context),
            "opd": self.resolve(self.opd, **context),
            "phase": self.resolve(self.phase, **context),
            "polarisation": self.polarisation,
        }


class Lens(Optic):
    """A residual refractive lens applied as an optical layer."""

    thickness: Array | BaseParametric
    n: Array | BaseParametric

    def __init__(
        self,
        thickness,
        n,
        transmission=None,
        opd=None,
        phase=None,
        polarisation=None,
        normalise=False,
        propagator=None,
    ):
        """
        Parameters
        ----------
        thickness : Array | BaseParametric, metres
            Material thickness relative to the ideal optic. This represents residual
            figure or fabrication errors; ideal focusing remains in the propagator.
        n : Array | BaseParametric
            Refractive index, optionally dependent on wavefront wavelength.
        """
        self.thickness = self.as_parametric(thickness)
        self.n = self.as_parametric(n)
        super().__init__(
            transmission=transmission,
            opd=opd,
            phase=phase,
            polarisation=polarisation,
            normalise=normalise,
            propagator=propagator,
        )

    def params(self, wavefront: Wavefront) -> dict[str, Any]:
        """Resolve the optic properties and residual refractive OPD once."""
        context = self.context(wavefront)
        thickness = self.resolve(self.thickness, **context)
        n = np.asarray(self.resolve(self.n, **context) - 1)
        if n.ndim:
            n = n[..., None, None]
        opd = self.resolve(self.opd, **context)
        opd = 0.0 if opd is None else opd
        return {
            "transmission": self.resolve(self.transmission, **context),
            "opd": opd + n * thickness,
            "phase": self.resolve(self.phase, **context),
            "polarisation": self.polarisation,
        }


class Wedge(Optic):
    """A thin refractive wedge applied as an optical layer."""

    angle: Array
    n: Array | BaseParametric
    reference_wavelength: Array | None

    def __init__(
        self,
        angle,
        n,
        reference_wavelength=None,
        transmission=None,
        opd=None,
        phase=None,
        polarisation=None,
        normalise=False,
        propagator=None,
    ):
        self.angle = np.asarray(angle, dtype=float)
        if self.angle.shape != (2,):
            raise ValueError("angle must have shape (2,).")
        self.n = self.as_parametric(n)
        self.reference_wavelength = (
            None
            if reference_wavelength is None
            else np.asarray(reference_wavelength, dtype=float)
        )
        super().__init__(
            transmission=transmission,
            opd=opd,
            phase=phase,
            polarisation=polarisation,
            normalise=normalise,
            propagator=propagator,
        )

    def params(self, wavefront: Wavefront) -> dict[str, Any]:
        """Resolve the optic properties and chromatic wedge OPD once."""
        context = self.context(wavefront)
        n = self.resolve(self.n, **context)
        index_difference = n - 1
        if self.reference_wavelength is not None:
            reference = wavefront.set(wavelength=self.reference_wavelength)
            index_difference = n - self.resolve(self.n, wavefront=reference)

        coordinates = wavefront.coordinates()
        x, y = coordinates[..., 0, :, :], coordinates[..., 1, :, :]
        thickness = x * np.tan(self.angle[0]) + y * np.tan(self.angle[1])
        index_difference = np.asarray(index_difference)
        if index_difference.ndim:
            index_difference = index_difference[..., None, None]
        opd = self.resolve(self.opd, **context)
        opd = 0.0 if opd is None else opd
        return {
            "transmission": self.resolve(self.transmission, **context),
            "opd": opd + index_difference * thickness,
            "phase": self.resolve(self.phase, **context),
            "polarisation": self.polarisation,
        }


class Interpolate(BaseOpticalLayer):
    """Interpolate a wavefront through a coordinate transformation."""

    transformation: BaseCoordTransform
    method: str
    complex: bool
    fill: Array

    def __init__(
        self,
        transformation,
        method="linear",
        complex=True,
        fill=0.0,
    ):
        if not isinstance(transformation, BaseCoordTransform):
            raise TypeError("transformation must be a BaseCoordTransform.")
        self.transformation = transformation
        self.method = str(method)
        self.complex = bool(complex)
        self.fill = np.asarray(fill, dtype=float)

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        return wavefront.interpolate(
            self.transformation,
            method=self.method,
            complex=self.complex,
            fill=self.fill,
        )


class Tilt(BaseOpticalLayer):
    """Tilt a wavefront by two angular coordinates."""

    angles: Array
    unit: str

    def __init__(self, angles, unit="rad"):
        self.angles = np.asarray(angles, dtype=float)
        if self.angles.shape != (2,):
            raise ValueError("angles must have shape (2,).")
        self.unit = str(unit)

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        return wavefront.tilt(self.angles, self.unit)
