"""Minimal prototype for coordinate-driven, parametric optics.

This file is intentionally separate from the public dLux API. It explores a design
where ``Optic`` is the standard field-applying layer, analytic apertures are small
parametric geometry objects, and ``DynamicOptic`` evaluates all of its fields in one
shared transformed coordinate frame.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as np
from jax import Array, vmap

import dLux.utils as dlu
from dLux.coordinates import BaseCoordTransform, DistortedCoords
from dLux.layers.optical_layers import OpticalLayer
from dLux.layers.polarised_layers import (
    BasePolarisingOptic,
    LinearPolariser,
    Retarder,
)
from dLux.parametric import Parametric
from dLux.wavefronts import Wavefront


class Optic(OpticalLayer):
    """Collapse local scalar effects into one phasor, apply polarisation, propagate."""

    transmission: Array | Parametric | None
    opd: Array | Parametric | None
    phase: Array | Parametric | None
    lens: Parametric | None
    polarisation: dict | None
    normalise: bool
    propagator: OpticalLayer | None

    def __init__(
        self,
        transmission=None,
        opd=None,
        phase=None,
        lens=None,
        polarisation=None,
        normalise=False,
        propagator=None,
    ):
        self.transmission = self.as_parametric(transmission)
        self.opd = self.as_parametric(opd)
        self.phase = self.as_parametric(phase)
        if lens is not None and not isinstance(lens, Parametric):
            raise TypeError("lens must be a Parametric object or None.")
        if polarisation is not None:
            polarisation = dlu.list2dictionary(
                (
                    list(polarisation)
                    if isinstance(polarisation, (list, tuple))
                    else [polarisation]
                ),
                True,
                BasePolarisingOptic,
            )
        self.lens = lens
        self.polarisation = polarisation
        self.normalise = bool(normalise)
        if propagator is not None and not isinstance(propagator, OpticalLayer):
            raise TypeError("propagator must be an OpticalLayer or None.")
        self.propagator = propagator

    def context(self, wavefront: Wavefront) -> dict[str, Any]:
        """Return the evaluation context shared by all optic leaves."""
        return {"wavefront": wavefront}

    def params(self, wavefront: Wavefront) -> dict[str, Any]:
        """Evaluate the physical parameters consumed by ``__call__``."""
        context = self.context(wavefront)
        return {
            "transmission": self.resolve(self.transmission, **context),
            "opd": self.resolve(self.opd, **context),
            "phase": self.resolve(self.phase, **context),
            "lens_opd": self.resolve(self.lens, **context),
            "polarisation": self.polarisation,
        }

    def phasor(self, wavefront: Wavefront, params: dict = None) -> Array:
        """Return the cumulative complex field applied at this optical plane.

        The wavefront supplies the coordinates and wavelengths required by dynamic
        and chromatic parameters. ``params`` lets ``__call__`` avoid evaluating those
        parameters twice.
        """
        if params is None:
            params = self.params(wavefront)
        transmission = 1.0 if params["transmission"] is None else params["transmission"]
        opd = 0.0 if params["opd"] is None else params["opd"]
        phase = 0.0 if params["phase"] is None else params["phase"]
        lens_opd = 0.0 if params["lens_opd"] is None else params["lens_opd"]

        wavenumber = wavefront._to_phasor_shape(wavefront.wavenumber)
        total_opd = wavefront._to_phasor_shape(opd + lens_opd)
        phase = wavefront._to_phasor_shape(phase)
        transmission = wavefront._to_phasor_shape(transmission)
        return transmission * np.exp(1j * (wavenumber * total_opd + phase))

    @staticmethod
    def polarisation_matrix(polarisation, wavefront: Wavefront) -> Array | None:
        """Evaluate and compose existing polarisation layers in listed order."""
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
        wavefront *= self.phasor(wavefront, params)
        polarisation = self.polarisation_matrix(params["polarisation"], wavefront)
        if polarisation is not None:
            wavefront = wavefront.apply_jones(polarisation)
        if self.normalise:
            wavefront = wavefront.normalise()
        if self.propagator is not None:
            wavefront = self.propagator(wavefront)
        return wavefront


class Lens(Parametric):
    """A refractive lens leaf that evaluates only its physical OPD contribution."""

    thickness: Array | Parametric
    n: Array | Parametric

    def __init__(self, thickness, n):
        self.thickness = (
            thickness if isinstance(thickness, Parametric) else np.asarray(thickness)
        )
        self.n = n if isinstance(n, Parametric) else np.asarray(n)

    def evaluate(self, *, wavefront, **context) -> Array:
        thickness = (
            self.thickness.evaluate(wavefront=wavefront, **context)
            if isinstance(self.thickness, Parametric)
            else self.thickness
        )
        n = (
            self.n.evaluate(wavefront=wavefront, **context)
            if isinstance(self.n, Parametric)
            else self.n
        )
        index_difference = np.asarray(n - 1)
        if index_difference.ndim:
            index_difference = index_difference[..., None, None]
        return index_difference * thickness


class TransformChain(BaseCoordTransform):
    """Apply coordinate transformations in the explicitly supplied order."""

    transformations: dict

    def __init__(self, transformations=()):
        self.transformations = dlu.list2dictionary(
            list(transformations), True, BaseCoordTransform
        )

    def __call__(self, coordinates: Array) -> Array:
        for transformation in self.transformations.values():
            coordinates = transformation(coordinates)
        return coordinates


class Translate(BaseCoordTransform):
    """Map global coordinates into a translated local coordinate frame."""

    offset: Array

    def __init__(self, offset):
        self.offset = np.asarray(offset, dtype=float)
        if self.offset.shape != (2,):
            raise ValueError("offset must have shape (2,).")

    def __call__(self, coordinates: Array) -> Array:
        return dlu.translate_coords(coordinates, self.offset)


class Rotate(BaseCoordTransform):
    """Map coordinates into a local frame rotated by ``angle`` radians."""

    angle: Array

    def __init__(self, angle):
        self.angle = np.asarray(angle, dtype=float)
        if self.angle.shape != ():
            raise ValueError("angle must be scalar.")

    def __call__(self, coordinates: Array) -> Array:
        return dlu.rotate_coords(coordinates, self.angle)


class Scale(BaseCoordTransform):
    """Actively scale an object by mapping coordinates into its local frame."""

    scale: Array

    def __init__(self, scale):
        scale = np.broadcast_to(np.asarray(scale, dtype=float), (2,))
        if np.any(scale == 0):
            raise ValueError("scale values must be non-zero.")
        self.scale = scale

    def __call__(self, coordinates: Array) -> Array:
        return coordinates / self.scale[:, None, None]


class Shear(BaseCoordTransform):
    """Apply the existing dLux coordinate-shear convention."""

    shear: Array

    def __init__(self, shear):
        self.shear = np.asarray(shear, dtype=float)
        if self.shear.shape != (2,):
            raise ValueError("shear must have shape (2,).")

    def __call__(self, coordinates: Array) -> Array:
        return dlu.shear_coords(coordinates, self.shear)


class Distort(BaseCoordTransform):
    """Apply a polynomial coordinate distortion."""

    distortion: DistortedCoords

    def __init__(self, order=1, coefficients=None):
        self.distortion = DistortedCoords(order, coefficients)

    @property
    def coefficients(self) -> Array:
        return self.distortion.distortion

    def __call__(self, coordinates: Array) -> Array:
        return self.distortion(coordinates)


class Shape(Parametric):
    """A geometry that evaluates to a transmission on supplied coordinates."""

    @property
    def extent(self) -> Array | None:
        """Return a finite bounding radius, or ``None`` when one is undefined."""
        return None


class SoftShape(Shape):
    """Base geometry with a common softened-edge width."""

    softening: Array

    def __init__(self, softening=1.0):
        self.softening = np.asarray(softening, dtype=float)

    def clip(self, pixel_scale) -> Array:
        return pixel_scale * self.softening / 2


class RadialShape(SoftShape):
    """Base softened geometry parameterised by a bounding radius."""

    radius: Array

    def __init__(self, radius, softening=1.0):
        super().__init__(softening)
        self.radius = np.asarray(radius, dtype=float)

    @property
    def extent(self) -> Array:
        return self.radius


class Circle(RadialShape):
    def evaluate(self, *, coordinates, pixel_scale, **kwargs) -> Array:
        return dlu.soft_circle(
            coordinates,
            self.radius,
            self.clip(pixel_scale),
        )


class RegularPolygon(RadialShape):
    nsides: int

    def __init__(self, nsides, radius, softening=1.0):
        super().__init__(radius, softening)
        self.nsides = int(nsides)

    def evaluate(self, *, coordinates, pixel_scale, **kwargs) -> Array:
        return dlu.soft_reg_polygon(
            coordinates,
            self.radius,
            self.nsides,
            self.clip(pixel_scale),
        )


class Spider(SoftShape):
    """A general set of occulting radial support arms with angles in degrees."""

    width: Array
    angles: Array

    def __init__(self, width, angles, softening=1.0):
        super().__init__(softening)
        self.width = np.asarray(width, dtype=float)
        self.angles = np.asarray(angles, dtype=float)

    def evaluate(self, *, coordinates, pixel_scale, **kwargs) -> Array:
        return dlu.soft_spider(
            coordinates,
            self.width,
            self.angles,
            self.clip(pixel_scale),
            invert=True,
        )


class Complement(Shape):
    """Invert any shape transmission without coupling inversion to edge softness."""

    shape: Shape

    def __init__(self, shape):
        if not isinstance(shape, Shape):
            raise TypeError("shape must be a Shape.")
        self.shape = shape

    @property
    def extent(self) -> Array | None:
        return self.shape.extent

    def evaluate(self, **context) -> Array:
        return 1 - self.shape.evaluate(**context)


class Intersection(Shape):
    """The product of several aperture transmissions."""

    shapes: dict

    def __init__(self, shapes):
        self.shapes = dlu.list2dictionary(list(shapes), True, Shape)

    @property
    def extent(self) -> Array:
        extents = [
            shape.extent for shape in self.shapes.values() if shape.extent is not None
        ]
        return None if not extents else np.max(np.array(extents))

    def evaluate(self, **context) -> Array:
        return np.prod(
            np.array([shape.evaluate(**context) for shape in self.shapes.values()]), 0
        )


class ApertureArray(Shape):
    """Vectorised copies of one shape at a set of aperture centres."""

    shape: Shape
    positions: Array

    def __init__(self, shape, positions):
        if not isinstance(shape, Shape):
            raise TypeError("shape must be a Shape.")
        self.shape = shape
        self.positions = np.asarray(positions, dtype=float)
        if self.positions.ndim != 2 or self.positions.shape[-1] != 2:
            raise ValueError("positions must have shape (n_apertures, 2).")

    @property
    def extent(self) -> Array:
        if self.shape.extent is None:
            raise TypeError("ApertureArray shapes must have a finite extent.")
        return np.max(np.linalg.norm(self.positions, axis=-1)) + self.shape.extent

    def local_coordinates(self, coordinates: Array) -> Array:
        return vmap(lambda position: Translate(position)(coordinates))(self.positions)

    def evaluate(self, *, coordinates, pixel_scale, **kwargs) -> Array:
        local_coordinates = self.local_coordinates(coordinates)
        evaluate = lambda coords: self.shape.evaluate(
            coordinates=coords,
            pixel_scale=pixel_scale,
            **kwargs,
        )
        transmissions = vmap(evaluate)(local_coordinates)
        return np.clip(transmissions.sum(0), 0.0, 1.0)


class DynamicOptic(Optic):
    """Evaluate aperture and aberrations in one transformed coordinate frame."""

    aperture: Shape
    transformation: BaseCoordTransform | None

    def __init__(
        self,
        aperture,
        transformation=None,
        opd=None,
        phase=None,
        lens=None,
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
            transmission=None,
            opd=opd,
            phase=phase,
            lens=lens,
            polarisation=polarisation,
            normalise=normalise,
            propagator=propagator,
        )

    def context(self, wavefront: Wavefront) -> dict[str, Any]:
        """Return the transformed coordinate context shared by every leaf."""
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
        """Use the aperture as transmission and resolve the remaining parameters."""
        context = self.context(wavefront)
        return {
            "transmission": self.aperture.evaluate(**context),
            "opd": self.resolve(self.opd, **context),
            "phase": self.resolve(self.phase, **context),
            "lens_opd": self.resolve(self.lens, **context),
            "polarisation": self.polarisation,
        }


if __name__ == "__main__":
    from dLux.layers import FFT
    from dLux.polynomials import DynamicZernikeBasis

    wavefront = Wavefront(1e-6, npixels=64, diameter=2.0)
    aperture = Intersection(
        [
            (
                "segments",
                ApertureArray(
                    Circle(0.2),
                    positions=np.array([[-0.4, 0.0], [0.0, 0.0], [0.4, 0.0]]),
                ),
            ),
            ("spiders", Spider(0.02, angles=np.array([0.0, 90.0]))),
        ]
    )
    optic = DynamicOptic(
        aperture=aperture,
        transformation=TransformChain([Rotate(0.1), Scale([1.0, 0.95]), Distort()]),
        opd=DynamicZernikeBasis(js=[1, 2, 3], coefficients=np.zeros(3)),
        lens=Lens(thickness=1e-6, n=1.5),
        polarisation=[Retarder(np.pi / 2, np.pi / 4), LinearPolariser(0.0)],
        normalise=True,
        propagator=FFT(pad=1),
    )
    local_phasor = optic.phasor(wavefront)
    output = optic(wavefront)

    assert local_phasor.shape == (64, 64)
    assert np.iscomplexobj(local_phasor)
    assert output.phasor.shape == (2, 2, 64, 64)
    assert np.isclose(output.power, 1.0)
