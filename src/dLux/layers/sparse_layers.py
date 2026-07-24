"""Sparse optical layers evaluated over positioned sub-apertures."""

from __future__ import annotations

import jax.numpy as np
import jax.tree as jtu
from jax import Array, vmap

from ..coord_specs import CoordSpec
from ..coordinates import AffineMap, DistortedCoords
from ..parametric import ParametricBasis
from ..wavefronts import Wavefront
from .dynamic_layers import BaseDynamicLayer
from .optical_layers import Optic

__all__ = ["SparseOptic", "SparseDynamicOptic"]


class SparseOptic(Optic):
    """Replicate one optic over a set of sub-aperture positions.

    Parameter coefficients are shared when they retain their native shape. A leading
    axis matching the number of positions gives each sub-aperture independent
    coefficients. The same convention applies to polynomial distortion arrays.
    """

    positions: Array

    def __init__(
        self,
        positions,
        transmission=None,
        opd=None,
        phase=None,
        normalise=False,
    ):
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[-1] != 2:
            raise ValueError("positions must have shape (n, 2).")
        self.positions = positions
        super().__init__(transmission, opd, phase, normalise)

    @property
    def n_apertures(self) -> int:
        """Return the number of positioned sub-apertures."""
        return len(self.positions)

    def _slice_local(self, index):
        """Select vectorised coefficient and distortion leaves for one aperture."""

        def select(leaf):
            if isinstance(leaf, ParametricBasis):
                shape = (self.n_apertures,) + leaf.basis_shape
                if leaf.coefficients.shape == shape:
                    return leaf.set(coefficients=leaf.coefficients[index])
            if isinstance(leaf, DistortedCoords):
                if (
                    leaf.distortion.ndim == 3
                    and leaf.distortion.shape[0] == self.n_apertures
                ):
                    return leaf.set(distortion=leaf.distortion[index])
            return leaf

        is_leaf = lambda leaf: isinstance(leaf, (ParametricBasis, DistortedCoords))
        return jtu.map(select, self, is_leaf=is_leaf)

    def _context_at(
        self,
        wavefront: Wavefront,
        position: Array,
        optic: SparseOptic,
    ) -> dict:
        coordinates = AffineMap(offset=-position)(wavefront.coordinates())
        return {
            "wavefront": wavefront,
            "coordinates": coordinates,
            "pixel_scale": wavefront.pixel_scale,
        }

    def _phasor_at(self, index, position, wavefront):
        optic = self._slice_local(index)
        context = self._context_at(wavefront, position, optic)
        params = {
            "transmission": optic.resolve(optic.transmission, **context),
            "opd": optic.resolve(optic.opd, **context),
            "phase": optic.resolve(optic.phase, **context),
        }
        return Optic.phasor(optic, wavefront, params)

    def phasor(self, wavefront: Wavefront, params: dict = None) -> Array:
        """Return the coherent sum of every positioned optic phasor."""
        indices = np.arange(self.n_apertures)
        phasors = vmap(self._phasor_at, in_axes=(0, 0, None))(
            indices, self.positions, wavefront
        )
        return phasors.sum(0)

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        phasor = wavefront.phasor * self.phasor(wavefront)
        wavefront = wavefront.set(phasor=phasor)
        return wavefront.normalise() if self.normalise else wavefront


class SparseDynamicOptic(BaseDynamicLayer, SparseOptic):
    """A sparse optic with an optional coordinate source and transformation.

    A shared transformation acts in the global coordinate frame before aperture
    placement. A transformation with a leading aperture axis acts independently in
    each aperture's local frame.
    """

    def __init__(
        self,
        positions,
        transmission=None,
        opd=None,
        phase=None,
        coordinates=None,
        transformation=None,
        normalise=False,
    ):
        BaseDynamicLayer.__init__(self, coordinates, transformation)
        SparseOptic.__init__(
            self,
            positions,
            transmission,
            opd,
            phase,
            normalise,
        )

    def _context_at(self, wavefront, position, optic):
        coordinate_source = optic.coordinates
        if coordinate_source is None:
            coordinates = wavefront.coordinates()
            pixel_scale = wavefront.pixel_scale
        elif isinstance(coordinate_source, CoordSpec):
            coordinates = optic._from_spec(coordinate_source)
            pixel_scale = coordinate_source.d
        else:
            coordinates = coordinate_source
            pixel_scale = wavefront.pixel_scale

        local_transformation = (
            isinstance(self.transformation, DistortedCoords)
            and self.transformation.distortion.ndim == 3
            and self.transformation.distortion.shape[0] == self.n_apertures
        )
        if optic.transformation is not None and not local_transformation:
            coordinates = optic.transformation(coordinates)
        coordinates = AffineMap(offset=-position)(coordinates)
        if optic.transformation is not None and local_transformation:
            coordinates = optic.transformation(coordinates)
        return {
            "wavefront": wavefront,
            "coordinates": coordinates,
            "pixel_scale": pixel_scale,
        }
