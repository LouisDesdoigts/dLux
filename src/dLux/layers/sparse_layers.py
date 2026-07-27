"""Sparse optical layers evaluated over centred sub-apertures."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as np
import jax.tree as jtu
from jax import Array, vmap

from ..grids import Affine, AffineMap, CoordTransform, DistortCoords, GridSpec
from ..parametric import Parametric, ParametricBasis, to_param
from ..fields import Wavefront
from .dynamic_layers import BaseDynamicLayer
from .optical_layers import OpticalLayer, Optic, _optic_phasor

__all__ = ["Interfere", "SparseOptic", "SparseDynamicOptic"]


def _slice(obj, index, size):
    """Select one centre from a shared or centre-vectorised object."""
    if isinstance(obj, ParametricBasis):
        params = (("coefficients", obj.basis_shape),)
    elif isinstance(obj, DistortCoords):
        params = (("distortion", obj.powers.shape),)
    elif isinstance(obj, AffineMap):
        params = (("matrix", (2, 2)), ("offset", (2,)))
    else:
        params = (
            ("translation", (2,)),
            ("rotation", ()),
            ("scale", (2,)),
            ("shear", (2,)),
        )

    values, local = {}, False
    for name, shape in params:
        value = getattr(obj, name)
        if value is None or value.shape == shape:
            values[name] = value
            continue
        if (
            isinstance(obj, ParametricBasis)
            and shape == (1,)
            and value.shape == (size,)
        ):
            values[name], local = value[index, None], True
            continue
        if value.shape[1:] != shape or value.shape[0] != size:
            raise ValueError(f"{name} must have shape {shape} or ({size},) + {shape}.")
        values[name], local = value[index], True
    return obj.set(**values), local


class Interfere(OpticalLayer):
    """Coherently sum the leading sub-aperture axis of a Wavefront."""

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        axis = wavefront.batch_ndim - 1
        size = wavefront.phasor.shape[axis]

        def collapse(value):
            if value is None or value.ndim <= 1:
                return value
            axes = [i for i, n in enumerate(value.shape[:-1]) if n == size]
            return np.take(value, 0, axis=axes[-1]) if axes else value

        spec = wavefront.spec
        spec = spec.set(d=collapse(spec.d), c=collapse(spec.c))
        return wavefront.set(phasor=wavefront.phasor.sum(axis), spec=spec)


class SparseOptic(Optic):
    """Generate locally sampled wavefronts over a set of sub-aperture centers.

    Parameter coefficients are shared when they retain their native shape. A leading
    axis matching the number of centers gives each sub-aperture independent
    coefficients. The same convention applies to polynomial distortion arrays.
    """

    transmission: Array | Parametric | None = eqx.field(converter=to_param)
    opd: Array | Parametric | None = eqx.field(converter=to_param)
    phase: Array | Parametric | None = eqx.field(converter=to_param)
    normalise: bool
    centers: Array

    def __init__(
        self, centers, transmission=None, opd=None, phase=None, normalise=False
    ):
        centers = np.asarray(centers, dtype=float)
        if centers.ndim != 2 or centers.shape[-1] != 2:
            raise ValueError("centers must have shape (n, 2).")
        self.centers = centers
        super().__init__(transmission, opd, phase, normalise)

    @property
    def n_apertures(self) -> int:
        """Return the number of centred sub-apertures."""
        return len(self.centers)

    def _slice_local(self, index):
        """Select parameters with a leading center axis for one aperture."""
        types = (ParametricBasis, DistortCoords, AffineMap, Affine)
        is_leaf = lambda leaf: isinstance(leaf, types)
        local_transform = False

        def select(leaf):
            nonlocal local_transform
            if not isinstance(leaf, types):
                return leaf
            leaf, local = _slice(leaf, index, self.n_apertures)
            local_transform |= local and isinstance(leaf, CoordTransform)
            return leaf

        return jtu.map(select, self, is_leaf=is_leaf), local_transform

    def _context_at(
        self, wavefront: Wavefront, center: Array, optic: SparseOptic, local=False
    ) -> dict:
        coordinates = AffineMap(offset=-center)(wavefront.coordinates)
        return {
            "wavefront": wavefront,
            "coordinates": coordinates,
            "pixel_scale": wavefront.pixel_scale,
        }

    def _phasor_at(self, index, center, wavefront):
        optic, local = self._slice_local(index)
        context = self._context_at(wavefront, center, optic, local)
        optic = optic.resolve(**context)
        return _optic_phasor(optic, wavefront)

    def phasor(self, wavefront: Wavefront, params: dict = None) -> Array:
        """Return the coherent sum of every centred optic phasor."""
        indices = np.arange(self.n_apertures)
        phasors = vmap(self._phasor_at, in_axes=(0, 0, None))(
            indices, self.centers, wavefront
        )
        return phasors.sum(0)

    def localise(self, wavefront: Wavefront) -> Wavefront:
        """Evaluate the optic on one locally centred field per sub-aperture."""
        indices = np.arange(self.n_apertures)

        def make_wavefront(index, center):
            spec = wavefront.spec.set(c=center / wavefront.spec.scale)
            local = wavefront.set(spec=spec)
            phasor = local.phasor * self._phasor_at(index, center, local)
            return phasor

        phasor = vmap(make_wavefront)(indices, self.centers)
        phasor = np.moveaxis(phasor, 0, wavefront.batch_ndim)
        spec = wavefront.spec.set(c=self.centers / wavefront.spec.scale)
        return wavefront.set(phasor=phasor, spec=spec)

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        """Apply the optic and append its sub-aperture axis to the wavefront."""
        wavefront = self.localise(wavefront)
        return wavefront.normalise() if self.normalise else wavefront


class SparseDynamicOptic(BaseDynamicLayer, SparseOptic):
    """A sparse optic with an optional coordinate source and transformation.

    A shared transformation acts in the global coordinate frame before aperture
    placement. A transformation with a leading aperture axis acts independently in
    each aperture's local frame.
    """

    def __init__(
        self,
        centers,
        transmission=None,
        opd=None,
        phase=None,
        coordinates=None,
        transformation=None,
        normalise=False,
    ):
        BaseDynamicLayer.__init__(self, coordinates, transformation)
        SparseOptic.__init__(self, centers, transmission, opd, phase, normalise)

    def _context_at(self, wavefront, center, optic, local=False):
        coordinate_source = optic.coordinates
        if coordinate_source is None:
            coordinates = wavefront.coordinates
            pixel_scale = wavefront.pixel_scale
        elif isinstance(coordinate_source, GridSpec):
            coordinates = coordinate_source.coordinates
            pixel_scale = coordinate_source.d
        else:
            coordinates = coordinate_source
            pixel_scale = wavefront.pixel_scale

        if optic.transformation is not None and not local:
            coordinates = optic.transformation(coordinates)
        coordinates = AffineMap(offset=-center)(coordinates)
        if optic.transformation is not None and local:
            coordinates = optic.transformation(coordinates)
        return {
            "wavefront": wavefront,
            "coordinates": coordinates,
            "pixel_scale": pixel_scale,
        }

    coordinates: Array | GridSpec | None
    transformation: CoordTransform | None
    transmission: Array | Parametric | None = eqx.field(converter=to_param)
    opd: Array | Parametric | None = eqx.field(converter=to_param)
    phase: Array | Parametric | None = eqx.field(converter=to_param)
    normalise: bool
    centers: Array
