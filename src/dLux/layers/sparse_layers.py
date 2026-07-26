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


def _slice_array(value, shape, index, size, name):
    """Select a shared or centre-vectorised array."""
    if value is None or value.shape == shape:
        return value, False
    if value.shape[1:] != shape or value.shape[0] != size:
        raise ValueError(f"{name} must have shape {shape} or ({size},) + {shape}.")
    return value[index], True


def _slice_basis(basis, index, size):
    """Select shared or centre-vectorised basis coefficients."""
    coefficients = basis.coefficients
    if coefficients.shape == basis.basis_shape:
        return basis
    if basis.basis_shape == (1,) and coefficients.ndim == 1:
        if coefficients.shape != (size,):
            raise ValueError(f"coefficients leading axis must match {size} centers.")
        return basis.set(coefficients=coefficients[index, None])
    coefficients, _ = _slice_array(
        coefficients, basis.basis_shape, index, size, "coefficients"
    )
    return basis.set(coefficients=coefficients)


def _slice_transform(transform, index, size):
    """Select shared or centre-vectorised transformation parameters."""
    values = {
        name: _slice_array(value, shape, index, size, name)[0]
        for name, value, shape in _transform_params(transform)
    }
    return transform.set(**values)


def _transform_params(transform):
    """Return transformation array names, values, and native shapes."""
    if isinstance(transform, DistortCoords):
        return (("distortion", transform.distortion, transform.powers.shape),)
    if isinstance(transform, AffineMap):
        return (
            ("matrix", transform.matrix, (2, 2)),
            ("offset", transform.offset, (2,)),
        )
    return tuple(
        (name, getattr(transform, name), shape)
        for name, shape in (
            ("translation", (2,)),
            ("rotation", ()),
            ("scale", (2,)),
            ("shear", (2,)),
        )
    )


def _transform_is_local(transform, size):
    """Whether any transformation parameter has a matching centre axis."""
    if transform is None:
        return False
    types = (DistortCoords, AffineMap, Affine)
    leaves = jtu.leaves(transform, is_leaf=lambda leaf: isinstance(leaf, types))
    return any(
        _slice_array(value, shape, 0, size, name)[1]
        for leaf in leaves
        if isinstance(leaf, types)
        for name, value, shape in _transform_params(leaf)
    )


class Interfere(OpticalLayer):
    """Coherently sum the leading sub-aperture axis of a Wavefront."""

    def __call__(self, wavefront: Wavefront) -> Wavefront:
        size = wavefront.phasor.shape[0]
        collapse = lambda value: (
            value[0]
            if value is not None and value.ndim > 1 and value.shape[0] == size
            else value
        )
        spec = wavefront.spec
        spec = spec.set(d=collapse(spec.d), c=collapse(spec.c))
        return wavefront.set(phasor=wavefront.phasor.sum(0), spec=spec)


class SparseOptic(Optic):
    """Replicate one optic over a set of sub-aperture centers.

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

        def select(leaf):
            if isinstance(leaf, ParametricBasis):
                return _slice_basis(leaf, index, self.n_apertures)
            if isinstance(leaf, (DistortCoords, AffineMap, Affine)):
                return _slice_transform(leaf, index, self.n_apertures)
            return leaf

        return jtu.map(select, self, is_leaf=is_leaf)

    def _context_at(
        self, wavefront: Wavefront, center: Array, optic: SparseOptic
    ) -> dict:
        coordinates = AffineMap(offset=-center)(wavefront.coordinates)
        return {
            "wavefront": wavefront,
            "coordinates": coordinates,
            "pixel_scale": wavefront.pixel_scale,
        }

    def _phasor_at(self, index, center, wavefront):
        optic = self._slice_local(index)
        context = self._context_at(wavefront, center, optic)
        optic = optic.resolve(**context)
        return _optic_phasor(optic, wavefront)

    def phasor(self, wavefront: Wavefront, params: dict = None) -> Array:
        """Return the coherent sum of every centred optic phasor."""
        indices = np.arange(self.n_apertures)
        phasors = vmap(self._phasor_at, in_axes=(0, 0, None))(
            indices, self.centers, wavefront
        )
        return phasors.sum(0)

    def wavefronts(self, wavefront: Wavefront) -> Wavefront:
        """Return one locally sampled Wavefront per aperture center."""
        indices = np.arange(self.n_apertures)

        def make_wavefront(index, center):
            spec = wavefront.spec.set(c=center / wavefront.spec.scale)
            local = wavefront.set(spec=spec)
            phasor = local.phasor * self._phasor_at(index, center, local)
            return phasor

        phasor = vmap(make_wavefront)(indices, self.centers)
        spec = wavefront.spec.set(c=self.centers / wavefront.spec.scale)
        return wavefront.set(phasor=phasor, spec=spec)

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

    def _context_at(self, wavefront, center, optic):
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

        local_transformation = _transform_is_local(
            self.transformation, self.n_apertures
        )
        if optic.transformation is not None and not local_transformation:
            coordinates = optic.transformation(coordinates)
        coordinates = AffineMap(offset=-center)(coordinates)
        if optic.transformation is not None and local_transformation:
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
