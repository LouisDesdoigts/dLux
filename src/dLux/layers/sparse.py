"""Sparse optical layers evaluated over centred sub-apertures."""

from __future__ import annotations

import jax.numpy as np
import jax.tree as jtu
from jax import Array, vmap

import dLux.utils as dlu

from ..grids import Affine, AffineMap, BaseCoordTransform, Distortion, GridSpec
from ..parametric import Parametric, ParametricBasis
from ..fields import Wavefront
from .dynamic import BaseDynamicLayer
from .optical import OpticalLayer, Optic

__all__ = ["Interfere", "SparseOptic", "SparseDynamicOptic"]


class Interfere(OpticalLayer):
    """Coherently sum the leading sub-aperture axis of a Wavefront."""

    def apply(self, wavefront: Wavefront) -> Wavefront:
        """Interfere the complete wavefront without generic leading-axis mapping.

        The final leading batch axis is interpreted as the sub-aperture axis and is
        consumed coherently, so `BaseOpticalLayer.apply` vectorisation is bypassed.
        """
        return self.apply_mono(wavefront)

    def apply_mono(self, wavefront: Wavefront) -> Wavefront:
        """Coherently collapse the final leading sub-aperture dimension.

        Complex phasors are summed along the axis immediately before intrinsic Jones
        and spatial axes. Matching vectorised sampling metadata are collapsed to the
        first sub-aperture value.
        """
        # Identify the sub-aperture axis and size
        axis = wavefront.batch_ndim - 1
        size = wavefront.phasor.shape[axis]

        # Define collapse of matching vectorised grid metadata
        def collapse(value):
            if value is None or value.ndim <= 1:
                return value
            axes = [i for i, n in enumerate(value.shape[:-1]) if n == size]
            return np.take(value, 0, axis=axes[-1]) if axes else value

        # Collapse the field and its realised grid metadata
        d = collapse(wavefront.grid.d)
        c = collapse(wavefront.grid.c)
        return wavefront.set(phasor=wavefront.phasor.sum(axis), d=d, c=c)


class SparseOptic(Optic):
    """Generate locally sampled wavefronts over a set of sub-aperture centers.

    Parameter coefficients are shared when they retain their native shape. A leading
    axis matching the number of centers gives each sub-aperture independent
    coefficients. The same convention applies to polynomial distortion arrays.

    Examples
    --------
    Propagate locally sampled apertures and coherently interfere their fields:

    ```python
    import dLux as dl

    # Construct a sparse optic from one locally sampled aperture
    grid = dl.GridSpec(n=64, diam=0.25, unit="m")
    optic = dl.SparseOptic(
        centers=[[-0.3, 0.0], [0.0, 0.3], [0.3, 0.0]],
        transmission=dl.Circle(diameter=0.2)(grid),
        opd=dl.DynamicZernikeBasis(order=4, diameter=0.2),
        normalise=True,
    )

    # Apply the optic to generate one field per aperture
    wavefront = dl.Wavefront(wavelength=650e-9, grid=grid)
    wavefront = optic(wavefront)

    # Propagate each aperture independently and interfere the fields
    focal_grid = dl.GridSpec(n=64, d=10, unit="mas")
    wavefront = dl.Fraunhofer(focal_grid)(wavefront)
    wavefront = dl.Interfere()(wavefront)
    ```
    """

    transmission: Array | Parametric | None
    opd: Array | Parametric | None
    phase: Array | Parametric | None
    normalise: bool
    centers: Array

    def __init__(
        self, centers, transmission=None, opd=None, phase=None, normalise=False
    ):
        """Initialise repeated locally sampled optics.

        Parameters
        ----------
        centers : ArrayLike
            Physical ``(x, y)`` centres with shape ``(n_apertures, 2)``.
        transmission : Array, Parametric, or None
            Shared or aperture-vectorised local amplitude transmission.
        opd, phase : Array, Parametric, or None
            Shared or aperture-vectorised OPD in metres and phase in radians.
        normalise : bool
            Renormalise the resulting wavefront power.
        """
        centers = dlu.to_value(centers)
        if centers.ndim != 2 or centers.shape[-1] != 2:
            raise ValueError("centers must have shape (n, 2).")
        self.centers = centers
        super().__init__(transmission, opd, phase, normalise)

    @property
    def n_apertures(self) -> int:
        """Return the number of centred sub-apertures."""
        return len(self.centers)

    @staticmethod
    def _slice(obj, index, size):
        """Select one centre from a shared or centre-vectorised object."""
        if isinstance(obj, ParametricBasis):
            params = (("coeffs", obj.shape),)
        elif isinstance(obj, Distortion):
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
                raise ValueError(
                    f"{name} must have shape {shape} or ({size},) + {shape}."
                )
            values[name], local = value[index], True
        return obj.set(**values), local

    def _slice_local(self, index):
        """Select parameters with a leading centre axis for one aperture."""
        types = (ParametricBasis, Distortion, AffineMap, Affine)
        is_leaf = lambda leaf: isinstance(leaf, types)
        local_transform = False

        def select(leaf):
            nonlocal local_transform
            if not isinstance(leaf, types):
                return leaf
            leaf, local = self._slice(leaf, index, self.n_apertures)
            local_transform |= local and isinstance(leaf, BaseCoordTransform)
            return leaf

        return jtu.map(select, self, is_leaf=is_leaf), local_transform

    def _context_at(
        self, wavefront: Wavefront, center: Array, optic: SparseOptic, local=False
    ) -> dict:
        """Build parametric context in one sub-aperture frame."""
        coordinates = AffineMap(offset=-center)(wavefront.coordinates)
        return {
            "wavefront": wavefront,
            "coordinates": coordinates,
            "pixel_scale": wavefront.pixel_scale,
        }

    def _phasor_at(self, index, center, wavefront):
        """Resolve and evaluate one centred local optic phasor."""
        optic, local = self._slice_local(index)
        context = self._context_at(wavefront, center, optic, local)
        optic = optic.resolve(**context)
        return optic._phasor(wavefront)

    def phasor(self, wavefront: Wavefront) -> Array:
        """Return the coherent sum of all locally centred optic phasors.

        The returned complex array broadcasts against ``wavefront.phasor`` and does
        not introduce a sub-aperture axis. Use `localise` to retain that axis.
        """
        indices = np.arange(self.n_apertures)
        phasors = vmap(self._phasor_at, in_axes=(0, 0, None))(
            indices, self.centers, wavefront
        )
        return phasors.sum(0)

    def localise(self, wavefront: Wavefront) -> Wavefront:
        """Evaluate one locally centred field per sub-aperture.

        The returned wavefront inserts the aperture axis after existing batch axes
        and stores per-aperture grid centres in the grid's declared unit.
        """
        indices = np.arange(self.n_apertures)

        def make_wavefront(index, center):
            local = wavefront.set(c=center / wavefront.grid.scale)
            phasor = local.phasor * self._phasor_at(index, center, local)
            return phasor

        phasor = vmap(make_wavefront)(indices, self.centers)
        phasor = np.moveaxis(phasor, 0, wavefront.batch_ndim)
        c = self.centers / wavefront.grid.scale
        return wavefront.set(phasor=phasor, c=c)

    def apply_mono(self, wavefront: Wavefront) -> Wavefront:
        """Apply the optic and append its sub-aperture axis to the wavefront.

        Optional normalisation is applied after localisation. The input wavefront is
        not mutated.
        """
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
        """Initialise repeated coordinate-dependent local optics.

        Parameters
        ----------
        centers : ArrayLike
            Physical ``(x, y)`` centres with shape ``(n_apertures, 2)``.
        transmission : Array, Parametric, or None
            Shared or aperture-vectorised local amplitude transmission.
        opd, phase : Array, Parametric, or None
            Shared or aperture-vectorised OPD in metres and phase in radians.
        coordinates : Array, GridSpec, or None
            Explicit coordinate source, or incident coordinates when omitted.
        transformation : BaseCoordTransform or None
            Shared global or aperture-vectorised local coordinate transformation.
        normalise : bool
            Renormalise the resulting wavefront power.
        """
        BaseDynamicLayer.__init__(self, coordinates, transformation)
        SparseOptic.__init__(self, centers, transmission, opd, phase, normalise)

    def _context_at(self, wavefront, center, optic, local=False):
        """Build dynamic context in a global or local transformed frame."""
        # Resolve the coordinate source and its physical sampling
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

        # Apply global transformation before aperture placement
        if optic.transformation is not None and not local:
            coordinates = optic.transformation(coordinates)
        coordinates = AffineMap(offset=-center)(coordinates)

        # Apply aperture-dependent transformation in the local frame
        if optic.transformation is not None and local:
            coordinates = optic.transformation(coordinates)
        return {
            "wavefront": wavefront,
            "coordinates": coordinates,
            "pixel_scale": pixel_scale,
        }

    coordinates: Array | GridSpec | None
    transformation: BaseCoordTransform | None
    transmission: Array | Parametric | None
    opd: Array | Parametric | None
    phase: Array | Parametric | None
    normalise: bool
    centers: Array
