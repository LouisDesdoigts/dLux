"""Sparse optical layers evaluated over centred sub-apertures."""

from __future__ import annotations

import jax.numpy as np
import jax.tree as jtu
import zodiax as zdx
from jax import Array, vmap

from ..utils.grids import Affine, AffineMap, BaseCoordTransform, Distortion, GridSpec
from ..parametric.bases import Basis, ImplicitBasis, FourierBasis
from ..parametric.polynomials import Polynomial
from ..fields import Wavefront
from .dynamic import BaseDynamicLayer
from .optical import BaseOpticalLayer, OpticalLayer, Optic

__all__ = ["Interfere", "SparseOptic", "SparseDynamicOptic"]


class Interfere(OpticalLayer):
    """Coherently sum the leading sub-aperture axis of a `Wavefront`.

    Complex phasors are added before intensity is formed, retaining interference
    between apertures. Inputs must have compatible wavelength and sampling metadata.
    """

    def apply(self, wavefront: Wavefront, **context) -> Wavefront:
        """Interfere the complete wavefront without generic leading-axis mapping.

        The final leading batch axis is interpreted as the sub-aperture axis and is
        consumed coherently, so `BaseOpticalLayer.apply` vectorisation is bypassed.
        """
        return self.apply_mono(wavefront, **context)

    def apply_mono(self, wavefront: Wavefront, **context) -> Wavefront:
        """Coherently collapse the final leading sub-aperture dimension.

        Complex phasors are summed along the axis immediately before intrinsic Jones
        and spatial axes. Sampling carrying that axis, including a shared singleton,
        is collapsed to its first sub-aperture value.
        """
        # Identify the sub-aperture axis and size
        axis = wavefront.batch_ndim - 1
        size = wavefront.phasor.shape[axis]
        batch = wavefront.phasor.shape[wavefront.wavelength.ndim : axis + 1]

        # Define collapse of matching vectorised grid metadata
        def collapse(value):
            if value is None or value.ndim <= 1:
                return value
            prefix = value.shape[:-1]
            if batch and prefix == batch:
                return np.take(value, 0, axis=len(prefix) - 1)
            if len(prefix) == wavefront.batch_ndim and prefix[axis] in (1, size):
                return np.take(value, 0, axis=axis)
            return value

        # Collapse the field and its realised grid metadata
        d = collapse(wavefront.grid.d)
        c = collapse(wavefront.grid.c)
        return wavefront.set(phasor=wavefront.phasor.sum(axis), d=d, c=c)


class SparseOptic(Optic):
    """Generate locally sampled wavefronts over a set of sub-aperture centers.

    Basis, ImplicitBasis (including DynamicZernikeBasis), FourierBasis and Polynomial
    coefficients are shared in their native coefficient shape. One leading axis
    matching the number of centres selects independent coefficients per aperture.
    Other arrays and expressions are shared; their axes are not inferred as aperture
    axes. Coordinate transforms support the same explicit leading aperture axis.

    Local Jones terms are unsupported and raise an error when configured. Apply a
    separate polarising layer before or after the sparse optic. Scalar modulation
    preserves an already polarised input wavefront.

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

    amplitude: Array | zdx.Expression | None = None
    jones: Array | zdx.Expression | None = None
    thickness: Array | zdx.Expression | None = None
    n: Array | zdx.Expression | None = None
    propagator: BaseOpticalLayer | None = None
    transmission: Array | zdx.Expression | None = None
    opd: Array | zdx.Expression | None = None
    phase: Array | zdx.Expression | None = None
    normalise: bool | None = None
    centers: Array

    def __init__(
        self,
        centers,
        transmission=None,
        opd=None,
        phase=None,
        normalise=None,
        *,
        alias=None,
    ):
        """Initialise repeated locally sampled optics.

        Parameters
        ----------
        centers : ArrayLike
            Physical ``(x, y)`` centres with shape ``(n_apertures, 2)``.
        transmission : Array, Parametric, or None
            Shared real amplitude transmission, or a basis with per-aperture
            coefficients.
        opd, phase : Array, Parametric, or None
            Shared OPD in metres and phase in radians, or bases with per-aperture
            coefficients.
        normalise : bool or None
            Renormalise the resulting wavefront power when true. Omit to leave
            normalisation unconfigured.
        """
        centers = zdx.as_array(centers, dtype=float)
        if centers.ndim != 2 or centers.shape[-1] != 2:
            raise ValueError("centers must have shape (n, 2).")
        self.centers = centers
        super().__init__(transmission, opd, phase, normalise)

        self.alias = alias

    @property
    def n_apertures(self) -> int:
        """Return the number of centred sub-apertures."""
        return len(self.centers)

    @staticmethod
    def _slice(obj, index, size, **context):
        """Select one centre from a shared or centre-vectorised object."""
        if isinstance(obj, (Basis, ImplicitBasis, FourierBasis, Polynomial)):
            # Select coefficients without generating modes in the wrong frame.
            if isinstance(obj, Basis):
                axes = obj.axes
            elif isinstance(obj, Polynomial):
                axes = 1
            else:
                axes = len(obj.shape)
            coeffs = zdx.resolve(obj.coeffs, **context)
            if coeffs is None:
                return obj, False

            # Retain compact per-aperture coefficients for a single implicit mode.
            if isinstance(obj, ImplicitBasis) and obj.shape == (1,):
                if coeffs.shape == (size,) and size != 1:
                    return obj.set(coeffs=coeffs[index, None]), True

            if coeffs.ndim == axes:
                return obj.set(coeffs=coeffs), False
            if coeffs.ndim == axes + 1 and coeffs.shape[0] == size:
                return obj.set(coeffs=coeffs[index]), True
            raise ValueError(
                f"coeffs must have {axes} coefficient axes, optionally preceded "
                f"by an aperture axis of size {size}."
            )

        if isinstance(obj, Distortion):
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
            if isinstance(value, zdx.Expression):
                value = zdx.resolve(value, **context)
            if value is None or value.shape == shape:
                values[name] = value
                continue
            if value.shape[1:] != shape or value.shape[0] != size:
                raise ValueError(
                    f"{name} must have shape {shape} or ({size},) + {shape}."
                )
            values[name], local = value[index], True
        return obj.set(**values), local

    def _slice_local(self, index, *, bases, **context):
        """Select either basis coefficients or coordinate transforms per aperture."""
        basis_types = (Basis, ImplicitBasis, FourierBasis, Polynomial)
        types = basis_types + (Distortion, AffineMap, Affine)
        is_leaf = lambda leaf: isinstance(leaf, types)
        local_transform = False

        def select(leaf):
            nonlocal local_transform
            if not isinstance(leaf, types):
                return leaf
            if isinstance(leaf, basis_types) != bases:
                return leaf
            leaf, local = self._slice(leaf, index, self.n_apertures, **context)
            local_transform |= local and isinstance(leaf, BaseCoordTransform)
            return leaf

        return jtu.map(select, self, is_leaf=is_leaf), local_transform

    def _context_at(
        self,
        wavefront: Wavefront,
        center: Array,
        optic: SparseOptic,
        local=False,
        **context,
    ) -> dict:
        """Build parametric context in one sub-aperture frame."""
        coordinates = AffineMap(offset=-center)(wavefront.coordinates)
        return {
            **context,
            "wavelengths": wavefront.wavelength,
            "coordinates": coordinates,
            "pixel_scale": wavefront.pixel_scale,
        }

    def _phasor_at(self, index, center, wavefront, **context):
        """Resolve and evaluate one centred local optic phasor."""
        if self.jones is not None:
            raise ValueError(
                "Sparse optics do not support local Jones terms; "
                "apply a separate polarising layer."
            )

        # Select geometry before constructing the authoritative local context.
        optic, local = self._slice_local(index, bases=False, **context)
        context = self._context_at(wavefront, center, optic, local, **context)

        # Resolve and select coefficients only after their coordinates exist.
        optic, _ = optic._slice_local(index, bases=True, **context)
        optic = optic.resolve(**context)
        return optic._phasor(wavefront)

    def phasor(self, wavefront: Wavefront, **context) -> Array:
        """Return the coherent sum of all locally centred optic phasors.

        The returned complex array broadcasts against ``wavefront.phasor`` and does
        not introduce a sub-aperture axis. Use `localise` to retain that axis.
        """
        indices = np.arange(self.n_apertures)
        phasors = vmap(lambda i, c: self._phasor_at(i, c, wavefront, **context))(
            indices, self.centers
        )
        return phasors.sum(0)

    def localise(self, wavefront: Wavefront, **context) -> Wavefront:
        """Evaluate one locally centred field per sub-aperture.

        The returned wavefront inserts the aperture axis after existing batch axes
        and stores per-aperture grid centres in the grid's declared unit.
        """
        indices = np.arange(self.n_apertures)

        def make_wavefront(index, center):
            local = wavefront.set(c=center / wavefront.grid.scale)
            phasor = local.phasor * self._phasor_at(index, center, local, **context)
            return phasor

        phasor = vmap(make_wavefront)(indices, self.centers)
        phasor = np.moveaxis(phasor, 0, wavefront.batch_ndim)
        c = self.centers / wavefront.grid.scale
        return wavefront.set(phasor=phasor, c=c)

    def apply_mono(self, wavefront: Wavefront, **context) -> Wavefront:
        """Apply the optic and append its sub-aperture axis to the wavefront.

        Optional normalisation is applied after localisation. The input wavefront is
        not mutated.
        """
        wavefront = self.localise(wavefront, **context)
        return wavefront.normalise() if self.normalise else wavefront


class SparseDynamicOptic(BaseDynamicLayer, SparseOptic):
    """A sparse optic with an optional coordinate source and transformation.

    A shared transformation acts in the global coordinate frame before aperture
    placement. A transformation with a leading aperture axis acts independently in
    each aperture's local frame.

    Transmission, OPD, and phase may be contextual `Parametric` objects evaluated on
    the local grids, allowing differentiable sparse geometry without materialising a
    global pupil.
    """

    amplitude: Array | zdx.Expression | None = None
    jones: Array | zdx.Expression | None = None
    thickness: Array | zdx.Expression | None = None
    n: Array | zdx.Expression | None = None
    propagator: BaseOpticalLayer | None = None

    def __init__(
        self,
        centers,
        transmission=None,
        opd=None,
        phase=None,
        coordinates=None,
        transformation=None,
        normalise=None,
        *,
        alias=None,
    ):
        """Initialise repeated coordinate-dependent local optics.

        Parameters
        ----------
        centers : ArrayLike
            Physical ``(x, y)`` centres with shape ``(n_apertures, 2)``.
        transmission : Array, Parametric, or None
            Shared real amplitude transmission, or a basis with per-aperture
            coefficients.
        opd, phase : Array, Parametric, or None
            Shared OPD in metres and phase in radians, or bases with per-aperture
            coefficients.
        coordinates : Array, GridSpec, or None
            Explicit coordinate source, or incident coordinates when omitted.
        transformation : BaseCoordTransform or None
            Shared global or aperture-vectorised local coordinate transformation.
        normalise : bool or None
            Renormalise the resulting wavefront power when true. Omit to leave
            normalisation unconfigured.
        """
        BaseDynamicLayer.__init__(self, coordinates, transformation)
        SparseOptic.__init__(self, centers, transmission, opd, phase, normalise)

        self.alias = alias

    def _context_at(self, wavefront, center, optic, local=False, **context):
        """Build dynamic context in a global or local transformed frame."""
        # Resolve the coordinate source and its physical sampling
        coordinate_source = optic.coordinates
        if coordinate_source is None:
            coordinates = wavefront.coordinates
            pixel_scale = wavefront.pixel_scale
        elif isinstance(coordinate_source, GridSpec):
            coordinate_source = coordinate_source.sampled(**context)
            coordinates = coordinate_source.coordinates
            pixel_scale = coordinate_source.d_si
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
            **context,
            "wavelengths": wavefront.wavelength,
            "coordinates": coordinates,
            "pixel_scale": pixel_scale,
        }

    coordinates: Array | GridSpec | None = None
    transformation: BaseCoordTransform | None = None
    transmission: Array | zdx.Expression | None = None
    opd: Array | zdx.Expression | None = None
    phase: Array | zdx.Expression | None = None
    normalise: bool | None = None
    centers: Array
