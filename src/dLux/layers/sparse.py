"""Sparse optical layers evaluated over centred sub-apertures."""

from __future__ import annotations

from typing import Any

import jax.numpy as np
import jax.tree as jtu
import zodiax as zdx
from jax import Array, vmap

import dLux.utils as dlu

from ..fields import Wavefront
from ..prebuilt.definitions import Def
from .optical import Optic, OpticalLayer

__all__ = ["SparseOptic"]


class SparseOptic(Optic):
    """Generate locally sampled wavefronts over a set of sub-aperture centers.

    ParametricBasis coefficients are shared in their native coefficient shape.
    Their coeffs_ndim property identifies the intrinsic coefficient axes. One leading
    axis matching the number of centres selects independent coefficients per aperture.
    A one-mode basis therefore uses (n_apertures, 1) for independent coefficients.
    Sampled transmission, OPD and phase are shared scalars or 2D arrays, or arrays
    shaped (n_apertures, ny, nx). A third axis always identifies apertures, never
    wavelengths. Expressions follow this same contract after resolution.
    Coordinate transforms support the same explicit leading aperture axis.

    An optional grid supplies the sampling coordinates. A shared transform acts
    before subtracting aperture centres; a transform with an aperture axis acts
    afterwards, in each local frame. Array-valued optical terms are applied
    directly; grid and transform only control expression evaluation.

    Inherited scalar, refractive and Jones effects use the same operations as
    Optic. Normalisation acts across the complete pupil before Jones losses.
    Aperture-specific Jones samples have shape (n_apertures, ny, nx, 2, 2);
    omitting the aperture axis shares the matrix across apertures. Local Jones
    operations produce explicit matrix axes in the returned field.

    A supplied grid_out propagates these fields after the local effects. The
    aperture axis remains separate until an explicit Wavefront.coherent_sum().

    Examples
    --------
    Propagate locally sampled apertures and coherently interfere their fields:

    ```python
    import dLux as dl
    import dLux.utils as dlu

    # Construct a sparse optic from one locally sampled aperture
    grid = dl.Grid(n=64, diam=0.25, unit="m")
    optic = dl.SparseOptic(
        centers=[[-0.3, 0.0], [0.0, 0.3], [0.3, 0.0]],
        transmission=dl.Circle(diameter=0.2)(grid),
        opd=dl.Basis(dl.utils.Zernike(order=4, diameter=0.2)),
        normalise=True,
    )

    # Apply the optic to generate one field per aperture
    wavefront = dl.Wavefront(wavelength=650e-9, grid=grid)
    wavefront = optic(wavefront)

    # Propagate each aperture independently and interfere the fields
    focal_grid = dl.Grid(n=64, d=10, unit="mas")
    wavefront = dl.PropagatorLayer(focal_grid)(wavefront)
    wavefront = wavefront.coherent_sum()
    ```
    """

    # Inherited fields
    abcd: Array | zdx.Expression | None = None
    grid_out: dlu.Grid | None = None
    focal_length: Any = None
    defocus: Any = None
    inverse: bool | None = False
    jones: Array | zdx.Expression | None = None
    thickness: Array | zdx.Expression | None = None
    n: Array | zdx.Expression | None = None
    amplitude: Array | zdx.Expression | None = None
    phase: Array | zdx.Expression | None = None
    transmission: Array | zdx.Expression | None = None
    opd: Array | zdx.Expression | None = None
    normalise: bool | None = None
    grid: dlu.Grid | None = None
    transform: dlu.BaseCoordTransform | None = None

    centers: Array

    def __init__(
        self,
        centers,
        transmission=None,
        opd=None,
        phase=None,
        normalise=None,
        *,
        amplitude=None,
        jones=None,
        thickness=None,
        n=None,
        grid=None,
        transform=None,
        dynamic=False,
        grid_out=None,
        focal_length=None,
        defocus=None,
        abcd=None,
        inverse=False,
        alias=None,
    ):
        """Initialise repeated locally sampled optics.

        Parameters
        ----------
        centers : ArrayLike
            Physical ``(x, y)`` centres with shape ``(n_apertures, 2)``.
        transmission : Array, Expression, or None
            Real amplitude transmission: shared scalar or (ny, nx) samples,
            (n_apertures, ny, nx) samples, or an expression/basis producing them.
        opd, phase : Array, Expression, or None
            OPD in metres and phase in radians, with the same sampling contract
            as transmission. Bases may have independent per-aperture coefficients.
        normalise : bool or None
            Normalise total power across all apertures to one, preserving relative
            aperture transmission. Omit to retain the incident amplitude.
        grid : Grid or None
            Sampling grid for expressions. None uses the incident wavefront grid.
            This does not resample the incident field or change its output grid.
        transform : BaseCoordTransform or None
            Shared global or aperture-dependent local coordinate transformation.
        dynamic : bool
            Build Def inputs as runtime expressions. Precomputed local expansions
            should be supplied as complete Basis objects: their sampling is then
            explicit, independent of the sparse optic's placement frame.
        grid_out : Grid or None
            Optional output sampling. Supplying it propagates each aperture after
            its local effects; None returns the locally sampled aperture fields.
        focal_length, defocus, abcd, inverse
            Inherited Propagator relay parameters. An Imager uses these parameters
            with its own output grid; its sparse pupil must leave grid_out unset.
        alias : mapping or sequence of pairs, optional
            User-defined Zodiax parameter aliases.
        amplitude : Array or Expression or None
            Real amplitude multiplier broadcast over the spatial samples. None is
            unity.
        jones : None
            Local Jones terms are not supported by SparseOptic; apply a separate
            Optic to the field.
        thickness : Array or Expression or None
            Material thickness in metres. Applied with n as excess optical path (n -
            1) * thickness.
        n : Array or Expression or None
            Dimensionless material index, paired with thickness.
        focal_length : Array or Expression or None
            Relay focal length in metres. Required to connect physical pupil and
            focal-plane lengths without a custom ABCD matrix.
        defocus : Array or Expression or None
            Axial displacement of the output plane in metres; None means zero.
        abcd : Array or Expression or None
            Custom canonical matrix with shape (..., 2, 2), defining relay scale and
            direction.
        inverse : bool or None
            Use inverse propagation for true. Forward is the default; a custom ABCD
            matrix sets this to None.
        """
        # Check centres and keep definitions dynamic until their local frame is known.
        centers = zdx.as_array(centers, dtype=float)
        if centers.ndim != 2 or centers.shape[-1] != 2:
            raise ValueError("centers must have shape (n, 2).")
        self.centers = centers
        for value in (transmission, opd, phase, amplitude, thickness, n, jones):
            if isinstance(value, Def) and not dynamic:
                sampling = grid if grid is not None else getattr(value, "grid", None)
                if sampling is not None:
                    raise ValueError(
                        "SparseOptic cannot cache Def inputs in a shared grid frame. "
                        "Use dynamic=True, or construct a local Basis first."
                    )
        super().__init__(
            transmission,
            opd,
            phase,
            normalise,
            amplitude=amplitude,
            jones=jones,
            thickness=thickness,
            n=n,
            grid=grid,
            transform=transform,
            dynamic=dynamic,
            grid_out=grid_out,
            focal_length=focal_length,
            defocus=defocus,
            abcd=abcd,
            inverse=inverse,
        )

        self.alias = alias

    @property
    def n_apertures(self) -> int:
        """Return the number of centred sub-apertures."""
        return len(self.centers)

    @staticmethod
    def _slice(obj, index, size, **context):
        """Select one centre from a shared or centre-vectorised object."""
        if isinstance(obj, dlu.ParametricBasis):
            # Select coefficients without generating modes in the wrong frame.
            axes = obj.coeffs_ndim
            coeffs = zdx.resolve(obj.coeffs, **context)
            if coeffs is None:
                return obj, False

            if coeffs.ndim == axes:
                return obj.set(coeffs=coeffs), False
            if coeffs.ndim == axes + 1 and coeffs.shape[0] == size:
                return obj.set(coeffs=coeffs[index]), True
            raise ValueError(
                f"coeffs must have {axes} coefficient axes, optionally preceded "
                f"by an aperture axis of size {size}."
            )

        # Describe the intrinsic parameter shapes for each coordinate-transform family.
        if isinstance(obj, dlu.Distortion):
            params = (("distortion", obj.powers.shape),)
        elif isinstance(obj, dlu.AffineMap):
            params = (("matrix", (2, 2)), ("offset", (2,)))
        else:
            params = (
                ("translation", (2,)),
                ("rotation", ()),
                ("scale", (2,)),
                ("shear", (2,)),
            )

        # Select one aperture only for parameters with a leading aperture axis.
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

        # Treat each basis or transform as a leaf when selecting an aperture.
        types = (
            (dlu.ParametricBasis,)
            if bases
            else (dlu.Distortion, dlu.AffineMap, dlu.Affine)
        )
        is_leaf = lambda leaf: isinstance(leaf, types)
        local_transform = False

        def select(leaf):
            nonlocal local_transform
            if not isinstance(leaf, types):
                return leaf
            leaf, local = self._slice(leaf, index, self.n_apertures, **context)
            local_transform |= local and isinstance(leaf, dlu.BaseCoordTransform)
            return leaf

        if bases:
            return jtu.map(select, self, is_leaf=is_leaf), False

        # Only the outer transform chooses between global and local placement.
        # Nested transforms are selected too, but stay in their own child frames.
        transform = jtu.map(select, self.transform, is_leaf=is_leaf)
        outer_is_local = local_transform
        optic = jtu.map(select, self.set(transform=None), is_leaf=is_leaf)
        return optic.set(transform=transform), outer_is_local

    def _context_at(
        self,
        wavefront: Wavefront,
        center: Array,
        optic: SparseOptic,
        local=False,
        **context,
    ) -> dict:
        """Sample the chosen grid in one global or locally transformed frame."""
        if optic.grid is not None:
            wavefront = wavefront.set(grid=optic.grid)
        # Start before Optic's transform: sparse placement determines its order.
        context = OpticalLayer.context(wavefront, **context)
        coordinates = context.pop("coordinates")

        # Shared mappings act before placement; aperture-specific ones act after.
        if optic.transform is not None and not local:
            coordinates = optic.transform(coordinates)
        coordinates = dlu.Affine(translation=center)(coordinates)
        if optic.transform is not None and local:
            coordinates = optic.transform(coordinates)

        return {**context, "coordinates": coordinates}

    def _prepare_at(self, index, center, wavefront, **context):
        """Resolve one aperture's fields in their authoritative local frame."""
        context = dict(
            context, wavelength=wavefront.wavelength, wavelengths=wavefront.wavelength
        )
        optic, local = self._slice_local(index, bases=False, **context)
        context = self._context_at(wavefront, center, optic, local, **context)

        # Resolve and select coefficients only after their coordinates exist.
        optic, _ = optic._slice_local(index, bases=True, **context)
        transmission = zdx.resolve(optic.transmission, **context)
        if transmission is not None:
            if transmission.ndim == 3:
                if transmission.shape[0] != self.n_apertures:
                    raise ValueError(
                        "transmission must have an aperture axis of size "
                        f"{self.n_apertures}; received {transmission.shape}."
                    )
                transmission = transmission[index]
            context["support"] = transmission != 0
        optic = optic.set(transmission=transmission).resolve_local(**context)

        # Sampled scalar terms may share a patch or carry one leading aperture
        # axis. Wavelength-dependent definitions have resolved at one wavelength.
        updates = {}
        for name in ("transmission", "opd", "amplitude", "phase", "thickness", "n"):
            value = getattr(optic, name)
            if value is None or value.ndim in (0, 2):
                continue
            if value.ndim != 3 or value.shape[0] != self.n_apertures:
                raise ValueError(
                    f"{name} must be scalar, (ny, nx), or "
                    f"({self.n_apertures}, ny, nx) with an aperture axis; "
                    f"received {value.shape}."
                )
            updates[name] = value[index]

        # Jones matrices end in their two intrinsic axes. An explicit aperture
        # axis may lead compact matrices or sampled spatial matrices.
        matrix = optic.jones
        if matrix is not None and matrix.ndim in (3, 5):
            if matrix.shape[0] != self.n_apertures:
                raise ValueError("Jones aperture axis must match centers.")
            updates["jones"] = matrix[index]
        return optic.set(**updates) if updates else optic

    def phasor(self, wavefront: Wavefront, **context) -> Array:
        """Return the coherent sum of all locally centred optic phasors.

        The returned complex array broadcasts against ``wavefront.phasor`` and does
        not introduce a sub-aperture axis. Use `localise` to retain that axis.
        """
        self = self.populate()
        if self.jones is not None:
            raise ValueError(
                "phasor returns a scalar multiplier; "
                "use localise or apply for Jones optics."
            )
        indices = np.arange(self.n_apertures)

        def sample(index, center):
            optic = self._prepare_at(index, center, wavefront, **context)
            return optic.scalar_phasor(wavefront)

        return vmap(sample)(indices, self.centers).sum(0)

    def localise(
        self, wavefront: Wavefront, *, normalise=False, **context
    ) -> Wavefront:
        """Apply local effects, adding an aperture axis before wavelength axes.

        Coordinates and coefficients are prepared per aperture. Optional
        normalisation uses total scalar pupil power before Jones application,
        preserving relative aperture transmission and subsequent polarisation loss.
        """
        if wavefront.batch_ndim:

            def localise(field):
                return self.localise(field, normalise=normalise, **context)

            return wavefront._map_wavefronts(localise)

        self = self.populate()
        indices = np.arange(self.n_apertures)
        polarised = self.jones is not None

        def make_wavefront(index, center):
            local = wavefront.set(c=center / dlu.unit_factor(wavefront.grid.unit))
            optic = self._prepare_at(index, center, local, **context)
            field = optic.apply_scalar(local, normalise=False)
            power = field.power
            if polarised:
                field = optic.apply_jones(field).materialise()
            return field.phasor, power

        phasor, power = vmap(make_wavefront)(indices, self.centers)
        if normalise:
            # Scalar normalisation commutes with the Jones multiplication. Use
            # the recorded pre-Jones power so polarisation losses remain visible.
            total = power.sum(0)
            scale = np.sqrt(total).reshape(
                total.shape + (1,) * (phasor.ndim - total.ndim)
            )
            phasor = phasor / scale

        # This mono field gains one aperture axis. The outer Wavefront mapper
        # restores wavelengths after it, keeping apertures and wavelengths separate.
        c = self.centers / dlu.unit_factor(wavefront.grid.unit)
        d = wavefront.grid.d
        return wavefront.set(
            phasor=phasor, d=d, c=c, jones=True if polarised else wavefront.jones
        )

    def apply_mono(self, wavefront: Wavefront, **context) -> Wavefront:
        """Apply the pupil with one normalisation shared by all apertures."""
        self = self.populate()
        wavefront = self.localise(wavefront, normalise=bool(self.normalise), **context)
        if self.grid_out is not None:
            wavefront = self.propagate(wavefront, **context)
        return wavefront
