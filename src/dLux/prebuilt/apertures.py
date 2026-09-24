"""Pupil geometry builders and their shared sampling calculations."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as np
import zodiax as zdx
import jax.random as jr
from jax import Array, vmap

import dLux.utils as dlu

from .definitions import Def
from ..layers.optical import Optic
from ..layers.sparse import SparseOptic

__all__ = [
    "ApertureBuilder",
    "SparseApertureBuilder",
    "SimpleCircular",
    "SegmentedHex",
    "NRMLike",
    "segmented_hex_cens",
    "non_redundant_support",
]


@eqx.filter_jit
def _hex_cens(rmax: float) -> Array:
    """Returns the centres of the six neighbouring hexagons.

    Parameters
    ----------
    rmax : float
        The circumradius of each hexagon.

    Returns
    -------
    centers : Array
        The six neighbour centres with shape (6, 2).
    """
    angles = np.deg2rad(60 * np.arange(6) + 30)
    return np.sqrt(3.0) * rmax * np.stack((np.cos(angles), np.sin(angles)), axis=-1)


@eqx.filter_jit
def segmented_hex_cens(nrings: int, rmax: float, gap: float = 0.0) -> Array:
    """Hex-segment centres including the central segment.

    Parameters
    ----------
    nrings : int
        Number of segment rings including the central segment.
    rmax : float
        Circumradius of one hexagonal segment.
    gap : float = 0.0
        Physical gap between neighbouring segments.

    Returns
    -------
    centers : Array
        The segment centres with shape (n_segments, 2).

    Convention
    ----------
    nrings = 1 -> 1 segment
    nrings = 2 -> 7 segments
    nrings = 3 -> 19 segments
    """
    if nrings < 1:
        raise ValueError("`nrings` must be >= 1.")

    rseg = rmax + gap / np.sqrt(3.0)

    cens = [np.zeros((1, 2))]

    if nrings >= 2:
        inner = _hex_cens(rseg)
        cens.append(inner)

    for i in range(2, nrings):
        outer = _hex_cens(i * rseg)
        cens.append(outer)

        end = np.roll(outer, -1, axis=0)
        fraction = np.linspace(0.0, 1.0, i + 1)[1:-1]
        mids = outer[:, None] + fraction[None, :, None] * (end - outer)[:, None]
        shaped = mids.reshape([6 * (i - 1), 2])
        cens.append(shaped)

    return np.concatenate(cens)


def non_redundant_support(apertures: Array) -> Array:
    """Returns a non-redundant support mask for overlapping sub-apertures.

    Parameters
    ----------
    apertures : Array
        The per-segment aperture masks with shape (n_segments, npixels, npixels).

    Returns
    -------
    support : Array
        A boolean support mask with the same shape as ``apertures`` and no pixel
        assigned to more than one segment.
    """
    # Get the individual and overlapping aperture supports
    aper_support = apertures > 0
    support_sum = aper_support.sum(0)

    # Select the strongest aperture wherever supports overlap
    indices = np.arange(len(apertures))[:, None, None]
    strongest = np.argmax(apertures, axis=0)
    selected = indices == strongest[None]

    # Keep ordinary supports and only select one overlapping aperture
    unique = support_sum <= 1
    return aper_support & (unique[None] | selected)


def _component(primary, obscurations, coordinates, pixel_scale):
    """Sample one local primary and remove its obscurations."""
    values = primary.resolve(coordinates=coordinates, pixel_scale=pixel_scale)
    transmission = values
    for shape in obscurations:
        transmission *= 1 - shape.resolve(
            coordinates=coordinates, pixel_scale=pixel_scale
        )
    return values, transmission


def _sample_aperture(
    primary,
    grid,
    obscurations=(),
    oversample=1,
    transform=None,
    *,
    centers=None,
    global_obscurations=(),
):
    """Sample transmission and primary support on a shared output grid.

    Repeated apertures sum before global obscurations. OPD support includes every
    pixel touched by the sampled primary.
    """
    fine = grid.oversample(oversample)
    coordinates = fine.transformed(transform)
    pixel_scale = fine.d_si

    # Evaluate either one primary or its repeated local coordinate frames.
    if centers is None:
        primaries, transmission = _component(
            primary, obscurations, coordinates, pixel_scale
        )
    else:

        def sample(center):
            local = dlu.Affine(translation=center)(coordinates)
            return _component(primary, obscurations, local, pixel_scale)

        primaries, components = vmap(sample)(centers)
        transmission = np.clip(components.sum(0), 0.0, 1.0)

    for shape in global_obscurations:
        transmission *= 1 - shape.resolve(
            coordinates=coordinates, pixel_scale=pixel_scale
        )
    transmission = dlu.downsample(transmission, oversample)

    # Reduce primary coverage on the same sampling as the transmission.
    primaries = dlu.downsample(primaries, oversample)
    support = primaries > 0 if centers is None else non_redundant_support(primaries)
    return transmission, support


def _initialise_basis(basis, coeffs=None, key=None, initial_shape=None):
    """Keep a complete expansion, replacing its coefficients only when requested."""
    if coeffs is not None and key is not None:
        raise ValueError("Provide only one of coeffs or key.")
    if key is not None:
        shape = basis.coeffs.shape if initial_shape is None else initial_shape
        coeffs = jr.normal(key, shape)
    elif coeffs is None:
        if initial_shape is None:
            return basis
        coeffs = np.broadcast_to(basis.coeffs, initial_shape)
    return basis.set(coeffs=zdx.as_array(coeffs))


class ApertureBuilder(Def):
    """Compose a primary, obscurations, and optional OPD definition on a grid.

    This is the general construction API. Named telescope builders retain simple
    numeric constructors and act as templates which assemble these definitions.
    Grid defines two spatial axes; scalar counts and spacings describe a square.

    Returns
    -------
    transmission : Array
        Returned by ``build`` when ``opd`` is ``None`` and support is not requested.
    transmission, support : tuple of Array
        Returned when ``opd`` is ``None`` and ``return_support=True``.
    transmission, basis : tuple[Array, ParametricBasis]
        Returned when ``opd`` is present.
    transmission, basis, support : tuple[Array, ParametricBasis, Array]
        Returned when ``opd`` is present and ``return_support=True``.

    Examples
    --------
    Compose an aperture, build its sampled arrays, or materialise it as an optic:

    ```python
    import jax.numpy as np
    import jax.random as jr

    import dLux as dl
    import dLux.utils as dlu

    # Construct a reusable aperture builder
    builder = dl.ApertureBuilder(
        primary=dl.Circle(diameter=1.0),
        obscurations=[
            dl.Circle(diameter=0.3),
            dl.Spider(width=0.02, angles=[0, 90]),
        ],
        opd=dl.ZernikeDef(
            orders=[2, 3],  # Skip piston, tip, and tilt
            norm="rms",
            unit="nm",
        ),
        oversample=5,
        key=jr.key(0),
    )

    # Build the sampled aperture and OPD basis directly
    grid = dl.Grid(n=128, diam=1.2, unit="m")
    transmission, basis = builder.build(grid)

    # Materialise the same definition as an optical layer
    optic = builder(grid)

    # Build a transformed version of the aperture
    transform = dl.Affine(rotation=np.deg2rad(10))
    rotated_optic = builder(grid, transform=transform)
    ```
    """

    primary: dlu.Shape
    obscurations: tuple
    opd: Def | dlu.ParametricBasis | None = None
    key: Array | None = None
    grid: dlu.Grid | None = None
    oversample: tuple[int, int] = eqx.field(static=True)

    def __init__(
        self,
        primary,
        obscurations=(),
        opd=None,
        oversample=5,
        key=None,
        *,
        grid=None,
        alias=None,
    ):
        """Initialise a dense aperture construction recipe.

        Parameters
        ----------
        primary : Shape
            Transmissive primary geometry.
        obscurations : list or tuple of Shape
            Geometries removed from the primary.
        opd : Def, ParametricBasis, or None
            Optional sampled OPD definition.
        oversample : int or tuple[int, int]
            Positive hard-edge sampling factors in physical ``(x, y)`` order.
        key : Array or None
            Default JAX random key for coefficient initialisation.
        grid : Grid or None
            Default sampling for build and call; a call-time grid replaces it.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        # Validate the primary and obscuration geometry
        if not isinstance(primary, dlu.Shape):
            raise TypeError("primary must be a Shape.")
        if not isinstance(obscurations, (list, tuple)):
            raise TypeError("obscurations must be a list or tuple of Shape objects.")
        obscurations = tuple(obscurations)
        if not all(isinstance(shape, dlu.Shape) for shape in obscurations):
            raise TypeError("obscurations must contain only Shape objects.")

        # Store the geometry and optional OPD definition
        self.primary = primary
        self.obscurations = obscurations
        if opd is not None and not isinstance(opd, (Def, dlu.ParametricBasis)):
            raise TypeError("opd must be a Def, a basis expansion, or None.")
        self.opd = opd
        self.key = key
        self.grid = grid
        self.oversample = dlu.as_size(oversample, 2, "oversample")

        self.alias = alias

    def _validate(self, grid, transform):
        """Validate the construction grid and optional coordinate transformation."""
        if not isinstance(grid, dlu.Grid):
            raise TypeError("grid must be a Grid.")
        if grid.n is None or grid.d is None:
            raise ValueError("grid must define n and d.")
        if grid.d.ndim != 1 or (grid.c is not None and grid.c.ndim != 1):
            raise ValueError("batched grids are not supported by ApertureBuilder.")
        if transform is not None and not isinstance(transform, dlu.BaseCoordTransform):
            raise TypeError("transform must be a BaseCoordTransform or None.")

    def _resolve_key(self, key, coeffs):
        """Resolve call-time coefficients and keys against the stored default."""
        if key is None and coeffs is None:
            return self.key
        return key

    def build(self, grid=None, transform=None, jit=True, return_support=False):
        """Build sampled aperture data on a two-dimensional grid.

        OPD factories return complete basis expansions, including their coefficient
        arrays and units. Existing expansions are retained without resampling.

        Parameters
        ----------
        grid : Grid
            One-dimensional square or explicit two-dimensional output sampling.
        transform : BaseCoordTransform or None
            Optional map into the aperture's local frame.
        jit : bool
            Compile the fixed-topology construction path.
        return_support : bool
            Append the primary support to the returned arrays.
        """
        grid = self.grid if grid is None else grid
        self._validate(grid, transform)
        build_fn = eqx.filter_jit(self._build) if jit else self._build
        return build_fn(grid, transform, return_support)

    def _aperture_data(self, grid, transform):
        """Sample the pupil transmission and support used by its OPD basis."""
        if self.primary.extent is None and isinstance(self.opd, Def):
            raise ValueError("primary must define an extent when opd is provided.")
        return _sample_aperture(
            self.primary, grid, self.obscurations, self.oversample, transform
        )

    def _build(self, grid, transform, return_support=False):
        """Build sampled transmission, OPD data, and optional support."""
        # Generate the aperture transmission and support
        transmission, support = self._aperture_data(grid, transform)

        # Return the transmission when no OPD is defined
        if self.opd is None:
            if return_support:
                return transmission, support
            return transmission

        # Generate the supported OPD data
        centers = getattr(self, "centers", None)
        basis = self._build_basis(grid, transform, support, centers)

        # Return the requested components
        if return_support:
            return transmission, basis, support
        return transmission, basis

    def _build_basis(self, grid, transform, support, centers=None):
        """Sample an OPD factory on its aperture support, or retain a ready basis."""
        basis = self.opd
        if isinstance(basis, Def):
            basis = basis(
                coordinates=grid.transformed(transform),
                support=support,
                diameter=2 * self.primary.extent,
                centers=centers,
            )
        return basis

    def _make_optic(
        self, transmission, basis, coeffs, key, normalise, *, centers=None, shared=False
    ):
        """Initialise coefficients once and package dense or local pupil samples.

        Sampling routines return a transmission and an optional basis. Dense
        bases already describe their coefficient axes; a local sparse basis may
        share its coefficients or receive one independent set per aperture.
        """
        key = self._resolve_key(key, coeffs)
        if basis is not None:
            shape = None
            if centers is not None:
                shape = basis.coeffs.shape[-basis.coeffs_ndim :]
                if not shared:
                    shape = (len(centers),) + shape
            basis = _initialise_basis(basis, coeffs, key, shape)

        if centers is not None:
            return SparseOptic(
                centers, transmission=transmission, opd=basis, normalise=normalise
            )
        return Optic(transmission=transmission, opd=basis, normalise=normalise)

    def __call__(
        self,
        grid=None,
        transform=None,
        coeffs=None,
        key=None,
        normalise=True,
        jit=True,
    ):
        """Materialise this definition as a globally sampled ``Optic``.

        ``coeffs`` and a call-time ``key`` are mutually exclusive. Explicit
        coefficients override the constructor key, while a call-time key replaces
        it. Omitting all three preserves the expansion's coefficients, which
        default to zeros in the basis factories. ``normalise`` retains the
        existing wavefront-normalisation meaning of ``Optic.normalise``.

        Parameters
        ----------
        grid : Grid
            One-dimensional square or explicit two-dimensional sampling grid.
        transform : BaseCoordTransform or None
            Optional map from grid coordinates into the aperture frame.
        coeffs : Array or None
            Explicit coefficients matching the sampled OPD basis shape.
        key : Array or None
            JAX random key overriding the constructor key.
        normalise : bool
            Renormalise wavefront power after applying the returned optic.
        jit : bool
            Compile the fixed-topology sampling calculation.

        Returns
        -------
        optic : Optic
            Sampled transmission and optional explicit OPD basis.
        """
        components = self.build(grid, transform, jit)
        if self.opd is None:
            transmission, basis = components, None
        else:
            transmission, basis = components
        return self._make_optic(transmission, basis, coeffs, key, normalise)


class SparseApertureBuilder(ApertureBuilder):
    """Repeat one local aperture shape over explicit ``(x, y)`` centres.

    Notes
    -----
    Calling the builder returns one globally sampled compound pupil by default.
    Passing ``sparse=True`` instead returns one local transmission plus explicit
    centres. Global obscurations cannot be represented by that shared local
    transmission.

    Examples
    --------
    Materialise the same repeated aperture globally or as a compact sparse optic:

    ```python
    import dLux as dl
    import dLux.utils as dlu

    # Construct a repeated sparse-aperture builder
    builder = dl.SparseApertureBuilder(
        subaperture=dl.Circle(diameter=0.2),
        centers=[
            [-0.3, 0.0],
            [0.0, 0.3],
            [0.3, 0.0],
        ],
        opd=dl.ZernikeDef(
            orders=[1, 2],
            norm="rms",
            unit="nm",
        ),
        oversample=5,
    )

    # Build the complete aperture on one global grid
    global_grid = dl.Grid(n=256, diam=1.0, unit="m")
    transmission, basis = builder.build(global_grid)
    optic = builder(global_grid)

    # Build one compact local aperture and retain its explicit centres
    local_grid = dl.Grid(n=64, diam=0.25, unit="m")
    sparse_optic = builder(local_grid, sparse=True)
    ```
    """

    primary: dlu.Shape
    obscurations: tuple
    opd: Def | dlu.ParametricBasis | None = None
    key: Array | None = None
    grid: dlu.Grid | None = None
    oversample: tuple[int, int] = eqx.field(static=True)
    centers: Array
    global_obscurations: tuple

    def __init__(
        self,
        subaperture,
        centers,
        obscurations=(),
        global_obscurations=(),
        opd=None,
        oversample=5,
        key=None,
        *,
        grid=None,
        alias=None,
    ):
        """Initialise a repeated sparse-aperture construction recipe.

        Parameters
        ----------
        subaperture : Shape
            Local geometry shared by every centre.
        centers : ArrayLike
            Physical ``(x, y)`` centres with shape ``(n_apertures, 2)``.
        obscurations : list or tuple of Shape
            Local obscurations repeated within each sub-aperture.
        global_obscurations : list or tuple of Shape
            Geometries removed after assembling the global pupil.
        opd : Def, ParametricBasis, or None
            Optional shared local OPD definition.
        oversample : int or tuple[int, int]
            Positive hard-edge sampling factors in physical ``(x, y)`` order.
        key : Array or None
            Default JAX random key for coefficient initialisation.
        grid : Grid or None
            Optional stored sampling; caller context supplies sampling when it is
            absent.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        # Validate and store the sub-aperture centres
        centers = zdx.as_array(centers, dtype=float)
        if centers.ndim != 2 or centers.shape[-1] != 2:
            raise ValueError("centers must have shape (n_apertures, 2).")
        self.centers = centers

        # Validate and store the global obscuration geometry
        if not isinstance(global_obscurations, (list, tuple)):
            raise TypeError(
                "global_obscurations must be a list or tuple of Shape objects."
            )
        global_obscurations = tuple(global_obscurations)
        if not all(isinstance(shape, dlu.Shape) for shape in global_obscurations):
            raise TypeError("global_obscurations must contain only Shape objects.")
        self.global_obscurations = global_obscurations

        # Initialise the shared local aperture definition
        super().__init__(subaperture, obscurations, opd, oversample, key, grid=grid)

        self.alias = alias

    def _aperture_data(self, grid, transform):
        """Sample the assembled transmission and independent aperture supports."""
        if self.primary.extent is None and isinstance(self.opd, Def):
            raise ValueError("subaperture must define an extent when opd is used.")
        return _sample_aperture(
            self.primary,
            grid,
            self.obscurations,
            self.oversample,
            transform,
            centers=self.centers,
            global_obscurations=self.global_obscurations,
        )

    def _sparse_data(self, grid, transform):
        """Sample local aperture fields, retaining separate global obscurations."""
        fine = grid.oversample(self.oversample)
        coordinates = fine.transformed(transform)
        _, transmission = _component(
            self.primary, self.obscurations, coordinates, fine.d_si
        )

        # Global supports are sampled at each aperture's actual pupil position.
        # Only this case needs one transmission array per aperture.
        if self.global_obscurations:
            global_coordinates = coordinates[None] + self.centers[:, :, None, None]
            for shape in self.global_obscurations:
                blocked = shape.resolve(
                    coordinates=global_coordinates, pixel_scale=fine.d_si
                )
                transmission = transmission * (1 - blocked)
        transmission = dlu.downsample(transmission, self.oversample)

        # The basis uses the local primary support, independently of global spiders.
        support = (
            self.primary.resolve(
                coordinates=grid.transformed(transform), pixel_scale=grid.d_si
            )
            > 0
        )
        basis = self._build_basis(grid, transform, support)
        return transmission, basis

    def __call__(
        self,
        grid=None,
        transform=None,
        coeffs=None,
        key=None,
        normalise=True,
        jit=True,
        sparse=False,
        shared=False,
    ):
        """Materialise this definition as an ``Optic`` or ``SparseOptic``.

        By default this uses the globally sampled ``Optic`` contract. With
        ``sparse=True``, ``shared=True`` keeps the native OPD coefficient shape for
        every aperture, while ``shared=False`` adds a leading aperture axis.
        Explicit coefficients may use either representation and override the stored
        key. A call-time key overrides the stored key and follows the selected layout.

        Parameters
        ----------
        grid : Grid
            One-dimensional square or explicit two-dimensional sampling grid.
        transform : BaseCoordTransform or None
            Optional map from grid coordinates into the aperture frame. Transforms
            require dense output and cannot be used with ``sparse=True``.
        coeffs : Array or None
            Explicit OPD coefficients. For sparse output, the leading aperture axis
            is present when ``shared=False`` and omitted when ``shared=True``.
        key : Array or None
            JAX random key overriding the constructor key. Mutually exclusive with
            call-time ``coeffs``.
        normalise : bool
            Renormalise wavefront power after applying the returned optic.
        jit : bool
            Compile the fixed-topology sampling calculation.
        sparse : bool
            Return a shared locally sampled `SparseOptic`; otherwise return a global
            densely sampled `Optic`.
        shared : bool
            Share native OPD coefficients between apertures when sparse, or add a
            leading independent aperture axis when false.

        Returns
        -------
        optic : Optic or SparseOptic
            Materialised dense or sparse pupil representation.
        """
        grid = self.grid if grid is None else grid

        # Materialise a global optic unless sparse output is requested
        if not sparse:
            return super().__call__(grid, transform, coeffs, key, normalise, jit)

        # Validate sparse construction requirements
        self._validate(grid, transform)
        if transform is not None:
            raise ValueError(
                "transform cannot be represented by one shared local SparseOptic; "
                "use sparse=False instead."
            )

        # Sample the shared local transmission and OPD basis
        build_fn = eqx.filter_jit(self._sparse_data) if jit else self._sparse_data
        transmission, basis = build_fn(grid, transform)

        return self._make_optic(
            transmission,
            basis,
            coeffs,
            key,
            normalise,
            centers=self.centers,
            shared=shared,
        )


class SimpleCircular(ApertureBuilder):
    """Pedagogical circular pupil assembled with a compact numeric API.

    Examples
    --------
    Construct a circular pupil with a secondary, supports, and low-order OPD:

    ```python
    import dLux as dl
    import dLux.utils as dlu

    # Construct a circular pupil with a secondary and radial supports
    pupil = dl.SimpleCircular(
        diameter=1.0,
        secondary_diameter=0.3,
        spider_width=0.02,
        spider_angles=[0, 120, 240],
        opd=dl.ZernikeDef(
            orders=[2, 3],  # Skip piston, tip, and tilt
            norm="rms",
            unit="nm",
        ),
    )

    # Materialise the pupil as an optical layer
    grid = dl.Grid(n=128, diam=1.2, unit="m")
    optic = pupil(grid)
    ```
    """

    primary: dlu.Shape
    obscurations: tuple
    opd: Def | dlu.ParametricBasis | None = None
    key: Array | None = None
    grid: dlu.Grid | None = None
    oversample: tuple[int, int] = eqx.field(static=True)

    def __init__(
        self,
        diameter,
        secondary_diameter=None,
        spider_width=None,
        spider_angles=None,
        opd=None,
        oversample=5,
        key=None,
        *,
        grid=None,
        alias=None,
    ):
        """Initialise a configurable circular pupil template.

        Parameters
        ----------
        diameter : float
            Primary diameter in metres.
        secondary_diameter : float or None
            Optional central obscuration diameter.
        spider_width, spider_angles : ArrayLike or None
            Radial-support width and angles in degrees; supply both or neither.
        opd : Def, ParametricBasis, or None
            Optional OPD definition.
        oversample : int or tuple[int, int]
            Hard-edge sampling factors.
        key : Array or None
            Default JAX random key for OPD coefficient initialisation.
        grid : Grid or None
            Optional stored sampling; caller context supplies sampling when it is
            absent.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        if (spider_width is None) != (spider_angles is None):
            raise ValueError("spider_width and spider_angles must both be provided.")
        primary = dlu.Circle(diameter)
        obscurations = []
        if secondary_diameter is not None:
            obscurations.append(dlu.Circle(secondary_diameter))
        if spider_width is not None:
            obscurations.append(dlu.Spider(spider_width, spider_angles))
        super().__init__(primary, obscurations, opd, oversample, key, grid=grid)

        self.alias = alias


class SegmentedHex(SparseApertureBuilder):
    """Pedagogical hexagonal tiling assembled with a compact numeric API.

    ``segment_diameter`` is the point-to-point diameter of the circumscribed circle;
    ``segment_f2f`` is the flat-to-flat width. Exactly one must be provided.
    Obscurations are global structures applied across the assembled pupil.

    Notes
    -----
    Both methods are numerically equivalent. Changing ``paste_method`` through an
    immutable update may trigger a separate JAX compilation. Compact OPD bases are
    evaluated directly at the output sampling and clipped by the corresponding hard
    segment support; only the transmission uses the configured oversampling.

    Examples
    --------
    Construct a segmented pupil with independent low-order segment errors:

    ```python
    import jax.random as jr

    import dLux as dl
    import dLux.utils as dlu

    # Construct a segmented pupil with independent low-order segment errors
    pupil = dl.SegmentedHex(
        nrings=3,
        segment_f2f=1.3,
        gap=0.01,
        opd=dl.ZernikeDef(
            orders=[1, 2],
            norm="rms",
            unit="nm",
        ),
    )

    # Materialise the segmented pupil as an optic
    grid = dl.Grid(n=256, diam=7.0, unit="m")
    optic = pupil(grid=grid, key=jr.key(0))
    ```
    """

    primary: dlu.Shape
    obscurations: tuple
    opd: Def | dlu.ParametricBasis | None = None
    key: Array | None = None
    grid: dlu.Grid | None = None
    oversample: tuple[int, int] = eqx.field(static=True)
    centers: Array
    global_obscurations: tuple
    paste_method: str

    def __init__(
        self,
        nrings,
        segment_diameter=None,
        segment_f2f=None,
        gap=0.0,
        remove_center=False,
        obscurations=(),
        opd=None,
        oversample=5,
        paste_method="scan",
        key=None,
        *,
        grid=None,
        alias=None,
    ):
        """Initialise an ideal hexagonally segmented pupil.

        Parameters
        ----------
        nrings : int
            Number of rings including the central segment.
        segment_diameter, segment_f2f : float or None
            Point-to-point and flat-to-flat size alternatives; supply exactly one.
        gap : float
            Non-negative edge-to-edge segment gap.
        remove_center : bool
            Whether to omit the central segment.
        obscurations : list or tuple of Shape
            Global obscurations applied after assembly.
        opd : Def, ParametricBasis, or None
            Optional per-segment OPD definition.
        oversample : int or tuple[int, int]
            Hard-edge sampling factors.
        paste_method : str
            ``"scan"`` for lower memory or ``"scatter"`` for greater parallelism.
        key : Array or None
            Default JAX random key for per-segment coefficient initialisation.
        grid : Grid or None
            Optional stored sampling; caller context supplies sampling when it is
            absent.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        # Resolve and validate the construction options
        if (segment_diameter is None) == (segment_f2f is None):
            raise ValueError("Provide exactly one of segment_diameter or segment_f2f.")
        nrings = dlu.as_size(nrings, 1, "nrings")[0]
        gap = zdx.as_array(gap, dtype=float)
        segment_f2f = zdx.as_array(segment_f2f, dtype=float)
        segment_diameter = zdx.as_array(segment_diameter, dtype=float)
        if segment_diameter is None:
            segment_diameter = 2 * segment_f2f / np.sqrt(3)
        if segment_diameter.ndim != 0 or gap.ndim != 0:
            raise ValueError("segment diameter and gap must be scalar.")
        if segment_diameter <= 0:
            raise ValueError("segment diameter must be greater than zero.")
        if gap < 0:
            raise ValueError("gap must be greater than or equal to zero.")

        paste_method = str(paste_method).lower()
        if paste_method not in ("scan", "scatter"):
            raise ValueError("paste_method must be either 'scan' or 'scatter'.")

        # Generate the requested ideal segment centres
        primary = dlu.RegularPolygon(6, segment_diameter)
        centers = segmented_hex_cens(nrings, segment_diameter / 2, gap)
        if remove_center:
            centers = centers[1:]

        # Initialise the general sparse-aperture definition
        super().__init__(
            primary,
            centers,
            global_obscurations=obscurations,
            opd=opd,
            oversample=oversample,
            key=key,
            grid=grid,
        )

        self.paste_method = paste_method

        self.alias = alias

    def build(self, grid=None, transform=None, jit=True, return_support=False):
        """Build a compact ideal or densely transformed segmented aperture.

        Parameters
        ----------
        grid : Grid
            Output pupil sampling.
        transform : BaseCoordTransform or None
            Optional global transform; transformed pupils use dense construction.
        jit : bool
            Compile the fixed-topology construction path.
        return_support : bool
            Return per-segment support and therefore use dense construction.
        """
        # Promote and validate the construction grid
        grid = self.grid if grid is None else grid
        self._validate(grid, transform)

        # Use global sampling when compact ideal placement is not possible
        if transform is not None or self.opd is not None or return_support:
            return super().build(grid, transform, jit, return_support)

        # Assemble compact transmission stamps through the selected path
        fine, spec = self._stamp_data(grid)
        build_fn = self._assemble_transmission
        build_fn = eqx.filter_jit(build_fn) if jit else build_fn
        return build_fn(fine, spec)

    def _stamp_data(self, grid):
        """Fix compact stamp sampling and integer placement before compilation."""
        fine = grid.oversample(self.oversample)
        spec = dlu.PasteSpec.from_grid(fine, self.centers, self.primary.extent)
        return fine, spec

    def _assemble_transmission(self, fine, spec):
        """Sample local transmissions, paste them, and apply global obscurations."""
        sample = lambda coords: _component(
            self.primary, self.obscurations, coords, spec.d
        )[1]
        components = vmap(sample)(spec.coordinates)
        transmission = np.clip(spec.paste(components, self.paste_method), 0.0, 1.0)
        for shape in self.global_obscurations:
            transmission *= 1 - shape(fine)
        return dlu.downsample(transmission, self.oversample)

    def _assemble_opd(self, spec):
        """Sample supported basis vectors on each stamp's native coordinates."""
        coordinates = spec.coordinates
        sample = lambda coords: self.primary.resolve(
            coordinates=coords, pixel_scale=spec.d
        )
        support = vmap(sample)(coordinates) > 0
        diameter = 2 * self.primary.extent
        sample = lambda coords, mask: self.opd(
            coordinates=coords, support=mask, diameter=diameter
        )
        basis = eqx.filter_vmap(sample)(coordinates, support)
        # Every stamp has independent coefficients; its axis belongs to the basis.
        basis = dlu.Basis(
            basis.basis, basis.coeffs, axes=basis.axes + 1, unit=basis.unit
        )
        return basis, support

    def _pasted_opd(self, grid, jit=True):
        """Prepare compact OPD sampling separately from layer packaging."""
        spec = dlu.PasteSpec.from_grid(grid, self.centers, self.primary.extent)
        build_fn = self._assemble_opd
        build_fn = eqx.filter_jit(build_fn) if jit else build_fn
        basis, support = build_fn(spec)
        return basis, support, spec

    def __call__(
        self,
        grid=None,
        transform=None,
        coeffs=None,
        key=None,
        normalise=True,
        jit=True,
        sparse=False,
        shared=False,
    ):
        """Materialise a global pasted optic or a genuinely sparse optic.

        Parameters
        ----------
        grid : Grid
            One-dimensional square or explicit two-dimensional pupil sampling.
        transform : BaseCoordTransform or None
            Optional global transform. Transformed pupils use dense construction.
        coeffs : Array or None
            Explicit per-segment OPD coefficients.
        key : Array or None
            JAX random key overriding the constructor key; mutually exclusive with
            call-time ``coeffs``.
        normalise, jit : bool
            Set output-optic normalisation and compiled construction respectively.
        sparse : bool
            Return a `SparseOptic` instead of the compactly assembled global `Optic`.
        shared : bool
            Share sparse OPD coefficients between segments when ``sparse=True``.

        Returns
        -------
        optic : Optic or SparseOptic
            Materialised ideal segmented pupil. Untransformed global OPD uses a
            compact `PastedBasis` by default.
        """
        grid = self.grid if grid is None else grid

        # Use sparse or transformed global construction when requested
        if sparse or transform is not None or isinstance(self.opd, dlu.ParametricBasis):
            return super().__call__(
                grid, transform, coeffs, key, normalise, jit, sparse, shared
            )

        # Promote and validate the ideal construction grid
        self._validate(grid, transform)

        # Generate the compactly assembled pupil transmission
        fine, spec = self._stamp_data(grid)
        build_fn = self._assemble_transmission
        build_fn = eqx.filter_jit(build_fn) if jit else build_fn
        transmission = build_fn(fine, spec)
        # Keep compact basis storage while sharing coefficient initialisation.
        opd = None
        if self.opd is not None:
            basis, _, spec = self._pasted_opd(grid, jit)
            opd = dlu.PastedBasis(
                basis.basis, spec, basis.coeffs, self.paste_method, unit=basis.unit
            )
        return self._make_optic(transmission, opd, coeffs, key, normalise)


class NRMLike(SparseApertureBuilder):
    """Simple NRM with ``(x, y)`` hole centres and one shared hole shape.

    This helper deliberately excludes global stops and obscurations. Sparse
    materialisation always gives every hole independent OPD coefficients.

    Returns
    -------
    Array or tuple of Array
        ``build`` returns the densely sampled global transmission, with OPD basis
        data and aperture support appended when requested.
    SparseOptic
        Calling with ``sparse=True`` creates one local pupil per centre with
        independent OPD coefficients.

    Examples
    --------
    Materialise the same hexagonal-hole mask globally or as a sparse optic:

    ```python
    import jax.random as jr

    import dLux as dl
    import dLux.utils as dlu

    # Construct a non-redundant mask with per-hole aberrations
    pupil = dl.NRMLike(
        centers=[
            [-0.4, 0.0],
            [0.0, 0.3],
            [0.4, 0.0],
        ],
        hole=dl.RegularPolygon(nsides=6, diameter=0.2),
        opd=dl.ZernikeDef(
            orders=[1, 2],
            norm="rms",
            unit="nm",
        ),
    )

    # Materialise the complete mask on a global grid
    global_grid = dl.Grid(n=256, diam=1.2, unit="m")
    optic = pupil(grid=global_grid, key=jr.key(0))

    # Materialise independent holes on a compact local grid
    local_grid = dl.Grid(n=64, diam=0.25, unit="m")
    sparse_optic = pupil(grid=local_grid, key=jr.key(0), sparse=True)
    ```
    """

    primary: dlu.Shape
    obscurations: tuple
    opd: Def | dlu.ParametricBasis | None = None
    key: Array | None = None
    grid: dlu.Grid | None = None
    oversample: tuple[int, int] = eqx.field(static=True)
    centers: Array
    global_obscurations: tuple

    def __init__(
        self, centers, hole, opd=None, oversample=5, key=None, *, grid=None, alias=None
    ):
        """Initialise a shared-shape non-redundant mask.

        Parameters
        ----------
        centers : ArrayLike
            Physical hole centres with shape ``(n_holes, 2)``.
        hole : Shape
            Geometry shared by every hole.
        opd : Def, ParametricBasis, or None
            Optional local OPD definition.
        oversample : int or tuple[int, int]
            Hard-edge sampling factors.
        key : Array or None
            Default JAX random key for per-hole coefficient initialisation.
        grid : Grid or None
            Optional stored sampling; caller context supplies sampling when it is
            absent.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        if not isinstance(hole, dlu.Shape):
            raise TypeError("hole must be a Shape.")

        super().__init__(
            hole, centers, opd=opd, oversample=oversample, key=key, grid=grid
        )

        self.alias = alias

    def __call__(
        self,
        grid=None,
        transform=None,
        coeffs=None,
        key=None,
        normalise=True,
        jit=True,
        sparse=False,
    ):
        """Materialise a global NRM or independent sparse holes.

        Parameters
        ----------
        grid : Grid
            One-dimensional square or explicit two-dimensional sampling grid.
        transform : BaseCoordTransform or None
            Optional map into the aperture frame. Transforms require dense output.
        coeffs : Array or None
            Explicit OPD coefficients. Sparse coefficients must begin with an axis
            matching the number of holes.
        key : Array or None
            JAX random key overriding the constructor key. Mutually exclusive with
            call-time ``coeffs``.
        normalise : bool
            Renormalise wavefront power after applying the returned optic.
        jit : bool
            Compile the fixed-topology sampling calculation.
        sparse : bool
            Return independent locally sampled holes as a ``SparseOptic``; otherwise
            return one globally sampled ``Optic``.

        Returns
        -------
        optic : Optic or SparseOptic
            Materialised global pupil or independent sparse-hole representation.
        """
        if sparse and coeffs is not None:
            coeffs = np.asarray(coeffs)
            if coeffs.ndim == 0 or coeffs.shape[0] != len(self.centers):
                raise ValueError("coeffs must have a leading axis matching the holes.")

        return super().__call__(
            grid, transform, coeffs, key, normalise, jit, sparse, False
        )
