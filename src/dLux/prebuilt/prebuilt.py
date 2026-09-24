"""Convenience constructors for sampled pupils and aberration bases.

Presets use dataclass replacement to retain the builder's optic type and fields.
Unlike a leaf update, this also validates static aliases through the constructor.
"""

from collections.abc import Callable
from dataclasses import replace

import equinox as eqx
import jax.numpy as np
import zodiax as zdx
from jax import Array

import dLux.utils as dlu

from .apertures import ApertureBuilder, NRMLike, SegmentedHex, SimpleCircular
from .definitions import Def

__all__ = ["ZernikeDef", "HSTAper", "JWSTAper", "JWSTNRMAper", "EuclidAper"]


class ZernikeDef(Def, dlu.Unitful):
    """Construct a Basis from sampled modes or a Zernike generator.

    A supplied grid or coordinate array samples the modes once into an ordinary
    ``Basis``. Without sampling, or with ``dynamic=True``, the same Basis contains
    a ``Zernike`` Expression. Generated objects retain the coefficients and unit;
    they do not retain this factory.

    Modes are dimensionless, optionally clipped and normalised on a support.
    Coefficients carry the amplitude in ``unit``. The unit conversion is applied
    after contraction, keeping sampled modes well-scaled for coefficient solves.

    Examples
    --------
    ```python
    import dLux as dl

    # Supplying a grid samples the modes once and creates editable coefficients.
    grid = dl.Grid(8, diam=1.0, unit="m")
    definition = dl.ZernikeDef([4, 5, 6], diameter=1.0, unit="nm")
    basis = definition(grid)
    print(basis)
    ```
    """

    # Inherited fields
    unit: str | None = eqx.field(kw_only=True, default=None)

    nolls: tuple[int, ...] = eqx.field(static=True)
    coeffs: Array | zdx.Expression | None = None
    diameter: Array | zdx.Expression | None = None
    n_sides: int = eqx.field(static=True, default=0)
    oversize: Array | zdx.Expression
    norm: str | None = eqx.field(static=True, default=None)
    method: str = eqx.field(static=True, default="padded")
    grid: dlu.Grid | None = None
    transform: Callable | None = None
    support: Array | zdx.Expression | None = None

    def __init__(
        self,
        nolls=None,
        order=None,
        orders=None,
        coeffs=None,
        diameter=None,
        n_sides=0,
        oversize=0.0,
        norm=None,
        method="padded",
        *,
        empty=False,
        unit=None,
        grid=None,
        transform=None,
        support=None,
        alias=None,
    ):
        """Store mode selection, coefficient values and optional sampling.

        Supply exactly one of ``nolls``, maximum radial ``order``, or selected
        radial ``orders``. ``diameter`` uses SI coordinate units and defaults to
        2.0 during evaluation. ``oversize`` is a fractional diameter enlargement.
        ``n_sides=0`` selects circular modes; values at least three select polygons.
        ``norm`` optionally selects l1, l2, max, rms or p2v mode normalisation.
        Omitted coefficients start at zero; ``empty=True`` instead retains None.

        Parameters
        ----------
        nolls : sequence of ints or None
            One-based Noll indices. Supply exactly one of nolls, order or orders.
        order : int or None
            Maximum radial order, including lower orders; alternative to nolls and
            orders.
        orders : sequence of ints or None
            Selected radial orders; alternative to nolls and order.
        coeffs : Array or Expression or None
            Expansion coefficients with the mode axis last. None creates editable
            zeros unless empty=True.
        diameter : ArrayLike or Expression or None
            Aperture diameter in SI coordinate units; None uses 2.0 during
            evaluation.
        n_sides : int
            Zero selects circular modes; values of at least three select the
            polygonal mapping.
        oversize : ArrayLike or Expression
            Fractional diameter enlargement; zero leaves diameter unchanged.
        norm : str or None
            Optional per-mode support normalisation: l1, l2, max, rms or p2v. None
            retains standard mode amplitudes.
        method : str
            Radial mode evaluation strategy: padded (default) or mapped.
        empty : bool
            Retain coeffs=None instead of allocating zeros when true; false by
            default.
        unit : str or None
            Output coefficient unit, for example nm. Conversion is applied after
            mode contraction; coordinates have their own grid unit.
        grid : Grid or None
            Optional stored sampling; caller context supplies sampling when it is
            absent.
        transform : callable or None
            Map evaluation coordinates into the input frame. None leaves coordinates
            unchanged.
        support : Array or Expression or None
            Optional mode-support mask on the evaluation grid. None uses the natural
            aperture support.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        self.nolls = tuple(dlu.Zernike.indices(nolls, order, orders))
        self.coeffs = dlu.initialise_coeffs(coeffs, (len(self.nolls),), empty=empty)
        self.diameter = zdx.as_array(diameter, dtype=float)
        self.n_sides = int(n_sides)
        self.oversize = zdx.as_array(oversize, dtype=float)
        self.norm = norm
        self.method = method
        self.unit = self.validate_unit(unit)
        self.grid = grid
        self.transform = transform
        self.support = zdx.as_array(support, dtype=float)
        self.alias = alias

    def __call__(
        self,
        grid=None,
        *,
        coordinates=None,
        dynamic=False,
        coeffs=None,
        transform=None,
        support=None,
        diameter=None,
        centers=None,
        alias=None,
        **context,
    ):
        """Build an expansion on supplied or stored sampling.

        Runtime inputs replace stored defaults. Explicit coordinates replace a
        stored grid. ``centers`` supplies an aperture axis before the mode axis,
        with independent coefficients of shape ``(n_apertures, n_modes)``.
        Omitted coefficients become stored zeros of the required shape.
        """
        self = self.populate()
        if grid is not None and coordinates is not None:
            raise ValueError("Supply exactly one of grid or coordinates, not both.")
        if coordinates is None:
            grid = self.grid if grid is None else grid
        # Construct the dimensionless generator independently of coefficient values.
        modes = dlu.Zernike(
            self.nolls,
            diameter=self.diameter if diameter is None else diameter,
            n_sides=self.n_sides,
            oversize=self.oversize,
            norm=self.norm,
            method=self.method,
            grid=grid,
            transform=self.transform if transform is None else transform,
            support=self.support if support is None else support,
            centers=centers,
        )

        # A factory can add an aperture axis to its stored shared coefficients.
        # An explicitly empty definition stays empty in the resulting expansion.
        coeffs = self.coeffs if coeffs is None else zdx.as_array(coeffs)
        shape = modes.coeffs_shape
        if centers is not None and isinstance(coeffs, Array) and coeffs.ndim == 1:
            coeffs = np.broadcast_to(coeffs, shape)

        # Both routes return Basis; only the representation of its modes differs.
        if not dynamic and (grid is not None or coordinates is not None):
            modes = modes.evaluate(coordinates=coordinates, **context)
        return dlu.Basis(
            modes,
            coeffs,
            axes=len(shape),
            shape=shape,
            empty=coeffs is None,
            unit=self.unit,
            alias=alias,
        )


def HSTAper(
    pupil_grid,
    *,
    diameter=2.4,
    secondary_diameter=0.305,
    spider_width=0.038,
    spider_angles=(0, 90, 180, 270),
    opd=None,
    coeffs=None,
    pupil_oversample=5,
    key=None,
    normalise=True,
    jit=True,
    focal_length=None,
    alias=None,
):
    """Construct an Optic with the representative circular HST pupil.

    ``pupil_grid`` defines the sampling; dimensions are metres and spider angles
    are degrees. ``opd`` supplies a basis definition, with optional ``coeffs`` or
    random ``key``. ``pupil_oversample`` controls aperture-edge sampling, while
    ``normalise`` normalises power after applying the pupil. ``focal_length``
    optionally defines a later relay; this constructor performs no propagation.
    """
    builder = SimpleCircular(
        diameter,
        secondary_diameter,
        spider_width,
        spider_angles,
        opd=opd,
        oversample=pupil_oversample,
        key=key,
    )
    pupil = builder(pupil_grid, coeffs=coeffs, normalise=normalise, jit=jit)
    focal_length = zdx.as_array(focal_length, dtype=float)
    return replace(pupil, grid=pupil_grid, focal_length=focal_length, alias=alias)


def JWSTAper(
    pupil_grid,
    *,
    segment_diameter=1.524,
    gap=0.007,
    spider_width=0.1,
    spider_angles=(30, 180, 330),
    opd=None,
    coeffs=None,
    pupil_oversample=5,
    paste_method="scan",
    key=None,
    normalise=True,
    jit=True,
    focal_length=None,
    sparse=False,
    alias=None,
):
    """Construct the representative 18-segment JWST Optic or SparseOptic.

    ``pupil_grid`` defines the sampling; dimensions are metres and spider angles
    are degrees. ``opd`` supplies a basis definition, with optional ``coeffs`` or
    random ``key``. ``pupil_oversample`` controls aperture-edge sampling, while
    ``normalise`` normalises power after applying the pupil. ``focal_length``
    optionally defines a later relay; this constructor performs no propagation.
    With ``sparse=True``, the grid samples each local aperture independently.
    """
    builder = SegmentedHex(
        nrings=3,
        segment_diameter=segment_diameter,
        gap=gap,
        remove_center=True,
        obscurations=(dlu.Spider(spider_width, spider_angles),),
        opd=opd,
        oversample=pupil_oversample,
        paste_method=paste_method,
        key=key,
    )
    pupil = builder(
        pupil_grid, coeffs=coeffs, normalise=normalise, jit=jit, sparse=sparse
    )
    focal_length = zdx.as_array(focal_length, dtype=float)
    return replace(pupil, grid=pupil_grid, focal_length=focal_length, alias=alias)


def JWSTNRMAper(
    pupil_grid,
    *,
    centers=None,
    hole_f2f=0.8,
    opd=None,
    coeffs=None,
    pupil_oversample=5,
    key=None,
    normalise=True,
    jit=True,
    sparse=False,
    focal_length=None,
    alias=None,
):
    """Construct the seven-hole JWST/NIRISS Optic or SparseOptic.

    ``pupil_grid`` defines the sampling; dimensions are metres and spider angles
    are degrees. ``opd`` supplies a basis definition, with optional ``coeffs`` or
    random ``key``. ``pupil_oversample`` controls aperture-edge sampling, while
    ``normalise`` normalises power after applying the pupil. ``focal_length``
    optionally defines a later relay; this constructor performs no propagation.
    With ``sparse=True``, the grid samples each local aperture independently.
    """
    if centers is None:
        centers = (
            (0.0, 2.64),
            (2.28631, 0.0),
            (-2.28631, 1.32),
            (2.28631, -1.32),
            (1.14315, -1.98),
            (-2.28631, -1.32),
            (-1.14315, -1.98),
        )
    diameter = 2 * zdx.as_array(hole_f2f, dtype=float) / np.sqrt(3)
    builder = NRMLike(
        centers,
        dlu.RegularPolygon(6, diameter),
        opd=opd,
        oversample=pupil_oversample,
        key=key,
    )
    pupil = builder(
        pupil_grid,
        coeffs=coeffs,
        normalise=normalise,
        jit=jit,
        sparse=sparse,
    )
    focal_length = zdx.as_array(focal_length, dtype=float)
    return replace(pupil, grid=pupil_grid, focal_length=focal_length, alias=alias)


def EuclidAper(
    pupil_grid,
    *,
    diameter=1.21,
    secondary_diameter=0.395,
    spider_width=0.012,
    spider_angles=(0, 120, 240),
    opd=None,
    coeffs=None,
    pupil_oversample=5,
    key=None,
    normalise=True,
    jit=True,
    focal_length=None,
    alias=None,
):
    """Construct an Optic with the approximate Euclid pupil.

    ``pupil_grid`` defines the sampling; dimensions are metres and spider angles
    are degrees. ``opd`` supplies a basis definition, with optional ``coeffs`` or
    random ``key``. ``pupil_oversample`` controls aperture-edge sampling, while
    ``normalise`` normalises power after applying the pupil. ``focal_length``
    optionally defines a later relay; this constructor performs no propagation.
    """
    shift = np.asarray((secondary_diameter / 2 - spider_width / 2, diameter / 2))
    obscurations = [dlu.Circle(secondary_diameter)]
    for angle in spider_angles:
        transform = dlu.Affine(
            translation=shift,
            rotation=np.deg2rad(angle + 30),
            order=("rotation", "translation"),
        )
        support = dlu.Rectangle(spider_width, diameter, transform=transform)
        obscurations.append(support)
    builder = ApertureBuilder(
        dlu.Circle(diameter),
        obscurations,
        opd=opd,
        oversample=pupil_oversample,
        key=key,
    )
    pupil = builder(pupil_grid, coeffs=coeffs, normalise=normalise, jit=jit)
    focal_length = zdx.as_array(focal_length, dtype=float)
    return replace(pupil, grid=pupil_grid, focal_length=focal_length, alias=alias)
