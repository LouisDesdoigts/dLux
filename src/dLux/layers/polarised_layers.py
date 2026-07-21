"""Polarised optical layers and parameterised polarisation fields."""

from __future__ import annotations
import equinox as eqx
import jax.numpy as np
import zodiax as zdx
import dLux.utils as dlu
from jax import Array


from .optical_layers import OpticalLayer
from ..wavefronts import Wavefront

__all__ = [
    "FieldDict",
    "ExplicitBasis",
    "PolarisingOptic",
    "UniformPolarisingOptic",
    "LinearPolariser",
    "Retarder",
    "SVLinearPolariser",
    "SVRetarder",
]


class FieldDict(zdx.Base):
    """
    A small dict-backed container with attribute access.

    Spatially varying polarisation layers store their raw parameter leaves, basis
    objects, and coefficients in separate `FieldDict`s. This keeps paths like
    `layer.parameters.angle`, `layer.basis.angle`, and `layer.coefficients.angle`
    explicit and optimisable while letting each layer support arbitrary field names.

    Attributes
    ----------
    values : dict
        Dictionary mapping field names to values.
    """

    values: dict

    def __init__(self: FieldDict, values: dict = None, **kwargs):
        """
        Parameters
        ----------
        values : dict = None
            Initial field dictionary.
        **kwargs
            Additional field values.
        """
        values = {} if values is None else dict(values)
        values.update(kwargs)
        self.values = values

    def __getitem__(self: FieldDict, key: str):
        """
        Returns a field by name.

        Returns
        -------
        value : Array
            Field value.
        """
        return self.values[key]

    def __getattr__(self: FieldDict, key: str):
        """
        Returns a field by attribute access.
        """
        try:
            return self.values[key]
        except KeyError as err:
            raise AttributeError(key) from err

    def get(self: FieldDict, key: str, default=None):
        """
        Returns a field by name, or `default` if it is absent.
        """
        return self.values.get(key, default)

    def __contains__(self: FieldDict, key: str) -> bool:
        """
        Returns whether a field is present.
        """
        return key in self.values


class ExplicitBasis(zdx.Base):
    """Leaf marker for a parameter evaluated from an explicit basis array."""


class FourierBasis(zdx.Base):
    """Leaf marker for a parameter evaluated from cached Fourier kernels."""


def _eval_leaf(leaf, basis, coefficients) -> Array:
    """
    Evaluates a parameter leaf using the matching basis and coefficients.

    Raw scalar and array leaves are returned directly. `ExplicitBasis` leaves use an
    explicit basis array with `dlu.eval_basis`, while `FourierBasis` leaves use cached
    Fourier kernels with `dlu.eval_fourier_basis`.
    """
    if isinstance(leaf, ExplicitBasis):
        if basis is None or coefficients is None:
            raise ValueError("Explicit basis leaves require basis and coefficients.")
        return dlu.eval_basis(basis, coefficients)

    if isinstance(leaf, FourierBasis):
        if basis is None or coefficients is None:
            raise ValueError("Fourier basis leaves require basis and coefficients.")
        return dlu.eval_fourier_basis(coefficients, *basis)

    return np.asarray(leaf, float)


class BasePolarisingOptic(OpticalLayer):
    """
    Base class for layers that apply a Jones matrix to a wavefront.

    Subclasses expose `jones` with shape `(2, 2, ...)`, matching the polarisation
    utility convention. The trailing axes may be empty for global optics or spatial
    for spatially varying optics.
    """

    jones: eqx.AbstractVar

    def __call__(self: PolarisingOptic, wavefront: Wavefront) -> Wavefront:
        """
        Applies the layer Jones matrix to the input wavefront.

        Parameters
        ----------
        wavefront : Wavefront
            Input wavefront. Non-polarised wavefronts are promoted by
            `Wavefront.apply_jones`.

        Returns
        -------
        wavefront : Wavefront
            Wavefront after applying the Jones matrix.
        """
        return wavefront.apply_jones(self.jones)


class PolarisingOptic(BasePolarisingOptic):
    """
    A polarising optic defined directly by a Jones matrix.

    Attributes
    ----------
    jones : Array
        Jones matrix with shape `(2, 2, ...)`.
    """

    jones: Array  # Concrete this as an array

    def __init__(self: PolarisingOptic, jones: Array):
        """
        Parameters
        ----------
        jones : Array
            Jones matrix with shape `(2, 2, ...)`.
        """
        self.jones = jones


class UniformPolarisingOptic(PolarisingOptic):
    """
    A spatially uniform Jones matrix optic.

    The input Jones matrix must have shape `(2, 2)`. If `orientation` is provided, the
    Jones matrix is rotated when the layer is applied.

    Attributes
    ----------
    jones : Array
        Spatially uniform Jones matrix with shape `(2, 2)`.
    orientation : Array or None
        Rotation angle in radians. If None, the Jones matrix is applied unchanged.
    """

    orientation: Array | None

    def __init__(
        self: UniformPolarisingOptic,
        jones: Array,
        orientation: Array | None = None,
    ):
        """
        Parameters
        ----------
        jones : Array
            Spatially uniform Jones matrix with shape `(2, 2)`.
        orientation : Array or None = None
            Rotation angle in radians.
        """
        self.orientation = orientation
        jones = np.asarray(jones)

        if jones.shape != (2, 2):
            raise ValueError("UniformPolarisingOptic requires a (2, 2) Jones matrix.")
        super().__init__(jones)

    def __call__(self: UniformPolarisingOptic, wavefront: Wavefront) -> Wavefront:
        """
        Applies the rotated Jones matrix to the input wavefront.

        Parameters
        ----------
        wavefront : Wavefront
            Input wavefront.

        Returns
        -------
        wavefront : Wavefront
            Wavefront after applying the rotated Jones matrix.
        """
        return wavefront.apply_jones(dlu.rotate_jones(self.jones, self.orientation))


class LinearPolariser(UniformPolarisingOptic):
    """
    A spatially uniform ideal linear polariser.

    `orientation` is the transmission-axis angle measured counter-clockwise from the
    horizontal x-axis.
    """

    def __init__(self: LinearPolariser, orientation: Array | None = None):
        """
        Parameters
        ----------
        orientation : Array or None = None
            Transmission-axis angle in radians.
        """
        super().__init__(dlu.linear_polariser(0.0), orientation)


class Retarder(UniformPolarisingOptic):
    """
    A spatially uniform retarder.

    `retardance` is the phase delay of the vertical component relative to horizontal.
    `orientation` rotates the retarder fast axis counter-clockwise from horizontal.
    """

    def __init__(
        self: Retarder,
        retardance: Array,
        orientation: Array | None = None,
    ):
        """
        Parameters
        ----------
        retardance : Array
            Retardance in radians.
        orientation : Array or None = None
            Fast-axis rotation angle in radians.
        """
        super().__init__(dlu.retarder(retardance, 0.0), orientation)


class SVLinearPolariser(BasePolarisingOptic):
    """
    A spatially varying ideal linear polariser.

    `angle` may be a scalar, array, or basis marker. The evaluated physical angle is
    exposed by `layer.angle`; the raw leaf, basis, and coefficients are available via
    `layer.parameters.angle`, `layer.basis.angle`, and `layer.coefficients.angle`.

    Attributes
    ----------
    parameters : FieldDict
        Raw parameter leaves.
    basis : FieldDict
        Basis data for basis-backed leaves.
    coefficients : FieldDict
        Coefficients for basis-backed leaves.
    """

    parameters: FieldDict
    basis: FieldDict
    coefficients: FieldDict

    def __init__(
        self: SVLinearPolariser,
        angle: Array,
        basis: dict = None,
        coefficients: dict = None,
    ):
        """
        Parameters
        ----------
        angle : Array
            Transmission-axis angle in radians, or a basis marker.
        basis : dict = None
            Basis data keyed by parameter name.
        coefficients : dict = None
            Basis coefficients keyed by parameter name.
        """
        self.parameters = FieldDict(angle=angle)
        self.basis = FieldDict(basis)
        self.coefficients = FieldDict(coefficients)

    @property
    def angle(self: SVLinearPolariser) -> Array:
        """
        Evaluated transmission-axis angle.

        Returns
        -------
        angle : Array
            Transmission-axis angle in radians.
        """
        return _eval_leaf(
            self.parameters.angle,
            self.basis.get("angle"),
            self.coefficients.get("angle"),
        )

    @property
    def jones(self: SVLinearPolariser) -> Array:
        """
        Returns the evaluated Jones matrix.

        Returns
        -------
        jones : Array
            Linear polariser Jones matrix with shape `(2, 2, ...)`.
        """
        return dlu.linear_polariser(self.angle)


class SVRetarder(BasePolarisingOptic):
    """
    A spatially varying retarder.

    Retardance and angle may be independently specified as scalars, arrays, or basis
    markers. Evaluated physical values are exposed by `layer.retardance` and
    `layer.angle`; raw leaves, basis data, and coefficients are stored in matching
    `FieldDict`s.

    Attributes
    ----------
    parameters : FieldDict
        Raw parameter leaves.
    basis : FieldDict
        Basis data for basis-backed leaves.
    coefficients : FieldDict
        Coefficients for basis-backed leaves.
    """

    parameters: FieldDict
    basis: FieldDict
    coefficients: FieldDict

    def __init__(
        self: SVRetarder,
        retardance: Array,
        angle: Array,
        basis: dict = None,
        coefficients: dict = None,
    ):
        """
        Parameters
        ----------
        retardance : Array
            Retardance in radians, or a basis marker.
        angle : Array
            Fast-axis angle in radians, or a basis marker.
        basis : dict = None
            Basis data keyed by parameter name.
        coefficients : dict = None
            Basis coefficients keyed by parameter name.
        """
        self.parameters = FieldDict(retardance=retardance, angle=angle)
        self.basis = FieldDict(basis)
        self.coefficients = FieldDict(coefficients)

    @property
    def retardance(self: SVRetarder) -> Array:
        """
        Evaluated retardance.

        Returns
        -------
        retardance : Array
            Retardance in radians.
        """
        return _eval_leaf(
            self.parameters.retardance,
            self.basis.get("retardance"),
            self.coefficients.get("retardance"),
        )

    @property
    def angle(self: SVRetarder) -> Array:
        """
        Evaluated fast-axis angle.

        Returns
        -------
        angle : Array
            Fast-axis angle in radians.
        """
        return _eval_leaf(
            self.parameters.angle,
            self.basis.get("angle"),
            self.coefficients.get("angle"),
        )

    @property
    def jones(self: SVRetarder) -> Array:
        """
        Returns the evaluated Jones matrix.

        Returns
        -------
        jones : Array
            Retarder Jones matrix with shape `(2, 2, ...)`.
        """
        return dlu.retarder(self.retardance, self.angle)
