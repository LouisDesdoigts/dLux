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
    "Parameter",
    "Constant",
    "Basis",
    "FourierParameter",
    "PolarisingOptic",
    "UniformPolarisingOptic",
    "LinearPolariser",
    "Retarder",
    "SVLinearPolariser",
    "SVRetarder",
]


class Parameter(zdx.Base):
    """
    Base class for scalar or spatially varying values evaluated at runtime.

    Polarisation layers use parameters for values such as angle and retardance. This
    keeps the layers independent of how those values are represented: a parameter may
    be a scalar, an array, an explicit basis expansion, or an implicit basis expansion.
    """

    def __call__(self: Parameter) -> Array:
        """
        Evaluates the parameter.

        Returns
        -------
        value : Array
            Scalar or spatially varying parameter value.
        """
        raise NotImplementedError("Parameter subclasses must implement __call__.")


class Constant(Parameter):
    """
    A scalar or array-valued parameter.

    Attributes
    ----------
    value : Array
        The scalar or array value returned when the parameter is evaluated.
    """

    value: Array

    def __init__(self: Constant, value: Array):
        """
        Parameters
        ----------
        value : Array
            Scalar or array value.
        """
        self.value = np.asarray(value, float)

    def __call__(self: Constant) -> Array:
        """
        Returns the stored value.

        Returns
        -------
        value : Array
            Scalar or array value.
        """
        return self.value


class Basis(Parameter):
    """
    An explicit basis parameter evaluated by `dlu.eval_basis`.

    Attributes
    ----------
    basis : Array
        Explicit basis array with shape `(..., n, n)`.
    coefficients : Array
        Coefficients with shape matching `basis.shape[:-2]`.
    """

    basis: Array
    coefficients: Array

    def __init__(self: Basis, basis: Array, coefficients: Array = None):
        """
        Parameters
        ----------
        basis : Array
            Explicit basis array with shape `(..., n, n)`.
        coefficients : Array = None
            Coefficients for the basis. Defaults to zeros with shape
            `basis.shape[:-2]`.
        """
        self.basis = np.asarray(basis, float)
        if coefficients is None:
            coefficients = np.zeros(self.basis.shape[:-2])
        self.coefficients = np.asarray(coefficients, float)

        if self.basis.shape[:-2] != self.coefficients.shape:
            raise ValueError(
                "The number of basis vectors must match the number of coefficients."
            )

    def __call__(self: Basis) -> Array:
        """
        Evaluates the explicit basis.

        Returns
        -------
        value : Array
            Evaluated basis array with shape `(n, n)`.
        """
        return dlu.eval_basis(self.basis, self.coefficients)


class FourierParameter(Parameter):
    """
    A Fourier-basis parameter evaluated from cached Fourier kernels.

    This stores Fourier evaluation kernels rather than an explicit `(modes, n, n)`
    basis array. The coefficients use the same `(x, y)` mode ordering as
    `dlu.eval_fourier_basis`.

    Attributes
    ----------
    coefficients : Array
        Fourier coefficients with shape `(n_modes_x, n_modes_y)`.
    kernels : tuple[Array, Array]
        Cached x and y Fourier evaluation kernels.
    """

    coefficients: Array
    kernels: tuple[Array, Array]

    def __init__(
        self: FourierParameter,
        npix: int | tuple[int, int],
        n_modes: int | tuple[int, int],
        coefficients: Array = None,
        scale: float = 1.0,
    ):
        """
        Parameters
        ----------
        npix : int or tuple[int, int]
            Output number of pixels in `(x, y)` order.
        n_modes : int or tuple[int, int]
            Number of Fourier modes in `(x, y)` order.
        coefficients : Array = None
            Fourier coefficients. Defaults to zeros.
        scale : float = 1.0
            Per-axis Fourier kernel scale.
        """
        self.kernels = dlu.fourier_kernels(n_modes, npix, scale)
        coefficient_shape = tuple(kernel.shape[1] for kernel in self.kernels)

        if coefficients is None:
            coefficients = np.zeros(coefficient_shape)
        self.coefficients = np.asarray(coefficients, float)

        if self.coefficients.shape != coefficient_shape:
            raise ValueError(
                "The Fourier coefficient array must match the number of modes in each "
                "dimension."
            )

    def update_kernels(
        self: FourierParameter, npix: int | tuple[int, int]
    ) -> FourierParameter:
        """
        Returns a copy with kernels updated for a new output size.

        Parameters
        ----------
        npix : int or tuple[int, int]
            Updated output number of pixels in `(x, y)` order.

        Returns
        -------
        parameter : FourierParameter
            Copy with updated Fourier kernels.
        """
        kernels = dlu.fourier_kernels(self.coefficients.shape, npix)
        return self.set(kernels=kernels)

    def __call__(self: FourierParameter) -> Array:
        """
        Evaluates the Fourier parameter.

        Returns
        -------
        value : Array
            Evaluated Fourier basis array.
        """
        return dlu.eval_fourier_basis(self.coefficients, *self.kernels)


def _as_parameter(value: Array | Parameter) -> Parameter:
    """Promote scalar and array inputs to a common parameter interface."""
    if isinstance(value, Parameter):
        return value
    return Constant(value)


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

    The angle may be a scalar, array, or `Parameter`. It is evaluated when `jones` is
    accessed, allowing basis-backed parameters to be optimised directly.

    Attributes
    ----------
    angle : Parameter
        Transmission-axis angle in radians.
    """

    angle: Parameter

    def __init__(self: SVLinearPolariser, angle: Array):
        """
        Parameters
        ----------
        angle : Array or Parameter
            Transmission-axis angle in radians.
        """
        self.angle = _as_parameter(angle)

    @property
    def jones(self: SVLinearPolariser) -> Array:
        """
        Returns the evaluated Jones matrix.

        Returns
        -------
        jones : Array
            Linear polariser Jones matrix with shape `(2, 2, ...)`.
        """
        return dlu.linear_polariser(self.angle())


class SVRetarder(BasePolarisingOptic):
    """
    A spatially varying retarder.

    Retardance and angle may be independently specified as scalars, arrays, or
    `Parameter` objects.

    Attributes
    ----------
    retardance : Parameter
        Retardance in radians.
    angle : Parameter
        Fast-axis angle in radians.
    """

    retardance: Parameter
    angle: Parameter

    def __init__(self: SVRetarder, retardance: Array, angle: Array):
        """
        Parameters
        ----------
        retardance : Array or Parameter
            Retardance in radians.
        angle : Array or Parameter
            Fast-axis angle in radians.
        """
        self.retardance = _as_parameter(retardance)
        self.angle = _as_parameter(angle)

    @property
    def jones(self: SVRetarder) -> Array:
        """
        Returns the evaluated Jones matrix.

        Returns
        -------
        jones : Array
            Retarder Jones matrix with shape `(2, 2, ...)`.
        """
        return dlu.retarder(self.retardance(), self.angle())
