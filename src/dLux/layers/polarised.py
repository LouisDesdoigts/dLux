from __future__ import annotations
import equinox as eqx
import dLux.utils as dlu
from jax import Array


from .optical_layers import BaseLayer, OpticalLayer
from ..wavefronts import Wavefront

__all__ = [
    "PolarisingOptic",
    "UniformPolarisingOptic",
    "LinearPolariser",
    "Retarder",
    "SVLinearPolariser",
    "SVRetarder",
]


class BasePolarisingOptic(OpticalLayer):
    jones: eqx.AbstractVar

    def __call__(self: PolarisingOptic, wavefront: Wavefront) -> Wavefront:
        return wavefront.apply_jones(self.jones)


class PolarisingOptic(BasePolarisingOptic):
    """
    A basic 'PolarisingOptic' class, which applies a polarisation transformation to the
    input wavefront.
    """

    jones: Array  # Concrete this as an array

    def __init__(self: PolarisingOptic, jones: Array):
        self.jones = jones


# Merge this with PolarisingOptic?
# NO - The top one is generic spatially varying
class UniformPolarisingOptic(PolarisingOptic):
    """
    A spatially uniform Jones matrix optic, which applies the same polarisation
    transformation across the entire wavefront. As such optics can be easily rotated,
    they also support an optional 'orientation' parameter, which rotates the Jones
    matrix by the specified orientation before applying it to the wavefront.
    """

    orientation: Array | None

    def __init__(
        self: UniformPolarisingOptic,
        jones: Array,
        orientation: Array | None = None,
    ):
        self.orientation = orientation

        if jones.shape != (2, 2):
            raise ValueError("UniformPolarisingOptic requires a (2, 2) Jones matrix.")
        super().__init__(jones)

    def __call__(self: UniformPolarisingOptic, wavefront: Wavefront) -> Wavefront:
        return wavefront.apply_jones(dlu.rotate_jones(self.jones, self.orientation))


class LinearPolariser(UniformPolarisingOptic):
    """
    A linear polariser, which can be oriented at any orientation. The Jones matrix for a
    linear polariser is given by:

    [[cos^2(theta), cos(theta)sin(theta)],
     [cos(theta)sin(theta), sin^2(theta)]]

    where theta is the orientation of the polariser's transmission axis relative to the
    horizontal.
    """

    def __init__(self: LinearPolariser, orientation: Array | None = None):
        super().__init__(dlu.linear_polariser(0.0), orientation)


class Retarder(UniformPolarisingOptic):
    """
    A retarder, which can be oriented at any orientation. The Jones matrix for a
    retarder is given by:

    [[1, 0],
     [0, exp(i * delta)]]

    where delta is the retardance of the retarder. The fast axis of the retarder is
    assumed to be horizontal, and the Jones matrix can be rotated to any orientation
    using the 'orientation' parameter.
    """

    def __init__(
        self: Retarder,
        retardance: Array,
        orientation: Array | None = None,
    ):
        super().__init__(dlu.retarder(retardance, 0.0), orientation)


### Spatially varying polarising optics ###
class SVLinearPolariser(BasePolarisingOptic):
    angle: Array

    def __init__(self: SVLinearPolariser, angle: Array):
        self.angle = angle

    @property
    def jones(self: SVLinearPolariser) -> Array:
        return dlu.linear_polariser(self.angle)


class SVRetarder(BasePolarisingOptic):
    retardance: Array
    angle: Array

    def __init__(self: SVRetarder, retardance: Array, angle: Array):
        self.retardance = retardance
        self.angle = angle

    @property
    def jones(self: SVRetarder) -> Array:
        return dlu.retarder(self.retardance, self.angle)


class MultiBasis(BaseLayer):
    basis: dict[Array]
    coefficients: dict[Array]

    # A dictionary of basis vectors and coefficients that correspond to each parameter
    # of the model. Produces an output dictionary of the evaluated basis and
    # coefficients

    def __init__(self: MultiBasis, basis: dict[Array], coefficients: dict[Array]):
        self.basis = basis
        self.coefficients = coefficients

    def __getitem__(self: MultiBasis, key: str) -> Array:
        """
        Provides access to a specific evaluated basis by its key via:

        ```python
        full_basis = MultiBasis(basis_dict, coeff_dict)
        output_a = full_basis['a']  # Accesses the evaluated basis for key 'a'
        ```

        """
        return dlu.eval_basis(self.basis[key], self.coefficients[key])

    def __getattr__(self: MultiBasis, key: str) -> Array:
        """
        Provides access to a specific evaluated basis by its key via:

        ```python
        full_basis = MultiBasis(basis_dict, coeff_dict)
        output_a = full_basis.a  # Accesses the evaluated basis for key 'a'
        ```

        """
        return self.basis[key]

    def __call__(self: MultiBasis) -> dict[Array]:
        pass


# TODO: Think more about implicit (fourier) basis vs explicit basis (holding an array)


class Basis(BaseLayer):

    def eval_basis(self):
        raise NotImplementedError("eval_basis must be implemented in subclasses.")


class ExplicitBasis(Basis):
    basis: Array
    coefficients: Array

    def eval_basis(self):
        return dlu.eval_basis(self.basis, self.coefficients)


class ImplicitBasis(Basis):
    coefficients: Array

    def eval_basis(self):
        pass


class FourierBasis(ImplicitBasis):

    def eval_basis(self):
        # Implement the evaluation of the Fourier basis functions here
        pass
