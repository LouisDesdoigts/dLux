"""Core optical-layer abstractions and reusable optical-layer mixins."""

from __future__ import annotations
from abc import abstractmethod
from typing import Any
import jax.numpy as np
import equinox as eqx
import zodiax as zdx
from jax import Array
import dLux.utils as dlu


from ..wavefronts import Wavefront
from ..parametric import Parametric

__all__ = [
    "BaseLayer",
    "OpticalLayer",
    "TransmissiveLayer",
    "AberratedLayer",
    "BasisLayer",
    "Tilt",
    "Normalise",
]


class BaseLayer(zdx.Base):
    """
    Abstract base class for all dLux layers.

    Layer objects define a callable transform interface that maps one target
    object to another (for example Wavefront -> Wavefront or PSF -> PSF).

    ??? abstract "UML"
        ![UML](../assets/uml/BaseLayer.png)
    """

    @abstractmethod
    def __call__(self: BaseLayer, target: Any) -> Any:  # pragma: no cover
        pass

    def apply(self: BaseLayer, target: Any) -> Any:
        """
        Backwards compatibility alias for `__call__`.

        Parameters
        ----------
        target : Any
            The object to operate on.

        Returns
        -------
        result : Any
            The transformed object.
        """
        return self(target)

    @staticmethod
    def resolve(value: Any, **kwargs: Any) -> Any:
        """Evaluate a parametric leaf, or return an ordinary value unchanged."""
        if isinstance(value, Parametric):
            return value.evaluate(**kwargs)
        return value

    @staticmethod
    def as_parametric(value: Any, dtype: Any = float) -> Any:
        """Preserve parametric leaves and convert ordinary values to arrays."""
        if value is None or isinstance(value, Parametric):
            return value
        return np.asarray(value, dtype=dtype)

    def __init_subclass__(cls, **kwargs):
        """Automatically inherit __call__ docstring from parent if child has none."""
        super().__init_subclass__(**kwargs)
        dlu.helpers.inherit_docstrings(cls, ["__call__"])


class OpticalLayer(BaseLayer):
    """
    The base optical layer class. Optical layer classes operate on `Wavefront` objects
    through their `apply` method, and are stored by the `OpticalSystem` classes.

    ??? abstract "UML"
        ![UML](../assets/uml/OpticalLayer.png)
    """

    @abstractmethod
    def __call__(
        self: OpticalLayer, wavefront: Wavefront
    ) -> Wavefront:  # pragma: no cover
        """
        Applies the layer to the wavefront.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to operate on.

        Returns
        -------
        wavefront : Wavefront
            The transformed wavefront.
        """


class TransmissiveLayer(OpticalLayer):
    """
    Base class to hold transmissive layers imbuing them with a transmission and
    normalise parameter.

    ??? abstract "UML"
        ![UML](../assets/uml/TransmissiveLayer.png)

    Attributes
    ----------
    transmission: Array | Parametric
        The Array of transmission values to be applied to the input wavefront.
    normalise: bool
        Whether to normalise the wavefront after passing through the optic.
    """

    transmission: Array
    normalise: bool

    def __init__(
        self: TransmissiveLayer,
        transmission: Array | Parametric = None,
        normalise: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        transmission: Array = None
            The array of transmission values to be applied to the input wavefront.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the optic.
        """
        self.transmission = self.as_parametric(transmission)
        self.normalise = bool(normalise)
        super().__init__(**kwargs)

    def __call__(self: TransmissiveLayer, wavefront: Wavefront) -> Wavefront:
        transmission = self.resolve(self.transmission, wavefront=wavefront)
        wavefront *= transmission
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront


class AberratedLayer(OpticalLayer):
    """
    Optical layer for holding static aberrations. Aberrations can be applied as either
    a phase or OPD, or both.

    ??? abstract "UML"
        ![UML](../assets/uml/AberratedLayer.png)

    Attributes
    ----------
    opd : Array, metres
        The Array of OPD values to be applied to the input wavefront.
    phase : Array, radians
        The Array of phase values to be applied to the input wavefront.
    """

    opd: Array | Parametric
    phase: Array | Parametric

    def __init__(
        self: AberratedLayer,
        opd: Array | Parametric = None,
        phase: Array | Parametric = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        opd : Array, metres = None
            The Array of OPD values to be applied to the input wavefront.
        phase : Array, radians = None
            The Array of phase values to be applied to the input wavefront.
        """
        self.opd = self.as_parametric(opd)
        self.phase = self.as_parametric(phase)

        if isinstance(self.opd, Array) and isinstance(self.phase, Array):
            if self.opd.shape != self.phase.shape:
                raise ValueError(
                    "opd and phase must have the same shape. Got "
                    f"shapes {self.opd.shape} and {self.phase.shape}."
                )
        super().__init__(**kwargs)

    def __call__(self: AberratedLayer, wavefront: Wavefront) -> Wavefront:
        wavefront = wavefront.add_opd(self.resolve(self.opd, wavefront=wavefront))
        wavefront = wavefront.add_phase(self.resolve(self.phase, wavefront=wavefront))
        return wavefront


class BasisLayer(OpticalLayer):
    """
    An OpticalLayer class that holds a set of basis vectors and coefficients, which are
    dot-producted at run time to produce the output. The basis can be applied as either
    an OPD, phase, or amplitude transmission according to ``effect``.

    ??? abstract "UML"
        ![UML](../assets/uml/BasisLayer.png)

    Attributes
    ----------
    basis: Array
        The object that evaluates coefficients into an array.
    coefficients: Array
        The array of coefficients to be applied to each basis vector.
    effect: str = "opd"
        How to apply the evaluated basis: ``"opd"``, ``"phase"``, or ``"amplitude"``.
    """

    basis: Array
    coefficients: Array
    effect: str = eqx.field(static=True)

    # NOTE: We need the None basis input for aberrated apertures
    def __init__(
        self: BasisLayer,
        basis: Array = None,
        coefficients: Array = None,
        effect: str = "opd",
        coefficient_shape: tuple[int, ...] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        basis: Array = None
            The explicit basis vectors.
        coefficients: Array = None
            The Array of coefficients to be applied to each basis vector. Defaults
            to zeros if `basis` is provided and `coefficients` is None.
        effect: str = "opd"
            How to apply the basis: ``"opd"``, ``"phase"``, or ``"amplitude"``.
        """
        super().__init__(**kwargs)

        if basis is not None:
            basis = np.asarray(basis, dtype=float)
            if coefficients is None:
                if coefficient_shape is None:
                    coefficient_shape = basis.shape[:-2]
                coefficients = np.zeros(coefficient_shape)
            else:
                coefficients = np.asarray(coefficients, dtype=float)
                coefficient_shape = coefficients.shape
            if basis.shape[: len(coefficient_shape)] != coefficient_shape:
                raise ValueError(
                    "The coefficient shape must match the leading basis dimensions."
                )

        self.basis = basis
        self.coefficients = coefficients
        if effect not in ("opd", "phase", "amplitude"):
            raise ValueError("effect must be 'opd', 'phase', or 'amplitude'.")
        self.effect = effect

    def eval_basis(self: BasisLayer) -> Array:
        """
        Calculates the dot product of the basis vectors and coefficients.

        Returns
        -------
        output : Array
            The output of the dot product between the basis vectors and coefficients.
        """
        return dlu.eval_basis(self.basis, self.coefficients)

    def solve_basis(self: BasisLayer, value: Array) -> Array:
        """Solve for coefficients representing ``value`` over this layer's basis."""
        return dlu.solve_basis(value, self.basis)

    def __call__(self: BasisLayer, wavefront: Wavefront) -> Wavefront:
        output = self.eval_basis()
        if self.effect == "phase":
            wavefront = wavefront.add_phase(output)
        elif self.effect == "opd":
            wavefront = wavefront.add_opd(output)
        else:
            wavefront *= output
        return wavefront


class Tilt(OpticalLayer):
    """
    Tilts the wavefront by the input (x, y) angles.

    ??? abstract "UML"
        ![UML](../assets/uml/Tilt.png)

    Attributes
    ----------
    angles : Array, radians
        The (x, y) angles by which to tilt the wavefront.
    """

    angles: Array

    def __init__(self: Tilt, angles: Array):
        """
        Parameters
        ----------
        angles : Array, radians
            The (x, y) angles by which to tilt the wavefront.
        """
        super().__init__()
        self.angles = np.asarray(angles, dtype=float)

        if self.angles.shape != (2,):
            raise ValueError("angles must be a 1d array of shape (2,).")

    def __call__(self: Tilt, wavefront: Wavefront) -> Wavefront:
        return wavefront.tilt(self.angles)


class Normalise(OpticalLayer):
    """
    Normalises the wavefront to unit intensity.

    ??? abstract "UML"
        ![UML](../assets/uml/Normalise.png)
    """

    def __call__(self: Normalise, wavefront: Wavefront) -> Wavefront:
        return wavefront.normalise()
