from __future__ import annotations
from abc import abstractmethod
from typing import Union
import jax.numpy as np
from zodiax import Base
from jax import Array
import dLux.utils as dlu


from ..wavefronts import Wavefront


__all__ = [
    "BaseLayer",
    "TransmissiveLayer",
    "AberratedLayer",
    "BasisLayer",
    "Tilt",
    "Normalise",
]


class BaseLayer(Base):
    @abstractmethod
    def apply(self, wavefront):  # pragma: no cover
        pass


class OpticalLayer(BaseLayer):
    """
    The base optical layer class. Optical layer classes operate on `Wavefront` objects
    though their `apply` method, and are stored by the `OpticalSystem` classes.
    """

    @abstractmethod
    def apply(self, wavefront) -> Wavefront:  # pragma: no cover
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
        ![UML](../../assets/uml/TransmissiveLayer.png)

    Attributes
    ----------
    transmission: Array
        The Array of transmission values to be applied to the input wavefront.
    normalise: bool
        Whether to normalise the wavefront after passing through the optic.
    """

    transmission: Array
    normalise: bool

    def __init__(
        self: OpticalLayer,
        transmission: Array = None,
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
        if transmission is not None:
            transmission = np.asarray(transmission, dtype=float)
        self.transmission = transmission
        self.normalise = bool(normalise)
        super().__init__(**kwargs)

    def apply(self: OpticalLayer, wavefront: Wavefront) -> Wavefront:
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
        wavefront *= self.transmission
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront


class AberratedLayer(OpticalLayer):
    """
    Optical layer for holding static aberrations. Aberrations can be applied as either
    a phase or OPD, or both.

    ??? abstract "UML"
        ![UML](../../assets/uml/AberratedLayer.png)

    Attributes
    ----------
    opd : Array, metres
        The Array of OPD values to be applied to the input wavefront.
    phase : Array, radians
        The Array of phase values to be applied to the input wavefront.
    """

    opd: Array
    phase: Array

    def __init__(
        self: OpticalLayer,
        opd: Array = None,
        phase: Array = None,
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
        if opd is not None:
            opd = np.asarray(opd, dtype=float)
        self.opd = opd

        if phase is not None:
            phase = np.asarray(phase, dtype=float)
        self.phase = phase

        if self.opd is not None and self.phase is not None:
            if self.opd.shape != self.phase.shape:
                raise ValueError(
                    "opd and phase must have the same shape. Got "
                    f"shapes {self.opd.shape} and {self.phase.shape}."
                )
        super().__init__(**kwargs)

    def apply(self: OpticalLayer, wavefront: Wavefront) -> Wavefront:
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
        wavefront += self.opd
        wavefront = wavefront.add_phase(self.phase)
        return wavefront


class BasisLayer(OpticalLayer):
    """
    An OpticalLayer class that holds a set of basis vectors and coefficients, which are
    dot-producted at run time to produce the output. The basis can be applied as either
    a phase or OPD, by setting the `as_phase` attribute.

    ??? abstract "UML"
        ![UML](../../assets/uml/BasisLayer.png)

    Attributes
    ----------
    basis: Union[Array, list]
        The set of basis vectors. Should in generate be a 3 dimensional array.
    coefficients: Array
        The array of coefficients to be applied to each basis vector.
    as_phase: bool = False
        Whether to apply the basis as a phase or OPD. If True the output is applied as
        a phase, else it is applied as an OPD.
    """

    basis: Union[Array, list]
    coefficients: Array
    as_phase: bool

    # NOTE: We need the None basis input for aberrated apertures
    def __init__(
        self: OpticalLayer,
        basis: Array = None,
        coefficients: Array = None,
        as_phase: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        basis: Union[Array, list]
            The set of basis vectors. Should in generate be a 3 dimensional array.
        coefficients: Array
            The Array of coefficients to be applied to each basis vector.
        as_phase: bool = False
            Whether to apply the basis as a phase or OPD. If True the output is applied
            as a phase, else it is applied as an OPD.
        """
        super().__init__(**kwargs)

        if basis is not None:
            basis = np.asarray(basis, dtype=float)
            if coefficients is None:
                coefficients = np.zeros(basis.shape[:-2])
            else:
                coefficients = np.asarray(coefficients, dtype=float)
                if basis.shape[:-2] != coefficients.shape:
                    raise ValueError(
                        "The number of basis vectors must be equal to "
                        "the number of coefficients."
                    )

        self.basis = basis
        self.coefficients = coefficients
        self.as_phase = bool(as_phase)

    def eval_basis(self: OpticalLayer) -> Array:
        """
        Calculates the dot product of the basis vectors and coefficients.

        Returns
        -------
        output : Array
            The output of the dot product between the basis vectors and coefficients.
        """
        return dlu.eval_basis(self.basis, self.coefficients)

    def apply(self: OpticalLayer, wavefront: Wavefront) -> Wavefront:
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
        output = self.eval_basis()
        if self.as_phase:
            wavefront = wavefront.add_phase(output)
        else:
            wavefront += output
        return wavefront


class Tilt(OpticalLayer):
    """
    Tilts the wavefront by the input (x, y) angles.

    ??? abstract "UML"
        ![UML](../../assets/uml/Tilt.png)

    Attributes
    ----------
    angles : Array, radians
        The (x, y) angles by which to tilt the wavefront.
    """

    angles: Array

    def __init__(self: OpticalLayer, angles: Array):
        """
        Parameters
        ----------
        angles : Array, radians
            The (x, y) angles by which to tilt the wavefront.
        """
        super().__init__()
        self.angles = np.asarray(angles, dtype=float)

        if self.angles.shape != (2,):
            raise ValueError("angles must have have (2,)")

    def apply(self: OpticalLayer, wavefront: Wavefront) -> Wavefront:
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
        return wavefront.tilt(self.angles)


class Normalise(OpticalLayer):
    """
    Normalises the wavefront to unit intensity.

    ??? abstract "UML"
        ![UML](../../assets/uml/Normalise.png)
    """

    def apply(self: OpticalLayer, wavefront: Wavefront) -> Wavefront:
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
        return wavefront.normalise()
