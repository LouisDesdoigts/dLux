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
    def apply(self: BaseLayer, wavefront):  # pragma: no cover
        pass


class OpticalLayer(BaseLayer):
    """
    Base class for optical layers. Primarily used for input type checking.

    Child classes must implement the apply method that takes in the
    wavefront as the first parameter.

    Note: I have chosen apply over apply as the method name for the layer
    to be applied to the wavefront. This is because even though it prevents
    the simple interface with Optax (not having to wrap in a list), because
    wavefront should in general not be able to be an object you take a gradient
    with respect to, it is just a latent class to store information throughout
    the calculation, plus its use of strings as a way to track parameters can
    make interactions with jax difficult.
    """

    @abstractmethod
    def apply(
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


##################
# Public Classes #
##################
class TransmissiveLayer(OpticalLayer):
    """
    Base class to hold transmissive layers imbuing them with a transmission and
    normalise parameter.

    Attributes
    ----------
    transmission: Array
        The Array of transmission values to be applied to the input wavefront.
    normalise: bool
        Whether to normalise the wavefront after passing through the
        optic.
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
            The Array of transmission values to be applied to the input
            wavefront.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
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
    Base class for aberration layers. Implements the opd and phase attributes.

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
    This class primarily exists to allow for the use of the class based basis
    used for dynamic aberrated apertures.

    Attributes
    ----------
    basis: Union[Array, list]
        The basis to use. Can be an array of a list of aberrations classes.
    coefficients: Array
        The Array of coefficients to be applied to each basis vector.
    as_phase: bool = False
        Whether to apply the basis as a phase or OPD. If True the basis is
        applied as a phase, else it is applied as an OPD.
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
    ) -> OpticalLayer:
        """
        Parameters
        ----------
        basis: Union[Array, list]
            The basis to use. Can be an array of a list of aberrations classes.
        coefficients: Array
            The Array of coefficients to be applied to each basis vector.
        phase: bool = False
            Whether to apply the basis as a phase phase or OPD. If True the
            basis is applied as a phase, else it is applied as an OPD.
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
    Tilts the wavefront by the input angles.

    Attributes
    ----------
    angles : Array, radians
        The (x, y) angles by which to tilt the wavefront.
    """

    angles: Array

    def __init__(self: OpticalLayer, angles: Array) -> OpticalLayer:
        """
        Constructor for the TiltWavefront class.

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
    """Normalises the wavefront to unit intensity."""

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
