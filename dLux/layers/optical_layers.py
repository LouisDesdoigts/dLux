from __future__ import annotations
from abc import abstractmethod
from typing import Union
import jax.numpy as np
from jax import Array
import dLux
import dLux.utils as dlu


__all__ = [
    "Optic",
    "BasisOptic",
    "Tilt",
    "Normalise",
]


Wavefront = lambda: dLux.wavefronts.Wavefront


class OpticalLayer(dLux.base.BaseOpticalLayer):
    """
    Base class for optical layers. Primarily used for input type checking.

    Child classes must implement the __call__ method that takes in the
    wavefront as the first parameter.

    Note: I have chosen __call__ over apply as the method name for the layer
    to be applied to the wavefront. This is because even though it prevents
    the simple interface with Optax (not having to wrap in a list), because
    wavefront should in general not be able to be an object you take a gradient
    with respect to, it is just a latent class to store information throughout
    the calculation, plus its use of strings as a way to track parameters can
    make interactions with jax difficult.
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


# class AberratedLayer(OpticalLayer):
#     """Base class for aberration layers. Primarily used for type checking."""


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

        self.basis = np.asarray(basis, dtype=float)
        if coefficients is None:
            self.coefficients = np.zeros(self.basis.shape[:-2])
        else:
            self.coefficients = np.asarray(coefficients, dtype=float)
            if self.basis.shape[:-2] != self.coefficients.shape:
                raise ValueError(
                    "The number of basis vectors must be equal to "
                    "the number of coefficients."
                )
        self.as_phase = bool(as_phase)

    def calculate(self: OpticalLayer) -> Array:
        """
        Calculates the dot product of the basis vectors and coefficients.
        """
        return dlu.eval_basis(self.basis, self.coefficients)


##################
# Public Classes #
##################
class Optic(TransmissiveLayer, AberratedLayer):
    """
    Optics class that holds both a transmission and OPD array.

    Attributes
    ----------
    transmission: Array
        The Array of transmission values to be applied to the input wavefront.
    opd : Array, metres
        The Array of OPD values to be applied to the input wavefront.
    phase : Array, radians
        The Array of phase values to be applied to the input wavefront.
    normalise: bool
        Whether to normalise the wavefront after passing through the
        optic.
    """

    def __init__(
        self: OpticalLayer,
        transmission: Array = None,
        opd: Array = None,
        phase: Array = None,
        normalise: bool = False,
    ):
        """
        Parameters
        ----------
        transmission: Array = None
            The Array of transmission values to be applied to the input
            wavefront.
        opd : Array, metres = None
            The Array of OPD values to be applied to the input wavefront.
        phase : Array, radians = None
            The Array of phase values to be applied to the input wavefront.
        normalise: bool = False
            Whether to normalise the wavefront after passing through the
            optic.
        """
        super().__init__(
            transmission=transmission,
            opd=opd,
            phase=phase,
            normalise=normalise,
        )

        if self.transmission is not None:
            if self.opd is not None:
                if self.transmission.shape != self.opd.shape:
                    raise ValueError(
                        "transmission and opd must have the same shape. Got "
                        f"shapes {self.opd.shape} and {self.phase.shape}."
                    )
            if self.phase is not None:
                if self.transmission.shape != self.phase.shape:
                    raise ValueError(
                        "transmission and phase must have the same shape. Got "
                        f"shapes {self.opd.shape} and {self.phase.shape}."
                    )

    def __call__(self: OpticalLayer, wavefront: Wavefront) -> Wavefront:
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
        wavefront += self.opd
        wavefront = wavefront.add_phase(self.phase)
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront


class BasisOptic(TransmissiveLayer, BasisLayer):
    """
    Adds an array of phase values to the input wavefront calculated from the
    Optical Path Difference (OPD). The OPDs are calculated from the basis
    arrays, and weighted by the coefficients, and converted to phases by the
    wavefront methods.

    Attributes
    ----------
    transmission: Array
        The Array of transmission values to be applied to the input wavefront.
    basis: Array, metres
        Arrays holding the pre-calculated basis vectors.
    coefficients: Array
        The Array of coefficients to be applied to each basis vector.
    phase : bool
        Whether to apply the basis as a phase phase or OPD. If True the basis
        is applied as a phase, else it is applied as an OPD.
    normalise : bool
        Whether to normalise the wavefront after passing through the
        optic.
    """

    def __init__(self: OpticalLayer, **kwargs) -> OpticalLayer:
        """
        Parameters
        ----------
        transmission: Array
            The Array of transmission values to be applied to the input
            wavefront.
        basis: Array, metres
            Arrays holding the pre-calculated basis vectors.
        coefficients: Array
            The Array of coefficients to be applied to each basis vector.
        phase : bool
            Whether to apply the basis as a phase phase or OPD. If True the
            basis is applied as a phase, else it is applied as an OPD.
        normalise : bool
            Whether to normalise the wavefront after passing through the
            optic.
        """
        super().__init__(**kwargs)

    def __call__(self: OpticalLayer, wavefront: Wavefront()) -> Wavefront():
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

        if self.as_phase:
            wavefront = wavefront.add_phase(self.calculate)
        else:
            wavefront += self.calculate()

        if self.normalise:
            wavefront = wavefront.normalise()
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

    def __call__(self: OpticalLayer, wavefront: Wavefront) -> Wavefront:
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
    """Normalises the wavefront."""

    def __call__(self: OpticalLayer, wavefront: Wavefront) -> Wavefront:
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
