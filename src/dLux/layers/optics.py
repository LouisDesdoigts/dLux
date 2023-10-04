from __future__ import annotations
from jax import Array


from ..wavefronts import Wavefront
from .optical_layers import (
    OpticalLayer,
    TransmissiveLayer,
    BasisLayer,
    AberratedLayer,
)


__all__ = [
    "Optic",
    "BasisOptic",
]


class Optic(TransmissiveLayer, AberratedLayer):
    """
    A basic 'Optic' class, which optionally applies a transmission, OPD and phase to
    the input wavefront, with the option for normalise after.

    ??? abstract "UML"
        ![UML](../../assets/uml/Optic.png)

    Attributes
    ----------
    transmission: Array
        The Array of transmission values to be applied to the input wavefront.
    opd : Array, metres
        The Array of OPD values to be applied to the input wavefront.
    phase : Array, radians
        The Array of phase values to be applied to the input wavefront.
    normalise: bool
        Whether to normalise the wavefront after passing through the optic.
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
            The Array of transmission values to be applied to the input wavefront.
        opd : Array, metres = None
            The Array of OPD values to be applied to the input wavefront.
        phase : Array, radians = None
            The Array of phase values to be applied to the input wavefront.
        normalise: bool = False
            Whether to normalise the wavefront after passing through the optic.
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
                        f"shapes {self.transmission.shape} and {self.opd.shape}."
                    )
            if self.phase is not None:
                if self.transmission.shape != self.phase.shape:
                    raise ValueError(
                        "transmission and phase must have the same shape. Got "
                        f"shapes {self.transmission.shape} and {self.phase.shape}."
                    )

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
        wavefront += self.opd
        wavefront = wavefront.add_phase(self.phase)
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront


class BasisOptic(TransmissiveLayer, BasisLayer):
    """
    A basic 'Optic' class, with a aberrations applied through a set of basis vectors
    coefficients. This can be applied either as an opd or phase, using the `as_phase`
    attribute. Also optionally applies a transmission and normalisation.

    ??? abstract "UML"
        ![UML](../../assets/uml/BasisOptic.png)

    Attributes
    ----------
    transmission: Array
        The Array of transmission values to be applied to the input wavefront.
    basis: Array, metres
        Arrays holding the pre-calculated basis vectors.
    coefficients: Array
        The Array of coefficients to be applied to each basis vector.
    as_phase : bool
        Whether to apply the basis as a phase phase or OPD. If True the basis is
        applied as a phase, else it is applied as an OPD.
    normalise : bool
        Whether to normalise the wavefront after passing through the optic.
    """

    def __init__(
        self: OpticalLayer,
        basis,
        transmission=None,
        coefficients=None,
        as_phase=False,
        normalise=False,
    ):
        """
        Parameters
        ----------
        basis: Array, metres
            Arrays holding the pre-calculated basis vectors.
        coefficients: Array = None
            The Array of coefficients to be applied to each basis vector.
        transmission: Array = None
            The Array of transmission values to be applied to the input wavefront.
        as_phase : bool = False
            Whether to apply the basis as a phase phase or OPD. If True the basis is
            applied as a phase, else it is applied as an OPD.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the optic.
        """
        super().__init__(
            transmission=transmission,
            basis=basis,
            coefficients=coefficients,
            as_phase=as_phase,
            normalise=normalise,
        )

    def apply(self: OpticalLayer, wavefront: Wavefront()) -> Wavefront():
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
            wavefront = wavefront.add_phase(self.eval_basis())
        else:
            wavefront += self.eval_basis()

        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront
