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

    def __init__(
        self: OpticalLayer,
        basis,
        transmission=None,
        coefficients=None,
        as_phase=False,
        normalise=False,
    ) -> OpticalLayer:
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
