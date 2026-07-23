"""Composite optical layers built from transmission and aberration components."""

from __future__ import annotations
from jax import Array


from ..wavefronts import Wavefront
from .optical_layers import TransmissiveLayer, BasisLayer, AberratedLayer
from ..parametric import Parametric

__all__ = [
    "Optic",
    "BasisOptic",
]


class Optic(TransmissiveLayer, AberratedLayer):
    """
    A basic 'Optic' class, which optionally applies a transmission, OPD and phase to
    the input wavefront, with optional normalisation afterward.

    ??? abstract "UML"
        ![UML](../assets/uml/Optic.png)

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
        self: Optic,
        transmission: Array | Parametric = None,
        opd: Array | Parametric = None,
        phase: Array | Parametric = None,
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

        if isinstance(self.transmission, Array):
            if isinstance(self.opd, Array):
                if self.transmission.shape != self.opd.shape:
                    raise ValueError(
                        "transmission and opd must have the same shape. Got "
                        f"shapes {self.transmission.shape} and {self.opd.shape}."
                    )
            if isinstance(self.phase, Array):
                if self.transmission.shape != self.phase.shape:
                    raise ValueError(
                        "transmission and phase must have the same shape. Got "
                        f"shapes {self.transmission.shape} and {self.phase.shape}."
                    )

    def __call__(self: Optic, wavefront: Wavefront) -> Wavefront:
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
        wavefront *= self.resolve(self.transmission, wavefront=wavefront)
        wavefront = wavefront.add_opd(self.resolve(self.opd, wavefront=wavefront))
        wavefront = wavefront.add_phase(self.resolve(self.phase, wavefront=wavefront))
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront


class BasisOptic(TransmissiveLayer, BasisLayer):
    """
    A basic 'BasisOptic' class, with aberrations applied through a set of basis-vector
    coefficients. The evaluated basis is applied according to ``effect``. Also
    optionally applies a transmission and normalisation.

    ??? abstract "UML"
        ![UML](../assets/uml/BasisOptic.png)

    Attributes
    ----------
    transmission: Array
        The Array of transmission values to be applied to the input wavefront.
    basis: Array, metres
        Arrays holding the pre-calculated basis vectors.
    coefficients: Array
        The Array of coefficients to be applied to each basis vector.
    effect : str
        How to apply the basis: ``"opd"``, ``"phase"``, or ``"amplitude"``.
    normalise : bool
        Whether to normalise the wavefront after passing through the optic.
    """

    def __init__(
        self: BasisOptic,
        basis: Array,
        transmission: Array | Parametric = None,
        coefficients: Array = None,
        normalise: bool = False,
        effect: str = "opd",
        coefficient_shape: tuple[int, ...] = None,
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
        effect : str = "opd"
            How to apply the basis: ``"opd"``, ``"phase"``, or ``"amplitude"``.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the optic.
        """
        super().__init__(
            transmission=transmission,
            basis=basis,
            coefficients=coefficients,
            normalise=normalise,
            effect=effect,
            coefficient_shape=coefficient_shape,
        )

    def __call__(self: BasisOptic, wavefront: Wavefront) -> Wavefront:
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
        transmission = self.resolve(self.transmission, wavefront=wavefront)
        wavefront *= transmission

        wavefront = BasisLayer.__call__(self, wavefront)

        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront
