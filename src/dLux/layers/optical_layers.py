"""Core optical-layer abstractions and reusable optical-layer mixins."""

from __future__ import annotations
from abc import abstractmethod
from typing import Any
import jax.numpy as np
import zodiax as zdx
from jax import Array
import dLux.utils as dlu


from ..wavefronts import Wavefront

__all__ = [
    "BaseLayer",
    "OpticalLayer",
    "TransmissiveLayer",
    "AberratedLayer",
    "BasisLayer",
    "Tilt",
    "Normalise",
    "FourierBasis",
]


class BaseLayer(zdx.Base):
    """
    Abstract base class for all dLux layers.

    Layer objects define a callable transform interface that maps one target
    object to another (for example Wavefront -> Wavefront or PSF -> PSF).

    ??? abstract "UML"
        ![UML](../../assets/uml/BaseLayer.png)
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

    def __init_subclass__(cls, **kwargs):
        """Automatically inherit __call__ docstring from parent if child has none."""
        super().__init_subclass__(**kwargs)
        dlu.helpers.inherit_docstrings(cls, ["__call__"])


class OpticalLayer(BaseLayer):
    """
    The base optical layer class. Optical layer classes operate on `Wavefront` objects
    through their `apply` method, and are stored by the `OpticalSystem` classes.

    ??? abstract "UML"
        ![UML](../../assets/uml/OpticalLayer.png)
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
        self: TransmissiveLayer,
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

    def __call__(self: TransmissiveLayer, wavefront: Wavefront) -> Wavefront:
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
        self: AberratedLayer,
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

    def __call__(self: AberratedLayer, wavefront: Wavefront) -> Wavefront:
        wavefront = wavefront.add_opd(self.opd)
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
    basis: Array | list
        The set of basis vectors. Should in general be a 3 dimensional array.
    coefficients: Array
        The array of coefficients to be applied to each basis vector.
    as_phase: bool = False
        Whether to apply the basis as a phase or OPD. If True the output is applied as
        a phase, else it is applied as an OPD.
    """

    basis: Array | list
    coefficients: Array
    as_phase: bool

    # NOTE: We need the None basis input for aberrated apertures
    def __init__(
        self: BasisLayer,
        basis: Array = None,
        coefficients: Array = None,
        as_phase: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        basis: Array | list = None
            The set of basis vectors. Should in general be a 3 dimensional array.
        coefficients: Array = None
            The Array of coefficients to be applied to each basis vector. Defaults
            to zeros if `basis` is provided and `coefficients` is None.
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

    def eval_basis(self: BasisLayer) -> Array:
        """
        Calculates the dot product of the basis vectors and coefficients.

        Returns
        -------
        output : Array
            The output of the dot product between the basis vectors and coefficients.
        """
        return dlu.eval_basis(self.basis, self.coefficients)

    def __call__(self: BasisLayer, wavefront: Wavefront) -> Wavefront:
        output = self.eval_basis()
        if self.as_phase:
            wavefront = wavefront.add_phase(output)
        else:
            wavefront = wavefront.add_opd(output)
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
        ![UML](../../assets/uml/Normalise.png)
    """

    def __call__(self: Normalise, wavefront: Wavefront) -> Wavefront:
        return wavefront.normalise()


class FourierBasis(OpticalLayer):
    """
    Optical layer for representing an OPD using a 2D real Fourier basis.

    ??? abstract "UML"
        ![UML](../../assets/uml/FourierBasis.png)

    Attributes
    ----------
    coefficients : Array
        The Fourier coefficients, ordered in `(x, y)` mode order.
    kernels : tuple[Array, Array]
        The cached Fourier evaluation kernels for the x and y axes.
    """

    coefficients: Array
    kernels: tuple[Array, Array]

    def __init__(
        self: FourierBasis,
        npix: int | tuple[int, int],
        n_modes: int | tuple[int, int],
        coefficients: Array = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        npix : int | tuple[int, int]
            The output number of pixels in `(x, y)` order.
        n_modes : int | tuple[int, int]
            The number of Fourier modes in `(x, y)` order.
        coefficients : Array = None
            The Fourier coefficients. Defaults to zeros if not provided.
        """
        self.kernels = dlu.fourier_kernels(n_modes, npix)
        coefficient_shape = tuple(kernel.shape[1] for kernel in self.kernels)

        if coefficients is None:
            coefficients = np.zeros(coefficient_shape)
        else:
            coefficients = np.asarray(coefficients, dtype=float)
            if coefficients.shape != coefficient_shape:
                raise ValueError(
                    "The Fourier coefficient array must match the number of "
                    "modes in each dimension."
                )

        self.coefficients = coefficients
        super().__init__(**kwargs)

    def update_kernels(self: FourierBasis, npix: int | tuple[int, int]) -> FourierBasis:
        """
        Returns a copy of the layer with kernels updated for a new output size.

        Parameters
        ----------
        npix : int | tuple[int, int]
            The updated output number of pixels in `(x, y)` order.

        Returns
        -------
        layer : FourierBasis
            A copy of the layer with updated Fourier kernels.
        """
        kernels = dlu.fourier_kernels(self.coefficients.shape, npix)
        return self.set(kernels=kernels)

    def eval_basis(self: FourierBasis) -> Array:
        """
        Evaluates the Fourier basis represented by the current coefficients.

        Returns
        -------
        output : Array
            The evaluated Fourier basis.
        """
        return dlu.eval_fourier_basis(self.coefficients, *self.kernels)

    def __call__(self: FourierBasis, wavefront: Wavefront) -> Wavefront:
        """
        Applies the evaluated Fourier basis to the input wavefront as an OPD.

        Parameters
        ----------
        wavefront : Wavefront
            The input wavefront.

        Returns
        -------
        wavefront : Wavefront
            The wavefront with the Fourier basis applied as an OPD.
        """
        return wavefront.add_opd(self.eval_basis())
