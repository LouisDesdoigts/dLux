from __future__ import annotations
from abc import abstractmethod
from typing import Union
import jax.numpy as np
from jax import Array
from zodiax import Base
import dLux


__all__ = [
    # OpticalLayer
    # TransmissiveLayer
    # AberratedLayer
    # ShapedLayer
    # BaseTransmissiveOptic
    # BaseOPDOptic
    # BasePhaseOptic
    # BaseBasisOptic
    "Optic",
    "PhaseOptic",
    "BasisOptic",
    "PhaseBasisOptic",
    "Tilt",
    "Normalise",
    "Rotate",
    "Flip",
    "Resize",
]


Wavefront = lambda: dLux.wavefronts.Wavefront


class OpticalLayer(Base):
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
    Base class to hold transmissive layers imbuing them with a normalise
    parameter.
    """

    normalise: bool

    def __init__(self: OpticalLayer, normalise: bool = False, **kwargs):
        """
        Parameters
        ----------
        normalise : bool = False
            Whether to normalise the wavefront after passing through the
            aperture.
        """
        self.normalise = bool(normalise)
        super().__init__(**kwargs)


class AberratedLayer(OpticalLayer):
    """
    Base class for aberration layers. Primarily used for input type checking.
    """


class ShapedLayer(OpticalLayer):
    """
    Base class used for layers that have a specified output shape.
    """

    @abstractmethod
    def applied_shape(self: OpticalLayer) -> int:
        """
        Returns the 'shape' of the layer, more specifically the required
        matching shape of the wavefront to be applied to.

        Returns
        -------
        shape : int
            The linear shape of the wavefront to be applied to.
        """


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
    """

    basis: Union[Array, list]
    coefficients: Array

    def __init__(
        self: OpticalLayer,
        basis: Array = None,
        coefficients: Array = None,
        **kwargs,
    ) -> OpticalLayer:
        """
        Parameters
        ----------
        basis: Union[Array, list]
            The basis to use. Can be an array of a list of aberrations classes.
        coefficients: Array
            The Array of coefficients to be applied to each basis vector.
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

    def calculate(
        self: OpticalLayer, basis: Array, coefficients: Array
    ) -> Array:
        """
        Performs an n-dimensional dot-product between the basis and
        coefficients arrays.

        Parameters
        ----------
        basis: Array
            The basis to use.
        coefficients: Array
            The Array of coefficients to be applied to each basis vector.
        """
        ndim = coefficients.ndim
        axes = (tuple(range(ndim)), tuple(range(ndim)))
        return np.tensordot(basis, coefficients, axes=axes)


class BaseTransmissiveOptic(TransmissiveLayer, ShapedLayer):
    """
    Base class for transmissive optics. Implements the transmission attribute
    and the `applied_shape` method.

    Attributes
    ----------
    transmission: Array
        The Array of transmission values to be applied to the input wavefront.
    """

    transmission: Array

    def __init__(
        self: OpticalLayer, transmission: Array, **kwargs
    ) -> OpticalLayer:
        """
        Parameters
        ----------
        transmission: Array
            The Array of transmission values to be applied to the input
            wavefront.
        """
        if transmission is not None:
            transmission = np.asarray(transmission, dtype=float)
        self.transmission = transmission
        super().__init__(**kwargs)

    @property
    def applied_shape(self: OpticalLayer) -> int:
        """
        Returns the 'shape' of the layer, more specifically the required
        matching shape of the wavefront to be applied to.

        Returns
        -------
        shape : int
            The linear shape of the wavefront to be applied to.
        """
        return self.transmission.shape


class BaseOPDOptic(AberratedLayer, ShapedLayer):
    """
    Base class for OPD optics. Implements the opd attribute.

    Attributes
    ----------
    opd : Array, metres
        The Array of OPD values to be applied to the input wavefront.
    """

    opd: Array

    def __init__(self: OpticalLayer, opd: Array, **kwargs) -> OpticalLayer:
        """
        Parameters
        ----------
        opd : Array, metres
            The Array of OPD values to be applied to the input wavefront.
        """
        if opd is not None:
            opd = np.asarray(opd, dtype=float)
        self.opd = opd
        super().__init__(**kwargs)


class BasePhaseOptic(AberratedLayer, ShapedLayer):
    """
    Base class for phase optics. Implements the phase attribute.

    Attributes
    ----------
    phase : Array, radians
        The Array of phase values to be applied to the input wavefront.
    """

    phase: Array

    def __init__(self: OpticalLayer, phase: Array, **kwargs) -> OpticalLayer:
        """
        Parameters
        ----------
        phase : Array, radians
            The Array of phase values to be applied to the input wavefront.
        """
        if phase is not None:
            phase = np.asarray(phase, dtype=float)
        self.phase = phase
        super().__init__(**kwargs)


class BaseBasisOptic(BaseTransmissiveOptic, BasisLayer, ShapedLayer):
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
    normalise: bool
        Whether to normalise the wavefront after passing through the
        optic.
    """

    def __init__(
        self: OpticalLayer,
        basis: Array,
        transmission: Array = None,
        coefficients: Array = None,
        normalise: bool = False,
    ) -> OpticalLayer:
        """
        Parameters
        ----------
        basis: Array, metres
            Arrays holding the pre-calculated basis vectors.
        transmission: Array = None
            The Array of transmission values to be applied to the input
            wavefront.
        coefficients: Array = None
            The Array of coefficients to be applied to each basis vector.
        normalise: bool = False
            Whether to normalise the wavefront after passing through the
        """
        super().__init__(
            transmission=transmission,
            basis=basis,
            coefficients=coefficients,
            normalise=normalise,
        )

    @property
    def applied_shape(self: OpticalLayer) -> int:
        """
        Returns the 'shape' of the layer, more specifically the required
        matching shape of the wavefront to be applied to.

        Returns
        -------
        shape : int
            The linear shape of the wavefront to be applied to.
        """
        return self.basis.shape[-2:]


##################
# Public Classes #
##################
class Optic(BaseTransmissiveOptic, BaseOPDOptic):
    """
    Optics class that holds both a transmission and OPD array.

    Attributes
    ----------
    transmission: Array
        The Array of transmission values to be applied to the input wavefront.
    opd : Array, metres
        The Array of OPD values to be applied to the input wavefront.
    normalise: bool
        Whether to normalise the wavefront after passing through the
        optic.
    """

    def __init__(
        self: OpticalLayer,
        transmission: Array = None,
        opd: Array = None,
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
        normalise: bool = False
            Whether to normalise the wavefront after passing through the
            optic.
        """
        super().__init__(
            transmission=transmission, opd=opd, normalise=normalise
        )

        if self.opd is not None and self.transmission is not None:
            if opd.shape != self.transmission.shape:
                raise ValueError(
                    "opd and transmission must have the same " "shape."
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
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront


class BasisOptic(BaseBasisOptic):
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
    """

    @property
    def opd(self: OpticalLayer) -> Array:
        """
        Calculates the total opd from the basis vectors and coefficients.

        Returns
        -------
        opd : Array, metres
            The total opd.
        """
        return self.calculate(self.basis, self.coefficients)

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
        wavefront += self.opd
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront


class PhaseOptic(BaseTransmissiveOptic, BasePhaseOptic):
    """
    Optics class that holds both a transmission and phase array.

    Attributes
    ----------
    transmission: Array
        The Array of transmission values to be applied to the input wavefront.
    phase : Array, radians
        The Array of phase values to be applied to the input wavefront.
    normalise: bool
        Whether to normalise the wavefront after passing through the
        optic.
    """

    def __init__(
        self: OpticalLayer,
        transmission: Array = None,
        phase: Array = None,
        normalise: bool = False,
    ) -> OpticalLayer:
        """
        Parameters
        ----------
        transmission: Array = None
            The Array of transmission values to be applied to the input
            wavefront.
        phase : Array, radians = None
            The Array of phase values to be applied to the input wavefront.
        normalise: bool = False
            Whether to normalise the wavefront after passing through the
            optic.
        """
        super().__init__(
            transmission=transmission, phase=phase, normalise=normalise
        )

        if self.phase is not None and self.transmission is not None:
            if phase.shape != self.transmission.shape:
                raise ValueError(
                    "phase and transmission must have the same " "shape."
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
        wavefront = wavefront.add_phase(self.phase)
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront


class PhaseBasisOptic(BaseBasisOptic):
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
    """

    @property
    def phase(self: OpticalLayer) -> Array:
        """
        Calculates the total phase from the basis vectors and coefficients.

        Returns
        -------
        phase : Array, radians
            The total phase.
        """
        return self.calculate(self.basis, self.coefficients)

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
        wavefront = wavefront.add_phase(self.phase)
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


class Rotate(OpticalLayer):
    """
    Applies a rotation to the wavefront using interpolation methods.

    Attributes
    ----------
    angle : Array, radians
        The angle by which to rotate the wavefront in the clockwise direction.
    order : int = 1
        The order of the interpolation to use. Only applies if fourier is
        False. Must be 0, 1, or 3.
    complex : bool = False
        Should the rotation be performed on the 'complex' (real, imaginary),
        as opposed to the default 'phasor' (amplitude, phase) arrays.
    """

    angle: Array
    order: int
    complex: bool

    def __init__(
        self: OpticalLayer,
        angle: Array,
        order: int = 1,
        complex: bool = False,
    ):
        """
        Constructor for the Rotate class.

        Parameters
        ----------
        angle: float, radians
            The angle by which to rotate the wavefront in the clockwise
            direction.
        order : int = 1
            The order of the interpolation to use. Must be 0, or 1.
        complex : bool = False
            Should the rotation be performed on the 'complex' (real,
            imaginary), as opposed to the default 'phasor' (amplitude, phase)
            arrays.
        """
        super().__init__()
        self.angle = np.asarray(angle, dtype=float)
        self.order = int(order)
        self.complex = bool(complex)

        if self.order not in (0, 1):
            raise ValueError("Order must be 0, 1")
        if self.angle.ndim != 0:
            raise ValueError(
                f"angle must be a zero-dimensional, has "
                f"{self.angle.ndim} dimensions."
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
        return wavefront.rotate(self.angle, self.order, self.complex)


class Flip(OpticalLayer):
    """
    Flips the wavefront about the input axes. Can be either an int, or a tuple
    of ints. This class uses the 'ij' indexing convention, ie axis 0 is the
    y-axis, and axis 1 is the x-axis.

    Attributes
    ----------
    axes : Union[tuple, int]
        The axes to flip the wavefront about. This class uses the 'ij' indexing
        convention, ie axis 0 is the y-axis, and axis 1 is the x-axis.
    """

    axes: Union[tuple[int], int]

    def __init__(self: OpticalLayer, axes: Union[tuple[int], int]):
        """
        Constructor for the Flip class.

        Parameters
        ----------
        axes : Union[tuple[int], int]
            The axes to flip the wavefront about. This class uses the 'ij'
            indexing convention, ie axis 0 is the y-axis, and axis 1 is the
            x-axis.
        """
        super().__init__()
        self.axes = axes

        if isinstance(self.axes, tuple):
            for axes in self.axes:
                if not isinstance(axes, int):
                    raise ValueError("All axes must be integers.")
        elif not isinstance(self.axes, int):
            raise ValueError("axes must be integers.")

    def __call__(self: OpticalLayer, wavefront: Wavefront) -> Wavefront:
        """
        Flips the wavefront about the input axes.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to flip on.

        Returns
        -------
        wavefront : Wavefront
            The flipped wavefront.
        """
        return wavefront.flip(self.axes)


class Resize(OpticalLayer):
    """
    Resizes the wavefront by either padding or cropping. If the Wavefront is
    larger than the desired size, then the wavefront is cropped. If the
    Wavefront is smaller than the desired size, then the wavefront is padded
    with zeros.

    Note this class only supports padding and cropping of even sizes to even
    sizes, and odd sizes to odd sizes.

    Attributes
    ----------
    npixels : tuple
        The desired output size of the wavefront.
    """

    npixels: int

    def __init__(self: OpticalLayer, npixels: int):
        """
        Constructor for the Resize class.

        Parameters
        ----------
        npixels : tuple
            The desired output size of the wavefront.
        """
        super().__init__()
        self.npixels = int(npixels)

    def __call__(self: OpticalLayer, wavefront: Wavefront) -> Wavefront:
        """
        Applies the layer to the wavefront.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to resize.

        Returns
        -------
        wavefront : Wavefront
            The resized wavefront.
        """
        if wavefront.npixels > self.npixels:
            return wavefront.crop_to(self.npixels)
        elif wavefront.npixels < self.npixels:
            return wavefront.pad_to(self.npixels)
        else:
            return wavefront
