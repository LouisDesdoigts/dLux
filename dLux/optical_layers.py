from __future__ import annotations
from abc import abstractmethod
import jax.numpy as np
from jax import vmap, Array
from jax.tree_util import tree_map
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
    "Rotate"]


class OpticalLayer(Base):
    """
    Base class for optical layers. Primarily used for input type checking.

    Child classes must implement the __call__ method that takes in the wavefront
    as the first parameter

    Note: I have chosen __call__ over apply as the method name for the layer 
    to be applied to the wavefront. This is because even though it prevents
    the simple interface with optax (not having to wrap in a list), becuase
    wavefront should in general not be able to be an object you take a gradient
    with respect to, it is just a latent class to store information throughout
    the calculation, plus its use of strings as a way to track parameters can
    make interactions with jax difficult.
    """


    @abstractmethod
    def __call__(self : OpticalLayer, 
        wavefront : Wavefront) -> Wavefront: # pragma: no cover
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
    Base class to hold tranmissive layers embuing them with a normalise 
    parameter.
    """
    normalise : bool

    def __init__(self      : OpticalLayer, 
                 normalise : bool = False,
                 **kwargs) -> OpticalLayer:
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


    @abstractmethod
    def applied_shape(self):
        """
        Returns the 'shape' of the layer, more specifically the required 
        matching shape of the waevefront to be applied to.
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
    basis        : Union[Array, list]
    coefficients : Array


    def __init__(self         : OpticalLayer,
                 basis        : Array = None,
                 coefficients : Array = None,
                 **kwargs) -> OpticalLayer:
        super().__init__(**kwargs)
        if basis is not None:
            basis = np.asarray(basis, dtype=float)
        self.basis = basis

        if self.basis is None:
            self.coefficients = None
        else:
            if hasattr(self.basis, 'shape'):
                coefficients = np.zeros(self.basis.shape[:-2])
            else:
                coefficients = np.zeros(len(self.basis))
            self.coefficients = np.asarray(coefficients, dtype=float)

        if isinstance(self.basis, Array):
            if self.basis.shape[:-2] != self.coefficients.shape:
                raise ValueError("The number of basis vectors must be equal to "
                    "the number of coefficients.")
        elif isinstance(self.basis, list):
            if len(self.basis) != len(self.coefficients):
                raise ValueError("The number of basis vectors must be equal to "
                    "the number of coefficients.")

    def calculate(self, basis, coefficients):
        ndim = coefficients.ndim
        axes = (tuple(range(ndim)), tuple(range(ndim)))
        return np.tensordot(basis, coefficients, axes=axes)

######### Optics #########
class BaseTransmissiveOptic(TransmissiveLayer, ShapedLayer):
    transmission: Array

    def __init__(self         : OpticalLayer,
                 transmission : Array,
                 **kwargs) -> OpticalLayer:
        if transmission is not None:
            transmission = np.asarray(transmission, dtype=float)
        self.transmission = transmission
        super().__init__(**kwargs)

    @property
    def applied_shape(self):
        return self.transmission.shape

class BaseOPDOptic(AberratedLayer, ShapedLayer):
    opd : Array

    def __init__(self : OpticalLayer, opd : Array, **kwargs) -> OpticalLayer:
        if opd is not None:
            opd = np.asarray(opd, dtype=float)
        self.opd = opd
        super().__init__(**kwargs)

    @property
    def applied_shape(self):
        return self.transmission.shape

class BasePhaseOptic(AberratedLayer, ShapedLayer):
    phase : Array

    def __init__(self : OpticalLayer, phase : Array, **kwargs) -> OpticalLayer:
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
    basis: Array, meters
        Arrays holding the pre-calculated basis vectors.
    coefficients: Array
        The Array of coefficients to be applied to each basis vector.
    normalise: bool
        Whether to normalise the wavefront after passing through the
        optic.
    """

    def __init__(self         : OpticalLayer,
                 basis        : Array,
                 transmission : Array = None,
                 coefficients : Array = None,
                 normalise    : bool = False) -> OpticalLayer:
        super().__init__(transmission=transmission, basis=basis, 
        coefficients=coefficients, normalise=normalise)
    
    @property
    def applied_shape(self):
        return self.basis.shape[-2:]


######################
### Public Classes ###
######################
class Optic(BaseTransmissiveOptic, BaseOPDOptic):

    def __init__(self         : OpticalLayer,
                 transmission : Array = None,
                 opd          : Array = None,
                 normalise    : bool = False) -> OpticalLayer:
        """
        
        """
        super().__init__(transmission=transmission, opd=opd, 
            normalise=normalise)
        
        if self.opd is not None and self.transmission is not None:
            if opd.shape != self.transmission.shape:
                raise ValueError("opd and transmission must have the same "
                    "shape.")
    
    def __call__(self : OpticalLayer, wavefront : Wavefront) -> Wavefront:
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
    basis: Array, meters
        Arrays holding the pre-calculated basis vectors.
    coefficients: Array
        The Array of coefficients to be applied to each basis vector.
    """

    @property
    def opd(self):
        return self.calculate(self.basis, self.coefficients)
    
    def __call__(self, wavefront):
        wavefront *= self.transmission
        wavefront += self.opd
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront


class PhaseOptic(BaseTransmissiveOptic, BasePhaseOptic):

    def __init__(self         : OpticalLayer,
                 transmission : Array = None,
                 phase        : Array = None,
                 normalise    : bool = False) -> OpticalLayer:
        """
        
        """
        super().__init__(transmission=transmission, phase=phase,
            normalise=normalise)

        if self.phase is not None and self.transmission is not None:
            if phase.shape != self.transmission.shape:
                raise ValueError("phase and transmission must have the same "
                    "shape.")
    
    def __call__(self : OpticalLayer, wavefront : Wavefront) -> Wavefront:
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
    basis: Array, meters
        Arrays holding the pre-calculated basis vectors.
    coefficients: Array
        The Array of coefficients to be applied to each basis vector.
    """

    @property
    def phase(self):
        return self.calculate(self.basis, self.coefficients)

    def __call__(self : OpticalLayer, wavefront : Wavefront) -> Wavefront:
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
    angles : Array


    def __init__(self  : OpticalLayer, angles : Array) -> OpticalLayer:
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


    def __call__(self : OpticalLayer, wavefront : Wavefront) -> Wavefront:
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
    
    def __call__(self : OpticalLayer, wavefront : Wavefront) -> Wavefront:
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
    angle   : Array
    order   : int
    complex : bool


    def __init__(self    : OpticalLayer,
                 angle   : Array,
                 order   : int  = 1,
                 complex : bool = False) -> OpticalLayer:
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
            Should the rotation be performed on the 'complex' (real, imaginary),
            as opposed to the default 'phasor' (amplitude, phase) arrays.
        """
        super().__init__()
        self.angle   = np.asarray(angle, dtype=float)
        self.order   = int(order)
        self.complex = bool(complex)

        if self.order not in (0, 1):
            raise ValueError("Order must be 0, 1")
        if self.angle.ndim != 0:
            raise ValueError(f"angle must be a zero-dimensional, has "
                f"{self.angle.ndim} dimensions.")


    def __call__(self : OpticalLayer, wavefront : Wavefront) -> Wavefront:
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