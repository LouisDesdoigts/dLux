from __future__ import annotations
import jax.numpy as np
from jax import vmap, Array
from jax.tree_util import tree_map
from jax.lax import stop_gradient
from zodiax import Base
from abc import abstractmethod
from inspect import signature
import dLux


__all__ = ["TiltWavefront", "NormaliseWavefront", 
           "ApplyBasisOPD", "AddPhase", "AddOPD", "TransmissiveOptic", 
           "Rotate"]


class OpticalLayer(Base):
    """
    # Optical layer equivilent to Base layer of other classes

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
        Appies the layer to the `Wavefront`.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to operate on.

        Returns
        -------
        wavefront : Wavefront
            The wavefront with the optical layer applied.
        """
        return

# For 'Aberration' Layers
class AberrationLayer(OpticalLayer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def get_opd(self):
        pass


# For 'Aperutre' Layers
class TransmissiveLayer(OpticalLayer):
    """
    Base class to hold tranmissive layers embuing them with a normalise 
    parameter.
    """
    normalise : bool

    def __init__(self, normalise=False, **kwargs):
        super().__init__(**kwargs)
        self.normalise = bool(normalise)


# For checking shape compatability with wavefronts
class ShapedLayer(OpticalLayer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @abstractmethod
    def shape(self):
        """
        Returns the 'shape' of the layer, more specifically the required 
        matching shape of the waevefront to be applied to.

        Note: Must have the @property decorator to work correctly
        """
        pass


###############################
### Calls Wavefront methods ###
###############################
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

        if self.angle.shape != (2,):
            raise ValueError("angles must have have (2,)")


    def __call__(self : OpticalLayer, wavefront : Wavefront) -> Wavefront:
        """
        Applies the tilt_angle to the phase of the wavefront.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to operate on.

        Returns
        -------
        wavefront : Wavefront
            The wavefront with the optical layer applied.
        """
        return wavefront.tilt_wavefront(self.tilt_angles)


class NormaliseWavefront(OpticalLayer):
    """
    Normalises the input wavefront using the in-built wavefront normalisation
    method.
    """
    

    def __init__(self : OpticalLayer) -> OpticalLayer:
        """
        Constructor for the NormaliseWavefront class.
        """
        super().__init__()


    def __call__(self : OpticalLayer, wavefront : Wavefront) -> Wavefront:
        """
        Normalises the wavefront.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to operate on.

        Returns
        -------
        wavefront : Wavefront
            The wavefront with the wavefront normalisation method applied.
        """
        return wavefront.normalise()


class AddPhase(ShapedLayer):
    """
    Adds an array of phase values to the wavefront.

    Attributes
    ----------
    phase: Array, radians
        The Array of phase values to be applied to the input wavefront.
    """
    phase : Array


    def __init__(self : OpticalLayer, phase : Array) -> OpticalLayer:
        """
        Constructor for the AddPhase class.

        Parameters
        ----------
        phase : Array, radians
            Array of phase values to be applied to the input wavefront.
        """
        super().__init__()
        self.phase = np.asarray(phase, dtype=float)


    @property
    def shape(self):
        return self.phase.shape


    def __call__(self : OpticalLayer, wavefront : Wavefront) -> Wavefront:
        """
        Adds the phase to the wavefront.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to operate on.

        Returns
        -------
        wavefront : Wavefront
            The wavefront with the phase added.
        """
        return wavefront.add_phase(phase)


class AddOPD(ShapedLayer):
    """
    Adds an Optical Path Difference (OPD) to the wavefront.

    Attributes
    ----------
    opd : Array, meters
        Array of OPD values to be applied to the input wavefront.
    """
    opd : Array


    def __init__(self : OpticalLayer, opd : Array) -> OpticalLayer:
        """
        Constructor for the ApplyOPD class.

        Parameters
        ----------
        opd : float, meters
            The Array of OPDs to be applied to the input wavefront.
        """
        super().__init__()
        self.opd = np.asarray(opd, dtype=float)


    @property
    def shape(self):
        return self.opd.shape


    def __call__(self : OpticalLayer, wavefront : Wavefront) -> Wavefront:
        """
        Apply the OPD array to the input wavefront.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to operate on.

        Returns
        -------
        wavefront : Wavefront
            The wavefront with the OPD added.
        """
        return wavefront.add_opd(self.opd)


class TransmissiveOptic(TransmissiveLayer, ShapedLayer):
    """
    Represents an arbitrary transmissive optic.

    Note this class does not normalise the 'transmission' between 0 and 1, but
    simply multiplies the wavefront amplitude by the transmision array.

    Attributes
    ----------
    transmission : Array
        An array representing the transmission of the optic.
    """
    transmission: Array


    def __init__(self         : OpticalLayer,
                 transmission : Array,
                 normalise    : bool = False) -> OpticalLayer:
        """
        Constructor for the TransmissiveOptic class.

        Parameters
        ----------
        transmission : Array
            The array representing the transmission of the aperture. This must
            a 0, 2 or 3 dimensional array with equal to that of the wavefront
            at time of aplication.
        """
        super().__init__(normalise=normalise)
        self.transmission = np.asarray(transmission, dtype=float)


    @property
    def shape(self):
        return self.transmission.shape
    

    def __call__(self : OpticalLayer, wavefront : Wavefront) -> Wavefront:
        """
        Applies the tranmission of the optical to the wavefront.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to operate on.

        Returns
        -------
        wavefront : Wavefront
            The wavefront with the tranmission applied.
        """
        wavefront *= self.transmission
        return wavefront.normalise() if self.normalise else wavefront


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
                f"{self.angle.ndims} dimensions.")


    def __call__(self : OpticalLayer, wavefront : Wavefront) -> Wavefront:
        """
        Applies the rotation to a wavefront.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to operate on.

        Returns
        -------
        wavefront : Wavefront
            The rotated wavefront.
        """
        return wavefront.rotate(self.angle, self.order, self.rotate)


####################################
### Layers that calculate Things ###
####################################
class ApplyBasisOPD(AberrationLayer, ShapedLayer):
    """
    Adds an array of phase values to the input wavefront calculated from the
    Optical Path Difference (OPD). The OPDs are calculated from the basis
    arrays, and weighted by the coefficients, and converted to phases by the
    wavefront methods.

    Attributes
    ----------
    basis: Array, meters
        Arrays holding the pre-calculated basis vectors.
    coefficients: Array
        The Array of coefficients to be applied to each basis vector.
    """
    basis        : Array
    coefficients : Array


    def __init__(self         : OpticalLayer,
                 basis        : Array,
                 coefficients : Array = None) -> OpticalLayer:
        """
        Constructor for the ApplyBasisOPD class.

        Parameters
        ----------
        basis : Array, meters
            The Array of basis polynomials. This should be a 3 dimensional Array
            with the first dimension being the number of basis vectors, and the
            last two dimensions being equal to the wavefront shape at the time
            of application to the wavefront.
        coefficients : Array = None
            The coefficients by which to weight the basis vectors. This must
            have the same length as the first dimension of the basis Array. If
            None is supplied an Array of zeros is constructed.
        """
        super().__init__()
        self.basis = np.asarray(basis, dtype=float)

        if coefficients is None:
            coefficients = np.zeros(self.basis.shape[0])
        self.coefficients = np.asarray(coefficients, dtype=float)

        if self.basis.shape[0] != self.coefficients.shape[0]:
            raise ValueError("The number of basis vectors must be equal to the"
                "number of coefficients.")
    

    @property
    def shape(self):
        return self.basis.shape[-2:]


    def get_opd(self : OpticalLayer) -> Array:
        """
        A function to calculate the total OPD from the basis vector and the
        coefficients.

        Returns
        -------
        OPD : Array, meters
            The total OPD calulated from the basis vectors and coefficients.
        """
        return np.dot(self.basis.T, self.coefficients)


    def __call__(self : OpticalLayer, wavefront : Wavefront) -> Wavefront:

        """
        Calculate and apply the appropriate phase shift to the wavefront.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to operate on.

        Returns
        -------
        wavefront : Wavefront
            The wavefront with the appropriate phase applied.
        """
        return wavefront.add_opd(self.get_opd())