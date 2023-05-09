from __future__ import annotations
import jax.numpy as np
from jax import vmap, Array
from jax.tree_util import tree_map
from jax.lax import stop_gradient
from zodiax import Base
from abc import ABC, abstractmethod
from inspect import signature
from dLux.utils.helpers import two_image_plot
from dLux.utils.units import convert_angular, convert_cartesian
import dLux


__all__ = ["CreateWavefront", "TiltWavefront", "NormaliseWavefront", 
           "ApplyBasisOPD", "AddPhase", "AddOPD", "TransmissiveOptic", 
           "ApplyBasisCLIMB", "Rotate"]


class OpticalLayer(Base, ABC):
    """
    A base Optical layer class to help with type checking throuhgout the rest
    of the software. Instantiates the apply method which inspects the function
    signature of the __call__ method in order to only pass and return the
    parameters dictionary if it is needed and modified.

    Attributes
    ----------
    name : str
        The name of the layer, which is used to index the layers dictionary.
    """
    name : str


    def __init__(self : OpticalLayer,
                 name : str = 'OpticalLayer') -> OpticalLayer:
        """
        Constructor for the OpticalLayer class.

        Parameters
        ----------
        name : str = 'OpticalLayer'
            The name of the layer, which is used to index the layers dictionary.
        """
        self.name = str(name)


    # Remove apply and rename to apply
    @abstractmethod
    def __call__(self : OpticalLayer, wavefront : Wavefront) -> Wavefront: # pragma: no cover
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


    # Remove and use apply name for __call__ 
    def apply(self : OpticalLayer, 
              wavefront, 
              parameters : dict) -> dict:
        """
        Unpacks the wavefront object from the parameters dictionary and applies
        the layers __call__ method upon it. This allows for extra meta-data
        to be passed between the layers to allow for more complex interactions
        between optical components (such as pupil-remapping).

        Parameters
        ----------
        parameters : dict
            A dictionary that must contain a "Wavefront" key with a
            corresponding dLux.wavefronts.Wavefront object.

        Returns
        -------
        parameters : dict
            A dictionary with the updated "Wavefront" key with the propagated
            wavefront object.
        """
        # Inspect apply function to see if it takes/returns the parameters dict
        input_parameters = signature(self.__call__).parameters

        # Method does not take in the parameters, update in place
        if 'parameters' not in input_parameters:
            # parameters["Wavefront"] = self.__call__(parameters["Wavefront"])
            wavefront = self.__call__(wavefront)

        # Method takes and return updated parameters
        elif input_parameters['returns_parameters'].default == True:
            wavefront, parameters = self.__call__(wavefront, parameters)

        # Method takes but does not return parameters
        else:
            wavefront = self.__call__(wavefront, parameters)

        # Return updated parameters dictionary
        return wavefront, parameters


class AberrationLayer(OpticalLayer):
    # No coefficinets here becuase it might be a static OPD

    def __init__(self, name, **kwargs):
        super().__init__(name = name, **kwargs)

    @abstractmethod
    def get_opd(self):
        pass


class TransmissiveLayer(OpticalLayer):
    """
    Base class to hold tranmissive layers embuing them with a normalise 
    parameter.
    """
    normalise : bool

    def __init__(self, normalise=False, **kwargs):
        super().__init__(**kwargs)
        self.normalise = bool(normalise)


class ShapedLayer(OpticalLayer):

    @abstractmethod
    def shape(self):
        """
        Returns the 'shape' of the layer, more specifically the required 
        matching shape of the waevefront to be applied to.

        Note: Must have the @property decorator to work correctly
        """
        pass

class CreateWavefront(OpticalLayer):
    """
    Initialises the relevant Wavefront class with the specified attributes.
    Also applies the tilt specified by the source object, defined in the
    parameters dictionary. All wavefronts are cosntructed in the Pupil plane.

    Attributes
    ----------
    npixels : int
        The number of pixels used to represent the wavefront.
    diameter: Array, meters
        The diameter of the wavefront in the Pupil plane.
    name : str
        The name of the layer, which is used to index the layers dictionary.
    """
    npixels        : int
    diameter       : Array


    def __init__(self     : OpticalLayer,
                 npixels  : int,
                 diameter : Array,
                 name     : str = 'CreateWavefront') -> OpticalLayer:
        """
        Constructor for the CreateWavefront class.

        Parameters
        ----------
        npixels : int
            The number of pixels used to represent the wavefront.
        diameter: Array, meters
            The diameter of the wavefront in the Pupil plane.
        name : str = 'CreateWavefront'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(name)
        self.npixels  = int(npixels)
        self.diameter = np.asarray(diameter, dtype=float)

        # Input checks
        assert self.diameter.ndim == 0, ("diameter must be "
        "a scalar array.")


    def __call__(self       : OpticalLayer,
                 wavelength : Arary,
                 offset     : Array = np.zeros(2)) -> Wavefront:
        """
        Constructs a wavefront obect based on the parameters of the class and
        the parameters within the parameters dictionary.

        Parameters
        ----------
        wavelength : Array
            The wavelength of the wavefront.
        offset : Array, radians, = np.zeros(2)
            The (x, y) offset from the optical axis of the source. Default
            value is (0, 0), on axis.

        Returns
        -------
        wavefront : Wavefront
            Returns the constructed wavefront with approprately set parameters,
            optionally tilted by offset.
        """
        wavefront = dLux.wavefronts.Wavefront(self.npixels, self.diameter, 
            wavelength)
        return wavefront.tilt_wavefront(offset)


class TiltWavefront(OpticalLayer):
    """
    Tilts the wavefront by the input tilt_angles.

    Attributes
    ----------
    tilt_angles : Array, radians
        The (x, y) angles by which to tilt the wavefront.
    name : str
        The name of the layer, which is used to index the layers dictionary.
    """
    tilt_angles : Array


    def __init__(self        : OpticalLayer,
                 tilt_angles : Array,
                 name        : str = 'TiltWavefront') -> OpticalLayer:
        """
        Constructor for the TiltWavefront class.

        Parameters
        ----------
        tilt_angles : Array, radians
            The (x, y) angles by which to tilt the wavefront.
        name : str = TiltWavefront
            The name of the layer, which is used to index the layers dictionary.
            Default is 'TiltWavefront'.
        """
        super().__init__(name)
        self.tilt_angles = np.asarray(tilt_angles, dtype=float)

        # Input checks
        assert self.tilt_angles.shape == (2,), \
        ("tilt_angles must be an array of shape (2,), ie (x, y).")


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

    Attributes
    ----------
    name : str
        The name of the layer, which is used to index the layers dictionary.
    """
    

    def __init__(self : OpticalLayer,
                 name : str = 'NormaliseWavefront') -> OpticalLayer:
        """
        Constructor for the NormaliseWavefront class.

        Parameters
        ----------
        name : string = 'NormaliseWavefront'
            The name of the layer, which is used to index the layers
            dictionary.
        """
        super().__init__(name)


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
    name : str
        The name of the layer, which is used to index the layers dictionary.
    """
    basis        : Array
    coefficients : Array


    def __init__(self         : OpticalLayer,
                 basis        : Array,
                 coefficients : Array = None,
                 name         : str = 'ApplyBasisOPD') -> OpticalLayer:
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
        name : str = 'ApplyBasisOPD'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(name)
        self.basis        = np.asarray(basis, dtype=float)
        self.coefficients = np.zeros(basis.shape[0]) if coefficients is None \
                            else np.asarray(coefficients, dtype=float)

        # Input checks
        assert self.basis.ndim == 3, \
        ("basis must be a 3 dimensional array, ie (nterms, npixels, npixels).")
        assert self.coefficients.ndim == 1 and \
        self.coefficients.shape[0] == self.basis.shape[0], \
        ("coefficients must be a 1 dimensional array with length equal to the "
        "First dimension of the basis array.")
    

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


class AddPhase(OpticalLayer, ShapedLayer):
    """
    Adds an array of phase values to the wavefront.

    Attributes
    ----------
    phase: Array, radians
        The Array of phase values to be applied to the input wavefront.
    name : str
        The name of the layer, which is used to index the layers dictionary.
    """
    phase : Array


    def __init__(self  : OpticalLayer,
                 phase : Array,
                 name  : str = 'AddPhase') -> OpticalLayer:
        """
        Constructor for the AddPhase class.

        Parameters
        ----------
        phase : Array, radians
            Array of phase values to be applied to the input wavefront. This
            must a 0, 2 or 3 dimensional array with equal to that of the 
            wavefront at time of aplication.
        name : str = 'AddPhase'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(name)
        self.phase = np.asarray(phase, dtype=float)

        # Input checks
        assert self.phase.ndim in (0, 2, 3), ("phase must be either a scalar "
        "array, or a 2 or 3 dimensional array.")


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
        return wavefront + self.phase / wavefront.wavenumber
    

    @property
    def shape(self):
        return self.phase.shape


class AddOPD(OpticalLayer, ShapedLayer):
    """
    Adds an Optical Path Difference (OPD) to the wavefront.

    Attributes
    ----------
    opd : Array, meters
        Array of OPD values to be applied to the input wavefront.
    name : str
        The name of the layer, which is used to index the layers dictionary.
    """
    opd : Array


    def __init__(self : OpticalLayer,
                 opd  : Array,
                 name : str = 'AddOPD') -> OpticalLayer:
        """
        Constructor for the ApplyOPD class.

        Parameters
        ----------
        opd : float, meters
            The Array of OPDs to be applied to the input wavefront. This must
            a 0, 2 or 3 dimensional array with equal to that of the wavefront
            at time of aplication.
        name : str = 'AddOPD'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(name)
        self.opd = np.asarray(opd, dtype=float)

        # Input checks
        assert self.opd.ndim in (0, 2, 3), ("opd must be either a scalar "
        "array, or a 2 or 3 dimensional array.")


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


    @property
    def shape(self):
        return self.opd.shape


class TransmissiveOptic(TransmissiveLayer, ShapedLayer):
    """
    Represents an arbitrary transmissive optic.

    Note this class does not normalise the 'transmission' between 0 and 1, but
    simply multiplies the wavefront amplitude by the transmision array.

    Attributes
    ----------
    transmission : Array
        An array representing the transmission of the optic.
    name : str
        The name of the layer, which is used to index the layers dictionary.
    """
    transmission: Array


    def __init__(self         : OpticalLayer,
                 transmission : Array,
                 normalise    : bool = False,
                 name         : str = 'TransmissiveOptic', 
                 **kwargs) -> OpticalLayer:
        """
        Constructor for the TransmissiveOptic class.

        Parameters
        ----------
        transmission : Array
            The array representing the transmission of the aperture. This must
            a 0, 2 or 3 dimensional array with equal to that of the wavefront
            at time of aplication.
        name : str = 'TransmissiveOptic'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(normalise=normalise, name=name, **kwargs)
        self.transmission = np.asarray(transmission, dtype=float)

        # Input checks
        assert self.transmission.ndim in (0, 2, 3), ("transmission must be "
        "either a scalar array, or a 2 or 3 dimensional array.")


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
        if self.normalise:
            return wavefront.normalise()
        return wavefront


    @property
    def shape(self):
        return self.transmission.shape



class Rotate(OpticalLayer):
    """
    Applies a rotation to the wavefront using interpolation methods.

    Attributes
    ----------
    angle : Array, radians
        The angle by which to rotate the wavefront in the clockwise direction.
    real_imaginary : bool
        Should the rotation be performed on the amplitude and phase array
        or the real and imaginary arrays.
    fourier : bool
        Should the rotation be done using fourier methods or interpolation.
    order : int = 1
        The order of the interpolation to use. Only applies if fourier is
        False. Must be 0, 1, or 3.
    padding : int
        The amount of padding to use if the fourier method is used.
    name : str
        The name of the layer, which is used to index the layers dictionary.
    """
    angle          : Array
    real_imaginary : bool
    fourier        : bool
    order          : int
    padding        : int


    def __init__(self           : OpticalLayer,
                 angle          : Array,
                 real_imaginary : bool = False,
                 fourier        : bool = False,
                 order          : int  = 1,
                 padding        : int  = None,
                 name           : str  = 'Rotate') -> OpticalLayer:
        """
        Constructor for the Rotate class.

        Parameters
        ----------
        angle: float, radians
            The angle by which to rotate the wavefront in the clockwise 
            direction.
        real_imaginary : bool = False
            Should the rotation be performed on the amplitude and phase array
            or the real and imaginary arrays.
        fourier : bool = False
            Should the fourier rotation method be used (True), or regular
            interpolation method be used (False).
        order : int = 1
            The order of the interpolation to use. Only applies if fourier is
            False. Must be 0, 1, or 3.
        padding : int = None
            The amount of fourier padding to use. Only applies if fourier is
            True.
        name : str = 'Rotate'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(name)
        self.angle          = np.asarray(angle, dtype=float)
        self.real_imaginary = bool(real_imaginary)
        if order not in (0, 1, 3):
            raise ValueError("Order must be 0, 1, or 3.")
        self.order = int(order)
        self.fourier        = bool(fourier)
        self.padding = padding if padding is None else int(padding)
        assert self.angle.ndim == 0, ("angle must be scalar array.")


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
        args = [self.angle, self.real_imaginary, self.fourier, self.order]
        args += [self.padding] if self.padding is not None else []
        return wavefront.rotate(*args)