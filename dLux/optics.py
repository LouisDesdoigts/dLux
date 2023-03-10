from __future__ import annotations
import jax.numpy as np
from jax import vmap
from jax.tree_util import tree_map
from jax.lax import stop_gradient
from zodiax import ExtendedBase
from abc import ABC, abstractmethod
from inspect import signature
from dLux.utils.helpers import two_image_plot
from dLux.utils.units import convert_angular, convert_cartesian
import dLux


__all__ = ["CreateWavefront", "TiltWavefront", "NormaliseWavefront", 
           "ApplyBasisOPD", "AddPhase", "AddOPD", "TransmissiveOptic", 
           "ApplyBasisCLIMB", "Rotate"]


Array = np.ndarray


class OpticalLayer(ExtendedBase, ABC):
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


    def apply(self : OpticalLayer, parameters : dict) -> dict:
        """
        Unpacks the wavefront object from the parameters dictionary and applies
        the layers __call__ method upon it.

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
            parameters["Wavefront"] = self.__call__(parameters["Wavefront"])

        # Method takes and return updated parameters
        elif input_parameters['returns_parameters'].default == True:
            wavefront, parameters = self.__call__(parameters["Wavefront"],
                                                  parameters)
            parameters["Wavefront"] = wavefront

        # Method takes but does not return parameters
        else:
            parameters["Wavefront"] = self.__call__(parameters["Wavefront"],
                                                    parameters)

        # Return updated parameters dictionary
        return parameters
    

    def summary(self            : OpticalLayer, 
                angular_units   : str = 'radians', 
                cartesian_units : str = 'meters', 
                sigfigs         : int = 4) -> str:
        """
        Returns a summary of the class.

        Parameters
        ----------
        angular_units : str = 'radians'
            The angular units to use in the summary. Options are 'radians', 
            'degrees', 'arcseconds' and 'arcminutes'.
        cartesian_units : str = 'meters'
            The cartesian units to use in the summary. Options are 'meters',
            'millimeters' and 'microns'.
        sigfigs : int = 4
            The number of significant figures to use in the summary.

        Returns
        -------
        summary : str
            A summary of the class.
        """
        return f"{self.name} layer has no summary method yet."


    def display(self            : OpticalLayer, 
                wavefront       : Wavefront,
                figsize         : tuple = (10, 4),
                dpi             : int = 120,
                angular_units   : str = 'radians', 
                cartesian_units : str = 'meters', 
                sigfigs         : int = 4) -> None:
        """
        Displays a plot of the wavefront amplitude and opd or phase.

        Parameters
        ----------
        wavefront : Wavefront
            The dummy wavefront to propagate though the optics.
        figsize : tuple = (10, 4)
            The size of the figure to display.
        cmap : str = 'inferno'
            The colour map to use.
        dpi : int = 120
            The resolution of the figure.
        angular_units : str = 'radians'
            The angular units to use in the summary. Options are 'radians', 
            'degrees', 'arcseconds' and 'arcminutes'.
        cartesian_units : str = 'meters'
            The cartesian units to use in the summary. Options are 'meters',
            'millimeters' and 'microns'.
        sigfigs : int = 4
            The number of significant figures to use in the summary.
        """
        amplitude = wavefront.amplitude
        phase = convert_angular(wavefront.phase, "radians", angular_units)
        two_image_plot(amplitude[0], phase[0], figsize=figsize, 
            titles=("Amplitude", "Phase"), cbar_labels=("Intensity", 
            f"Phase ({angular_units})"), cmaps=('inferno', 'twilight'), 
            bounds=(None, 2*np.pi), dpi=dpi)


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
    wavefront_type: str
        Determines the type of wavefront class to create. Currently supports
        'Cartesian', 'Angular', 'FarFieldFresnel'.
    name : str
        The name of the layer, which is used to index the layers dictionary.
    """
    npixels        : int
    diameter       : Array
    wavefront_type : str


    def __init__(self           : OpticalLayer,
                 npixels        : int,
                 diameter       : Array,
                 wavefront_type : str = 'Cartesian',
                 name           : str = 'CreateWavefront') -> OpticalLayer:
        """
        Constructor for the CreateWavefront class.

        Parameters
        ----------
        npixels : int
            The number of pixels used to represent the wavefront.
        diameter: Array, meters
            The diameter of the wavefront in the Pupil plane.
        wavefront_type: str = 'Cartesian'
            Determines the type of wavefront class to create. Currently supports
            'Cartesian', 'Angular', 'FarFieldFresnel'.
        name : str = 'CreateWavefront'
            The name of the layer, which is used to index the layers dictionary.
            Default is 'CreateWavefront'.
        """
        super().__init__(name)
        self.npixels        = int(npixels)
        self.diameter       = np.asarray(diameter, dtype=float)
        self.wavefront_type = str(wavefront_type)

        # Input checks
        assert self.diameter.ndim == 0, ("diameter must be "
        "a scalar array.")
        assert wavefront_type in ('Cartesian', 'Angular', 'FarFieldFresnel'), \
        ("wavefront_type must be either 'Cartesian', 'Angular' or "
         "'FarFieldFresnel'")


    def __call__(self               : OpticalLayer,
                 wavefront          : Wavefront,
                 parameters         : dict,
                 returns_parameters : bool = True) -> Wavefront:
        """
        Constructs a wavefront obect based on the parameters of the class and
        the parameters within the parameters dictionary.

        Parameters
        ----------
        wavefront : None
            Any empty None type input to the class in order to maintain the
            input conventions determied by the apply method of OpticalLayers.
        parameters : dict
            A dictionary of parameters needed to construct the wavefront.
        returns_parametrs: bool = True
            Determines if the class returns the parameters dictionary.

        Returns
        -------
        wavefront, parameters : (Wavefront, dict)
            Returns the constructed wavefront and the updated parameters
            dictionary. If returns_parameters is False, only the wavefront is
            returned.
        """
        # Get the wavelength
        wavelength = parameters["wavelength"]

        # Determine the pixel scale
        pixel_scale = self.diameter/self.npixels

        # Construct normalised Amplitude
        amplitude = np.ones((1, self.npixels, self.npixels))
        amplitude /= np.linalg.norm(amplitude)

        # Construct empty phases
        phase = np.zeros(((1, self.npixels, self.npixels)))

        # Get correct Wavefront type
        wavefront_constructor = getattr(dLux.wavefronts,
                                        self.wavefront_type + "Wavefront")

        # Construct Wavefront
        wavefront = wavefront_constructor(wavelength, pixel_scale, amplitude,
                                      phase, dLux.wavefronts.PlaneType.Pupil)

        # Tilt wavefront from source offset
        wavefront = wavefront.tilt_wavefront(parameters["offset"])

        # Kill PlaneType Gradients
        is_leaf = lambda x: isinstance(x, dLux.wavefronts.PlaneType)
        kill_gradient = lambda x: stop_gradient(x.value) if is_leaf(x) else x
        wavefront = tree_map(kill_gradient, wavefront, is_leaf=is_leaf)

        # Update the parameters dictionary with the constructed wavefront
        parameters["Wavefront"] = wavefront

        # Return either the wavefront or wavefront and parameters dictionary
        if returns_parameters:
            return wavefront, parameters
        else:
            return wavefront
    

    def summary(self            : OpticalLayer, 
                angular_units   : str = 'radians', 
                cartesian_units : str = 'meters', 
                sigfigs         : int = 4) -> str:
        """
        Returns a summary of the class.

        Parameters
        ----------
        angular_units : str = 'radians'
            The angular units to use in the summary. Options are 'radians', 
            'degrees', 'arcseconds' and 'arcminutes'.
        cartesian_units : str = 'meters'
            The cartesian units to use in the summary. Options are 'meters',
            'millimeters' and 'microns'.
        sigfigs : int = 4
            The number of significant figures to use in the summary.

        Returns
        -------
        summary : str
            A summary of the class.
        """
        return (f"{self.name}: Creates a {self.wavefront_type} wavefront of " 
                f"size {self.npixels} pixels and diameter {self.diameter} m.")


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


    def summary(self            : OpticalLayer, 
                angular_units   : str = 'radians', 
                cartesian_units : str = 'meters', 
                sigfigs         : int = 4) -> str:
        """
        Returns a summary of the class.

        Parameters
        ----------
        angular_units : str = 'radians'
            The angular units to use in the summary. Options are 'radians', 
            'degrees', 'arcseconds' and 'arcminutes'.
        cartesian_units : str = 'meters'
            The cartesian units to use in the summary. Options are 'meters',
            'millimeters' and 'microns'.
        sigfigs : int = 4
            The number of significant figures to use in the summary.

        Returns
        -------
        summary : str
            A summary of the class.
        """
        angles = convert_angular(self.tilt_angles, 'radians', angular_units)
        return (f"{self.name}: Tilts the wavefront by {angles:.{sigfigs}} "
                f"{angular_units} in the (x, y) dimension.")


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
    

    def summary(self            : OpticalLayer, 
                angular_units   : str = 'radians', 
                cartesian_units : str = 'meters', 
                sigfigs         : int = 4) -> str:
        """
        Returns a summary of the class.

        Parameters
        ----------
        angular_units : str = 'radians'
            The angular units to use in the summary. Options are 'radians', 
            'degrees', 'arcseconds' and 'arcminutes'.
        cartesian_units : str = 'meters'
            The cartesian units to use in the summary. Options are 'meters',
            'millimeters' and 'microns'.
        sigfigs : int = 4
            The number of significant figures to use in the summary.

        Returns
        -------
        summary : str
            A summary of the class.
        """
        return f"{self.name}: Normalises the wavefront to unity power."


class ApplyBasisOPD(OpticalLayer):
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
    

    def summary(self            : OpticalLayer, 
                angular_units   : str = 'radians', 
                cartesian_units : str = 'meters', 
                sigfigs         : int = 4) -> str:
        """
        Returns a summary of the class.

        Parameters
        ----------
        angular_units : str = 'radians'
            The angular units to use in the summary. Options are 'radians', 
            'degrees', 'arcseconds' and 'arcminutes'.
        cartesian_units : str = 'meters'
            The cartesian units to use in the summary. Options are 'meters',
            'millimeters' and 'microns'.
        sigfigs : int = 4
            The number of significant figures to use in the summary.

        Returns
        -------
        summary : str
            A summary of the class.
        """
        return (f"{self.name}: Applies an Optical Path Difference (OPD) to the "
            f"wavefront calculated from the basis vectors and coefficients.")


class AddPhase(OpticalLayer):
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
        return wavefront.add_phase(self.phase)
    

    def summary(self            : OpticalLayer, 
                angular_units   : str = 'radians', 
                cartesian_units : str = 'meters', 
                sigfigs         : int = 4) -> str:
        """
        Returns a summary of the class.

        Parameters
        ----------
        angular_units : str = 'radians'
            The angular units to use in the summary. Options are 'radians', 
            'degrees', 'arcseconds' and 'arcminutes'.
        cartesian_units : str = 'meters'
            The cartesian units to use in the summary. Options are 'meters',
            'millimeters' and 'microns'.
        sigfigs : int = 4
            The number of significant figures to use in the summary.

        Returns
        -------
        summary : str
            A summary of the class.
        """
        return f"{self.name}: Add an array of phase values to the wavefront."


class AddOPD(OpticalLayer):
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
    

    def summary(self            : OpticalLayer, 
                angular_units   : str = 'radians', 
                cartesian_units : str = 'meters', 
                sigfigs         : int = 4) -> str:
        """
        Returns a summary of the class.

        Parameters
        ----------
        angular_units : str = 'radians'
            The angular units to use in the summary. Options are 'radians', 
            'degrees', 'arcseconds' and 'arcminutes'.
        cartesian_units : str = 'meters'
            The cartesian units to use in the summary. Options are 'meters',
            'millimeters' and 'microns'.
        sigfigs : int = 4
            The number of significant figures to use in the summary.

        Returns
        -------
        summary : str
            A summary of the class.
        """
        return (f"{self.name}: Add an array of Optical Path Differences (OPD) "
                "to the wavefront.")


class TransmissiveOptic(OpticalLayer):
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
                 name         : str = 'TransmissiveOptic') -> OpticalLayer:
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
        super().__init__(name)
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
        return wavefront.multiply_amplitude(self.transmission)
    

    def summary(self            : OpticalLayer, 
                angular_units   : str = 'radians', 
                cartesian_units : str = 'meters', 
                sigfigs         : int = 4) -> str:
        """
        Returns a summary of the class.

        Parameters
        ----------
        angular_units : str = 'radians'
            The angular units to use in the summary. Options are 'radians', 
            'degrees', 'arcseconds' and 'arcminutes'.
        cartesian_units : str = 'meters'
            The cartesian units to use in the summary. Options are 'meters',
            'millimeters' and 'microns'.
        sigfigs : int = 4
            The number of significant figures to use in the summary.

        Returns
        -------
        summary : str
            A summary of the class.
        """
        return (f"{self.name}: Applies an array of tranmission values to the "
                "Wavefront.")


class ApplyBasisCLIMB(OpticalLayer):
    """
    Adds an array of binary phase values to the input wavefront from a set of
    continuous basis vectors. This uses the CLIMB algorithm in order to
    generate the binary values in a continous manner as described in the
    paper Wong et al. 2021. The basis vectors are taken as an Optical Path
    Difference (OPD), and applied to the phase of the wavefront. The ideal
    wavelength parameter described the wavelength that will have a perfect
    anti-phase relationship given by the Optical Path Difference.

    Note: Many of the methods in the class still need doccumentation.
    Note: This currently only outputs 256 pixel arrays and uses a 3x oversample,
    therefore requiring a 768 pixel basis array.

    Attributes
    ----------
    basis: Array
        Arrays holding the continous pre-calculated basis vectors.
    coefficients: Array
        The Array of coefficients to be applied to each basis vector.
    ideal_wavelength : Array
        The target wavelength at which a perfect anti-phase relationship is
        applied via the OPD.
    name : str
        The name of the layer, which is used to index the layers dictionary.
    """
    basis            : Array
    coefficients     : Array
    ideal_wavelength : Array


    def __init__(self             : OpticalLayer,
                 basis            : Array,
                 ideal_wavelength : Array,
                 coefficients     : Array = None,
                 name             : str   = 'ApplyBasisCLIMB') -> OpticalLayer:
        """
        Constructor for the ApplyBasisCLIMB class.

        Parameters
        ----------
        basis : Array
            Arrays holding the continous pre-calculated basis vectors. This must
            be a 3d array of shape (nterms, npixels, npixels), with the final
            two dimensions matching that of the wavefront at time of
            application. This is currently required to be a nx768x768 shaped
            array. 
        ideal_wavelength : Array
            The target wavelength at which a perfect anti-phase relationship is
            applied via the OPD.
        coefficients : Array = None
            The Array of coefficients to be applied to each basis vector. This
            must be a one dimensional array with leading dimension equal to the
            leading dimension of the basis vectors. Default is None which
            initialises an array of zeros.
        name : str = 'ApplyBasisCLIMB'
            The name of the layer, which is used to index the layers dictionary.
        """
        super().__init__(name)
        self.basis            = np.asarray(basis, dtype=float)
        self.ideal_wavelength = np.asarray(ideal_wavelength, dtype=float)
        self.coefficients     = np.array(coefficients).astype(float) \
                    if coefficients is not None else np.zeros(len(self.basis))

        # Inputs checks
        assert self.basis.ndim == 3, \
        ("basis must be a 3 dimensional array, ie (nterms, npixels, npixels).")
        assert self.basis.shape[-1] == 768, \
        ("Basis must have shape (n, 768, 768).")
        assert self.coefficients.ndim == 1 and \
        self.coefficients.shape[0] == self.basis.shape[0], \
        ("coefficients must be a 1 dimensional array with length equal to the "
        "First dimension of the basis array.")
        assert self.ideal_wavelength.ndim == 0, ("ideal_wavelength must be a "
                                                 "scalar array.")


    def __call__(self : OpticalLayer, wavefront : Wavefront) -> Wavefront:
        """
        Generates and applies the binary OPD array to the wavefront in a
        differentiable manner.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to operate on.

        Returns
        -------
        wavefront : Wavefront
            The wavefront with the binary OPD applied.
        """
        latent = self.get_opd(self.basis, self.coefficients)
        binary_phase = np.pi*self.CLIMB(latent, ppsz=wavefront.npixels)
        opd = self.phase_to_opd(binary_phase, self.ideal_wavelength)
        return wavefront.add_opd(opd)


    def opd_to_phase(self, opd, wavel):
        return 2*np.pi*opd/wavel


    def phase_to_opd(self, phase, wavel):
        return phase*wavel/(2*np.pi)


    def get_opd(self, basis, coefficients):
        return np.dot(basis.T, coefficients)


    def get_total_opd(self):
        return self.get_opd(self.basis, self.coefficients)


    def get_binary_phase(self):
        latent = self.get_opd(self.basis, self.coefficients)
        binary_phase = np.pi*self.CLIMB(latent)
        return binary_phase


    def lsq_params(self, img):
        xx, yy = np.meshgrid(np.linspace(0,1,img.shape[0]),
                             np.linspace(0,1,img.shape[1]))
        A = np.vstack([xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]).T
        matrix = np.linalg.inv(np.dot(A.T,A)).dot(A.T)
        return matrix, xx, yy, A


    def lsq(self, img):
        matrix, _, _, _ = self.lsq_params(img)
        return np.dot(matrix,img.ravel())


    def area(self, img, epsilon = 1e-15):
        a,b,c = self.lsq(img)
        a = np.where(a==0,epsilon,a)
        b = np.where(b==0,epsilon,b)
        c = np.where(c==0,epsilon,c)
        x1 = (-b-c)/(a) # don't divide by zero
        x2 = -c/(a) # don't divide by zero
        x1, x2 = np.min(np.array([x1,x2])), np.max(np.array([x1,x2]))
        x1, x2 = np.max(np.array([x1,0])), np.min(np.array([x2,1]))

        dummy = x1 + (-c/b)*x2-(0.5*a/b)*x2**2 - (-c/b)*x1+(0.5*a/b)*x1**2

        # Set the regions where there is a defined gradient
        dummy = np.where(dummy>=0.5,dummy,1-dummy)

        # Colour in regions
        dummy = np.where(np.mean(img)>=0,dummy,1-dummy)

        # rescale between 0 and 1?
        dummy = np.where(np.all(img>0),1,dummy)
        dummy = np.where(np.all(img<=0),0,dummy)

        # undecided region
        dummy = np.where(np.any(img==0),np.mean(dummy>0),dummy)

        # rescale between 0 and 1
        dummy = np.clip(dummy, 0, 1)

        return dummy

    def CLIMB(self, wf, ppsz = 256):
        psz = ppsz * 3

        dummy = np.array(wf.split(ppsz))
        dummy = np.array(dummy.split(ppsz, axis = 2))
        subarray = dummy[:,:,0,0]

        flat = dummy.reshape(-1, 3, 3)
        vmap_mask = vmap(self.area, in_axes=(0))

        soft_bin = vmap_mask(flat).reshape(ppsz, ppsz)

        return soft_bin
    

    def summary(self            : OpticalLayer, 
                angular_units   : str = 'radians', 
                cartesian_units : str = 'meters', 
                sigfigs         : int = 4) -> str:
        """
        Returns a summary of the class.

        Parameters
        ----------
        angular_units : str = 'radians'
            The angular units to use in the summary. Options are 'radians', 
            'degrees', 'arcseconds' and 'arcminutes'.
        cartesian_units : str = 'meters'
            The cartesian units to use in the summary. Options are 'meters',
            'millimeters' and 'microns'.
        sigfigs : int = 4
            The number of significant figures to use in the summary.

        Returns
        -------
        summary : str
            A summary of the class.
        """
        return (f"{self.name}: Applies a binary OPD to the Wavefront using the "
                "CLIMB algorithm.")


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


    def summary(self            : OpticalLayer, 
                angular_units   : str = 'radians', 
                cartesian_units : str = 'meters', 
                sigfigs         : int = 4) -> str:
        """
        Returns a summary of the class.

        Parameters
        ----------
        angular_units : str = 'radians'
            The angular units to use in the summary. Options are 'radians', 
            'degrees', 'arcseconds' and 'arcminutes'.
        cartesian_units : str = 'meters'
            The cartesian units to use in the summary. Options are 'meters',
            'millimeters' and 'microns'.
        sigfigs : int = 4
            The number of significant figures to use in the summary.

        Returns
        -------
        summary : str
            A summary of the class.
        """
        angle = convert_angular(self.angle, 'radians', angular_units)
        
        if self.fourier:
            method = f"a Fourier method with padding of {self.padding}"
        else:
            method = "an Interpolation method of order 1"

        if self.real_imaginary:
            wf_type = "real and imaginary arrays"
        else:
            wf_type = "amplitude and phase arrays"

        return (f"{self.name}: Applies a {angle:.{sigfigs}} {angular_units} "
                f"rotation to the wavefront {wf_type} using {method}.")