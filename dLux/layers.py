"""
layers.py
---------
A layer is a mathematical abstraction of an optical interaction. 
The physical interpretation of most layers is straightforward but 
some are not obvious. This structure was chosen because of the 
constrainst of automatic differentiation using `equinox`.

Concrete Classes 
----------------
- CreateWavefront
- TiltWavefront
- CircularAperture
- NormaliseWavefront
- ApplyBasisOPD
- AddPhase
- TransmissiveOptic
- ApplyBasisCLIMB
"""
from __future__ import annotations
import jax
import jax.numpy as np
import equinox as eqx
import dLux
import abc


__author__ = "Louis Desdoigts"
__date__ = "05/07/2022"


__all__ = ["AddPhase", "TransmissiveOptic", "ApplyBasisCLIMB", 
           "ApplyBasisOPD", "ApplyOPD", "CircularAperture", "CreateWavefront",
           "NormaliseWavefront", "TiltWavefront", "CompoundAperture",
           "InformationConservingRotation"]


class OpticalLayer(dLux.base.Base, abc.ABC):
    """
    A base Optical layer class to help with type checking throuhgout the rest
    of the software.
    """
    
    
    @abc.abstractmethod
    def __call__(self : OpticalLayer, params_dict : dict) -> dict:
        """
        Abstract method for Optical Layers
        """
        return


class CreateWavefront(OpticalLayer):
    """
    Initialises an on-axis input wavefront

    Attributes
    ----------
    npix : int
        The number of pixels along one side that represent the 
        wavefront.
    pixel_scale: float, meters/pixel
        The pixelscae of each array between each layer operation
        Its value is automatically calculated from the input values 
    wavefront_size: float, meters
        Width of the array representing the wavefront in physical units
    wavefront_type: string
        Determines the type of wavefront class to create. Currently
        supports 'Cartesian', 'Angular', 'FarFieldFresnel'
    name : string
        The name of the layer, which is used to index the layers
        dictionary. Default is 'Wavefront Creation'
    """
    npix : int
    wavefront_size : float
    # offset : float
    wavefront_type : str = eqx.static_field()
    name : str = eqx.static_field()


    def __init__(self, npix, wavefront_size, offset, wavefront_type='Cartesian', name='Wavefront Creation'):
        """
        Parameters
        ----------
        npix : int
            The number of pixels along one edge of the wavefront.
        wavefront_size : float, meters
            The physical dimensions of the wavefront in units of
            (radians) meters.
        wavefront_type: string
            Determines the type of wavefront class to create. Currently
            supports 'Cartesian', 'Angular', 'FarFieldFresnel'
        name : string
            The name of the layer, which is used to index the layers
            dictionary. Default is 'Wavefront Creation'
        """
        self.npix = int(npix)
        self.wavefront_size = np.array(wavefront_size).astype(float)
        # self.offset = np.asarray(offset, float)
        assert wavefront_type in ['Cartesian', 'Angular', 'FarFieldFresnel'], \
        "wavefront_type must be either 'Cartesian', 'Angular' or 'FarFieldFresnel'"
        self.wavefront_type = str(wavefront_type)
        self.name = name


    def __call__(self, params_dict):
        """
        Creates a safe wavefront by populating the amplitude and 
        phase arrays as well as the pixel_scale. 

        Parameters
        ----------
        params_dict : dict
            A dictionary of the parameters. The following condition
            must be satisfied:
            ```py
            params_dict.get("wavelength") != None
            params_dict.get("offset") != None
            ```
            where "wavelength" points to an array of wavelengths in meters
            and offset points to an x-y tilt of the wavefront. 

        Returns
        -------
        params_dict : dict
            The `params_dict` parameter with the `Wavefront` entry 
            updated.
        """
        wavel = params_dict["wavelength"]
        # offset = params_dict["offset"]
        
        pixel_scale = self.wavefront_size/self.npix
        plane_type = dLux.PlaneType.Pupil
        
        x_angle, y_angle = params_dict["offset"]
        x_positions, y_positions = dLux.utils.coordinates \
                                  .get_pixel_coordinates(self.npix, pixel_scale)
        wavenumber = 2 * np.pi / wavel
        phase = - wavenumber * (x_positions * x_angle + y_positions * y_angle)
        
        phase = np.expand_dims(phase, 0)
        
        # phase = np.zeros([1, self.npix, self.npix])
        amplitude = np.ones([1, self.npix, self.npix])
        amplitude /= np.linalg.norm(amplitude)
        
        if self.wavefront_type == 'Cartesian':
            wavefront = dLux.CartesianWavefront(
                                        wavel, 
                                        # offset,
                                        pixel_scale,
                                        plane_type,
                                        amplitude, 
                                        phase)
            
        elif self.wavefront_type == 'Angular':
            wavefront = dLux.AngularWavefront(
                                        wavel, 
                                        # offset,
                                        pixel_scale,
                                        plane_type,
                                        amplitude, 
                                        phase)
            
        elif self.wavefront_type == 'FarFieldFresnel':
            wavefront = dLux.FarFieldFresnelWavefront(
                                        wavel, 
                                        # offset,
                                        pixel_scale,
                                        plane_type,
                                        amplitude, 
                                        phase)
            
        # Kill PlaneType Gradients
        is_leaf = lambda x: isinstance(x, dLux.PlaneType)
        def kill_gradient(x):
            if is_leaf(x):
                return jax.lax.stop_gradient(x.value)
            else:
                return x
        wavefront = jax.tree_map(kill_gradient, wavefront, is_leaf=is_leaf)

        params_dict["Wavefront"] = wavefront
        return params_dict


# TODO: Talk to @Louis abot incorporating this into the Wavefronts 
# because it is not differentiable and only relies on the state of 
# the wavefront. 
# I now think it might be a good idea to add this into the wavefront 
# class and then leave the call function as a simple call to it.
# This would allow other class to apply arbitrary tilts tracked in 
# that class.
class TiltWavefront(OpticalLayer):
    """ 
    Tilts the wavefront by the tilt_angles.

    Attributes
    ----------
    tilt_angles : Array, radians
        The (x, y) angles by which to tilt the wavefront.
    """
    tilt_angles = Array
    name : str = eqx.static_field()

    def __init__(self, tilt_angles, name='Wavefront Tilt'):
        """
        Parameters
        ----------
        tilt_angles : Array, radians
            The (x, y) angles by which to tilt the wavefront.
        name : string
            The name of the layer, which is used to index the layers
            dictionary. Default is 'Wavefront Tilt'
        """
        self.tilt_angles = np.asarray(tilt_angles, dtype=float)
        self.name = name
    
    def __call__(self, params_dict):
        """
        Applies the tilt_angle to the phase of the wavefront.

        Parameters
        ----------
        params_dict : dict
            A dictionary of the parameters. The following condition
            must be satisfied:
            ```py
            params_dict.get("Wavefront") != None
            ```

        Returns
        -------
        params_dict : dict
            The `params_dict` parameter with the `Wavefront` entry 
            updated. 
        """
        wavefront = params_dict["Wavefront"]
        tilted_wavefront = wavefront.tilt_wavefront(self.tilt_angles)
        params_dict["Wavefront"] = wavefront.tilt_wavefront(self.tilt_angles)
        return params_dict
        
        # x_angle, y_angle = self.tilt_angles
        # x_positions, y_positions = wavefront.get_pixel_coordinates()
        # wavenumber = 2 * np.pi / wavefront.get_wavelength()
        # phase = - wavenumber * (x_positions * x_angle + \
        #     y_positions * y_angle)
        # params_dict["Wavefront"] = wavefront.add_phase(phase)
        # return params_dict

    
class CircularAperture(OpticalLayer):
    """
    Multiplies the input wavefront by a pre calculated circular binary 
    (float) mask that fills the size of the array

    Attributes
    ----------
    npix : int 
        The number of pixels along one side of the aperture. This 
        parameters is not differentiable.
    array : Array[float]
        A binary `float` array describing the apertures location.
        The parameter is differentiable but refers to _Notes_ for 
        a known bug.
    name : string
        The name of the layer, which is used to index the layers
        dictionary. Default is 'Circular Aperture'
    """
    npix : int
    array : float
    name : str = eqx.static_field()

    
    def __init__(self, npix, rmin=0., rmax=1., eps=1e-8, 
                                 name='Circular Aperture'):
        """
        Parameters
        ----------
        npix : int
            The number of pixels along one side of the aperture when 
            represented as an array.
        rmin : float = 0.
            The inner radius of the Annular aperture. Note that the 
            description `Circular` is a misnomer. Additionally notice 
            that this parameter must satisfy the condition
            ```py
            0 <= rmin < rmax
            ```
        rmax : float = 1.
            The outer radius of the Anular aperture. Note that this 
            must satisfy the condition.
            ```py
            rmin < rmax <= 1.
            ``` 
        eps : float = 1e-08
            A small constant that is added to the binary aperture array
            to stablize autodiff. This parameter should not be changed 
            very often.
        name : string
            The name of the layer, which is used to index the layers
            dictionary. Default is 'Circular Aperture'
        """
        self.npix = int(npix)
        self.array = self.create_aperture(self.npix, 
            rmin = rmin, rmax = rmax) + eps
        self.name = name
   
 
    def create_aperture(self, npix, rmin, rmax):     
        """
        Produces the aperture array from the parameters; `rmin`, `rmax` 
        and `npix`.

        Parameters
        ----------
        npix : int
            The number of pixels along the side of the array that is 
            used to represent the aperture. 
        rmin : float = 0.
            The inner radius of the annular opening. This is a unitless
            quantity that must satisfy the following condition:
            ```py
            0 <= rmin < rmax
            ```
        rmax : float = 1.
            The outer radius of the annular opening. This is a unitless
            quantity and must satisfy the condition:
            ```py
            rmin < rmax <= 1.
            ```
        """   
        centre = (npix - 1.) / 2.
        normalised_coordinates = (np.arange(npix) - centre) / centre
        stacked_grids = np.array(np.meshgrid(normalised_coordinates,
            normalised_coordinates)) 
        radial_coordinates = np.sqrt(np.sum(stacked_grids ** 2, axis = 0))
        aperture = np.logical_and(radial_coordinates <= rmax,   
            radial_coordinates > rmin).astype(float)
        return aperture.astype(float)
   
 
    def __call__(self, params_dict):
        """
        Apply the aperture to the wavefront. Note that the name 
        `CircularAperture` is a misnomer since because this Module
        can also represent annular openings.

        Parameters
        ----------
        params_dict : dict
            A dictionary of the parameters. The following condition
            must be satisfied:
            ```py
            params_dict.get("Wavefront") != None
            ```

        Returns
        -------
        params_dict : dict
            The `params_dict` parameter with the `Wavefront` entry 
            updated. 
        """
        # Get relevant parameters
        wavefront = params_dict["Wavefront"]        
        wavefront = wavefront.multiply_amplitude(self.array)

        # Update Wavefront Object
        params_dict["Wavefront"] = wavefront
        return params_dict

    
class NormaliseWavefront(OpticalLayer):
    """ 
    Normalises the input wavefront using the in-built normalisation 
    
    Attributes
    ----------
    name : string
        The name of the layer, which is used to index the layers
        dictionary. Default is 'Wavefront Tilt'
    """
    name : str = eqx.static_field()
        
    def __init__(self, name='Wavefront Normalisation'):
        """
        Parameters
        ----------
        name : string
            The name of the layer, which is used to index the layers
            dictionary. Default is 'Wavefront Normalisation'
        """
        self.name = name
    
    
    def __call__(self, params_dict):
        """
        Normalise the wavefront. 

        Parameters
        ----------
        params_dict : dict
            A dictionary of the parameters. The following condition
            must be satisfied:
            ```py
            params_dict.get("Wavefront") != None
            ```

        Returns
        -------
        params_dict : dict
            The `params_dict` parameter with the `Wavefront` entry 
            updated. 
        """
        # Get relevant parameters
        wavefront = params_dict["Wavefront"]
        wavefront = wavefront.normalise()
        params_dict["Wavefront"] = wavefront
        return params_dict
    

class ApplyBasisOPD(OpticalLayer):
    """
    Adds an array of phase values to the input wavefront calculated 
    from the OPD. The phases are calculated from the basis 
    arrays, and weighted by the coefficients.
     
    Parameters
    ----------
    nterms: int, equinox.static_field
        The number of zernike terms to apply, ignoring the first two 
        radial terms: Piston, Tip, Tilt. This is not a differentiable
        parameter.
    basis: jax.numpy.ndarray
        Arrays holding the pre-calculated basis vectors. This is not a 
        differentiable parameter.
    coeffs: jax.numpy.ndarray
        Array of shape (nterns) of coefficients to be applied to each 
        basis vector.
    name : string
        The name of the layer, which is used to index the layers
        dictionary. Default is 'Apply Basis OPD'
    """
    npix: int
    basis: float
    coeffs: float
    name : str = eqx.static_field()
    
    

    def __init__(self, basis, coeffs=None, name='Apply Basis OPD'):
        """
        Parameters
        ----------
        basis : Array
            The basis polynomials (2-dimensional) to use calculating 
            the phase difference. It is assumed that the polynomials 
            have been evaluated on a square surface. That is, the 
            following condition is satisfied:
            ```py
            basis.shape[-1] == basis.shape[-2]
            ```
            The array should be 3-dimensional unless a single 
            basis vector is getting used. The leading axes (dimesion)
            should match the number of coefficients. 
        coeffs : Array = None
            The coefficients by which to weight the basis vectors.
            This is assumed to be one dimensional and have the same 
            length as the leading dimension of the `basis`. That is,
            the following conditions are met:
            ```py
            coeffs.shape = (n, )
            basis.shape = (n, m, m)
            ```
            If `None`, the default value is passed it is interpretted 
            as `np.zeros((n, ))`.
        name : string
            The name of the layer, which is used to index the layers
            dictionary. Default is 'Apply Basis OPD'
        """
        self.basis = np.array(basis).astype(float)
        self.npix = self.basis.shape[-1]
        self.coeffs = np.zeros(len(self.basis)) if coeffs is None \
                 else np.array(coeffs).astype(float)
        self.name = name


    def __call__(self, params_dict):
        """
        Calculate and apply the appropriate phase shift to the 
        wavefront.         

        Parameters
        ----------
        params_dict : dict
            A dictionary of the parameters. The following condition
            must be satisfied:
            ```py
            params_dict.get("Wavefront") != None
            ```
            It is also assumed that `self.basis.shape[-1] == 
            wavefront.shape[0]` and that both are square. 

        Returns
        -------
        params_dict : dict
            The `params_dict` parameter with the `Wavefront` entry 
            updated. 
        """
        # Get relevant parameters
        wavefront = params_dict["Wavefront"]
        wavefront = wavefront.add_opd(self.get_total_opd())
        params_dict["Wavefront"] = wavefront
        return params_dict
    

    def get_total_opd(self):
        """
        A convinience function to calculate the phase shift from the
        coefficients and the basis vectors. 

        Returns
        -------
        phase_shift : Array
            The per-pixel phase shift of the wavefront.
        """
        return np.dot(self.basis.T, self.coeffs)
    

class AddPhase(OpticalLayer):
    """ 
    Takes in an array of phase values and adds them to the phase term of the 
    input wavefront. ie wavelength independent
    
    This would represent a geometric phase optic like the TinyTol liquid crystal Pupil.
    
    Parameters
    ----------
    npix : int
        The number of pixels along the edge of the wavefront. This 
        is not a differentiable parameter.
    phase_array: float, radians
        Array of phase values to be applied to the input wavefront.
    name : string
        The name of the layer, which is used to index the layers
        dictionary. Default is 'Add Phase'
    """
    npix: int
    phase_array : float
    name : str = eqx.static_field()
    
    

    def __init__(self, phase_array, name='Add Phase'):
        """
        Parameters
        ----------
        phase_array : float, radians
            Array of phase values to be applied to the input wavefront.
        name : string
            The name of the layer, which is used to index the layers
            dictionary. Default is 'Add Phase'
        """
        self.phase_array = np.array(phase_array).astype(float)
        self.npix = int(phase_array.shape[0])
        self.name = name
    
    
    def __call__(self, params_dict):
        """
        Apply the phase shift represented by this `Layer` to the 
        wavefront.        

        Parameters
        ----------
        params_dict : dict
            A dictionary of the parameters. The following conditions
            must be satisfied:
            ```py
            params_dict.get("Wavefront") != None
            params_dict.get("Wavefront").shape == self.phase_array.shape
            ```

        Returns
        -------
        params_dict : dict
            The `params_dict` parameter with the `Wavefront` entry 
            updated. 
        """
        # Get relevant parameters
        wavefront = params_dict["Wavefront"]
        wavefront = wavefront.add_phase(self.phase_array)
        params_dict["Wavefront"] = wavefront
        return params_dict
    

class ApplyOPD(OpticalLayer):
    """ 
    Takes in an array representing the Optical Path Difference (OPD) and 
    applies the corresponding phase difference to the input wavefront. 

    This would represent an etched reflective optic, or phase plate
    
    Parameters
    ----------
    opd_array : float
        Array of OPD values to be applied to the input wavefront in 
        units of meters. This is a differentiable parameter.
    npix : int
        The number of pixels along the leading edge of the `opd_array`
        stored for debugging purposes.
    name : string
        The name of the layer, which is used to index the layers
        dictionary. Default is 'Apply OPD'
    """
    npix: int
    opd_array: float
    name : str = eqx.static_field()
    

    def __init__(self, opd_array, name='Apply OPD'):
        """
        Parameters
        ----------
        opd_array : float, meters
            The per pixel optical path differences.
        name : string
            The name of the layer, which is used to index the layers
            dictionary. Default is 'Apply OPD'
        """
        self.opd_array = np.array(opd_array).astype(float)
        self.npix = int(opd_array.shape[0])
        self.name = name
        
    def __call__(self, params_dict):
        """
        Apply the optical path difference represented by this `Layer`
        to the `Wavefront`.

        Parameters
        ----------
        params_dict : dict
            A dictionary of the parameters. The following conditions
            must be satisfied:
            ```py
            params_dict.get("Wavefront") != None
            params_dict.get("Wavefront").shape == self.phase_array.shape
            ```

        Returns
        -------
        params_dict : dict
            The `params_dict` parameter with the `Wavefront` entry 
            updated.
        """
        # Get relevant parameters
        wavefront = params_dict["Wavefront"]
        wavefront = wavefront.add_opd(self.opd_array)
        params_dict["Wavefront"] = wavefront
        return params_dict
    

class TransmissiveOptic(OpticalLayer):
    """ 
    Represents an arbitrary transmissive optic in the optical path. 
    
    Note this class does not normalise the 'transmission' between 
    0 and 1, but simply multiplies the wavefront amplitude by the 
    TransmissiveOptic.transmision array.

    Attributes
    ----------
    npix : int, eqx.static_field()
        The number of pixels along the leading edge of the wavefront.
    transmission : float
        An array representing the transmission of the aperture.
    name : string
        The name of the layer, which is used to index the layers
        dictionary. Default is 'Transmissive Optic'
    """
    npix: int
    transmission: float
    name : str = eqx.static_field()
    

    def __init__(self, transmission, name='Transmissive Optic'):
        """
        Parameters
        ----------
        transmission : Array[float]
            The array representing the transmission of the aperture.
        name : string
            The name of the layer, which is used to index the layers
            dictionary. Default is 'Transmissive Optic'
        """
        self.transmission = np.array(transmission).astype(float)
        self.npix = transmission.shape[0]
        self.name = name
        

    def __call__(self, params_dict):
        """
        Pass the wavefront through the `Aperture`.

        Parameters
        ----------
        params_dict : dict
            A dictionary of the parameters. The following conditions
            must be satisfied:
            ```py
            params_dict.get("Wavefront") != None
            params_dict.get("Wavefront").shape == self.aperture.shape
            ```

        Returns
        -------
        params_dict : dict
            The `params_dict` parameter with the `Wavefront` entry 
            updated.
        """
        # Get relevant parameters
        wavefront = params_dict["Wavefront"]
        wavefront = wavefront.multiply_amplitude(self.transmission)
        params_dict["Wavefront"] = wavefront
        return params_dict


class CompoundAperture(OpticalLayer):
    """
    Applies a series of soft-edged, circular aperture and occulters, 
    defined by their physical (x, y) positions and radii.
    Coordinates are defined paraxilly with physical units.
    All parameters are differentiable.
    
    NOTE: Needs unit testing
    
    Attributes
    ----------
    aperture_radii : Array[float], meters
        The radii of the apertures
    aperture_coords : Array[float], meters
        The (x, y) coordinates of the centers of the apertures
    occulter_radii : Array[float], meters
        The radii of the occulters
    occulter_coords : Array[float], meters
        The (x, y) coordinates of the centers of the occulters
    name : string
        The name of the layer, which is used to index the layers
        dictionary. Default is 'Compound Aperture'
    """
    aperture_radii: float
    aperture_coords: float
    occulter_radii: float
    occulter_coords: float
    name : str = eqx.static_field()
    
    
    def __init__(self, aperture_radii, aperture_coords=None, 
                 occulter_radii=None, occulter_coords=None,
                 name='Compound Aperture'):
        """
        Parameters
        ----------
        aperture_radii : Array[float], meters
            The radii of the apertures
        aperture_coords : Array[float], meters
            The (x, y) coordinates of the centers of the apertures
        occulter_radii : Array[float], meters
            The radii of the occulters
        occulter_coords : Array[float], meters
            The (x, y) coordinates of the centers of the occulters
        name : string
            The name of the layer, which is used to index the layers
            dictionary. Default is 'Compound Aperture'
        """
        self.aperture_radii = np.zeros(1)  if aperture_radii is None else \
                                np.array(aperture_radii).astype(float)
        self.occulter_radii = np.array([]) if occulter_radii is None else \
                                np.array(occulter_radii).astype(float)
        
        if self.aperture_radii.shape == ():
            self.aperture_coords = np.zeros([1, 2])
        else:
            self.aperture_coords = np.zeros([len(self.aperture_radii), 2]) \
            if aperture_coords is None \
            else np.array(aperture_coords).astype(float)
        
        if self.occulter_radii.shape == ():
            self.occulter_coords = np.zeros([1, 2])
        else:
            self.occulter_coords = np.zeros([len(self.occulter_radii), 2]) \
            if occulter_coords is None \
            else np.array(occulter_coords).astype(float)
        
        self.name = name
            

    def get_aperture(self, radius, center, xycoords, aper, vmin=1e-8, vmax=1):
        """
        Constructs a soft-edged aperture or occulter 
        This function is general and probably be moved into utils 
        since it is very general
        
        Parameters
        ----------
        radius : float, meters
            The radius of the aperture/occulter
        center : Array[float], meters
            The (x, y) center of the aperture/occulter, as measured
            from the optical axis
        xycoords : Array[float], meters
            The ((x, y), npix, npix) coordinate arrays to calculate the
            apertures/occulters within
        aper : bool
            Determines whether an aperture (True) or occulter (False)
            is calculated
        """
        
        # Shift coordinates
        xycoords -= center.reshape(2, 1, 1)
        rcoords = np.hypot(xycoords[0], xycoords[1])
        thetacoords = np.arctan2(xycoords[1], xycoords[0])

        # Wrap theta values around circle
        thetas_mapped = (thetacoords + np.pi/4)%(np.pi/2) - np.pi/4
        
        # Calculate projected pixel size
        npix = xycoords.shape[-1]
        pixel_scale = (np.max(xycoords) - np.min(xycoords))/(npix-1)
        alpha = (pixel_scale/2)*np.hypot(1, np.tan(thetas_mapped))
        
        # Get projected radial distance
        distance = radius - rcoords
        if not aper:
            distance *= -1
        
        # Fit linear slop slong projected pixel sizes/radial distances
        m = (vmax-vmin)/(2*alpha)
        b = (vmax-vmin)/2
        grey = m * distance + b
        
        # Clip to desired range
        grey_out = np.clip(grey, a_min=vmin, a_max=vmax)
        return grey_out
    
    def construct_aperture(self, diameter, npixels):
        """
        Constructs the various apertures and occulters from the and 
        combines them into a single transmission array
        
        Parameters
        ----------
        diameter : float, meters
            The diameter of the wavefront to calculate the aperture on
        npixels : int
            The linear size of the array to calculate the aperture on
        """
        
        # Map aperture function
        mapped_aperture = jax.vmap(self.get_aperture, 
                               in_axes=(0, 0, None, None))
        
        # Generate coordinate grid
        pixel_scale = diameter/npixels
        xycoords = dLux.utils.get_pixel_coordinates(npixels, pixel_scale)
        
        # Generate aperture/occulters
        outer_apers = mapped_aperture(self.aperture_radii, \
                                      self.aperture_coords, xycoords, True)
        inner_apers = mapped_aperture(self.occulter_radii, \
                                      self.occulter_coords, xycoords, False)
        
        # Bound values 
        outer_comb = np.clip(outer_apers.sum(0), a_min=0., a_max=1.)
        inner_comb = np.prod(inner_apers, axis=0)
        
        # Combine
        return outer_comb * inner_comb
    
    
    def __call__(self, params_dict):
        """
        Generates and applies the output transmission array to the Wavefront
        
        Parameters
        ----------
        params_dict : dict
            A dictionary of the parameters. The following conditions
            must be satisfied:
            ```py
            params_dict.get("Wavefront") != None
            ```
        Returns
        -------
        params_dict : dict
            The `params_dict` parameter with the `Wavefront` entry 
            updated.
        """
        WF = params_dict["Wavefront"]
        wavefront_diameter = WF.get_diameter()
        wavefront_npixels = WF.get_npixels()
        aper = self.construct_aperture(wavefront_diameter, wavefront_npixels)
        WF = WF.multiply_amplitude(aper)

        # Update Wavefront Object
        params_dict["Wavefront"] = WF
        return params_dict


class ApplyBasisCLIMB(OpticalLayer):
    """
    Adds an array of phase values to the input wavefront calculated 
    from the OPD
     
    Parameters
    ----------
    npix: int, equinox.static_field
        The number of pixels of the basis, before downsampling through CLIMB.
    basis: jax.numpy.ndarray, equinox.static_field
        Arrays holding the pre-calculated basis terms. This is a 
        differentiable parameter.
    coefficients: jax.numpy.ndarray
        Array of shape (nterns) of coefficients to be applied to each 
        Zernike term. This is a differentiable parameter.
    ideal_wavel : float
        The wavelength 
    name : string
        The name of the layer, which is used to index the layers
        dictionary. Default is 'CLIMB'
    """
    npix: int
    basis: float
    coeffs: float
    ideal_wavel: float
    name : str = eqx.static_field()
    

    # TODO: This will need to be reviewed by @LouisDesdoigts
    def __init__(self, basis, coeffs, ideal_wavel, name='CLIMB'):
        """
        Parameters
        ----------
        basis : Array
            The basis functions of the zernike polynomials precomputed
            over a square grid. If there is more than one order of the 
            Zernike polynomials getting used it is assumed that the 
            results are stored in a 3-dimensional array with the 
            leading dimension matching the number of polynomial terms
            to use. That is, the following conditions are met:
            ```py
            basis.shape = (n, m, m)
            coeffs.shape = (n, )
            ```
            where, `n` and `m` are integers representianf the number
            polynomial terms and the wavefront dimensions respectively.
        coeffs : Array
            The coefficients of the zernike polynomials defined in 
            basis. These must satisfy the condition that is described 
            above. 
        ideal_wavel : float
            The target wavelength that results in a perfect half-wave
            step. Ie the output OPD will be ideal_wavel/2
        name : string
            The name of the layer, which is used to index the layers
            dictionary. Default is 'CLIMB'
        """
        self.npix = int(basis.shape[-1])
        self.basis = np.array(basis).astype(float)
        self.coeffs = np.array(coeffs).astype(float)
        self.ideal_wavel = np.array(ideal_wavel).astype(float)
        self.name = name


    def __call__(self, params_dict):
        """
        
        """
        # Get relevant parameters
        wavefront = params_dict["Wavefront"]
        wavel = wavefront.wavelength

        # Get basis phase
        latent = self.get_opd(self.basis, self.coeffs)
        binary_phase = np.pi*self.CLIMB(latent)
        opd = self.phase_to_opd(binary_phase, self.ideal_wavel)
        wavefront = wavefront.add_opd(opd)

        params_dict["Wavefront"] = wavefront
        return params_dict
    

    def opd_to_phase(self, opd, wavel):
        return 2*np.pi*opd/wavel
    

    def phase_to_opd(self, phase, wavel):
        return phase*wavel/(2*np.pi)
    

    def get_opd(self, basis, coefficients):
        return np.dot(basis.T, coefficients)
    

    def get_total_opd(self):
        return self.get_opd(self.basis, self.coeffs)
    

    def get_binary_phase(self):
        latent = self.get_opd(self.basis, self.coeffs)
        binary_phase = np.pi*self.CLIMB(latent)
        return binary_phase
    

    def lsq_params(self, img):
        xx, yy = np.meshgrid(np.linspace(0,1,img.shape[0]),np.linspace(0,1,img.shape[1]))
        A = np.vstack([xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]).T
        matrix = np.linalg.inv(np.dot(A.T,A)).dot(A.T)
        return matrix, xx, yy, A


    def lsq(self, img):
        matrix, _, _, _ = self.lsq_params(img)
        return np.dot(matrix,img.ravel())


    def area(self, img, epsilon = 1e-15):

        a,b,c = self.lsq(img)
        a, b, c = np.where(a==0,epsilon,a), np.where(b==0,epsilon,b), np.where(c==0,epsilon,c)
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
        vmap_mask = jax.vmap(self.area, in_axes=(0))

        soft_bin = vmap_mask(flat).reshape(ppsz, ppsz)

        return soft_bin


class InformationConservingRotation(OpticalLayer):
    """
    Imagine the following scenario: you rotate and image, under the hood the 
    code interpolates into the image. You rotate the same image again and the 
    image is re-interpolated based on the old interpolations. Repeating this 
    the image will become increasingly distorted even once it is rotated back 
    into the initial coordinate system. This code aims to combat this by 
    reducing the information loss in rotations. The cost is computational 
    efficiency so this should be used with caution. 

    Parameters
    ----------
    alpha: float, radians
        The amount by which to rotate the wavefront. 
    padding: int 
        A factor by which to pad the array in the Fourier Space Representation.
        This defaults to 4 but for more efficient less accurate code use a 
        smaller number.
    """
    alpha: float
    padding: int


    def __init__(self, alpha: float, padding: int = 2):
        """
        Parameters
        ----------
        alpha: float, radians 
            The amount by which to rotate the image.
        padding: int
            This parameter cannot be differentiated and will default to 4.
            Change the parameter to 2 for a performance boost but it may 
            not result in lower accuracy.
        """
        self.alpha = np.asarray(alpha).astype(float)
        self.padding = int(padding)


    def __rotate(
            self,                                                                      
            image: float,                                                             
            rotation: float) -> float:                                                          
        """
        A simple rotation that is not information preserving. 

        Parameters
        ----------
        image: float
            The image to rotate.
        rotation: float, radians
            The amount to rotate the image. 

        Returns 
        -------
        image: float
            The rotated image. 
        """
        npix = image.shape[0]                                                          
        centre = (npix - 1) / 2                                                        
        x_pixels, y_pixels = dLux.utils.get_pixel_positions(npix)                                 
        rs, phis = dLux.utils.cart2polar(x_pixels, y_pixels)                                      
        phis += rotation                                                               
        coordinates_rot = np.roll(dLux.utils.polar2cart(rs, phis) + centre,                       
            shift=1, axis=0)                                                           
        rotated = jax.scipy.ndimage.map_coordinates(
            image, coordinates_rot, order=1)                     
        return rotated  


    def _rotate(self, image: float, alpha: float, pad: int) -> float:
        """
        Rotates the image by some amount. In the process the image is padded,
        when entering the fourier space so that FFTs can be used. 

        Parameters
        ----------
        image: Matrix
            The image to rotate. Notice that the Fourier methods we have used do 
            not cope so well with had edges in the image. 
        alpha: float, radians
            The amount by which to rotate the image. 
        pad: int = 2
            A padding factor. 

        Returns
        -------
        image: Matrix
            The input image rotated and resampled onto the intial grid. This often 
            leads to cropping of the corners in the rotated plane.         
        """
        in_shape = image.shape
        image_shape = np.array(in_shape, dtype=int) + 3 
        image = np.full(image_shape, np.nan, dtype=float)\
            .at[1 : in_shape[0] + 1, 1 : in_shape[1] + 1]\
            .set(image)

        # FFT rotation only work in the -45:+45 range
        # So I need to work out how to determine the quadrant that 
        # alpha is in and hence the 
        # number of required pi/2 rotations and angle in radians. 
        half_pi_to_1st_quadrant = alpha // (np.pi / 2)
        angle_in_1st_quadrant = - alpha + (half_pi_to_1st_quadrant * np.pi / 2)

        image = np.rot90(image, half_pi_to_1st_quadrant)\
            .at[:-1, :-1]\
            .get()  

        width, height = image.shape
        left_corner = int(((pad - 1) / 2.) * width)
        right_corner = int(((pad + 1) / 2.) * width)
        top_corner = int(((pad - 1) / 2.) * height)
        bottom_corner = int(((pad + 1) / 2.) * height)

        # Make the padded array 
        out_shape = (width * pad, height * pad)
        padded_image = np.full(out_shape, np.nan, dtype=float)\
            .at[left_corner : right_corner, top_corner : bottom_corner]\
            .set(image)

        padded_mask = np.ones(out_shape, dtype=bool)\
            .at[left_corner : right_corner, top_corner : bottom_corner]\
            .set(np.where(np.isnan(image), True, False))
        
        # Rotate the mask, to know what part is actually the image
        padded_mask = self.__rotate(padded_mask, -angle_in_1st_quadrant)

        # Replace part outside the image which are NaN by 0, and go into 
        # Fourier space.
        padded_image = np.where(np.isnan(padded_image), 0. , padded_image)

        uncentered_angular_displacement = np.tan(angle_in_1st_quadrant / 2.)
        centered_angular_displacement = -np.sin(angle_in_1st_quadrant)

        uncentered_frequencies = np.fft.fftfreq(out_shape[0])
        centered_frequencies = np.arange(-out_shape[0] / 2., out_shape[0] / 2.)

        pi_factor = -2.j * np.pi * np.ones(out_shape, dtype=float)

        uncentered_phase = np.exp(
            uncentered_angular_displacement *\
            ((pi_factor * uncentered_frequencies).T *\
            centered_frequencies).T)

        centered_phase = np.exp(
            centered_angular_displacement *\
            (pi_factor * centered_frequencies).T *\
            uncentered_frequencies)

        f1 = np.fft.ifft(
            (np.fft.fft(padded_image, axis=0).T * uncentered_phase).T, axis=0)
        
        f2 = np.fft.ifft(
            np.fft.fft(f1, axis=1) * centered_phase, axis=1)

        rotated_image = np.fft.ifft(
            (np.fft.fft(f2, axis=0).T * uncentered_phase).T, axis=0)\
            .at[padded_mask]\
            .set(np.nan)
        
        return np.real(rotated_image\
            .at[left_corner + 1 : right_corner - 1, 
                top_corner + 1 : bottom_corner - 1]\
            .get()).copy()


    def __call__(self, params: dict) -> dict:
        """
        Applying the information preserving rotation to a wavefront. 

        Parameters
        ----------
        params: dict
            A dictionary that must contain a "Wavefront" entry.

        Returns 
        -------
        params: dict
            The same dictionary but the "Wavefront" entry has been updated.
        """
        wavefront = params["Wavefront"]
        field = wavefront.get_complex_form()
        rotated_field = self._rotate(field, self.alpha, self.padding)
        # TODO: Fix updates
        rotated_wavefront = wavefront\
            .set_phase(np.angle(rotated_field))\
            .set_amplitude(np.abs(rotated_field))
        params["Wavefront"] = rotated_wavefront
        return params
