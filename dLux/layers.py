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
__author__ = "Louis Desdoigts"
__date__ = "05/07/2022"


import dLux
import jax
import jax.numpy as np
import equinox as eqx
from dLux.wavefronts import PlaneType


class CreateWavefront(eqx.Module):
    """ 
    Initialises an on-axis input wavefront

    Parameters
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
        supports 'Cartesian' and 'Angular'
    """
    npix : int
    wavefront_size : float
    pixel_scale : float
    wavefront_type : str = eqx.static_field()


    def __init__(self, npix, wavefront_size, wavefront_type='Cartesian'):
        """
        Parameters
        ----------
        npix : int
            The number of pixels along one edge of the wavefront.
        wavefront_size : float, meters
            The physical dimensions of the wavefront in units of
            (radians) meters.
        """
        self.npix = int(npix)
        self.wavefront_size = np.array(wavefront_size).astype(float)
        self.pixel_scale = np.array(wavefront_size / npix)\
            .astype(float)
        assert wavefront_type in ['Cartesian', 'Angular'], "wavefront_type \
        must be either 'Cartesian' or 'Angular'"
        self.wavefront_type = str(wavefront_type)


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
            params_dict.get("Wavefront") != None
            ```

        Returns
        -------
        params_dict : dict
            The `params_dict` parameter with the `Wavefront` entry 
            updated. 
        """        
        wavel = params_dict["wavelength"]
        offset = params_dict["offset"]
        
        phase = np.zeros([1, self.npix, self.npix])
        amplitude = np.ones([1, self.npix, self.npix])
        amplitude /= np.linalg.norm(amplitude)
        
        # TODO: Make jax safe
        if self.wavefront_type is 'Cartesian':
            wavefront = dLux.CartesianWavefront(wavel, offset)
        elif self.wavefront_type is 'Angular':
            wavefront = dLux.AngularWavefront(wavel, offset)

        params_dict["Wavefront"] = wavefront\
            .set_phase(phase)\
            .set_amplitude(amplitude)\
            .set_plane_type(0)\
            .set_pixel_scale(self.pixel_scale)
        return params_dict


# TODO: Talk to @Louis abot incorporating this into the Wavefronts 
# because it is not differentiable and only relies on the state of 
# the wavefront. 
# I now think it might be a good idea to add this into the wavefront 
# class and then leave the call function as a simple call to it.
# This would allow other class to apply arbitrary tilts tracked in 
# that class.
class TiltWavefront(eqx.Module):
    """ 
    Applies a paraxial tilt by adding a phase slope
    """
    def __call__(self, params_dict):
        """
        Applies a tilt to the phase of the wavefront according to the
        offset that is stored in the `Wavefront`.

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
        x_angle, y_angle = wavefront.get_offset()
        x_positions, y_positions = wavefront.get_pixel_positions()
        wavenumber = 2 * np.pi / wavefront.get_wavelength()
        phase = - wavenumber * (x_positions * x_angle + \
            y_positions * y_angle)
        params_dict["Wavefront"] = wavefront.add_phase(phase)
        return params_dict

    
class CircularAperture(eqx.Module):
    """
    Multiplies the input wavefront by a pre calculated circular binary 
    (float) mask that fills the size of the array
    
    Notes
    -----
    - There is a known bug where gradients become `nan` if phase 
      operation are applied after this layer

    Attributes
    ----------
    npix : int 
        The number of pixels along one side of the aperture. This 
        parameters is not differentiable.
    array : Array[float]
        A binary `float` array describing the apertures location.
        The parameter is differentiable but refers to _Notes_ for 
        a known bug.
    """
    npix : int
    array : float

    
    def __init__(self, npix, rmin=0., rmax=1., eps=1e-8):
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
        """
        self.npix = int(npix)
        self.array = self.create_aperture(self.npix, 
            rmin = rmin, rmax = rmax) + eps
   
 
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

    
class NormaliseWavefront(eqx.Module):
    """ 
    Normalises the input wavefront using the in-built normalisation 
    algorithm of the wavefront. 
    """                
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
    

class ApplyBasisOPD(eqx.Module):
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
    """
    npix: int
    basis: float
    coeffs: float
    

    def __init__(self, basis, coeffs=None):
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
        """
        self.basis = np.array(basis).astype(float)
        self.npix = self.basis.shape[-1]
        self.coeffs = np.zeros(len(self.basis)) if coeffs is None \
                 else np.array(coeffs).astype(float)


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
    

class AddPhase(eqx.Module):
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
    """
    npix: int
    phase_array : float
    

    def __init__(self, phase_array):
        """
        Parameters
        ----------
        phase_array : float, radians
            Array of phase values to be applied to the input wavefront.  
        """
        self.phase_array = np.array(phase_array).astype(float)
        self.npix = int(phase_array.shape[0])
    
    
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
    

class ApplyOPD(eqx.Module):
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
    """
    npix: int
    opd_array: float
    

    def __init__(self, opd_array):
        """
        Parameters
        ----------
        opd_array : float, meters
            The per pixel optical path differences.
        """
        self.opd_array = np.array(opd_array).astype(float)
        self.npix = int(opd_array.shape[0])
        
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
    

class TransmissiveOptic(eqx.Module):
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
    """
    npix: int
    transmission: float
    

    def __init__(self, transmission):
        """
        Parameters
        ----------
        transmission : Array[float]
            The array representing the transmission of the aperture.
        """
        self.transmission = np.array(transmission).astype(float)
        self.npix = transmission.shape[0]
        

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


class ApplyBasisCLIMB(eqx.Module):
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
    """
    npix: int
    basis: float
    coeffs: float
    ideal_wavel: float
    

    # TODO: This will need to be reviewed by @LouisDesdoigts
    def __init__(self, basis, coeffs, ideal_wavel):
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
            The wavelength that perfectly CLIMBs.
        """
        # TODO: I have no idea what ideal_wavel is 
        self.npix = int(basis.shape[-1])
        self.basis = np.array(basis).astype(float)
        self.coeffs = np.array(coeffs).astype(float)
        self.ideal_wavel = np.array(ideal_wavel).astype(float)


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

