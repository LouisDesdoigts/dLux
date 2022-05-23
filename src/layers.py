from .base import Layer
import jax.numpy as np
from jax.numpy import ndarray
from equinox import static_field
from .jaxinterp2d import interp2d
from .zernike import zernike_basis

__all__ = [
    
    # Optics Layers
    'CreateWavefront', 'TiltedWavefront', 'CircularAperture', 'Wavefront2PSF', 
    'NormaliseWavefront', 'AddPhase', 'ApplyOPD', 'ApplyZernike', 'ThinLens', 
    
    # Instrument Layers
   'ApplyPixelResponse',
    
    # Generic Layers
   'Pad', 'Crop', 'MultiplyArray', 'AddScalar', 'AddArray', 
   'MultiplyScalar', 'Interpolator', 'InvertXY', 'InvertX', 'InvertY'

]








""" All Classes in this script inherit from the Layer() base classand must
define 2 static parameters, size_in and size_out.

Each child class can either be some optical or array operation, transform
or neural network like operation. Thier __call__() function must be 
differentiable in order for things to work and follow this formatting:

    def __call__(self, wavelength, wavefront, pixelscale, offset):
        # Do things
        return wavefront, pixelscale

wavefront must be input as an array with shape (size_in, size_in) and it 
must be returned with shape (size_out, size_out).

Parameters
----------
size_in: int, equinox.static_field
    defines the linear size of the input wavefront to the __call__()
    function
size_out: int, equinox.static_field
    defines the linear size of the output wavefront to the __call__()
    function
"""







###################################################
############## Optical Layers #####################
###################################################

class CreateWavefront(Layer):
    """ 
    Initialises an on-axis input wavefront

    Parameters
    ----------
    pixelscale: float, equinox.static_field
        Units: meters/pixel
        The pixelscae of each array between each layer operation
        Its value is automatically calculated from the input values
        
    wavefront_size: float
        Units: meters
        Width of the array representing the wavefront in physical units
        
    """
    pixelscale: float = static_field()
    wavefront_size: float
    
    def __init__(self, size, wavefront_size):
        self.size_in = size
        self.size_out = size
        self.wavefront_size = wavefront_size
        self.pixelscale = wavefront_size/size
    
    def __call__(self, dummy_wavefront, wavel, dummy_offset, dummy_pixelscale):
        """
        pixelscale input is always None - take definition from class property
        
        """
        npix = self.size_in
        wavefront = np.ones([npix, npix]) * np.exp(1j * np.zeros([npix, npix]))
        return wavefront, self.pixelscale
    
class TiltedWavefront(Layer):
    """ 
    Initialises an input wavefront

    Parameters
    ----------
    pixelscale: float, equinox.static_field
        Units: meters/pixel
        The pixelscae of each array between each layer operation
        Its value is automatically calculated from the input values
        
    wavefront_size: float
        Units: meters
        Width of the array representing the wavefront in physical units
        
    """
    pixelscale: float = static_field()
    wavefront_size: float
    _XX: ndarray
    _YY: ndarray
    
    def __init__(self, size, wavefront_size, shift=0.):
        self.size_in = size
        self.size_out = size
        self.wavefront_size = wavefront_size
        self.pixelscale = wavefront_size/size
    
        c = self.size_in//2
        xs = np.arange(-c, c) + shift # Shift what is the optical axis
        self._XX, self._YY = self.pixelscale * np.array(np.meshgrid(xs, xs))
    
    def __call__(self, dummy_wavefront, wavel, offset, dummy_pixelscale):
        """
        offset: (offset_x, offset_y) - measured in radians deviation from the optical axis
        pixelscale input is always None - take definition from class property
        
        """
        xangle, yangle = offset
        npix = self.size_in
        tiltphasor = np.exp(-2.0j * np.pi * (self._YY*xangle + self._XX*yangle) / wavel)
        wavefront = tiltphasor * np.ones([npix, npix]) * np.exp(1j * np.zeros([npix, npix]))
        return wavefront, self.pixelscale
    
class CircularAperture(Layer):
    """
    Multiplies the input wavefront by a pre calculated circular binary mask
    that fills the size of the array
    __call__() is a mirror of MultiplyArray(Layer)
    """
    array: ndarray = static_field()
    def __init__(self, size):
        self.size_in = size
        self.size_out = size
        self.array = self._create_aperture(size)
    
    def __call__(self, wavefront, dummy_wavel, dummy_offset, pixelscale):
        """
        
        """
        wavefront_out = np.multiply(wavefront, self.array)
        return wavefront_out, pixelscale
    
    def _create_aperture(self, npix):
        """
        
        """
        xs = np.arange(-npix//2, npix//2)
        XX, YY = np.meshgrid(xs, xs)
        RR = np.hypot(XX, YY)
        aperture = RR < npix//2
        return aperture
    
class Wavefront2PSF(Layer):
    """ 
    Returns the modulus squared of the input wavefront
    """
    def __init__(self, size):
        self.size_in = size
        self.size_out = size
        
    def __call__(self, wavefront, dummy_wavel, dummy_offset, pixelscale):
        psf = np.abs(wavefront)**2
        return psf, pixelscale
    
class NormaliseWavefront(Layer):
    """ 
    Normalises the input wavefront
    """
    def __init__(self, size):
        self.size_in = size
        self.size_out = size
    
    def __call__(self, wavefront, dummy_wavel, dummy_offset, pixelscale):
        norm_factor = np.sqrt(np.sum(np.abs(wavefront)**2))
        norm_wavefront = wavefront/norm_factor
        return norm_wavefront, pixelscale
    
    
class ApplyZernike(Layer):
    """
    NEW DOCTRING:
        TO DO
        
    OLD DOCSTRING:
    Adds an array of phase values to the input wavefront calculated from the OPD
     
    Parameters
    ----------
    nterms: int, equinox.static_field
        The number of zernike terms to apply, ignoring the first two radial
        terms: Piston, Tip, Tilt
        
    basis: jax.numpy.ndarray, equinox.static_field
        Arrays holding the pre-calculated zernike basis terms
        
    coefficients: jax.numpy.ndarray
        Array of shape (nterns) of coefficients to be applied to each 
        Zernike term
    """
    basis: ndarray = static_field()
    coefficients: ndarray
    names: list = static_field()
    
    def __init__(self, size, coefficients, indexes=None):
        self.size_in = size
        self.size_out = size
        self.coefficients = np.array(coefficients)
        
        # Load basis
        indexes = np.arange(len(coefficients)) if indexes is None else indexes
        
        if np.max(indexes) >= 22:
            raise ValueError("Zernike indexes above 22 not supported")
            
        full_basis = np.array(np.nan_to_num(
                zernike_basis(nterms=np.max(indexes)+1, npix=size)))
        
        self.basis = np.array([full_basis[indx] for indx in indexes])
        
        # Names - Helper
        all_names = ['Piston', 'Tilt X', 'Tilt Y',
                     'Focus', 'Astigmatism 45', 'Astigmatism 0',
                     'Coma Y', 'Coma X',
                     'Trefoil Y', 'Trefoil X',
                     'Spherical', '2nd Astig 0', '2nd Astig 45',
                     'Tetrafoil 0', 'Tetrafoil 22.5',
                     '2nd coma X', '2nd coma Y', '3rd Astig X', '3rd Astig Y',
                     'Pentafoil X', 'Pentafoil Y', '5th order spherical']
        self.names = [all_names[indx] for indx in indexes]
        
    def __call__(self, complex_array, wavel, dummy_offset, pixelscale):
        amplitude = np.abs(complex_array)
        phase = np.angle(complex_array)
        zernike_opd = np.dot(self.basis.T, self.coefficients)
        zernike_phase = self._opd_to_phase(zernike_opd, wavel)
        phase_out = phase + zernike_phase
        wavefront_out = amplitude * np.exp(1j*phase_out)
        return wavefront_out, pixelscale
    
    def _opd_to_phase(self, opd, wavel):
        return 2*np.pi*opd/wavel
    
    def get_total_opd(self):
        return np.dot(self.basis.T, self.coefficients)
    
class ApplyZernikeOld(Layer):
    """ Old function, will be deleted eventually
    Adds an array of phase values to the input wavefront calculated from the OPD
    
    Currently relies on poppy to import zernikes
    To Do: 
     - Check units output from poppy basis 
     - Check order of terms output by poppy
     
    Parameters
    ----------
    nterms: int, equinox.static_field
        The number of zernike terms to apply, ignoring the first two radial
        terms: Piston, Tip, Tilt
        
    basis: jax.numpy.ndarray, equinox.static_field
        Arrays holding the pre-calculated zernike basis terms
        
    coefficients: jax.numpy.ndarray
        Array of shape (nterns) of coefficients to be applied to each 
        Zernike term
    """
    nterms: int = static_field()
    basis: ndarray = static_field()
    coefficients: ndarray
    
    def __init__(self, size, nterms, coefficients, defocus=True):
        self.size_in = size
        self.size_out = size
        self.nterms = nterms
        self.coefficients = coefficients
        
        # Load basis
        if defocus:
            self.basis = np.array(np.nan_to_num(
                zernike_basis(nterms=nterms+3, npix=size)[3:])).T
            print("Ignoring Piston Tip Tilt")
        else:
            self.basis = np.array(np.nan_to_num(
                zernike_basis(nterms=nterms+4, npix=size)[4:])).T
            print("Ignoring Piston Tip Tilt Defocus")
        
    def __call__(self, complex_array, wavel, dummy_offset, pixelscale):
        amplitude = np.abs(complex_array)
        phase = np.angle(complex_array)
        zernike_opd = np.dot(self.basis, self.coefficients)
        zernike_phase = self._opd_to_phase(zernike_opd, wavel)
        phase_out = phase + zernike_phase
        wavefront_out = amplitude * np.exp(1j*phase_out)
        return wavefront_out, pixelscale
    
    def _opd_to_phase(self, opd, wavel):
        return 2*np.pi*opd/wavel
    
    def get_total_opd(self):
        return np.dot(self.basis, self.coefficients)
    
class ThinLens(Layer):
    """
    Applies the thin-lens formula
    To Do:
    Check if the center of r_coords is on the corner or center of a pixel
    
    Parameters
    ----------
    pixelscale: float equinox.static_field
        Units: meters/pixel
        The pixelscae of each array between each layer operation
        Its value is automatically calculated from the input values
    r_coords: jax.numpy.ndarray equinox.static_field
        Pre-calcualted array defining the physical radial distance from the
        array center
    f: float equinox.static_field
        Units: meters
    """
    pixelscale: float = static_field()
    r_coords: ndarray = static_field()
    f: float
    
    def __init__(self, size, f, aperture):
        self.size_in = size
        self.size_out = size
        self.f = f
        self.pixelscale = aperture/size # m/pix ie pixel size (OF THE APERTURE)

        # Check if this matches PSF centering
        xs = np.arange(0, size) - size//2 
        XX, YY = np.meshgrid(xs, xs)
        x_coords = XX * self.pixelscale
        y_coords = YY * self.pixelscale
        self.r_coords = np.hypot(x_coords, y_coords)
        
    
    def __call__(self, wavefront, wavel, dummy_offset, pixelscale):
        """
        k: Wavenumber
        f: Focal length (m)
        x/y_coords: spatial coordinate system (m)
        """
        k = 2*np.pi / wavel
        wavefront_out = wavefront * np.exp(-0.5j * k * self.r_coords**2 * 1/self.f)
        return wavefront_out, pixelscale
    
class AddPhase(Layer):
    """ 
    
    Takes in an array of phase values and adds them to the phase term of the 
    input wavefront. ie wavelength independent
    
    This would represent a geometric phase optic like the TinyTol Pupil
    
    Parameters
    ----------
    array: jax.numpy.ndarray, equinox.static_field
        Units: radians
        Array of phase values to be applied to the input wavefront
    """
    array: ndarray
    def __init__(self, size, array):
        self.size_in = size
        self.size_out = size
        self.array = array
    
    def __call__(self, complex_array, dummy_wavel, dummy_offset, pixelscale):
        """
        
        """
        amplitude = np.abs(complex_array)
        phase = np.angle(complex_array) + self.array
        wavefront_out = amplitude * np.exp(1j*phase)
        return wavefront_out, pixelscale
    
class ApplyOPD(Layer):
    """ 
    
    Takes in an array representing the Optical Path Difference (OPD) and 
    applies the corresponding phase difference to the input wavefront. 

    This would represent an etched reflective optic, or phase plate
    
    Parameters
    ----------
    array: jax.numpy.ndarray, equinox.static_field
        Units: radians
        Array of OPD values to be applied to the input wavefront
    """
    array: ndarray
    def __init__(self, size, array):
        self.size_in = size
        self.size_out = size
        self.array = array
    
    def __call__(self, complex_array, wavel, dummy_offset, pixelscale):
        """
        
        """
        amplitude = np.abs(complex_array)
        phase = np.angle(complex_array)
        phase_in = self._opd_to_phase(self.array, wavel)
        phase_out = phase + phase_in
        wavefront_out = amplitude * np.exp(1j*phase_out)
        return wavefront_out, pixelscale
    
    def _opd_to_phase(self, opd, wavel):
        return 2*np.pi*opd/wavel
    
    
    
    
class PadToWavel(Layer):
    """ 
    To Do
    Implement this as an aleternative to interpolate
     -> How to do this with static array sizes since size out depends on wavel?
     -> Probably not possible
    
    Possibly pre-calculate array sizes and store than in osys object?
    """
    pass












######################################################
############## Instrumental Layers ###################
######################################################
    
    
class ApplyPixelResponse(Layer):
    pixel_response: ndarray
    
    def __init__(self, size, pixel_response):
        self.size_in = size
        self.size_out = size
        self.pixel_response = pixel_response
        
    def __call__(self, image):
        image *= self.pixel_response
        return image
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

######################################################
############## Generic Array Ops #####################
######################################################

class Pad(Layer):
    def __init__(self, size_in, size_out):
        self.size_in = size_in
        self.size_out = size_out
    
    def __call__(self, wavefront, dummy_wavel, dummy_offset, pixelscale):
        """
        Pads the input to the given size
        Places the array in the center
        """
        c, s = self.size_out//2, self.size_in//2
        padded = np.zeros([self.size_out, self.size_out], dtype=wavefront.dtype)
        wavefront_out = padded.at[c-s:c+s, c-s:c+s].set(wavefront)
        return wavefront_out, pixelscale
    
class Crop(Layer):
    
    def __init__(self, size_in, size_out):
        self.size_in = size_in
        self.size_out = size_out
    
    def __call__(self, wavefront, dummy_wavel, dummy_offset, pixelscale):
        """
        Crops the input to the given size
        Crops from the center
        """
        c, s = self.size_in//2, self.size_out//2
        wavefront_out = wavefront[c-s:c+s, c-s:c+s]
        return wavefront_out, pixelscale

class MultiplyArray(Layer):
    """
    Multiplies the input wavefront by an array
    """
    array: ndarray
    def __init__(self, size, array):
        self.size_in = size
        self.size_out = size
        self.array = array
    
    def __call__(self, wavefront, dummy_wavel, dummy_offset, pixelscale):
        wavefront_out = np.multiply(wavefront, self.array)
        return wavefront_out, pixelscale

class AddScalar(Layer):
    """
    Add a scalar to the input wavefront
    """
    value: float
    def __init__(self, size, value):
        self.size_in = size
        self.size_out = size
        self.value = value
        
    def __call__(self, wavefront, dummy_wavel, dummy_offset, pixelscale):
        wavefront_out = np.add(wavefront, self.value)
        return wavefront_out, pixelscale

class AddArray(Layer):
    """
    Adds an array of values to the input wavefront
    """
    array: ndarray
    def __init__(self, size, array):
        self.size_in = size
        self.size_out = size
        self.array = array
    
    def __call__(self, wavefront, dummy_wavel, dummy_offset, pixelscale):
        wavefront_out = np.add(wavefront, self.array)
        return wavefront_out, pixelscale
    
class MultiplyScalar(Layer):
    """
    Multiplies the input wavefront by a scalar
    """
    value: float
    def __init__(self, size, value):
        self.size_in = size
        self.size_out = size
        self.value = value
        
    def __call__(self, wavefront, dummy_wavel, dummy_offset, pixelscale):
        wavefront_out = np.multiply(wavefront, self.value)
        return wavefront_out, pixelscale
    
class Interpolator(Layer):
    pixelscale_out: float = static_field()
    
    def __init__(self, size_in, size_out, pixelscale_out):
        self.size_in = size_in
        self.size_out = size_out
        self.pixelscale_out = pixelscale_out
        

    def __call__(self, wavefront, wavel, dummy_offset, pixelscale):
        """
        NOTE: Poppy pads all arrays by 2 pixels before interpolating to reduce 
        edge effects - We will not do that here, chosing instead to have
        all layers as minimal as possible, and have guidelines around best 
        practice to get the best results
        """
        # Resample
        wavefront = self._interpolate(wavefront, pixelscale)
        
        # enforce conservation of energy:
        pixscale_ratio = pixelscale / self.pixelscale_out
        wavefront *= 1. / pixscale_ratio

        return wavefront, self.pixelscale_out
        

    def _interpolate(self, wavefront, pixelscale_in):
        x_in = self._make_axis(self.size_in, pixelscale_in)
        y_in = self._make_axis(self.size_in, pixelscale_in)
        x_out = self._make_axis(self.size_out, self.pixelscale_out)
        y_out = self._make_axis(self.size_out, self.pixelscale_out)

        # New Method
        shape_out = (self.size_out, self.size_out)
        XX_out, YY_out = np.meshgrid(x_out, y_out)
        
        # # Real and imag
        # real = interp2d(XX_out.flatten(), YY_out.flatten(), x_in, y_in, wavefront.real).reshape(shape_out)
        # imag = interp2d(XX_out.flatten(), YY_out.flatten(), x_in, y_in, wavefront.imag).reshape(shape_out)
        # new_wf = real + 1j * imag
        
        # Mag and Phase
        mag = interp2d(XX_out.flatten(), YY_out.flatten(), x_in, y_in, np.abs(wavefront)).reshape(shape_out)
        phase = interp2d(XX_out.flatten(), YY_out.flatten(), x_in, y_in, np.angle(wavefront)).reshape(shape_out)
        new_wf = mag * np.exp(1j*phase)
        
        return new_wf

    def _make_axis(self, npix, step):
        """ Helper function to make coordinate axis for interpolation """
        return step * np.arange(-npix // 2, npix // 2, dtype=np.float64)
    
class InvertXY(Layer):
    """
    Layer for axis invertions
    NOTE: Untested
    """
    
    def __init__(self, size):
        self.size_in = size
        self.size_out = size
        
    def __call__(self, array, dummy_wavel, dummy_offset, dummy_pixelscale):
        return array[::-1, ::-1]
    
class InvertX(Layer):
    """
    Layer for axis invertions
    NOTE: Untested
    """
    
    def __init__(self, size):
        self.size_in = size
        self.size_out = size
        
    def __call__(self, array, dummy_wavel, dummy_offset, dummy_pixelscale):
        return array[:, ::-1]
    
class InvertY(Layer):
    """
    Layer for axis invertions
    NOTE: Untested
    """
    
    def __init__(self, size):
        self.size_in = size
        self.size_out = size
        
    def __call__(self, array, dummy_wavel, dummy_offset, dummy_pixelscale):
        return array[::-1]
    
