from .base import Layer
import jax.numpy as np
from jax.numpy import ndarray
from equinox import static_field
from poppy.zernike import zernike_basis

class CreateWavefront(Layer):
    """
    Initialises an input wavefront
    x and y should be in radians (measured from the optical axis)
        How to pass in aperture to this robustly? As a property of the osys?
        What if we want to optimise the aperture size? 
        Shuould this exist outside of the system?
        Always propagate on axis and and shift with the offset term in MFT?
    To Do: Test this properly
    """
    pixelscale: float = static_field
    optic_size: float
    
    def __init__(self, size, optic_size):
        """
        size: Size of the array
        
        array_size: This physical size of the input wavefront (m)
            This value is used to determine the pixelscale and coordinate
            arrays that are tracked throughout propagation for fresnel
        """
        self.size_in = size
        self.size_out = size
        self.optic_size = optic_size
        self.pixelscale = optic_size/size
    
    def __call__(self, dummy_wavefront, wavel, offset, dummy_pixelscale):
        """
        offset: (offset_x, offset_y) - measured in radians deviation from the optical axis
        
        pixelscale input is always None - take definition from class property
        """
        
        npix = self.size_in
        xangle, yangle = offset
        V, U = np.indices([npix, npix], dtype=float)
        V -= (npix - 1) / 2.0
        V *= self.pixelscale
        U -= (npix - 1) / 2.0
        U *= self.pixelscale

        tiltphasor = np.exp(-2.0j * np.pi * (U*xangle + V*yangle) / wavel)
        wavefront = tiltphasor * np.ones([npix, npix]) * np.exp(1j * np.zeros([npix, npix]))
        return wavefront, self.pixelscale
    
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
    
class AddPhase(Layer):
    """
    Adds an array of values to the input wavefront
    """
    array: ndarray
    def __init__(self, size, array):
        self.size_in = size
        self.size_out = size
        self.array = array
    
    def __call__(self, complex_array, dummy_wavel, dummy_offset, pixelscale):
        
        amplitude = np.abs(complex_array)
        phase = np.angle(complex_array) + self.array
        wavefront_out = amplitude * np.exp(1j*phase)
        return wavefront_out, pixelscale
    
class ApplyOPD(Layer):
    """
    Adds an array of phase values to the input wavefront calculated from the OPD
    """
    array: ndarray
    def __init__(self, size, array):
        self.size_in = size
        self.size_out = size
        self.array = array
    
    def __call__(self, complex_array, wavel, dummy_offset, pixelscale):
        amplitude = np.abs(complex_array)
        phase = np.angle(complex_array)
        phase_in = self._opd_to_phase(self.array, wavel)
        phase_out = phase + phase_in
        wavefront_out = amplitude * np.exp(1j*phase_out)
        return wavefront_out, pixelscale
    
    def _opd_to_phase(self, opd, wavel):
        return 2*np.pi*opd/wavel
    
class ApplyZernike(Layer):
    """
    Adds an array of phase values to the input wavefront calculated from the OPD
    
    Currently relies on poppy to import zernikes
    """
    nterms: int = static_field
    basis: ndarray = static_field
    coefficients: ndarray
    
    def __init__(self, size, nterms, coefficients):
        self.size_in = size
        self.size_out = size
        self.nterms = nterms
        self.coefficients = coefficients
        
        # Load basis
        self.basis = np.array(np.nan_to_num(
            zernike_basis(nterms=nterms+3, npix=size)[3:])).T
        print("Note Zernike Ignores Piston Tip Tilt")
        
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
    pixelscale: float = static_field
    r_coords: ndarray = static_field
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
    
class PadToWavel(Layer):
    """
    To Do
    Implement this as an aleternative to interpolate
     -> How to do this with static array sizes since size out depends on wavel?
     -> Probably not possible
    
    Possibly pre-calculate array sizes and store than in osys object?
    """
    pass