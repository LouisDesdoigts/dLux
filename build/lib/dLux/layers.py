from .base import Layer
import jax.numpy as np
from jax.numpy import ndarray
from equinox import static_field
from .jaxinterp2d import interp2d
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
    
    
class ApplyPixelResponse(Layer):
    pixel_response: ndarray
    
    def __init__(self, size, pixel_response):
        self.size_in = size
        self.size_out = size
        self.pixel_response = pixel_response
        
    def __call__(self, image):
        image *= self.pixel_response
        return image