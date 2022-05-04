from .base import Layer
import jax.numpy as np
from jax.numpy import ndarray
from equinox import static_field
from jaxinterp2d import interp2d
    
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