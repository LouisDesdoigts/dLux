import jax
import jax.numpy as np
import equinox as eqx
import dLux


__all__ = ["ApplyPixelResponse", "ApplyJitter", 
           "ApplySaturation", "AddConstant"]


class ApplyPixelResponse(dLux.base.Base):
    """
    
    """
    pixel_response: np.ndarray
    
    def __init__(self, pixel_response):
        """
        
        """
        self.pixel_response = np.array(pixel_response)
        
    def __call__(self, image):
        """
        
        """
        image *= self.pixel_response
        return image
    
class ApplyJitter(dLux.base.Base):
    """
    Convolves the output image with a gaussian kernal
    """
    kernel_size: int
    sigma: float
    
    def __init__(self, sigma, kernel_size=25):
        self.kernel_size = int(kernel_size)
        self.sigma = np.array(sigma).astype(float)
        
    def __call__(self, image):
        """
        
        """
        # Generate distribution
        x = np.linspace(-10, 10, self.kernel_size)
        window = jax.scipy.stats.norm.pdf(x,          scale=self.sigma) * \
                 jax.scipy.stats.norm.pdf(x[:, None], scale=self.sigma)
        
        # Normalise
        window /= np.sum(window)
        
        # Convolve with image
        image_out = jax.scipy.signal.convolve(image, window, mode='same')
        return image_out
    
class ApplySaturation(dLux.base.Base):
    """
    Reduces any values above self.saturation to self.saturation
    """
    saturation: float
    
    def __init__(self, saturation):
        self.saturation = np.array(saturation).astype(float)
        
    def __call__(self, image):
        """
        
        """
        # Apply saturation
        image_out = np.minimum(image, self.saturation)
        return image_out
    
class AddConstant(dLux.base.Base):
    """
    Add a constant to the output image.
    Typically used as the mean of the detector noise
    """
    value: float
    
    def __init__(self, value):
        self.value = np.array(value).astype(float)
        
    def __call__(self, image):
        """
        
        """
        return image + self.value