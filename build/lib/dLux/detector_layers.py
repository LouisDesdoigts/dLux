from .base import Layer
from jax.numpy import ndarray

class ApplyPixelResponse(Layer):
    pixel_response: ndarray
    
    def __init__(self, size, pixel_response):
        self.size_in = size
        self.size_out = size
        self.pixel_response = pixel_response
        
    def __call__(self, image):
        image *= self.pixel_response
        return image