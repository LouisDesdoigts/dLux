from __future__ import annotations
import jax.numpy as np
from jax.scipy.signal import convolve
from jax import vmap, Array
from zodiax import Base
import dLux.utils as dlu
import dLux


__all__ = ["Image"]


class Image(Base):
    image       : Array
    pixel_scale : Array
    
    def __init__(self : Image, image : Array, pixel_scale : Array) -> Image:
        """
        Constructor for the wavefront.

        Parameters
        ----------
        """
        self.image = np.asarray(image, dtype=float)
        self.pixel_scale = np.asarray(pixel_scale, dtype=float)

        
    @property
    def npixels(self):
        """
        
        """
        return self.image.shape[-1]


    def downsample(self, n):
        """
        
        """
        return dlu.downsample(self.image, n, 'sum')


    def convolve(self, other):
        """
        
        """
        return self.set('image', convolve(self.image, other, mode='same'))
        

    def rotate(self, angle, order): # Match poppy
        """
        
        """
        return self.set('image', dlu.rotate(self.image, angle, order=1))


    def __mul__(self, other):
        """
        
        """
        return self.multiply('image', other)
    
    def __imul__(self, other):
        """
        
        """
        return self.__mull__(other)
    
    def __add__(self, other):
        """
        
        """
        return self.add('image', other)
    
    def __iadd__(self, other):
        """
        
        """
        return self.__add__(other)
    
    def __sub__(self, other):
        """
        
        """
        return self.subtract('image', other)
    
    def __isub__(self, other):
        """
        
        """
        return self.__sub__(other)
    
    def __div__(self, other):
        """
        
        """
        return self.divide('image', other)
    
    def __idiv__(self, other):
        """
        
        """
        return self.__div__(other)