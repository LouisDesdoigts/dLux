from __future__ import annotations
import jax.numpy as np
from jax import vmap, Array
from zodiax import Base
# from dLux.utils.coordinates import get_pixel_positions
# from dLux.utils.interpolation import interpolate_field, rotate_field
import dLux.utils as dlu
import dLux


__all__ = ["Image"]


class Image(Base):
    image       : Array
    pixel_scale : Array
    units       : str

    @property
    def npixels(self):
        pass
    
    def __init__(self        : Image,
                 image       : Array,
                 pixel_scale : Array,
                 units       : str) -> Image:
        """
        Constructor for the wavefront.

        Parameters
        ----------
        """
        self.image = np.asarray(image, dtype=float)
        self.pixel_scale = np.asarray(pixel_scale, dtype=float)

        if units not in ['Angular', 'Cartesian']:
            raise ValueError("units must be either 'Angular' or 'Cartesian'.")
        self.units = units
        

    def downsample(self):
        pass

    def rotate(self): # Match poppy
        pass

    def interpolate(self):
        pass

    def crop(self):
        pass

    def distort(self): # Match poppy
        pass

    def add_noise(self): # ?
        pass