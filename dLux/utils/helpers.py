import jax.numpy as np
from numpy.random import rand, normal
from jax.scipy.ndimage import map_coordinates


__all__ = ["unif_rand", "norm_rand", "opd2phase", "phase2opd",
    "get_ppf", "rad2arcsec", "arcsec2rad", "cart2polar", "polar2cart",
    "scale_mask", "nyquist_pix_airy", "nyquist_pix"]


# Random Functions
def unif_rand(shape):
    return np.array(rand(np.prod(np.array(shape))).reshape(shape))

def norm_rand(mean, deviation, shape):
    return np.array(normal(mean, deviation, shape))


# Transformations
def rad2arcsec(values):
    return values * 3600 * 180 / np.pi

def arcsec2rad(values):
    return values * np.pi / (3600 * 180)

def cart2polar(x, y):
    return np.array([np.hypot(x, y), np.arctan2(y, x)])

def polar2cart(r, phi):
    return np.array([r*np.cos(phi), r*np.sin(phi)])


# Optics Functions
def opd2phase(opd, wavel):
    return 2*np.pi*opd/wavel

def phase2opd(phase, wavel):
    return phase*wavel/(2*np.pi)

def get_ppf(wavels,
            mean = True,
            aperture = 0.125,
            fl = 1.32,
            osamp = 5, 
            det_pixsize = 6.5e-6
            ):
    """
    Returns number of pixels per fringe, defaulting to toliman values
    """
    
    det_pixsize /= osamp
    if mean:
        wavels = wavels.mean()
    
    # Calcs for fringe units
    angular_fringe = wavels/aperture
    physical_fringe = angular_fringe * fl
    pix_per_fringe = det_pixsize/physical_fringe
    return pix_per_fringe


# Mask Loading
def scale_mask(mask, size=512):
    map_coords = map_coordinates
    xs = np.linspace(0, mask.shape[0], size)
    coords = np.array(np.meshgrid(xs, xs))
    mask_out = np.array(map_coords(mask, coords, order=0))
    return mask_out


# Pixsize calc helper
def nyquist_pix_airy(Nyq_rate, wavel, optic_size, focal_length):
    """ Assumes Airy disk """
    airy_fringe = focal_length * 1.22*wavel / optic_size
    det_pixelsize = 1/Nyq_rate * 0.5 * airy_fringe
    return det_pixelsize

def nyquist_pix(Nyq_rate, wavel, optic_size, focal_length):
    """ Calcualtes based on linear aperture size """
    fringe = focal_length * wavel / optic_size
    det_pixelsize = 1/Nyq_rate * 0.5 * fringe
    return det_pixelsize

