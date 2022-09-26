import jax.numpy as np
from numpy.random import rand, normal
from jax.scipy.ndimage import map_coordinates
from collections import OrderedDict


__all__ = ["unif_rand", "norm_rand", "opd2phase", "phase2opd",
    "get_ppf", "rad2arcsec", "arcsec2rad", "cart2polar", "polar2cart",
    "scale_mask", "nyquist_pix_airy", "nyquist_pix", "list_to_dict",
    "rotate"]


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
    """ 
    Calcualtes the pixel physical linear pixel size required 
    to sample at Nyq_rate x Nyquist sampling for the optical 
    system based on linear aperture size. Ie Nyq_rate = 1 would
    sample at *exactly* the nyquist rate, 2 pixels per fringe"""
    fringe = focal_length * wavel / optic_size
    det_pixelsize = 1/Nyq_rate * 0.5 * fringe
    return det_pixelsize


# List to dictionary function
def list_to_dict(list_in, ordered=True):
    """
    Converts some input list of dLux layers and converts them into
    an OrderedDict with the correct structure.
    """
    # Construct names list and identify repeats
    names, repeats = [], []
    for i in range(len(list_in)):

        # Check for name attribute
        if hasattr(list_in[i], 'name') and list_in[i].name is not None:
            name = list_in[i].name

        # Else take name from object
        else:
            name = str(list_in[i]).split('(')[0]

        # Check for Repeats
        if name in names:
            repeats.append(name)
        names.append(name)

    # Get list of unique repeats
    repeats = list(set(repeats))

    # Iterate over repeat names
    for i in range(len(repeats)):

        idx = 0
        # Iterate over names list and append index value to name
        for j in range(len(names)):
            if repeats[i] == names[j]:
                names[j] = names[j] + '_{}'.format(idx)
                idx += 1

    # Turn list into Dictionary
    dict_out = OrderedDict() if ordered else {}
    for i in range(len(names)):
        dict_out[names[i]] = list_in[i]

    return dict_out


def rotate(
        image: float, 
        rotation: float) -> float:
    """
    Rotate an image by some amount.

    Parameters
    ----------
    image: matrix
        The image to rotate.
    rotation: float, radians
        The amount to rotate clockwise from the positive x axis. 

    Returns 
    -------
    image: matrix
        The rotated image. 
    """
    npix = image.shape[0]
    centre = (npix - 1) / 2
    x_pixels, y_pixels = get_pixel_positions(npix)
    rs, phis = cart2polar(x_pixels, y_pixels)
    phis += rotation
    coordinates_rot = np.roll(polar2cart(rs, phis) + centre, 
        shift=1, axis=0)
    rotated = map_coordinates(image, coordinates_rot, order=1)
    return rotated
