name = "utils"

# Import as modules
from . import coordinates, helpers, interpolation, math, optics, units

# Dont import all functions from modules
from .coordinates import (
    cart_to_polar as cart_to_polar,
    polar_to_cart as polar_to_cart,
    pixel_coords as pixel_coords,
    pixel_coordinates as pixel_coordinates,
)
from .helpers import list_to_dictionary as list_to_dictionary
from .interpolation import (
    scale_array as scale_array,
    downsample as downsample,
    generate_coordinates as generate_coordinates,
    scale as scale,
    rotate as rotate,
    fourier_rotate as fourier_rotate,
)
from .math import (
    factorial as factorial,
    triangular_number as triangular_number,
)
from .optics import (
    opd_to_phase as opd_to_phase,
    phase_to_opd as phase_to_opd,
    get_fringe_size as get_fringe_size,
    get_pixels_per_fringe as get_pixels_per_fringe,
    get_pixel_scale as get_pixel_scale,
    get_airy_pixel_scale as get_airy_pixel_scale,
)
from .units import (
    rad_to_arcsec as rad_to_arcsec,
    rad_to_deg as rad_to_deg,
    rad_to_arcmin as rad_to_arcmin,
    deg_to_rad as deg_to_rad,
    deg_to_arcmin as deg_to_arcmin,
    deg_to_arcsec as deg_to_arcsec,
    arcmin_to_rad as arcmin_to_rad,
    arcmin_to_deg as arcmin_to_deg,
    arcmin_to_arcsec as arcmin_to_arcsec,
    arcsec_to_rad as arcsec_to_rad,
    arcsec_to_deg as arcsec_to_deg,
    arcsec_to_arcmin as arcsec_to_arcmin,
)


# Add to __all__
modules = [
    coordinates,
    helpers,
    interpolation,
    units,
    optics,
    math,
]

__all__ = [module.__all__ for module in modules]
