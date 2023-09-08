name = "utils"

# Import as modules
from . import (
    propagation,
    coordinates,
    helpers,
    interpolation,
    math,
    optics,
    units,
    array_ops,
    zernikes,
    source,
    geometry,
)

# Dont import all functions from modules
from .propagation import FFT as FFT, MFT as MFT, fresnel_MFT as fresnel_MFT
from .coordinates import (
    cart2polar as cart2polar,
    polar2cart as polar2cart,
    pixel_coords as pixel_coords,
    nd_coords as nd_coords,
    translate_coords as translate_coords,
    compress_coords as compress_coords,
    shear_coords as shear_coords,
    rotate_coords as rotate_coords,
)
from .helpers import list2dictionary as list2dictionary, map2array as map2array
from .interpolation import (
    scale_array as scale_array,
    generate_coordinates as generate_coordinates,
    scale as scale,
    rotate as rotate,
)
from .math import (
    factorial as factorial,
    triangular_number as triangular_number,
    eval_basis as eval_basis,
    nandiv as nandiv,
)
from .optics import (
    opd2phase as opd2phase,
    phase2opd as phase2opd,
    fringe_size as fringe_size,
)
from .units import (
    rad2arcsec as rad2arcsec,
    rad2deg as rad2deg,
    rad2arcmin as rad2arcmin,
    deg2rad as deg2rad,
    deg2arcmin as deg2arcmin,
    deg2arcsec as deg2arcsec,
    arcmin2rad as arcmin2rad,
    arcmin2deg as arcmin2deg,
    arcmin2arcsec as arcmin2arcsec,
    arcsec2rad as arcsec2rad,
    arcsec2deg as arcsec2deg,
    arcsec2arcmin as arcsec2arcmin,
)
from .source import (
    fluxes_from_contrast as fluxes_from_contrast,
    positions_from_sep as positions_from_sep,
)
from .array_ops import (
    pad_to as pad_to,
    crop_to as crop_to,
    resize as resize,
    downsample as downsample,
)
from .zernikes import (
    zernike_name as zernike_name,
    noll_indices as noll_indices,
    zernike_factors as zernike_factors,
    eval_radial as eval_radial,
    eval_azimuthal as eval_azimuthal,
    zernike as zernike,
    zernike_fast as zernike_fast,
    polike as polike,
    polike_fast as polike_fast,
)

from .geometry import (
    gen_coords as gen_coords,
    shift_and_scale as shift_and_scale,
    circ_distance as circ_distance,
    square_distance as square_distance,
    rectangle_distance as rectangle_distance,
    reg_polygon_distance as reg_polygon_distance,
    spider_distance as spider_distance,
    circle as circle,
    annulus as annulus,
    square as square,
    rectangle as rectangle,
    reg_polygon as reg_polygon,
    spider as spider,
    soft_circle as soft_circle,
    soft_annulus as soft_annulus,
    soft_square as soft_square,
    soft_rectangle as soft_rectangle,
    soft_reg_polygon as soft_reg_polygon,
    soft_spider as soft_spider,
)


# Add to __all__
modules = [
    propagation,
    coordinates,
    helpers,
    interpolation,
    units,
    optics,
    math,
    array_ops,
    zernikes,
    source,
    geometry,
]

__all__ = [module.__all__ for module in modules]
