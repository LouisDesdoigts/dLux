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
from .helpers import (
    map2array as map2array,
    list2dictionary as list2dictionary,
    insert_layer as insert_layer,
    remove_layer as remove_layer,
)
from .interpolation import (
    # scale_array as scale_array,
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
    zernike_basis as zernike_basis,
    polike as polike,
    polike_fast as polike_fast,
    polike_basis as polike_basis,
)

from .geometry import (
    combine as combine,
    soften as soften,
    circle as circle,
    square as square,
    rectangle as rectangle,
    reg_polygon as reg_polygon,
    spider as spider,
    soft_circle as soft_circle,
    soft_square as soft_square,
    soft_rectangle as soft_rectangle,
    soft_reg_polygon as soft_reg_polygon,
    soft_spider as soft_spider,
)
