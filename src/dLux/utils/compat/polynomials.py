"""Compatibility imports for polynomial functions now beside their classes."""

from ..numerical.polynomials import (
    polynomial_powers,
    polynomial_basis,
    eval_polynomial,
    solve_polynomial,
)
from ..optical.zernikes import Zernike

from .zernikes import (
    zernike_name,
    noll_indices,
    zernike_factors,
    eval_radial,
    eval_azimuthal,
    zernike,
    zernike_fast,
    zernike_basis,
    polike,
    polike_fast,
    polike_basis,
)

__all__ = [
    "polynomial_powers",
    "polynomial_basis",
    "eval_polynomial",
    "solve_polynomial",
    "zernike_name",
    "noll_indices",
    "radial_orders_to_indices",
    "zernike_factors",
    "eval_radial",
    "eval_azimuthal",
    "zernike",
    "zernike_fast",
    "zernike_basis",
    "polike",
    "polike_fast",
    "polike_basis",
]

radial_orders_to_indices = Zernike.radial_orders_to_indices
