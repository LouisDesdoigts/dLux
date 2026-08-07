"""Resolve fluxes and positions for astronomical source models."""

import jax.numpy as np
from jax import Array

__all__ = ["fluxes_from_contrast", "positions_from_sep"]


def fluxes_from_contrast(mean_flux: float, contrast: float) -> Array:
    """Return two component fluxes with the requested mean and contrast."""
    return 2 * np.array([contrast * mean_flux, mean_flux]) / (1 + contrast)


def positions_from_sep(
    position: Array, separation: float, position_angle: float
) -> Array:
    """Return two on-sky positions about a mean position.

    ``position`` and ``separation`` are in radians, ``position_angle`` is measured
    counter-clockwise, and the result has shape ``(2, 2)`` in ``(x, y)`` order.
    """
    r, phi = separation / 2, position_angle
    sep_vec = np.array([r * np.sin(phi), r * np.cos(phi)])
    return np.array([position + sep_vec, position - sep_vec])
