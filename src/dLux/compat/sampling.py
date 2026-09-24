"""Translate old field sampling once, without storing duplicate metadata."""

from math import prod
import jax.numpy as np
from ..utils.grids import Grid
from .warnings import warn_deprecated


def wavefront_sampling(grid, npixels, diameter, pixel_scale, center, phasor):
    if isinstance(grid, Grid):
        if any(value is not None for value in (npixels, diameter, pixel_scale, center)):
            raise TypeError("Supply grid or released sampling arguments, not both.")
        return grid, phasor
    if grid is not None:
        if npixels is not None:
            raise TypeError("npixels was supplied twice.")
        npixels = grid
    if npixels is None:
        raise TypeError("Supply a Grid or npixels with diameter/pixel_scale.")
    if diameter is None and pixel_scale is None:
        raise ValueError("Released sampling requires diameter or pixel_scale.")
    warn_deprecated(
        "Wavefront(..., npixels, diameter/pixel_scale)",
        "Wavefront(..., grid)",
        "Grid(n=npixels, diam=diameter, unit='m'); legacy initial amplitude is "
        "retained",
    )
    grid = Grid(n=npixels, diam=diameter, d=pixel_scale, c=center, unit="m")
    # The release initialised amplitude to 1/n². Preserve its power only on this route.
    if phasor is None:
        phasor = np.ones(grid.shape, dtype=complex) / prod(grid.n)
    return grid, phasor


def psf_sampling(data, grid, pixel_scale):
    if isinstance(grid, Grid):
        if pixel_scale is not None:
            raise TypeError("Supply grid or pixel_scale, not both.")
        return grid
    if pixel_scale is not None and grid is not None:
        raise TypeError("pixel_scale was supplied twice.")
    spacing = grid if pixel_scale is None else pixel_scale
    if spacing is None:
        raise TypeError("Supply a Grid or pixel_scale.")
    warn_deprecated(
        "PSF(data, pixel_scale)",
        "PSF(data, grid)",
        "Grid(n=data.shape[-2:][::-1], d=pixel_scale); no physical unit is inferred",
    )
    spacing = np.asarray(spacing)
    # Released leading pixel-scale axes label independent images, not x/y components.
    spacing = np.stack((spacing, spacing), axis=-1)
    return Grid(n=data.shape[-2:][::-1], d=spacing)
