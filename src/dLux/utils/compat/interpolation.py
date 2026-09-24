"""Retained interpolation functions backed by the Interp array operation."""

from __future__ import annotations

import jax.numpy as np
import interpax as ipx
from jax import Array

import dLux.utils as dlu
from .. import grids
from ..numerical.interpolation import Interp, interp
from ..propagation.fft import fourier_shift

__all__ = ["interpolate", "interp", "scale", "rotate", "fourier_shift"]


def interpolate(x, knots, values, *, method="linear", extrapolate=0.0, period=None):
    """Retained 1D table interface backed by Interp(ndim=1).

    Values have shape (K, *V); queries of shape Q return Q + V. Knots are a
    strictly increasing vector. Periodic calls delegate directly to Interpax.
    """
    x, knots, values = np.asarray(x), np.asarray(knots), np.asarray(values)
    if knots.ndim != 1 or values.ndim == 0 or values.shape[0] != knots.size:
        raise ValueError("values must start with an axis matching a knot vector.")
    if period is not None:
        evaluate = lambda values: ipx.Interpolator1D(
            knots, values, method=method, extrap=extrapolate, period=period
        )(x)
        if np.iscomplexobj(values):
            return evaluate(values.real) + 1j * evaluate(values.imag)
        return evaluate(values)
    fill = extrapolate
    if extrapolate is True:
        fill = None
    elif extrapolate is False:
        fill = np.nan
    elif extrapolate is None:
        fill = 0.0
    operation = Interp(
        np.moveaxis(values, 0, -1),
        knots,
        ndim=1,
        method=method,
        fill=fill,
    )
    output = operation.sample((x,))
    value_ndim = values.ndim - 1
    axes = tuple(range(value_ndim, output.ndim)) + tuple(range(value_ndim))
    return output.transpose(axes)


def scale(
    array: Array,
    npixels: int | tuple[int, int],
    ratio: float | Array,
    method: str = "linear",
    complex: bool = True,
) -> Array:
    """Paraxially interpolate a 2D array using per-axis sampling ratios.

    Parameters
    ----------
    array : Array
        The input field to interpolate, either in amplitude and phase, or real
        and imaginary.
    npixels : int | tuple[int, int]
        Output size in physical ``(x, y)`` axis order.
    ratio : float | Array
        Output sampling relative to the input along each physical axis.
    method : str = "linear"
        The interpolation method.
    complex : bool = True
        If the input array is complex, interpolate the real and imaginary components
        when True, or the amplitude and phase components when False.

    Returns
    -------
    array : Array
        The interpolated array.
    """
    # The released ratio is output/input pixel spacing, not object magnification.
    size_out = dlu.as_size(npixels, 2, "npixels")
    ratio = dlu.as_axis(ratio, 2, "ratio")
    grid_out = grids.Grid(n=size_out, d=1.0)
    return Interp(
        scale=1 / ratio, grid_out=grid_out, method=method, complex=complex, fill=0.0
    )(array)


def rotate(
    array: Array, angle: Array, method: str = "linear", complex: bool = True
) -> Array:
    """Rotate an array clockwise by the angle, retaining its input shape.

    Parameters
    ----------
    array : Array
        The array to rotate.
    angle : float | Array, radians
        The angle to rotate the array by.
    method : str = "linear"
        The interpolation method.
    complex : bool = True
        If the input array is complex, interpolate the real and imaginary components
        when True, or the amplitude and phase components when False.

    Returns
    -------
    array : Array
        The rotated array.
    """
    return Interp(rotation=np.asarray(angle), method=method, complex=complex, fill=0.0)(
        array
    )
