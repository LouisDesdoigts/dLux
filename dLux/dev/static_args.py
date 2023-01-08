import equinox as eqx
import jax
import jax.numpy as np
import functools as ft

DLUX_ARCHITECTURE = "CPU"


def hypotenuse_cpu(ccoords: float) -> float:
    x: float = jax.lax.index_in_dim(ccoords, 0, keepdims=False)
    y: float = jax.lax.index_in_dim(ccoords, 1, keepdims=False)
    x_sq: float = jax.lax.integer_pow(x, 2)
    y_sq: float = jax.lax.integer_pow(y, 2)
    return jax.lax.sqrt(x_sq + y_sq)


def hypotenuse_gpu(ccoords: float) -> float:
    return jax.lax.sqrt(jax.lax.integer_pow(ccoords, 2).sum(axis = 0))


# +
hypotenuse: callable

if DLUX_ARCHITECTURE == "CPU":
    hypotenuse: callable = hypotenuse_cpu
elif DLUX_ARCHITECTURE == "GPU":
    hypotenuse: callable = hypotenuse_gpu


# -

def coords(n: int, rad: float) -> float:
    arange: float = jax.lax.iota(float, n)
    max_: float = np.array(n - 1, dtype=float)
    grid: float = arange * 2. * rad / max_ - rad
    s: int = grid.size
    shape: tuple = (1, s, s) 
    x: float = jax.lax.broadcast_in_dim(grid, shape, (2,))
    y: float = jax.lax.broadcast_in_dim(grid, shape, (1,))
    return jax.lax.concatenate([x, y], 0)


def cartesian_to_polar(coords: float) -> float:
    x: float = jax.lax.index_in_dim(coords, 0)
    y: float = jax.lax.index_in_dim(coords, 1)
    x_sq: float = jax.lax.integer_pow(x, 2)
    y_sq: float = jax.lax.integer_pow(y, 2)
    hypot: float = jax.lax.sqrt(x_sq + y_sq)
    return jax.lax.concatenate([hypot, jax.lax.atan2(x, y)], 0)


ccoords: float = coords(1024, 1.)


class CircularAperture(eqx.Module):
    def linear_soften(self: object, distances: float, nsoft: float, pixel_scale: float) -> float:
        lower: float = jax.lax.full_like(distances, 0., dtype=float)
        upper: float = jax.lax.full_like(distances, 1., dtype=float)
        inside: float = jax.lax.max(distances, lower)
        scaled: float = inside / nsoft / pixel_scale
        aperture: float = np.nanmin(scaled, upper)
        return aperture


def soften_v1() -> float:



def circular_aperture_v0(ccoords: float, r: float, pixel_scale: float, nsoft: float) -> float:
    rho: float = hypotenuse(ccoords)
    return soften_v0(r - rho, nsoft, pixel_scale)
