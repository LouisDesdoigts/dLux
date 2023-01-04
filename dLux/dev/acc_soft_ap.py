import jax 
import jax.numpy as np 
import functools as ft

npix: int = 10000
x: float = np.ones((npix, npix), dtype=float)

occulting: bool = True


@ft.partial(jax.jit, static_argnums=0, inline=True)
def occulting_v0(occulting: bool, x: float) -> float:
    y: float
    if occulting:
        y: float = (1. - x)
    else:
        y: float = x
    return y


@ft.partial(jax.jit, inline=True)
def occulting_v1(occulting: bool, x: float) -> float:
    occ: float = occulting.astype(float)
    return (1. - x) * occ + (1. - occ) * x


@ft.partial(jax.jit, inline=True)
def occulting_v2(occulting: bool, x: float) -> float:
    occ: float = occulting.astype(float)
    return occ + x - 2. * x * occ 


# Specify the width of the soft fringe in pixels then use .at and set the value to .5. Discuss this with louis. 

# +
# So I want to do a soft edged aperture. 
# I also want to test if using the inline=True option is a performance boon.
# I also want to test writing my own XLA code and binding it to python. 
# -

@ft.partial(jax.jit, inline=True, static_argnums=0)
def coords(n: int, rad: float) -> float:
    arange: float = jax.lax.iota(float, n)
    max_: float = np.array(n - 1, dtype=float)
    axes: float = arange * 2. * rad / max_ - rad
    return np.asarray(np.meshgrid(axes, axes))


# +
@ft.partial(jax.jit, inline=True)
def hypotenuse(coords: float) -> float:
    return np.sqrt(jax.lax.integer_pow(coords, 2).sum(axis = 0))

@ft.partial(jax.jit, inline=True)
def _hypotenuse(x: float, y: float) -> float:
    x_sq: float = jax.lax.integer_pow(x, 2)
    y_sq: float = jax.lax.integer_pow(y, 2)
    return jax.lax.sqrt(x_sq + y_sq)


# -

@ft.partial(jax.jit, inline=True)
def get_pixel_scale(ccoords: float) -> float:
    first: float = jax.lax.slice(ccoords, (0, 0, 0), (1, 1, 1))
    second: float = jax.lax.slice(ccoords, (0, 0, 1), (1, 1, 2))
    return (second - first).reshape(1, 1)


@ft.partial(jax.jit, inline=True)
def cart_to_polar(coords: float) -> float:
    x: float = coords[0]
    y: float = coords[1]
    return jax.lax.concatenate([_hypotenuse(x, y), jax.lax.atan2(x, y)], 0)


cart_coords: float = coords(100, 1.)
pol_coords: float = cart_to_polar(cart_coords)


@ft.partial(jax.jit, inline=True)
def soft_annular_aperture_v0(rmin: float, rmax: float, ccoords: float) -> float:
    r: float = hypotenuse(ccoords)
    pixel_scale: float = get_pixel_scale(ccoords)
    ann_ap: float = np.logical_and((rmin < r), (r < rmax)).astype(float)
    in_bound: float = np.logical_and((rmin - pixel_scale) < r, r < (rmin + pixel_scale))
    out_bound: float = np.logical_and((rmax - pixel_scale) < r, r < (rmax + pixel_scale))
    bound: float = np.logical_or(in_bound, out_bound)
    return jax.lax.select(bound, jax.lax.full_like(ann_ap, .5), ann_ap)


@ft.partial(jax.jit, inline=True)
def soft_annular_aperture_v1(rmin: float, rmax: float, ccoords: float) -> float:
    r: float = hypotenuse(ccoords)
    pixel_scale: float = get_pixel_scale(ccoords)
        
    ann_ap: float = np.logical_and((rmin < r), (r < rmax)).astype(float)
    bounds: float = np.logical_and((rmin - pixel_scale) < r, r < (rmax + pixel_scale)).astype(float)
    return (ann_ap + bounds) / 2.


ccoords: float = cart_coords
rmin: float = np.array([[.5]], dtype=float)
rmax: float = np.array([[1.]], dtype=float)

import matplotlib as mpl
import matplotlib.pyplot as plt

# %matplotlib qt
plt.imshow(soft_annular_aperture(rmin, rmax, cart_coords))

# %%timeit
soft_annular_aperture_v1(rmin, rmax, cart_coords)

jax.make_jaxpr(soft_annular_aperture_v1)(rmin, rmax, ccoords)


