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

@ft.partial(jax.jit, inline=True)
def mesh(grid: float) -> float:
    s: int = grid.size
    shape: tuple = (1, s, s) 
    x: float = jax.lax.broadcast_in_dim(grid, shape, (2,))
    y: float = jax.lax.broadcast_in_dim(grid, shape, (1,))
    return jax.lax.concatenate([x, y], 0)


@ft.partial(jax.jit, inline=True, static_argnums=0)
def coords(n: int, rad: float) -> float:
    arange: float = jax.lax.iota(float, n)
    max_: float = np.array(n - 1, dtype=float)
    axes: float = arange * 2. * rad / max_ - rad
    return mesh(axes)


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
    x: float = jax.lax.index_in_dim(coords, 0)
    y: float = jax.lax.index_in_dim(coords, 1)
    return jax.lax.concatenate([_hypotenuse(x, y), jax.lax.atan2(x, y)], 0)


@ft.partial(jax.jit, inline=True)
def soft_annular_aperture(rmin: float, rmax: float, ccoords: float) -> float:
    r: float = hypotenuse(ccoords)
    pixel_scale: float = _get_pixel_scale(ccoords)
    ann_ap: float = ((rmin < r) & (r < rmax)).astype(float)
    bounds: float = (((rmin - pixel_scale) < r) & (r < (rmax + pixel_scale))).astype(float)
    return (ann_ap + bounds) / 2.


@ft.partial(jax.jit, inline=True)
def soft_circular_aperture(r: float, ccoords: float) -> float:
    rho: float = hypotenuse(ccoords)
    pixel_scale: float = _get_pixel_scale(ccoords)
    circ: float = (rho < r).astype(float)
    edges: float = (rho < (r + pixel_scale)).astype(float)
    return (circ + edges) / 2.


@ft.partial(jax.jit, inline=True)
def _soft_square_aperture(width: float, ccoords: float) -> float:
    pixel_scale: float = get_pixel_scale(ccoords, (1, 1, 1))
    acoords: float = jax.lax.abs(ccoords)
    square: float = (acoords < width).prod(axis = 0).astype(float)
    edges: float = (acoords < (width + pixel_scale)).prod(axis = 0).astype(float)
    return (square + edges) / 2.


@ft.partial(jax.jit, inline=True)
def soft_square_aperture(width: float, ccoords: float) -> float:
    pixel_scale: float = get_pixel_scale(ccoords, (1, 1, 1))
    acoords: float = jax.lax.abs(ccoords)
    x: float = jax.lax.index_in_dim(acoords, 0)
    y: float = jax.lax.index_in_dim(acoords, 1)
    square: float = ((x < width) & (y < width)).astype(float)
    edges: float = ((x < (width + pixel_scale)) & (y < (width + pixel_scale))).astype(float)
    return ((square + edges) / 2.).squeeze()


@ft.partial(jax.jit, inline=True)
def soft_rectangular_aperture(width: float, height: float, ccoords: float) -> float:
    pixel_scale: float = get_pixel_scale(ccoords, (1, 1, 1))
    acoords: float = jax.lax.abs(ccoords)
    x: float = jax.lax.index_in_dim(acoords, 0)
    y: float = jax.lax.index_in_dim(acoords, 1)
    square: float = ((x < width) & (y < height)).astype(float)
    edges: float = ((x < (width + pixel_scale)) & (y < (height + pixel_scale))).astype(float)
    return ((square + edges) / 2.).squeeze()



jax.numpy.broadcast_shapes((1, 100, 100), (100,))


@ft.partial(jax.jit, inline=True, static_argnums=0)
def soft_regular_polygonal_aperture(n: float, rmax: float, ccoords: float) -> float:
    alpha: float = np.pi / n
    rho: float = jax.lax.index_in_dim(pcoords, 0, axis=2)
    phi: float = jax.lax.index_in_dim(pcoords, 1, axis=2)
    x: float = jax.lax.index_in_dim(ccoords, 0, axis=2)
    y: float = jax.lax.index_in_dim(ccoords, 1, axis=2)
    spikes: float = jax.lax.iota(float, n) * 2. * alpha
    ms: float = -1. / jax.lax.tan(spikes)
    sgn: float = np.where(jax.lax.ge(spikes, np.pi), 1., -1.)
    dists: float = sgn * (ms * x - y) / jax.lax.sqrt(1 + ms ** 2)
    dists: float = np.where(lax.eq(np.abs(ms), np.inf), x, dists)
    edges: float = jax.lax.lt(dists, rmax)
    pol: float = edges.prod(axis = -1) 
    return pol 


ccoords: float = _coords(100, np.array([1.], dtype=float))
n: int = 8
rmax: float = .8

# %%timeit
soft_regular_polygonal_aperture(n, rmax, ccoords).block_until_ready()

pred: float = jax.lax.squeeze(wedge.astype(int)
hex_: float = jax.lax.select_n(pred, *dists)

# I need to look at moving the leading dimension to the bask as perhaps was intended by the `jax` creators. This might allow me to simplify the code considerably.

rmin: float = np.array([[.5]], dtype=float)
rmax: float = np.array([[1.]], dtype=float)
width: float = np.array([[[.8]]], dtype=float)
height: float = np.array([[[.9]]], dtype=float)
ccoords: float = coords(100, 1.)

# +
import matplotlib as mpl
import matplotlib.pyplot as plt

# %matplotlib qt
# -

plt.imshow(soft_annular_aperture(rmin, rmax, ccoords))

plt.imshow(soft_circular_aperture(rmax, ccoords))

plt.imshow(soft_square_aperture(width, ccoords)[0])

plt.imshow(soft_rectangular_aperture(width, height, ccoords))

# %%timeit
soft_circular_aperture(rmax, ccoords).block_until_ready()

# %%timeit
soft_annular_aperture(rmin, rmax, ccoords).block_until_ready()

# %%timeit
soft_square_aperture(width, ccoords).block_until_ready()

from jax.interpreters import xla

help(xla.xla_call)

help(xla.XlaBuilder)


