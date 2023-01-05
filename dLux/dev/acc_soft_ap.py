# +
import jax 
import jax.numpy as np 
import functools as ft
import matplotlib as mpl
import matplotlib.pyplot as plt

# %matplotlib qt
# -

npix: int = 10000
x: float = np.ones((npix, npix), dtype=float)

occulting: bool = True


@ft.partial(jax.jit, inline=True)
def occulting(occulting: bool, x: float) -> float:
    occ: float = occulting.astype(float)
    return (1. - x) * occ + (1. - occ) * x


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


@ft.partial(jax.jit, inline=True, static_argnums=3)
def soft_annular_aperture(
        rmin: float, 
        rmax: float, 
        ccoords: float, 
        nsoft: float = 3) -> float:
    """
    Generate an annular aperture.
    
    Parameters
    ----------
    rmin: Array, meters
        The inner radius of the annulus.
    rmax: Array, meters
        The outer radius of the annulus. 
    ccoords: Array, meters
        The cartesian_coordinates.
    nsoft: Array, pixels
        The number of soft pixels. 
        
    Returns
    -------
    aperture: Array
        An anulus.
    """
    r: float = jax.lax.expand_dims(hypotenuse(ccoords), (-1,))
    
    pixel_scale: float = get_pixel_scale(ccoords)
    bounds: float = jax.lax.iota(float, nsoft) * pixel_scale
    rmins: float = rmin - bounds
    rmaxs: float = rmax + bounds
        
    aps: float = ((rmins < r) & (r < rmaxs)).astype(float)
    return aps.sum(axis = -1) / nsoft


with jax.profiler.trace("tmp/jax-trace", create_perfetto_link=True):
    soft_annular_aperture(rmin, rmax, ccoords).block_until_ready()

# +

import dLux as dl


# -

@jax.value_and_grad
@jax.jit
def annular_power(rmax: float) -> float:
    ccoords: float = coords(100, np.array([1.], dtype=float))
    annulus: float = dl.AnnularAperture(rmax, 1.)._aperture(ccoords)
    return annulus.sum()


comp_dl_annular_aperture: callable = jax.jit(dl.AnnularAperture(1., .5)._aperture, inline=True)

dl_annular_aperture: callable = dl.AnnularAperture(1., .5)._aperture

jax.make_jaxpr(dl_annular_aperture)(ccoords)

jax.make_jaxpr(soft_annular_aperture)(rmin, rmax, ccoords)

# %%timeit
comp_soft_annular_aperture(rmin, rmax, ccoords).block_until_ready()

soft_annular_aperture

# %%timeit
dl_annular_aperture(ccoords).block_until_ready()

plt.imshow(dl_annular_aperture(ccoords))

plt.imshow(soft_annular_aperture(rmin, rmax, ccoords))


@ft.partial(jax.jit, inline=True)
def soft_circular_aperture(r: float, ccoords: float, nsoft: float = 1.) -> float:
    rho: float = hypotenuse(ccoords)
    pixel_scale: float = _get_pixel_scale(ccoords)
    circ: float = (rho < r).astype(float)
    edges: float = (rho < (r + pixel_scale)).astype(float)
    return (circ + edges) / 2.


@ft.partial(jax.jit, inline=True)
def soft_square_aperture(width: float, ccoords: float) -> float:
    pixel_scale: float = get_pixel_scale(ccoords)
    acoords: float = jax.lax.abs(ccoords)
    x: float = jax.lax.index_in_dim(acoords, 0)
    y: float = jax.lax.index_in_dim(acoords, 1)
    square: float = ((x < width) & (y < width)).astype(float)
    edges: float = ((x < (width + pixel_scale)) & (y < (width + pixel_scale))).astype(float)
    return ((square + edges) / 2.).squeeze()


@ft.partial(jax.jit, inline=True)
def soft_rectangular_aperture(width: float, height: float, ccoords: float) -> float:
    pixel_scale: float = get_pixel_scale(ccoords)
    acoords: float = jax.lax.abs(ccoords)
    x: float = jax.lax.index_in_dim(acoords, 0)
    y: float = jax.lax.index_in_dim(acoords, 1)
    square: float = ((x < width) & (y < height)).astype(float)
    edges: float = ((x < (width + pixel_scale)) & (y < (height + pixel_scale))).astype(float)
    return ((square + edges) / 2.).squeeze()


@ft.partial(jax.jit, inline=True, static_argnums=0)
def soft_regular_polygonal_aperture(n: float, rmax: float, ccoords: float) -> float:
    alpha: float = np.pi / n
    pcoords: float = cart_to_polar(ccoords)
    pixel_scale: float = get_pixel_scale(ccoords)
        
    rho: float = jax.lax.index_in_dim(pcoords, 0)
    phi: float = jax.lax.index_in_dim(pcoords, 1)
    x: float = jax.lax.index_in_dim(ccoords, 0)
    y: float = jax.lax.index_in_dim(ccoords, 1)
        
    spikes: float = jax.lax.broadcasted_iota(float, (n, 1, 1), 0) * 2. * alpha
    ms: float = -1. / jax.lax.tan(spikes)
    sgn: float = jax.lax.select(
        jax.lax.ge(spikes, np.pi), 
        jax.lax.full_like(spikes, 1., dtype=float), 
        jax.lax.full_like(spikes, -1., dtype=float)
    )
    
    npix: int = x.shape[-1]
    shape: tuple = (n, npix, npix)
    dims: tuple = (0, 1, 2)
        
    dists: float = jax.lax.select(
        jax.lax.broadcast_in_dim(lax.eq(np.abs(ms), np.inf), shape, dims),
        jax.lax.broadcast_in_dim(x, shape, dims),
        sgn * (ms * x - y) / jax.lax.sqrt(1 + jax.lax.integer_pow(ms, 2))
    )
        
    ap: float = (dists < rmax).astype(float).prod(axis = 0) 
    ed: float = (dists < (rmax + pixel_scale)).astype(float).prod(axis = 0) 
    return (ap + ed) / 2.    


rmin: float = np.array([[.5]], dtype=float)
rmax: float = np.array([[1.]], dtype=float)
width: float = np.array([[[.8]]], dtype=float)
height: float = np.array([[[.9]]], dtype=float)
n: int = 6
prmax: float = np.array(.8, dtype=float)
ccoords: float = coords(1024, np.array([1.], dtype=float))

plt.imshow(soft_annular_aperture(rmin, rmax, ccoords))

plt.imshow(soft_circular_aperture(rmax, ccoords))

plt.imshow(soft_square_aperture(width, ccoords)[0])

plt.imshow(soft_rectangular_aperture(width, height, ccoords))

plt.imshow(soft_regular_polygonal_aperture(n, prmax, ccoords))

# %%timeit
soft_circular_aperture(rmax, ccoords).block_until_ready()

# %%timeit
soft_annular_aperture(rmin, rmax, ccoords).block_until_ready()

# %%timeit
soft_square_aperture(width, ccoords).block_until_ready()

# %%timeit
soft_regular_polygonal_aperture(n, prmax, ccoords).block_until_ready()

# %%timeit
_soft_square_aperture(width, ccoords)

# %%timeit
soft_square_aperture(width, ccoords)
