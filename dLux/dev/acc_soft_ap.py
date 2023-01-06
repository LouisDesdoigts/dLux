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


def occulting(occulting: bool, x: float) -> float:
    occ: float = occulting.astype(float)
    return (1. - x) * occ + (1. - occ) * x


# +
def mesh(grid: float) -> float:
    s: int = grid.size
    shape: tuple = (1, s, s) 
    x: float = jax.lax.broadcast_in_dim(grid, shape, (2,))
    y: float = jax.lax.broadcast_in_dim(grid, shape, (1,))
    return jax.lax.concatenate([x, y], 0)

def coords(n: int, rad: float) -> float:
    arange: float = jax.lax.iota(float, n)
    max_: float = np.array(n - 1, dtype=float)
    axes: float = arange * 2. * rad / max_ - rad
    return mesh(axes)


# +
def hypotenuse(ccoords: float) -> float:
    x: float = jax.lax.index_in_dim(ccoords, 0)
    y: float = jax.lax.index_in_dim(ccoords, 1)
    return _hypotenuse(x, y)

def _hypotenuse(x: float, y: float) -> float:
    x_sq: float = jax.lax.integer_pow(x, 2)
    y_sq: float = jax.lax.integer_pow(y, 2)
    return jax.lax.sqrt(x_sq + y_sq)


# -

def get_pixel_scale(ccoords: float) -> float:
    first: float = jax.lax.slice(ccoords, (0, 0, 0), (1, 1, 1))
    second: float = jax.lax.slice(ccoords, (0, 0, 1), (1, 1, 2))
    return (second - first)


def cart_to_polar(coords: float) -> float:
    x: float = jax.lax.index_in_dim(coords, 0)
    y: float = jax.lax.index_in_dim(coords, 1)
    return jax.lax.concatenate([_hypotenuse(x, y), jax.lax.atan2(x, y)], 0)


def soft_annular_aperture(
        rmin: float, 
        rmax: float, 
        ccoords: float) -> float:
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
    rmins: float = rmin - pixel_scale * 3.
    rmaxs: float = rmax + pixel_scale * 3.
        
    aps: float = ((rmin < r) & (r < rmax)).astype(float)
    edges: float = ((rmins < r) & (r < rmaxs)).astype(float)
    return (aps + edges) / 2.


import dLux as dl


@jax.value_and_grad
def annular_power(rmax: float) -> float:
    ccoords: float = coords(100, np.array([1.], dtype=float))
    annulus: float = dl.AnnularAperture(rmax, 1.)._aperture(ccoords)
    return annulus.sum()


r: float = jax.lax.squeeze(hypotenuse(ccoords), (0,))

circ: float = jax.lax.lt(r, 1.).astype(float)
perim: float = (jax.lax.lt(r, 1.025) & ~jax.lax.lt(r, 1.)).astype(float)

fig: object = plt.figure()
axes: object = fig.subplots(1, 2)
_: object = axes[0].imshow(circ)
_: object = axes[1].imshow(perim)

comp_dl_annular_aperture: callable = jax.jit(dl.AnnularAperture(1., .5)._aperture, inline=True)

dl_annular_aperture: callable = dl.AnnularAperture(1., .5)._aperture

comp_soft_annular_aperture: callable = jax.jit(soft_annular_aperture, inline=True)


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


# This is a barrier.

def rotate(coordinates: float, rotation: float) -> float:
    x, y = coordinates[0], coordinates[1]
    new_x = np.cos(-rotation) * x + np.sin(-rotation) * y
    new_y = -np.sin(-rotation) * x + np.cos(-rotation) * y
    return np.array([new_x, new_y])


def translate(coordinates: float, centre: float) -> float:
    return coordinates - centre[:, None, None]


def strain(coordinates: float, strain: float) -> float:
    trans_coordinates: float = np.transpose(coordinates, (0, 2, 1))
    return coordinates + trans_coordinates * strain[:, None, None]


def compress(coordinates: float, compression: float) -> float:
    return coordinates * compression[:, None, None]


def transform_coords(
        coordinates: float, 
        centre: float, 
        compression: float, 
        strains: float,
        rotation: float) -> float:
    
    is_trans = (centre != np.zeros((2,), float)).any()
    coordinates = jax.lax.cond(
        is_trans,
        lambda coords: translate(coords, centre),
        lambda coords: coords, 
        coordinates
    )

    is_compr = (compression != np.ones((2,), float)).any()
    coordinates = jax.lax.cond(
        is_compr,
        lambda coords: compress(coords, compression),
        lambda coords: coords,
        coordinates
    )

    is_strain = (strains != np.zeros((2,), float)).any()
    coordinates = jax.lax.cond(
        is_strain,
        lambda coords: strain(coords, strains),
        lambda coords: coords,
        coordinates
    )

    is_rot = (rotation != 0.)
    coordinates = jax.lax.cond(
        is_rot,
        lambda coords: rotate(coords, rotation),
        lambda coords: coords,
        coordinates
    )

    return coordinates


def soften(distances: float, softening: float) -> float:
    steepness = 3. / softening * distances.shape[-1]
    return (np.tanh(steepness * distances) + 1.) / 2.


def soft_edged(coordinates: float, rmin: float, rmax: float, softening: float) -> float:
    rho = np.hypot(coordinates[0], coordinates[1])
    return soften(rho - rmin, softening) * soften(- rho + rmax, softening)


def hard_edged(coordinates:float, rmin: float, rmax: float) -> float:
    rho = np.hypot(coordinates[0], coordinates[1])
    return ((rho > rmin) * (rho < rmax)).astype(float)


def annular_aperture_v0(
        coordinates: float, 
        centre: float, 
        compression: float, 
        strains: float,
        rotation: float,
        softening: float,
        occulting: bool, 
        rmin: float,
        rmax: float) -> float:
    coordinates = transform_coords(coordinates, centre, compression, strains, rotation) 

    aperture = jax.lax.cond(
        (softening != 0.).any(),
        lambda coords: soft_edged(coords, rmin, rmax, softening),
        lambda coords: hard_edged(coords, rmin, rmax),
        coordinates
    )

    jax.lax.cond(
        occulting,
        lambda ap: (1. - ap),
        lambda ap: ap,
        aperture
    )

    return aperture


def annular_aperture_v1(
        coordinates: float, 
        centre: float, 
        compression: float, 
        strains: float,
        rotation: float,
        softening: float,
        occulting: bool, 
        rmin: float,
        rmax: float) -> float:
    
    is_trans = (centre != np.zeros((2,), float)).any()
    coordinates = jax.lax.cond(
        is_trans,
        lambda coords: translate(coords, centre),
        lambda coords: coords, 
        coordinates
    )

    is_compr = (compression != np.ones((2,), float)).any()
    coordinates = jax.lax.cond(
        is_compr,
        lambda coords: compress(coords, compression),
        lambda coords: coords,
        coordinates
    )

    is_strain = (strains != np.zeros((2,), float)).any()
    coordinates = jax.lax.cond(
        is_strain,
        lambda coords: strain(coords, strains),
        lambda coords: coords,
        coordinates
    )

    is_rot = (rotation != 0.)
    coordinates = jax.lax.cond(
        is_rot,
        lambda coords: rotate(coords, rotation),
        lambda coords: coords,
        coordinates
    )
    
    aperture = jax.lax.cond(
        (softening != 0.).any(),
        lambda coords: soft_edged(coords, rmin, rmax, softening),
        lambda coords: hard_edged(coords, rmin, rmax),
        coordinates
    )

    jax.lax.cond(
        occulting,
        lambda ap: (1. - ap),
        lambda ap: ap,
        aperture
    )

    return aperture


def simp_annular_aperture(coordinates: float) -> float:
    coordinates = transform_coords(coordinates, centre_, compression_, strain_, rotation_) 

    aperture = jax.lax.cond(
        (softening_ != 0.).any(),
        lambda coords: soft_edged(coords, rmin, rmax, softening_),
        lambda coords: hard_edged(coords, rmin, rmax),
        coordinates
    )

    jax.lax.cond(
        occulting_,
        lambda ap: (1. - ap),
        lambda ap: ap,
        aperture
    )

    return aperture


comp_annular_aperture: callable = jax.jit(annular_aperture_v0)

# %%timeit
comp_annular_aperture(
    ccoords, 
    centre_, 
    compression_, 
    strain_, 
    rotation_, 
    softening_, 
    occulting_,
    rmin,
    rmax
)

centre_: float = np.zeros((2,), dtype=float)
strain_: float = np.zeros((2,), dtype=float)
rotation_: float = np.zeros((), dtype=float)
compression_: float = np.ones((2,), dtype=float)
softening_: float = np.ones((), dtype=float)
occulting_: bool = False

dl_comp_annular_aperture: callable = jax.jit(dl.AnnularAperture(1., .5)._aperture)

# %%timeit
dl_comp_annular_aperture(ccoords)

comp_simp_annular_aperture: callable = jax.jit(simp_annular_aperture)

# %%timeit
comp_simp_annular_aperture(ccoords)


