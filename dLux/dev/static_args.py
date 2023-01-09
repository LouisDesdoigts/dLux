import equinox as eqx
import jax
import jax.numpy as np
import functools as ft
import matplotlib as mpl
import matplotlib.pyplot as plt

# %matplotlib qt

mpl.rcParams["image.cmap"] = "Greys"

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


def cartesian_to_polar(ccoords: float) -> float:
    x: float = jax.lax.index_in_dim(ccoords, 0)
    y: float = jax.lax.index_in_dim(ccoords, 1)
    x_sq: float = jax.lax.integer_pow(x, 2)
    y_sq: float = jax.lax.integer_pow(y, 2)
    hypot: float = jax.lax.sqrt(x_sq + y_sq)
    return jax.lax.concatenate([hypot, jax.lax.atan2(x, y)], 0)


def normalise(arr: float) -> float:
    return (arr - arr.min()) / (arr.max() - arr.min())


ccoords: float = coords(1024, 1.)


class Wavefront(eqx.Module):
    wavelength: float
    radius: float
    npix: float
    pixel_scale:float
        
    def __init__(self: object, wavelength: float, radius: float, npix: int) -> object:
        self.wavelength = np.asarray(wavelength).astype(float)
        self.radius = np.asarray(radius).astype(float)
        self.npix = int(npix)
        self.pixel_scale = np.asarray(2. * radius / npix).astype(float)
    
    @ft.partial(jax.jit, inline=True, static_argnums=0)
    def __call__(self: object) -> float:
        return coords(self.npix, self.radius)


class Aperture(eqx.Module):
    nsoft: int
    radius: float
        
    def __init__(self: object, nsoft: int, radius: float) -> object:
        self.nsoft = int(nsoft)
        self.radius = np.asarray(radius).astype(float)
                
    def __call__(self: object, wavefront: object) -> float:
        ccoords: float = wavefront()
        pixel_scale: float = wavefront.pixel_scale
        rho: float = hypotenuse(ccoords)
        distances: float = self.radius - rho
        lower: float = jax.lax.full_like(distances, 0., dtype=float)
        upper: float = jax.lax.full_like(distances, 1., dtype=float)
        inside: float = jax.lax.max(distances, lower)
        scaled: float = inside / self.nsoft / pixel_scale
        aperture: float = jax.lax.min(scaled, upper)
        return aperture


wavelength: float = 550e-09
radius: float = 1.
npix: int = 1024
nsoft: int = 3

wavefront: object = Wavefront(wavelength, radius, npix)
aperture: object = Aperture(nsoft, radius)

aberrations: float = jax.random.normal(jax.random.PRNGKey(0), (npix, npix))

pupil_data: float = normalise(normalise(aberrations) + aperture(wavefront))

plt.imshow(pupil_data)
plt.colorbar()

model_args: object = jax.tree_map(lambda _: False, aperture)

model_args: object = eqx.tree_at(lambda x: x.radius, model_args, True)


@ft.partial(eqx.filter_jit, inline=True, args=(True, True, False))
def loss(data: float, model: object, wavefront: object) -> float:
    psf: float = model(wavefront)
    return jax.lax.integer_pow(data - psf, 2).sum()


def circ_ap_func(
        radius: float, 
        x: float, 
        y: float,
        rotation: float, 
        nsoft: float,
        pixel_scale: float) -> float:
    # Passing arguments to safe types. 
    centre: float = np.asarray([x, y]).astype(float)
    radius: float = np.asarray(radius).astype(float)
    rotation: float = np.asarray(rotation).astype(float)
    nsoft: float = np.asarray(nsoft).astype(float)
    
    # Organising coords
    ccoords: float = coords(npix, radius)
    
    # Translation 
    ccoords: float = ccoords - centre[:, None, None]
        
    # Rotation
    sin_alpha: float = jax.lax.sin(rotation)
    cos_alpha: float = jax.lax.cos(rotation)
    x: float = jax.lax.index_in_dim(ccoords, 0)
    y: float = jax.lax.index_in_dim(ccoords, 1)
    new_x: float = x * cos_alpha - y * sin_alpha
    new_y: float = x * sin_alpha + y * cos_alpha
    ccoords: float = jax.lax.concatenate([new_x, new_y], 0)        
        
    # Transformation 
    rho: float = hypotenuse(ccoords)
        
    # Linear softening
    distances: float = radius - rho
    lower: float = jax.lax.full_like(distances, 0., dtype=float)
    upper: float = jax.lax.full_like(distances, 1., dtype=float)
    inside: float = jax.lax.max(distances, lower)
    scaled: float = inside / nsoft / pixel_scale
    aperture: float = jax.lax.min(scaled, upper)
    return aperture


x: float = 0.
y: float = 0.
rotation: float = 0.
pixel_scale: float = 2. * radius / npix

jit_circ_ap_func: callable = jax.jit(circ_ap_func)
static_jit_circ_ap_func: callable = jax.jit(circ_ap_func, inline=True, static_argnums=(1, 2, 3, 4))

# %%timeit
jit_circ_ap_func(radius, x, y, rotation, nsoft, pixel_scale).block_until_ready()

# %%timeit
static_jit_circ_ap_func(radius, x, y, rotation, nsoft, pixel_scale).block_until_ready()

plt.imshow()


