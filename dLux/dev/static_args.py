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
npix: int = 128
nsoft: int = 3

wavefront: object = Wavefront(wavelength, radius, npix)
aperture: object = Aperture(nsoft, radius)

aberrations: float = jax.random.normal(jax.random.PRNGKey(0), (npix, npix))

pupil_psf: float = normalise(normalise(aberrations) + aperture(wavefront))

plt.imshow(pupil_psf)
plt.colorbar()


