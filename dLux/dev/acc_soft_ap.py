import jax 
import jax.lax as lax
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


@ft.partial(jax.jit, inline=True)
def hypotenuse(coords: float) -> float:
    return np.sqrt(jax.lax.integer_pow(coords, 2).sum(axis = 0))


@ft.partial(jax.jit, inline=True)
def cart_to_polar_v0(coords: float) -> float:
    return jax\
        .lax\
        .full_like(coords, 0.)\
        .at[0]\
        .set(hypotenuse(coords))\
        .at[1]\
        .set(jax.lax.atan2(coords[0], coords[1]))


@ft.partial(jax.jit, inline=True)
def cart_to_polar_v0(coords: float) -> float:
    empty: float = jax.lax.broadcast(0., coords.shape)
    return jax\
        .lax\
        .full_like(coords, 0.)\
        .at[0]\
        .set(hypotenuse(coords))\
        .at[1]\
        .set(jax.lax.atan2(coords[0], coords[1]))


jax.make_jaxpr(cart_to_polar_v0)(coords)

# %%timeit
cart_to_polar_v0(coords)

import dLux as dl

# %%timeit
sl/

jax.lax.atan2





