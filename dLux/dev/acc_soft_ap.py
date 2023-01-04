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


jax.make_jaxpr(occulting_v0, static_argnums=0)(occulting, x)

# %%timeit
occulting_v0(occulting, x)


@ft.partial(jax.jit, inline=True)
def occulting_v1(occulting: bool, x: float) -> float:
    occ: float = occulting.astype(float)
    return (1. - x) * occ + (1. - occ) * x


jax.make_jaxpr(occulting_v1)(occulting, x)

# %%timeit
occulting_v1(occulting, x)


@ft.partial(jax.jit, inline=True)
def occulting_v2(occulting: bool, x: float) -> float:
    occ: float = occulting.astype(float)
    return occ + x - 2. * x * occ 


jax.make_jaxpr(occulting_v2)(occulting, x)

# %%timeit 
socculting_v2(occulting, x)


# Specify the width of the soft fringe in pixels then use .at and set the value to .5. Discuss this with louis. 

# +
# So I want to do a soft edged aperture. 
# I also want to test if using the inline=True option is a performance boon.
# I also want to test writing my own XLA code and binding it to python. 
# -

@ft.partial(jax.jit, inline=True, static_argnums=0)
def coords_v1(n: int, rad: float) -> float:
    axes: float = np.linspace(-rad, rad, n, endpoint=True)
    return np.asarray(np.meshgrid(axes, axes))


@ft.partial(jax.jit, inline=True, static_argnums=0)
def coords_v2(n: int, rad: float) -> float:
    arange: float = jax.lax.iota(float, n)
    max_: float = np.array(n - 1, dtype=float)
    axes: float = arange * 2. * rad / max_ - rad
    return np.asarray(np.meshgrid(axes, axes))


arange: float = jax.lax.iota(float, 100)
max_: float = np.array(100 - 1, dtype=float)
axes: float = arange * 2. / max_ - 1.

test: float = np.tile(axes, (1, 100, 1))

# %%timeit
coords_v1(100, 1.)

# %%timeit
coords_v2(100, 1.)

jax.make_jaxpr(coords_v1, static_argnums=0)(100, 1.)

jax.make_jaxpr(coords_v2, static_argnums=0)(100, 1.)

coords: float = 
