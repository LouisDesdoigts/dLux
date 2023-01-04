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
occulting_v2(occulting, x)


