import jax 

def coords(n: int, rad: float) -> float:
    arange: float = jax.lax.iota(float, n)
    max_: float = np.array(n - 1, dtype=float)
    axes: float = arange * 2. * rad / max_ - rad
    
    s: int = axes.size
    shape: tuple = (1, s, s) 
    
    x: float = jax.lax.broadcast_in_dim(axes, shape, (2,))
    y: float = jax.lax.broadcast_in_dim(axes, shape, (1,))
    return jax.lax.concatenate([x, y], 0)

@jax.jit
def hypotenuse_v0(ccoords: float) -> float:
    return jax.lax.sqrt(jax.lax.integer_pow(ccoords, 2).sum(axis = 0))

@jax.jit
def hypotenuse_v1(ccoords: float) -> float:
    x: float = ccoords[0]
    y: float = ccoords[1]
    return np.hypot(x, y)

@jax.jit
def hypotenuse_v2(ccoords: float) -> float:
    x: float = ccoords[0]
    y: float = ccoords[1]
    x_sq: float = jax.lax.integer_pow(x, 2)
    y_sq: float = jax.lax.integer_pow(y, 2)
    return jax.lax.sqrt(x_sq + y_sq)

@jax.jit
def hypotenuse_v3(ccoords: float) -> float:
    x: float = jax.lax.index_in_dim(ccoords, 0)
    y: float = jax.lax.index_in_dim(ccoords, 1)
    x_sq: float = jax.lax.integer_pow(x, 2)
    y_sq: float = jax.lax.integer_pow(y, 2)
    return jax.lax.sqrt(x_sq + y_sq)

ccoords: float = coords(100, 1.)

%%timeit
hypotenuse_v0(ccoords)

%%timeit
hypotenuse_v1(ccoords)

%%timeit
hypotenuse_v2(ccoords)

%%timeit
hypotenuse_v3(ccoords)
