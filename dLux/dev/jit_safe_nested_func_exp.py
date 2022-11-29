import jax 
import jax.numpy as np
import jax.tree_util as tu

@jax.jit
def outer_func(i: int) -> callable:
    vec: list = np.zeros((10, 1), dtype=float).at[i].set(1.)

    @tu.Partial
    @jax.jit
    def inner_func(mat: list, vec: list = vec.copy()) -> list:
        return np.matmul(mat, vec)

    return inner_func

mat: list = np.ones((10, 10), dtype=float)
inner_func: callable = outer_func(0)       
out: list = inner_func(mat)

print(out)

