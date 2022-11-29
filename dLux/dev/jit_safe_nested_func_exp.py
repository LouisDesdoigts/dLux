import jax 
import jax.numpy as np
import jax.tree_util as tu


class Vector(object):
    length: int
    collection: float

    def __init__(self, *args: float):
        self.length = len(args)
        self.collection = list(args)
        

@jax.jit
def outer_func(i: int) -> callable:
    # vec: list = np.zeros((10, 1), dtype=float).at[i].set(1.)
    vec: list = [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    i: int = 3

    @tu.Partial
    @jax.jit
    def inner_func(mat: list) -> list:
        vec_: list = np.array(vec)
        return i * mat @ vec_

    return inner_func

with jax.checking_leaks() as jcl:
    mat: list = np.ones((10, 10), dtype=float)
    inner_func: callable = outer_func(0)       
    out: list = inner_func(mat)

    print(out)

