# So this is the experiment where I attempt to make `jit` safe 
# generators using `jax`. 

import jax 
import jax.numpy as np 
import functools as ft 
import jax.tree_util as jtu


@jax.jit
def count() -> int:
    i: int = 0
    while True:
        yield i
        i: int = i + 1


counter: iter = count()
zero: int = next(counter)
one: int = next(counter)
two: int = next(counter)

print("Zero: ", zero)
print("One: ", one)
print("Two: ", two)

