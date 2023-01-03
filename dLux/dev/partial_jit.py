from __future__ import annotations

import jax
import functools
import equinox
import typing
import multipledispatch

T: type = typing.TypeVar('T')

class Array(equinox.Module, typing.Generic[T]):
    x: T
    
    def __init__(self, x: T) -> Array[T]:
        self.x = jax.numpy.asarray(x)

    @functools.partial(jax.jit, inline=True)
    def __matmul__(self, y: T) -> Array[T]:
        if self.x.ndim == 0:
            return Array(self.x * y)
            
        return Array(self.x @ y)

    @functools.partial(jax.jit, inline=True)
    @multipledispatch.dispatch(int)
    def __getitem__(self, y: int) -> Array[T]:
        return Array(self.x[y])

    @functools.partial(jax.jit, inline=True)
    def __add__(self, y: T) -> Array[T]:
        return Array(self.x + y)
        
    @functools.partial(jax.jit, inline=True)
    def __sub__(self, y: T) -> Array[T]:
        return Array(self.x - y)

