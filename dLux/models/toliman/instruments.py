from __future__ import annotations
import jax.numpy as np
import dLux.utils as dlu
from jax import Array, vmap
import matplotlib.pyplot as plt
import dLux


__all__ = ["Toliman"]


class Toliman(dLux.instruments.BaseInstrument):
    source : None
    optics : None
    
    def __init__(self, optics, source):
        self.optics = optics
        self.source = source
        super().__init__()
    
    def __getattr__(self, key):
        for attribute in self.__dict__.values():
            if hasattr(attribute, key):
                return getattr(attribute, key)
        # if key in self.sources.keys():
        #     return self.sources[key]
        raise AttributeError(f"{self.__class__.__name__} has no attribute "
        f"{key}.")
    
    def normalise(self):
        return self.set('source', self.source.normalise())
    
    def model(self):
        return self.optics.model(self.source)

    def full_model(self):
        return self.optics.full_model(self.source)
    
    def perturb(self, X, parameters):
        for parameter, x in zip(parameters, X):
            self = self.add(parameter, x)
        return self