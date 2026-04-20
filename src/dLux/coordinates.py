import zodiax as zdx
import jax.numpy as np

__all__ = ["Spec", "PadSpec", "CoordSpec"]


class Spec(zdx.Base):
    pass


class PadSpec(Spec):
    pad: int
    crop: int
    c: float

    def __init__(self, pad=1, crop=1, c=0.0):
        self.pad = int(pad)
        self.crop = int(crop)
        self.c = c


class CoordSpec(Spec):
    n: int
    d: float
    c: float

    def __init__(self, n=None, d=None, c=0.0):
        self.n = n
        self.d = d
        self.c = c

    @property
    def xs(self):
        if self.d is None:
            raise ValueError("d must be specified to calculate coordinates.")
        return self.c + (np.arange(self.n) - (self.n - 1) / 2) * self.d

    @property
    def fov(self):
        if self.d is None:
            raise ValueError("d must be specified to calculate FOV.")
        return self.n * self.d

    @property
    def extent(self):
        if self.d is None:
            raise ValueError("d must be specified to calculate extent.")
        return self.c - (self.n / 2) * self.d, self.c + (self.n / 2) * self.d
