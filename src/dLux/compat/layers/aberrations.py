"""Released Zernike calls backed by the current mode generator."""

import dLux.utils as dlu
from ..warnings import warn_deprecated

__all__ = ["Zernike", "ZernikeBasis"]


class Zernike(dlu.Zernike):
    """A current one-mode generator with the released calculate method.

    Coordinates use the released unit-radius convention (diameter 2). The
    inherited diameter field may be updated explicitly for physical coordinates.
    resolve still returns a mode stack; calculate removes the single mode axis.
    """

    def __init__(self, j):
        warn_deprecated(
            "Zernike(j)",
            "dlu.Zernike",
            "Zernike(nolls=[j]); resolve returns a mode stack",
        )
        super().__init__(nolls=[j], diameter=2.0)

    def calculate(self, coordinates, nsides=0):
        """Return one mode shaped (ny, nx), on a circle or regular polygon."""
        # The old second argument is polygon sides, never diameter.
        return self.resolve(coordinates=coordinates, n_sides=nsides)[0]


class ZernikeBasis(dlu.Zernike):
    """A current mode generator with the released calculate_basis method."""

    def __init__(self, js):
        warn_deprecated("ZernikeBasis", "dlu.Zernike", "Zernike(nolls=js)")
        super().__init__(nolls=js, diameter=2.0)

    def calculate_basis(self, coordinates, nsides=0):
        """Return modes shaped (n_modes, ny, nx), using diameter 2 by default."""
        return self.resolve(coordinates=coordinates, n_sides=nsides)
