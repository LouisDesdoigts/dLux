"""Residual refractive optical layers."""

from __future__ import annotations

import jax.numpy as np
import zodiax as zdx
from jax import Array

from ..fields import Wavefront
from .optical import OpticalLayer

__all__ = ["RefractiveOptic", "Wedge"]


class RefractiveOptic(OpticalLayer):
    """Apply a refractive thickness profile as optical path difference.

    Parameters
    ----------
    thickness : Array or Parametric
        Scalar or sampled material thickness in meters.
    n : Array or Parametric
        Wavelength-dependent or fixed refractive index.

    Examples
    --------
    Apply a dispersive refractive thickness profile to a chromatic wavefront:

    ```python
    import jax.numpy as np

    import dLux as dl
    import dLux.utils as dlu

    # Construct a sampled refractive thickness profile
    thickness = 1e-3 * dlu.gaussian(std=20, npixels=(128, 128), extent=64)
    optic = dl.RefractiveOptic(
        thickness=thickness,
        n=dl.CauchyIndex(coeffs=[1.5, 0.01]),
    )

    # Construct a chromatic wavefront
    grid = dl.GridSpec(n=128, diam=1.0, unit="m")
    wavelengths = np.linspace(600e-9, 700e-9, 5)
    wavefront = dl.Wavefront(wavelength=wavelengths, grid=grid)

    # Apply the wavelength-dependent optical path
    wavefront = optic(wavefront)
    ```
    """

    thickness: Array | zdx.Expression
    n: Array | zdx.Expression

    def __init__(self, thickness, n, *, alias=None):
        """Initialise a refractive thickness profile.

        Parameters
        ----------
        thickness : Array or Parametric
            Material thickness in metres.
        n : Array or Parametric
            Refractive index evaluated at the wavefront wavelength.
        """
        self.thickness = zdx.as_array(thickness, dtype=float)
        self.n = zdx.as_array(n, dtype=float)

        self.alias = alias

    def apply_mono(self, wavefront: Wavefront, **context) -> Wavefront:
        """Apply the resolved refractive optical path to one wavefront.

        Thickness is measured in metres and refractive index is resolved at the
        wavefront wavelength. The resulting OPD is applied without mutating the input.
        """
        self = self.resolve(**self.context(wavefront, **context))
        n = np.asarray(self.n - 1)
        if n.ndim:
            n = n[..., None, None]
        return wavefront.add_opd(n * self.thickness)


class Wedge(OpticalLayer):
    """Apply the optical path of a thin refractive wedge.

    The two wedge angles define a linear thickness ramp. A constant or parametric
    refractive index is resolved at the incident wavelength and converted into OPD
    before being applied to the wavefront.

    Parameters
    ----------
    angle : ArrayLike
        Two wedge angles in radians along the physical ``(x, y)`` axes.
    n : Array or Parametric
        Wavelength-dependent or fixed refractive index.
    """

    angle: Array
    n: Array | zdx.Expression

    def __init__(self, angle, n, *, alias=None):
        """Initialise a refractive wedge.

        Parameters
        ----------
        angle : ArrayLike
            Two-component wedge slope.
        n : Array or Parametric
            Refractive index evaluated at the wavefront wavelength.
        """
        self.angle = zdx.as_array(angle, dtype=float)
        if self.angle.shape != (2,):
            raise ValueError("angle must have shape (2,).")
        self.n = zdx.as_array(n, dtype=float)

        self.alias = alias

    def apply_mono(self, wavefront: Wavefront, **context) -> Wavefront:
        """Apply the wavelength-dependent optical path of the wedge.

        Wedge angle is in radians, coordinates and thickness are in metres, and the
        refractive index is resolved at each wavefront wavelength.
        """
        # Resolve the sampled wedge thickness
        self = self.resolve(**self.context(wavefront, **context))
        coordinates = wavefront.coordinates
        x, y = coordinates[..., 0, :, :], coordinates[..., 1, :, :]
        thickness = x * np.tan(self.angle[0]) + y * np.tan(self.angle[1])

        # Convert thickness into refractive optical path
        n = np.asarray(self.n - 1)
        if n.ndim:
            n = n[..., None, None]
        return wavefront.add_opd(n * thickness)
