"""
Atmospheric turbulence models and related functions.
Adapted from hcipy (https://github.com/ehpor/hcipy)
"""

import jax
import jax.numpy as np

from .bessel import kv

__all__ = [
    "Cn_squared_from_fried_parameter",
    "fried_parameter_from_Cn_squared",
    "phase_covariance_von_karman",
    "phase_structure_function_von_karman",
    "power_spectral_density_von_karman",
]


def fried_parameter_from_Cn_squared(cn2, wavelength=500e-9):
    return (0.423 * cn2 * (2 * np.pi / wavelength) ** 2) ** (-3 / 5)


def Cn_squared_from_fried_parameter(r0, wavelength=500e-9):
    return r0 ** (-5 / 3) / (0.423 * (2 * np.pi / wavelength) ** 2)


def phase_covariance_von_karman(r0, L0):
    def covariance(coords):
        radius = np.maximum(np.hypot(coords[0], coords[1]), 1e-10)
        z = 2 * np.pi * radius / L0
        return (
            (L0 / r0) ** (5 / 3)
            * jax.scipy.special.gamma(11 / 6)
            / (2 ** (5 / 6) * np.pi ** (8 / 3))
            * (24 / 5 * jax.scipy.special.gamma(6 / 5)) ** (5 / 6)
            * z ** (5 / 6)
            * kv(5 / 6, z)
        )

    return covariance


def phase_structure_function_von_karman(r0, L0):
    def structure(coords):
        radius = np.hypot(coords[0], coords[1])
        z = 2 * np.pi * np.maximum(radius, 1e-10) / L0
        result = (
            (L0 / r0) ** (5 / 3)
            * 2 ** (1 / 6)
            * jax.scipy.special.gamma(11 / 6)
            / np.pi ** (8 / 3)
            * (24 / 5 * jax.scipy.special.gamma(6 / 5)) ** (5 / 6)
            * (
                jax.scipy.special.gamma(5 / 6) / 2 ** (1 / 6)
                - z ** (5 / 6) * kv(5 / 6, z)
            )
        )
        return np.where(radius == 0, 0.0, result)

    return structure


def power_spectral_density_von_karman(r0, L0):
    def psd(coords):
        u = np.hypot(coords[0], coords[1])

        return (
            0.0229
            * ((u**2 + (2 * np.pi / L0) ** 2) / (2 * np.pi) ** 2) ** (-11 / 6)
            * r0 ** (-5 / 3)
        )

    return psd
