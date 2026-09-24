"""Released constructors which return current optical layers."""

import dLux.utils as dlu
from ...layers.optical import OpticalLayer, TiltLayer as Tilt
from ...layers.unified import NormaliseLayer as Normalise
from ...utils.sequential import BaseLayer
from ..warnings import warn_deprecated
from .optics import Optic

__all__ = [
    "BaseLayer",
    "OpticalLayer",
    "TransmissiveLayer",
    "AberratedLayer",
    "BasisLayer",
    "Tilt",
    "Normalise",
    "FourierBasis",
]


def TransmissiveLayer(transmission=None, normalise=False):
    warn_deprecated(
        "TransmissiveLayer", "Optic", "Optic(transmission=..., normalise=...)"
    )
    return Optic(transmission=transmission, normalise=normalise)


def AberratedLayer(opd=None, phase=None):
    warn_deprecated("AberratedLayer", "Optic", "Optic(opd=..., phase=...)")
    return Optic(opd=opd, phase=phase)


def BasisLayer(basis, coefficients=None, as_phase=False):
    warn_deprecated(
        "BasisLayer", "Optic", "Optic(opd=Basis(basis, coeffs=...)); update opd.coeffs"
    )
    definition = dlu.Basis(basis, coeffs=coefficients)
    return Optic(phase=definition) if as_phase else Optic(opd=definition)


def FourierBasis(npix, n_modes, coefficients=None):
    warn_deprecated(
        "FourierBasis layer",
        "Optic",
        "Optic(opd=dlu.FourierBasis(...)); update opd.coeffs",
    )
    return Optic(opd=dlu.FourierBasis(npix, n_modes, coeffs=coefficients))
