"""Released ABCD constructors, evaluated entirely by current propagation kernels."""

import dLux.utils as dlu
from .propagators import _Propagation
from ..warnings import warn_deprecated

ABCDElement = dlu.ABCD
__all__ = [
    "ABCDElement",
    "ABCDFreeSpace",
    "ABCDLens",
    "ABCDMirror",
    "ABCDConjugatePlane",
    "ABCDPropagator",
    "MFTPropagator",
    "FFTPropagator",
    "ASMPropagator",
]


def ABCDFreeSpace(distance):
    warn_deprecated("ABCDFreeSpace", "ABCD", "ABCD(free_space=distance)")
    return dlu.ABCD(free_space=distance)


def ABCDLens(focal_length):
    warn_deprecated("ABCDLens", "ABCD", "ABCD(lens=focal_length)")
    return dlu.ABCD(lens=focal_length)


def ABCDMirror(radius):
    warn_deprecated("ABCDMirror", "ABCD", "ABCD(radius=radius)")
    return dlu.ABCD(radius=radius)


def ABCDConjugatePlane(focal_length):
    warn_deprecated("ABCDConjugatePlane", "ABCD", "ABCD(fraunhofer=focal_length)")
    return dlu.ABCD(fraunhofer=focal_length)


def ABCDPropagator(ABCDs, spec):
    warn_deprecated(
        "ABCDPropagator",
        "dlu.Propagator",
        "Propagator(grid_out, abcd=ABCDChain(elements))",
    )
    return _Propagation(dlu.Propagator(spec, abcd=dlu.ABCDChain(ABCDs)))


MFTPropagator = ABCDPropagator


def FFTPropagator(ABCDs, spec):
    warn_deprecated(
        "FFTPropagator(ABCDs, spec)",
        "dlu.FFTPropagator",
        "FFTPropagator(spec, abcd=ABCDChain(elements))",
    )
    if isinstance(spec, dlu.Grid):
        if spec.d is not None:
            raise ValueError(
                "FFT output sampling can not specify d; use MFTPropagator."
            )
        spec = dlu.FFTSpec(n=spec.n, c=spec.c)
    return _Propagation(dlu.FFTPropagator(spec, abcd=dlu.ABCDChain(ABCDs)))


def ASMPropagator(distance, spec):
    warn_deprecated("ASMPropagator", "dlu.FreeSpace", "FreeSpace(distance, resize=...)")
    if isinstance(spec, dlu.Grid):
        if spec.d is not None or spec.c is not None:
            raise ValueError("ASM output sampling can not specify d or c.")
        spec = dlu.ResizeSpec(n=spec.n)
    elif isinstance(spec, dlu.FFTSpec):
        # The released PadSpec used c=0 even for centre-preserving ASM resizing.
        if spec.c is not None and bool((spec.c != 0).any()):
            raise ValueError("ASM output sampling can not specify a nonzero c.")
        spec = dlu.ResizeSpec(n=spec.n, pad=spec.pad, crop=spec.crop)
    return _Propagation(dlu.FreeSpace(distance, spec))
