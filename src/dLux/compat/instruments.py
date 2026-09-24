"""Released instrument convenience constructors."""

from math import prod
import jax.numpy as np
import dLux.utils as dlu
from ..layers.optical import Optic
from ..systems.imagers import Imager
from ..systems.detector import Detector
from .warnings import warn_deprecated
from .optical_systems import _Imager, _PointWeightsImager

Instrument = Imager
__all__ = ["Instrument", "Telescope", "Dither"]


def Telescope(optics, source, detector=None):
    warn_deprecated(
        "Telescope", "Imager", "imager.set(source=source, detector=detector)"
    )
    if not isinstance(optics, Imager):
        raise TypeError(
            "Telescope needs an Imager (including Angular/CartesianOpticalSystem); "
            "arbitrary layered propagation needs an explicit imaging plane."
        )
    if detector is not None and not isinstance(detector, Detector):
        detector = Detector(detector.layers)
    # Retain only the point-weight photometry adapter; all modelling stages and
    # return types come from the current Imager.
    # The released entrance field had amplitude 1/n² rather than unit power.
    # Encode that input factor as an ordinary layer; later pupil normalisation
    # still acts in its original position in the sequence.
    pupil = optics.pupil
    image_grid, oversample = optics.image_grid, optics.oversample
    if isinstance(optics, _Imager):
        incident = Optic(transmission=1 / np.sqrt(prod(optics.pupil_grid.n)))
        pupil = dlu.Sequential([("incident", incident), ("pupil", pupil)])
        # Released detectors own explicit Downsample layers. Treat the optical
        # subpixels as this Imager's native grid to avoid integrating them twice.
        image_grid, oversample = optics.propagation_grid, 1
    return _PointWeightsImager(
        optics.pupil_grid,
        image_grid,
        pupil,
        source=source,
        detector=detector,
        oversample=oversample,
        filter=optics.filter,
        bandpass=optics.bandpass,
        jitter=optics.jitter,
    )


def Dither(*args, **kwargs):
    raise TypeError(
        "Use jax.vmap over source positions with Imager instead of Dither; output "
        "axes are explicit."
    )
