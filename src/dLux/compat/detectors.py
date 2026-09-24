"""Released detector return flags over current detector sequences."""

from ..systems.detector import DetectorSystem
from .warnings import warn_deprecated

Detector = DetectorSystem
BaseDetector = DetectorSystem
__all__ = ["BaseDetector", "Detector", "LayeredDetector"]


class _Detector(DetectorSystem):
    def __call__(self, psf, return_psf=False):
        output = self.apply(psf)
        return output if return_psf else output.data

    def model(self, psf, return_psf=False):
        return self(psf, return_psf)


def LayeredDetector(layers):
    warn_deprecated(
        "LayeredDetector",
        "DetectorSystem",
        "DetectorSystem(layers); call returns an Intensity",
    )
    return _Detector(layers)
