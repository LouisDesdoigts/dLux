"""Coronagraph-specific optical propagation layers."""

from .optical import BaseOpticalLayer, OpticalLayer
from .propagation import Fraunhofer

__all__ = ["SoummerFPM"]


class SoummerFPM(OpticalLayer):
    """Apply a focal-plane optical layer with the Soummer MFT algorithm.

    The focal-plane field is evaluated only on the propagator's compact output grid.
    The difference introduced by ``optic`` is inverse transformed and subtracted
    from the original pupil field. This supports scalar amplitude and phase optics,
    parametric optics, and Jones optics that promote the wavefront to a polarised
    representation.

    This implementation requires a forward MFT ``Fraunhofer`` propagator. The same
    propagator is configured for inverse propagation when returning the modified
    field to the input pupil grid.

    Parameters
    ----------
    optic : BaseOpticalLayer
        Optical layer applied on the sampled focal-plane grid. It should act as the
        identity outside the compact region represented by ``focal_spec``.
    propagator : Fraunhofer
        Forward ``Fraunhofer`` propagator configured with ``method="mft"``.

    References
    ----------
    Soummer, R., Pueyo, L., Sivaramakrishnan, A., & Vanderbei, R. J. (2007),
    "Fast computation of Lyot-style coronagraph propagation", Optics Express,
    15(24), 15935--15951. https://doi.org/10.1364/OE.15.015935
    """

    optic: BaseOpticalLayer
    propagator: Fraunhofer

    def __init__(self, optic, propagator):
        if not isinstance(optic, BaseOpticalLayer):
            raise TypeError("optic must be a BaseOpticalLayer.")
        if not isinstance(propagator, Fraunhofer):
            raise TypeError("propagator must be a Fraunhofer layer.")
        if propagator.method != "mft":
            raise ValueError("SoummerFPM requires an MFT Fraunhofer propagator.")
        if propagator.inverse:
            raise ValueError("SoummerFPM requires a forward propagator.")
        self.optic = optic
        self.propagator = propagator

    def context(self, wavefront):
        """Return focal-plane context used to resolve the wrapped optic."""
        return {
            "wavefront": wavefront,
            "coordinates": wavefront.coordinates,
            "pixel_scale": wavefront.spec.d * wavefront.spec.scale,
            "spec": wavefront.spec,
        }

    def apply_mono(self, wavefront):
        """Apply the compact focal-plane optic and return to the input pupil."""
        # Propagate to and apply the compact focal-plane optic
        focal = self.propagator(wavefront)
        optic = self.optic.resolve(**self.context(focal))
        difference = focal - optic(focal)

        # Inverse propagate only the field introduced by the optic
        inverse = self.propagator.set(spec=wavefront.spec, inverse=True)
        pupil_difference = inverse(difference)

        # Subtract the focal-plane modification from the original pupil
        return wavefront - pupil_difference
