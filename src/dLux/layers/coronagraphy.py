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
        """Initialise a Soummer focal-plane-mask operation.

        Parameters
        ----------
        optic : BaseOpticalLayer
            Arbitrary focal-plane optical mask, including complex or polarising masks.
        propagator : Fraunhofer
            MFT-based focal propagator used forward and in reverse.
        """
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
        """Return the focal-plane context used to resolve the wrapped optic.

        The mapping contains the focal ``wavefront``, SI-valued ``coordinates``,
        per-axis ``pixel_scale`` in canonical units, and the realised focal ``grid``.
        """
        return {
            "wavefront": wavefront,
            "coordinates": wavefront.coordinates,
            "pixel_scale": wavefront.grid.d * wavefront.grid.scale,
            "grid": wavefront.grid,
        }

    def apply_mono(self, wavefront):
        """Apply the compact focal-plane optic and return to the input pupil.

        The input is propagated to the configured compact MFT grid, only the field
        difference introduced by the optic is inverse propagated, and that difference
        is subtracted from the original pupil wavefront. The returned grid therefore
        matches the input pupil grid.
        """
        # Propagate to and apply the compact focal-plane optic
        focal = self.propagator(wavefront)
        optic = self.optic.resolve(**self.context(focal))
        difference = focal - optic(focal)

        # Inverse propagate only the field introduced by the optic
        inverse = self.propagator.set(grid=wavefront.grid, inverse=True)
        pupil_difference = inverse(difference)

        # Subtract the focal-plane modification from the original pupil
        return wavefront - pupil_difference
