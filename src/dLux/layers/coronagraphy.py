"""Coronagraph-specific optical propagation layers."""

import zodiax as zdx

from .optical import OpticalLayer
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
    optic : zodiax.Module
        Callable optical layer or composition, such as ``Sequential``, applied on
        the sampled focal grid. It must return a wavefront on that same grid and
        act as the identity outside the compact region represented by the grid.
    propagator : Fraunhofer
        Forward ``Fraunhofer`` propagator configured with ``method="mft"``.

    References
    ----------
    Soummer, R., Pueyo, L., Sivaramakrishnan, A., & Vanderbei, R. J. (2007),
    "Fast computation of Lyot-style coronagraph propagation", Optics Express,
    15(24), 15935--15951. https://doi.org/10.1364/OE.15.015935

    Examples
    --------
    Apply an opaque mask on a compact focal grid and return to the pupil:

    ```python
    import dLux as dl

    # Construct the pupil and compact focal-plane grids
    pupil_grid = dl.GridSpec(n=128, diam=1.0, unit="m")
    focal_grid = dl.GridSpec(n=64, d=5, unit="mas")

    # Make the focal-plane mask
    fpm_diam = 100 * focal_grid.scale  # 100 mas in radians
    fpm = dl.Complement(dl.Circle(diameter=fpm_diam))

    # Construct the Soummer propagation layer
    propagation = dl.SoummerFPM(
        optic=dl.Optic(transmission=fpm),
        propagator=dl.Fraunhofer(focal_grid),
    )

    # Construct the entrance pupil and wavefront
    pupil = dl.SimpleCircular(diameter=0.9)(pupil_grid)
    wavefront = dl.Wavefront(wavelength=650e-9, grid=pupil_grid)
    wavefront = pupil(wavefront)

    # Apply the focal-plane mask and return to the pupil
    wavefront = propagation(wavefront)
    ```
    """

    optic: zdx.Module
    propagator: Fraunhofer

    def __init__(self, optic, propagator, *, alias=None):
        """Initialise a Soummer focal-plane-mask operation.

        Parameters
        ----------
        optic : zodiax.Module
            Callable optical mask or composition with ``apply(wavefront, **context)``.
            It must preserve focal sampling; complex and Jones masks are supported.
        propagator : Fraunhofer
            MFT-based focal propagator used forward and in reverse.
        """
        if not isinstance(optic, zdx.Module) or not callable(
            getattr(optic, "apply", None)
        ):
            raise TypeError("optic must be a callable optical layer or composition.")
        if not isinstance(propagator, Fraunhofer):
            raise TypeError("propagator must be a Fraunhofer layer.")
        if propagator.method != "mft":
            raise ValueError("SoummerFPM requires an MFT Fraunhofer propagator.")
        if propagator.inverse:
            raise ValueError("SoummerFPM requires a forward propagator.")
        self.optic = optic
        self.propagator = propagator

        self.alias = alias

    def apply_mono(self, wavefront, **context):
        """Apply the compact focal-plane optic and return to the input pupil.

        The input is propagated to the configured compact MFT grid, only the field
        difference introduced by the optic is inverse propagated, and that difference
        is subtracted from the original pupil wavefront. The returned grid therefore
        matches the input pupil grid.
        """
        # Propagate to and apply the compact focal-plane optic
        focal = self.propagator(wavefront, **context)
        difference = focal - self.optic(focal, **context)

        # Inverse propagate only the field introduced by the optic
        inverse = self.propagator.set(grid=wavefront.grid, inverse=True)
        pupil_difference = inverse(difference, **context)

        # Subtract the focal-plane modification from the original pupil
        return wavefront - pupil_difference
