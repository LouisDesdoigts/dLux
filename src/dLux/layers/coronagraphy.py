"""Coronagraph-specific optical propagation layers."""

import zodiax as zdx

import dLux.utils as dlu
from .optical import OpticalLayer

__all__ = ["SoummerFPMLayer"]


class SoummerFPMLayer(OpticalLayer):
    """Apply a focal-plane optical layer with the Soummer MFT algorithm.

    The focal-plane field is evaluated only on the propagator's compact output grid.
    The difference introduced by ``optic`` is inverse transformed and subtracted
    from the original pupil field. This supports scalar amplitude and phase optics,
    parametric optics, and Jones optics that promote the wavefront to a polarised
    representation.

    This implementation requires a forward numerical MFT ``Propagator``. The same
    propagator is configured for inverse propagation when returning the modified
    field to the input pupil grid.

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
    import dLux.utils as dlu

    # Construct the pupil and compact focal-plane grids
    pupil_grid = dl.Grid(n=128, diam=1.0, unit="m")
    focal_grid = dl.Grid(n=64, d=5, unit="mas")

    # Make the focal-plane mask
    fpm_diam = 100 * dlu.unit_factor(focal_grid.unit)  # 100 mas in radians
    fpm = dl.Complement(dl.Circle(diameter=fpm_diam))

    # Construct the Soummer propagation layer
    propagation = dl.SoummerFPMLayer(
        optic=dl.Optic(transmission=fpm),
        propagator=dlu.Propagator(focal_grid),
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
    propagator: dlu.Propagator

    def __init__(self, optic, propagator, *, alias=None):
        """Initialise a Soummer focal-plane-mask operation.

        Parameters
        ----------
        optic : zodiax.Module
            Callable optical mask or composition with ``apply(wavefront, **context)``.
            It must preserve focal sampling; complex and Jones masks are supported.
        propagator : dlu.Propagator
            MFT-based focal propagator used forward and in reverse.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        if not isinstance(optic, zdx.Module) or not callable(
            getattr(optic, "apply", None)
        ):
            raise TypeError("optic must be a callable optical layer or composition.")
        if not isinstance(propagator, dlu.Propagator) or not isinstance(
            propagator.grid_out, dlu.Grid
        ):
            raise TypeError("propagator must be a numerical explicit-grid Propagator.")
        if propagator.abcd is not None or propagator.defocus is not None:
            raise ValueError(
                "SoummerFPMLayer requires an undefocused conjugate Propagator."
            )
        if propagator.inverse:
            raise ValueError("SoummerFPMLayer requires a forward propagator.")
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
        self = self.populate()
        focal = wavefront.propagate(self.propagator, **context)
        masked = self.optic(focal, **context)
        return self.return_to_pupil(wavefront, focal, masked, **context)

    def return_to_pupil(self, pupil, focal, masked, **context):
        """Recombine a supplied focal-mask correction with its original pupil.

        Keeping these fields explicit lets a system share the pupil or focal field
        with another optical branch without repeating propagation.
        """
        self = self.populate()
        difference = focal - masked
        inverse = self.propagator.set(grid_out=pupil.grid, inverse=True)
        pupil_difference = difference.propagate(inverse, **context)
        return pupil - pupil_difference
