"""Released basis-optic convenience constructor."""

import jax.numpy as np
import zodiax as zdx
import dLux.utils as dlu
from ...layers.optical import Optic as CurrentOptic
from ..warnings import warn_deprecated

__all__ = ["Optic", "BasisOptic"]


class Optic(CurrentOptic):
    """Translate released array broadcasting before applying the current optic.

    The released API allowed modulation arrays to expand singleton spatial axes.
    Keep that behaviour at this boundary only, retaining the input pixel spacing.
    New Optic objects continue to require modulation to fit their field sampling.
    """

    def __init__(self, transmission=None, opd=None, phase=None, normalise=False):
        """Retain released modulation broadcasting and warn about migration.

        Parameters
        ----------
        transmission : ArrayLike or Expression or None
            Real amplitude transmission; None is unity. Released singleton spatial
            axes may expand to the broadcast modulation shape.
        opd : ArrayLike or Expression or None
            Optical path difference in metres; None is zero.
        phase : ArrayLike or Expression or None
            Additional phase in radians; None is zero.
        normalise : bool
            Set transmitted power to one when true; false by default.
        """
        warn_deprecated("compat.Optic", "Optic", "Optic(transmission=..., opd=...)")
        super().__init__(transmission, opd, phase, normalise)
        shapes = [
            np.shape(value)
            for value in (self.transmission, self.opd, self.phase)
            if value is not None and not isinstance(value, zdx.Expression)
        ]
        np.broadcast_shapes(*shapes)

    def apply_mono(self, wavefront, **context):
        local = self.resolve_local(**self.context(wavefront, **context))
        values = (local.transmission, local.opd, local.phase)
        shape = np.broadcast_shapes(
            wavefront.phasor.shape,
            *(np.shape(value) for value in values if value is not None),
        )
        if shape != wavefront.phasor.shape:
            wavefront = wavefront.set(
                phasor=np.broadcast_to(wavefront.phasor, shape),
                grid=wavefront.grid.set(n=shape[-2:][::-1]),
            )
        return CurrentOptic.apply_mono(local, wavefront, **context)


def BasisOptic(
    basis, transmission=None, coefficients=None, as_phase=False, normalise=False
):
    warn_deprecated(
        "BasisOptic",
        "Optic",
        "Optic(opd=Basis(...)); coefficients now live at opd.coeffs",
    )
    basis = dlu.Basis(basis, coeffs=coefficients)
    return Optic(
        transmission=transmission,
        normalise=normalise,
        phase=basis if as_phase else None,
        opd=None if as_phase else basis,
    )
