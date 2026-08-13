"""User-facing optical elements and direct wavefront operations."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import jax.numpy as np
import equinox as eqx
from jax import Array

import dLux.utils as dlu

from ..parametric import Parametric, ParametricHolder
from ..fields import Wavefront

__all__ = [
    "BaseLayer",
    "BaseOpticalLayer",
    "OpticalLayer",
    "TransmissiveLayer",
    "AberratedLayer",
    "Optic",
    "Tilt",
]


class BaseLayer(ParametricHolder):
    """Base class for callable transformations of dLux objects."""

    @abstractmethod
    def apply(self, target: Any) -> Any:
        """Transform a compatible target and return a new object.

        Subclasses define the accepted target type and must preserve immutable dLux
        semantics: applying a layer does not mutate the input object.
        """

    def __call__(self, target: Any) -> Any:
        """Call :meth:`apply` using concise layer syntax."""
        return self.apply(target)


class BaseOpticalLayer(BaseLayer):
    """Base class for layers that transform wavefronts."""

    @abstractmethod
    def apply_mono(self, wavefront: Wavefront) -> Wavefront:
        """Transform one monochromatic wavefront.

        Implement this method when defining a standard optical layer. It receives a
        wavefront with no mapped leading batch axis and must return a wavefront;
        `apply` supplies recursive leading-axis vectorisation.
        """

    def apply(self, wavefront: Wavefront) -> Wavefront:
        """Apply the monochromatic operation over all leading wavefront axes.

        Parameters
        ----------
        wavefront : Wavefront
            Scalar or polarised wavefront with optional leading wavelength and batch
            axes. Sampling metadata are vectorised only when they share the mapped
            leading axis.

        Returns
        -------
        wavefront : Wavefront
            Transformed wavefront with leading axes and realised grid metadata
            restored. The input wavefront is not mutated.
        """
        # Apply directly to non-wavefront targets and scalar wavefronts
        if not isinstance(wavefront, Wavefront):
            return self.apply_mono(wavefront)
        axes = wavefront._mapped_axis
        if axes is None:
            return self.apply_mono(wavefront)

        # Define application to one monochromatic field
        def apply_one(phasor, wavelength, d, c):
            wavefront_i = wavefront.set(
                phasor=phasor,
                wavelength=wavelength,
                d=d,
                c=c,
            )
            return self.apply(wavefront_i)

        # Vectorise the layer over leading wavefront dimensions
        apply_one = eqx.filter_vmap(apply_one, in_axes=axes)
        output = apply_one(
            wavefront.phasor, wavefront.wavelength, wavefront.grid.d, wavefront.grid.c
        )

        # Remove axes introduced for grid values that were not vectorised
        c = output.grid.c
        c = c[0] if c is not None and axes[3] is None else c
        d = output.grid.d[0] if axes[2] is None else output.grid.d

        # Restore the realised wavefront grid
        return output.set(d=d, c=c)


class OpticalLayer(BaseOpticalLayer):
    """Public contract for layers that transform wavefronts."""

    @staticmethod
    def context(wavefront: Wavefront) -> dict[str, Any]:
        """Return the standard parametric context for an optical layer.

        The mapping contains the input ``wavefront`` and is passed to every
        `Parametric` leaf during `resolve`.
        """
        return {"wavefront": wavefront}


class TransmissiveLayer(OpticalLayer):
    """Apply a transmission with optional output normalisation.

    Parameters
    ----------
    transmission : Array, Parametric, or None
        Scalar or sampled amplitude transmission.
    normalise : bool
        Normalise the resulting wavefront to unit power.
    """

    transmission: Array | Parametric | None
    normalise: bool

    def __init__(self, transmission=None, normalise=False):
        """Initialise a transmissive optical layer.

        Parameters
        ----------
        transmission : Array, Parametric, or None
            Scalar or sampled amplitude transmission, or a parametric resolving one.
        normalise : bool
            Renormalise wavefront power after applying the transmission.
        """
        self.transmission = dlu.to_value(transmission, optional=True, types=Parametric)
        self.normalise = bool(normalise)

    def apply_mono(self, wavefront: Wavefront) -> Wavefront:
        """Apply the resolved amplitude transmission to one wavefront.

        Parametric transmission is resolved from the optical context, reshaped for
        phasor broadcasting, and multiplied into a new wavefront. Unit-power
        normalisation is applied afterward when configured.
        """
        self = self.resolve(**self.context(wavefront))
        if self.transmission is not None:
            transmission = wavefront._to_phasor_shape(self.transmission)
            wavefront = wavefront.set(phasor=wavefront.phasor * transmission)
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront


class AberratedLayer(OpticalLayer):
    """Apply optical-path and phase aberrations to a wavefront.

    Parameters
    ----------
    opd : Array, Parametric, or None
        Optical path difference in meters.
    phase : Array, Parametric, or None
        Additional phase in radians.
    """

    opd: Array | Parametric | None
    phase: Array | Parametric | None

    def __init__(self, opd=None, phase=None):
        """Initialise optical-path and phase aberrations.

        Parameters
        ----------
        opd : Array, Parametric, or None
            Optical path difference in metres.
        phase : Array, Parametric, or None
            Wavelength-independent phase in radians.
        """
        self.opd = dlu.to_value(opd, optional=True, types=Parametric)
        self.phase = dlu.to_value(phase, optional=True, types=Parametric)

    def apply_mono(self, wavefront: Wavefront) -> Wavefront:
        """Apply resolved OPD and phase aberrations to one wavefront.

        OPD is interpreted in metres and converted using the wavefront wavelength;
        phase is interpreted directly in radians. The input is not mutated.
        """
        self = self.resolve(**self.context(wavefront))
        wavefront = wavefront.add_opd(self.opd)
        return wavefront.add_phase(self.phase)


class Optic(TransmissiveLayer, AberratedLayer):
    """Represent a scalar physical optic evaluated at one plane.

    Parameters
    ----------
    transmission : Array, Parametric, or None
        Scalar or sampled amplitude transmission.
    opd : Array, Parametric, or None
        Optical path difference in meters.
    phase : Array, Parametric, or None
        Additional phase in radians.
    normalise : bool
        Normalise the resulting wavefront to unit power.

    Examples
    --------
    Combine amplitude transmission, OPD, and phase in one optical layer:

    ```python
    import jax.random as jr

    import dLux as dl

    # Construct the sampled optical components
    grid = dl.GridSpec(n=128, diam=1.2, unit="m")
    transmission = dl.Circle(diameter=1.0)(grid)
    opd = 10e-9 * jr.normal(jr.key(0), transmission.shape)
    phase = 0.1 * jr.normal(jr.key(1), transmission.shape)

    # Combine the transmission, OPD, and phase into one optic
    optic = dl.Optic(
        transmission=transmission,
        opd=opd,
        phase=phase,
        normalise=True,
    )

    # Apply the optic to a wavefront
    wavefront = dl.Wavefront(wavelength=650e-9, grid=grid)
    wavefront = optic(wavefront)
    ```
    """

    transmission: Array | Parametric | None
    opd: Array | Parametric | None
    phase: Array | Parametric | None
    normalise: bool

    def __init__(self, transmission=None, opd=None, phase=None, normalise=False):
        """Initialise a combined transmissive and aberrated optic.

        Parameters
        ----------
        transmission : Array, Parametric, or None
            Scalar or sampled amplitude transmission.
        opd : Array, Parametric, or None
            Optical path difference in metres.
        phase : Array, Parametric, or None
            Wavelength-independent phase in radians.
        normalise : bool
            Renormalise wavefront power after applying the optic.
        """
        TransmissiveLayer.__init__(self, transmission, normalise)
        AberratedLayer.__init__(self, opd, phase)

    def phasor(self, wavefront: Wavefront) -> Array:
        """Resolve and return the optic's cumulative complex field multiplier.

        Transmission, OPD in metres, and phase in radians are resolved from the
        wavefront context. The returned array is reshaped to broadcast against the
        wavefront phasor but is not applied or normalised.
        """
        self = self.resolve(**self.context(wavefront))
        return self._phasor(wavefront)

    def apply_mono(self, wavefront: Wavefront) -> Wavefront:
        """Apply the resolved complex optic multiplier to one wavefront.

        Returns a new wavefront and optionally normalises it to unit power according
        to the optic's ``normalise`` setting.
        """
        phasor = wavefront.phasor * self.phasor(wavefront)
        wavefront = wavefront.set(phasor=phasor)
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront

    def _phasor(self, wavefront: Wavefront) -> Array:
        """Combine resolved optical terms into one complex field."""
        # Resolve absent optical terms to their identities
        transmission = 1.0 if self.transmission is None else self.transmission
        opd = 0.0 if self.opd is None else self.opd
        phase = 0.0 if self.phase is None else self.phase

        # Promote every term to the wavefront field shape
        wavenumber = wavefront._to_phasor_shape(wavefront.wavenumber)
        transmission = wavefront._to_phasor_shape(transmission)
        opd = wavefront._to_phasor_shape(opd)
        phase = wavefront._to_phasor_shape(phase)

        # Construct the combined scalar phasor
        return transmission * np.exp(1j * (wavenumber * opd + phase))


class Tilt(OpticalLayer):
    """Tilt a wavefront by two angular coordinates.

    Parameters
    ----------
    angles : ArrayLike
        Two angular offsets in ``(x, y)`` order.
    unit : str
        Angular unit associated with ``angles``.

    Examples
    --------
    Apply one physical angular offset across a chromatic wavefront:

    ```python
    import jax.numpy as np

    import dLux as dl

    # Construct a chromatic wavefront
    grid = dl.GridSpec(n=128, diam=1.0, unit="m")
    wavelengths = np.linspace(600e-9, 700e-9, 5)
    wavefront = dl.Wavefront(wavelength=wavelengths, grid=grid)

    # Apply an angular offset in convenient physical units
    tilt = dl.Tilt(angles=[20.0, -10.0], unit="mas")
    wavefront = tilt(wavefront)
    ```
    """

    angles: Array
    unit: str

    def __init__(self, angles, unit="rad"):
        """Initialise an angular wavefront tilt.

        Parameters
        ----------
        angles : ArrayLike
            Two-component ``(x, y)`` angular offset.
        unit : str
            Supported angular unit for ``angles``.
        """
        self.angles = dlu.to_value(angles, name="angles")
        if self.angles.shape != (2,):
            raise ValueError("angles must have shape (2,).")
        self.unit = dlu.canonical_unit(unit, dimension="angle", name="tilt unit")

    def apply_mono(self, wavefront: Wavefront) -> Wavefront:
        """Apply the configured angular tilt to one wavefront.

        Angles use the layer's declared unit and physical ``(x, y)`` order. The
        returned wavefront retains its sampling grid.
        """
        return wavefront.tilt(self.angles, self.unit)
