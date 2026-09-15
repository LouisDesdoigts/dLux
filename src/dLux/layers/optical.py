"""User-facing optical elements and direct wavefront operations."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import jax.numpy as np
import equinox as eqx
import zodiax as zdx
from jax import Array

import dLux.utils as dlu

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


class BaseLayer(zdx.Module):
    """Base contract for callable immutable transformations of dLux objects.

    Concrete layer families define the target accepted by `apply`. Layers contain
    their differentiable parameters and return transformed objects without mutating
    their inputs. ``**context`` is forwarded unchanged through application; pass
    ``state=zdx.State(...)`` to resolve external ``zdx.StateRef`` values. Layers
    consume state without advancing it or storing it on the returned field.
    """

    @abstractmethod
    def apply(self, target: Any, **context) -> Any:
        """Transform a compatible target and return a new object.

        Subclasses define the accepted target type and must preserve immutable dLux
        semantics: applying a layer does not mutate the input object. Accept
        ``**context`` to participate in contextual system evaluation.
        """

    def __call__(self, target: Any, **context) -> Any:
        """Call :meth:`apply` using concise layer syntax."""
        return self.apply(target, **context)


class BaseOpticalLayer(BaseLayer):
    """Base contract for layers that transform scalar or polarised wavefronts.

    Subclasses implement the monochromatic `apply_mono` operation. `apply` maps it
    over wavelength and compatible leading axes, while `__call__` provides concise
    layer syntax with the same vectorisation contract.
    """

    @abstractmethod
    def apply_mono(self, wavefront: Wavefront, **context) -> Wavefront:
        """Transform one monochromatic wavefront.

        Implement this method when defining a standard optical layer. It receives a
        wavefront with no mapped leading batch axis and must return a wavefront;
        `apply` supplies recursive leading-axis vectorisation.
        """

    def apply(self, wavefront: Wavefront, **context) -> Wavefront:
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
            return self.apply_mono(wavefront, **context)
        axes = wavefront._mapped_axis
        if axes is None:
            return self.apply_mono(wavefront, **context)

        # Define application to one monochromatic field
        def apply_one(phasor, wavelength, weight, d, c):
            wavefront_i = wavefront.set(
                phasor=phasor,
                wavelength=wavelength,
                weight=weight,
                d=d,
                c=c,
            )
            return self.apply(wavefront_i, **context)

        # Vectorise the layer over leading wavefront dimensions
        apply_one = eqx.filter_vmap(apply_one, in_axes=axes)
        output = apply_one(
            wavefront.phasor,
            wavefront.wavelength,
            wavefront.weight,
            wavefront.grid.d,
            wavefront.grid.c,
        )

        # Remove axes introduced for metadata values that were not vectorised
        wavelength = output.wavelength
        if axes[1] is None:
            wavelength = wavelength[0]
        weights = output.weight
        if weights is not None and axes[2] is None:
            weights = weights[0]
        c = output.grid.c
        c = c[0] if c is not None and axes[4] is None else c
        d = output.grid.d[0] if axes[3] is None else output.grid.d

        # Restore the realised wavefront grid
        return output.set(
            wavelength=wavelength,
            weight=weights,
            d=d,
            c=c,
        )


class OpticalLayer(BaseOpticalLayer):
    """Public extension point for user-defined monochromatic optical layers.

    Implement `apply_mono(wavefront, **context)` for one monochromatic `Wavefront`.
    Inherited application handles chromatic and otherwise vectorised wavefronts.
    The result must retain physically consistent wavelength and grid metadata.
    """

    @staticmethod
    def context(wavefront: Wavefront, **context) -> dict[str, Any]:
        """Return the standard parametric context for an optical layer.

        Spatial definitions obtain coordinates and pixel sampling from the incident
        grid. Chromatic definitions receive wavelengths separately. Explicit caller
        context, such as external state, is forwarded; an additional coordinate
        input is rejected by spatial definitions rather than silently overwritten.
        """
        return {
            **context,
            "grid": wavefront.grid,
            "wavelengths": wavefront.wavelength,
        }


class TransmissiveLayer(OpticalLayer):
    """Apply a transmission with optional output normalisation.

    Transmission is real-valued amplitude scaling. Use OPD, phase, or Jones terms
    for phase and complex field effects. Values broadcast to the incoming phasor
    while preserving its wavelength, polarisation, and other leading axes.

    Parameters
    ----------
    transmission : Array, Parametric, or None
        Real scalar or sampled amplitude transmission.
    normalise : bool or None
        Normalise the resulting wavefront to unit power when true. Omit to leave
        normalisation unconfigured.
    """

    transmission: Array | zdx.Expression | None = None
    normalise: bool | None = None

    def __init__(self, transmission=None, normalise=None, *, alias=None):
        """Initialise a transmissive optical layer.

        Parameters
        ----------
        transmission : Array, Parametric, or None
            Scalar or sampled amplitude transmission, or a parametric resolving one.
        normalise : bool or None
            Renormalise wavefront power after applying the transmission when true.
            Omit to leave normalisation unconfigured.
        """
        transmission = zdx.as_array(transmission)
        if not isinstance(transmission, zdx.Expression) and np.iscomplexobj(
            transmission
        ):
            raise TypeError("transmission must be real-valued; use phase or opd.")
        self.transmission = zdx.as_array(transmission, dtype=float)
        self.normalise = None if normalise is None else bool(normalise)

        self.alias = alias

    def apply_mono(self, wavefront: Wavefront, **context) -> Wavefront:
        """Apply the resolved amplitude transmission to one wavefront.

        Parametric transmission is resolved from the optical context, reshaped for
        phasor broadcasting, and multiplied into a new wavefront. Unit-power
        normalisation is applied afterward when configured.
        """
        self = self.resolve(**self.context(wavefront, **context))
        if self.transmission is not None:
            if np.iscomplexobj(self.transmission):
                raise TypeError("transmission must be real-valued; use phase or opd.")
            transmission = wavefront._to_phasor_shape(self.transmission)
            wavefront = wavefront.set(phasor=wavefront.phasor * transmission)
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront


class AberratedLayer(OpticalLayer):
    """Apply optical-path and phase aberrations to a wavefront.

    OPD is converted to phase using the wavefront wavelength, while phase is applied
    directly. Values broadcast to the incoming phasor and may be arrays or contextual
    `Parametric` objects in concrete layers.

    Parameters
    ----------
    opd : Array, Parametric, or None
        Optical path difference in meters.
    phase : Array, Parametric, or None
        Additional phase in radians.
    """

    opd: Array | zdx.Expression | None = None
    phase: Array | zdx.Expression | None = None

    def __init__(self, opd=None, phase=None, *, alias=None):
        """Initialise optical-path and phase aberrations.

        Parameters
        ----------
        opd : Array, Parametric, or None
            Optical path difference in metres.
        phase : Array, Parametric, or None
            Wavelength-independent phase in radians.
        """
        self.opd = zdx.as_array(opd, dtype=float)
        self.phase = zdx.as_array(phase, dtype=float)

        self.alias = alias

    def apply_mono(self, wavefront: Wavefront, **context) -> Wavefront:
        """Apply resolved OPD and phase aberrations to one wavefront.

        OPD is interpreted in metres and converted using the wavefront wavelength;
        phase is interpreted directly in radians. The input is not mutated.
        """
        self = self.resolve(**self.context(wavefront, **context))
        wavefront = wavefront.add_opd(self.opd)
        return wavefront.add_phase(self.phase)


class Optic(TransmissiveLayer, AberratedLayer):
    """Combine optional scalar, refractive, and Jones effects at one plane.

    Local terms resolve on the incident grid. Amplitude and transmission multiply
    the field, while OPD, phase, and refractive thickness contribute phase. A Jones
    matrix promotes scalar inputs to polarised wavefronts. Optional unit-power
    normalisation follows these local operations, then the propagator transports
    the complete field to its output grid. Unconfigured terms are skipped.

    Parameters
    ----------
    transmission : Array, Parametric, or None
        Real scalar or sampled amplitude transmission.
    opd : Array, Parametric, or None
        Optical path difference in meters.
    phase : Array, Parametric, or None
        Additional phase in radians.
    normalise : bool or None
        Normalise the resulting wavefront to unit power when true. Omit to leave
        normalisation unconfigured.
    amplitude : Array, Parametric, or None
        Additional dimensionless amplitude multiplier, broadcast like transmission.
    jones : Array, Parametric, or None
        Complex Jones matrix with shape ``(2, 2, ...)``; trailing axes are spatial.
    thickness : Array, Parametric, or None
        Scalar or sampled material thickness in metres.
    n : Array, Parametric, or None
        Dimensionless refractive index. When both ``n`` and ``thickness`` are set,
        add ``(n - 1) * thickness`` to OPD, relative to propagation in vacuum.
    propagator : BaseOpticalLayer or None
        Propagation layer called after local operations and normalisation. It owns
        output sampling and handles the complete wavelength and batch population.

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

    transmission: Array | zdx.Expression | None = None
    opd: Array | zdx.Expression | None = None
    phase: Array | zdx.Expression | None = None
    amplitude: Array | zdx.Expression | None = None
    jones: Array | zdx.Expression | None = None
    thickness: Array | zdx.Expression | None = None
    n: Array | zdx.Expression | None = None
    propagator: BaseOpticalLayer | None = None
    normalise: bool | None = None

    def __init__(
        self,
        transmission=None,
        opd=None,
        phase=None,
        normalise=None,
        *,
        amplitude=None,
        jones=None,
        thickness=None,
        n=None,
        propagator=None,
        alias=None,
    ):
        """Initialise optional local optical terms and outgoing propagation.

        Parameters
        ----------
        transmission : Array, Parametric, or None
            Real scalar or sampled amplitude transmission.
        opd : Array, Parametric, or None
            Optical path difference in metres.
        phase : Array, Parametric, or None
            Wavelength-independent phase in radians.
        normalise : bool or None
            Renormalise wavefront power after applying the optic when true. Omit to
            leave normalisation unconfigured.
        amplitude : Array, Parametric, or None
            Scalar or sampled amplitude multiplier. Defaults to no multiplication.
        jones : Array, Parametric, or None
            Jones matrix of shape ``(2, 2, ...)``. Defaults to no Jones operation.
        thickness : Array, Parametric, or None
            Thickness in metres, broadcast to the field. Defaults to no thickness.
        n : Array, Parametric, or None
            Refractive index, broadcast to the field. Refraction is skipped unless
            both this and ``thickness`` are supplied; both default to ``None``.
        propagator : BaseOpticalLayer or None
            Outgoing propagation layer. Defaults to remaining at the incident grid.
        alias : mapping or sequence of pairs, optional
            Additional Zodiax parameter aliases.
        """
        TransmissiveLayer.__init__(self, transmission, normalise)
        AberratedLayer.__init__(self, opd, phase)

        self.amplitude = zdx.as_array(amplitude, dtype=float)
        self.jones = zdx.as_array(jones, dtype=None)
        self.thickness = zdx.as_array(thickness, dtype=float)
        self.n = zdx.as_array(n, dtype=None)
        if propagator is not None and not isinstance(propagator, BaseOpticalLayer):
            raise TypeError("propagator must be a BaseOpticalLayer or None.")
        self.propagator = propagator

        self.alias = alias

    def phasor(self, wavefront: Wavefront, **context) -> Array:
        """Resolve and return the optic's cumulative complex field multiplier.

        Amplitude, transmission, OPD in metres, phase in radians, and refractive
        thickness are resolved together with Jones terms from the wavefront
        context, so expressions can share linked values. The returned array
        broadcasts against the wavefront phasor and includes only scalar
        modulation. Jones effects, normalisation, and propagation are excluded.
        """
        local = self._resolve_local(wavefront, **context)
        return local._phasor(wavefront)

    def apply(self, wavefront: Wavefront, **context) -> Wavefront:
        """Apply local effects over leading axes, then propagate the full field."""
        if self.propagator is None:
            return super().apply(wavefront, **context)
        local = self.set(propagator=None)
        wavefront = local(wavefront, **context)
        return self.propagator(wavefront, **context)

    def apply_mono(self, wavefront: Wavefront, **context) -> Wavefront:
        """Apply local scalar and Jones effects to one monochromatic wavefront.

        Normalisation follows local modulation. :meth:`apply` owns outgoing
        propagation after restoring all wavelength and batch axes.
        """
        local = self._resolve_local(wavefront, **context)
        jones = local.jones
        phasor = wavefront.phasor * local._phasor(wavefront)
        wavefront = wavefront.set(phasor=phasor)
        if jones is not None:
            wavefront = wavefront.apply_jones(jones)
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront

    def _resolve_local(self, wavefront, **context):
        """Resolve local terms together so shared links retain their owners."""
        names = ("transmission", "amplitude", "opd", "phase", "thickness", "n", "jones")
        values = {name: getattr(self, name) for name in names}
        values = zdx.resolve(values, **self.context(wavefront, **context))
        return self.set(**values)

    def _phasor(self, wavefront: Wavefront) -> Array:
        """Combine resolved optical terms into one complex field."""
        if np.iscomplexobj(self.transmission):
            raise TypeError("transmission must be real-valued; use phase or opd.")

        # Resolve absent optical terms to their identities
        transmission = 1.0 if self.transmission is None else self.transmission
        opd = 0.0 if self.opd is None else self.opd
        phase = 0.0 if self.phase is None else self.phase

        # Promote every term to the wavefront field shape
        wavenumber = wavefront._to_phasor_shape(wavefront.wavenumber)
        transmission = wavefront._to_phasor_shape(transmission)
        opd = wavefront._to_phasor_shape(opd)
        phase = wavefront._to_phasor_shape(phase)

        # Apply optional amplitude and excess material optical path
        if self.amplitude is not None:
            transmission = transmission * wavefront._to_phasor_shape(self.amplitude)
        if self.thickness is not None and self.n is not None:
            thickness = wavefront._to_phasor_shape(self.thickness)
            index = wavefront._to_phasor_shape(self.n)
            opd = opd + (index - 1) * thickness

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

    def __init__(self, angles, unit="rad", *, alias=None):
        """Initialise an angular wavefront tilt.

        Parameters
        ----------
        angles : ArrayLike
            Two-component ``(x, y)`` angular offset.
        unit : str
            Supported angular unit for ``angles``.
        """
        self.angles = zdx.as_array(angles, dtype=float)
        if self.angles.shape != (2,):
            raise ValueError("angles must have shape (2,).")
        self.unit = dlu.canonical_unit(unit, dimension="angle", name="tilt unit")

        self.alias = alias

    def apply_mono(self, wavefront: Wavefront, **context) -> Wavefront:
        """Apply the configured angular tilt to one wavefront.

        Angles use the layer's declared unit and physical ``(x, y)`` order. The
        returned wavefront retains its sampling grid.
        """
        angles = zdx.resolve(self.angles, **self.context(wavefront, **context))
        return wavefront.tilt(angles, self.unit)
