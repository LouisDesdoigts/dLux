"""Optical effects, their combined Optic, and Wavefront propagation adapters."""

from __future__ import annotations

from typing import Any

import jax.numpy as np
import zodiax as zdx
from jax import Array

import dLux.utils as dlu

from ..fields import Wavefront
from ..prebuilt.definitions import Def

__all__ = [
    "OpticalLayer",
    "ComplexLayer",
    "BaseOptic",
    "RefractiveLayer",
    "JonesLayer",
    "PropagatorLayer",
    "Optic",
    "TiltLayer",
    "WedgeLayer",
]


class OpticalLayer(dlu.BaseLayer):
    """Transform a Wavefront, mapping apply_mono over independent samples.

    Subclasses implement apply_mono for one monochromatic field. Layers supporting
    complete chromatic fields override apply; Wavefront owns their axis alignment.
    """

    def apply_mono(self, wavefront: Wavefront, **context) -> Wavefront:
        """Transform one scalar-wavelength field on one grid.

        The phasor is (ny, nx), or (2, 2, ny, nx) for explicit Jones samples.
        Layer parameters are shared across mapped samples; resolve expressions
        using self.context(wavefront, **context). Ordinary layers retain the batch
        layout; a specialised layer such as SparseOptic may introduce apertures.
        """
        raise NotImplementedError(
            "Implement apply_mono(wavefront, **context) for automatic batching. "
            "Released layers may still implement __call__ directly."
        )

    def apply(self, wavefront: Wavefront, **context) -> Wavefront:
        """Apply a monochromatic operation across independent field samples."""
        if isinstance(wavefront, Wavefront):
            return wavefront._map_wavefronts(self.apply_mono, **context)
        return self.apply_mono(wavefront, **context)

    @staticmethod
    def context(wavefront: Wavefront, **context) -> dict[str, Any]:
        """Supply incident SI coordinates, pixel scale and wavelengths.

        Plain Expressions and Sampled definitions receive the same coordinate
        context. The grid is used here only, avoiding two sampling inputs downstream.
        """
        context = dict(
            context, wavelength=wavefront.wavelength, wavelengths=wavefront.wavelength
        )
        context.pop("grid", None)
        context.pop("coordinates", None)
        grid = wavefront.grid.resolve(**context)
        return {
            **context,
            "coordinates": grid.coordinates,
            "pixel_scale": grid.d_si,
        }


class ComplexLayer(OpticalLayer):
    """Multiply a field by amplitude and a wavelength-independent phase.

    Amplitude is dimensionless and phase is in radians. Inputs may be arrays or
    Expressions sampled on the incident grid. The layer preserves its sampling.
    """

    amplitude: Array | zdx.Expression | None = None
    phase: Array | zdx.Expression | None = None

    def __init__(self, amplitude=None, phase=None, *, alias=None):
        self.amplitude = zdx.as_array(amplitude, dtype=float)
        self.phase = zdx.as_array(phase, dtype=float)
        self.alias = alias

    def modulation(self, wavefront: Wavefront) -> tuple[Array, Array]:
        """Return resolved amplitude and phase aligned to the field axes."""
        amplitude = 1.0 if self.amplitude is None else self.amplitude
        phase = 0.0 if self.phase is None else self.phase
        return (
            wavefront._to_phasor_shape(amplitude),
            wavefront._to_phasor_shape(phase),
        )

    def apply_mono(self, wavefront: Wavefront, **context) -> Wavefront:
        self = self.resolve(**self.context(wavefront, **context))
        amplitude, phase = self.modulation(wavefront)
        return wavefront.set(phasor=wavefront.phasor * amplitude * np.exp(1j * phase))


class BaseOptic(OpticalLayer):
    """Apply real amplitude transmission and optical path difference in metres.

    Optional normalisation sets the resulting field power to one. In a combined
    Optic this happens after all scalar effects and before Jones polarisation.
    """

    transmission: Array | zdx.Expression | None = None
    opd: Array | zdx.Expression | None = None
    normalise: bool | None = None

    def __init__(self, transmission=None, opd=None, normalise=None, *, alias=None):
        transmission = zdx.as_array(transmission)
        if not isinstance(transmission, zdx.Expression) and np.iscomplexobj(
            transmission
        ):
            raise TypeError("transmission must be real-valued; use phase or opd.")
        self.transmission = zdx.as_array(transmission, dtype=float)
        self.opd = zdx.as_array(opd, dtype=float)
        self.normalise = None if normalise is None else bool(normalise)
        self.alias = alias

    def resolve_local(self, **context) -> BaseOptic:
        """Resolve transmission first so sampled OPD can use its footprint."""
        self = self.populate()
        transmission = zdx.resolve(self.transmission, **context)
        if transmission is not None:
            context = {**context, "support": transmission != 0}
        return self.set(transmission=transmission).resolve(**context)

    def modulation(self, wavefront: Wavefront) -> tuple[Array, Array]:
        """Return resolved transmission and OPD phase aligned to field axes."""
        if np.iscomplexobj(self.transmission):
            raise TypeError("transmission must be real-valued; use phase or opd.")
        transmission = 1.0 if self.transmission is None else self.transmission
        opd = 0.0 if self.opd is None else self.opd
        return wavefront._to_phasor_shape(transmission), wavefront.phase_from_opd(opd)

    def apply_mono(self, wavefront: Wavefront, **context) -> Wavefront:
        self = self.resolve_local(**self.context(wavefront, **context))
        amplitude, phase = self.modulation(wavefront)
        wavefront = wavefront.set(
            phasor=wavefront.phasor * amplitude * np.exp(1j * phase)
        )
        return wavefront.normalise() if self.normalise else wavefront


class RefractiveLayer(OpticalLayer):
    """Apply excess material optical path ``(n - 1) * thickness``.

    Thickness is in metres. The refractive index may be complex to represent
    absorption, or an Expression evaluated at the incident wavelengths.
    """

    thickness: Array | zdx.Expression | None = None
    n: Array | zdx.Expression | None = None

    def __init__(self, thickness=None, n=None, *, alias=None):
        self.thickness = zdx.as_array(thickness, dtype=float)
        self.n = zdx.as_array(n)
        self.alias = alias

    def modulation(self, wavefront: Wavefront) -> tuple[Array, Array]:
        """Return unit amplitude and resolved material phase on the field axes."""
        phase = wavefront._to_phasor_shape(0.0)
        if self.thickness is not None and self.n is not None:
            opd = (self.n - 1) * self.thickness
            phase = wavefront.phase_from_opd(opd)
        return wavefront._to_phasor_shape(1.0), phase

    def apply_mono(self, wavefront: Wavefront, **context) -> Wavefront:
        self = self.resolve(**self.context(wavefront, **context))
        _, phase = self.modulation(wavefront)
        return wavefront.set(phasor=wavefront.phasor * np.exp(1j * phase))


class JonesLayer(OpticalLayer):
    """Apply a numerical Jones definition through Wavefront's polarisation API."""

    jones: Array | zdx.Expression | None = None

    def __init__(self, jones=None, *, alias=None):
        self.jones = zdx.as_array(jones)
        self.alias = alias

    def apply_mono(self, wavefront: Wavefront, **context) -> Wavefront:
        self = self.resolve(**self.context(wavefront, **context))
        return self.apply_jones(wavefront)

    def apply_jones(self, wavefront: Wavefront) -> Wavefront:
        """Apply an already-resolved Jones matrix, preserving field sampling."""
        if self.jones is None:
            return wavefront
        # One monochromatic layer has a matrix or a sampled matrix field.
        if self.jones.ndim not in (2, 4):
            raise ValueError("A local Jones layer needs (2, 2) or (ny, nx, 2, 2).")
        return wavefront.apply_jones(self.jones, spatial=self.jones.ndim == 4)


class PropagatorLayer(OpticalLayer, dlu.Propagator):
    """A numerical Propagator with Wavefront input and output.

    Grid, focal length, defocus, ABCD and inverse parameters are inherited directly.
    Wavefront owns the wavelength, population and Jones-axis adaptation. Alternative
    numerical algorithms can be applied with ``wavefront.propagate(definition)``.
    """

    # Inherited fields
    abcd: Array | zdx.Expression | None = None
    grid_out: dlu.Grid | None = None
    focal_length: Any = None
    defocus: Any = None
    inverse: bool | None = False

    def __init__(
        self,
        grid_out=None,
        focal_length=None,
        defocus=None,
        *,
        abcd=None,
        inverse=False,
        alias=None,
    ):
        dlu.Propagator.__init__(
            self,
            grid_out,
            focal_length,
            defocus,
            abcd=abcd,
            inverse=inverse,
            alias=alias,
        )

    def propagate(self, wavefront: Wavefront, **context) -> Wavefront:
        """Apply this object's propagation parameters to an incident field."""
        return wavefront.propagate(self, **context)

    def apply(self, wavefront: Wavefront, **context) -> Wavefront:
        return self.propagate(wavefront, **context)

    def apply_mono(self, wavefront: Wavefront, **context) -> Wavefront:
        return self.apply(wavefront, **context)


class Optic(BaseOptic, ComplexLayer, RefractiveLayer, JonesLayer, PropagatorLayer):
    """Combine inherited scalar, complex, refractive and Jones effects at one plane.

    Each parent supplies its own fields and numerical operation. Transmission,
    OPD, amplitude, phase, thickness, index and Jones terms are stored directly on
    the optic, with ordinary update paths such as ``opd.coeffs``.

    Evaluation uses one local grid and coordinate transform. Scalar phases combine
    before exponentiation, optional normalisation follows scalar modulation, and
    Jones effects follow normalisation so polarisation losses are retained.
    A supplied ``grid_out`` propagates the resulting field after these effects.
    None retains the incident sampling. Focal length defines the relay, not a
    second local lens phase.
    """

    # Inherited fields
    abcd: Array | zdx.Expression | None = None
    grid_out: dlu.Grid | None = None
    focal_length: Any = None
    defocus: Any = None
    inverse: bool | None = False
    jones: Array | zdx.Expression | None = None
    thickness: Array | zdx.Expression | None = None
    n: Array | zdx.Expression | None = None
    amplitude: Array | zdx.Expression | None = None
    phase: Array | zdx.Expression | None = None
    transmission: Array | zdx.Expression | None = None
    opd: Array | zdx.Expression | None = None
    normalise: bool | None = None

    grid: dlu.Grid | None = None
    transform: dlu.BaseCoordTransform | None = None

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
        grid=None,
        transform=None,
        dynamic=False,
        grid_out=None,
        focal_length=None,
        defocus=None,
        abcd=None,
        inverse=False,
        alias=None,
    ):
        """Initialise inherited fields, consuming Def inputs on local sampling.

        ``dynamic=True`` asks definitions for runtime expressions. Otherwise a
        supplied grid permits sampling at construction. Arrays and Expressions
        are retained; factories are not stored in the optic.
        """
        # Choose sampling before constructing deferred optical definitions.
        if grid is not None and not isinstance(grid, dlu.Grid):
            raise TypeError("grid must be a Grid or None.")
        if transform is not None and not isinstance(transform, dlu.BaseCoordTransform):
            raise TypeError("transform must be a BaseCoordTransform or None.")

        def construct(value, support=None):
            if not isinstance(value, Def):
                return value
            context = {}
            sampling = grid if grid is not None else getattr(value, "grid", None)
            if sampling is not None and not dynamic:
                sampling = sampling.resolve()
                context = dict(
                    coordinates=sampling.transformed(transform),
                    pixel_scale=sampling.d_si,
                )
                if support is not None:
                    context["support"] = zdx.resolve(support, **context) != 0
            return value(dynamic=dynamic, **context)

        # Each parent initialises its own fields; shared Module state is set last.
        transmission = construct(transmission)
        BaseOptic.__init__(self, transmission, construct(opd, transmission), normalise)
        ComplexLayer.__init__(
            self, construct(amplitude), construct(phase, transmission)
        )
        RefractiveLayer.__init__(self, construct(thickness, transmission), construct(n))
        JonesLayer.__init__(self, construct(jones))
        self.grid = grid
        self.transform = transform
        PropagatorLayer.__init__(
            self,
            grid_out,
            focal_length,
            defocus,
            abcd=abcd,
            inverse=inverse,
            alias=alias,
        )

    def context(self, wavefront: Wavefront, **context) -> dict[str, Any]:
        """Build local coordinates once, retaining incident pixel sampling."""
        if self.grid is not None:
            wavefront = wavefront.set(grid=self.grid)
        context = super().context(wavefront, **context)
        coordinates = context.pop("coordinates")
        if self.transform is not None:
            coordinates = self.transform(coordinates, **context)
        return {**context, "coordinates": coordinates}

    def modulation(self, wavefront: Wavefront) -> tuple[Array, Array]:
        """Combine the inherited amplitude and phase operations on resolved fields."""
        amplitude = wavefront._to_phasor_shape(1.0)
        phase = wavefront._to_phasor_shape(0.0)
        for layer in (BaseOptic, ComplexLayer, RefractiveLayer):
            local_amplitude, local_phase = layer.modulation(self, wavefront)
            amplitude = amplitude * local_amplitude
            phase = phase + local_phase
        return amplitude, phase

    def scalar_phasor(self, wavefront: Wavefront) -> Array:
        """Combine already-resolved scalar effects with one exponentiation."""
        amplitude, phase = self.modulation(wavefront)
        return amplitude * np.exp(1j * phase)

    def phasor(self, wavefront: Wavefront, **context) -> Array:
        """Return local scalar modulation, excluding normalisation and Jones terms."""
        self = self.populate()
        local = self.resolve_local(**self.context(wavefront, **context))
        return local.scalar_phasor(wavefront)

    def apply_scalar(self, wavefront: Wavefront, *, normalise=False) -> Wavefront:
        """Apply prepared scalar effects, optionally normalising their output."""
        wavefront = wavefront.set(
            phasor=wavefront.phasor * self.scalar_phasor(wavefront)
        )
        return wavefront.normalise() if normalise else wavefront

    def apply_local(
        self, wavefront: Wavefront, *, normalise=False, polarisation=True, **context
    ) -> Wavefront:
        """Resolve supplied local context, apply scalar terms, then Jones effects.

        Normalisation is explicitly controlled here so sparse owners can normalise
        jointly across apertures before applying their Jones effects.
        """
        context = dict(
            context, wavelength=wavefront.wavelength, wavelengths=wavefront.wavelength
        )
        local = self.resolve_local(**context)
        wavefront = local.apply_scalar(wavefront, normalise=normalise)
        if polarisation:
            wavefront = local.apply_jones(wavefront)
        return wavefront

    def apply(self, wavefront: Wavefront, **context) -> Wavefront:
        """Apply the optic to a complete field with Wavefront-owned alignment."""
        return wavefront._map_wavefronts(self.apply_mono, **context)

    def apply_mono(self, wavefront: Wavefront, **context) -> Wavefront:
        self = self.populate()
        context = self.context(wavefront, **context)
        wavefront = self.apply_local(
            wavefront, normalise=bool(self.normalise), **context
        )
        if self.grid_out is not None:
            wavefront = self.propagate(wavefront, **context)
        return wavefront


class TiltLayer(OpticalLayer, dlu.Unitful):
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
    grid = dl.Grid(n=128, diam=1.0, unit="m")
    wavelengths = np.linspace(600e-9, 700e-9, 5)
    wavefront = dl.Wavefront(wavelength=wavelengths, grid=grid)

    # Apply an angular offset in convenient physical units
    tilt = dl.TiltLayer(angles=[20.0, -10.0], unit="mas")
    wavefront = tilt(wavefront)
    ```
    """

    angles: Array | zdx.Expression
    unit: str | None = "rad"

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
        if not isinstance(self.angles, zdx.Expression) and self.angles.shape != (2,):
            raise ValueError("angles must have shape (2,).")
        self.unit = self.validate_unit(unit, dimension="angle", name="tilt unit")

        self.alias = alias

    def apply_mono(self, wavefront: Wavefront, **context) -> Wavefront:
        """Apply the configured angular tilt to one wavefront.

        Angles use the layer's declared unit and physical ``(x, y)`` order. The
        returned wavefront retains its sampling grid.
        """
        angles = zdx.resolve(self.angles, **self.context(wavefront, **context))
        return wavefront.tilt(angles, self.unit)


class WedgeLayer(OpticalLayer):
    """Apply the optical path of a thin refractive wedge.

    The two wedge angles define a linear thickness ramp. A constant or parametric
    refractive index is resolved at the incident wavelength and converted into OPD
    before being applied to the wavefront.

    Parameters
    ----------
    angle : ArrayLike
        Two wedge angles in radians along the physical ``(x, y)`` axes.
    n : Array or Expression
        Wavelength-dependent or fixed refractive index.
    """

    angle: Array | zdx.Expression
    n: Array | zdx.Expression

    def __init__(self, angle, n, *, alias=None):
        """Initialise a refractive wedge.

        Parameters
        ----------
        angle : ArrayLike
            Two-component wedge slope.
        n : Array or Expression
            Refractive index evaluated at the wavefront wavelength.
        """
        self.angle = zdx.as_array(angle, dtype=float)
        if not isinstance(self.angle, zdx.Expression) and self.angle.shape != (2,):
            raise ValueError("angle must have shape (2,).")
        self.n = zdx.as_array(n, dtype=float)

        self.alias = alias

    def apply_mono(self, wavefront: Wavefront, **context) -> Wavefront:
        """Apply the wavelength-dependent optical path of the wedge.

        Wedge angle is in radians, coordinates and thickness are in metres, and the
        refractive index is resolved at each wavefront wavelength.
        """
        # Resolve the sampled wedge thickness
        self = self.resolve(**self.context(wavefront, **context))
        coordinates = wavefront.coordinates
        x, y = coordinates[..., 0, :, :], coordinates[..., 1, :, :]
        thickness = x * np.tan(self.angle[0]) + y * np.tan(self.angle[1])

        # Convert thickness into refractive optical path
        return wavefront.add_opd((self.n - 1) * thickness)
