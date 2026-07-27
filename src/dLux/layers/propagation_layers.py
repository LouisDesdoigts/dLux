"""Physical and ABCD-based wavefront propagation layers."""

from __future__ import annotations

import jax.numpy as np
import zodiax as zdx
from jax import Array

import dLux.utils as dlu

from ..grids import BaseGridSpec, GridSpec, ResizeSpec
from .optical_layers import OpticalLayer

__all__ = [
    "ABCDElement",
    "ABCDFreeSpace",
    "ABCDLens",
    "ABCDMirror",
    "ABCDFraunhofer",
    "Propagator",
    "FocalPropagator",
    "ABCDPropagator",
    "FreeSpace",
    "Fraunhofer",
    "Fresnel",
]


def _propagation_inputs(wf):
    """Broadcast wavelength and coordinate axes over non-spatial field dimensions."""
    wavelength = np.asarray(wf.wavelength)
    extra = wf.phasor.ndim - wavelength.ndim - 2
    wavelength = wavelength.reshape(wavelength.shape + (1,) * extra)
    x, y = wf.axes
    if wf.is_polarised:
        x, y = x[..., None, None, :], y[..., None, None, :]
    return wavelength, x, y


def _propagate_mft(wf, spec, ABCD=None, **kwargs):
    """Propagate every field to an explicit output grid."""
    wavelength, x, y = _propagation_inputs(wf)
    axes_out = spec.axes

    def propagate(field, lam, x, y):
        if ABCD is None:
            return dlu.MFT(field, lam, (x, y), axes_out, **kwargs)
        return dlu.ABCD_MFT(field, lam, (x, y), axes_out, ABCD)

    propagate = np.vectorize(propagate, signature="(n,m),(),(m),(n)->(p,q)")
    return wf.set(phasor=propagate(wf.phasor, wavelength, x, y), spec=spec)


def _propagate_fft(wf, spec, unit, ABCD=None, **kwargs):
    """Propagate every field at native FFT sampling."""
    center = dlu.as_axis(spec.c, 2, "c")
    center = None if center is None else center * dlu.unit_factor(unit)
    padding = spec.padding

    def propagate(field, wavelength, x, y):
        fn = dlu.FFT if ABCD is None else dlu.ABCD_FFT
        inputs = kwargs if ABCD is None else {"ABCD": ABCD}
        field, axes = fn(
            field, wavelength, (x, y), output_center=center, **padding, **inputs
        )
        return field, *axes

    propagate = np.vectorize(propagate, signature="(n,m),(),(m),(n)->(p,q),(q),(p)")
    wavelength, x, y = _propagation_inputs(wf)
    field, x, y = propagate(wf.phasor, wavelength, x, y)
    if wf.is_polarised:
        x, y = x[..., 0, 0, :], y[..., 0, 0, :]

    if any(f > 1 for f in spec.crop_factor):
        nx, ny = spec.crop_size(field.shape)
        sy, sx = (field.shape[-2] - ny) // 2, (field.shape[-1] - nx) // 2
        field, x, y = (
            field[..., sy : sy + ny, sx : sx + nx],
            x[..., sx : sx + nx],
            y[..., sy : sy + ny],
        )

    scale = dlu.unit_factor(unit)
    d = np.stack((x[..., 1] - x[..., 0], y[..., 1] - y[..., 0]), -1) / scale
    c = np.stack(((x[..., -1] + x[..., 0]) / 2, (y[..., -1] + y[..., 0]) / 2), -1)
    spec = wf.spec.set(n=field.shape[-2:][::-1], d=d, c=c / scale, unit=unit)
    return wf.set(phasor=field, spec=spec)


def _propagate_free_space(wf, spec, distance, crop):
    """Propagate every field over a free-space distance."""
    wavelength, x, y = _propagation_inputs(wf)
    padding = spec.padding
    propagate = np.vectorize(
        lambda field, lam, x, y: dlu.ASM(
            field, lam, (x, y), distance, crop=crop, **padding
        ),
        signature="(n,m),(),(m),(n)->(n,m)" if crop else "(n,m),(),(m),(n)->(p,q)",
    )
    field = propagate(wf.phasor, wavelength, x, y)
    spec = wf.spec if crop else wf.spec.set(n=field.shape[-2:][::-1])
    return wf.set(phasor=field, spec=spec)


def _validate_grid(spec, name, ndim=2, angular=None):
    """Validate a complete propagation grid and its coordinate unit."""
    if spec.n is None or spec.d is None or spec.unit is None:
        raise ValueError(f"The {name} GridSpec requires n, d, and unit.")
    if spec.ndim != ndim:
        raise ValueError(f"The {name} GridSpec must have {ndim} dimensions.")
    try:
        dlu.unit_factor_to_rad(spec.unit)
        is_angular = True
    except ValueError:
        is_angular = False
    if angular is not None and is_angular != angular:
        unit_type = "angular" if angular else "physical"
        raise ValueError(f"The {name} GridSpec must use {unit_type} units.")
    return is_angular


def _validate_method(method, spec, types):
    """Validate a propagation method and its required specification type."""
    method = str(method).lower()
    if method not in types:
        methods = "', '".join(types)
        raise ValueError(f"method must be '{methods}'.")
    if not isinstance(spec, types[method]):
        raise TypeError(
            f"{method.upper()} propagation requires a {types[method].__name__}."
        )
    return method


class ABCDElement(zdx.Base):
    """Base class for elements represented by an ABCD matrix."""


class ABCDFreeSpace(ABCDElement):
    """A free-space propagation element represented by an ABCD matrix."""

    distance: float

    def __init__(self, distance):
        self.distance = np.asarray(distance, float)

    @property
    def abcd(self):
        """Return the analytic ABCD matrix for free-space propagation."""
        return dlu.abcd_free_space(self.distance)


class ABCDLens(ABCDElement):
    """A thin lens represented by an ABCD matrix."""

    focal_length: float

    def __init__(self, focal_length):
        self.focal_length = np.asarray(focal_length, float)

    @property
    def abcd(self):
        """Return the analytic ABCD matrix for the lens."""
        return dlu.abcd_lens(self.focal_length)


class ABCDMirror(ABCDElement):
    """A curved mirror represented by an ABCD matrix."""

    radius: float

    def __init__(self, radius):
        self.radius = np.asarray(radius, float)

    @property
    def abcd(self):
        """Return the analytic ABCD matrix for the mirror."""
        return dlu.abcd_mirror(self.radius)


class ABCDFraunhofer(ABCDElement):
    """A far-field transform represented by an ABCD matrix."""

    focal_length: float

    def __init__(self, focal_length):
        self.focal_length = np.asarray(focal_length, float)

    @property
    def abcd(self):
        """Return the analytic ABCD matrix for far-field propagation."""
        return dlu.abcd_fraunhofer(self.focal_length)


class Propagator(OpticalLayer):
    """Base propagation layer holding an output sampling specification."""

    spec: BaseGridSpec

    def __init__(self, spec):
        if not isinstance(spec, (GridSpec, ResizeSpec)):
            raise TypeError("spec must be a GridSpec or ResizeSpec.")
        self.spec = spec.broadcast(2)

    def validate(self, wavefront):
        """Validate the input coordinate specification."""
        _validate_grid(wavefront.spec, "input", angular=False)


class FocalPropagator(Propagator):
    """Base propagation layer with optional physical focal scaling."""

    spec: BaseGridSpec
    focal_length: Array | None

    def __init__(self, spec, focal_length=None):
        super().__init__(spec)
        self.focal_length = (
            None if focal_length is None else np.asarray(focal_length, dtype=float)
        )

    def validate(self, wavefront):
        """Validate the input and explicitly requested output coordinates."""
        super().validate(wavefront)
        if isinstance(self.spec, ResizeSpec):
            return
        angular = _validate_grid(self.spec, "output", wavefront.spec.ndim)
        if self.focal_length is None and not angular:
            raise ValueError(
                "Propagation without a focal length requires angular output units."
            )
        if self.focal_length is not None and angular:
            raise ValueError(
                "Propagation with a focal length requires physical output units."
            )


class Fraunhofer(FocalPropagator):
    """Conjugate-plane propagation using an MFT or FFT."""

    spec: BaseGridSpec
    focal_length: Array | None
    method: str

    def __init__(self, spec, focal_length=None, method="mft"):
        method = _validate_method(method, spec, {"mft": GridSpec, "fft": ResizeSpec})
        super().__init__(spec, focal_length)
        self.method = method

    def __call__(self, wavefront):
        self.validate(wavefront)
        if self.method == "fft":
            unit = "rad" if self.focal_length is None else wavefront.spec.unit
            return _propagate_fft(
                wavefront, self.spec, unit, focal_length=self.focal_length
            )
        return _propagate_mft(wavefront, self.spec, focal_length=self.focal_length)


class Fresnel(FocalPropagator):
    """Defocused focal propagation using an FFT, MFT, or LCT."""

    spec: BaseGridSpec
    focal_length: Array | None
    defocus: Array
    method: str

    def __init__(self, spec, defocus=0.0, focal_length=None, method="lct"):
        types = {"fft": ResizeSpec, "mft": GridSpec, "lct": GridSpec}
        method = _validate_method(method, spec, types)
        super().__init__(spec, focal_length)
        self.method = method
        self.defocus = np.asarray(defocus, dtype=float)

    def __call__(self, wavefront):
        self.validate(wavefront)
        if self.method == "fft":
            unit = "rad" if self.focal_length is None else wavefront.spec.unit
            kwargs = {"focal_length": self.focal_length, "defocus": self.defocus}
            return _propagate_fft(wavefront, self.spec, unit, **kwargs)
        return _propagate_mft(
            wavefront, self.spec, focal_length=self.focal_length, defocus=self.defocus
        )


class ABCDPropagator(Propagator):
    """Propagate through an ordered ABCD system using an LCT or FFT."""

    spec: BaseGridSpec
    ABCDs: dict
    method: str

    def __init__(self, ABCDs, spec, method="lct"):
        super().__init__(spec)
        method = _validate_method(
            method, self.spec, {"lct": GridSpec, "fft": ResizeSpec}
        )

        elements = list(ABCDs.items()) if isinstance(ABCDs, dict) else ABCDs
        self.ABCDs = dlu.list2dictionary(
            elements, ordered=True, allowed_types=(ABCDElement,)
        )
        if not self.ABCDs:
            raise ValueError("ABCDs must contain at least one element.")
        self.method = method

    @property
    def abcd(self) -> Array:
        """Return the composed ABCD matrix."""
        return dlu.compose_abcd([element.abcd for element in self.ABCDs.values()])

    def validate(self, wavefront):
        """Validate physical ABCD input and output coordinates."""
        Propagator.validate(self, wavefront)
        if isinstance(self.spec, ResizeSpec):
            return
        _validate_grid(self.spec, "output", wavefront.spec.ndim, angular=False)

    def __call__(self, wavefront):
        self.validate(wavefront)
        if self.method == "fft":
            return _propagate_fft(
                wavefront, self.spec, unit=wavefront.spec.unit, ABCD=self.abcd
            )
        return _propagate_mft(wavefront, self.spec, ABCD=self.abcd)


class FreeSpace(Propagator):
    """Paraxial angular-spectrum propagation over a free-space distance."""

    spec: BaseGridSpec
    distance: Array
    crop: bool

    def __init__(self, distance, spec=None, crop=True):
        if spec is None:
            spec = ResizeSpec()
        if not isinstance(spec, ResizeSpec):
            raise TypeError("FreeSpace spec must be a ResizeSpec.")
        super().__init__(spec)
        self.distance = np.asarray(distance, dtype=float)
        self.crop = bool(crop)

    def __call__(self, wavefront):
        self.validate(wavefront)
        return _propagate_free_space(wavefront, self.spec, self.distance, self.crop)
