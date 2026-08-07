"""Physical and ABCD-based wavefront propagation layers."""

from __future__ import annotations

import jax.numpy as np
from jax import Array

import dLux.utils as dlu

from ..base import Base
from ..grids import BaseGridSpec, GridSpec, ResizeSpec
from .optical import OpticalLayer

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
    # Broadcast wavelength over intrinsic non-spatial field axes
    wavelength = np.asarray(wf.wavelength)
    extra = wf.phasor.ndim - wavelength.ndim - 2
    wavelength = wavelength.reshape(wavelength.shape + (1,) * extra)

    # Promote coordinate axes over Jones dimensions when required
    x, y = wf.axes
    if wf.is_polarised:
        x, y = x[..., None, None, :], y[..., None, None, :]
    return wavelength, x, y


def _propagate_mft(wf, spec, ABCD=None, **kwargs):
    """Propagate every field to an explicit output grid."""
    # Resolve input and requested output coordinate axes
    wavelength, x, y = _propagation_inputs(wf)
    axes_out = spec.axes

    # Define propagation of one monochromatic field
    def propagate(field, lam, x, y):
        if ABCD is None:
            return dlu.MFT(field, lam, (x, y), axes_out, **kwargs)
        return dlu.ABCD_MFT(field, lam, (x, y), axes_out, ABCD)

    # Vectorise propagation over leading field dimensions
    signature = "(n,m),(),(m),(n)->(p,q)"
    propagate = np.vectorize(propagate, signature=signature)
    phasor = propagate(wf.phasor, wavelength, x, y)

    # Update the propagated field and requested grid
    return wf.set(phasor=phasor, spec=spec)


def _propagate_fft(
    wf, spec, unit=None, ABCD=None, focal_length=None, inverse=False, **kwargs
):
    """Propagate every field at native FFT sampling."""
    # Resolve the output units, centre, and padding
    if unit is None:
        unit = "m" if inverse else "rad"
        if focal_length is not None:
            unit = wf.spec.unit

    scale = dlu.unit_factor(unit)
    center = dlu.as_axis(spec.c, 2, "c")
    center = None if center is None else center * scale
    padding = spec.padding

    # Configure the FFT propagation function and inputs
    if ABCD is None:
        fn = dlu.FFT
        inputs = {"focal_length": focal_length, "inverse": inverse, **kwargs}
    else:
        fn = dlu.ABCD_FFT
        inputs = {"ABCD": ABCD}

    # Define propagation over one monochromatic field
    def propagate(field, wavelength, x, y):
        field, axes = fn(
            field,
            wavelength,
            (x, y),
            output_center=center,
            **padding,
            **inputs,
        )
        return field, *axes

    # Vectorise propagation over the leading field axes
    signature = "(n,m),(),(m),(n)->(p,q),(q),(p)"
    propagate = np.vectorize(propagate, signature=signature)
    wavelength, x, y = _propagation_inputs(wf)
    field, x, y = propagate(wf.phasor, wavelength, x, y)

    # Remove the polarisation axes from the output coordinates
    if wf.is_polarised:
        x, y = x[..., 0, 0, :], y[..., 0, 0, :]

    # Crop the propagated field and coordinate axes
    field = spec.crop_array(field)
    x, y = spec.crop_axes((x, y))

    # Construct the realised output grid
    spec = GridSpec.from_axes((x, y), unit)

    # Update the propagated wavefront
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


class ABCDElement(Base):
    """Base class for elements represented by an ABCD matrix."""


class ABCDFreeSpace(ABCDElement):
    """A free-space propagation element represented by an ABCD matrix."""

    distance: float

    def __init__(self, distance):
        self.distance = dlu.to_value(distance)

    @property
    def abcd(self):
        """Return the analytic ABCD matrix for free-space propagation."""
        return dlu.abcd_free_space(self.distance)


class ABCDLens(ABCDElement):
    """A thin lens represented by an ABCD matrix."""

    focal_length: float

    def __init__(self, focal_length):
        self.focal_length = dlu.to_value(focal_length)

    @property
    def abcd(self):
        """Return the analytic ABCD matrix for the lens."""
        return dlu.abcd_lens(self.focal_length)


class ABCDMirror(ABCDElement):
    """A curved mirror represented by an ABCD matrix."""

    radius: float

    def __init__(self, radius):
        self.radius = dlu.to_value(radius)

    @property
    def abcd(self):
        """Return the analytic ABCD matrix for the mirror."""
        return dlu.abcd_mirror(self.radius)


class ABCDFraunhofer(ABCDElement):
    """A far-field transform represented by an ABCD matrix."""

    focal_length: float

    def __init__(self, focal_length):
        self.focal_length = dlu.to_value(focal_length)

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

    def apply(self, wavefront):
        """Propagate the complete vectorised wavefront state."""
        return self.apply_mono(wavefront)

    def validate(self, wavefront):
        """Validate the input coordinate specification."""
        _validate_grid(wavefront.spec, "input", angular=False)


class FocalPropagator(Propagator):
    """Base focal propagation layer with an explicit propagation direction."""

    spec: BaseGridSpec
    focal_length: Array | None
    inverse: bool

    def __init__(self, spec, focal_length=None, inverse=False):
        super().__init__(spec)
        self.focal_length = dlu.to_value(focal_length, optional=True)
        self.inverse = bool(inverse)

    def validate(self, wavefront):
        """Validate the input and explicitly requested output coordinates."""
        if isinstance(self.spec, ResizeSpec):
            angular = self.inverse and self.focal_length is None
            _validate_grid(wavefront.spec, "input", angular=angular)
            return
        if self.inverse:
            angular = self.focal_length is None
            _validate_grid(wavefront.spec, "input", angular=angular)
            _validate_grid(self.spec, "output", wavefront.spec.ndim, angular=False)
            return
        super().validate(wavefront)
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
    """Propagate between conjugate planes using an MFT or FFT.

    Parameters
    ----------
    spec : GridSpec or ResizeSpec
        Explicit MFT output grid or FFT resizing specification.
    focal_length : float or None
        Focal length in meters. Omit for angular focal-plane coordinates.
    method : {"mft", "fft"}
        Numerical propagation method.
    inverse : bool
        Propagate from the focal plane back to a physical pupil plane.
    """

    spec: BaseGridSpec
    focal_length: Array | None
    inverse: bool
    method: str

    def __init__(self, spec, focal_length=None, method="mft", inverse=False):
        method = _validate_method(method, spec, {"mft": GridSpec, "fft": ResizeSpec})
        super().__init__(spec, focal_length, inverse)
        self.method = method

    def apply_mono(self, wavefront):
        """Propagate a wavefront between conjugate planes."""
        self.validate(wavefront)
        if self.method == "fft":
            return _propagate_fft(
                wavefront,
                self.spec,
                focal_length=self.focal_length,
                inverse=self.inverse,
            )
        return _propagate_mft(
            wavefront, self.spec, focal_length=self.focal_length, inverse=self.inverse
        )


class Fresnel(FocalPropagator):
    """Propagate between defocused focal planes using an FFT, MFT, or LCT.

    Parameters
    ----------
    spec : GridSpec or ResizeSpec
        Explicit MFT/LCT output grid or FFT resizing specification.
    defocus : float
        Longitudinal defocus distance in meters.
    focal_length : float or None
        Focal length in meters. Omit for angular focal-plane coordinates.
    method : {"fft", "mft", "lct"}
        Numerical propagation method.
    inverse : bool
        Reverse MFT or LCT propagation direction. Inverse FFT propagation is not
        currently supported.
    """

    spec: BaseGridSpec
    focal_length: Array | None
    inverse: bool
    defocus: Array
    method: str

    def __init__(
        self, spec, defocus=0.0, focal_length=None, method="lct", inverse=False
    ):
        types = {"fft": ResizeSpec, "mft": GridSpec, "lct": GridSpec}
        method = _validate_method(method, spec, types)
        if inverse and method == "fft":
            raise ValueError(
                "Inverse Fresnel propagation is not supported with method='fft'."
            )
        super().__init__(spec, focal_length, inverse)
        self.method = method
        self.defocus = dlu.to_value(defocus)

    def apply_mono(self, wavefront):
        """Propagate a wavefront between defocused focal planes."""
        self.validate(wavefront)
        if self.method == "fft":
            return _propagate_fft(
                wavefront,
                self.spec,
                focal_length=self.focal_length,
                defocus=self.defocus,
            )
        return _propagate_mft(
            wavefront,
            self.spec,
            focal_length=self.focal_length,
            defocus=self.defocus,
            inverse=self.inverse,
        )


class ABCDPropagator(Propagator):
    """Propagate through an ordered ABCD system using an LCT or FFT.

    Parameters
    ----------
    ABCDs : sequence or dict
        Ordered ``ABCDElement`` objects composing the optical system.
    spec : GridSpec or ResizeSpec
        Explicit LCT output grid or FFT resizing specification.
    method : {"lct", "fft"}
        Numerical propagation method.
    """

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

    def __getattr__(self, key):
        """Raise grid parameters, named elements, and element parameters."""
        return dlu.resolve_attr(self, key, self.spec, self.ABCDs)

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

    def apply_mono(self, wavefront):
        """Propagate a wavefront through the composed ABCD system."""
        self.validate(wavefront)
        if self.method == "fft":
            return _propagate_fft(
                wavefront, self.spec, unit=wavefront.spec.unit, ABCD=self.abcd
            )
        return _propagate_mft(wavefront, self.spec, ABCD=self.abcd)


class FreeSpace(Propagator):
    """Paraxial angular-spectrum propagation over a free-space distance.

    Parameters
    ----------
    distance : float
        Signed propagation distance in meters.
    spec : ResizeSpec or None
        Optional padding, cropping, or output-size specification.
    crop : bool
        Crop the propagated array according to ``spec``.
    """

    spec: BaseGridSpec
    distance: Array
    crop: bool

    def __init__(self, distance, spec=None, crop=True):
        if spec is None:
            spec = ResizeSpec()
        if not isinstance(spec, ResizeSpec):
            raise TypeError("FreeSpace spec must be a ResizeSpec.")
        super().__init__(spec)
        self.distance = dlu.to_value(distance)
        self.crop = bool(crop)

    def apply_mono(self, wavefront):
        """Propagate a wavefront over the configured free-space distance."""
        self.validate(wavefront)

        # Define and vectorise monochromatic angular-spectrum propagation
        wavelength, x, y = _propagation_inputs(wavefront)
        prop_fn = lambda field, lam, x, y: dlu.ASM(
            field, lam, (x, y), self.distance, crop=False, **self.spec.padding
        )
        propagate = np.vectorize(prop_fn, signature="(n,m),(),(m),(n)->(p,q)")
        field = propagate(wavefront.phasor, wavelength, x, y)

        # Apply optional output cropping and update the realised grid
        field = self.spec.crop_array(field) if self.crop else field
        spec = wavefront.spec.resize(field.shape[-2:][::-1])
        return wavefront.set(phasor=field, spec=spec)
