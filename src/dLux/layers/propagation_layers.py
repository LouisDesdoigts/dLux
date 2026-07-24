"""Physical and ABCD-based wavefront propagation layers."""

from __future__ import annotations

import jax.numpy as np
from abcdLux import abcd, asm, lct
from jax import Array

import dLux.utils as dlu

from ..abcd import BaseABCDElement
from ..coordinates import BaseSpec, CoordSpec, PadSpec
from .optical_layers import OpticalLayer

__all__ = [
    "Propagator",
    "FocalPropagator",
    "ABCDPropagator",
    "ASM",
    "Fraunhofer",
    "Fresnel",
]


class Propagator(OpticalLayer):
    """Base propagation layer holding an output sampling specification."""

    spec: BaseSpec

    def __init__(self, spec):
        if not isinstance(spec, (CoordSpec, PadSpec)):
            raise TypeError("spec must be a CoordSpec or PadSpec.")
        self.spec = spec.broadcast(2) if isinstance(spec, CoordSpec) else spec

    def validate(self, wavefront):
        """Validate the input coordinate specification."""
        spec = wavefront.spec
        if spec.n is None or spec.d is None or spec.unit is None:
            raise ValueError("The input Wavefront requires n, d, and unit.")
        if spec.ndim != 2:
            raise ValueError("Propagation requires two-dimensional coordinates.")
        try:
            dlu.unit_factor_to_rad(spec.unit)
        except ValueError:
            return
        raise ValueError("Input Wavefront coordinates must use physical units.")

    def propagate_lct(self, wavefront, matrix):
        """Propagate to an explicit output grid through the abcdLux LCT."""
        wavelength = np.asarray(wavefront.wavelength)
        extra = wavefront.phasor.ndim - wavelength.ndim - 2
        wavelength = wavelength.reshape(wavelength.shape + (1,) * extra)
        x, y = wavefront.axes
        propagate = np.vectorize(
            lambda field, lam, x, y: lct.lct_prop(
                u_in=field,
                spec_in=(x, y),
                spec_out=self.spec.axes,
                lam=lam,
                ABCD=matrix,
            ),
            signature="(n,m),(),(m),(n)->(p,q)",
        )
        field = propagate(wavefront.phasor, wavelength, x, y)
        return wavefront.set(phasor=field, spec=self.spec)

    def propagate_fft(self, wavefront, matrix, unit):
        """Propagate at native FFT sampling through the abcdLux LCT."""
        if np.asarray(wavefront.wavelength).ndim:
            raise ValueError(
                "FFT propagation requires a monochromatic Wavefront because its "
                "output sampling depends on wavelength."
            )
        npad = tuple(size * self.spec.pad for size in wavefront.spec.n)
        field, axes = lct.lct_prop_fft(
            u_in=wavefront.phasor,
            spec_in=wavefront.axes,
            lam=wavefront.wavelength,
            ABCD=matrix,
            npad=npad,
        )
        if self.spec.crop > 1:
            ny, nx = (size // self.spec.crop for size in field.shape[-2:])
            sy = (field.shape[-2] - ny) // 2
            sx = (field.shape[-1] - nx) // 2
            field = field[..., sy : sy + ny, sx : sx + nx]
            axes = (
                axes[0][sx : sx + nx],
                axes[1][sy : sy + ny],
            )
        x, y = axes
        d = np.asarray((x[1] - x[0], y[1] - y[0]))
        c = np.asarray(((x[-1] + x[0]) / 2, (y[-1] + y[0]) / 2))
        scale = dlu.unit_factor(unit)
        spec = wavefront.spec.set(
            n=field.shape[-2:][::-1],
            d=d / scale,
            c=c / scale,
            unit=unit,
        )
        return wavefront.set(phasor=field, spec=spec)


class FocalPropagator(Propagator):
    """Base propagation layer with optional physical focal scaling."""

    focal_length: Array | None

    def __init__(self, spec, focal_length=None):
        super().__init__(spec)
        self.focal_length = (
            None if focal_length is None else np.asarray(focal_length, dtype=float)
        )

    def validate(self, wavefront):
        """Validate the input and explicitly requested output coordinates."""
        super().validate(wavefront)
        if isinstance(self.spec, PadSpec):
            return
        if self.spec.n is None or self.spec.d is None or self.spec.unit is None:
            raise ValueError("The output CoordSpec requires n, d, and unit.")
        if self.spec.ndim != wavefront.spec.ndim:
            raise ValueError("Input and output coordinate dimensionality must match.")
        try:
            dlu.unit_factor_to_rad(self.spec.unit)
            angular = True
        except ValueError:
            angular = False
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

    method: str

    def __init__(self, spec, focal_length=None, method="mft"):
        method = str(method).lower()
        if method not in ("mft", "fft"):
            raise ValueError("method must be 'mft' or 'fft'.")
        if method == "mft" and not isinstance(spec, CoordSpec):
            raise TypeError("MFT propagation requires a CoordSpec.")
        if method == "fft" and not isinstance(spec, PadSpec):
            raise TypeError("FFT propagation requires a PadSpec.")
        super().__init__(spec, focal_length)
        self.method = method

    def __call__(self, wavefront):
        self.validate(wavefront)
        matrix = abcd.abcd_fraunhofer(
            1.0 if self.focal_length is None else self.focal_length
        )
        if self.method == "fft":
            unit = "rad" if self.focal_length is None else wavefront.spec.unit
            return self.propagate_fft(wavefront, matrix, unit)
        return self.propagate_lct(wavefront, matrix)


class Fresnel(FocalPropagator):
    """Defocused focal propagation using an FFT, MFT, or LCT."""

    defocus: Array
    method: str

    def __init__(self, spec, defocus=0.0, focal_length=None, method="lct"):
        method = str(method).lower()
        if method not in ("fft", "mft", "lct"):
            raise ValueError("method must be 'fft', 'mft', or 'lct'.")
        if method in ("mft", "lct") and not isinstance(spec, CoordSpec):
            raise TypeError("MFT and LCT propagation require a CoordSpec.")
        if method == "fft" and not isinstance(spec, PadSpec):
            raise TypeError("FFT propagation requires a PadSpec.")
        super().__init__(spec, focal_length)
        self.method = method
        self.defocus = np.asarray(defocus, dtype=float)

    def __call__(self, wavefront):
        self.validate(wavefront)
        focal_length = 1.0 if self.focal_length is None else self.focal_length
        matrix = abcd.compose_abcd(
            [
                abcd.abcd_fraunhofer(focal_length),
                abcd.abcd_free_space(self.defocus),
            ]
        )
        if self.method == "fft":
            unit = "rad" if self.focal_length is None else wavefront.spec.unit
            return self.propagate_fft(wavefront, matrix, unit)
        return self.propagate_lct(wavefront, matrix)


class ABCDPropagator(Propagator):
    """Propagate through an ordered ABCD system using an LCT or FFT."""

    ABCDs: dict
    method: str

    def __init__(self, ABCDs, spec, method="lct"):
        super().__init__(spec)
        method = str(method).lower()
        if method not in ("lct", "fft"):
            raise ValueError("method must be 'lct' or 'fft'.")
        if method == "fft" and not isinstance(self.spec, PadSpec):
            raise TypeError("FFT propagation requires a PadSpec.")
        if method == "lct" and not isinstance(self.spec, CoordSpec):
            raise TypeError("LCT propagation requires a CoordSpec.")

        elements = list(ABCDs.items()) if isinstance(ABCDs, dict) else ABCDs
        self.ABCDs = dlu.list2dictionary(
            elements,
            ordered=True,
            allowed_types=(BaseABCDElement,),
        )
        if not self.ABCDs:
            raise ValueError("ABCDs must contain at least one element.")
        self.method = method

    @property
    def abcd(self) -> Array:
        """Return the composed ABCD matrix."""
        return abcd.compose_abcd([element.abcd for element in self.ABCDs.values()])

    def validate(self, wavefront):
        """Validate physical ABCD input and output coordinates."""
        Propagator.validate(self, wavefront)
        if isinstance(self.spec, PadSpec):
            return
        if self.spec.n is None or self.spec.d is None or self.spec.unit is None:
            raise ValueError("The output CoordSpec requires n, d, and unit.")
        if self.spec.ndim != wavefront.spec.ndim:
            raise ValueError("Input and output coordinate dimensionality must match.")
        try:
            dlu.unit_factor_to_rad(self.spec.unit)
        except ValueError:
            return
        raise ValueError("ABCD output coordinates must use physical units.")

    def __call__(self, wavefront):
        self.validate(wavefront)
        if self.method == "fft":
            return self.propagate_fft(
                wavefront,
                self.abcd,
                unit=wavefront.spec.unit,
            )
        return self.propagate_lct(wavefront, self.abcd)


class ASM(Propagator):
    """Paraxial angular-spectrum propagation over a free-space distance."""

    distance: Array

    def __init__(self, distance, spec=None):
        if spec is None:
            spec = PadSpec()
        if not isinstance(spec, PadSpec):
            raise TypeError("ASM spec must be a PadSpec.")
        super().__init__(spec)
        self.distance = np.asarray(distance, dtype=float)

    def __call__(self, wavefront):
        self.validate(wavefront)
        wavelength = np.asarray(wavefront.wavelength)
        extra = wavefront.phasor.ndim - wavelength.ndim - 2
        wavelength = wavelength.reshape(wavelength.shape + (1,) * extra)
        npad = tuple(size * self.spec.pad for size in wavefront.spec.n)
        x, y = wavefront.axes
        propagate = np.vectorize(
            lambda field, lam, x, y: asm.asm_prop(
                u_in=field,
                spec_in=(x, y),
                lam=lam,
                z=self.distance,
                npad=npad,
                crop=False,
            ),
            signature="(n,m),(),(m),(n)->(p,q)",
        )
        field = propagate(wavefront.phasor, wavelength, x, y)
        if self.spec.crop > 1:
            ny, nx = (size // self.spec.crop for size in field.shape[-2:])
            sy = (field.shape[-2] - ny) // 2
            sx = (field.shape[-1] - nx) // 2
            field = field[..., sy : sy + ny, sx : sx + nx]
        return wavefront.set(
            phasor=field,
            spec=wavefront.spec.set(n=field.shape[-2:][::-1]),
        )
