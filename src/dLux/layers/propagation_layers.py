"""Fourier, angular-spectrum, and ABCD propagation layers."""

from __future__ import annotations
import jax.numpy as np
from abcdLux import abcd, lct, asm
import dLux.utils as dlu

from ..abcd import BaseABCDElement
from ..coordinates import CoordSpec, PadSpec
from .optical_layers import OpticalLayer

__all__ = [
    "Propagator",
    "FFT",
    "MFT",
    "ABCDPropagator",
    "MFTPropagator",
    "FFTPropagator",
    "ASMPropagator",
]


class Propagator(OpticalLayer):
    """Base class for direct Fourier propagators."""

    focal_length: float | None
    inverse: bool

    def __init__(self, focal_length: float | None = None, inverse: bool = False):
        if focal_length is not None:
            focal_length = np.asarray(focal_length, float)
        self.focal_length = focal_length
        self.inverse = bool(inverse)


class FFT(Propagator):
    """Propagate a wavefront using its FFT propagation interface."""

    pad: int
    crop: int
    center: bool

    def __init__(
        self,
        focal_length: float = None,
        inverse: bool = False,
        pad: int = 1,
        crop: int = 1,
        center: bool = True,
    ):
        super().__init__(focal_length=focal_length, inverse=inverse)
        self.pad = int(pad)
        self.crop = int(crop)
        self.center = bool(center)

    def __call__(self, wavefront):
        spec = CoordSpec(c=0.0) if self.center else None
        size_out = wavefront.npixels * self.pad // self.crop
        return wavefront.propagate_FFT(
            pad=self.pad,
            focal_length=self.focal_length,
            inverse=self.inverse,
            spec_out=spec,
        ).resize(size_out)


class MFT(Propagator):
    """Propagate a wavefront using its matrix Fourier transform interface."""

    npixels: int
    pixel_scale: float

    def __init__(
        self,
        npixels: int,
        pixel_scale: float,
        focal_length: float = None,
        inverse: bool = False,
    ):
        super().__init__(focal_length=focal_length, inverse=inverse)
        self.pixel_scale = np.asarray(pixel_scale, float)
        self.npixels = int(npixels)

    def __call__(self, wavefront):
        return wavefront.propagate(
            npixels=self.npixels,
            pixel_scale=self.pixel_scale,
            focal_length=self.focal_length,
            inverse=self.inverse,
        )


###################
### Propagators ###
###################
class ABCDPropagator(OpticalLayer):
    """
    Propagator defined by a composition of ABCD elements.

    ??? abstract "UML"
        ![UML](../assets/uml/ABCDPropagator.png)

    Attributes
    ----------
    ABCDs : dict
        Dictionary of ABCD elements in propagation order.
    spec : CoordSpec | PadSpec
        Output coordinate specification.
    abcd : Array, property
        The composed ABCD matrix for this propagator.
    """

    ABCDs: dict
    spec: CoordSpec

    def __init__(self, ABCDs, spec):
        """
        Parameters
        ----------
        ABCDs : list[BaseABCDElement] | dict[str, BaseABCDElement]
            ABCD elements to compose into a single propagation transform.
        spec : CoordSpec | PadSpec
            Output coordinate specification.
        """
        self.ABCDs = dlu.list2dictionary(ABCDs, True, allowed_types=(BaseABCDElement,))
        self.spec = spec

    def __getattr__(self, key):
        """Resolve missing attributes via `spec` or child ABCD elements."""
        if hasattr(self.spec, key):
            return getattr(self.spec, key)
        if key in self.ABCDs.keys():
            return self.ABCDs[key]
        for layer in list(self.ABCDs.values()):
            if hasattr(layer, key):
                return getattr(layer, key)
        raise dlu.missing_attribute_error(self, key, list(self.ABCDs.keys()))

    @property
    def abcd(self):
        """
        Returns
        -------
        matrix : Array
            The composed ABCD matrix for this propagator.
        """
        return abcd.compose_abcd([m.abcd for m in self.ABCDs.values()])


class MFTPropagator(ABCDPropagator):
    """
    A matrix Fourier transform (MFT) propagator represented by an ABCD matrix.

    Note: Always returns a wavefront in an 'Intermediate' plane, even if the
    propagation is to a conjugate plane. Future ABCDWavefronts may enable better plane
    tracking, but the present Wavefront class is not compatible with this formulation.

    ??? abstract "UML"
        ![UML](../assets/uml/MFTPropagator.png)

    Parameters
    ----------
    ABCDs: list[BaseABCDElement]
        A list of ABCD elements to compose into the overall ABCD matrix for the
        propagation.
    spec: CoordSpec
        The coordinate specification of the output wavefront.
    """

    def __call__(self, wavefront):
        """
        Propagate a wavefront using an LCT-based matrix Fourier transform.

        Parameters
        ----------
        wavefront : Wavefront
            Input wavefront to propagate.

        Returns
        -------
        wavefront : Wavefront
            Propagated wavefront with updated field and output specification.
        """
        # Define input-output coordinates
        spec_in = wavefront.spec
        spec_out = self.spec

        # Propagate the field
        field = lct.lct_prop(
            u_in=wavefront.phasor,
            spec_in=spec_in.xs,
            spec_out=spec_out.xs,
            lam=wavefront.wavelength,
            ABCD=self.abcd,
        )

        # Update wavefront
        return wavefront.set(phasor=field, spec=spec_out)


class FFTPropagator(ABCDPropagator):
    """
    FFT-based ABCD propagator with optional padding and cropping.

    Note: Always returns a wavefront in an 'Intermediate' plane, even if the
    propagation is to a conjugate plane. Future ABCDWavefronts may enable better plane
    tracking, but the present Wavefront class is not compatible with this formulation.

    ??? abstract "UML"
        ![UML](../assets/uml/FFTPropagator.png)

    Parameters
    ----------
    ABCDs : list[BaseABCDElement] | dict[str, BaseABCDElement]
        ABCD elements to compose into a single propagation transform.
    spec : CoordSpec | PadSpec
        Output coordinate specification. If `CoordSpec` is provided, `d` must be None.
    """

    def __init__(self, ABCDs, spec):
        """
        Parameters
        ----------
        ABCDs : list[BaseABCDElement] | dict[str, BaseABCDElement]
            ABCD elements to compose into a single propagation transform.
        spec : CoordSpec | PadSpec
            Output coordinate specification. If `CoordSpec` is provided, `d` must be
            None.
        """
        if isinstance(spec, CoordSpec) and (spec.d is not None):
            raise ValueError("FFTPropagator CoordSpec can not specify d.")
        super().__init__(ABCDs=ABCDs, spec=spec)

    def __call__(self, wavefront):
        """
        Propagate a wavefront using an FFT-based LCT approximation.

        Parameters
        ----------
        wavefront : Wavefront
            Input wavefront to propagate.

        Returns
        -------
        wavefront : Wavefront
            Propagated wavefront with updated field and sampling metadata.
        """

        # Get the input spec
        spec_in = wavefront.spec
        lam = wavefront.wavelength

        # Handle the two spec options
        if isinstance(self.spec, CoordSpec):
            n_padded = self.spec.n
        else:
            n_padded = tuple(value * self.spec.pad for value in spec_in.n)

        # Get the effective focal length for the ABCD system
        fl = abcd.abcd_effective_focal_length(self.abcd)

        # Calculate the phase ramp for the FFT propagation offset
        d_fft, c_fft = dlu.FFT_spec(n_padded, spec_in.d, lam, fl)
        in_ramp = dlu.FFT_ramp(spec_in.xs, lam, c_fft - self.spec.c, fl)

        # Calculate the output phase ramp correction
        spec_out = CoordSpec(n=n_padded, c=self.spec.c, d=d_fft)

        # FFT-based LCT propagation
        field, spec_out_xys = lct.lct_prop_fft(
            u_in=wavefront.phasor * in_ramp,
            spec_in=wavefront.xs,
            lam=wavefront.wavelength,
            ABCD=self.abcd,
            npad=spec_out.n,
        )

        # Apply the crop if specified
        if isinstance(self.spec, PadSpec) and self.spec.crop > 1:
            n_out = field.shape[0] // self.spec.crop
            field = dlu.crop_to(field, n_out)

        # # Get the output coordinates
        xs_out, ys_out = spec_out_xys
        dx_out = xs_out[1] - xs_out[0]
        dy_out = ys_out[1] - ys_out[0]
        spec_out = CoordSpec(
            n=field.shape[-2:][::-1],
            d=(dx_out, dy_out),
            c=self.spec.c,
            unit=spec_in.unit,
        )

        # Update wavefront
        return wavefront.set(phasor=field, spec=spec_out)


class ASMPropagator(OpticalLayer):
    """
    Angular Spectrum Method (ASM) propagator.

    Note: Always returns a wavefront in an 'Intermediate' plane, even if the
    propagation is to a conjugate plane. Future ABCDWavefronts may enable better plane
    tracking, but the present Wavefront class is not compatible with this formulation.

    ??? abstract "UML"
        ![UML](../assets/uml/ASMPropagator.png)
    """

    distance: float
    spec: CoordSpec

    def __init__(self, distance, spec):
        """
        Parameters
        ----------
        distance : float
            Propagation distance.
        spec : CoordSpec | PadSpec
            Output specification. If `CoordSpec` is provided, `d` and `c` must be None.
        """
        self.distance = np.asarray(distance, float)
        if isinstance(spec, CoordSpec):
            if spec.d is not None or spec.c is not None:
                raise ValueError("ASMPropagator CoordSpec can not specify d or c.")
        self.spec = spec

    def __getattr__(self, key):
        """Resolve missing attributes via `spec`."""
        if hasattr(self.spec, key):
            return getattr(self.spec, key)
        raise dlu.missing_attribute_error(self, key)

    def __call__(self, wavefront):
        """
        Propagate a wavefront using the angular spectrum method.

        Parameters
        ----------
        wavefront : Wavefront
            Input wavefront to propagate.

        Returns
        -------
        wavefront : Wavefront
            Propagated wavefront with updated field.
        """

        # Get padding
        if isinstance(self.spec, CoordSpec):
            n_padded = self.spec.n
        else:
            n_padded = wavefront.npixels * self.spec.pad

        # Propagate the field
        field = asm.asm_prop(
            u_in=wavefront.phasor,
            spec_in=wavefront.xs,
            lam=wavefront.wavelength,
            z=self.distance,
            npad=n_padded,
            crop=False,
        )

        # Apply the crop if specified
        if isinstance(self.spec, PadSpec) and self.spec.crop > 1:
            n_out = field.shape[0] // self.spec.crop
            field = dlu.crop_to(field, n_out)

        # ASM preserves the physical sampling while padding/cropping changes the
        # sampled spatial shape.
        n_out = field.shape[-2:][::-1]
        spec_out = wavefront.spec.set(n=n_out)
        return wavefront.set(phasor=field, spec=spec_out)


class Fraunhofer(ABCDPropagator):
    """
    Placeholder for a dedicated Fraunhofer propagator.

    ??? abstract "UML"
        ![UML](../assets/uml/Fraunhofer.png)
    """

    spec_out: tuple[int, float]
    focal_length: float

    def __init__(self):
        raise NotImplementedError("Fraunhofer propagator is not yet implemented.")


class Fresnel(ABCDPropagator):
    """
    Placeholder for a dedicated Fresnel propagator.

    ??? abstract "UML"
        ![UML](../assets/uml/Fresnel.png)
    """

    spec_out: tuple[int, float]
    focal_length: float
    defocus: float

    def __init__(self):
        raise NotImplementedError("Fresnel propagator is not yet implemented.")
