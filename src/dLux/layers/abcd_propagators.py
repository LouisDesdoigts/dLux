from __future__ import annotations
import zodiax as zdx
import jax.numpy as np
from abcdLux import abcd, lct, asm
import dLux.utils as dlu

from .propagators import OpticalLayer
from ..coordinates import CoordSpec, PadSpec

__all__ = [
    "ABCDElement",
    "ABCDFreeSpace",
    "ABCDLens",
    "ABCDMirror",
    "ABCDConjugatePlane",
    "ABCDPropagator",
    "MFTPropagator",
    "FFTPropagator",
    "ASMPropagator",
]


class ABCDElement(zdx.Base):
    """
    An ABCD element is a layer that can be represented by an ABCD matrix. This is a
    base class for such elements, and should not be used directly.
    """

    pass


class ABCDFreeSpace(ABCDElement):
    """
    A free space propagation element represented by an ABCD matrix.
    """

    distance: float

    def __init__(self, distance):
        self.distance = np.array(distance, float)

    @property
    def abcd(self):
        """Analytic ABCD matrix for free space propagation"""
        return abcd.abcd_free_space(self.distance)


class ABCDLens(ABCDElement):
    """
    A lens element represented by an ABCD matrix. Note this element alone does not
    produce the standard 'pupil-focal Fourier relationship' as that is between the front
    and back focal planes. To represent a true pupil-focal plane propagation, apply a
    free space propagation of the focal length both before _and_ after the lens, or use
    the ABCDConjugatePlane element.
    """

    focal_length: float

    def __init__(self, focal_length):
        self.focal_length = np.array(focal_length, float)

    @property
    def abcd(self):
        """Analytic ABCD matrix for a lens"""
        return abcd.abcd_lens(self.focal_length)


class ABCDMirror(ABCDElement):
    """
    A mirror element represented by an ABCD matrix.
    """

    radius: float

    def __init__(self, radius):
        self.radius = np.array(radius, float)

    @property
    def abcd(self):
        """Analytic ABCD matrix for a mirror"""
        return abcd.abcd_mirror(self.radius)


class ABCDConjugatePlane(ABCDElement):
    """
    A conjugate plane element represented by an ABCD matrix.
    """

    focal_length: float

    def __init__(self, focal_length):
        self.focal_length = np.array(focal_length, float)

    @property
    def abcd(self):
        """Analytic ABCD matrix for a conjugate plane"""
        return abcd.abcd_fraunhofer(self.focal_length)


###################
### Propagators ###
###################
class ABCDPropagator(OpticalLayer):
    """
    Arbitrarily chained ABCD matrices
    """

    ABCDs: dict
    spec: CoordSpec

    def __init__(self, ABCDs, spec):
        self.ABCDs = dlu.list2dictionary(ABCDs, True, allowed_types=(ABCDElement,))
        self.spec = spec

    def __getattr__(self, key):
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
        """Analytic defocused ABCD matrix"""
        return abcd.compose_abcd([m.abcd for m in self.ABCDs.values()])


class MFTPropagator(ABCDPropagator):
    """
    A matrix Fourier transform (MFT) propagator represented by an ABCD matrix.

    Note: Always returns a wavefront in an 'Intermediate' plane, even if the
    propagation is to a conjugate plane. Future ABCDWavefronts may enable better plane
    tracking, but the present Wavefront class is not compatible with this formulation.

    Parameters
    ----------
    ABCDs: list[ABCDElement]
        A list of ABCD elements to compose into the overall ABCD matrix for the
        propagation.
    spec: CoordSpec
        The coordinate specification of the output wavefront.
    """

    def __call__(self, wavefront):
        """
        Propagates the updates the wavefront
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
        return wavefront.set(phasor=field).set_spec(spec_out)


class FFTPropagator(ABCDPropagator):
    """
    FFT-based propagator with optional zero-padding and cropping.

    Note: Always returns a wavefront in an 'Intermediate' plane, even if the
    propagation is to a conjugate plane. Future ABCDWavefronts may enable better plane
    tracking, but the present Wavefront class is not compatible with this formulation.

    pad: int
        The zero-padding factor to apply to the `Wavefront` before propagation. In
        general, this should be greater than 2 to avoid aliasing if the wavefront has
        not already been padded
    crop: int
        The factor by which to crop the output field after propagation. Generally used
        to return the wavefront to the original size after a padded FFT-based
        propagation, and should generally be the same as the original pad factor.
    """

    def __init__(self, ABCDs, spec):
        if isinstance(spec, CoordSpec) and (spec.d is not None):
            raise ValueError("FFTPropagator CoordSpec can not specify d.")
        super().__init__(ABCDs=ABCDs, spec=spec)

    def __call__(self, wavefront):
        """
        Propagates the updates the wavefront
        """

        # Get the input spec
        spec_in = wavefront.spec
        lam = wavefront.wavelength

        # Handle the two spec options
        if isinstance(self.spec, CoordSpec):
            n_padded = self.spec.n
        else:
            n_padded = spec_in.n * self.spec.pad

        # Get the effective focal length for the ABCD system
        fl = abcd.abcd_effective_focal_length(self.abcd)

        # Calculate the phase ramp for the FFT propagation offset
        # NOTE: !! This FFT pixel offset correction may not be valid here !!
        d_fft, c_fft = dlu.fft_spec(n_padded, spec_in.d, lam, fl)
        in_ramp = dlu.fft_phase_ramp(spec_in.xs, lam, c_fft - self.spec.c, fl)

        # # Calculate the output phase ramp correction
        spec_out = CoordSpec(n=n_padded, c=self.spec.c, d=d_fft)
        # shift = dlu.fft_spec(spec_out.n, spec_out.d, lam, fl)[1]
        # out_ramp = dlu.fft_phase_ramp(spec_out.xs, lam, shift, fl)

        # FFT-based LCT propagation
        field, spec_out_xys = lct.lct_prop_fft(
            u_in=wavefront.phasor * in_ramp,
            spec_in=wavefront.spec.xs,
            lam=wavefront.wavelength,
            ABCD=self.abcd,
            # npad=spec_out.n - spec_in.n,
            npad=spec_out.n,  # - spec_in.n,
        )
        # field *= out_ramp

        # Apply the crop if specified
        if isinstance(self.spec, PadSpec):
            n_out = field.shape[0] // self.spec.crop
            field = dlu.crop_to(field, n_out)

        # # Get the output coordinates
        xs_out, ys_out = spec_out_xys
        dx_out = xs_out[1] - xs_out[0]

        # Update wavefront
        return wavefront.set(phasor=field, pixel_scale=dx_out, center=self.spec.c)


class ASMPropagator(OpticalLayer):
    """
    Angular Spectrum Method (ASM) propagator

    Note: Always returns a wavefront in an 'Intermediate' plane, even if the
    propagation is to a conjugate plane. Future ABCDWavefronts may enable better plane
    tracking, but the present Wavefront class is not compatible with this formulation.
    """

    distance: float
    pad: int
    crop: int

    def __init__(self, distance, pad=1, crop=1):
        raise NotImplementedError("ASMPropagator is not yet implemented.")
        self.distance = float(distance)
        self.pad = int(pad)
        self.crop = int(crop)

    def __call__(self, wavefront):

        # Get spec and padding
        # spec_in = (wavefront.npixels, wavefront.pixel_scale)
        # spec_in =
        # pad_to =

        # Propagate the field
        field = asm.asm_prop(
            u_in=wavefront.phasor,
            spec_in=wavefront.spec,
            wavelength=wavefront.wavelength,
            distance=self.distance,
            npad=wavefront.npixels * self.pad,
            crop=False,
        )

        # crop if requested
        if self.crop > 1:
            crop_to = field.shape[0] // self.crop
            field = dlu.crop_to(field, crop_to)

        # Update wavefront
        return wavefront.set(phasor=field, plane="Intermediate")


class Fraunhofer(ABCDPropagator):

    spec_out: tuple[int, float]
    focal_length: float

    def __init__(self):
        raise NotImplementedError("Fraunhofer propagator is not yet implemented.")


class Fresnel(ABCDPropagator):

    spec_out: tuple[int, float]
    focal_length: float
    defocus: float

    def __init__(self):
        raise NotImplementedError("Fresnel propagator is not yet implemented.")


# class CoordSpec(zdx.Base):
#     N: int | tuple[int, ...]  # Num pixels
#     d: float | tuple[float, ...]  # pixel scale
#     s: float | tuple[float, ...]  # offset

#     def __init__(self):
#         raise NotImplementedError("CoordSpec is not yet implemented.")
