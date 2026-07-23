"""Propagation layers and ABCD-system elements."""

from __future__ import annotations
import jax.numpy as np
import zodiax as zdx
from abcdLux import abcd, lct, asm
import dLux.utils as dlu

from ..coord_specs import CoordSpec, PadSpec
from .unified_layers import BaseOpticalLayer

__all__ = [
    "BaseABCDElement",
    "PropagatorLayer",
    "Propagator",
    "FFT",
    "MFT",
    "ABCDFreeSpace",
    "ABCDLens",
    "ABCDMirror",
    "ABCDConjugatePlane",
    "ABCDPropagator",
    "MFTPropagator",
    "FFTPropagator",
    "ASMPropagator",
]


class BaseABCDElement(zdx.Base):
    """Base class for elements represented by an ABCD matrix."""


class PropagatorLayer(BaseOpticalLayer):
    """Optionally propagate a wavefront with another optical layer."""

    propagator: BaseOpticalLayer | None

    def __init__(self, propagator=None):
        if propagator is not None and not isinstance(propagator, BaseOpticalLayer):
            raise TypeError("propagator must be a BaseOpticalLayer or None.")
        self.propagator = propagator

    def __call__(self, wavefront):
        return wavefront if self.propagator is None else self.propagator(wavefront)


class Propagator(BaseOpticalLayer):
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


class ABCDFreeSpace(BaseABCDElement):
    """
    A free space propagation element represented by an ABCD matrix.

    ??? abstract "UML"
        ![UML](../assets/uml/ABCDFreeSpace.png)

    Attributes
    ----------
    distance : float
        The free-space propagation distance.
    abcd : Array, property
        The analytic ABCD matrix for free-space propagation.
    """

    distance: float

    def __init__(self, distance):
        """
        Parameters
        ----------
        distance : float
            The free-space propagation distance.
        """
        self.distance = np.array(distance, float)

    @property
    def abcd(self):
        """
        Returns
        -------
        matrix : Array
            The analytic ABCD matrix for free-space propagation.
        """
        return abcd.abcd_free_space(self.distance)


class ABCDLens(BaseABCDElement):
    """
    A lens element represented by an ABCD matrix. Note this element alone does not
    produce the standard 'pupil-focal Fourier relationship' as that is between the front
    and back focal planes. To represent a true pupil-focal plane propagation, apply a
    free space propagation of the focal length both before _and_ after the lens, or use
    the ABCDConjugatePlane element.

    ??? abstract "UML"
        ![UML](../assets/uml/ABCDLens.png)

    Attributes
    ----------
    focal_length : float
        The lens focal length.
    abcd : Array, property
        The analytic ABCD matrix for the lens.
    """

    focal_length: float

    def __init__(self, focal_length):
        """
        Parameters
        ----------
        focal_length : float
            The lens focal length.
        """
        self.focal_length = np.array(focal_length, float)

    @property
    def abcd(self):
        """
        Returns
        -------
        matrix : Array
            The analytic ABCD matrix for a lens.
        """
        return abcd.abcd_lens(self.focal_length)


class ABCDMirror(BaseABCDElement):
    """
    A mirror element represented by an ABCD matrix.

    ??? abstract "UML"
        ![UML](../assets/uml/ABCDMirror.png)

    Attributes
    ----------
    radius : float
        The mirror radius of curvature.
    abcd : Array, property
        The analytic ABCD matrix for the mirror.
    """

    radius: float

    def __init__(self, radius):
        """
        Parameters
        ----------
        radius : float
            The mirror radius of curvature.
        """
        self.radius = np.array(radius, float)

    @property
    def abcd(self):
        """
        Returns
        -------
        matrix : Array
            The analytic ABCD matrix for a mirror.
        """
        return abcd.abcd_mirror(self.radius)


class ABCDConjugatePlane(BaseABCDElement):
    """
    A conjugate plane element represented by an ABCD matrix. This produces the classic
    'pupil-focal Fourier relationship' seen in fourier/physical optics.

    ??? abstract "UML"
        ![UML](../assets/uml/ABCDConjugatePlane.png)

    Attributes
    ----------
    focal_length : float
        The effective focal length of the conjugate-plane transform.
    abcd : Array, property
        The analytic ABCD matrix for conjugate-plane propagation.
    """

    focal_length: float

    def __init__(self, focal_length):
        """
        Parameters
        ----------
        focal_length : float
            The effective focal length that defines the conjugate-plane transform.
        """
        self.focal_length = np.array(focal_length, float)

    @property
    def abcd(self):
        """
        Returns
        -------
        matrix : Array
            The analytic ABCD matrix for a conjugate-plane propagation.
        """
        return abcd.abcd_fraunhofer(self.focal_length)


###################
### Propagators ###
###################
class ABCDPropagator(BaseOpticalLayer):
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
        return wavefront.set(phasor=field).set_spec(spec_out)


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
            n_padded = spec_in.n * self.spec.pad

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
            spec_in=wavefront.spec.xs,
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

        # Update wavefront
        return wavefront.set(phasor=field, pixel_scale=dx_out, center=self.spec.c)


class ASMPropagator(BaseOpticalLayer):
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

        # Update wavefront
        return wavefront.set(phasor=field)


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
