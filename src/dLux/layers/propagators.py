"""Optical-layer propagators for FFT, MFT, and far-field Fresnel propagation."""

from __future__ import annotations

from .optical_layers import OpticalLayer
from ..wavefronts import Wavefront
from ..coordinates import CoordSpec

__all__ = ["MFT", "FFT"]


class Propagator(OpticalLayer):
    """
    Base propagator class, instantiating the focal_length attribute.

    ??? abstract "UML"
        ![UML](../../assets/uml/Propagator.png)

    Attributes
    ----------
    focal_length : float
        The effective focal length of the focusing optic this class represents. Note if
        `focal_length` is None, pixel_scales are assumed to be in radians/pixel, else
        meters/pixel.
    inverse: bool
        If False, propagate forward through the system. If True, propagate backward
        through the system.
    """

    focal_length: float | None
    inverse: bool

    def __init__(
        self: Propagator, focal_length: float | None = None, inverse: bool = False
    ):
        """
        Parameters
        ----------
        focal_length : float = None
            The effective focal length of the focusing optic this class represents. Note
            if `focal_length` is None, pixel_scales are assumed to be in radians/pixel,
            else meters/pixel.
        inverse : bool = False
            If False, propagate forward through the system. If True, propagate
            backward through the system.
        """
        super().__init__()

        if focal_length is not None:
            focal_length = float(focal_length)
        self.focal_length = focal_length
        self.inverse = bool(inverse)


class FFT(Propagator):
    """
    Propagates a `Wavefront` using the FFT algorithm.

    ??? abstract "UML"
        ![UML](../../assets/uml/FFT.png)

    Attributes
    ----------
    focal_length : float
        The focal_length of the lens/mirror this propagator represents. If None, the
        output pixel_scale has units radians/pixel, else meters/pixels.
    inverse: bool
        If False, propagate forward through the system. If True, propagate backward
        through the system.
    pad : int
        The zero-padding factor to apply to the `Wavefront` before propagation. In
        general, this should be greater than 2 to avoid aliasing.
    crop : int
        The cropping factor to apply to the `Wavefront` after propagation. In general,
        this should only be applied after a corresponding padding factor has been
        applied to avoid aliasing.
    """

    pad: int
    crop: int
    center: bool

    def __init__(
        self: FFT,
        focal_length: float = None,
        inverse: bool = False,
        pad: int = 1,
        crop: int = 1,
        center: bool = True,
    ):
        """
        Parameters
        ----------
        focal_length : float = None
            The focal_length of the lens/mirror this propagator represents. If None, the
            output pixel_scale has units radians/pixel, else meters/pixels.
        inverse: bool = False
            If False, propagate forward through the system. If True, propagate
            backward through the system.
        pad : int = 1
            The zero-padding factor to apply to the `Wavefront` before propagation. In
            general, this should be greater than 2 to avoid aliasing.
        crop : int = 1
            The cropping factor to apply to the `Wavefront` after propagation. In
            general, this should only be applied after a corresponding padding factor
            has been applied to avoid aliasing.
        center : bool = True
            If True, the output coordinates are centered at 0. If False, output
            coordinates use the default transform convention.
        """
        super().__init__(focal_length=focal_length, inverse=inverse)
        self.pad = int(pad)
        self.crop = int(crop)
        self.center = bool(center)

    def __call__(self: FFT, wavefront: Wavefront) -> Wavefront:
        """
        Applies the layer to the wavefront.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to operate on.

        Returns
        -------
        wavefront : Wavefront
            The transformed wavefront.
        """
        if self.center:
            spec = CoordSpec(c=0.0)
        else:
            spec = None
        size_out = wavefront.npixels * self.pad // self.crop
        return wavefront.propagate_FFT(
            pad=self.pad,
            focal_length=self.focal_length,
            inverse=self.inverse,
            spec_out=spec,
        ).resize(size_out)


class MFT(Propagator):
    """
    Propagates a `Wavefront` using the MFT algorithm, allowing for the pixel_scale and
    number of pixels to be specified in the output plane.

    ??? abstract "UML"
        ![UML](../../assets/uml/MFT.png)

    Attributes
    ----------
    npixels : int
        The number of pixels in the output plane.
    pixel_scale : float, meters/pixel or radians/pixel
        The pixel scale in the output plane. Has units of radians/pixel if focal_length
        is None, else meters/pixel.
    focal_length : float, meters
        The focal_length of the lens/mirror this propagator represents. If None, the
        output pixel_scale has units radians/pixel, else meters/pixels.
    inverse: bool
        If False, propagate forward through the system. If True, propagate backward
        through the system.
    """

    npixels: int
    pixel_scale: float

    def __init__(
        self: MFT,
        npixels: int,
        pixel_scale: float,
        focal_length: float = None,
        inverse: bool = False,
    ):
        """
        Parameters
        ----------
        npixels : int
            The number of pixels in the output plane.
        pixel_scale : float, meters/pixel or radians/pixel
            The pixel scale in the output plane. Has units of radians/pixel if
            focal_length is None, else meters/pixel.
        focal_length : float = None, meters
            The focal_length of the lens/mirror this propagator represents. If None,
            the output pixel_scale has units radians/pixel, else meters/pixels.
        inverse: bool = False
            If False, propagate forward through the system. If True, propagate
            backward through the system.
        """
        super().__init__(focal_length=focal_length, inverse=inverse)

        self.pixel_scale = float(pixel_scale)
        self.npixels = int(npixels)

    def __call__(self: MFT, wavefront: Wavefront) -> Wavefront:
        """
        Applies the layer to the wavefront.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to operate on.

        Returns
        -------
        wavefront : Wavefront
            The transformed wavefront.
        """
        return wavefront.propagate(
            npixels=self.npixels,
            pixel_scale=self.pixel_scale,
            focal_length=self.focal_length,
            inverse=self.inverse,
        )
