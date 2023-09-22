from __future__ import annotations
import jax.numpy as np
from jax import Array


from .optical_layers import OpticalLayer
from ..wavefronts import Wavefront


__all__ = ["MFT", "FFT", "ShiftedMFT", "FarFieldFresnel"]


class Propagator(OpticalLayer):
    """
    Base propagator class, instantiating the focal_length attribute.

    Attributes
    ----------
    focal_length : float
        The effective focal length of the focusing optic this class represents. Note if
        `focal_length` is None, pixel_scales are assumed to be in radians/pixel, else
        meters/pixel.
    """

    focal_length: float

    def __init__(self: Propagator, focal_length: float = None):
        """
        Parameters
        ----------
        focal_length : float = None
            The effective focal length of the focusing optic this class represents. Note
            if `focal_length` is None, pixel_scales are assumed to be in radians/pixel,
            else meters/pixel.
        """
        super().__init__()

        if focal_length is not None:
            focal_length = float(focal_length)

        self.focal_length = focal_length


class FFT(Propagator):
    """
    Propagates a `Wavefront` using the FFT algorithm.

    ??? abstract "UML"
        ![UML](../../assets/uml/FFT.png)

    # TODO: Update padding to take in units of npixels, rather than factor.

    Attributes
    ----------
    focal_length : float
        The focal_length of the lens/mirror this propagator represents. If None, the
        output pixel_scale has units radians/pixel, else meters/pixels.
    pad : int
        The zero-padding factor to apply to the `Wavefront` before propagation. In
        general, this should be greater than 2 to avoid aliasing.
    """

    pad: int

    def __init__(self: Propagator, focal_length: float = None, pad: int = 2):
        """
        Parameters
        ----------
        focal_length : float = None
            The focal_length of the lens/mirror this propagator represents. If None, the
            output pixel_scale has units radians/pixel, else meters/pixels.
        pad : int = 2
            The zero-padding factor to apply to the `Wavefront` before propagation. In
            general, this should be greater than 2 to avoid aliasing.
        """
        super().__init__(focal_length=focal_length)
        self.pad = int(pad)

    def apply(self: Propagator, wavefront: Wavefront) -> Wavefront:
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
        return wavefront.propagate_FFT(self.focal_length, self.pad)


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
    """

    npixels: int
    pixel_scale: float

    def __init__(
        self: Propagator,
        npixels: int,
        pixel_scale: float,
        focal_length: float = None,
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
        """
        super().__init__(focal_length=focal_length)

        self.pixel_scale = float(pixel_scale)
        self.npixels = int(npixels)

    def apply(self: Propagator, wavefront: Wavefront) -> Wavefront:
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
            self.npixels,
            self.pixel_scale,
            self.focal_length,
        )


class ShiftedMFT(MFT):
    """
    Propagates a `Wavefront` using the MFT algorithm, allowing for the pixel_scale and
    number of pixels to be specified in the output plane. Also optionally allows for a
    shift to be applied to the wavefront in the output plane.

    ??? abstract "UML"
        ![UML](../../assets/uml/ShiftedMFT.png)

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
    shift : Array
        The (x, y) shift to apply to the wavefront in the output plane.
    pixel : bool
        If True the shift value is assumed to be in units of pixels, else the physical
        units of the output plane (ie radians if focal_length is None, else meters).
    """

    shift: Array
    pixel: bool

    def __init__(
        self: Propagator,
        npixels: int,
        pixel_scale: float,
        shift: Array,
        focal_length: float = None,
        pixel: bool = False,
    ):
        """
        Parameters
        ----------
        npixels : int
            The number of pixels in the output plane.
        pixel_scale : float, meters/pixel or radians/pixel
            The pixel scale in the output plane. Has units of radians/pixel if
            focal_length is None, else meters/pixel.
        shift : Array
            The (x, y) shift to apply to the wavefront in the output plane.
        focal_length : float = None, meters
            The focal_length of the lens/mirror this propagator represents. If None,
            the output pixel_scale has units radians/pixel, else meters/pixels.
        pixel : bool = False
            If True the shift value is assumed to be in units of pixels, else the
            physical units of the output plane (ie radians if focal_length is None,
            else meters).
        """
        super().__init__(
            pixel_scale=pixel_scale,
            npixels=npixels,
            focal_length=focal_length,
        )

        self.shift = np.asarray(shift, float)
        self.pixel = bool(pixel)

        if self.shift.shape != (2,):
            raise ValueError(
                f"Shift must be a 2D array, got {self.shift.shape}."
            )

    def apply(self: Propagator, wavefront: Wavefront) -> Wavefront:
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
            self.npixels,
            self.pixel_scale,
            self.focal_length,
            self.shift,
            self.pixel,
        )


class FarFieldFresnel(ShiftedMFT):
    """
    Propagates a `Wavefront` using the MFT algorithm, modified to allows for more
    accurate representations of wavefront behaviour in the far-field regime, a few
    wavelengths away from focus in either direction. Allows for the pixel_scale and
    number of pixels to be specified in the output plane, and optionally allows for a
    shift to be applied to the wavefront in the output plane.

    ??? abstract "UML"
        ![UML](../../assets/uml/FarFieldFresnel.png)

    Attributes
    ----------
    npixels : int
        The number of pixels in the output plane.
    pixel_scale : float, meters/
        The pixel scale in the output plane.
    focal_length : float, meters
        The focal_length of the lens/mirror this propagator represents.
    shift : Array
        The (x, y) shift to apply to the wavefront in the output plane.
    pixel : bool
        If True the shift value is assumed to be in units of pixels, else the physical
        units of the output plane.
    focal_shift : float, meters
        The shift in the propagation distance of the wavefront from focus.
    """

    focal_shift: float

    def __init__(
        self: Propagator,
        npixels: int,
        pixel_scale: float,
        focal_length: float,
        focal_shift: float,
        shift: Array = np.zeros(2),
        pixel: bool = False,
    ):
        """
        Parameters
        ----------
        npixels : int
            The number of pixels in the output plane.
        pixel_scale : float, meters/
            The pixel scale in the output plane.
        focal_length : float, meters
            The focal_length of the lens/mirror this propagator represents.
        focal_shift : float, meters
            The shift in the propagation distance of the wavefront from focus.
        shift : Array = np.zeros(2)
            The (x, y) shift to apply to the wavefront in the output plane.
        pixel : bool = False
            If True the shift value is assumed to be in units of pixels, else the
            physical units of the output plane.
        """
        self.focal_shift = float(focal_shift)
        super().__init__(
            shift=shift,
            pixel=pixel,
            focal_length=focal_length,
            pixel_scale=pixel_scale,
            npixels=npixels,
        )

    def apply(self: Propagator, wavefront: Wavefront) -> Wavefront:
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
        return wavefront.propagate_fresnel(
            self.npixels,
            self.pixel_scale,
            self.focal_length,
            self.focal_shift,
            self.shift,
            self.pixel,
        )
