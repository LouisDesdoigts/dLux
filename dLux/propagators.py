from __future__ import annotations
import jax.numpy as np
from jax import Array
import dLux.utils as dlu
import dLux


__all__ = ["MFT", "FFT", "ShiftedMFT", "FarFieldFresnel"]


Wavefront = dLux.wavefronts.Wavefront


class Propagator(dLux.optical_layers.OpticalLayer):
    """
    An abstract class to store the various properties of the propagation of
    some wavefront.

    Attributes
    ----------
    focal_length : float, metres
        The effective focal length of the lens/mirror this propagator
        represents. If None, the output pixel_scales are taken to be
        radians/pixel, else they are taken to be in metres/pixel.
    inverse : bool
        Should the propagation be performed in the inverse direction.
    """

    focal_length: float
    inverse: bool

    def __init__(
        self: Propagator, focal_length: float = None, inverse: bool = False
    ):
        """
        Constructor for the Propagator.

        Parameters
        ----------
        inverse : bool = False
            Should the propagation be performed in the inverse direction.
        """
        super().__init__()

        if focal_length is not None:
            focal_length = float(focal_length)

        self.focal_length = focal_length
        self.inverse = bool(inverse)


class FFT(Propagator):
    """
    A Propagator class designed to propagate a wavefront to a plane using a
    Fast Fourier Transform.

    # TODO: Update padding to take in units of npixels, rather than factor.

    Attributes
    ----------
    focal_length : float, metres
        The focal_length of the lens/mirror this propagator represents.
    pad : int
        The amount of padding to apply to the wavefront before propagating.
    inverse : bool
        Should the propagation be performed in the inverse direction.
    """

    pad: int

    def __init__(
        self: Propagator,
        focal_length: float = None,
        pad: int = 2,
        inverse: bool = False,
    ) -> Propagator:
        super().__init__(focal_length=focal_length, inverse=inverse)
        self.pad = int(pad)

    def __call__(self: Propagator, wavefront: Wavefront) -> Wavefront:
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
        if self.inverse:
            return wavefront.IFFT(self.pad, self.focal_length)
        else:
            return wavefront.FFT(self.pad, self.focal_length)


class MFT(Propagator):
    """
    A Propagator class designed to propagate a wavefront to a plane that is
    defined in Cartesian units (ie metres/pixel).

    Attributes
    ----------
    npixels : int
        The number of pixels in the output plane.
    pixel_scale : float, metres/pixel or radians/pixel
        The pixel scale in the output plane, measured in metres per pixel.
    oversample : float
        The amount of oversampling in the output plane.
    unit : str
        The output unit of the propagation. If inverse is False or focal_length
        is not None it is automatically set to 'meters', otherwise it can be
        either 'arcseconds' or 'radians'.
    focal_length : float, metres
        The effective focal length of the lens/mirror this propagator
        represents. If None, the pixel_scale is taken to be in radians/pixel,
        else it is taken to be in metres/pixel.
    inverse : bool
        Should the propagation be performed in the inverse direction.
    """

    npixels: int
    oversample: float
    pixel_scale: float
    unit: str

    def __init__(
        self: Propagator,
        npixels: int,
        pixel_scale: float,
        oversample: float = 1.0,
        unit: str = "radians",
        focal_length: float = None,
        inverse: bool = False,
    ):
        """
        Constructor for VariableSampling propagators.

        Parameters
        ----------
        npixels : int
            The number of pixels in the output plane.
        pixel_scale : float, metres/pixel or radians/pixel
            The pixel scale in the output plane, measured in radians per pixel
            if focal_length is None, else metres per pixel.
        oversample : float = 1.
            The amount of oversampling in the output plane.
        unit : str = 'radians'
            The output unit of the propagation. If inverse is False or
            focal_length is not None it is automatically set to 'meters',
            otherwise it can be either 'arcseconds' or 'radians'.
        focal_length : float = None, metres
            The focal_length of the lens/mirror this propagator represents.
            If None, the pixel_scale is taken to be in radians/pixel, else it
            is taken to be in metres/pixel.
        inverse : bool = False
            Should the propagation be performed in the inverse direction.
        """
        super().__init__(focal_length=focal_length, inverse=inverse)

        if self.inverse or self.focal_length is not None:
            self.unit = "meters"
        elif unit not in ["arcseconds", "radians"]:
            raise ValueError(
                f"Unit must be one of 'arcseconds' or 'radians', got {unit}."
            )

        self.unit = unit
        self.oversample = float(oversample)
        self.pixel_scale = float(pixel_scale)
        self.npixels = int(npixels)

    def __call__(self: Propagator, wavefront: Wavefront) -> Wavefront:
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
        if self.inverse:
            return wavefront.IMFT(
                self.npixels,
                self.pixel_scale / self.oversample,
                focal_length=self.focal_length,
            )
        else:
            pixel_scale = self.pixel_scale / self.oversample
            if self.unit == "arcseconds":
                pixel_scale = dlu.arcsec_to_rad(pixel_scale)
            return wavefront.MFT(
                self.npixels,
                pixel_scale,
                focal_length=self.focal_length,
            )


class ShiftedMFT(MFT):
    """
    A Propagator class designed to propagate a wavefront to a plane that is
    defined in Cartesian units (ie metres/pixel), with a variable output
    sampling in that plane.

    Attributes
    ----------
    npixels : int
        The number of pixels in the output plane.
    pixel_scale : float, metres/pixel or radians/pixel
        The pixel scale in the output plane, measured in metres per pixel.
    oversample : float
        The amount of oversampling in the output plane.
    unit : str
        The output unit of the propagation. If inverse is False or focal_length
        is not None it is automatically set to 'meters', otherwise it can be
        either 'arcseconds' or 'radians'.
    focal_length : float, metres
        The effective focal length of the lens/mirror this propagator
        represents. If None, the pixel_scale is taken to be in radians/pixel,
        else it is taken to be in metres/pixel.
    shift : Array
        The (x, y) shift to apply to the wavefront in the output plane.
    pixel : bool
        If True the shift value is assumed to be in units of pixels, else the
        physical units of the output plane (ie radians if focal_length is None,
        else metres).
    inverse : bool
        Should the propagation be performed in the inverse direction.
    """

    shift: Array
    pixel: bool

    def __init__(
        self: Propagator,
        npixels: int,
        pixel_scale: float,
        shift: Array,
        oversample: float = 1.0,
        unit: str = "radians",
        focal_length: float = None,
        pixel: bool = False,
        inverse: bool = False,
    ):
        """
        Constructor for VariableSampling propagators.

        Parameters
        ----------
        npixels : int
            The number of pixels in the output plane.
        pixel_scale : float, metres/pixel or radians/pixel
            The pixel scale in the output plane, measured in metres or radians
            per pixel for Cartesian or Angular Wavefront respectively.
        shift : Array
            The (x, y) shift to apply to the wavefront in the output plane.
        oversample : float = 1.
            The amount of oversampling in the output plane.
        unit : str = 'radians'
            The output unit of the propagation. If inverse is False or
            focal_length is not None it is automatically set to 'meters',
            otherwise it can be either 'arcseconds' or 'radians'.
        focal_length : float = None, metres
            The effective focal_length of the lens/mirror this propagator
            represents. If None, the pixel_scale is taken to be in
            radians/pixel, else it is taken to be in metres/pixel.
        pixel : bool = False
            If True the shift value is assumed to be in units of pixels, else
            the physical units of the output plane (ie radians if focal_length
            is None, else metres).
        inverse : bool = False
            Should the propagation be performed in the inverse direction.
        """
        super().__init__(
            pixel_scale=pixel_scale,
            npixels=npixels,
            oversample=oversample,
            focal_length=focal_length,
            inverse=inverse,
            unit=unit,
        )

        self.shift = np.asarray(shift, float)
        self.pixel = bool(pixel)

        if shift.shape != (2,):
            raise TypeError(f"Shift must be a 2D array, got {shift.shape}.")

    def __call__(self: Propagator, wavefront: Wavefront) -> Wavefront:
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
        if self.inverse:
            return wavefront.shifted_IMFT(
                self.npixels,
                self.pixel_scale / self.oversample,
                self.shift,
                self.focal_length,
                self.pixel,
            )
        else:
            pixel_scale = self.pixel_scale / self.oversample
            if self.unit == "arcseconds":
                pixel_scale = dlu.arcsec_to_rad(pixel_scale)

            return wavefront.shifted_MFT(
                self.npixels,
                pixel_scale,
                self.shift,
                self.focal_length,
                self.pixel,
            )


class FarFieldFresnel(ShiftedMFT):
    """
    A propagator class to for Far-Field fresnel propagations. This classes
    implements algorithms that use quadratic phase factors to better represent
    out-of-plane behaviour of wavefronts, close to the focal plane. This class
    is designed to work on Cartesian wavefronts, i.e. pixel units are in
    metres/pixel in the output plane.

    Attributes
    ----------
    npixels : int
        The number of pixels in the output plane.
    pixel_scale : float, metres/pixel
        The pixel scale in the output plane, measured in metres per pixel.
    focal_length : float, metres
        The focal_length of the lens/mirror this propagator represents.
    focal_shift : float, metres
        The shift in the propagation distance of the wavefront.
    oversample : float
        The amount of oversampling in the output plane.
    unit : str
        The output unit of the propagation. If inverse is False or focal_length
        is not None it is automatically set to 'meters', otherwise it can be
        either 'arcseconds' or 'radians'.
    shift : Array
        The (x, y) shift to apply to the wavefront in the output plane.
    pixel : bool
        Should the shift value be considered in units of pixels, or in the
        physical units of the output plane (ie pixels or metres, radians). True
        interprets the shift value in pixel units.
    inverse : bool
        Should the propagation be performed in the inverse direction.
    """

    focal_shift: float

    def __init__(
        self: Propagator,
        npixels: int,
        pixel_scale: float,
        focal_length: float,
        focal_shift: float,
        oversample: float = 1.0,
        unit: str = "radians",
        shift: Array = np.zeros(2),
        pixel: bool = False,
        inverse: bool = False,
    ):
        """
        Constructor for the CartesianFresnel propagator

        Parameters
        ----------
        npixels : int
            The number of pixels in the output plane.
        pixel_scale : float, metres/pixel
            The pixel scale in the output plane, measured in metres per pixel.
        focal_length : float, metres
            The focal_length of the lens/mirror this propagator represents.
        focal_shift : float, metres
            The shift in the propagation distance of the wavefront.
        oversample : float = 1.
            The amount of oversampling in the output plane.
        unit : str = 'radians'
            The output unit of the propagation. If inverse is False or
            focal_length is not None it is automatically set to 'meters',
            otherwise it can be either 'arcseconds' or 'radians'.
        shift : Array = np.array([0., 0.])
            The (x, y) shift to apply to the wavefront in the output plane.
        pixel : bool = False
            Should the shift value be considered in units of pixel, or in the
            physical units of the output plane (ie pixels or metres, radians).
        inverse : bool = False
            Should the propagation be performed in the inverse direction.
        """
        if inverse:
            raise NotImplementedError(
                "Inverse propagation not implemented " "for CartesianFresnel."
            )

        self.focal_shift = float(focal_shift)

        super().__init__(
            shift=shift,
            pixel=pixel,
            focal_length=focal_length,
            pixel_scale=pixel_scale,
            oversample=oversample,
            npixels=npixels,
            inverse=inverse,
            unit=unit,
        )

    def __call__(self: Propagator, wavefront: Wavefront) -> Wavefront:
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
        pixel_scale = self.pixel_scale / self.oversample
        if self.unit == "arcseconds":
            pixel_scale = dlu.arcsec_to_rad(pixel_scale)
        return wavefront.shifted_fresnel_prop(
            self.npixels,
            pixel_scale,
            self.shift,
            self.focal_length,
            self.focal_shift,
            self.pixel,
        )
