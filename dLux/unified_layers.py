from __future__ import annotations
from typing import Union
import dLux

__all__ = [
    "Rotate",
    "Flip",
    "Resize",
]

OpticalLayer = lambda: dLux.optical_layers.OpticalLayer
DetectorLayer = lambda: dLux.detector_layers.DetectorLayer
Wavefront = lambda: dLux.wavefronts.Wavefront
PSF = lambda: dLux.psfs.PSF


class UnifiedLayer(OpticalLayer(), DetectorLayer()):
    """
    Base class for unified layers that can be applied to either wavefronts or
    PSFs.
    """


class Resize(UnifiedLayer):
    """
    Resizes either a wavefront or PSF by either padding or cropping.

    Note this class only supports padding and cropping of even sizes to even
    sizes, and odd sizes to odd sizes to ensure all operations are paraxial.

    Attributes
    ----------
    npixels : int
        The desired output size.
    """

    npixels: int

    def __init__(self: UnifiedLayer, npixels: int):
        """
        Constructor for the Resize class.

        Parameters
        ----------
        npixels : tuple
            The desired output size.
        """
        super().__init__()
        self.npixels = int(npixels)

    def __call__(
        self: UnifiedLayer, input: Union[Wavefront(), PSF()]
    ) -> Union[Wavefront(), PSF()]:
        """
        Resizes the input.

        Parameters
        ----------
        input : Union[Wavefront, PSF]
            The input to resize.

        Returns
        -------
        input : Union[Wavefront, PSF]
            The resized input.
        """
        return input.resize(self.npixels)


class Rotate(UnifiedLayer):
    """
    Rotates either a wavefront or PSF by a given angle. This is done using
    interpolation methods. The 'complex' input only has an effect if the input
    is a wavefront.

    Attributes
    ----------
    angle : Array, radians
        The angle by which to rotate the input in the clockwise direction.
    order : int = 1
        The order of the interpolation to use. Must be 0 or 1.
    complex : bool = False
        Should the rotation be performed on the 'complex' (real, imaginary),
        as opposed to the default 'phasor' (amplitude, phase) arrays. Only
        applies if the input is a wavefront.
    """

    angle: float
    order: int
    complex: bool

    def __init__(
        self: UnifiedLayer,
        angle: float,
        order: int = 1,
        complex: bool = False,
    ):
        """
        Constructor for the Rotate class.

        Parameters
        ----------
        angle: float, radians
            The angle by which to rotate the input in the clockwise direction.
        order : int = 1
            The order of the interpolation to use. Must be 0, or 1.
        complex : bool = False
            Should the rotation be performed on the 'complex' (real,
            imaginary), as opposed to the default 'phasor' (amplitude, phase)
            arrays. Only applies if the input is a wavefront.
        """
        super().__init__()
        self.angle = float(angle)
        self.order = int(order)
        self.complex = bool(complex)

        if self.order not in (0, 1):
            raise ValueError("Order must be 0, 1")

    def __call__(
        self: UnifiedLayer, input: Union[Wavefront(), PSF()]
    ) -> Union[Wavefront(), PSF()]:
        """
        Applies the rotation to the input.

        Parameters
        ----------
        input : Union[Wavefront, PSF]
            The input to rotate.

        Returns
        -------
        input : Union[Wavefront, PSF]
            The rotated input.
        """
        if isinstance(input, PSF):
            return input.rotate(self.angle, self.order)
        else:
            return input.rotate(self.angle, self.order, self.complex)


class Flip(UnifiedLayer):
    """
    Flips either a wavefront or PSF about the input axes. Can be either an int,
    or a tuple of ints. This class uses the 'ij' indexing convention, ie axis 0
    is the y-axis, and axis 1 is the x-axis.

    Attributes
    ----------
    axes : Union[tuple, int]
        The axes to flip the input about. This class uses the 'ij' indexing
        convention, ie axis 0 is the y-axis, and axis 1 is the x-axis.
    """

    axes: Union[tuple[int], int]

    def __init__(self: UnifiedLayer, axes: Union[tuple[int], int]):
        """
        Constructor for the Flip class.

        Parameters
        ----------
        axes : Union[tuple[int], int]
            The axes to flip the input about. This class uses the 'ij'
            indexing convention, ie axis 0 is the y-axis, and axis 1 is the
            x-axis.
        """
        super().__init__()
        self.axes = axes

        if isinstance(self.axes, tuple):
            for axes in self.axes:
                if not isinstance(axes, int):
                    raise ValueError("All axes must be integers.")
        elif not isinstance(self.axes, int):
            raise ValueError("axes must be integers.")

    def __call__(
        self: UnifiedLayer, input: Union[Wavefront(), PSF()]
    ) -> Union[Wavefront(), PSF()]:
        """
        Flips the input about the input axes.

        Parameters
        ----------
        input : Union[Wavefront, PSF]
            The input to flip.

        Returns
        -------
        input : Union[Wavefront, PSF]
            The flipped input.
        """
        return input.flip(self.axes)
