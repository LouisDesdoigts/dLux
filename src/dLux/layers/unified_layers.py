"""Unified layers that operate on either wavefronts or PSFs."""

from __future__ import annotations

__all__ = ["UnifiedLayer", "Rotate", "Flip", "Resize"]


from .optical_layers import OpticalLayer
from .detector_layers import DetectorLayer
from ..wavefronts import Wavefront
from ..psfs import PSF


class UnifiedLayer(OpticalLayer, DetectorLayer):
    """
    Base class for unified layers that can be applied to either wavefronts or PSFs.
    """


class Resize(UnifiedLayer):
    """
    Resizes either a wavefront or PSF by either padding or cropping. Note this class
    only supports padding and cropping of even sizes to even sizes, and odd sizes to
    odd sizes to ensure all operations are paraxial.

    ??? abstract "UML"
        ![UML](../../assets/uml/Resize.png)

    Attributes
    ----------
    npixels : int
        The desired output size.
    """

    npixels: int

    def __init__(self: Resize, npixels: int):
        """
        Parameters
        ----------
        npixels : int
            The desired output size.
        """
        super().__init__()
        self.npixels = int(npixels)

    def apply(self: Resize, target: Wavefront | PSF) -> Wavefront | PSF:
        """
        Resizes the input.

        Parameters
        ----------
        target : Wavefront | PSF
            The input to resize.

        Returns
        -------
        target : Wavefront | PSF
            The resized input.
        """
        return target.resize(self.npixels)


class Rotate(UnifiedLayer):
    """
    Rotates either a wavefront or PSF by a given angle. This is done using
    interpolation methods. The 'complex' input only has an effect if the input is a
    wavefront.

    ??? abstract "UML"
        ![UML](../../assets/uml/Rotate.png)

    Attributes
    ----------
    angle : float, radians
        The angle by which to rotate the input in the clockwise direction.
    method : str
        The interpolation method.
    complex : bool
        Should the rotation be performed on the 'complex' (real, imaginary), as opposed
        to the default 'phasor' (amplitude, phase) arrays. Only applies if the input is
        a wavefront.
    """

    angle: float
    method: str
    complex: bool

    def __init__(
        self: Rotate,
        angle: float,
        method: str = "linear",
        complex: bool = False,
    ):
        """
        Parameters
        ----------
        angle : float, radians
            The angle by which to rotate the input in the clockwise direction.
        method : str = "linear"
            The interpolation method.
        complex : bool = False
            Should the rotation be performed on the 'complex' (real, imaginary), as
            opposed to the default 'phasor' (amplitude, phase) arrays. Only applies if
            the input is a wavefront.
        """
        super().__init__()
        self.angle = float(angle)
        self.method = str(method)
        self.complex = bool(complex)

    def apply(self: Rotate, target: Wavefront | PSF) -> Wavefront | PSF:
        """
        Applies the rotation to the input.

        Parameters
        ----------
        target : Wavefront | PSF
            The input to rotate.

        Returns
        -------
        target : Wavefront | PSF
            The rotated input.
        """
        if isinstance(target, PSF):
            return target.rotate(self.angle, self.method)
        return target.rotate(self.angle, self.method, self.complex)


class Flip(UnifiedLayer):
    """
    Flips either a wavefront or PSF about the input axes. Can be either an int or a
    tuple of ints. This class uses the 'ij' indexing convention, ie axis 0 is the
    y-axis, and axis 1 is the x-axis.

    ??? abstract "UML"
        ![UML](../../assets/uml/Flip.png)

    Attributes
    ----------
    axes : tuple | int
        The axes to flip the input about. This class uses the 'ij' indexing convention,
        ie axis 0 is the y-axis, and axis 1 is the x-axis.
    """

    axes: tuple[int] | int

    def __init__(self: Flip, axes: tuple[int] | int):
        """
        Parameters
        ----------
        axes : tuple | int
            The axes to flip the input about. This class uses the 'ij' indexing
            convention, ie axis 0 is the y-axis, and axis 1 is the x-axis.
        """
        super().__init__()
        self.axes = axes

        if isinstance(self.axes, tuple):
            for axis in self.axes:
                if not isinstance(axis, int):
                    raise ValueError("All axes must be integers.")
        elif not isinstance(self.axes, int):
            raise ValueError("axes must be an int or tuple of ints.")

    def apply(self: Flip, target: Wavefront | PSF) -> Wavefront | PSF:
        """
        Flips the input about the input axes.

        Parameters
        ----------
        target : Wavefront | PSF
            The input to flip.

        Returns
        -------
        target : Wavefront | PSF
            The flipped input.
        """
        return target.flip(self.axes)
