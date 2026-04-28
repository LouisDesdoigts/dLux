import zodiax as zdx
import jax.numpy as np

__all__ = ["Spec", "PadSpec", "CoordSpec"]


class Spec(zdx.Base):
    """
    Abstract base class for coordinate/sampling specifications.

    ??? abstract "UML"
        ![UML](../../assets/uml/Spec.png)
    """

    pass


class PadSpec(Spec):
    """
    Coordinate specification defined via integer padding and cropping factors
    relative to an input grid size.

    ??? abstract "UML"
        ![UML](../../assets/uml/PadSpec.png)

    Attributes
    ----------
    pad : int
        Factor by which to increase the grid size. The padded grid will have
        ``n * pad`` pixels along each axis.
    crop : int
        Factor by which to reduce the grid size after processing. The cropped
        grid will have ``n_out // crop`` pixels along each axis.
    c : float
        Centre coordinate of the grid, in metres.
    """

    pad: int
    crop: int
    c: float

    def __init__(self, pad=1, crop=1, c=0.0):
        """
        Parameters
        ----------
        pad : int = 1
            Grid size increase factor.
        crop : int = 1
            Grid size reduction factor applied after processing.
        c : float = 0.0
            Centre coordinate of the grid, in metres.
        """
        self.pad = int(pad)
        self.crop = int(crop)
        self.c = c


class CoordSpec(Spec):
    """
    Coordinate specification defined explicitly by number of pixels, pixel
    scale, and centre offset.

    ??? abstract "UML"
        ![UML](../../assets/uml/CoordSpec.png)

    Attributes
    ----------
    n : int
        Number of pixels along each axis.
    d : float
        Pixel scale (spacing between adjacent pixels), in metres.
    c : float
        Centre coordinate of the grid, in metres.
    """

    n: int
    d: float
    c: float

    def __init__(self, n=None, d=None, c=0.0):
        """
        Parameters
        ----------
        n : int = None
            Number of pixels along each axis.
        d : float = None
            Pixel scale in metres.
        c : float = 0.0
            Centre coordinate of the grid, in metres.
        """
        self.n = n
        self.d = d
        self.c = c

    @property
    def xs(self):
        """
        1D array of pixel centre coordinates along one axis.

        Returns
        -------
        xs : Array
            Coordinates of pixel centres, in metres, centred on `c`.
        """
        if self.d is None:
            raise ValueError("d must be specified to calculate coordinates.")
        return self.c + (np.arange(self.n) - (self.n - 1) / 2) * self.d

    @property
    def fov(self):
        """
        Total field of view of the grid.

        Returns
        -------
        fov : float
            Field of view in metres, equal to ``n * d``.
        """
        if self.d is None:
            raise ValueError("d must be specified to calculate FOV.")
        return self.n * self.d

    @property
    def extent(self):
        """
        Coordinate range (min, max) of the grid edges.

        Returns
        -------
        extent : tuple[float, float]
            ``(lower_edge, upper_edge)`` coordinates in metres.
        """
        if self.d is None:
            raise ValueError("d must be specified to calculate extent.")
        return self.c - (self.n / 2) * self.d, self.c + (self.n / 2) * self.d
