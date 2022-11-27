import matplotlib.pyplot as plt
import matplotlib as mpl 
import dLux as dl
import jax.numpy as np
import abc
import typing

mpl.rcParams["text.usetex"] = True

Array = typing.TypeVar("Array")
Layer = typing.TypeVar("Layer")


class Spider(dl.Aperture, abc.ABC):
    """
    An abstraction on the concept of an optical spider for a space telescope.
    These are the things that hold up the secondary mirrors. For example,
    """
    def __init__(self, x_offset: float, y_offset: float, softening: bool):
        super().__init__(x_offset, y_offset, False, softening)


    def _strut(self, angle: float, coordinates: Array) -> Array:
        """
        Generates a representation of a single strut in the spider. This is 
        more complex than you might imagine since the strut can point in 
        any direction. 

        Parameters
        ----------
        angle: float, radians
            The angle that this strut points as measured from the positive 
            x-axis in radians. 

        Returns
        -------
        strut: float
            The soft edged strut. 
        """
        x, y = coordinates[0][:, ::-1], coordinates[1][:, :]
        perp = np.tan(angle)
        gradient = np.tan(angle)
        dist = np.abs(y - gradient * x) / np.sqrt(1 + gradient ** 2)
        theta = np.arctan2(y, x) + np.pi 
        theta = np.where(theta > angle, theta - angle, theta + 2 * np.pi - angle)
        theta = np.where(theta > 2 * np.pi, theta - 2 * np.pi, theta)

        # So the current problem is that I need to translate the coordinates 
        # around by angle and then return them to the range [0, 2 pi].

        strut = np.where((theta > np.pi / 2.) & (theta < 3. * np.pi / 2.), 1., dist)

        # This is all a hot mess. That is what this is. So how do I fix it?
        # Well it is obviously not trivial. I to take an array of angles and
        # essentially rotate it by angle. So I need to find all of the 
        # points where theta is greater than angle and then I subtract 
        # angle from theta in these points. Where theta is less than
        # angle I want to add two pi - angle. Consider the case of angle equals
        # three pi on two. The fourth quadrant has three pi on two subtracted 
        # giving it a range of zero to pi on two. Yes I think that this will 
        # work. 

        return strut


class UniformSpider(Spider):
    """
    A spider with equally-spaced, equal-width struts. This is of course the 
    most common and simplest implementation of a spider. Gradients can be 
    taken with respect to the width of the struts and the global rotation 
    as well as the centre of the spider.

    Parameters
    ----------
    number_of_struts: int 
        The number of struts to equally space around the circle. This is not 
        a differentiable parameter. 
    width_of_struts: float, meters
        The width of each strut. 
    rotation: float, radians
        A global rotation to apply to the entire spider. 
    """
    number_of_struts: int
    width_of_struts: float
    rotation: float


    def __init__(
            self: Layer,
            x_offset: float,
            y_offset: float, 
            number_of_struts: int, 
            width_of_struts: float, 
            rotation: float,
            softening: bool) -> Layer:
        """
        Parameters
        ----------
        radius_of_spider: float, meters
            The physical width of the spider. For the moment it is assumed to 
            be embedded within a circular aperture.         
        center_of_spicer: Array, meters 
            The [x, y] center of the spider.
        number_of_struts: int 
            The number of struts to equally space around the circle. This is not 
            a differentiable parameter. 
        width_of_struts: float, meters
            The width of each strut. 
        rotation: float, radians
            A global rotation to apply to the entire spider.
        """ 
        super().__init__(x_offset, y_offset, softening)
        self.number_of_struts = int(number_of_struts)
        self.rotation = np.asarray(rotation).astype(float)
        self.width_of_struts = np.asarray(width_of_struts).astype(float)


    def _metric(self, coordinates: Array) -> Array:
        coordinates = self._translate(coordinates)
        angles = np.linspace(0, 2 * np.pi, self.number_of_struts, 
            endpoint=False)
        angles += self.rotation
        struts = np.array([self._strut(angle, coordinates) for angle in angles]) - self.width_of_struts / 2.
        softened = self._soften(struts)
        return softened.prod(axis=0)
        
 
    def __call__(self: Layer, params: dict) -> dict:
        """
        Apply the spider to a wavefront, as it propagates through the spider. 

        Parameters
        ----------
        params: dict
            A dictionary of parameters that contains a "Wavefront" key. 

        Returns 
        -------
        params: dict 
            The same dictionary with the "Wavefront" value updated.
        """
        aperture = self._spider()
        wavefront = params["Wavefront"]
        wavefront = wavefront\
            .set_amplitude(wavefront.get_amplitude() * aperture)\
            .set_phase(wavefront.get_phase() * aperture)
        params["Wavefront"] = wavefront
        return params


coordinates = dl.utils.get_pixel_coordinates(24, 2. / 24.)

# Uniform Spider Testing
even_soft_unif_spider = UniformSpider(0., 0., 4., .1, 0., softening=True)
odd_soft_unif_spider = UniformSpider(0., 0., 3., .1, 0., softening=True)

plt.imshow(even_soft_unif_spider._aperture(coordinates))
plt.colorbar()
plt.show()

plt.imshow(odd_soft_unif_spider._aperture(coordinates))
plt.colorbar()
plt.show()
