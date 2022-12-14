import jax.numpy as np
import jax 
import dLux
import abc
import typing


from apertures import *

jax.config.update("jax_enable_x64", True)

Array = typing.TypeVar("Array")
Wavefront = typing.TypeVar("Wavefront")


class CompositeAperture(ApertureLayer):
    """
    Represents an aperture that contains more than one single 
    aperture. The smaller sub-apertures are stored in a dictionary
    pytree and are so acessible by user defined name.
    
    This class should be used if you want to learn the parameters
    of the entire aperture without learning the individual components.
    This is often going to be useful for pupils with spiders since 
    the connection implies that changes to once are likely to 
    affect one another.

    Parameters:
    -----------
    apertures: dict(str, Aperture)
       The apertures that make up the compound aperture. 
    centre: float, meters
        The x coordinate of the centre of the aperture.
    strain: Array
        Linear stretching of the x and y axis representing a 
        strain of the coordinate system.
    compression: Array 
        The x and y compression of the coordinate system. This 
        is a constant. 
    rotation: float, radians
        The rotation of the aperture away from the positive 
        x-axis. 
    """
    centre: Array
    strain: Array
    compression: Array
    rotation: Array
    apertures: dict
    

    def __init__(self   : ApertureLayer, 
            apertures   : dict,
            centre      : Array = [0., 0.], 
            strain      : Array = [0., 0.],
            compression : Array = [1., 1.],
            rotation    : Array = 0.) -> ApertureLayer:
        """
        The default aperture is dis-allows the learning of all 
        parameters. 

        Parameters
        ----------
        centre: float, meters
            The centre of the coordinate system along the x-axis.
        softening: bool = False
            True if the aperture is soft edged otherwise False. A
            soft edged aperture has a small layer of non-binary 
            pixels. This is to prevent undefined gradients. 
        occulting: bool = False
            True if the aperture is occulting else False. An 
            occulting aperture is zero inside and one outside. 
        strain: Array
            Linear stretching of the x and y axis representing a 
            strain of the coordinate system.
        compression: Array 
            The x and y compression of the coordinate system. This 
            is a constant. 
        rotation: float, radians
            The rotation of the aperture away from the positive 
            x-axis. 
        apertures : dict
           The aperture objects stored in a dictionary of type
           {str : Aperture} where the Aperture is a subclass of the 
           Aperture.
        """
        super().__init__()
        self.centre = np.asarray(centre).astype(float)
        self.strain = np.asarray(strain).astype(float)
        self.compression = np.asarray(compression).astype(float)
        self.rotation = np.asarray(rotation).astype(float)
        self.apertures = apertures


    def __getitem__(self: ApertureLayer, key: str) -> ApertureLayer:
        """
        Get one of the apertures from the collection using a name 
        based lookup.

        Parameters:
        -----------
        key: str
           The name of the aperture to lookup. See the class doc
           string for more information.

        Returns:
        --------
        layer: Aperture
           The layer that was stored under the name `key`. 
        """
        return self.apertures[key]


    def __setitem__(self, key: str, value: ApertureLayer) -> None:
        """
        Assign a new value to one of the aperture mirrors.
        Parameters
        ----------
        key : str
           The name of the segement to replace for example "B1-7".
        value : ApertureLayer
           The new value to assign to that segement.
        """
        self.apertures[key] = value


    def __call__(self, wavefront: Wavefront) -> Wavefront:
        """
        Apply the aperture to an incoming wavefront.
        Parameters
        ----------
        parameters : dict
           A dictionary containing the parameters of the model. 
           The dictionary must satisfy `parameters.get("Wavefront")
           != None`. 
        Returns
        -------
        parameters : dict
           The parameter, parameters, with the "Wavefront"; key
           value updated. 
        """
        wavefront = wavefront.multiply_amplitude(
           self._aperture(
               wavefront.pixel_coordinates()))
        return parameters


    @abc.abstractmethod
    def _aperture(self: ApertureLayer, coordinates: Array) -> Array:
        """
        Evaluates the aperture. 

        Parameters:
        -----------
        coordinates: Array, meters
           The coordinates of the paraxial array. 

        Returns 
        -------
        aperture : Matrix
           An aperture generated by combining all of the sub 
           apertures that were stored. 
        """


class CompoundAperture(CompositeAperture):
    """
    Represents an aperture that contains more than one single 
    aperture. The smaller sub-apertures are stored in a dictionary
    pytree and are so acessible by user defined name. The 
    `CompoundAperture` contains overlapping apertures that 
    may or may not be occulting. The goal is mainly to represent
    `AnnularAperture`s that have `UniformSpider`s embedded. This
    class should not be used to represent multiple apertures 
    that are not connected. Doing so will result in a zero 
    output.
    
        
    This class should be used if you want to learn the parameters
    of the entire aperture without learning the individual components.
    This is often going to be useful for pupils with spiders since 
    the connection implies that changes to once are likely to 
    affect one another.

    Parameters:
    -----------
    apertures: dict(str, Aperture)
       The apertures that make up the compound aperture. 
    centre: float, meters
        The x coordinate of the centre of the aperture.
    strain: Array
        Linear stretching of the x and y axis representing a 
        strain of the coordinate system.
    compression: Array 
        The x and y compression of the coordinate system. This 
        is a constant. 
    rotation: float, radians
        The rotation of the aperture away from the positive 
        x-axis. 
    """


    def __init__(
            self        : ApertureLayer,
            apertures   : dict,
            centre      : Array = [0., 0.], 
            strain      : Array = [0., 0.],
            compression : Array = [1., 1.],
            rotation    : Array = 0.) -> ApertureLayer:
        """
        Parameters:
        -----------
        apertures: dict(str, Aperture)
           The apertures that make up the compound aperture. 
        centre: float, meters
            The x coordinate of the centre of the aperture.
        strain: Array
            Linear stretching of the x and y axis representing a 
            strain of the coordinate system.
        compression: Array 
            The x and y compression of the coordinate system. This 
            is a constant. 
        rotation: float, radians
            The rotation of the aperture away from the positive 
            x-axis. 
        """
        super().__init__(apertures,
            centre = centre,
            strain = strain,
            compression = compression,
            rotation = rotation)

    def _coordinates(self: ApertureLayer, coords: Array) -> Array:
        
        rotation: float = self.rotation # Shorter reference.
        x_trans: float = np.array([np.cos(rotation), np.sin(rotation)])
        y_trans: float = np.array([-np.sin(rotation), np.cos(rotation)])
        new_x: float = np.cos(self.rotation) * x + np.sin(self.rotation) * y
        new_y: float = -np.sin(self.rotation) * x + np.cos(self.rotation) * y
        coords: float = np.array([new_x, new_y])


        return coordinates - self.centre[:, None, None]

        trans_coords: Array = np.transpose(coords, (0, 2, 1))
        return coords + trans_coords * self.strain[:, None, None]

        return coords * self.compression[:, None, None]

        is_trans = (self.centre != np.zeros((2,), float)).any()
        coords: Array = jax.lax.cond(is_trans,
            lambda: self._translate(coords),
            lambda: coords)

        is_compr: bool = (self.compression != np.ones((2,), float)).any()
        coords: Array = jax.lax.cond(is_compr,
            lambda: self._compress(coords),
            lambda: coords)

        is_strain: bool = (self.strain != np.zeros((2,), float)).any()
        coords: Array = jax.lax.cond(is_strain,
            lambda: self._strain(coords),
            lambda: coords)

        is_rot: bool = (self.rotation != 0.)
        coords: Array = jax.lax.cond(is_rot,
            lambda: self._rotate(coords),
            lambda: coords)

        return coords


    def _aperture(self, coordinates: Array) -> Array:
        """
        Evaluates the aperture. 

        Parameters:
        -----------
        coordinates: Array, meters
           The coordinates of the paraxial array. 

        Returns 
        -------
        aperture : Matrix
           An aperture generated by combining all of the sub 
           apertures that were stored. 
        """
        coords: float = self.apertures.values[]
        return np.stack([ap._aperture(coordinates) 
           for ap in self._apertures.values()]).prod(axis=0)

grid: float = np.linspace(0., 1., 128) - .5
coords: float = np.array(np.meshgrid(grid, grid))


@jax.jit
def rotate_v1(rotation: float, coords: float) -> float:
    x: float = coords[0]
    y: float = coords[1]
    new_x: float = np.cos(rotation) * x + np.sin(rotation) * y
    new_y: float = -np.sin(rotation) * x + np.cos(rotation) * y
    coords: float = np.array([new_x, new_y])
    return coords


jax.make_jaxpr(rotate_v1)(np.pi / 2., coords)

# %%timeit
rotate_v1(np.pi / 2., coords)


@jax.jit
def rotate_v2(rotation: float, coords: float) -> float:
    x_trans: float = np.array([np.cos(rotation), np.sin(rotation)])
    y_trans: float = np.array([-np.sin(rotation), np.cos(rotation)])
    new_x: float = (x_trans[:, None, None] * coords).sum(axis=0)
    new_y: float = (y_trans[:, None, None] * coords).sum(axis=0)
    coords: float = np.array([new_x, new_y])


jax.make_jaxpr(rotate_v2)(np.pi / 2., coords)

# %%timeit
rotate_v2(np.pi / 2., coords)

comp_ap: ApertureLayer = CompoundAperture(
    apertures = {
        "pupil": CircularAperture(1.),
        "obstruction": CircularAperture(.5, occulting=True),
    })

# +
test_plots_of_aps({
    
})


# -

class MultiAperture(CompositeAperture):
    """
    Represents an aperture that contains more than one single 
    aperture. The smaller sub-apertures are stored in a dictionary
    pytree and are so acessible by user defined name. The 
    `MultiAperture` is used to represent apertures that are 
    not overlapping. We can add `CompoundAperture`s into 
    `MultiAperture` to create a combination of the two affects.

    Attributes
    ----------
    apertures : dict(str, ApertureLayer)
       The apertures that make up the compound aperture. 
    """


    def __init__(self: ApertureLayer, apertures: dict) -> ApertureLayer:
        """
        Parameters
        ----------
        apertures : dict
           The aperture objects stored in a dictionary of type
           {str : ApertureLayer} where the ApertureLayer is a subclass of the 
           ApertureLayer.
        """
        super().__init__(apertures)


    def _aperture(self, coordinates: Array) -> Array:
        """
        Evaluates the aperture. 

        Parameters:
        -----------
        coordinates: Array, meters
           The coordinates of the paraxial array. 

        Returns 
        -------
        aperture : Matrix
           An aperture generated by combining all of the sub 
           apertures that were stored. 
        """
        return np.stack([ap._aperture(coordinates) 
           for ap in self.apertures.values()]).sum(axis=0)


    def to_list(self: ApertureLayer) -> list:
        """
        Returns:
        --------
        layers: list
           A list of `Aperture` objects that comprise the 
           `MultiAperture`.
        """
        return list(self.apertures.values())


