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

    Parameters:
    -----------
    apertures: dict(str, Aperture)
       The apertures that make up the compound aperture. 
    """
    apertures: dict


    def __init__(self: ApertureLayer, apertures: dict) -> ApertureLayer:
        """
        Parameters
        ----------
        apertures : dict
           The aperture objects stored in a dictionary of type
           {str : Aperture} where the Aperture is a subclass of the 
           Aperture.
        """
        super().__init__()
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

help(DynamicAperture)


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

    Parameters:
    -----------
    apertures: dict(str, ApertureLayer)
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
           for ap in self._apertures.values()]).prod(axis=0)

comp_ap: ApertureLayer = CompoundAperture({
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


