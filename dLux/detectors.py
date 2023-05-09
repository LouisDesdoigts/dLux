from __future__ import annotations
import jax.numpy as np
from jax import Array
from zodiax import Base
from collections import OrderedDict
from copy import deepcopy
# from typing import Union
# from abc import abstractmethod
import dLux


class BaseDetector(Base):
    
    @abstractmethod
    def normalise(self): # pragma: no cover
        pass

    @abstractmethod
    def model(self, optics, detector=None): # pragma: no cover
        pass

class Detector(Base):
    """
    A high level class desgined to model the behaviour of some detectors
    response to some psf.

    Attributes
    ----------
    layers: dict
        A collections.OrderedDict of 'layers' that define the transformations
        and operations upon some input psf as it interacts with the detector.
    """
    layers : OrderedDict


    def __init__(self : Detector, layers : list) -> Instrument:
        """
        Constructor for the Detector class.

        Parameters
        ----------
        layers : list
            An list of dLux detector layer classes that define the instrumental
            effects for some detector.
        """
        # Ensure input is a list
        assert isinstance(layers, list), ("Input layers must be a list, it is" \
        " automatically converted to a dictionary.")

        # Ensure all entries are dLux layers
        for layer in layers:
            assert isinstance(layer, dLux.detectors.DetectorLayer), \
            ("All entries within layers must be a "
             "dLux.detectors.DetectorLayer object.")

        # Construct layers
        self.layers = dLux.utils.list_to_dictionary(layers)


    def __getattr__(self : Detector, key : str) -> object:
        """
        Magic method designed to allow accessing of the various items within
        the layers dictionary of this class via the 'class.attribute' method.

        Parameters
        ----------
        key : str
            The key of the item to be searched for in the layers dictionary.

        Returns
        -------
        item : object
            The item corresponding to the supplied key in the layers dictionary.
        """
        if key in self.layers.keys():
            return self.layers[key]
        else:
            raise AttributeError("'{}' object has no attribute '{}'"\
                                 .format(type(self), key))


    def apply_detector(self : Instrument, image : Array) -> Array:
        """
        Applied the stored detector layers to the input image.

        Parameters
        ----------
        image : Array
            The input 'image' to the detector to be transformed.

        Returns
        -------
        image : Array
            The ouput 'image' after being transformed by the detector layers.
        """
        # Input type checking
        assert isinstance(image, np.ndarray), "Input must be a jax array."
        assert image.ndim == 2, "Input image must a 2d array."

        # Apply detector layers
        for key, layer in self.layers.items():
            image = layer(image)
        return image


    # def debug_apply_detector(self  : Instrument, 
    #                          image : Array) -> Array:
    #     """
    #     Applied the stored detector layers to the input image, storing and
    #     returning the intermediate states of the image and layers.

    #     Parameters
    #     ----------
    #     image : Array
    #         The input 'image' to the detector to be transformed.

    #     Returns
    #     -------
    #     image : Array
    #         The ouput 'image' after being transformed by the detector layers.
    #     intermediate_images : list
    #         The intermediate states of the image.
    #     intermediate_layers : list
    #         The intermediate states of each layer after being applied to the
    #         image.
    #     """
    #     # Input type checking
    #     assert isinstance(image, np.ndarray), "Input must be a jax array."
    #     assert image.ndim == 2, "Input image must a 2d array."

    #     # Apply detector layers
    #     intermediate_images = []
    #     intermediate_layers = []
    #     for key, layer in self.layers.items():
    #         image = layer(image)
    #         intermediate_images.append(image.copy())
    #         intermediate_layers.append(deepcopy(layer))
    #     return image, intermediate_images, intermediate_layers
    

    def summarise(self : Detector) -> None: # pragma: no cover
        """
        Prints a summary of all the layers in the detector.
        """
        print("Text summary:")
        keys = self.layers.keys()
        for key in keys:
            layer = self.layers[key]
            print(f"  {layer.summary()}")
        print('\n')


    def plot(self : Optics, image : Array) -> None: # pragma: no cover
        """
        Prints the summary of all the layers and then plots a image as it
        propagates through the detector layer.

        Parameters
        ----------
        iamge : Array
            The image to propagate through the detector.
        """
        self.summarise()
        keys = self.layers.keys()
        for key in keys:
            layer = self.layers[key]
            print(f"{layer.summary()}")
            image = layer(image)
            layer.display(image)


    def model(self : Detector, image: Array) -> Array:
        """
        A function to apply the detector layers to the input image.

        Parameters
        ----------
        image: Array
            The image to be transformed by the detector layers.

        Returns
        -------
        image : Array
            The image after being transformed by the detector layers.
        """
        return self.apply_detector(image)
