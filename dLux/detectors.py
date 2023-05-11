from __future__ import annotations
import jax.numpy as np
from jax import Array
from zodiax import Base
from collections import OrderedDict
from copy import deepcopy
from abc import abstractmethod
import dLux


__all__ = ["Detector", "BaseDetector"]

DetectorLayer = lambda : dLux.detector_layers.DetectorLayer

class BaseDetector(Base):
    
    @abstractmethod
    def model(self, image): # pragma: no cover
        pass

class Detector(BaseDetector):
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

            A list of ∂Lux 'layers' that define the transformations and
            operations upon some input wavefront through an optical system.
            The entried can either be dLux DetectorLayers, or tuples of the
            form (DetectorLayer, key), with the key being used as the dictionary
            key for the layer.
        """
        # Ensure input is a list
        if not isinstance(layers, list):
            raise ValueError("Input layers must be a list, it is"
                " automatically converted to a dictionary.")
        # Ensure all entries are dLux layers
        for layer in layers:
            if isinstance(layer, tuple):
                layer = layer[0]
            if not isinstance(layer, DetectorLayer()):
                raise ValueError("All entries within layers must be an "
                    "DetectorLayer object.")

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


    def model(self : Detector, image: Array) -> Array:
        """
        Applied the stored detector layers to the input image.

        Parameters
        ----------
        image : Array
            The input psf to be transformed.

        Returns
        -------
        image : Array
            The ouput 'image' after being transformed by the detector layers.
        """
        if image.ndim != 2:
            raise ValueError("image must be a 2d array.")

        # Apply detector layers
        for key, layer in self.layers.items():
            image = layer(image)
        return image