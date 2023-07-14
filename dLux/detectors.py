from __future__ import annotations
from abc import abstractmethod
from collections import OrderedDict
from jax import Array
from zodiax import Base
import dLux.utils as dlu
import dLux

__all__ = ["LayeredDetector"]

DetectorLayer = lambda: dLux.detector_layers.DetectorLayer


class BaseDetector(Base):
    @abstractmethod
    def model(self, image):  # pragma: no cover
        pass


class LayeredDetector(BaseDetector):
    """
    A high level class designed to model the behaviour of some detectors
    response to some psf.

    Attributes
    ----------
    layers: dict
        A collections.OrderedDict of 'layers' that define the transformations
        and operations upon some input psf as it interacts with the detector.
    """

    layers: OrderedDict

    def __init__(self: BaseDetector, layers: list):
        """
        Constructor for the Detector class.

        Parameters
        ----------
        layers : list
            An list of dLux detector layer classes that define the instrumental
            effects for some detector.

            A list of ∂Lux 'layers' that define the transformations and
            operations upon some input wavefront through an optical system.
            The entries can either be dLux DetectorLayers, or tuples of the
            form (DetectorLayer, key), with the key being used as the
            dictionary key for the layer.
        """
        self.layers = dlu.list_to_dictionary(layers, True, DetectorLayer())
        super().__init__()

    def __getattr__(self: BaseDetector, key: str) -> object:
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
            The item corresponding to the supplied key in the layers
            dictionary.
        """
        if key in self.layers.keys():
            return self.layers[key]
        else:
            raise AttributeError(
                "'{}' object has no attribute '{}'".format(type(self), key)
            )

    def model(self: BaseDetector, image: Array) -> Array:
        """
        Applied the stored detector layers to the input image.

        Parameters
        ----------
        image : Array
            The input psf to be transformed.

        Returns
        -------
        image : Array
            The output 'image' after being transformed by the detector layers.
        """
        for key, layer in self.layers.items():
            image = layer(image)
        return image.image
