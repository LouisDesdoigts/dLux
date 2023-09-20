from __future__ import annotations
from collections import OrderedDict
from abc import abstractmethod
from typing import Union
from zodiax import Base
from jax import Array
import dLux.utils as dlu

from .layers.detector_layers import DetectorLayer
from .psfs import PSF


__all__ = ["BaseDetector", "LayeredDetector"]


class BaseDetector(Base):
    @abstractmethod
    def model(self, psf):  # pragma: no cover
        pass


class LayeredDetector(BaseDetector):
    """
    Applies a series of detector layers to some input psf.

    ??? abstract "UML"
        ![UML](../../assets/uml/LayeredDetector.png)

    Attributes
    ----------
    layers: OrderedDict
        A series of `DetectorLayer` transformations to apply to the input psf.
    """

    layers: OrderedDict

    def __init__(self: LayeredDetector, layers: list):
        """
        Parameters
        ----------
        layers : list
            A list of DetectorLayer objects to apply to the input psf.
        """
        self.layers = dlu.list2dictionary(layers, True, DetectorLayer)
        super().__init__()

    def __getattr__(self: LayeredDetector, key: str) -> object:
        """
        Raises the individual layers via their keys.

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

    def model(
        self: LayeredDetector, psf: PSF, return_psf: bool = False
    ) -> Array:
        """
        Applied the detector layers to the input psf.

        Parameters
        ----------
        psf : PSF
            The input psf to be transformed.

        Returns
        -------
        psf : PSF
            The output psf after being transformed by the detector layers.
        """
        for key, layer in self.layers.items():
            psf = layer.apply(psf)
        if return_psf:
            return psf
        return psf.data

    def insert_layer(
        self: LayeredDetector, layer: Union[DetectorLayer, tuple], index: int
    ) -> LayeredDetector:
        """
        Inserts a layer into the layers dictionary at a specified index. This function
        calls the list2dictionary function to ensure all keys remain unique. Note that
        this can result in some keys being modified if they are duplicates. The input
        'layer' can be a tuple of (key, layer) to specify a key, else the key is taken
        as the class name of the layer.

        Parameters
        ----------
        layer : Any
            The layer to be inserted.
        index : int
            The index at which to insert the layer.

        Returns
        -------
        detector : LayeredDetector
            The updated detector.
        """
        return self.set(
            "layers",
            dlu.insert_layer(self.layers, layer, index, DetectorLayer),
        )

    def remove_layer(self: LayeredDetector, key: str) -> LayeredDetector:
        """
        Removes a layer from the layers dictionary, specified by its key.

        Parameters
        ----------
        key : str
            The key of the layer to be removed.

        Returns
        -------
        detector : LayeredDetector
            The updated detector.
        """
        return self.set("layers", dlu.remove_layer(self.layers, key))
