from __future__ import annotations
from collections import OrderedDict
from typing import Union
from jax import Array
import dLux.utils as dlu
import dLux

__all__ = ["LayeredDetector"]

DetectorLayer = lambda: dLux.detector_layers.DetectorLayer
PSF = lambda: dLux.psfs.PSF


class LayeredDetector(dLux.base.BaseDetector):
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

    def __init__(self: LayeredDetector, layers: list):
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
        self.layers = dlu.list2dictionary(layers, True, DetectorLayer())
        super().__init__()

    def __getattr__(self: LayeredDetector, key: str) -> object:
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

    def model(
        self: LayeredDetector, psf: PSF(), return_psf: bool = False
    ) -> Array:
        """
        Applied the stored detector layers to the input psf.

        Parameters
        ----------
        PSF : Array
            The input psf to be transformed.

        Returns
        -------
        psf : Array
            The output 'psf' after being transformed by the detector layers.
        """
        for key, layer in self.layers.items():
            psf = layer(psf)
        if return_psf:
            return psf
        return psf.data

    def insert_layer(
        self: LayeredDetector, layer: Union[DetectorLayer, tuple], index: int
    ) -> LayeredDetector:
        """
        Inserts a layer into the layers dictionary at the given index using the
        list.insert method. Note this method may require the names of some
        parameters to be

        Parameters
        ----------
        layer : Union[DetectorLayer, tuple]
            The layer to insert into the layers dictionary. Can either be a
            single DetectorLayer, or you can specify the layers key by passing
            in a tuple of (DetectorLayer, key).
        index : int
            The index to insert the layer at.
        """
        layers_list = list(zip(self.layers.values(), self.layers.keys()))
        layers_list.insert(index, layer)
        new_layers = dlu.list2dictionary(layers_list, True, DetectorLayer())
        return self.set("layers", new_layers)

    def remove_layer(self: LayeredDetector, key: str) -> LayeredDetector:
        """
        Removes a layer from the layers dictionary indexed at 'key' using the
        dict.pop(key) method.

        Parameters
        ----------
        key : str
            The key of the layer to remove.
        """
        self.layers.pop(key)
        return self
