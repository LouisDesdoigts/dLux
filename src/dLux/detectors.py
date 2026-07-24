"""Detector models that apply detector-layer transformations to PSFs."""

from __future__ import annotations
from collections import OrderedDict
from typing import Any
from abc import abstractmethod

from jax import Array
import zodiax as zdx
import dLux.utils as dlu

from .layers.detector_layers import BaseDetectorLayer
from .psfs import PSF

__all__ = ["BaseDetector", "LayeredDetector"]


class BaseDetector(zdx.Base):
    """Base class for detector models."""

    @abstractmethod
    def __call__(
        self,
        psf: PSF,
        return_psf: bool = False,
    ) -> Array | PSF:  # pragma: no cover
        """Apply the detector model to a PSF."""

    def model(self, psf: PSF, return_psf: bool = False) -> Array | PSF:
        """Backwards-compatible alias for calling the detector."""
        return self(psf, return_psf)


class LayeredDetector(BaseDetector):
    """
    Applies a series of detector layers to an input PSF.

    ??? abstract "UML"
        ![UML](../assets/uml/LayeredDetector.png)

    Attributes
    ----------
    layers: OrderedDict
        A series of `BaseDetectorLayer` transformations to apply to the input PSF.
    """

    layers: OrderedDict

    def __init__(
        self: LayeredDetector,
        layers: list[BaseDetectorLayer | tuple[str, BaseDetectorLayer]],
    ):
        """
        Parameters
        ----------
        layers : list[BaseDetectorLayer | tuple[str, BaseDetectorLayer]]
            A list of BaseDetectorLayer objects to apply to the input PSF. List entries
            can be tuples of (key, layer) to specify a key, else the key is taken as
            the class name of the layer.
        """
        self.layers = dlu.list2dictionary(layers, True, BaseDetectorLayer)
        super().__init__()

    def __getattr__(self: LayeredDetector, key: str) -> Any:
        """
        Raises the individual layers via their keys.

        Parameters
        ----------
        key : str
            The key of the item to be searched for in the layers dictionary.

        Returns
        -------
        item : Any
            The item corresponding to the supplied key in the layers
            dictionary.
        """
        if key in self.layers.keys():
            return self.layers[key]
        for layer in list(self.layers.values()):
            if hasattr(layer, key):
                return getattr(layer, key)
        raise dlu.helpers.missing_attribute_error(
            self,
            key,
            list(self.layers.keys()),
        )

    def __call__(
        self: LayeredDetector,
        psf: PSF,
        return_psf: bool = False,
    ) -> Array | PSF:
        """
        Applies the detector layers to the input PSF.

        Parameters
        ----------
        psf : PSF
            The input PSF to be transformed.
        return_psf : bool = False
            Should the PSF object be returned instead of the PSF array?

        Returns
        -------
        result : Array | PSF
            If `return_psf` is False, returns the PSF array.
            If `return_psf` is True, returns the PSF object.
        """
        for _, layer in self.layers.items():
            psf = layer(psf)
        if return_psf:
            return psf
        return psf.data

    def insert_layer(
        self: LayeredDetector,
        layer: BaseDetectorLayer | tuple[str, BaseDetectorLayer],
        index: int,
    ) -> LayeredDetector:
        """
        Inserts a layer into the layers dictionary at a specified index. This function
        calls the list2dictionary function to ensure all keys remain unique. Note that
        this can result in some keys being modified if they are duplicates. The input
        'layer' can be a tuple of (key, layer) to specify a key, else the key is taken
        as the class name of the layer.

        Parameters
        ----------
        layer : BaseDetectorLayer | tuple[str, BaseDetectorLayer]
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
            dlu.insert_layer(self.layers, layer, index, BaseDetectorLayer),
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
