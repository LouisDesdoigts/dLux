from __future__ import annotations
from jax import Array
from zodiax import Base
from abc import abstractmethod

__all__ = [
    "BaseInstrument",
    "BaseSpectrum",
    "BaseSourceObject",
    "BaseOpticalLayer",
    "BaseOptics",
    "BaseDetectorLayer",
    "BaseDetector",
]


class BaseInstrument(Base):
    @abstractmethod
    def model(self):  # pragma: no cover
        pass


class BaseSpectrum(Base):
    @abstractmethod
    def normalise(self):  # pragma: no cover
        pass


class BaseSourceObject(Base):
    # TODO: Add this to allow custom sources

    @abstractmethod
    def normalise(self):  # pragma: no cover
        pass

    @abstractmethod
    def model(self, optics):  # pragma: no cover
        pass


class BaseOpticalLayer(Base):
    @abstractmethod
    def __call__(self: BaseOpticalLayer, wavefront):  # pragma: no cover
        pass


class BaseOptics(Base):
    @abstractmethod
    def propagate_mono(
        self: BaseOptics,
        wavelength: Array,
        offset: Array,
        return_wf: bool,
    ) -> Array:  # pragma: no cover
        pass

    @abstractmethod
    def propagate(
        self: BaseOptics,
        wavelengths: Array,
        offset: Array,
        weights: Array,
        return_wf: bool,
    ):
        pass

    @abstractmethod
    def model(
        self: BaseOptics,
        source: BaseSourceObject,
        return_wf: bool = False,
    ) -> Array:
        pass


class BaseDetectorLayer(Base):
    @abstractmethod
    def __call__(self: BaseDetectorLayer, psf):  # pragma: no cover
        pass


class BaseDetector(Base):
    @abstractmethod
    def model(self, psf):  # pragma: no cover
        pass
