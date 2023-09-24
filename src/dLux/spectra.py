from __future__ import annotations
from abc import abstractmethod
import jax.numpy as np
from zodiax import Base
from jax import vmap, Array

__all__ = ["BaseSpectrum", "Spectrum", "PolySpectrum"]


class BaseSpectrum(Base):
    @abstractmethod
    def normalise(self):  # pragma: no cover
        pass


class SimpleSpectrum(BaseSpectrum):
    """
    Base class for arbitrary spectral parametrisations.

    Attributes
    ----------
    wavelengths : Array, metres
        The array of wavelengths at which the spectrum is defined.
    """

    wavelengths: Array

    def __init__(self: Spectrum, wavelengths: Array):
        """
        Parameters
        ----------
        wavelengths : Array, metres
            The array of wavelengths at which the spectrum is defined.
        """
        self.wavelengths = np.asarray(wavelengths, dtype=float)
        super().__init__()


class Spectrum(SimpleSpectrum):
    """
    A simple spectrum class using wavelengths and weights.


    ??? abstract "UML"
        ![UML](../../assets/uml/Spectrum.png)

    Attributes
    ----------
    wavelengths : Array, metres
        The array of wavelengths at which the spectrum is defined.
    weights : Array
        The relative weights of each wavelength.
    """

    weights: Array

    def __init__(self: Spectrum, wavelengths: Array, weights: Array = None):
        """
        Parameters
        ----------
        wavelengths : Array, metres
            The array of wavelengths at which the spectrum is defined.
        weights : Array = None
            The relative weights of each wavelength. Input weights
            are automatically normalised to a sum of 1.
        """
        super().__init__(wavelengths)
        if weights is None:
            in_shape = self.wavelengths.shape
            weights = np.ones(in_shape) / in_shape[-1]

        weights = np.asarray(weights, dtype=float)
        if weights.ndim == 2:
            self.weights = weights / weights.sum(-1)[:, None]
        else:
            self.weights = weights / weights.sum()

        if self.weights.ndim == 1:
            if self.wavelengths.shape != self.weights.shape:
                raise ValueError(
                    "wavelengths and weights must have the same " "shape."
                )
        else:
            if self.wavelengths.shape != self.weights.shape[-1:]:
                raise ValueError(
                    "wavelengths and weights must have the same " "shape."
                )

    def normalise(self: Spectrum) -> Spectrum:
        """
        Returns a normalised spectrum object, where the weights are normalised to a
        sum of 1.

        Returns
        -------
        spectrum : Spectrum
            The spectrum object with the normalised spectrum.
        """
        if self.weights.ndim == 2:
            weight_sum = self.weights.sum(-1)[:, None]
        else:
            weight_sum = self.weights.sum()
        return self.divide("weights", weight_sum)


class PolySpectrum(SimpleSpectrum):
    """
    Implements a generic polynomial spectrum, such as a linear spectrum.

    This implements a polynomial as follows: f(x) = c0 + c1*x + c2*x^2 + ... + cn*x^n

    ??? abstract "UML"
        ![UML](../../assets/uml/PolySpectrum.png)

    Attributes
    ----------
    wavelengths : Array, metres
        The array of wavelengths at which the spectrum is defined.
    coefficients : Array
        The array of polynomial coefficient values.
    """

    coefficients: Array

    def __init__(self: Spectrum, wavelengths: Array, coefficients: Array):
        """
        Parameters
        ----------
        wavelengths : Array, metres
            The array of wavelengths at which the spectrum is defined.
        coefficients : Array
            The array of polynomial coefficient values.
        """
        super().__init__(wavelengths)
        self.coefficients = np.asarray(coefficients, dtype=float)

        if self.coefficients.ndim != 1:
            raise ValueError("Coefficients must be a 1d array.")

    def _eval_weight(self: Spectrum, wavelength: Array) -> Array:
        """
        Evaluates the polynomial function at the supplied wavelength.

        Parameters
        ----------
        wavelength : Array, metres
            The wavelength at which to evaluate the polynomial function.

        Returns
        -------
        weight : Array
            The relative weight of the supplied wavelength.
        """
        return np.array(
            [
                self.coefficients[i] * wavelength**i
                for i in range(len(self.coefficients))
            ]
        ).sum()

    @property
    def weights(self: Spectrum) -> Array:
        """
        Gets the relative spectral weights by evaluating the polynomial function at the
        internal wavelengths. Output weights are automatically normalised to a sum of
        1.

        Returns
        -------
        weights : Array
            The normalised relative weights of each wavelength.
        """
        weights = vmap(self._eval_weight)(self.wavelengths)
        return weights / weights.sum()

    def normalise(self: Spectrum) -> Spectrum:
        """
        Calculated weights are automatically normalised, so this method simply returns
        an unmodified object.

        Returns
        --------
        spectrum : Spectrum
            The unmodified spectrum object
        """
        return self
