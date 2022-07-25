"""
dLux/dev/source.py
------------------
Sources represent observable objects. In terms of implementation they 
are primarily just glorified containers for the information pertaining
to the state of the system. 
"""

import equinox as eqx
import jax.numpy as np
import abc
import typing


Vector = typing.TypeVar("Vector")
Matrix = typing.TypeVar("Matrix")
Tensor = typing.TypeVar("Tensor")
Source = typing.TypeVar("Source")


class Source(eqx.Module, abc.ABC):
    """
    Models an astrophyical target.

    Attributes
    ----------
    spectrum : Vector, meters
        The wavelengths that make up a discrete representation of the 
        spectrum. 
    weigths : Vector
        Weights associated with the expression of the different 
        wave-lengths in the spectrum.
    position : Vector
        The position of the source. 
    resolved : bool
        Answers the question, is the source resolved. 
    """
    spectrum : Vector
    weights : Vector
    position : Vector
    resolved : bool
     

    def __init__(self : Source, spectrum : Vector, 
            weights : Vector, position : Vector,
            resolved : bool) -> Source:
        """
        Parameters
        ----------
        spectrum : Vector, meters
            The wavelengths that make up a discrete representation of the 
            spectrum. 
        weigths : Vector
            Weights associated with the expression of the different 
            wave-lengths in the spectrum.
        position : Vector
            The position of the source. 
        resolved : bool
            Answers the question, is the source resolved. 
        """
        self.spectrum = np.asarray(spectrum).astype(float)
        self.weights = np.asarray(weigths).astype(float)
        self.position = np.asarray(position).astype(float)
        self.resolved = bool(resolved)


    def get_spectrum(self : Source) -> Vector:
        """
        Returns
        -------
        spectrum : Vector[float], meters
            The wavelengths that make up the spectrum of the source.
        """
        return self.spectrum


    def get_weigths(self : Source) -> Vector:
        """
        Returns 
        -------
        weights : Vector[float]
            The weight associated elementwise with each of the 
            wavelengths in the spectrum. 
        """
        return self.weigths


    def get_position(self : Source) -> Vector:
        """
        Returns
        -------
        position : Vector[float]
            The position of the start via some parametrisation. The 
            exact parametrisation is determined by the subclass 
            implementation.
        """
        return self.position


    def is_resolved(self : Source) -> bool:
        """
        Returns
        -------
        resolved : bool
            True if the source is resolved and false otherwise. 
        """
        return self.resolved


class PointSource(Source):
    """
    An astrophysical point source. Examples include, single stars,
    quasars and very distant galaxies. 
    """
    def __init__(self : Source, spectrum : Vector, 
            weights : Vector, position : Vector,
            resolved : bool) -> Source:
        """
        Parameters
        ----------
        spectrum : Vector, meters
            The wavelengths that make up a discrete representation of the 
            spectrum. 
        weigths : Vector
            Weights associated with the expression of the different 
            wave-lengths in the spectrum.
        position : Vector
            The position of the source. 
        resolved : bool
            Answers the question, is the source resolved. 
        """
        super().__init__(spectrum, weights, position, resolved)


class ExtendedSource(Source):
    """
    An extended source. Examples include a star with an orbiting 
    protoplanetary disk or an unresolved galaxy.

    Attributes
    ----------
    fluxes : Matrix
        The relative intensity of the extended source mapped over a 
        pixel grid. 
    """
    fluxes : Matrix 


    def __init__(self : Source, spectrum : Vector, 
            weights : Vector, position : Vector,
            resolved : bool, fluxes : Matrix) -> Source:
        """
        Parameters
        ----------
        spectrum : Vector, meters
            The wavelengths that make up a discrete representation of the 
            spectrum. 
        weigths : Vector
            Weights associated with the expression of the different 
            wave-lengths in the spectrum.
        position : Vector
            The position of the source. 
        resolved : bool
            Answers the question, is the source resolved. 
        fluxes : Matrix
            The relative intensity of the points that make up the 
            extended source.
        """
        super().__init__(spectrum, weights, position, resolved)
        self.fluxes = np.asarray(fluxes).astype(float)


    def get_fluxes(self : Source) -> Matrix:
        """
        Returns
        -------
        fluxes : Matrix[float]
            The relative intensities of the different pixels that make 
            up the extended source distribution.
        """
        return self.fluxes


class CompoundSource(eqx.Module):
    """
    Represents multiple sources. The primary use cases for this 
    class is in modelling binaries.

    Attributes
    ----------
    sources : list[Source]
        The point or extended sources that are getting modelled.
    weights : Vector
        The relative weightings of the different sources in the 
        ensamble. 
    """
    sources : list[Source]
    weights : Vector


    def __init__(self : Source, sources : list) -> Source:
        """
        Parameters
        ----------
        sources : list[Source]
            The sources to model in the compound source.
        """
        self.sources = list(sources)


    def get_sources(self : Source) -> list:
        """
        Returns
        -------
        sources : list
            The sources that are getting modelled. 
        """
        return self.sources


    def get_spectrum(self : Sources) -> Vector: # combine the spectra
        """
        Combine the spectra of the sources that are getting modelled. 
        Note that this operation is not going to have a well defined 
        array size and so is inherently un-jit-able. 

        Returns
        -------
        spectrum : Vector
            The unique spectral elements of the compound sources.
        """
        return np.unique(np.concatenate(
            [source.get_spectrum() for source in self.get_sources()]))


    def get_weigths(self : Sources) -> Vector: # spectral weights of ensamble
        """
        Combined weights of the different wavelengths in the combined 
        spectra of the compound source.

        Returns
        -------
        weights : Vector
            The wieght of each wavelength in the combined source,
            elementwise.
        """
#        spectrum = self.get_spectrum()
#        net_weights = np.zeros(spectrum.shape)
#        total_spectrum = np.concatenate(
#            [source.get_spectrum() for source in self.get_sources()])
#
#        for wavelength in spectrum:
#            np.where(== wavelength)
        # TODO: The implementation of this is non-trivial and I have 
        # elected to leave it until later. 


# TODO: Decide on a parametrisation for the below. 
def Binary(CompoundSource):
    """
    Model of binary stars. 
    """
    def __init__(self : Source, 
            first_star_spectrum : Vector, 
            first_star_spectral_weights : Vector, 
            first_star_position : Vector,
            first_star_resolved : bool,
            second_star_spectrum : Vector,
            second_star_spectral_weights : Vector,
            second_star_position : Vector,
            second_star_resolved : bool) -> Source:
        """
        The documentation below is general and can be applied equally
        to both those parameters labelled first_star and second_star.

        Parameters
        ----------
        spectrum : Vector, meters
            The wavelengths that make up a discrete representation of the 
            spectrum. 
        weigths : Vector
            Weights associated with the expression of the different 
            wave-lengths in the spectrum.
        position : Vector
            The position of the source. 
        resolved : bool
            Answers the question, is the source resolved.
        """ 
        first_star = PointSource(first_star_spectrum, 
            first_star_spectral_weights, first_star_position,
            first_star_resolved)
        second_star = PointSource(second_star_spectrum,
            second_star_spectral_weights, second_star_position,
            second_star_resolved)
        super().__init__([first_star, second_star])    


    def get_separation_vector(self : Source) -> Vector:
        """
        Returns
        -------
        separation : Vector
            The separation of the stars in the binary. 
            The first element is the x coordinate and the   
            second is the y coordinate.
        """           
        first, second = self.get_sources()     
        return first.get_position() - second.get_position()


    def get_separation_polar(self : Source) -> float:
        """
        Returns
        -------
        separation : float
            The total separation of the two stars in the binary.
        """
        separations = self.get_separation_vector()
        return np.array([np.hypot(separations), 
            np.arctan2(separations[1], separations[0])])
