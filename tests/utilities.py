from __future__ import annotations
from abc import ABC, abstractmethod


class UtilityUser(ABC):
    """
    The base utility class. These utility classes are designed to
    define safe constructors and constants for testing. These
    classes are for testing purposes only.
    """
    utility : Utility


class Utility(ABC):
    """

    """
    def __init__(self : Utility) -> Utility:
        """

        """
        return


    @abstractmethod
    def construct(self : Utility) -> object:
        """

        """
        return