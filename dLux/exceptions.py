class DimensionError(Exception):
    """
    This error is raised in the place of a ValueError
    when a user attempts to initialise a dLux object 
    with an attribute that has incorrect or inconsistent
    dimensions. 
    """


    def __init__(self: object, message: str) -> object:
        """
        Parameters
        ----------
        message: str
            A detailed, situational error message that is 
            correctly formatted. 
        """
        super().__init__(message)
