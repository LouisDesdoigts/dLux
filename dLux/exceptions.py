attr_dims_message = f"""
I was expecting dimensions that could be broadcast with
{}. Instead I recieved dimensions {} when initialising 
the attribute {}.
"""


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


def validate_attr_dims(
        attr_shape: tuple, 
        correct_shape: tuple, 
        attr_name: str) -> None:
    """
    Confirm that an initialised attribute has the correct shape
    and raise a helpful error if it does not.

    Parameters
    ----------
    attr_shape: tuple
        The shape of the attribute to be initialised.
    correct_shape: tuple
        A shape that meets the necessary conditions of
        the attribute. 
    attr_name: str
        The name of the attribute that is getting 
        initialised
    """
    if attr_shape != correct_shape:
        raise DimensionError(
            attr_dims_message.format(
                correct_shape, attr_shape, attr_name))
        
