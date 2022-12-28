bc_attr_dims_message = """
I was expecting dimensions that could be broadcast with
{}. Instead I recieved dimensions {} when initialising 
the attribute {}.
"""


eq_attr_dims_message = """
I was expecting dimensions {} but I recieved dimensions
{} when initialising the attribute {}.
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


def validate_eq_attr_dims(
        attr_shape: tuple, 
        correct_shape: tuple, 
        attr_name: str) -> None:
    """
    Confirm that an initialised attribute has the correct shape
    and raise a helpful error if it does not. Correct in this 
    case implies equality between `attr_shape` and `correct_shape`.

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
            eq_attr_dims_message.format(
                correct_shape, attr_shape, attr_name))


def validate_bc_attr_dims(
        attr_shape: tuple, 
        correct_shape: tuple,
        attr_name: str) -> None:
    """
    Confirm that an initialised attribute has the correct shape
    and raise a helpful error if it does not. Correct in this 
    case is the relaxed "broadcastable" version.

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
    try:
        np.broacdcast_shapes(attr_shape, correct_shape)
    except ValueError:
        raise DimensionError(
            bc_attr_dims_message.format(
                correct_shape, attr_shape, attr_name))
        
