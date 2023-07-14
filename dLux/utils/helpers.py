from collections import OrderedDict

__all__ = ["list_to_dictionary"]


def list_to_dictionary(
    list_in: list, ordered: bool, allowed_types: tuple = ()
) -> dict:
    """
    Converts some input list of dLux layers and converts them into an
    OrderedDict with the correct structure, ensuring that all keys are unique.

    Parameters
    ----------
    list_in : list
        The list of dLux Layers to be converted into a dictionary.
    ordered : bool
        Whether to return an ordered or regular dictionary.
    allowed_types : tuple
        The allowed types of layers to be included in the dictionary.

    Returns
    -------
    dictionary : dict
        The equivalent dictionary or ordered dictionary.
    """
    # Construct names list and identify repeats
    names, repeats = [], []
    for item in list_in:
        # Check for specified names
        if isinstance(item, tuple):
            item, name = item
        else:
            name = item.__class__.__name__

        # Check input types
        if allowed_types != () and not isinstance(item, allowed_types):
            raise TypeError(
                f"Item {name} is not an allowed type, got " f"{type(item)}"
            )

        # Check for Repeats
        if name in names:
            repeats.append(name)
        names.append(name)

    # Get list of unique repeats
    repeats = list(set(repeats))

    # Iterate over repeat names
    for i in range(len(repeats)):
        # Iterate over names list and append index value to name
        idx = 0
        for j in range(len(names)):
            if repeats[i] == names[j]:
                names[j] = names[j] + "_{}".format(idx)
                idx += 1

    # Turn list into Dictionary
    dict_out = OrderedDict() if ordered else {}
    for i in range(len(names)):
        # Check for spaces in names
        if " " in names[i]:
            raise ValueError(f"Names can not contain spaces, got {names[i]}")

        # Add to dict
        if isinstance(list_in[i], tuple):
            item = list_in[i][0]
        else:
            item = list_in[i]
        dict_out[names[i]] = item
    return dict_out
