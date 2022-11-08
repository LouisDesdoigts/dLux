from __future__ import annotations
import jax.numpy as np
from jax.tree_util import tree_map
from equinox import tree_at, Module, is_array, filter as eqx_filter
from optax import adam, multi_transform
from typing import Union, NewType, Any, Callable
from abc import ABC
import dLux


# Types
Array  = np.ndarray
Pytree = NewType("Pytree", object)
Leaf   = Any


###############
### Classes ###
###############
class Base(ABC, Module):
    """
    An abstract base class that is used to give a user-friendly API for working
    with PyTrees, specifically using Equniox. This can be thought of as
    extending the equinox.Module class.

    If you are unfamiliar with PyTrees, check out this jax tutorial and have a
    look a the equinox docs to see how they can be extending to give
    object-oriented jax:
     - https://jax.readthedocs.io/en/latest/jax-101/05.1-pytrees.html
     - https://docs.kidger.site/equinox/

    New concept: Union[str, list]s
        In order to make working with PyTrees easier, there is one concept that
        is introduced, the paths object. A path object is not a unique or new
        class of object type, just a helpfull way of thinking about navigating
        PyTrees. We define a 'path' here as string. Each path object refers to
        a unique 'leaf' in the PyTree. Each nested item should be joined with a
        dot ie '.', as if you were trying to access a class attribute. For the
        case of lists or tuples, you still access them with the 'dot' method.
        Here are some example path objects:
            path1 = 'object.param.value'
            path2 = 'p1.p2.p3'
            path3 = 'item.2.4.value'

    This functionality can be futher extended (as is done with many of the dLux
    classes) using the __getattr__ method, allowing classes to access attributes
    using the . method, which also translates to paths.

    This class implements a series of concrete methods designed to match the
    jax.numpy.ndarray.at[]. methods ie:

        - pytree.get()
        - pytree.set()
        - pytree.add()
        - pytree.multiply()
        - pytree.divide()
        - pytree.power()
        - pytree.min()
        - pytree.max()
        - pytree.apply()
        - pytree.apply_args()
    """


    ######################
    ### Hidden methods ###
    ######################
    def _get_leaf(self   : Pytree,
                  pytree : Pytree,
                  path   : Union[str, list]) -> Leaf:
        """
        A hidden class desinged to recurse down a pytree following the path,
        returning the leaf at the end of the path.

        Base case: len(path) == 1
            In this case the leaf referred to by the single path entry is
            returned (and hence recursively sent up to the initial call).

        Recursive case: len(path) > 1
            In this case the function takes the PyTree like object referred to
            by the first entry in path, and recursively calls this function
            with this new pytree object and the path without the first entry.

        Parameters
        ----------
        pytree : Pytree
            The pytee object to recurse though.
        path : Union[str, list]
            The path to recurse down.

        Returns
        -------
        leaf : Leaf
            The leaf object specified at the end of the path object.
        """
        key = path[0]
        if hasattr(pytree, key):
            pytree = getattr(pytree, key)
        elif isinstance(pytree, dict):
            pytree = pytree[key]
        elif isinstance(pytree, (list, tuple)):
            pytree = pytree[int(key)]
        else:
            raise ValueError("key: {} not found in object: {}".format(key,
                                                            type(pytree)))

        # Return param if at the end of path, else recurse
        return pytree if len(path) == 1 else self._get_leaf(pytree, path[1:])


    def _get_leaves(self : Pytree, paths : list_like) -> list:
        """
        Returns a list of leaves specified by the paths.

        Parameters
        ----------
        paths : Union[str, list]
            A list/tuple of nested paths. Note path objects can only be
            nested a single time.
        pmap : dict = None
            A dictionary of absolute paths.

        Returns
        -------
        leaves : list
            The list of leaf objects specified by the paths object
        """
        return [self._get_leaf(self, path) for path in paths]


    def _unwrap(self      : Pytree,
                paths     : Union[str, list],
                values_in : list = None,
                pmap      : dict = None) -> list:
        """
        Unwraps the provided paths in to the correct list-based format for the
        _get_leaves and _get_leaf methods, returning a single dimensional list
        of input paths.

        Parameters
        ----------
        paths : Union[str, list]
            A list/tuple of nested paths to unwrap.
        values_in : list = None
            The list of values to be unwrapped.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
        paths, values : list, list
            The list of unwrapped paths or paths and values.
        """
        # Get keys
        keys = pmap.keys() if pmap is not None else []

        # Inititalise empty lists
        paths_out, values_out = [], []

        # Make sure values is list
        values = values_in if isinstance(values_in, list) else [values_in]

        # Repeat values to match length of paths
        values = values * len(paths) if len(values) == 1 else values
        assert len(values) == len(paths), ("Something odd has happened, this "
        "is likely due to a missmatch between the input paths and values.")

        # Iterate over paths and values
        for path, value in zip(paths, values):

            # Recurse and add in the case of list inputs
            if isinstance(path, list):
                new_paths, new_values = self._unwrap(path, [value], pmap)
                paths_out  += new_paths
                values_out += new_values

            # Get the absolute path and append
            elif path in keys:
                paths_out.append(pmap[path])
                values_out.append(value)

            # Union[str, list] must already be absolute
            else:
                paths_out.append(path)
                values_out.append(value)

        # Return
        return paths_out if values_in is None else (paths_out, values_out)


    def _format(self   : Pytree,
                paths  : Union[str, list],
                values : list = None,
                pmap   : dict = None) -> list:
        """
        Formats the provided paths in to the correct list-based format for the
        _get_leaves and _get_leaf methods, returning a single dimensional list
        of input paths, with the 'path map' (pmap) values applied.

        Parameters
        ----------
        paths : Union[str, list]
            A list/tuple of nested paths to unwrap.
        values : list = None
            The list of values to be unwrapped.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
        paths, values : list, list
            The list of unwrapped paths or paths and values.
        """
        # Nested/multiple inputs
        if isinstance(paths, list):

            # If there is nesting, ensure correct dis
            if len(paths) > 1 and values is not None \
                and True in [isinstance(p, list) for p in paths]:
                assert isinstance(values, list) and len(values) == len(paths), \
                ("If a list of paths is provided, the list of values must be "
                 "of equal length.")

            # Its a list - iterate and unbind all the keys
            if values is not None:
                flat_paths, new_values = self._unwrap(paths, values, pmap)
            else:
                flat_paths = self._unwrap(paths, pmap=pmap)

            # Turn into seperate strings
            new_paths = [path.split('.') if '.' in path else [path] \
                         for path in flat_paths]

        # Un-nested/singular input
        else:
            # Get from dict if it extsts
            keys = pmap.keys() if pmap is not None else []
            paths = pmap[paths] if paths in keys else paths

            # Turn into seperate strings
            new_paths = [paths.split('.') if '.' in paths else [paths]]
            new_values = [values]

        # Return
        return new_paths if values is None else (new_paths, new_values)


    ########################
    ### Jax like methods ###
    ########################
    def get(self  : Pytree,
            paths : str,
            pmap  : dict = None) -> Leaf:
        """
        Get the leaf specified by path.

        Parameters
        ----------
        paths : Union[str, list]
            A list/tuple of nested paths to unwrap.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
        leaf, leaves : Any, list
            The leaf or list of leaves specified by paths.
        """
        new_paths = self._format(paths, pmap=pmap)
        values = self._get_leaves(new_paths)
        return values[0] if len(new_paths) == 1 else values


    def set(self   : Pytree,
            paths  : Union[str, list],
            values : Union[Any, list],
            pmap   : dict = None) -> Pytree:
        """
        Set the leaves specified by paths with values.

        Parameters
        ----------
        paths : Union[str, list]
            A path or list of paths or list of nested paths.
        values : Union[Any, list]
            The list of values to set at the leaves specified by paths.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
        pytree : Pytree
            The pytree with leaves specified by paths updated with values.
        """
        new_paths, new_values = self._format(paths, values, pmap)

        # Define 'where' function and update pytree
        get_leaves_fn = lambda pytree: pytree._get_leaves(new_paths)
        return tree_at(get_leaves_fn, self, new_values,
                      is_leaf = lambda leaf: leaf is None)


    def add(self   : Pytree,
            paths  : Union[str, list],
            values : Union[Any, list],
            pmap   : dict = None) -> Pytree:
        """
        Add to the the leaves specified by paths with values.

        Parameters
        ----------
        paths : Union[str, list]
            A path or list of paths or list of nested paths.
        values : Union[Any, list]
            The list of values to add to the leaves specified by paths.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
        pytree : Pytree
            The pytree with values added to leaves specified by paths.
        """
        new_paths, new_values = self._format(paths, values, pmap)
        new_values = [leaf + value for value, leaf in zip(new_values, \
                                                   self._get_leaves(new_paths))]

        # Define 'where' function and update pytree
        get_leaves_fn = lambda pytree: pytree._get_leaves(new_paths)
        return tree_at(get_leaves_fn, self, new_values,
                      is_leaf = lambda leaf: leaf is None)


    def multiply(self   : Pytree,
                 paths  : Union[str, list],
                 values : Union[list, Any],
                 pmap   : dict = None) -> Pytree:
        """
        Multiplies the the leaves specified by paths with values.

        Parameters
        ----------
        paths : Union[str, list]
            A path or list of paths or list of nested paths.
        values : Union[Any, list]
            The list of values to multiply the leaves specified by paths.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
        pytree : Pytree
            The pytree with values multiplied by leaves specified by paths.
        """
        new_paths, new_values = self._format(paths, values, pmap)
        new_values = [leaf * value for value, leaf in zip(new_values, \
                                                   self._get_leaves(new_paths))]

        # Define 'where' function and update pytree
        get_leaves_fn = lambda pytree: pytree._get_leaves(new_paths)
        return tree_at(get_leaves_fn, self, new_values,
                      is_leaf = lambda leaf: leaf is None)


    def divide(self   : Pytree,
               paths  : Union[str, list],
               values : Union[list, Any],
               pmap   : dict = None) -> Pytree:
        """
        Divides the the leaves specified by paths with values.

        Parameters
        ----------
        paths : Union[str, list]
            A path or list of paths or list of nested paths.
        values : list
            The list of values to divide the leaves specified by paths.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
        pytree : Pytree
            The pytree with values divided by leaves specified by paths.
        """
        new_paths, new_values = self._format(paths, values, pmap)
        new_values = [leaf / value for value, leaf in zip(new_values, \
                                                   self._get_leaves(new_paths))]

        # Define 'where' function and update pytree
        get_leaves_fn = lambda pytree: pytree._get_leaves(new_paths)
        return tree_at(get_leaves_fn, self, new_values,
                      is_leaf = lambda leaf: leaf is None)


    def power(self   : Pytree,
              paths  : Union[str, list],
              values : Union[list, Any],
              pmap   : dict = None) -> Pytree:
        """
        Raises th leaves specified by paths to the power of values.

        Parameters
        ----------
        paths : PAth
            A path or list of paths or list of nested paths.
        values : list
            The list of values to take the leaves specified by paths to the
            power of.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
        pytree : Pytree
            The pytree with the leaves specified by paths raised to the power
            of values.
        """
        new_paths, new_values = self._format(paths, values, pmap)
        new_values = [leaf ** value for value, leaf in zip(new_values, \
                                                   self._get_leaves(new_paths))]

        # Define 'where' function and update pytree
        get_leaves_fn = lambda pytree: pytree._get_leaves(new_paths)
        return tree_at(get_leaves_fn, self, new_values,
                      is_leaf = lambda leaf: leaf is None)


    def min(self   : Pytree,
            paths  : Union[str, list],
            values : Union[list, Any],
            pmap   : dict = None) -> Pytree:
        """
        Updates the leaves specified by paths with the minimum value of the
        leaves and values.

        Parameters
        ----------
        paths : Union[str, list]
            A path or list of paths or list of nested paths.
        values : list
            The list of values to take the minimum of and the leaf.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
        pytree : Pytree
            The pytree with the leaves specified by paths updated with the
            minimum value of the leaf and values.
        """
        new_paths, new_values = self._format(paths, values, pmap)
        new_values = [np.minimum(leaf, value) for value, leaf in \
                                  zip(new_values, self._get_leaves(new_paths))]

        # Define 'where' function and update pytree
        get_leaves_fn = lambda pytree: pytree._get_leaves(new_paths)
        return tree_at(get_leaves_fn, self, new_values,
                      is_leaf = lambda leaf: leaf is None)


    def max(self   : Pytree,
            paths  : Union[str, list],
            values : Union[list, Any],
            pmap   : dict = None) -> Pytree:
        """
        Updates the leaves specified by paths with the maximum value of the
        leaves and values.

        Parameters
        ----------
        paths : Union[str, list]
            A path or list of paths or list of nested paths.
        values : list
            The list of values to take the maximum of and the leaf.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
        pytree : Pytree
            The pytree with the leaves specified by paths updated with the
            maximum value of the leaf and values.
        """
        new_paths, new_values = self._format(paths, values, pmap)
        new_values = [np.maximum(leaf, value) for value, leaf in \
                                  zip(new_values, self._get_leaves(new_paths))]

        # Define 'where' function and update pytree
        get_leaves_fn = lambda pytree: pytree._get_leaves(new_paths)
        return tree_at(get_leaves_fn, self, new_values,
                      is_leaf = lambda leaf: leaf is None)


    def apply(self  : Pytree,
              paths : Union[str, list],
              fns   : Union[list, Callable],
              pmap  : dict = None) -> Pytree:
        """
        Applies the functions within fns the leaves specified by paths.

        Parameters
        ----------
        paths : Union[str, list]
            A path or list of paths or list of nested paths.
        fns : Union[list, Callable]
            The list of functions to apply to the leaves.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
        pytree : Pytree
            The pytree with fns applied to the leaves specified by paths.
        """
        new_paths, new_fns = self._format(paths, fns, pmap)
        new_values = [fn(leaf) for fn, leaf in zip(new_fns, \
                                               self._get_leaves(new_paths))]

        # Define 'where' function and update pytree
        get_leaves_fn = lambda pytree: pytree._get_leaves(new_paths)
        return tree_at(get_leaves_fn, self, new_values,
                      is_leaf = lambda leaf: leaf is None)


    def apply_args(self  : Pytree,
                   paths : Union[str, list],
                   fns   : Union[list, Any],
                   args  : Union[list, tuple],
                   pmap  : dict = None) -> Pytree:
        """
        Applies the functions within fns the leaves specified by paths, while
        also passing in args to the function.

        Parameters
        ----------
        paths : Union[str, list]
            A path or list of paths or list of nested paths.
        fns : Union[list, Callable]
            The list of functions to apply to the leaves.
        args : Union[list, tuple]
            The tupe or list of tuples of extra arguments to pass into fns.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
        pytree : Pytree
            The pytree with fns applied to the leaves specified by paths with
            the extra args passed in.
        """
        new_paths, new_fns = self._format(paths, fns, pmap)
        new_paths, new_args = self._format(paths, args, pmap)
        new_values = [fn(leaf, *args) for fn, args, leaf in zip(new_fns, \
                                        new_args, self._get_leaves(new_paths))]

        # Define 'where' function and update pytree
        get_leaves_fn = lambda pytree: pytree._get_leaves(new_paths)
        return tree_at(get_leaves_fn, self, new_values,
                      is_leaf = lambda leaf: leaf is None)


class ExtendedBase(Base):
    """
    This class extends the Base class, with the goal of giving access to a
    seris of functions designed to help interface with a series of jax-based
    optimisation packages such as optax and numpyro.
    """


    #########################
    ### Equinox functions ###
    #########################
    def get_args(self  : Pytree,
                 paths : Union[str, list],
                 pmap  : dict = None) -> Pytree:
        """
        Returns an 'args' object, to be used in conjunction with the Equinox
        filter functions. 'args' is a Pytree with a matching tree strucutre,
        but with boolean values at the leaves. This is primarily used by the
        Equinox.filter_grad() and Equinox.filter_value_and_grad() functions,
        passed in as the optional 'arg' argument. It is used to either turn on
        or off gradient calculations with respect to each leaf.

        Parameters
        ----------
        paths : Union[str, list]
            A path or list of paths or list of nested paths.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
        args : Pytree
            An pytree of matching structre with boolean values at the leaves.
        """
        args = tree_map(lambda _: False, self)
        paths = paths if isinstance(paths, list) else [paths]
        values = len(paths) * [True]
        return args.set(paths, values, pmap)


    #######################
    ### Optax functions ###
    #######################
    def get_param_spec(self     : Pytree,
                       paths    : Union[str, list],
                       groups   : Union[str, list],
                       get_args : bool = False,
                       pmap     : dict = None) -> Pytree:
        """
        Returns 'param_spec' object, to be used in conjunction with the
        Optax.multi_transform functions. The param_spec is a pytree of matching
        strucutre that has strings assigned to every node, denoting the group
        that is belongs to. Each of these groups can then have unique optimiser
        objects assigned to them. This is typically used to assign different
        learning rates to different parameters.

        Note this sets the default or non-trainable group to 'null'.

        Parameters
        ----------
        paths : Union[str, list]
            A path or list of paths or list of nested paths.
        groups : Union[str, list]
            A string or list of strings, denoting which group to assign the
            corresponding leaves denoted by paths to.
        get_args : bool = False
            Return a corresponding args pytree or not.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
        param_spec : Pytree
            An pytree of matching structre with string values at the leaves
            specified by groups.
        """
        param_spec = tree_map(lambda _: "null", self)
        param_spec = param_spec.set(paths, groups, pmap)

        return param_spec if not get_args \
            else (param_spec, self.get_args(paths, pmap))


    def get_optimiser(self       : Pytree,
                      paths      : Union[str, list],
                      optimisers : Union[optax.GradientTransformation, list],
                      get_args   : bool = False,
                      pmap       : dict = None) -> tuple:
        """
        Returns an Optax.GradientTransformion object, with the optimisers
        specified by optimisers applied to the leaves specified by paths.

        Parameters
        ----------
        paths : Union[str, list]
            A path or list of paths or list of nested paths.
        optimisers : Union[optax.GradientTransformation, list]
            A optax.GradientTransformation or list of
            optax.GradientTransformation objects to be applied to the leaves
            specified by paths.
        get_args : bool = False
            Return a corresponding args pytree or not.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
        (optimiser, state) : tuple
            A tuple of (Optax.GradientTransformion, optax.MultiTransformState)
            objects, with the optimisers applied to the leaves specified by
            paths, and the initialised optimisation state.
        """
        # Construct groups and get param_spec
        groups = [str(i) for i in range(len(optimisers))]
        param_spec = self.get_param_spec(paths, groups, pmap=pmap)

        # Generate optimiser dictionary
        opt_dict = dict([(groups[i], optimisers[i]) \
                         for i in range(len(groups))])

        # Assign the null group
        opt_dict["null"] = adam(0.0)

        # Get optimiser object
        optim = multi_transform(opt_dict, param_spec)

        # Get filtered optimiser
        opt_state = optim.init(eqx_filter(self, is_array))

        return (optim, opt_state) if not get_args \
            else (optim, opt_state, self.get_args(paths, pmap))


    #########################
    ### Numpyro functions ###
    #########################
    def update_and_model(self      : Pytree,
                         model_fn  : str,
                         paths     : Union[str, list],
                         values    : Union[Any, list],
                         pmap      : dict = None,
                         **kwargs) -> Any:
        """
        Updates the leaves speficied by paths with values, and then calls the
        function specified by the string model_fn, returning whatever is
        returnd by the model_fn. Any extra positional arguments or key-word
        arguments are passed through to the modelling function.

        This function is desigend to be used in conjunction with numpyro.
        Please go through the 'Pytree interface' tutorial to see how this
        is used.

        Parameters
        ----------
        model_fn : str
            A string specifying which model function to call.
        paths : Union[str, list]
            A path or list of paths or list of nested paths.
        values : Union[Any, list]
            The list of values to set at the leaves specified by paths.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
         : Any
            Whatever object is returned by model_fn.
        """
        return getattr(self.set(paths, values, pmap), model_fn)(**kwargs)



    #######################
    ### Other Functions ###
    #######################
    def apply_and_model(self      : Pytree,
                        model_fn  : str,
                        paths     : Union[str, list],
                        fns       : Union[Callable, list],
                        pmap      : dict = None,
                        **kwargs) -> object:
        """
        Applies the functions specified by fns to the leaves speficied by
        paths, and then calls the function specified by the string model_fn,
        returning whatever is returnd by the model_fn. Any extra positional
        arguments or keyword arguments are passed through to the modelling
        function.

        Parameters
        ----------
        model_fn : str
            A string specifying which model function to call.
        paths : Union[str, list]
            A path or list of paths or list of nested paths.
        fns : Union[Callable, list]
            The list of functions to apply to the leaves.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
         : Any
            Whatever object is returned by model_fn.
        """
        return getattr(self.apply(paths, fns, pmap), model_fn)(**kwargs)


    # Method 1, get first
#     def __getattr__(self, key):
#         """

#         """
#         # Found it, nice work
#         dict_like = self.__dict__
#         if key in dict_like.keys():
#             return dict_like[key]

#         # Expand and iterate though items
#         for value in dict_like.values():

#             # Dictionary, call the recursive method
#             if isinstance(value, dict):
#                 try:
#                     return _recurse_dict(value, key)
#                 except ValueError as e:
#                     pass

#             # dLux object, recurse
#             if isinstance(value, Base):
#                 try:
#                     return getattr(value, key)
#                 except ValueError as e:
#                     pass

#         # Not found, raise error
#         raise ValueError("'{}' object has no attribute '{}'"\
#                              .format(type(self), key))


#     def _recurse_dict(self, dict_like, key):
#         """

#         """
#         # Return item if it exists
#         if key in dict_like:
#             return dict_like[key]

#         # Iterate through values
#         for value in dict_like.values():

#             # Value is a dict, Recurse
#             if isinstance(value, dict):
#                 try:
#                     return self._recurse_dict(value, key)
#                 except ValueError as e:
#                     pass

#             # Value is a dLux object, recall the getattr method
#             if isinstance(value, Base):
#                 try:
#                     return getattr(value, key)
#                 except ValueError as e:
#                     pass

#         # Nothing found, raise Error
#         raise ValueError("'{}' not found.".format(key))


#     # Method 2, get all
#     def __getattr__(self, key):
#         """

#         """
#         return self._get_all(key, [])


#     def _get_all(self, key, values):

#         # Found it, nice work
#         dict_like = self.__dict__
#         if key in dict_like.keys():
#             values.append(dict_like[key])

#         # Expand and iterate though items
#         for value in dict_like.values():

#             # Dictionary, call the recursive method
#             if isinstance(value, dict):
#                 # values.append(self._recurse_dict(value, key, values))
#                 values = self._recurse_dict(value, key, values)

#             # dLux object, recurse
#             if isinstance(value, Base):
#                 # values.append(value._get_all(key, values))
#                 values = value._get_all(key, values)

#         return values


#     def _recurse_dict(self, dict_like, key, values):
#         """

#         """
#         # Return item if it exists
#         if key in dict_like:
#             values.append(dict_like[key])

#         # Iterate through values
#         for value in dict_like.values():

#             # Value is a dict, Recurse
#             if isinstance(value, dict):
#                 # values.append(self._recurse_dict(value, key, values))
#                 values = self._recurse_dict(value, key, values)

#             # Value is a dLux object, recall the getattr method
#             if isinstance(value, Base):
#                 # values.append(value._get_all(key, values))
#                 values = value._get_all(key, values)

#         return values