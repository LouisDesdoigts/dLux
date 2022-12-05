# Contributing Guide!

---
---

## Typing conventions
So we have worked out a general typing convetion for the software. There are two main data-types we care about, Jax data types and others. We denote any jax-types with the `Array` typing hint. This is simply defined as as a`jax.numpy.ndarray`, so at the start of any files one should have `Array = np.ndarray`. *Any* jax arrays should use this type-hit, regardless of data-type. All other type-hints for dLux type objects should refer to the *base* object type assuming it is not defined inside that script, ie:
> dLux.wavefronts.Wavefront
> dLux.propagators.Propagator
> dLux.optics.OpticalLayer
> dLux.base.Instrument

For classes defined in the script, add `from __future__ import annotations` to the first line of the script in order to be able to reference these types within the script.


---
## Class Inheritance
dLux classes are built from [`Equinox`](https://github.com/patrick-kidger/equinox) and [`Zodiax`](https://github.com/LouisDesdoigts/zodiax). Any new class types should inherit from either the `zodiax.Base` or `zodiax.ExtendedBase` classes, which inherit from `equinox.Module`, giving the full functionality from both of these packages.

---
## Code Style & Formatting

> All imported functionality should inport the specific method used, as opposed to an full package.

For example rather than importing whole package:
```python
import abc

class Foo(abc.ABC):
    pass
```

We want to specfically import the function used:
```python
from abc import ABC

class Foo(ABC):
    pass
```

> Functions and methods with more than two input should spread those inputs over multiple lines, using spaces to format alignment of type hints and default values to help with code readability

For example rather than this:
```python
def Foo(self: Foo, a: Array = None, b: Array=np.ones(2), c: int=2) -> float:
    pass
```

Format like this:
```python
def Foo(self : Foo,
        a    : Array = None,
        b    : Array = np.ones(2),
        c    : int   = 2) -> float:
    pass
```

### Constructors
> Data type enforcing for jax-type arrays: Constructors should be able to take in lists, numpy arrays and python floats, but ensure that they are correctly formatted into jax type arrays like follows:

```python
self.parameter = np.asarray(input_parameter, dtype=float)
```

This ensures that there is flexibilty in the inputs types and that all data type will match the defauly jax 32 or 64 bit types.


> All constructor methods should use assert statements to ensure that inputs are correctly formatted at init time. The goal should be if no errors are thrown at init time then all the class methods should work as intended and have correct dimensionality. For ∂Lux this would typically be testing jax-type and array dimensionality. Do be sure that these enforce correct dimensionality, not just that errors are not thrown.

For example if an attribute is a scalar, most methods will work identically if it is zero or one dimenional. For example, enforcing zero-dimensionality for scalars:
```python
input_parameter = [1e3]

self.parameter = np.asarray(input_parameter, dtype=float)
assert self.parameter.ndim == 0, \
("input_parameter must a scalar array, ie zero dimensional")
```

### Non-differentiable parameters
> By default, all parameters should be jax-types. However, parameters that define array shapes or logical flow can be python types (ie int, bool). This prevents issues with jax tracing through arrays of integers which can cause jax errors at runtime.

### Internal logic flow
> Most internal logic within ∂Lux should be acheived using regular python logic. In some rare cases, `jax.lax.cond` should be used, but *only* if the internal logical flow can change during run time. Ie any logic that uses a boolean class attribute can not change during run time, and so should use regular python logic. This helps jax trace through the code and reduces compile time.

### Setter and Getters
> By default, getters and setter should be avoided. Most getting and setting can be acheived with nice syntax using the `Zodaix` methods. There are however some exceptions.

> Setter: Some class types such as `Wavefronts` track a lot of parameters that change a lot throughout runtime and so the setter methods are used in conjunction with assert statements to ensure that parameters are not set incorrectly and that the errors are raised at the point where the issue arises, as opposed to some other method recieving an incorrectly shaped array.

> Getters: For some classes we want to have parameterised values, ie binary stars using sepration and field angle. However, optical modelling methods use cartesian position inputs, so the base `Source` class implements an abstract `.get_position()` method to be overwritten by all child classes. In the case of the `BinarySource` class this method uses the separation and field angle to generate these cartesian values at run time. This means that all other classes in ∂Lux can assume cartesian positional values and use a single method when working with any `Source` class. Futhermore, getters can be used to generate useful values. For example wavefronts are store pixel scale and npixles, but implement a class property method for the diameter, which is much more usefull for the `Propagator` classes.


### Hidden methods
> In general hidden class methods should be avoided and classes should try to implement methods that allow for them to used stand-alone. For example the `CompoundAperture` class implements the `construct_combined_aperture` and `get_aperture` methods that allow for users to dirctly construct and output the internally stored apertures, or use the `make_aperture` method to construcut individual apertures.


---
## Units:
All units wihin dLux are SI!

---
## Non-Circular imports
To prevent circular imports, you can not import any specific classes or methods within the package, you instead need to import the whole packge (ie `import dLux`) and then in the code, refer to specific function (ie `dLux.sources.Source`). This is inconvenient, but just the way python works.

> exception: The utils package
> The utils package is the collection of functions that operate *independently* of the package, and so so can be imported as normal, ie `from dLux.utils.coordinates import radians_to_arcseconds` etc.

---
## Adding to doccumentation
When adding to the doccumentation, there are a few steps that need to be done:
1. Create the class or function and add the docstring
2. Create a .md file in the docs/ directory
3. Point to the class/function as so `::: dLux.module.new_class`
4. Edit the mkdocs.yml file to add the new class
5. Run `mkdocs serve` to deploy the docs locally at `localhost:8000` (put this into a browser)

---
## Building and running unit-tests
Any added functionality should be unit-tested! This is done with pytest in the tests/ directory. Please have a look at the testing module and try to create tests the match the style!

There are three main things we want to test:
1. Constructor assert statements and input types. Ensure that the correct error is thrown and that classes cant be constructed with non compatiable inputs.
2. Test that the different run time logic branches evaluate.
3. Test that the different run time logic branches do not return nan or infinite values.