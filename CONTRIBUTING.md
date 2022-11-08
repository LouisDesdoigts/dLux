# Contributing Guide!

---

## Typing conventions
So we have worked out a general typing convetion for the software. There are two main data-types we care about, Jax data types and others. We denote any jax-types with the `Array` typing hint. This is simply defined as as a`jax.numpy.ndarray`, so at the start of any files one should have `Array = np.ndarray`. *Any* jax arrays should use this type-hit, regardless of data-type. All other type-hints for dLux type objects should refer to the *base* object type, ie:
> dLux.wavefronts.Wavefront
> dLux.propagators.Propagator
> dLux.optics.OpticalLayer
> dLux.base.Instrument

## Units:
All units wihin dLux are SI!

## Non-Circular imports
To prevent circular imports, you can not import any specific classes or methods within the package, you instead need to import the whole packge (ie `import dLux`) and then in the code, refer to specific function (ie `dLux.sources.Source`). This is inconvenient, but just the way python works.

> exception: The utils package
> The utils package is the collection of functions that operate *independently* of the package, and so so can be imported as normal, ie `from dLux.utils.coordinates import radians_to_arcseconds` etc.

## Adding to doccumentation
When adding to the doccumentation, there are a few steps that need to be done:
1. Create the class or function and add the docstring
2. Create a .md file in the docs/ directory
3. Point to the class/function as so `::: dLux.module.new_class`
4. Edit the mkdocs.yml file to add the new class
5. Run `mkdocs serve` to deploy the docs locally at `localhost:8000` (put this into a browser)

## Building and running unit-tests
Any added functionality should be unit-tested! This is done with pytest in the tests/ directory. Please have a look at the testing module and try to create tests the match the style!