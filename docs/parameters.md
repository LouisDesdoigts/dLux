# Parameters and immutable updates

dLux models are Equinox pytrees with Zodiax parameter paths. A path selects a nested
leaf using dot-separated names, while dLux raises useful child attributes so common
parameters can often be reached without spelling every intermediate object.

## Inspect parameters

Use direct attributes while exploring a model and `get()` when building reusable
parameter selections:

```python
coeffs = optics.pupil.coeffs
coeffs = optics.get("pupil.coeffs")

params = optics.get(
    ["pupil.coeffs", "focus.focal_length"],
    as_dict=True,
)
```

`as_dict=True` retains path names, which is normally the most convenient form for an
optimiser or inference routine. Short raised paths such as `"coeffs"` are allowed when
they resolve from the current object. If several children expose the same name, use a
layer or collection name to make the intended path explicit.

## Update models immutably

Updates return a new object; they do not modify the original:

```python
updated = optics.set("pupil.coeffs", new_coeffs)
updated = optics.set(**{"pupil.coeffs": new_coeffs})
```

The arithmetic methods apply the corresponding operation at selected leaves:

```python
flat = optics.multiply("pupil.coeffs", 0.0)
scaled = optics.multiply("pupil.coeffs", 1e-9)
shifted = source.add("position", offset)
```

`add`, `multiply`, `divide`, `power`, `min`, and `max` follow the same path/value
contract as `set`. Prefer the operation that states the intended initialisation or
constraint rather than manually retrieving and replacing a leaf.

## Select fitted and fixed leaves

An optimisation parameter dictionary is an explicit filter over the model. Include
only fitted leaves; omitted leaves remain fixed:

```python
params = {
    "pupil.coeffs": optics.get("pupil.coeffs"),
    "position": source.get("position"),
    "flux": source.get("flux"),
}
```

Update the relevant model inside the objective, then evaluate the ordinary forward
model. This keeps the parameterisation separate from the optical calculation and
allows different parameter groups to use different scaling or optimisers.

For parameters distributed across several top-level objects, apply the same mapping
to each object and verify that every path was consumed. A shared package helper for
this pattern is planned; until then, keep the helper local to the application and
raise on unused paths rather than silently dropping misspelled parameters.

## Shared and derived parameters

Store a genuinely shared physical parameter once where possible and derive dependent
values during evaluation. Repeating the same value in multiple leaves creates
independent optimisation parameters unless the application updates all paths
together.

Derived arrays, coordinate grids, and sampled bases should not normally appear in the
fitted dictionary. Fit their compact physical parameters and allow the corresponding
parametric or builder contract to evaluate them.

## JAX transformations

Take gradients with respect to the parameter dictionary and reconstruct the model
inside the transformed function:

```python
import equinox as eqx
import jax.numpy as np
import zodiax as zdx


@eqx.filter_value_and_grad
def loss(params, optics, source, data):
    optics = optics.set(**{"pupil.coeffs": params["pupil.coeffs"]})
    source = source.set(
        position=params["position"],
        flux=params["flux"],
    )
    model = optics.model(source)
    z_score = zdx.z_score(model.data, data.data, data.std)
    return np.mean(z_score**2)
```

The keys and pytree structure of `params` are static topology under JIT. Change values
between calls, but construct a new compiled objective when changing the selected path
set or leaf shapes.

See [Getting Started](tutorials/introductory/getting_started.md) for a complete staged
optimisation and the [Zodiax documentation](https://github.com/LouisDesdoigts/zodiax)
for the underlying path and optimiser utilities.
