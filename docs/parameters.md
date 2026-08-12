# Parameters, paths, and immutable updates

dLux objects are immutable Equinox pytrees. Their leaves contain physical parameters,
sampled arrays, and static model structure; changing a model means returning a new
tree with selected leaves replaced. Zodiax paths provide a concise way to locate those
leaves inside nested sources, systems, layers, and parametrics.

For example, an optical system may contain a layer named `pupil`, whose OPD is a
`Basis`, whose fitted leaf is `coeffs`:

```text
OpticalSystem
└── pupil: Optic
    └── opd: Basis
        └── coeffs
```

The fully qualified path is `"pupil.opd.coeffs"`. dLux raises useful child attributes,
so `"pupil.coeffs"` and sometimes simply `"coeffs"` can resolve to the same leaf.
Short paths are convenient for simple models; qualified paths remain explicit when
several children expose the same name.

## Inspect parameters

Use direct attributes while exploring a model and `get()` when the path itself is part
of a reusable calculation:

```python
coeffs = optics.pupil.coeffs
coeffs = optics.get("pupil.coeffs")

params = optics.get(
    ["pupil.coeffs", "focus.focal_length"],
    as_dict=True,
)
```

`as_dict=True` retains the requested path names. This is useful when paths define a
fitted parameter set, but `get()` is also a general inspection mechanism and is not
specific to optimisation.

## Update models immutably

`set()` follows the same paths and returns a new object; it never modifies the
original:

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

## Update more than one object

An inference model often has parameters split between an optical system and a source.
`dlu.update` applies one path mapping across several top-level objects:

```python
params = {
    "pupil.coeffs": new_coeffs,
    "position": new_position,
    "flux": new_flux,
}
optics, source = dlu.update(params, optics, source)
```

Strict mode is the default. Objects are checked in positional order, each path is
consumed by its first match, and unused paths raise an error. This catches misspelled
or incorrectly qualified paths. With `strict=False`, matching paths are applied to
every object and unmatched paths are ignored.

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

import dLux.utils as dlu


@eqx.filter_value_and_grad
def loss(params, optics, source, data):
    optics, source = dlu.update(params, optics, source)
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
