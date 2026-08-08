# Differentiable programming in dLux

Treat JAX, Equinox, and Zodiax as one integrated object and transformation model.

- JAX owns arrays, tracing, compilation, vectorisation, autodiff, and functional
  control flow.
- Equinox owns PyTree-based modules, filtered transformations, static topology, and
  immutable object updates.
- Zodiax owns nested parameter paths, raised attributes, parameter selection, and
  optimisation-facing model manipulation.

Before changing a class, identify its differentiable leaves, structural values,
leading-axis contract, and parameter paths. Keep values dynamic unless they determine
topology. Resolve shapes and Python collections before tracing, then compile the
fixed-shape calculation rather than a pass-through wrapper.

Ordinary optical layers implement one monochromatic operation and rely on the shared
application contract for leading axes. Do not expose users to manual batching for
ordinary wavelength, source, or parameter populations.

Test transformations through the public object path. Eager execution alone does not
establish `jit`, `vmap`, gradient, Hessian, or nested-transformation compatibility.
Normalisation and stochastic operations require particular care because a numerically
reasonable expression can represent the wrong derivative or statistical model.

Parameter raising should make every meaningful end leaf reachable from the nearest
high-level model. Short paths are the goal; ambiguity is acceptable and is normally
resolved through layer or collection names. Preserve consistent errors across direct
attribute access and Zodiax path operations.
