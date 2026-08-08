# Numerical and transformation validation

Choose checks based on the contract:

- fields: shape, dtype, units, coordinates, normalisation, and leading axes;
- propagation: complex field, phase convention, centring, sampling, padding, cropping,
  forward/reverse meaning, and energy where physically applicable;
- parametrics: resolved value, raised paths, broadcasting, gradients, and Hessians
  where meaningful;
- systems and sources: semantic batch axes, intermediate planes, flux ownership, and
  equivalence between scalar and promoted inputs;
- builders: sampled support, basis masking, topology, JIT reuse, and dense/sparse
  agreement within the stated domain;
- compatibility: warning category, migration text, legacy result, and removal version.

Use tolerances justified by precision and algorithm. Do not silently loosen a test to
hide a convention mismatch. A reference comparison should state which implementation
defines the expected physics and which differences are intentionally allowed.

Eager execution does not establish compatibility with `jit`, `vmap`, gradients,
Hessians, or nested transformations. Exercise the complete public object path for
every transformation the API promises and check both values and output structure.
