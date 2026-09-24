# Differentiable programming in dLux

JAX owns numerical arrays and transformations, Equinox owns immutable PyTrees and
filtered transforms, and Zodiax owns named parameter paths, shared links and numerical
definitions. Use these tools directly instead of wrapping them in another framework.

## Leaves and construction

Use `jax.numpy as np` everywhere in package code, including validation. Convert numeric
leaves with `zdx.as_array`; preserve Expressions until their local inputs are available.
Dtype requests use Python categories, not explicit precision widths. JAX precision is
configured by the caller at initialisation.

Trainable or runtime-varying numbers should be arrays. Python floats, booleans, strings
and integer leaves inside tuples are static under `eqx.filter_jit`, which recompiles
when their values change. That is useful for fixed topology and options, but
not for parameters users expect to fit. An ordinary Python field is often sufficient;
`eqx.field(static=True)` instead makes the value part of the PyTree definition,
not an ordinary leaf. It is not a general `frozen=True` feature: constructor checks
and immutable tree replacement have different roles. Current Zodiax rejects arrays
inside explicitly static fields, including nested containers. Never make a JAX array static to work around tracing, indexing or printing problems.

Use ordinary constructor assignment and inherited immutable updates. Validate useful
shapes and exclusive choices there. For concrete-value checks that might see tracers,
distinguish concrete setup from array evaluation. Sampling topology such as a
BandPass range must be concrete, while building an Interp inside a JIT routine can be
useful. Do not impose a blanket ban on constructors inside JIT or scatter Tracer checks
without establishing which check needs concrete values. Keep Python value tests out
of compiled execution; do not add `error_if`, callbacks or a
second validation protocol. Do not promise that later `.set()` values are revalidated.

## Evaluation

Separate shape/topology decisions from array arithmetic. Fixed Python loops over a
small number of operations unroll during tracing and can be clearer than scan/vmap
machinery. Use vmap when independent batches need mapping; use scan when a substantial
sequential operation actually benefits. Benchmark before claiming a faster strategy.

Follow the [resolution contract](contracts.md#resolution-and-composition).
Prepare child definitions using the context they consume, then write the calculation
as ordinary array operations. Do not eagerly resolve a spatial/spectral definition in
an unrelated parent's coordinate frame. Shared Linked values must be populated from
the containing owner before branches are evaluated separately.

Keep physical axis names distinct from array order. Use Wavefront's shared methods for
wavelength, population and Jones-axis alignment. Compare vectorised calculations with
independent scalar evaluations; matching shape alone cannot detect accidentally paired
axes of equal length.

Prepared interpolators and propagators are explicit snapshots. Re-prepare after
changing their inputs. Put preparation inside the differentiated function when fitting
those inputs; otherwise gradients through the frozen snapshot cannot recover them.

## Validation

Test the full public path used by a model, rather than only a private kernel. Eager/JIT
agreement should compare values and structure. Gradient checks should exercise meaningful
sensitivity; finite zero gradients through a hard mask do not demonstrate a useful
fitting contract. Use finite differences or an analytic derivative where appropriate.

Do not demand every transformation for every input. Add Hessian/nested-transform cases
where the science or a demonstrated failure requires them. Normalisation and very large
physical spectral values deserve particular attention to numerical scale. State the
precision, algorithm and tolerance used; do not broaden tolerances to conceal an
unexplained disagreement.
