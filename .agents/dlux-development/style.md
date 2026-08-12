# dLux style and review reference

Use this reference when writing or reviewing dLux code. The goal is concise code with
visible structure, not mechanical minimalism or formatter-defined layout.

## Algorithmic recipe layout

Organise non-trivial functions into logically distinct blocks. Separate the blocks
with whitespace and introduce them with short comments. Reading the comments alone
should provide a useful outline of the algorithm.

Prefer:

```python
# Generate the oversampled aperture components
fine = grid.oversample(self.oversample)
primary = self._evaluate(self.primary, fine, transform)
eval_fn = lambda s: 1 - self._evaluate(s, fine, transform)
obscurations = [eval_fn(s) for s in self.obscurations]

# Combine and downsample the transmission and support
transmissions = np.stack([primary, *obscurations])
transmission = dlu.downsample(transmissions.prod(0), self.oversample)
support = dlu.downsample(primary, self.oversample) > 0

# Get the physical diameter required for OPD generation
extent = self.primary.extent
if extent is None and self.opd is not None:
    raise ValueError("primary must define an extent when opd is provided.")
diameter = np.asarray(0.0) if extent is None else 2 * extent

# Package the sampled aperture data
return ApertureData(transmission, support, diameter)
```

Avoid combining sampling, composition, validation, and packaging in one uninterrupted
block. Comments should identify the purpose of a stage, not narrate each assignment.

## Visual rhythm

Adjacent lines should have similar visual weight where the calculation permits it.
Reorder independent assignments, shorten repeated access, or introduce a domain
alias to avoid long-short-long patterns.

Avoid:

```python
d = output.spec.d[0] if axes[2] is None else output.spec.d
c = output.spec.c
c = c[0] if c is not None and axes[3] is None else c
```

Prefer:

```python
# Remove axes introduced for grid values that were not vectorised
c = output.spec.c
c = c[0] if c is not None and axes[3] is None else c
d = output.spec.d[0] if axes[2] is None else output.spec.d
```

Aliases must earn their line. Use one when it shortens several later expressions,
names a physical concept, or balances a complete block.

Avoid:

```python
obs = self.global_obscurations
eval_fn = lambda shape: 1 - self._evaluate(shape, fine, transform)
obscurations = [eval_fn(shape) for shape in obs]
```

Prefer:

```python
eval_fn = lambda s: 1 - self._evaluate(s, fine, transform)
obscurations = [eval_fn(s) for s in self.global_obscurations]
```

Short loop variables are appropriate when the collection name and block establish
their meaning.

## Intentional line wrapping

Do not minimise physical line count. Preserve trailing commas in a deliberate
vertical structure:

```python
return np.linspace(
    mean[i] - extent * stds[i],
    mean[i] + extent * stds[i],
    npix_arr[i],
)
```

Keep a complete operation on one line when it fits clearly:

```python
groups.append(ZernikeGroup(indices, ns, ms, np.stack(coeffs), np.stack(k)))
```

Avoid vertically expanding a simple call without exposing useful structure. If an
expression wraps poorly, try this order:

1. shorten repeated concepts with a clear local name;
2. use concise variables in a small comprehension;
3. extract a repeated expression into a local function;
4. keep the semantic operation on one line if it now fits;
5. otherwise use an intentional multiline layout.

For a condition that genuinely wraps, nested checks are easier to scan than a ragged
multiline boolean:

```python
if transformation is not None:
    if not isinstance(transformation, CoordTransform):
        raise TypeError("transformation must be a CoordTransform or None.")
```

Do not nest a condition that already fits comfortably on one line.

## Expose numerical stages

Avoid visually stacking transformation construction and invocation:

```python
convolved = eqx.filter_vmap(convolve_fn)(
    images,
    kernels,
)
```

Prefer a focused mapped function followed by a simple call:

```python
@jax.vmap
def convolve_fn(image, kernel):
    return jsp.signal.convolve(image, kernel, mode="same")

convolved = convolve_fn(images, kernels)
```

For contractions, use concise axis names and expose repeated leaves when this makes
the block denser and more regular:

```python
# Contract the coefficient and basis dimensions
ndim = len(self.shape)
b_ax = tuple(range(ndim))
coeffs = self.coeffs
c_ax = tuple(range(coeffs.ndim - ndim, coeffs.ndim))
weights = np.tensordot(coeffs, self.basis, axes=(c_ax, b_ax))
```

## Constructors and classes

The constructor is the first method in every concrete class. Use a linear conversion,
validation, and assignment flow with whitespace between stages:

```python
def __init__(self, width, angles, edge=None, invert=False):
    super().__init__(edge, invert)
    self.width = dlu.to_value(width)
    self.angles = dlu.to_value(angles)

    if self.width <= 0:
        raise ValueError("width must be greater than zero.")
    if self.angles.ndim != 1:
        raise ValueError("angles must be a one-dimensional array.")
```

Do not hide ordinary semantic validation in field converters, `__check_init__`, or a
chain of generic helpers. A shared converter is useful only when it defines a real
public concept.

Within a class, use this order:

1. class docstring and annotated fields;
2. constructor;
3. derived properties;
4. primary public and magic methods;
5. secondary public methods;
6. private implementation methods.

Document the constructor of every public concrete class. Its docstring must define
every argument, accepted type, shape, unit, default, exclusive combination,
broadcasting rule, validation condition, and important construction-time consequence.
Simple constructors may be concise, but still need a useful `Parameters` section;
"Initialise the object" is never sufficient. Keep conceptual behaviour, physical
contracts, limitations, and complete examples on the class docstring rather than
duplicating them in `__init__`.

Preserve Equinox's default object representation. The printed tree should honestly
show the object's leaves and structure; do not add custom representations, hidden
fields, or declarative ordering machinery merely to make an awkward class design look
cleaner. Improve the object model itself when its default representation is unclear.

## Module organisation

Place module-level functions after imports and `__all__`, before the first class.
Order utility functions by dependency and public workflow; do not scatter free
functions between classes. Keep related functions contiguous rather than scattering
one domain concept across broad hard, soft, validation, or helper sections. Keep
imports at module scope unless an unavoidable annotation-only cycle requires
`TYPE_CHECKING`.

Use `super()` to follow the MRO rather than naming a parent implementation directly.
Prefer concise positional arguments for an established internal signature when the
whole call remains clear. Do not bind a parent method or construct an argument tuple
for one call.

Resolve Python topology before compilation. When optional JIT is part of a builder
contract, transform the bound method that performs the calculation:

```python
build_fn = eqx.filter_jit(self._build) if jit else self._build
return build_fn(grid, transform)
```

Do not create module-level pass-through functions and separately compiled aliases
whose only purpose is to call the real method.

## Abstraction review

An abstraction should define a public extension point, encode a domain concept,
centralise behaviour that must remain identical, or remove meaningful repetition.
Do not extract a helper solely to shorten one caller. Search its call sites first.

Before implementing numerical behaviour, search the complete package for the
operation and its physical synonyms. Reuse the canonical utility even when its
current signature needs a small generalisation. Do not reproduce Gaussian,
interpolation, propagation, coordinate, normalisation, padding, cropping, or basis
evaluation mathematics inside a layer when `dLux.utils` already owns that contract.

For registries and parsers, keep canonical data separate from aliases, prefix rules,
validation, and error presentation. A one-off spelling exception usually indicates
that canonicalisation has been modelled at the wrong level. Lay resolution out as a
short sequence of direct lookup, alias resolution, optional prefix handling, and
dimension validation rather than one nested conditional block.

Keep a private helper beside its owning implementation when it depends on core dLux
objects or represents only that module's structure. Move it into `dLux.utils` only
when it defines a reusable array-oriented numerical contract. Utilities must remain
independent of grids, fields, layers, parametrics, builders, sources, systems, and
other higher-level dLux objects.

Use `super()` for the next implementation in the MRO. Prefer concise positional
arguments for an established internal signature when the complete call stays clear;
use keywords when they materially explain meaning or skip optional positions.

Avoid local imports. If modules appear circular, first remove redundant concrete type
checks, use structural dispatch when the callee owns validation, or move the shared
object into a module that can import both contracts normally.

## Documentation review

Documentation depth follows contract importance:

- substantial public APIs document parameters and returns plus units, shapes, axes,
  leading dimensions, normalisation, limitations, and relevant JAX behaviour;
- public concrete classes document both their overall class contract and constructor;
- simple wrappers, aliases, properties, and obvious transformations use one precise
  sentence;
- abstract methods document the requirements placed on implementations;
- private helpers need detail only for non-obvious numerical or structural contracts.

Do not add a full NumPy-style template that only repeats the signature. Do not leave
a blank line before closing quotes. Use British scientific vocabulary, preserve
existing public identifier spelling, and never claim checks the code does not make.

## Final review

For every modified function, ask:

- Can the algorithm be recovered from the block comments?
- Does each block perform one distinct job?
- Are setup, validation, calculation, and return stages visually separate?
- Are adjacent lines reasonably balanced?
- Did formatter-driven wrapping hide an operation that needs a name?
- Are local names concise without becoming ambiguous?
- Does the code preserve units, axes, leading dimensions, normalisation, and JAX
  behaviour?
- Do numerical names and docstrings distinguish exact quantities from directional,
  approximate, or boundary-coordinate constructions?
- Does it look like the surrounding dLux module and the clearest v0.15 code?
