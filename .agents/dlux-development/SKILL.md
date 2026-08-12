---
name: dlux-development
description: Develop, review, or redesign the dLux differentiable optical modelling package, including public APIs, architecture, compatibility, and releases. Use dlux-testing for test design, dlux-documentation for documentation workflows, and dlux-usage for external application code.
---

# dLux development

Read the repository `AGENTS.md` first. Use this skill to reason about how a change
fits the dLux object model and numerical contracts. Read `style.md` before writing or
substantially restructuring code, `differentiable-programming.md` when JAX, Equinox,
or Zodiax behaviour is involved, and `releases.md` for release work.

## Establish the change

1. Inspect the target source, public exports, tests, and generated documentation.
2. Inspect neighbouring implementations and relevant v0.15 code for established
   behaviour and visual style. Preserve the style principles, not obsolete APIs.
3. Describe the public contract in plain language:
   - accepted objects and returned objects;
   - units and coordinate plane;
   - array shape, physical-axis order, and leading-axis meaning;
   - scalar promotion and broadcasting;
   - normalisation ownership;
   - differentiable leaves, static topology, JIT, and vectorisation;
   - approximation or fidelity limits.
4. Search the full package for existing utilities and extension points before adding
   machinery. Compose or extend the canonical implementation instead of rebuilding
   its numerical operation locally.
5. Keep exploratory notebooks and working Markdown untracked unless the user asks to
   publish them.

## Place the implementation

Use the lowest layer that can own the behaviour without importing higher-level dLux
objects:

- `utils`: pure, array-oriented numerical functions with explicit inputs;
- `grids`: sampling specifications and coordinate transforms;
- `fields`: wavefront, intensity, and detector-image data plus physical operations;
- `parametric`: differentiable or context-resolved value generation;
- `layers`: transformations applied to fields or detector data;
- `builders`: setup-time generation of sampled components or layers;
- `prebuilt`: named templates made from general builders, with documented fidelity;
- `sources`: spectral and spatial emission models;
- `systems`: ordered optical and detector orchestration;
- `compatibility`: explicit, warning-backed migration surfaces for released APIs.

Keep `utils` independent from the core object model. Within core modules, call its
functions through `dlu`. Move a helper into `utils` only when it has a meaningful,
array-oriented numerical contract and genuine reuse beyond its original caller. Do
not use `utils` as a holding area for one-use core helpers, and never make it import
grids, fields, layers, parametrics, builders, sources, systems, or other higher-level
dLux objects.

## Preserve object contracts

### Public extension contracts

Keep a base class public when users may reasonably implement custom behaviour at that
level. A class is not private merely because most users instantiate its concrete
children. Export supported extension points deliberately and document the contract
subclasses must implement.

Audit class methods with the same discipline. A method is either a supported public
operation with a complete callable contract or an implementation detail with a
private name. Do not retain redundant convenience routes without a concrete benefit,
and do not privatise useful behaviour merely to avoid documenting it. Check released
history before removing a public method; superseded released routes follow the normal
compatibility and warning policy.

### Fields and grids

Treat physical axes and array axes separately and document their ordering. Preserve
leading dimensions rather than assuming every field is two-dimensional. Let
`GridSpec` and resize specifications own coordinate recovery, cropping, padding, and
sampling metadata where those operations form part of their contract.

### Parametrics

A parametric resolves a value; it does not apply that value to a field or orchestrate
a system. Keep differentiable leaves explicit. Resolve context through established
parameter paths and raise nested leaves so simple models can use short paths.

### Optical layers

Implement the monochromatic transformation in `apply_mono`. `BaseOpticalLayer.apply`
owns recursive leading-axis vectorisation, and `__call__` provides normal callable
syntax. Override `apply` only for complete-field behaviour such as propagation or
semantic-axis consumption such as interference.

Do not force related layer families to use cosmetically identical primitive methods.
Optical layers use `apply_mono` because their base owns wavelength vectorisation;
detector layers operate on a complete `Intensity` through `apply`. Shared inheritance
should encode shared behaviour, not erase meaningful semantic differences.

### Construction and sparse evaluation

Separate construction topology from fixed-shape evaluation. Store compact indices,
starts, residual offsets, or shape specifications rather than large materialised
masks. For repeated geometry, expose parallel and memory-efficient strategies only
when both have a real use case, and document the trade-off concisely.

### Propagation

Treat forward and reverse propagation as physical contracts. Validate sampling,
phase, centring, padding, and cropping against the complete complex field. An inverse
ABCD matrix is not an inverse propagation operation. Prefer one composable propagator
with an explicit supported reverse operation over paired ad hoc classes.

### Sources and spectra

Define wavelength and spectral-weight axes explicitly. Scalar public inputs should
promote consistently. Preserve flux and normalisation ownership across different
wavelength sample counts, and consider how normalisation affects gradients and
higher derivatives.

## Implement for JAX deliberately

- Convert numeric public inputs consistently, then perform semantic validation in a
  readable constructor.
- Treat constructor validation as setup-time feedback, not a persistent invariant:
  immutable updates can replace any leaf. Do not add compiled runtime checks merely
  to defend against later `.set()` calls. Where a complex mutable contract benefits
  from an explicit diagnostic, provide an opt-in `validate()` method and document
  that users decide when to call it.
- Resolve collection lengths, output shapes, mode groups, and other Python topology
  before tracing.
- Keep values dynamic when they do not determine topology.
- Prefer `vmap` for independent regular batches and `scan` when sequential updates
  materially reduce memory. Benchmark representative sizes and backends before
  declaring one universally faster.
- Avoid dynamic-shape workarounds, hidden runtime checks, and duplicated compiled
  wrappers around the real calculation.
- Test the public eager, JIT, gradient, and vectorised routes that the API promises.

## Maintain compatibility intentionally

Only released APIs require compatibility. Keep aliases and migration warnings in the
compatibility layer where possible. A warning must name the old API, the replacement,
an explicit before/after usage example, and the planned removal version. Do not add
empty legacy modules when module-like compatibility can be provided centrally.

## Name APIs deliberately

Do not mechanically regularise the public API. Retain concise scientific acronyms and
established short identifiers unless they create a concrete ambiguity or encode the
wrong contract. Consistency alone is not sufficient reason to lengthen or rename an
API, especially when doing so expands the compatibility surface.

## Prepare releases deliberately

Use `releases.md` for version changes, compatibility audits, migration guides,
changelogs, support matrices, and release validation. Release notes are part of
package development and must cover the full public diff and migration surface.

## Documentation and generated files

Put API documentation in source docstrings. Substantial public classes should include
small examples modelled on Equinox documentation: one complete interaction, without
turning the docstring into a tutorial. Keep private documentation concise unless a
hidden numerical contract needs explanation.

Edit tutorial notebooks, not exported tutorial Markdown. Use the
`dlux-documentation` skill for documentation structure, generated API pages, UML
diagrams, and tutorial publishing.

## Validation workflow

During design, use small notebooks or scripts to expose numerical choices without
committing those artefacts. After approval:

1. add focused behavioural tests;
2. exercise eager and promised JAX transformations;
3. run formatting without surrendering intentional visual layout;
4. run the full suite;
5. inspect the complete diff for generated files and unrelated changes;
6. commit only when explicitly requested.
