# dLux agent guide

This file defines the repository-wide working agreement for AI-assisted development
in dLux. It is intentionally vendor-neutral. Use the task-specific material under
`.agents/` when it matches the work; do not load unrelated skills or references.

Available workflows are:

- `dlux-development` for package implementation, API design, compatibility, and releases;
- `dlux-testing` for behavioural, numerical, and transformation validation;
- `dlux-documentation` for docstrings, tutorials, API generation, and UML diagrams;
- `dlux-usage` for building models, instruments, simulations, and inference with dLux.

## Working agreement

- Inspect the affected source, tests, neighbouring modules, and relevant history
  before proposing a new pattern.
- Establish the public and physical contract before changing an API, including units,
  shapes, axes, normalisation, vectorisation, and differentiation where relevant.
- Prefer the smallest coherent change and discuss broad redesigns before implementing
  them. Preserve unrelated user work.
- During exploratory API development, do not update tests until the direction is
  approved. Once approved, test public behaviour rather than implementation details.
- Do not commit unless explicitly asked. Before a requested commit, inspect the full
  diff and run validation proportional to the change.

## Package boundaries

- `dLux.utils` is the independent, functional numerical foundation. It must not
  import the core object model. Internal core code calls utilities through
  `import dLux.utils as dlu`.
- Core modules own fields, grids, sources, and systems. `dLux.parametric` owns
  differentiable or context-resolved values; `dLux.layers` owns transformations
  applied to fields or detector data.
- `builders` owns setup-time construction. `prebuilt` owns named templates composed
  from the general builders and must document their fidelity.
- Declare public objects through module `__all__` exports and document them at source.
- Put module imports at module scope. Resolve unclear ownership instead of hiding
  circular imports inside functions where a cleaner design is available.

## Code and API principles

- Optimise for a new graduate student being able to read, verify, and extend the code.
- Lay non-trivial calculations out as an algorithmic recipe: concise logical blocks,
  separated by whitespace and introduced by useful comments.
- Keep adjacent lines visually balanced. Preserve trailing commas in intentional
  multiline structures, but keep a complete operation on one line when it fits
  clearly.
- Use concise domain names and avoid abstractions that do not encode a shared concept,
  invariant, extension point, or meaningful reuse.
- Put `__init__` first in every concrete class. Keep module-level functions before
  classes and order classes from base contracts to concrete implementations.
- Separate construction-time topology and validation from compiled numerical
  evaluation. Do not add runtime checks or JAX control flow without considering their
  tracing, compilation, and execution costs.
- Preserve units, physical-axis ordering, leading dimensions, normalisation ownership,
  and differentiability explicitly. Never infer a physical inverse from a convenient
  numerical operation without deriving and validating it.
- Use British scientific vocabulary in prose and new APIs while retaining the exact
  spelling of existing public identifiers.

## Documentation and generated files

- Give substantial public APIs complete contracts and concise examples. Keep simple
  wrappers and private helpers brief unless they carry a non-obvious contract.
- Treat API Markdown, UML diagrams, and exported tutorial Markdown as generated files.
  Update docstrings, exports, notebooks, or generators and then regenerate them.
- Keep exploratory notebooks and working Markdown untracked unless publication is
  explicitly requested.

## Validation

- Run focused tests while iterating and the full suite before a requested commit.
- Exercise eager, JIT, vectorised, and differentiated paths only where the public
  contract promises them. Coverage is a diagnostic, not a reason for low-value tests.
- Format with `black .` and lint with `ruff check .`; inspect the result for degraded
  visual layout rather than treating formatter output as the design target.
- Run the fast test suite with `pytest`; include slow tests only when the affected
  contract requires them.
- Build documentation with `zensical build --strict` after changing documentation,
  public exports, generated API pages, or navigation.

Before finishing, review every modified function for clear stages, package ownership,
physical correctness, and consistency with the surrounding dLux code.
