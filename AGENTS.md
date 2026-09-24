# dLux agent guide

This repository develops a research-oriented, differentiable optics package. Code
should be understandable and extendable by a new postgraduate student with limited
Python experience. Prefer visible numerical steps and small, useful contracts over
framework machinery that tries to prevent every possible misuse.

## Working agreement

- Follow the user's current scope. Code, tests, documentation and compatibility can
  be separate passes; permission for one does not imply a rewrite of all four.
- Preserve existing local work. Do not commit or push without an explicit request.
  Do not include agent guidance, review notes or exploratory files in a requested
  code commit unless the user also asks for those files.
- Read the implementation and its consumers before changing an interface. State
  meaningful units, shapes, normalisation and ownership before implementing it.
- Make clear, small improvements directly within an authorised review. Record
  substantial or ambiguous design changes for discussion instead of silently
  broadening the architecture.
- Treat `docs/tutorials/**` as exported output, including its overview/index Markdown.
  Edit notebooks or source indexes in the tutorial repository and export/synchronise;
  never patch those generated pages directly. Module Overviews under `docs/overviews/`
  are separate authored guides.
- Treat accepted development changes as the current API. Do not preserve abandoned,
  unreleased designs merely because an old test, notebook or guide uses them.

## Skills

Read the relevant workflow and its linked references:

- [Development](.agents/dlux-development/SKILL.md): implementation and architecture.
- [Testing](.agents/dlux-testing/SKILL.md): behavioral and numerical validation.
- [Documentation](.agents/dlux-documentation/SKILL.md): user explanations and generation.
- [Usage](.agents/dlux-usage/SKILL.md): models, simulations and inference.

The current ownership and numerical contracts are collected in
[contracts.md](.agents/dlux-development/contracts.md). Check actual source when
working in an area that has changed; report conflicts rather than restoring stale
behaviour. Keep shared contracts in this reference instead of copying them into
all four skills.

## Implementation principles

- Ordinary models inherit `zodiax.Module`. Use Expressions for definitions that can
  resolve to arrays and domain base classes only when they provide a useful shared
  contract. Do not reintroduce a dLux Base alias or empty marker hierarchy.
- Use ordinary assignments in constructors and inherited immutable updates elsewhere.
  No generic field-assignment helpers, custom `.set()` interception, automatic aliases,
  or blanket `**kwargs` constructor forwarding.
- Convert numerical leaves with `zdx.as_array`, preserving Expressions and allowed
  None values. Request Python dtype categories such as `float` or `complex`; precision
  comes from JAX configuration, not per-class precision arguments.
- Use `jax.numpy as np`; do not import ordinary NumPy in package code. Do not add
  `equinox.error_if`, callbacks, or runtime validation frameworks. Useful shape and
  setup checks are welcome; invalid use may fail naturally.
- Keep numerical parameters differentiable and topology explicit. Simple Python
  loops over a small fixed collection are often clearer and trace away under JIT.
- Preserve Equinox printing. Declare meaningful defaults on fields, so ordinary
  settings disappear from the representation without custom printing or pdoc. Keep
  interpolation methods visible even at their constructor default; printing and
  static/JIT choices are separate decisions.
- Prefer `import dLux.utils as dlu` in core. Inside utility subpackages use clear
  relative ownership imports to avoid facade cycles. Utilities normally own numerical
  work, but a hard ban on core imports must not force unnecessary abstractions.
- Reuse existing interpolation, sampling, basis and propagation operations. Keep
  one-use helpers near their caller; a generic helper file is not a destination for
  calculations that already have an owner.

## Validation and reporting

During design, use small local experiments without rewriting tests unless authorised.
Once test updates are requested, migrate them to the accepted API and preserve
meaningful behavioural checks. Do not retain obsolete APIs just to satisfy tests,
blanket-skip failures, or loosen tolerances to hide an unexplained mismatch.

Use the intended project environment and verify the actual import locations. Run
focused checks while editing, then the full suite when the task requests it. Do not repeatedly run the full suite
  during an unsettled design pass. Report
whether sibling checkouts or installed dependency releases were used. Distinguish
passing assertions, coverage gates, and a remote Python-version matrix.

Format and lint changed Python files, then review the result for readability. Do not
run repository-wide formatting that touches unrelated work. Do not regenerate or
build documentation during a code-only task. Keep review notes local unless the user
asks to publish them.
