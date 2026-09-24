---
name: dlux-development
description: Implement, simplify, restructure or review dLux package code and its public numerical contracts. Use for changes to the package itself, including architecture and compatibility decisions.
---

# dLux development

Read [AGENTS.md](../../AGENTS.md), then [contracts.md](contracts.md) for current
ownership and behaviour. Use [style.md](style.md) for implementation layout and
[differentiable-programming.md](differentiable-programming.md) for JAX decisions.
Read [releases.md](releases.md) only for release work.

## Work from a concrete use case

Start with the ordinary user operation and its input/output types. Identify the owner
of sampling, units, coordinates, normalisation and any batch axes. Inspect callers
and existing utilities before adding another representation of the same operation.

Keep research extensions simple: a short documented subclass contract is preferable
to hooks, fallback dispatch, inferred topology and validation intended to handle
arbitrary user implementations. Valid numerical inputs and scientifically meaningful
combinations are caller responsibilities.

Retain useful shape checks. Do not add runtime callbacks, semantic validation trees,
cache invalidation interceptors, automatic aliases or compatibility modes to canonical
classes. Keep constructor assignments ordinary; resolve definitions where their local
inputs become available.

## Resolution and composition

Use the [resolution contract](contracts.md#resolution-and-composition). Ordinary
Modules retain their identity as containers; Expressions produce values when ready.
Resolve consumed operands in their local context and use the existing Unresolved
signal for missing inputs. Do not build another resolver or require every stored
operator/unused field to become an array.

An Expression's `self.resolve()` re-enters its own calculation; it is not a child-
preparation shortcut. Ordinary Module operations can use `self = self.resolve(...)`.
Use `resolve_children` only when preparing all children in that context is intended;
a custom override is not required for each subclass. Coordinate wrappers select their
local frame before evaluating children. Distinct scientific inputs need distinct names,
not a framework to guess the intended coordinate frame.

Populate the shared owner before splitting Linked/Deferred components. The outermost
Zodiax resolve populates links by default; a detached child cannot find sibling owners.
Do not create aliases implicitly.

## Review structure

A base class should own shared behaviour or define a useful extension contract.
Check whether subclassing encodes the actual operation, or just borrows an unrelated
attribute name. In particular, dLux coefficient models do not need Zodiax Transform's
active `x`; basis classes store `coeffs` and use the parametric contract they require.

For each proposed helper or class ask what duplication, invariant or user concept it
owns. Do not introduce a wrapper only to rename an argument or pass through a call.
Do not remove a public extension point solely because it is small. Keep composition
separate from reductions: Sequential passes an output to the next operation; Combine
reduces independently evaluated values; Imager adds Scene components incoherently.

Review source files for:

- duplicated numerical operations or preparation routes;
- unnecessary inheritance, empty markers and legacy methods in canonical classes;
- large functions mixing coordinate preparation, mathematics and packaging;
- unit/shape ownership that changes between consumers;
- private helper chains that hide a simple calculation;
- import cycles, facade shadowing, stale exports and file placement;
- checks or Python abstractions that survive unnecessarily into compiled work.

Make unambiguous local improvements. Record larger changes with the actual file,
example or evidence, consequence and proposed direction. Do not turn a review into
an unrequested new framework or complete backwards-compatibility rewrite.

## Compatibility and validation

Released compatibility belongs in `compat/` or `utils/compat/`. Canonical
classes follow the accepted current API. Prefer constructors that create current
objects over reconstructing old implementations; preserve only explicitly supported
translation paths. Old class identity and private layout need not survive. Check the
untouched 0.15.1 reference before promising a legacy adapter; do not preserve every
intermediate development snapshot.

Update authorised tests around public results, independent references and meaningful
JAX transformations. Run against the checked-out package. Preparation snapshots need
checks before and after explicit re-preparation, not automatic invalidation machinery.
Formatting, tests and coverage are separate evidence; report unresolved gates honestly.

Documentation work and commits require the scope supplied by the user. Do not run
API generators, edit exported tutorials or stage review/agent files as a side effect
of a code review.
