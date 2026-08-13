---
name: dlux-testing
description: Design, add, review, or diagnose tests for dLux, including numerical optics, public behaviour, JAX transformations, gradients, vectorisation, compatibility, coverage, and regression validation. Do not use merely to run an existing test command without interpreting results.
---

# dLux testing

Read the repository `AGENTS.md` and `validation.md`, then establish the public contract
being tested.

Test behaviour rather than implementation structure. Prefer complete outputs and
physically meaningful invariants over individual intermediate expressions. Select the
smallest representative cases that expose units, axes, leading dimensions,
normalisation, sampling, and parameter behaviour.

Do not invent analytic or broadly labelled "physics correctness" regressions without
a defined public contract, established convention, or demonstrated failure mode. A
test should protect behaviour users rely on; speculative physical expectations make
the suite rigid without necessarily improving correctness.

During exploratory API work, wait for approval before rewriting tests. Once the
contract is accepted:

1. add focused eager tests;
2. add JIT, vectorisation, differentiation, or higher-order checks only where those
   behaviours are promised;
3. compare against analytic results, independent implementations, or preserved
   regressions where appropriate;
4. test actionable validation and compatibility behaviour;
5. run the affected module, then the fast suite, then required slow tests;
6. use coverage to find unexamined contracts, not to justify low-value assertions.

Keep stochastic generation separate from deterministic model evaluation. Fix and
split random keys explicitly, and test distributions or seeded outcomes at the level
the public API guarantees.

Use a representative transformation matrix rather than applying every transformation
to every input. A public numerical path normally needs an eager reference plus the
smallest JIT, vectorisation, or differentiation cases that expose its promised
contract. Add Hessian or nested-transformation checks only where higher derivatives
are scientifically used or especially vulnerable to normalisation and control-flow
choices.

A transformation merely executing is not enough. Where meaningful sensitivity is
promised, check that gradients respond to a representative perturbation rather than
silently accepting an everywhere-zero hard boundary. For batching, compare values
against independently evaluated scalar cases; an output-shape assertion alone does
not establish broadcasting semantics.

Keep compatibility behaviour in the dedicated deprecation tests. Assert the warning
category, removal version, replacement, and before/after migration example as well as
the preserved result.

Run examples and tests against the checked-out package rather than an unrelated
installed release. Verify the interpreter and import path first; use the project
environment with `PYTHONPATH=src` when needed. Never install dependencies into the
system or base Python merely because the intended environment is incomplete.

Treat public documentation examples as lightweight executable contracts. Check that
they run and return the documented container and representative shape without turning
every example into a separate physics regression.
