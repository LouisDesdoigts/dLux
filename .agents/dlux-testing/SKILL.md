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
