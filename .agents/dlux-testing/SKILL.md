---
name: dlux-testing
description: Update, design or diagnose dLux tests for current public behavior, numerical optics, JAX transformations and regression validation. Use when interpreting or changing tests, not merely invoking an existing command.
---

# dLux testing

Read [AGENTS.md](../../AGENTS.md), [validation.md](validation.md) and the relevant
[current contracts](../dlux-development/contracts.md). Use current code and accepted
user decisions to distinguish a real regression from an obsolete API expectation.

When tests are authorised, organise them by package ownership: fields,
systems (including sources), layers, prebuilt and the utility subpackages. Existing
root source tests need not move as a side effect of a narrow fix. Keep released adapters in
compatibility tests. dLux no longer owns a serialization subsystem; use Zodiax for a
small integration check on representative dLux objects, without reproducing its suite.

Preserve valid behavioural coverage during a migration. Move/rewrite assertions for
consolidated classes, remove tests solely for deliberately removed unreleased APIs,
and add coverage for the new shared contract. Do not make canonical code imitate an
abandoned design to turn old tests green. Do not blanket skip/xfail or weaken numerical
assertions. Record a genuinely unsettled contract rather than choosing one silently.

Use small meaningful examples, independent references and invariants. A typical public
numerical path needs an eager result and representative JIT, batching or derivative
checks. Avoid applying every transformation to every input or mirroring the exact
implementation in its expected value. Check that meaningful gradients respond to a
perturbation; shape and finite-value assertions alone are insufficient.

Source tests primarily check declarative inputs and parameterisations. End-to-end
photometry, Scene summation, resolved convolution and detector/noise order belong to
Imager tests. Test that detector effects happen once after all source contributions.
Keep OpticalSystem's retained array-returning entry points covered alongside Imager's
staged output contracts. Use fixed split random keys for simulation and test the contract actually promised.

Verify the interpreter and imported package/dependency locations. Run focused files
while editing, then the requested full suite. The configured `slow` marker labels tests;
plain pytest does not automatically exclude them. Use `-m 'not slow'` only when a fast
subset is intended and say that it was a subset.

Coverage is a separate configured gate, not a reason to create low-value tests or lower
a threshold. When a scoped 100% target is requested, distinguish useful behaviour tests from tests
chasing validation tails. Report the missing lines rather than changing exclusions.
Report failures of assertions, coverage and environment separately. A failed full run
plus a passing focused rerun is not a fresh green full run. A run
using sibling source checkouts is not proof that the published dependency constraints
work. Do not install into a system/base environment as a fallback.

Documentation examples can provide useful contracts when documentation is in scope.
A test update does not independently authorise editing or regenerating documentation.

For compatibility checks, keep the reference release/tests untouched. Report supported
numerical comparisons separately from failures due to deliberate structural changes;
do not recreate old classes just to satisfy identity assertions. Use the release-file
selection when testing an artifact; a whole development checkout may include excluded
experiments and produce a different coverage denominator.
