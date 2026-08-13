---
name: dlux-usage
description: Use the current dLux API to build optical systems, sources, detectors, simulations, differentiable models, inference, optimisation, or instrument examples. Use for application code rather than modifying dLux itself, and verify the installed or checked-out version before adapting legacy examples.
---

# Using dLux

First determine the dLux version and available public API. Tutorials and related dLux
repositories may illustrate valuable scientific patterns but can use older contracts;
translate them through current docs and source rather than copying them unchanged.

Read `model-building.md` for the package mental model and construction checks.

Build the deterministic forward model before adding noise, optimisation, or inference:

1. define the physical question and desired output;
2. choose pupil and focal grids with explicit units and sampling;
3. build sources, optical layers, propagators, systems, and detector response;
4. identify fixed, fitted, shared, and derived parameters;
5. validate scalar behaviour, intermediate planes, flux, and sampling;
6. exercise promised vectorised and differentiated paths;
7. add observations, likelihoods, optimisation, or inference;
8. report approximations and validation limits with the resulting code.

Prefer public dLux objects and concise parameter paths. Use utilities directly when
they are the intended public numerical interface, not to recreate core object
behaviour manually.

Keep the core contracts explicit: physical coordinates use `(x, y)` order while
sampled arrays follow NumPy spatial-axis order; objects update immutably; optical
layers preserve native vectorisation; detector layers transform `Intensity`
deterministically, while uncertainty and noise belong to an explicitly constructed
`Image`.
