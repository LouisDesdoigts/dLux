# Tutorials

Use this page to find the shortest route to what you want to build. The current
tutorials are being reorganized into a compact learning path, focused how-to guides,
and complete scientific case studies. Until that work is complete, the tables below
route to the existing material and call out when a page covers several topics.

## Start here

| Goal | Tutorial | Level | What it covers |
|---|---|---:|---|
| Understand systems, sources, fields, and images | [Optical Systems, Sources, and Detectors](../optical_systems.md) | Beginner | The current dLux object model and a first forward model |
| Build and differentiate a first model | [A Basic Overview](../phase_retrieval.md) | Beginner | Optical construction, simulated data, gradients, and phase retrieval |
| Construct telescope pupils | [Apertures in dLux](../apertures.md) | Beginner--intermediate | Built-in, custom, segmented, and dynamic apertures |
| Calibrate many optical and detector parameters | [Instrumental Calibration](../calibration.md) | Advanced | Joint phase, source, jitter, and flat-field calibration |

Suggested order for a new user:

1. [Optical Systems, Sources, and Detectors](../optical_systems.md)
2. [Apertures in dLux](../apertures.md)
3. [A Basic Overview](../phase_retrieval.md)
4. One application or propagation guide from the sections below

## I want to build or extend an optical system

| Goal | Best current tutorial | Relevant section |
|---|---|---|
| Create a custom optical layer | [Instrumental Calibration](../calibration.md) | Creating a custom layer |
| Work with parametric or dynamic apertures | [Apertures in dLux](../apertures.md) | Custom and dynamic apertures |
| Build a multi-plane coronagraph | [Coronagraphs](../coronagraphs.md) | Multi-plane system, FPM, and Lyot stop |
| Use a finely sampled focal-plane mask efficiently | [Coronagraphs](../coronagraphs.md) | Soummer MFT-style coronagraph |
| Model a sparse pupil efficiently | [Sparse Propagation](../sparse_propagation.md) | Subaperture propagation |
| Model defocused/reimaged systems | [ABCD Fresnel](../abcd_fresnel.md) | ABCD/LCT propagation and phase diversity |
| Model physical free-space propagation | [ASM Fresnel](../asm.md) | Gratings, colour, and multiple physical planes |

## I want to solve an inverse or design problem

| Goal | Best current tutorial | Techniques |
|---|---|---|
| Recover pupil aberrations | [A Basic Overview](../phase_retrieval.md) | Phase retrieval with gradient optimization |
| Perform phase-diverse retrieval | [ABCD Fresnel](../abcd_fresnel.md) | Positive and negative defocus |
| Jointly calibrate optics and detector | [Instrumental Calibration](../calibration.md) | High-dimensional differentiable calibration |
| Optimize a coronagraph | [Coronagraphs](../coronagraphs.md) | FPM and Lyot-stop optimization |
| Design an information-rich phase mask | [Phase Mask Design](../mask_design.md) | Gradient energy, Fisher information, and CRLBs |

## Current coverage gaps

Dedicated guides for parametric classes, custom propagators, detector layers,
polarization, refractive optics, uncertainty-aware inference, WFS/DM systems,
multi-arm instruments, and stateful closed-loop simulations are planned but not yet
available. `TUTORIALS_AND_FEATURES_REVIEW.md` in the source repository tracks the
proposed curriculum and implementation roadmap.
