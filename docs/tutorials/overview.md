# dLux tutorials

dLux models optical systems, sources, detectors, and inference problems as one
differentiable object graph. The tutorials are organised into a short starting point,
introductory workflows, guides to how the package works, complete advanced case
studies, and migrated versions of useful older tutorials.

## Getting started

[`Getting started`](introductory/getting_started.md) builds a small optical system,
generates a model image, and recovers a compact set of phase parameters. It is the
shortest complete introduction to the dLux modelling workflow.

## Introductory tutorials

The [`introductory`](introductory/overview.md) tutorials contain the complete starting
workflow and introduce the principal ready-to-use tools that extend it:

- end-to-end optical modelling, exposure simulation, and parameter recovery;
- named prebuilt components and their supported customisation;
- basic optics, propagators, and their physical and sampling contracts.

## Basics

The [`basics`](basics/overview.md) tutorials explain the reusable package model and
methods:

- fields, grids, wavefronts, and optical-layer interactions;
- optical systems, sources, spectra, detectors, and images;
- custom, dynamic, dense, sparse, and segmented apertures;
- custom layers, parametric types, context resolution, and raised parameter paths;
- optimisation, sampling, derivatives, likelihoods, and uncertainty estimation.

## Advanced

The [`advanced`](advanced/overview.md) tutorials use these pieces to solve complete
scientific and instrumental problems, including JWST-like field inference,
coronagraphy, sparse-aperture calibration, phase-diversity phase retrieval, detector
calibration, phase-mask design, and wavefront sensing.

## Old tutorials

The [`old`](old/optical_systems.md) collection preserves migrated versions of the
previously published tutorials while the new suite is developed. These examples use
the current compatibility surface but retain their historical structure and are not
the primary route for learning the package.

## How the pieces fit together

### Low-level model

The low-level objects define sampled physical state and the transformations applied to
it:

- **Grids** define array sampling, physical spacing, centres, dimensionality, and
  units.
- **Fields** carry sampled data and its grid contract. A wavefront retains complex
  optical amplitude and wavelength information; intensity represents deterministic
  sampled optical or detector signal; an image represents realised detector data and
  optional uncertainty.
- **Layers** apply one optical or detector transformation to a field and return the
  resulting field with its updated sampling and physical state.
- **Parametrics** resolve differentiable values from their stored parameters and the
  context supplied by a source, field, layer, or system.

Their central relationships are:

```text
grid -> field
parametric + context -> resolved value
layer + field -> transformed field
```

> **UML placeholder:** grids, fields, wavefronts, intensities, images, parametrics, and
> optical/detector layers.

### High-level model

The high-level objects assemble these transformations into complete models:

- **Sources and spectra** define incident wavelengths, spectral weights, positions,
  spatial structure, and flux.
- **Systems** hold ordered, named layers and define the boundary between optical
  propagation and detector response.
- **Builders** use grids and definitions to construct static arrays or ready-to-use
  layers at setup time.
- **Prebuilt components** provide named configurations composed from the general
  builders. Their fidelity, provenance, omissions, and supported overrides are part of
  their contract.

Their central relationships are:

```text
grid + builder -> layer -> system
prebuilt -> configured builders and layers
source + optical system -> intensity
intensity + detector system -> transformed intensity
intensity -> explicitly constructed image -> simulated exposure
```

> **UML placeholder:** sources, spectra, optical systems, detector systems, builders,
> prebuilt components, and their complete modelling flow.

### Utilities

`dLux.utils` is the independent functional numerical foundation. It provides coordinate,
aperture, propagation, interpolation, polynomial, normalisation, and statistical
operations without depending on the dLux object model. Core objects and layers use
these functions internally, while users can also call them directly for custom models
and validation.

> **UML placeholder:** the relationship between the object model and the independent
> numerical modules in `dLux.utils`.

Each final diagram should briefly explain the role of its objects and link directly to
the tutorials that use them. The diagrams orient the reader; they do not replace the
physical and numerical contracts described in the tutorials and API reference.

## Inference and methods

The completed page will also provide a task-based index for Zodiax, Optax, Optimistix,
NumPyro, BlackJAX, Jacobians, Hessians, Fisher information, noise models, likelihoods,
posterior diagnostics, and the tutorials in which each method is used.
