---
name: dlux-usage
description: Build optical models, simulations and inference code using the current dLux API. Use for application code rather than changes to dLux itself; translate older examples through the checked-out interfaces.
---

# Using dLux

Confirm the package version and actual imports before adapting examples. Read
[model-building.md](model-building.md) for the normal workflow and the relevant
[current contracts](../dlux-development/contracts.md) for shapes and ownership.

Start from the user's optical problem and desired output. Preserve OpticalSystem when
adapting an established layer sequence; use Imager for stored source, photometry and
detector stages. Establish a deterministic model, sampling and expected photons before
adding noise or inference. Keep
model construction short; use existing numerical definitions and prebuilt pupils
rather than manually reproducing their mathematics.

Use arrays directly until a parameterisation adds value. A Basis or other Expression
can replace an array in an Optic. Units, aliases, external State and unbound operators
are optional advanced tools, not prerequisites for an introductory example.

Choose fitting parameters through explicit Zodiax paths and immutable updates. Populate
shared Linked parameters at the containing model. Let fields and systems handle their
supported wavelength, source and Jones axes rather than creating ad hoc vmap wrappers.

Validate sampling, flux and representative gradients before fitting. State relevant
approximations, such as resolved-source convolution or coarse optical wavelength
sampling. Keep stochastic simulation separate from the deterministic prediction, and
use Imager.simulate when detector noise/gain ordering matters.

Do not treat parameter lookup as a search over the printed representation: defaults may
be hidden, properties are not necessarily stored leaves, and descendants can repeat
names. Use paths such as `pupil.opd.coeffs` and `source.spectrum.flux`, checking the actual
container. Imager PSF excludes filter/QE/flux weights; OpticalSystem's propagate/model
methods intentionally retain weighted, array-returning conveniences.
