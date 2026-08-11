# dLux model-building reference

The core flow is:

`grid -> builder or parametric -> layer -> optical system -> source -> intensity -> detector system -> image`

Grids define sampled coordinates and units. Builders materialise ideal sampled
components or layers. Parametrics resolve differentiable values from context. Optical
layers transform wavefronts, propagators move between planes, and systems order those
operations. Sources provide wavelength, position, flux, and spatial distributions.
Detector systems transform focal intensity into an observation.

Keep wavelength, source, aperture, exposure, filter, and parameter axes semantically
distinct. Let dLux own ordinary vectorisation instead of manually mapping scalar
models. Use raised Zodiax paths to select and update fitted leaves. Keep stochastic
observations outside the deterministic forward model so simulation, optimisation,
Hessians, and posterior calculations share the same physical prediction.

Validate sampling and units before fitting. Separate fixed, fitted, shared, and derived
parameters, state model fidelity and data provenance, and return complete runnable
code with the parameter paths users are expected to change.

Use direct attributes to inspect an object and Zodiax paths to construct reusable
parameter selections. `get(..., as_dict=True)` builds an optimisation-facing mapping;
`set`, `add`, `multiply`, `divide`, `power`, `min`, and `max` return immutable updated
objects. Omit fixed leaves from the fitted dictionary, qualify ambiguous leaves with a
layer or collection name, and fail on unused paths when updating several top-level
objects.
