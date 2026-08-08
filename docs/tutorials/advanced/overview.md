# Advanced case studies

These tutorials demonstrate how compact dLux components compose into complex
instruments and real inference or design problems. Each case study should state its
scientific question, model graph, inferred parameters, data and likelihood, validation
target, runtime, limitations, and the changes needed for research-grade use.

## Planned case studies

### JWST-like field and instrument inference

Infer a vectorised stellar field with different positions, fluxes, and effective
temperatures while jointly fitting segment aberrations, jitter, detector rotation,
and gain from repeated noisy observations. Introduce a custom observation class once
direct model construction becomes cumbersome.

### Polarimetric coronagraph and SCC

Build a Soummer-style coronagraph, liquid-crystal vortex mask, polarimetric
aberrations, chromatic refractive effects, and an SCC reference path. Include design or
calibration with a non-Optax optimisation route.

### Sparse JWST AMI calibration and binary inference

Calibrate a distorted sparse aperture on a reference star, materialise the recovered
instrument, and infer a close binary with MCMC. Explore multiple filters and a
spectral-information basis if they fit the scientific narrative.

### Phase-diversity phase retrieval

Use physically sampled defocused observations and Fresnel or ABCD propagation to
recover wavefront error. Compare focused and deliberately diverse measurements,
explain the relevant degeneracies, and validate the recovered phase in both pupil and
detector planes.

### Existing and specialist problems

- Phase-mask and information-optimal design
- High-dimensional flat-field, pixel-response, gain, and detector calibration
- Vector-Zernike WFS inference and atmospheric phase screens
- Future multi-arm coronagraph, DM, WFS, and closed-loop systems

Custom observation classes should be taught inside one or two of these case studies,
where their value follows naturally from the complexity of the model.
