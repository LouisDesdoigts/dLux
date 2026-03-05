# ChangeLog

This project now tracks stabilization and API-cleanup changes prior to 1.0.

## Unreleased

### Type Hints & Consistency Targets

- Use `|` union syntax; avoid `typing.Union`/`typing.Optional` patterns.
- Keep type hints lightweight and readability-first in public APIs.
- Prefer runtime-safe typing patterns; avoid invalid callable-style annotations.
- Avoid unnecessary `TYPE_CHECKING` indirection unless needed for import-cycle safety.
- Standardize to explicit package aliases for third-party core libs:
  - `import zodiax as zdx`
  - `import equinox as eqx`
- In `sources.py`, avoid core `import jax`; use targeted JAX symbol imports plus package-style imports for `jax.numpy`, `jax.scipy`, and `jax.tree`.
- Use module `__all__` as the source of truth and auto-reexport those symbols in package `__init__` modules.
- Avoid duplicate implementation logic; centralize shared package plumbing helpers.
- Avoid builtin-shadowing API parameter names (e.g. use `use_sum`, `allowed_type`).
- Pre-1.0: prefer clean API direction over compatibility shims when renaming internal keyword APIs.
- During overhaul, rely on pre-commit for lint/format; do not add explicit standalone lint/type-check gate commands.
- Defer test gate execution until the dedicated testing phase.

### Logging Scope

- This changelog tracks consistency decisions and major direction changes for final review.
- It is intentionally non-exhaustive and does not need to list every small code edit.

### Consistency Audit Checklist

- [x] Union syntax uses `|` (no `typing.Union`/`typing.Optional`).
- [x] Public-facing hints remain lightweight/readable (no unnecessary shape/dtype verbosity).
- [x] No invalid callable-style annotations in type positions.
- [x] `TYPE_CHECKING` is only used where import-cycle safety requires it.
- [x] `zodiax`/`equinox` imports use explicit aliases (`zdx`, `eqx`).
- [x] `sources.py` avoids core `import jax`.
- [x] Package `__init__` reexports are driven by child-module `__all__`.
- [x] Shared package reexport logic is centralized (no duplicated `_reexport` implementations).
- [x] APIs avoid builtin-shadowing parameter names.
- [x] Pre-1.0 cleanup does not reintroduce removed compatibility shims.

Current package-wide audit status: 10/10 checks passing (`src/dLux`, 9 automated + 1 manual).

### Changed

- Standardized import conventions (`zodiax`/`equinox` alias style; no core `import jax` in `sources.py`).
- Standardized type-hint direction (lightweight annotations and `|` union syntax).
- Consolidated export/interface/API naming consistency updates across `src/dLux`.
- Simplified package APIs so top-level and sub-package namespaces are auto-populated from child-module `__all__`.
- Centralized package reexport logic into one internal helper to reduce duplicate code.
- Began step-1 API flow refinement in `sources.py` and `wavefronts.py` with clearer source subclass signatures, shared return-mode/position validators, and wavefront module cleanup.
- Extended step-1 `sources.py` flow cleanup with consistent return-mode validation across source types, safer default wavelength/weight handling for composite sources, and robust tuple/list handling in `Scene` construction.
- Extended step-1 `wavefronts.py` flow cleanup with simple inline 2D input validation for tilt/propagation shifts, safer `None` shift defaults, and corrected peak-mode normalisation to use PSF peak intensity.
- Extended step-1 `optical_systems.py` flow cleanup with safer `None` offset defaults, clearer offset shape errors, and small annotation/attribute consistency fixes.
- Extended step-1 `instruments.py` flow cleanup with safer source tuple validation, clearer type/error messaging, and model signature consistency updates.
- Extended step-1 `detectors.py` flow cleanup with clearer attribute errors, model return-signature consistency, and improved detector model docstrings.
- Extended step-1 `layers/apertures.py` flow cleanup with corrected wavefront coordinate evaluation in aperture application paths and targeted attribute/signature consistency fixes.
- Extended step-1 `layers/optics.py` cleanup with class-accurate self annotations and concise wording fixes in public docstrings.
- Extended step-1 `layers/optical_layers.py` cleanup with class-accurate self annotations and concise validation/doc wording fixes.
- Extended step-1 `layers/propagators.py` cleanup with class-accurate method annotations, clearer shift-shape validation text, and safer `None` default handling for far-field Fresnel shift.
- Extended step-1 `layers/detector_layers.py` cleanup with class-accurate method annotations and minimal constructor validation for integer kernel sizes.
- Extended step-1 `layers/unified_layers.py` cleanup with class-accurate annotations and concise docs/validation-message wording fixes.
- Extended step-1 `layers/aberrations.py` cleanup with concise constructor/doc wording fixes, consistent index normalisation, and minor control-flow simplification.
- Added a second-pass `layers/detector_layers.py` polish to keep `ApplyJitter.sigma` as a readable scalar float and clarify detector-layer parameter docs.
- Added final orchestration polish in `optical_systems.py` and `instruments.py` with class-accurate self annotations and concise source/doc typing clarifications.
- Added a core-model pass in `spectra.py` with class-accurate annotations, lightweight wavelength-shape validation, and clearer spectrum shape-error wording.
- Added a core-model pass in `psfs.py` and `transformations.py` with a PSF arithmetic field-target fix (`data` vs `phasor`) plus concise transformation doc/message/typing consistency cleanups.
- Added package-wide sweep follow-up fixes for remaining doc typos in `layers/apertures.py` and class-accurate layer-mutation self annotations in `optical_systems.py`.
- Removed temporary pre-1.0 compatibility shims for renamed utility keywords.

### Notes

- These changes are part of the pre-1.0 stabilization/overhaul process and may include API cleanup that is intentionally not backward-compatible yet.
