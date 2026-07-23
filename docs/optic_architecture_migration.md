# Optic architecture migration

This document tracks the v0.17 refactor that replaces the inheritance-heavy dynamic
aperture and basis-layer systems with composable parametric optic components. Tests
are intentionally left untouched until the source migration is structurally complete.

## Target module layout

- Abstract contracts live with their respective parameter, coordinate, layer, source,
  spectrum, optical-system, detector, and propagation implementations.
- `dLux.layers.unified_layers`: operations that can act on both wavefronts and PSFs.
- `dLux.parametric.shapes`: analytic, composite, and sparse aperture geometries that
  evaluate to transmissions.
- `dLux.coord_specs`, `dLux.coordinates`, and `dLux.affine`: flat modules separating
  sampling specifications, coordinate transformations, and affine operations.
- `dLux.layers.optical_layers`: high-level user-facing `Optic`, `DynamicOptic`,
  interpolation, and small wavefront operations.
- `dLux.parametric.bases` / `dLux.parametric.polynomials`: reusable
  parameterisations consumed by optic properties.
- `dLux.layers.propagation_layers`: direct Fourier, ABCD, LCT, and ASM propagation.
- `dLux.parametric.refractive`: refractive-index and residual optical-path models
  accepted by an optic's physical-property leaves.

## Decisions

- `Optic` is the standard physical element at one optical plane.
- Local scalar effects reduce to one complex phasor before wavefront multiplication.
- `params(wavefront)` returns a plain dictionary and is evaluated once per call.
- `polarisation` stores an ordered dictionary of existing polarising optics.
- `Lens` and `Wedge` are optical layers whose material models contribute through OPD.
- Propagation is optional and occurs after all local effects and normalisation.
- Dynamic optics share one transformed coordinate context across aperture, OPD, phase,
  and polarisation parameterisations.
- Heterogeneous ordered collections use `dlu.list2dictionary(..., ordered=True, ...)`.
- Affine operations retain semantic JAX leaves and are explicitly ordered by `Affine`;
  `TransformChain` composes affine and nonlinear transformations.
- Softening and transmission inversion are independent. `Complement` inverts shapes.
- Sparse apertures use vectorised local coordinates rather than recursive layer trees.

## Migration checklist

### Dependency audit

- [x] Inventory old aperture, optic, basis-layer, transform, lens, polarisation, and
  propagator references.
- [x] Confirm the refactor branch starts from `pol-hack` with chromatic and polarised
  wavefront support.
- [x] Record documentation and helper migrations in this tracker.

### Foundations

- [x] Keep each abstract base class in its respective domain module.
- [x] Keep `BaseLayer`, `BaseOpticalLayer`, and `BaseUnifiedLayer` in
  `unified_layers.py`.
- [x] Retain standalone `TransmissiveLayer`, `AberratedLayer`, and `Normalise`
  implementations and compose their contracts through `Optic`.
- [x] Add an ordered `Affine` container whose translation, rotation, scaling, shearing,
  and raw-matrix operations retain semantic JAX leaves.
- [x] Add ordered `TransformChain`.
- [x] Split coordinate specifications, general transformations, and affine operations
  into the flat `dLux.coord_specs`, `dLux.coordinates`, and `dLux.affine` modules.
- [x] Remove the fixed-order `CoordTransform` and redundant atomic affine classes.
- [x] Add `dlu.tilt_opd` and call it from `Wavefront.tilt`.

### Shapes and apertures

- [x] Add `Shape`, `SoftShape`, and `RadialShape` parameterisations.
- [x] Add circle, square/rectangle, regular polygon, and spider geometries.
- [x] Add `Complement`, `Intersection`, and `Union` composition.
- [x] Add vectorised `ApertureArray` for sparse and segmented pupils.
- [ ] Define global versus per-subaperture aberration coordinate conventions.
- [ ] Migrate telescope and sparse-aperture helpers to the new shapes.
- [x] Remove the obsolete dynamic aperture layer hierarchy.

### Unified optics

- [x] Rebuild `Optic` as a direct `BaseOpticalLayer` with transmission, OPD, phase,
  polarisation, normalisation, and optional propagation.
- [x] Add `DynamicOptic` with a shared transformed coordinate context.
- [x] Implement `Lens` and `Wedge` as optical layers and reuse existing polarisation
  models as optic properties.
- [x] Move user-facing interpolation and wavefront operations into `optical_layers`.
- [x] Remove or hide `TransmissiveLayer`, `AberratedLayer`, and `Normalise`.
- [x] Remove `BasisLayer` and `BasisOptic`; use parametric properties instead.

### Integration and cleanup

- [x] Consolidate `optics` into `optical_layers` and remove the redundant module.
- [x] Move unified target operations into `unified_layers`.
- [x] Remove the redundant interpolation-based `Rotate` layer; use `Interpolate`
  with `Affine.rotate`.
- [x] Remove the unused instrument abstraction and concrete instrument wrappers.
- [x] Consolidate direct and ABCD propagation APIs into `propagation_layers`.
- [x] Group parametric bases, polynomials, shapes, and refractive models under
  `dLux.parametric`.
- [x] Group layer implementations under the reinitialised `dLux.layers` package.
- [x] Rename the consolidated propagator module to `propagation_layers.py`.
- [x] Update public exports and API documentation pages for the consolidated modules.
- [ ] Regenerate affected UML diagrams.
- [ ] Update tutorials after the source design stabilises.
- [ ] Remove compatibility aliases after internal consumers have migrated.
- [ ] Update tests only after the structural source migration is complete.
- [ ] Restore full test and documentation coverage before merging.

## Compatibility gaps discovered during implementation

- `Lens` and `Wedge` remain standalone optical layers while their refractive-index
  inputs remain parametric physical-property models.
- Existing tests and tutorials still construct `BasisOptic`; the class has been removed
  from the new high-level optic implementation.
- Existing tests and tutorials import the removed dynamic aperture and layer-mixin
  classes. These will be migrated in the deliberately deferred test/documentation
  phase.
- Atomic transform names currently overlap with unified interpolation-layer names and
  are intentionally not top-level re-exported until the old layer API is removed.
