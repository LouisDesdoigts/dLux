# ChangeLog

---

## V0.16.0

### 🚀 Highlights
- Rebuilt dLux around coordinate-aware field containers, composable parametric
  models, and explicit contracts between optical and detector systems.
- Replaced the legacy propagation classes with a unified set of Fraunhofer,
  Fresnel, free-space, and ABCD propagation layers.
- Added first-class support for chromatic and polarised wavefronts throughout the
  optical modelling stack.
- Added grid-aware dense, sparse, dynamic, and segmented aperture construction for
  building realistic differentiable instruments without materialising unnecessary
  full-array bases.

### ✨ New Features
- **Fields and grids:** added `BaseField`, `ContinuousField`, and `DiscreteField`
  contracts for `Wavefront`, `PolarisedWavefront`, deterministic `Intensity`, and
  realised `Image`, all backed by a multidimensional `GridSpec`.
- **Images:** added explicit `Intensity` to `Image` conversion, Poisson and read-noise
  simulation, standard-deviation and derived-variance tracking, and Fourier amplitude
  and power spectra. Statistical residuals and likelihoods remain in Zodiax.
- **Coordinates:** added broadcastable grid specifications, `ResizeSpec`,
  coordinate transformations, affine maps, polynomial distortions, ordered
  transformation composition, construction from regular axes, oversampling, and
  unit-aware plotting extents.
- **Parametrics:** added general explicit and implicit bases, dynamic coordinate
  evaluation, interpolation, arbitrary-dimensional polynomial models, refractive
  index models, selected polynomial degrees, and reusable hard or softened shape
  definitions. `Basis` replaces the former `ExplicitBasis` name.
- **Spectral parametrics:** added normalised `SpectralPolynomial`, `SpectralBasis`,
  and `Blackbody` models for wavelength-dependent source weights.
- **Aperture construction:** added grid-aware builders for dense and sparse optics,
  Zernike OPD definitions, and configurable HST-, JWST-, JWST NRM-, and Euclid-like
  templates. Named templates describe their intended fidelity rather than claiming
  exact observatory models.
- **Geometry:** added hard and softened convex polygons, differentiable vertices, and
  opt-in convexity validation alongside the existing circular, rectangular, polygon,
  and spider primitives.
- **Segmented apertures:** added compact pasted transmission and `PastedBasis`
  construction, with parallel and memory-efficient placement strategies for large
  segmented pupils such as ELT-scale apertures.
- **Dynamic optics:** added coordinate-aware dynamic optical layers and sparse
  optic foundations that compose shapes, aberrations, positions, and local or
  global transformations.
- **Sparse optics:** added shared and per-aperture parametric evaluation,
  position-vectorised wavefront propagation through optical systems, and explicit
  coherent interference of propagated sub-apertures.
- **Propagation:** added `Fraunhofer`, `Fresnel`, `FreeSpace`, and
  `ABCDPropagator`, including FFT, MFT, and LCT routes with coordinate-unit
  validation and vectorised chromatic sampling. Fraunhofer and Fresnel propagation
  support explicit reverse propagation where the numerical route has a defined
  physical inverse.
- **Optical systems:** added a single layered `OpticalSystem`, a dedicated
  `DetectorSystem`, intermediate-state debugging, and optional wavefront returns
  from propagation.
- **Serialisation:** added `Base.save(...)`, template-based `Base.load(...)`, and
  functional `dLux.save(...)`/`dLux.load(...)` interfaces for validated `.dlux`
  archives of realised JAX PyTrees. Imported Equinox classes are reconstructed
  automatically; `like=` remains available as a structural and local-class resolver.
- **Detector modelling:** added deterministic `Sensitivity`, `Convolve`, `Jitter`,
  `Bias`, `Gain`, and `Saturation` layers. Responses may be fixed arrays or
  parametrics, allowing spatial and nonlinear detector models to use the same layer
  contract.
- **Sources:** added composable `Spectrum`, `Source`, and `BinarySource` models
  with parametric wavelengths, weights, positions, fluxes, and resolved
  distributions. `Source` now supports vectorised populations of positions, fluxes,
  spectra, and resolved distributions through one leading-axis contract.
- **Coronagraphy:** added `SoummerFPM`, a modular Soummer-style focal-plane-mask
  layer built around a Fraunhofer MFT propagator and an arbitrary optical mask layer.
- **Optical layers:** added `RefractiveOptic` and `Wedge` layers for static or
  parametrically generated thickness and refractive-index profiles.
- **Polarisation:** added polarised wavefront propagation, Stokes evaluation, and
  uniform or spatially varying parametric polariser and retarder fields.
- Added shared interpolation methods and layers for complex wavefronts and real
  intensity fields ([#302](https://github.com/LouisDesdoigts/dLux/issues/302)).
- Added a normalised multidimensional Gaussian utility with explicit physical-axis
  ordering and batched mean or covariance support.
- Added central unit parsing, canonicalisation, validation, and contextual errors for
  grids, wavelengths, positions, fluxes, and relative source distributions.
- Added consistent raised-parameter paths across nested fields, sources, parametrics,
  layers, and systems, with clearer errors for unresolved paths.
- Added `dLux.utils.update(...)` for applying one parameter-path mapping across one or
  more Zodiax objects with strict unused-path detection.
- Added official Python 3.14 support.

### ⚠️ Breaking Changes
- Replaced `CoordSpec` and `PadSpec` with `GridSpec` and `ResizeSpec`, and moved
  coordinate specifications and transformations into `dLux.grids`.
- `ResizeSpec` now uses the concise `pad` and `crop` attribute names.
- Consolidated wavefronts, deterministic intensities, and detector images into
  `dLux.fields`; removed the former wavefront, PSF, detector, spectrum, and scene
  module structure. `PSF` is retained only as a deprecated compatibility wrapper for
  `Intensity`.
- Replaced the separate layered, angular, Cartesian, and parametric optical-system
  classes with `OpticalSystem`; detector processing now lives in
  `DetectorSystem`.
- `OpticalSystem.model(...)` now takes a source and returns an `Intensity`.
  `DetectorSystem.model(...)` applies deterministic detector responses and also
  returns an `Intensity`; construct an `Image` explicitly to simulate observations.
- Replaced the legacy FFT, MFT, and ASM propagation-layer classes with the new
  propagation contracts; `ASM` is now represented by `FreeSpace`.
- Reworked aperture and basis aberrations around dynamic optics and general
  parametric interfaces, replacing the specialised aperture-layer hierarchy
  ([#331](https://github.com/LouisDesdoigts/dLux/issues/331)).
- Unified hard and softened geometry through explicit edge definitions. Aperture
  builders now materialise a layer when called and return sampled arrays through
  their explicit `build(...)` methods.
- Renamed abstract extension contracts consistently: coordinate transforms derive
  from `BaseCoordTransform`, aperture construction derives from `BaseBuilder`, and
  OPD recipes derive from `BaseOPDDef`. `Distortion` is the general coordinate-field
  transformation; the released concrete `CoordTransform` remains a deprecated
  affine compatibility wrapper.
- Standardised aperture and polygon sizes on diameter, with polygon diameters
  referring to their enclosing circles.
- `LinearPolariser` and `Retarder` now cover uniform and spatially varying fields;
  their axis parameter is named `angle`, and `SVLinearPolariser` and `SVRetarder`
  have been removed.
- Optical layers define their monochromatic operation through `apply_mono(...)`;
  `apply(...)` owns leading-axis vectorisation and `__call__(...)` remains its concise
  callable interface.
- Renamed public parameter leaves and constructor keywords from `coefficients` to
  `coeffs`. Warning-backed aliases preserve released uses through dLux 0.17.

### ⏳ Deprecations and Compatibility
- Added a central compatibility layer for supported dLux 0.14 and 0.15 interfaces.
  Safe aliases emit actionable warnings with direct migration examples; legacy
  contracts that cannot preserve their physical meaning raise migration errors.
- Legacy module import paths are exposed without retaining empty compatibility
  modules throughout the package.
- Deprecated interfaces are scheduled for removal in dLux 0.17.0.
- Retained `PSF`, `CoordSpec`, `PadSpec`, `CoordTransform`, `DistortedCoords`, legacy
  detector-layer names, system names, source wrappers, propagator wrappers, and
  `coefficients` aliases where their released behaviour can be translated safely.
- Added a task-oriented [0.16 migration guide](https://louisdesdoigts.github.io/dLux/latest/migration/)
  covering grids, fields, systems, sources, parametrics, apertures, propagation, and
  custom layers.

### 🐛 Bug Fixes
- Corrected FFT coordinate centring and restored explicit final-wavefront returns.
- Standardised the Collins phase across Fourier propagation routes, with FFT and
  MFT forward, inverse, centring, and mixed-route behaviour checked against common
  correctness contracts.
- Wavefront and intensity interpolation share the established interpolation utility
  while preserving complex and real-valued data requirements.
- Expanded propagation and array operations to preserve leading vectorisation
  dimensions and non-square spatial shapes where supported.
- Standardised batched coordinate transformations and vectorised optical application
  so semantic leading axes are preserved through nested models.
- Corrected source spectral evaluation by using normalised wavelength coordinates,
  optional spectral-weight normalisation, and explicit `linear`, `log`, and `ln`
  distribution conventions.
- Corrected finite-grid jitter construction for scalar, axis-aligned, correlated,
  and zero-width kernels while preserving differentiability.

### 📚 Documentation and Testing
- Rebuilt the tests around public behavioural contracts, shared JAX transformation
  checks, and finite-output assertions.
- Automated API page and inheritance-diagram generation from public module exports.
- Added dedicated installation, citation, and publications pages and removed the
  empty FAQ and manually maintained UML image assets.
- Added repository guidance and a dLux development skill for consistent AI-assisted
  implementation, review, testing, documentation, and external model construction.
- Added a dedicated 0.14/0.15 migration guide, parameter-path guide, compatibility
  API reference, and deprecation test suite.
- Reorganised tutorials into introductory, basics, advanced, and retained legacy
  routes, with Getting Started published as the reference end-to-end workflow.
- Replaced package-scale class diagrams with compact package and module maps while
  retaining local inheritance diagrams on generated API pages.
- Added a maintained contributor register covering software, science, tutorials,
  documentation, testing, and review.

### 🎉 Contributors
- [Jaren Ashcraft (@Jashcraf)](https://github.com/Jashcraf) contributed the
  polarisation foundations, including polarised wavefronts, Stokes evaluation, and
  uniform and spatially varying polarisation layers.
- [Adam Taras (@ataras2)](https://github.com/ataras2) contributed to the v0.16
  development series.

## V0.15.1

### ✨ New Features
- Aperture transformations now accept any `BaseCoordTransform`, including
  `DistortedCoords` ([#332](https://github.com/LouisDesdoigts/dLux/issues/332)).
- Static circular, HST-like, and Euclid-like apertures now accept `array_diameter`
  for independently sizing the output grid
  ([#328](https://github.com/LouisDesdoigts/dLux/issues/328)).
- `zernike_fast(...)` and `polike_fast(...)` now accept an optional `diameter`
  ([#275](https://github.com/LouisDesdoigts/dLux/issues/275)).
- Added `solve_basis(array, basis)` for recovering basis coefficients with a
  least-squares solve ([#299](https://github.com/LouisDesdoigts/dLux/issues/299)).

### ⚠️ Breaking Changes
- Coordinate transformations now live in `dLux.coordinates`, affine operations in
  `dLux.affine`, and sampling specifications in `dLux.coord_specs`; the former
  transformation and coordinate-package paths have been removed.
- Continuous scalar parameters are now consistently stored as scalar JAX arrays
  rather than Python floats
  ([#338](https://github.com/LouisDesdoigts/dLux/issues/338)).

### 🐛 Bug Fixes
- Fixed transmission, softening, and Zernike basis generation for apertures larger
  than their sampling grids
  ([#305](https://github.com/LouisDesdoigts/dLux/issues/305),
  [#328](https://github.com/LouisDesdoigts/dLux/issues/328)).
- Composite apertures now apply aberrations regardless of normalisation and no longer
  fail when they contain no aberrated sub-apertures
  ([#320](https://github.com/LouisDesdoigts/dLux/issues/320)).

### 🎉 New Contributors
- 🌟 [Matthijs Mars (@MatthijsMars)](https://github.com/MatthijsMars) made their first
  contribution by expanding aperture transformations to support all
  `BaseCoordTransform` implementations
  ([#333](https://github.com/LouisDesdoigts/dLux/pull/333)).
- 🌟 [Yinzi Xin (@yinzi-xin)](https://github.com/yinzi-xin) made their first
  contribution by adding independently sized aperture grids through
  `array_diameter`, including tests
  ([#336](https://github.com/LouisDesdoigts/dLux/pull/336)).

## V0.15.0

### 🚀 Highlights
- **Complex wavefronts:** `Wavefront` now stores the complex electric-field `phasor`
  directly, improving field arithmetic and avoiding undefined gradients when fields
  cancel ([#291](https://github.com/LouisDesdoigts/dLux/issues/291),
  [#295](https://github.com/LouisDesdoigts/dLux/issues/295)).
- **Sampling-aware propagation:** FFT and MFT propagation now track coordinate
  sampling, support inverse propagation, and correct centring-dependent phase ramps.
- **Callable models:** optical layers, detector layers, unified layers, and detectors
  now share a consistent `object(target)` interface.
- **Expanded modelling toolkit:** this release adds ABCD propagation, static telescope
  apertures, cached Fourier bases, mask-design tools, and NaN-safe norms.

### ✨ New Features
- **Wavefronts:** added `from_phasor(...)`, `to_psf()`, coordinate specifications,
  configurable power or peak normalisation, angular units for `tilt(...)`, and direct
  complex-field arithmetic
  ([#288](https://github.com/LouisDesdoigts/dLux/issues/288)).
- **Coordinates:** added `Spec`, `PadSpec`, and `CoordSpec`; expanded
  `pixel_coords(...)` to support diameter, radius, pixel scale, and FFT-style
  centring; and added generic angular-unit conversion helpers.
- **Propagation:** added ABCD elements and ABCD-backed MFT, FFT, and ASM propagators,
  plus coordinate sampling, inverse propagation, padding, cropping, and phase-ramp
  utilities.
- **Optical systems:** added `ParametricOpticalSystem`, output field-of-view metadata,
  wavefront initialisation, intermediate-state debugging, and physically consistent
  `sqrt(weight)` spectral weighting
  ([#268](https://github.com/LouisDesdoigts/dLux/issues/268)).
- **Layers:** publicly exported `OpticalLayer`, added the cached `FourierBasis` OPD
  layer, and added the no-op `Lambda` layer
  ([#301](https://github.com/LouisDesdoigts/dLux/issues/301),
  [#315](https://github.com/LouisDesdoigts/dLux/issues/315)).
- **Apertures:** added circular, segmented, sparse, HST-like, JWST-like, and
  Euclid-like static aperture builders, with optional support masks for basis
  normalisation.
- **Numerical utilities:** added CLIMB-style `soft_binarise(...)`, masked NaN-safe
  norms, n-dimensional Gaussian kernels, cached Fourier helpers, and fill values for
  resizing and padding
  ([#276](https://github.com/LouisDesdoigts/dLux/issues/276),
  [#317](https://github.com/LouisDesdoigts/dLux/issues/317)).
- **Detectors and PSFs:** added `BaseDetector`, made layered detectors callable,
  expanded PSF arithmetic, and updated `ApplyJitter` with pixel-based sigma and
  oversampled kernel generation
  ([#262](https://github.com/LouisDesdoigts/dLux/issues/262)).
- **Sources and spectra:** added consistent return-mode and shape validation, 2D
  spectral weights, inferred default weight shapes, and consistent `PSF` outputs.

### ⚠️ Breaking Changes
- `Wavefront` construction changed from `Wavefront(npixels, diameter, wavelength)` to
  `Wavefront(wavelength, npixels, diameter=...)`, with `pixel_scale` available as an
  alternative to `diameter`.
- `Wavefront` stores `phasor`; `amplitude` and `phase` are now derived properties and
  should be modified through `phasor`, `add_phase(...)`, or `add_opd(...)`.
- `wavefront.coordinates` became the method `wavefront.coordinates()`.
- Wavefront arithmetic now acts on complex fields. In particular, `wavefront + opd`
  no longer applies OPD; use `wavefront.add_opd(opd)`.
- Custom layers should implement `__call__(...)`, and optical systems now invoke layers
  as callables rather than through `.apply(...)`.
- The Fourier-transform sign convention and propagation interfaces changed. FFT
  propagation now produces corrected coordinates and phases after centring
  ([#300](https://github.com/LouisDesdoigts/dLux/issues/300)).
- Removed `ShiftedMFT`, `FarFieldFresnel`, `fresnel_MFT(...)`,
  `fresnel_phase_factors(...)`, and `quadratic_phase(...)`.
- Resolved-source convolution no longer supports `return_wf=True`, because image-plane
  convolution cannot preserve coherent wavefront information.
- Minimum requirements increased to Python 3.10 and Zodiax 0.5, with
  `abcdLux>=0.0.2` added as a runtime dependency.

### ⏳ Deprecations
- `.apply(...)` remains available on base layers and coordinate transformations as a
  compatibility alias, but new code should use the callable interface.

### 🐛 Bug Fixes
- Fixed NaN-prone and undefined-gradient wavefront arithmetic by retaining complex
  fields throughout propagation and numerical operations
  ([#291](https://github.com/LouisDesdoigts/dLux/issues/291),
  [#295](https://github.com/LouisDesdoigts/dLux/issues/295)).
- Fixed FFT coordinate centring, propagation phase ramps, and forward/inverse Fourier
  sign consistency ([#300](https://github.com/LouisDesdoigts/dLux/issues/300)).
- Fixed propagation paths that operated on amplitude instead of the full complex
  phasor.
- Fixed `nandiv(...)` for Python float inputs and safe division under JAX NaN debugging
  ([#269](https://github.com/LouisDesdoigts/dLux/issues/269)).
- Fixed `factorial(0)`, associated Zernike factor evaluation, resizing edge cases, and
  regular-polygon aperture initialisation.
- Improved validation and error reporting across coordinates, sources, spectra,
  detectors, apertures, optical systems, and transformations.
