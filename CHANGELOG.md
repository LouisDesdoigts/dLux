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

### ✨ New Features
- **Fields and grids:** added `BaseField`, `ContinuousField`, and `DiscreteField`
  contracts for `Wavefront`, `PolarisedWavefront`, `PSF`, and `Image`, all backed
  by a multidimensional `GridSpec`.
- **Images:** added Poisson and read-noise simulation, variance and error tracking,
  Gaussian and Poisson likelihoods, and Fourier amplitude and power spectra.
- **Coordinates:** added broadcastable grid specifications, `ResizeSpec`,
  coordinate transformations, affine maps, polynomial distortions, and ordered
  transformation composition.
- **Parametrics:** added general explicit and implicit bases, dynamic coordinate
  evaluation, interpolation, arbitrary-dimensional polynomial models, refractive
  index models, and reusable shape definitions.
- **Dynamic optics:** added coordinate-aware dynamic optical layers and sparse
  optic foundations that compose shapes, aberrations, positions, and local or
  global transformations.
- **Sparse optics:** added shared and per-aperture parametric evaluation,
  position-vectorised wavefront propagation through optical systems, and explicit
  coherent interference of propagated sub-apertures.
- **Propagation:** added `Fraunhofer`, `Fresnel`, `FreeSpace`, and
  `ABCDPropagator`, including FFT, MFT, and LCT routes with coordinate-unit
  validation and vectorised chromatic sampling.
- **Optical systems:** added a single layered `OpticalSystem`, a dedicated
  `DetectorSystem`, intermediate-state debugging, and optional wavefront returns
  from propagation.
- **Sources:** added composable `Spectrum`, `Source`, and `BinarySource` models
  with parametric wavelengths, weights, positions, fluxes, and resolved
  distributions.
- **Optical layers:** added `RefractiveOptic` and `Wedge` layers for static or
  parametrically generated thickness and refractive-index profiles.
- **Polarisation:** added polarised wavefront propagation, Stokes evaluation, and
  uniform or spatially varying parametric polariser and retarder fields.
- Added shared interpolation methods and layers for complex wavefronts and real
  PSFs ([#302](https://github.com/LouisDesdoigts/dLux/issues/302)).

### ⚠️ Breaking Changes
- Replaced `CoordSpec` and `PadSpec` with `GridSpec` and `ResizeSpec`, and moved
  coordinate specifications and transformations into `dLux.grids`.
- `ResizeSpec` now uses the concise `pad` and `crop` attribute names.
- Consolidated wavefronts, PSFs, and detector images into `dLux.fields`; removed
  the former wavefront, PSF, detector, spectrum, and scene module structure.
- Replaced the separate layered, angular, Cartesian, and parametric optical-system
  classes with `OpticalSystem`; detector processing now lives in
  `DetectorSystem`.
- `OpticalSystem.model(...)` now takes a source and returns a `PSF`, while
  `DetectorSystem.model(...)` takes a `PSF` and returns an `Image`.
- Replaced the legacy FFT, MFT, and ASM propagation-layer classes with the new
  propagation contracts; `ASM` is now represented by `FreeSpace`.
- Reworked aperture and basis aberrations around dynamic optics and general
  parametric interfaces, replacing the specialised aperture-layer hierarchy
  ([#331](https://github.com/LouisDesdoigts/dLux/issues/331)).
- Standardised aperture and polygon sizes on diameter, with polygon diameters
  referring to their enclosing circles.
- `LinearPolariser` and `Retarder` now cover uniform and spatially varying fields;
  their axis parameter is named `angle`, and `SVLinearPolariser` and `SVRetarder`
  have been removed.

### 🐛 Bug Fixes
- Corrected FFT coordinate centring and restored explicit final-wavefront returns.
- Standardised the Collins phase across Fourier propagation routes, with FFT and
  MFT forward, inverse, centring, and mixed-route behaviour checked against common
  correctness contracts.
- Wavefront and PSF interpolation share the established interpolation utility
  while preserving complex and real-valued data requirements.
- Expanded propagation and array operations to preserve leading vectorisation
  dimensions and non-square spatial shapes where supported.

### 📚 Documentation and Testing
- Rebuilt the tests around public behavioural contracts, shared JAX transformation
  checks, and finite-output assertions.
- Automated API page and inheritance-diagram generation from public module exports.
- Added dedicated installation, citation, and publications pages and removed the
  empty FAQ and manually maintained UML image assets.

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
