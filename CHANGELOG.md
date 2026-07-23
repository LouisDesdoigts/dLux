# ChangeLog

---

## V0.17.0

### ✨ New Features
- Polarisation layers now accept scalar, sampled, or parametric angle and retardance
  fields, including bases evaluated dynamically from wavefront coordinates.

### ⚠️ Breaking Changes
- `LinearPolariser` and `Retarder` now cover both uniform and spatially varying
  fields. Their axis parameter is consistently named `angle`, and the redundant
  `SVLinearPolariser` and `SVRetarder` classes have been removed.

## V0.16.0

### ✨ New Features
- Added a general parametric basis interface, including explicit and generated
  bases with coefficient evaluation and solving support.
- Added polynomial basis models that integrate directly with the parametric
  basis API.
- Added coordinate-transform interpolation methods to `Wavefront` and `PSF`
  ([#302](https://github.com/LouisDesdoigts/dLux/issues/302)).
- Added dedicated `Interpolate` optical and `ApplyInterpolation` detector layers
  for use in layered propagation and detector models.

### ⚠️ Breaking Changes
- Reworked basis aberrations around the general `BasisLayer` and parametric basis
  interfaces, replacing the previous specialised aberration-layer module and
  standardising OPD, phase, and amplitude effects
  ([#331](https://github.com/LouisDesdoigts/dLux/issues/331)).

### ⏳ Deprecations
- No APIs are deprecated in this release.

### 🐛 Bug Fixes
- Wavefront and PSF interpolation now share the established interpolation utility
  while preserving their distinct complex and real-valued data requirements.

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
- Coordinate transformation classes now live in `dLux.coordinates`; the
  `dLux.transformations` module has been removed.
- Continuous scalar parameters are now consistently stored as scalar JAX arrays
  rather than Python floats
  ([#338](https://github.com/LouisDesdoigts/dLux/issues/338)).

### ⏳ Deprecations
- No APIs are deprecated in this release.

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
