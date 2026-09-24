# ChangeLog

---

## V0.16.0 (unreleased)

### 🚀 Highlights

- Added `Imager`, `Coronagraph` and `Interferometer` with separate Wavefront, PSF,
  photometry, Intensity and Image stages. Stored inputs can be overridden for one
  calculation; named input dictionaries produce named outputs.
- Retained `OpticalSystem.propagate()` and `.model()` as familiar array-returning
  entry points, backed by the current field and source calculations.
- Added unit-aware grids, reusable numerical propagators, array parameterisations,
  Jones optics and bin-integrated photometry. Utilities are available from both
  `dLux` and `dLux.utils`.
- Added explicit compatibility constructors for supported 0.15.1 workflows. They
  construct current objects rather than reproducing the old object hierarchy.

### ✨ New capabilities

- `Grid` describes 2D spatial sampling; `Axis` describes 1D sampling. Fields retain
  their sampling through propagation, resizing and interpolation. Partial grids
  can be completed when an operation has the required information.
- One `Wavefront` handles scalar, chromatic and Jones fields. Uniform Jones matrices
  remain separate until spatial materialisation is needed. `Jones`, `JonesMatrix`
  and `JonesChain` describe polarisation independently of optical layers.
- `Optic` combines transmission, OPD, phase, Jones and refractive effects. Generated
  shapes and `Basis` expressions can replace arrays; coefficients default to zeros
  when their shape is known. Basis units scale the contracted output.
- `HSTAper`, `JWSTAper`, `JWSTNRMAper` and `EuclidAper` construct sampled pupils.
  Builders and `ZernikeDef` prepare ordinary Optic/SparseOptic and Basis objects.
- Numerical `Propagator`, `FFTPropagator`, `LCT`, `SpectralPropagator` and `FreeSpace`
  support explicit grids and reusable preparation. ABCD definitions resolve to
  matrices; prepared FFT routes can compose array operations through Series.
- `BandPass` specifies equal-width bins by their outer edges. Spectrum, Blackbody,
  PolySpectrum and Filter definitions can retain finer native sampling. Photometry
  integrates spectrum × filter × QE into optical-bin Weights, respecting each
  curve's support boundaries.
- Source geometry is independent of brightness. Point/Binary/Companions expressions
  resolve angular positions; ContrastFlux allocates primary and companion fluxes.
  Scenes combine source components with their own spectra and resolved profiles.
- Detector separates field resampling, deterministic response, pixel integration,
  photon/read noise and gain. Imager supports image oversampling and jitter blur.

### ⚠️ Changed interfaces and conventions

- Declared dependencies now require Zodiax 0.6.0 or later and Interpax; the declared
  minimum Python version is 3.10.

- `Spectrum(value, bandpass=..., flux=...)` is a density model, not the released
  discrete-weight constructor. A numeric flux is total incident photons in the
  calculation BandPass; `None` preserves absolute density. Only photon flux units
  are supported. Current PolySpectrum uses a unit baseline and fixed normalised
  wavelength coordinates; old polynomial coefficients need conversion.
- Current `PointSource` positions default to arcseconds. Binary positions locate
  the primary; PA defaults to degrees and contrast to log10(companion/primary).
  The explicit compat constructors translate released radian/midpoint/mean-flux
  conventions.
- Current `FFT`, `MFT`, `ASM`, propagators, Spectrum and FourierBasis take precedence
  at the package root. Released signatures are available through `dLux.compat` or
  `dLux.utils.compat`. Field-operation layers use names such as `ResizeLayer` and
  `InterpolateLayer` to distinguish them from numerical operators.
- Grids use physical `(x, y)` metadata and array `(ny, nx)` order. Wavefront batches
  lead wavelength axes; explicit Jones axes sit directly before spatial axes.
  ABCD and numerical Jones matrices both use trailing `(..., 2, 2)` axes.
- Wavefront has no spectral-weight leaf. Imager's PSF is unweighted; photometry
  supplies weights before Intensity and detector Image formation. OpticalSystem
  retains its weighted propagation and default array-return conventions.
- Array-valued fields store `array` and `grid`. Coefficients use `coeffs` on their
  parameterisation. Fitted paths and old scalar-sampling assumptions need review.
- Interp defaults to `cubic2`, real/imaginary interpolation for complex values, and
  extrapolation with `fill=None`. It interpolates values without flux compensation.
  Shape edges are hard unless softening is supplied. Padding/cropping rejects
  parity changes that would shift the centre by half a pixel.
- Current Jitter measures width in actual pixels. The ApplyJitter adapter converts
  the released fixed-domain width; moving its sigma unchanged changes the blur.

### ⏳ Deprecations and migration

- Supported released constructors and numerical functions live in `compat` and
  `utils.compat`. Warning-backed entry points give replacements and a planned
  removal version of **0.17.0**; some direct aliases do not emit separate warnings.
- Adapters cover sampled optical systems, source/spectrum constructors, simple
  apertures, detector operations, coordinates and propagation. Transmission-only
  compound apertures construct nested Combine expressions.
- Old class identity, automatic parameter paths and implicit aberrated-aperture
  support are not restored. Unsupported physical or structural translations require
  explicit current models rather than silent approximations.
- See the [migration guide](https://louisdesdoigts.github.io/dLux/latest/migration/)
  for before/after constructions, changed units, coefficient paths and limitations.

### 🐛 Numerical fixes and retained behaviour

- Unified FFT/MFT centring and the general ABCD phase convention. Released FFT
  adapters preserve supported old transform conventions where appropriate.
- Preserved the released Affine rotation direction, Fourier coefficient mapping
  and Gaussian utility axis order, with numerical reference comparisons.
- Corrected Euclid's spider orientation. Its current aperture builder uses one
  shared edge-sampling route rather than the old helper's extra circular oversampling.
- Unified spectral integration for stored and explicit BandPass inputs. Integration
  splits at curve support boundaries instead of extending response outside support.
- Consolidated leading-batch handling, interpolation and optical preparation across
  numerical operations, fields and systems.

### 📚 Documentation and validation

- Rebuilt Module Overviews and API navigation around the current package structure;
  generated inheritance diagrams include the methods defining each contract.
- Restored constructor contracts and executable examples in principal class
  docstrings. Added focused Bases, Prebuilt and photometry guides.
- Reworked behavioral/JIT tests and added comparisons against the untouched 0.15.1
  reference. Compatibility remains partial; passing supported numerical comparisons
  does not imply the old object structure is preserved.
- Known limitation: Zernike coordinate/diameter derivatives can be non-finite at
  the exact origin. Final release-artifact and supported-Python validation are
  still pending.

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
