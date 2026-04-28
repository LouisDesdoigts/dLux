# ChangeLog

---

## V0.15.0

### Wavefront
- Replaced stored `amplitude` and `phase` with a stored complex `phasor`.
- `amplitude`, `phase`, `real`, `imaginary`, `complex`, `polar`, `power`, and `psf` are now derived from `phasor`.
- `psf` now returns the current-plane intensity, `abs(phasor) ** 2`.
- Constructor signature changed to `Wavefront(wavelength, npixels, ...)`.
- Initialisation now accepts exactly one of `diameter` or `pixel_scale`.
- Added optional 1D `center` metadata and `spec` / `xs` helpers.
- Added `Wavefront.from_phasor(...)`, `Wavefront.to_psf()`, and `Wavefront.set_spec(...)`.
- `coordinates` is now a method and supports optional scaling and polar output.
- Arithmetic operators now act directly on the complex field.
- `+` no longer means “add OPD”; use `add_opd(...)`.
- `*` no longer applies optical layers; layers are called explicitly.
- Added `-`, `/`, and in-place variants for wavefront arithmetic.
- `add_phase(...)` and `add_opd(...)` now multiply by complex phase factors.
- `tilt(...)` now supports angular unit conversion via `unit=...`.
- `normalise(...)` now supports `mode="power"` and `mode="peak"`.
- `flip`, `resize`, `rotate`, and `scale_to` now operate on the complex field by default.

### Coordinates and Sampling
- Added `Spec`, `PadSpec`, and `CoordSpec` objects for sampling metadata.
- `CoordSpec` provides `xs`, `fov`, and `extent`.
- `pixel_coords(...)` now accepts exactly one of `diameter`, `radius`, or `pixel_scale`.
- `pixel_coords(...)` added FFT-style centering via `fft_style=True`.
- `nd_coords(...)` has stricter casting and validation for pixel counts, scales, offsets, and indexing.
- Added `DistortedCoords` transformation using the existing polynomial distortion utilities.

### Propagation
- FFT and MFT propagation now operate on `phasor`.
- Fourier transform sign convention changed.
- FFT propagation now tracks output sampling through `CoordSpec`.
- FFT propagation added support for inverse propagation, explicit centering, padding, cropping, and phase-ramp correction.
- MFT propagation added inverse propagation support.
- Removed `ShiftedMFT` and `FarFieldFresnel` from the standard propagator layer API.
- Added ABCD/LCT propagation infrastructure:
  - `ABCDElement`
  - `ABCDFreeSpace`
  - `ABCDLens`
  - `ABCDMirror`
  - `ABCDConjugatePlane`
  - `ABCDPropagator`
  - `MFTPropagator`
  - `FFTPropagator`
  - `ASMPropagator`
- Added placeholder `Fraunhofer` and `Fresnel` ABCD propagator classes.

### Optical Systems
- Added `ParametricOpticalSystem`.
- Added `fov = psf_npixels * psf_pixel_scale`.
- Added `initialise_wavefront(...)` to layered optical systems.
- Added `debug_propagate_mono(...)` for collecting intermediate wavefronts.
- Optical systems now apply layers via `layer(wavefront)`.
- `return_wf=True` and `return_psf=True` are now mutually exclusive.
- Polychromatic propagation now weights phasors by `sqrt(weight)` before summing intensities.
- Offsets are now normalised internally from `None` to a `(2,)` zero vector.
- Layer attribute lookup now searches both layer keys and layer attributes.

### Layers
- Layer interface moved from `.apply(...)` to `__call__(...)`.
- `.apply(...)` remains as a compatibility alias on base layer classes.
- Added `Lambda`, a no-op unified layer.
- Added `FourierBasis`, an optical OPD layer using cached real Fourier kernels.
- `TransmissiveLayer`, `AberratedLayer`, `BasisLayer`, `Tilt`, and `Normalise` now use `__call__`.
- `AberratedLayer` and `BasisLayer` now apply OPD explicitly through `add_opd(...)`.
- Unified `Resize`, `Rotate`, and `Flip` now operate through `__call__`.

### Apertures
- Refactored aperture layers around `ApertureLayer`, `BaseDynamicAperture`, and `DynamicAperture`.
- Apertures now multiply the wavefront phasor by their transmission.
- Dynamic apertures now use `wavefront.coordinates()` and `wavefront.pixel_scale`.
- Added/standardised `extent` and `nsides` properties for dynamic apertures.
- Improved transformation type checking and missing-attribute errors.

### Sources and Spectra
- Added shared validation for return modes, wavelength arrays, spectra, and 2D positions.
- Sources reject simultaneous `return_wf=True` and `return_psf=True`.
- `Spectrum` now supports 2D weights with wavelength shape checked on the trailing axis.
- Binary and point-resolved sources now infer default weight arrays from wavelengths or spectra.
- Resolved-source models now explicitly reject `return_wf=True` because convolution destroys coherent wavefront information.
- Composite source outputs now construct `PSF` objects consistently when `return_psf=True`.
- Source and scene attribute errors are more explicit.

### Detectors and PSFs
- Added `BaseDetector`.
- `LayeredDetector` is now callable; `.model(...)` delegates to `__call__(...)`.
- Detector layers now use `__call__`; `.apply(...)` remains as a compatibility alias.
- Detector attribute lookup now searches layer keys and layer attributes.
- `ApplyJitter` now takes `sigma` in pixels, adds `oversample`, and removes pixel-scale dependence.
- Detector kernels and downsampling factors now validate positive integer sizes.
- PSF arithmetic now uses a unified implementation over `.data`.
- PSF arithmetic now supports `+`, `-`, `*`, `/`, `None`, arrays, scalars, and other `PSF` objects.

### Utilities
- Added generic angular conversion helpers:
  - `unit_factor_to_rad(...)`
  - `convert(...)`
- Existing angular conversions now use the generic conversion backend.
- Angular units now support aliases and SI-style prefixes such as `mas`, `mrad`, and `uarcsec`.
- Added n-dimensional `gaussian(...)`.
- Added `mv_gaussian(...)` stub that currently raises `NotImplementedError`.
- Fixed `factorial(0)`.
- Updated `nandiv(...)` to avoid evaluating division by zero under JAX NaN debugging.
- Added `soft_binarise(...)` using the CLIMB-style soft binarisation algorithm.
- Added `utils.norms` with NaN-safe norm functions, including peak-to-valley support and mask broadcasting.
- Added static aperture builders:
  - `circular_aperture(...)`
  - `segmented_aperture(...)`
  - `sparse_aperture(...)`
  - `hst_like(...)`
  - `jwst_like(...)`
  - `euclid_like(...)`
- Static aperture builders can optionally return support masks for basis normalisation.
- Added cached real Fourier basis helpers:
  - `fourier_kernel_1d(...)`
  - `fourier_kernels(...)`
  - `eval_fourier_basis(...)`
- Added `fft_spec(...)` and `fft_phase_ramp(...)` helpers.

### Zernikes and Basis Evaluation
- Added caching for static Zernike components.
- Updated Zernike code for safe `factorial(0)` behaviour.
- Added cached Fourier-basis evaluation support for the new `FourierBasis` layer.

### Public API and Internals
- Added centralised export helpers in `_exports.py`.
- Top-level and subpackage exports now follow module `__all__`.
- Added public exports for new coordinate specs, aperture utilities, norm utilities, ABCD propagators, `Lambda`, and `FourierBasis`.
- Standardised imports around `zodiax as zdx` and `equinox as eqx`.
- Modernised type hints to `|` unions.
- Removed dead imports and legacy compatibility shims.
- Improved error messages across wavefronts, sources, detectors, apertures, coordinates, and transformations.
