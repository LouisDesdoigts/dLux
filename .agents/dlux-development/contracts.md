# Current dLux ownership and contracts

This is a working map of the current API, not a record of every intermediate design.
Check the implementation and accepted user decisions when it conflicts with either.
Read the relevant section rather than loading the whole map for a narrow task.

## Owners

| Owner | Responsibility |
| --- | --- |
| `utils/grids` | Spec/UnitSpec/Axis/Grid, coordinate maps, ResizeSpec/FFTSpec/PasteSpec and Field |
| `utils/numerical` | Polynomial, Interp, units, norms, Gaussians, shapes, Profile and Jitter |
| `utils/bases` | ParametricBasis, Basis, FourierBasis, PastedBasis and CLIMB |
| `utils/sampled.py`, `utils/batching.py` | Context-selected coordinates and explicit numerical broadcasting |
| `utils/sequential.py` | BaseLayer, Sequential, Series and Combine |
| `utils/astrometry.py` | Positions, Point, Binary and Companions Expressions |
| `utils/photometry` | Spectra, Filter, BandPass, Weights, flux definitions and integration |
| `utils/optical` | Jones, refractive indices, Zernike and optical phase calculations |
| `utils/propagation` | ABCD, physical propagators and prepared FFT/MFT/ASM operations |
| `fields` | Wavefront, Intensity, PSF and Image |
| `layers` | Wavefront/intensity adapters and local optical/detector effects |
| `systems` | OpticalSystem, DetectorSystem, Imager/Coronagraph/Interferometer and Detector |
| `systems/sources.py` | Source, PointSource, BinarySource, ResolvedSource and Scene |
| `prebuilt` | Def, pupil builders, ZernikeDef and named aperture constructors |
| `compat/`, `utils/compat/` | Supported released constructors/functions and migration errors |

Utility layout follows numerical purpose, not one inheritance hierarchy. There is no
canonical parametrics package. Utilities are exported through dLux and dLux.utils;
current names take precedence over compat names. Layer suffixes distinguish numerical
operators from field adapters. Experimental raytrace/GaussBeam, source_estimation and
calib remain outside the selected release; core modelling must not depend on them.

## Resolution and composition

- Ordinary Module.resolve retains the container and resolves ready children.
- Expression.resolve produces its value when ready, otherwise a partially resolved
  definition. Implement its calculation in evaluate and resolve consumed operands
  there with zdx.resolve. During nested evaluation, an unresolved Expression operand
  raises Unresolved; an ordinary Module may still remain partially resolved.
- Required context belongs in the evaluate signature. For alternatives such as grid
  or coordinates, raise Unresolved when neither is available. Other errors propagate.
- resolve_children retains the Expression while preparing children in the supplied
  context; it does not promise all leaves become arrays. Custom overrides are optional.
  Stored callable operators and unused fields must not determine readiness.
- Do not call self.resolve inside that Expression's evaluate: it re-enters the same
  expression. An ordinary Module operation can use self = self.resolve(...).
- The outer resolve populates links by default. Populate the shared owner before
  separating components that rely on Linked/Deferred siblings. Aliases are user-defined.

Coordinate-changing owners select the local frame before evaluating children. Each
context name has one scientific meaning; do not infer frames from matching names or
shapes. Reusing an incoming name for a different derived input can resolve a child too
early during partial resolution. Prefer distinct names when both must coexist.

Sampled is an Expression with optional UnitSpec grid, callable transform and static
strict=False. sampling_context selects runtime or stored sampling, applies the map
once and passes SI coordinates to the calculation. Runtime input replaces the stored
choice; strict rejects replacing a stored non-None grid/transform, not immutable set
updates. This generates values on coordinates; it does not interpolate stored arrays.
Fixed bases do not need sampling leaves. Spectral evaluation uses wavelengths, not a
pupil's coordinates.

BaseLayer.__call__ delegates to apply(target, **context). Sequential passes an explicit
target through named steps. Series adds Expression behaviour, has no x leaf, and can
resolve(target=array); subclasses may supply their own initial value. Preserve stored
steps as callable operators. Combine instead resolves children independently and reduces
their broadcast values using sum/product/union/intersection.

## Sampling, interpolation and preparation

Physical metadata uses (x, y); spatial arrays end in (ny, nx). An additional image axis
is a batch, not an inferred spatial dimension. Spec defines n/ndim/shape; UnitSpec adds
regular coordinates and units. ResizeSpec, FFTSpec and PasteSpec inherit Spec without
unitful coordinates. Axis is 1D; Grid is always 2D with xy indexing, including partial
Grid definitions. Scalar n/d/c/diam inputs expand over both Grid axes. There is no
separate realised Grid class or grid.broadcast(ndim) step.

Grid/Axis derive coordinates in SI from their stored n/d/c/unit. Stored d, c, fov and
default extent retain their declared units. Explicit Axis vectors must be regular;
irregular numerical knots may stay raw arrays without becoming Specs.

| Operation | Contract |
| --- | --- |
| Grid.resize | New geometry at fixed spacing/centre; parity may change |
| resize_samples / array pad-crop | Retain centre and existing sample positions; reject parity changes |
| oversample/downsample | Change counts and spacing together, retaining extent |
| FFTSpec.c | Fourier output centre, including fractional offsets handled by phase ramps |
| Field | Real/complex samples with a UnitSpec; shared array views and sampling methods |

Interp is the common 1D/2D/3D interpolator. ndim counts trailing sampled axes, with
ordinary leading batches. grid_in accepts sampling or physical-order axis vectors;
grid_out specifies output sampling. Tuple queries describe arbitrary point shapes.
Stacked queries have (..., ndim, *query_shape) with ndim query axes. In 1D a scalar or
vector is also accepted. Raw knots and queries must use consistent units.

Interp defaults to cubic2, complex=True (real/imaginary interpolation), and fill=None
(extrapolation). complex=False interpolates amplitude/phase; a numeric fill supplies
outside values. The low-level interp function retains its linear default. Do not add
normalisation, flux conservation, cell-area corrections or coordinate-Jacobian scaling.
A bound Interp without an output grid, transform or query remains unresolved; explicit
apply can use its input grid. sample supports repeated queries and prepare returns an
explicit Interpax snapshot. Re-prepare after changing its defining inputs.

Positive Affine rotation turns coordinates anticlockwise and sampled objects clockwise,
matching the released convention. Interp/Field use that same coordinate-map sign. This
is not a rule for Jones fast-axis or Gaussian orientation angles.

## Fields, optics and batches

Numerical Jones/ABCD matrices have trailing (..., 2, 2), Mueller matrices (..., 4, 4)
and Stokes vectors (..., 4). Wavefront stores axes as
(*population, *wavelength, [2, 2,] ny, nx). Its jones leaf is None for scalar fields,
a compact matrix for uniform Jones response, or literal True for explicit Jones axes.
Compact source-only matrices need singleton wavelength axes; use Wavefront methods to
align/materialise them. Wavefront stores no photometric weights.

OpticalLayer.apply_mono is the simple extension point for one monochromatic operation.
Wavefront's shared mapping handles population/wavelength axes. Field's numerical array
mapping has a different contract. Whole-field propagation can override apply; do not
repeat Jones-axis inference in every consumer.

Optic inherits BaseOptic, ComplexLayer, RefractiveLayer, JonesLayer and PropagatorLayer.
Its leaves remain flat (opd, phase, thickness, jones, focal_length, etc.). Local scalar
effects combine before Jones operations; normalisation precedes Jones losses.
grid_out=None stops after local effects; an output grid enables propagation. A local
grid/transform supplies expression coordinates without interpolating supplied arrays.
Focal length configures the relay rather than adding a second local lens phase.

Numerical batch axes have no physical meaning. Use explicit singleton axes, the helpers
in batching.py, or vmap for independent populations; equal lengths do not establish
pairing. ParametricBasis.coeffs_ndim counts intrinsic coefficient axes. SparseOptic
selects an optional leading aperture axis before local evaluation and interferes
aperture fields coherently. Apertures do not pair with wavelengths. Do not change its
unsettled broader contract as part of an unrelated batching cleanup.

## Bases and construction

Basis owns coeffs and sampled/generated modes. axes=None infers supplied coefficient
array dimensions; explicit axes preserves preceding coefficient batches. Expression
coefficients need known topology without constructor evaluation. Omitted coefficients
become stored zeros when shape is known; empty=True requests deferred coefficients.
Fixed Basis/FourierBasis/PastedBasis have no sampling-grid leaves. Their optional unit
is a string scaling output after contraction, with inverse scaling for coefficient
solves. Unitful declares/validates units; conversions use unit_factor directly, without
scale/factor properties. Refractive scale is an existing wavelength convention, not
an output-unit conversion.

Def in prebuilt/definitions.py is a constructor recipe, not a runtime optical layer.
ZernikeDef constructs Basis: supplied sampling calculates modes once; absent sampling
or dynamic=True retains a Zernike Expression. Optic consumes Def inputs during
construction. HSTAper/JWSTAper/JWSTNRMAper/EuclidAper construct Optic or SparseOptic,
not Imager. Zernike geometry/support and Basis coefficient units are distinct.

## Propagation

Physical BasePropagator subclasses are Modules. Calls take (array, grid_in, wavelength);
return_grid=True also returns the realised grid. prepare returns a numerical operation,
often MFT or Series, for those inputs without mutating the definition. Re-prepare after
input changes; prepare inside differentiation to retain those dependencies. Preparation
validates required resolved sampling and units; it is not automatic cache invalidation.

Numerical FFT/MFT/Chirp use right-aligned leading batches. Wavefront.propagate aligns
wavelength/population axes and shares a preparation across Jones components. Numerical
operations may introduce broadcast batches; the field adapter preserves its existing
batch contract. Do not infer parameter roles by tree-walking names or equal lengths.

Supplied ABCD determines direction and stores inverse=None; ordinary forward propagation
uses inverse=False. ABCD.inverse means optical traversal reversal, not matrix inversion.
Verify complex phase and realised sampling, not intensity alone. FreeSpace prepares the
physical recipe; ASM is the array operation. PropagatorLayer inherits Propagator fields
and adapts Wavefront input/output; neither it nor Optic nests a propagator leaf.

## Sources and photometry

Source stores one BaseSpectrum and positions resolving to (C, 2) radians. Brightness
lives at spectrum.flux; the Source flux constructor argument sets that leaf. A scalar
applies to every position. Omitted positions are on-axis. PointSource defaults to
arcseconds; raw Source positions and Point default to radians. Positions/Binary/
Companions use arcseconds by default. Binary position locates the primary and PA runs
from +y towards +x in degrees. Contrast defaults to log10(companion/primary), or a direct
ratio with log=False. Scene groups additive named sources with optional offsets.

BaseSpectrum owns sample/integrate and flux. Spectrum wraps arrays or Expressions;
PolySpectrum uses a unit baseline and fixed normalised wavelength coordinates;
Blackbody supplies photon radiance. Numeric flux gives incident photons in the current
calculation BandPass; None keeps absolute density. Flux supports photons only. Weights
are integrated results and cannot be fed to Source as density spectra.

BandPass uses static Python endpoints, equal-width n bins and a unit. range denotes
outer edges; wavelengths are centres. It determines Imager's optical sampling while
Spectrum/Filter/QE can retain finer native sampling. Raw native knots may be irregular.
Do not restore n_optical, units dictionaries or a second Imager wavelengths leaf.

| Spectral operation | Output |
| --- | --- |
| spectrum.sample(wavelengths) / spectrum(wavelengths) | Unnormalised point density shape |
| spectrum(BandPass) | Integrated Weights, with flux applied |
| spectrum.integrate(BandPass) | Total incident count |
| Filter.sample | Dimensionless response values |
| Filter(weights) | Counts multiplied by centre-sampled response |
| Filter.integrate(BandPass) | Effective bandwidth in metres |
| Photometry(spectrum, filter=..., qe=...)(BandPass) | Integrated density × responses |

Sampled curves default to cubic2. Native support boundaries and knots split integration
intervals; original interpolants are evaluated within their support. BandPass-backed
curves extend to its edges and are zero outside; raw native vectors stop at first/last
knots. Separate curve integrals must not be multiplied. Losses remain after incident
flux assignment. Prepared responses are snapshots; analytic narrow features still need
adequate integration sampling.

## System stages

| Entry point | Result |
| --- | --- |
| OpticalSystem.propagate/model | Arrays by default; explicit PSF/Wavefront return flags |
| Imager.wavefront / psf | Unweighted optical responses, with optical losses retained |
| Imager.photometry | Integrated flux/filter/QE weights, without optical propagation |
| Imager.intensity | Fine expected signal, including profiles/jitter and field resampling |
| Imager.model / image / call | Detector Image |
| Imager.simulate | Photon/read-noise realisation, then gain |
| DetectorSystem | Standalone Intensity → Intensity sequence |

OpticalSystem retains source modelling and shares numerical stages with Imager; it is
not just a sequence lacking model methods. Source.model is a deprecated forwarding
adapter, not another canonical modelling engine. Imager stages accept optional public
overrides; internal stages receive selected concrete inputs. Populate Linked owners
before separating components. image_grid denotes native pixels, oversample refines
propagation, and local pupil/plane slots must have grid_out=None to avoid a second relay.

Source lists/Scenes add incoherent contributions. Named input dictionaries request
independent outputs: match keys, share scalars, and fall back to stored defaults for
missing keys. There is no Cartesian product. None overrides keep stored values; clear
optional settings with an immutable update. Imager PSFs use (C, L, ny, nx), with aperture
axes summed coherently first; they exclude source flux, filter and QE weighting.

Detector owns resampling, deterministic layers, downsampling, noise and gain. QE enters
Photometry. Response layers act on expected intensity before pixel integration; simulate
then draws Poisson counts, adds read noise and applies gain. Do this once after combining
sources. Pixel-unit jitter uses native pixels before further detector binning. Image
stores arrays and sampling, without uncertainty/noise-model metadata.

Gaussian functions are fixed continuous shapes. Profile owns sampled pixel masses and
optional normalisation; its Gaussian handling supplies the discrete zero-width limit.
Jitter is an Expression using Profile on displacement coordinates. Resolved morphology
convolves fine point-response intensity before complex detector resampling. Pupil spatial
Jones is supported without reconstruction; scalar/compact-Jones resampling uses an
effective field retaining the point-response phase. Explicit Jones reconstruction is
not supported in that path. Resolved wavefront output remains the point response;
Coronagraph rejects resolved convolution. Shared-PSF Fourier shifting is not implemented.
