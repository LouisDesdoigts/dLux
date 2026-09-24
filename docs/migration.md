# Migrating from dLux 0.15.1 to 0.16

The main changes are explicit sampling with `Grid`, array parameterisations inside
`Optic`, and separate optical, photometric and detector stages. `OpticalSystem`
retains `.propagate()` and `.model()` with array returns by default. `Imager`
adds stored source, bandpass and detector settings with methods for each stage.

## Start with the compatibility constructors

For an existing script, use the released namespace explicitly:

```python
import dLux.compat as old
import dLux.utils.compat as old_utils
import jax.numpy as np

optics = old.AngularOpticalSystem(
    wf_npixels=16, diameter=1.0,
    layers=[old.CircularAperture(radius=0.5, normalise=True)],
    psf_npixels=8, psf_pixel_scale=0.1,
)
source = old.PointSource(wavelengths=np.array([600e-9]), flux=1000)
image_array = optics.model(source)
```

These constructors translate supported arguments into current objects. They do not
restore the old class hierarchy, parameter paths or every Wavefront method.
The [compatibility reference](API/compat/overview.md) lists the retained entry points.
Deprecation warnings identify replacements and schedule removal for **0.17.0**.
Python may hide `DeprecationWarning`; use `python -W default your_script.py` when
migrating. Some retained functions and direct aliases do not emit their own warning.

The declared requirements are Python 3.10 or later, Zodiax 0.6.0 or later, and
Interpax. Use the declared dependency versions rather than an older Zodiax checkout.

Current names take precedence in `import dLux as dl`, including `Spectrum`,
`PolySpectrum`, `FourierBasis`, `FFT`, `MFT` and the numerical propagators.
Old code using those names must choose `compat` explicitly. Utilities are also
available directly from `dLux`; `dLux.utils` remains their implementation namespace.

## OpticalSystem: keep the familiar entry point

Replace scalar entrance sampling with a Grid and describe the final relay explicitly.
This is the current equivalent of a simple layered imaging system:

```python
import dLux as dl
import jax.numpy as np

pupil_grid = dl.Grid(16, diam=1.0, unit="m")
image_grid = dl.Grid(8, d=0.1, unit="arcsec")
pupil = dl.Optic(transmission=dl.Circle(1.0), normalise=True)
optics = dl.OpticalSystem({
    "pupil": pupil,
    "focus": dl.Propagator(grid_out=image_grid),
}, grid=pupil_grid)

psf_array = optics.propagate(np.array([550e-9, 650e-9]))
psf = optics.propagate(np.array([550e-9, 650e-9]), return_psf=True)
band = dl.BandPass(range=(550, 650), n=2, unit="nm")
source = dl.PointSource(flux=1000)
image_array = optics.model(source, bandpass=band)
```

- `propagate_mono(wavelength)` uses one wavelength in metres.
- `propagate(wavelengths, weights=...)` sums point responses; omitted weights sum
  to one, and explicit weights retain their supplied total. Offsets use radians.
- `model(source, bandpass=...)` includes source brightness and resolved morphology.
  It does not apply a detector.
- Both modelling routes return arrays by default. `return_psf=True` retains a PSF
  container; `return_wf=True` requests complex fields instead. The latter is not
  an intensity representation for a resolved source.
- `optics(wavefront)` applies the sequence to an existing Wavefront.

Summed wavelengths must share an output grid. Use an explicit angular image grid
for broadband imaging; summing native FFT outputs with different pixel scales is
not a physically equivalent replacement.

## Imager: inspect the modelling stages

An Imager owns the final image grid and derives its relay. Its pupil must not
also specify `grid_out`, which would propagate twice.

```python
imager = dl.Imager(
    pupil_grid, image_grid, pupil=pupil, source=source, bandpass=band,
    filter=dl.Filter(0.8), detector=dl.Detector(qe=0.9, read_noise=2.0),
)
psf = imager.psf()                  # Unweighted source/wavelength responses
weights = imager.photometry()      # Integrated counts after filter and QE
intensity = imager.intensity()     # Combined source prediction before readout
image = imager.model()             # Expected detector image
companion_image = imager.model(source=dl.PointSource(position=[0.2, 0], flux=100))
```

`psf()` returns source-by-wavelength samples `(C, L, ny, nx)`; it does not apply
spectral brightness or filter losses. `model()`/`image()` return an Image, while
`simulate(key, ...)` adds photon and read noise. These return conventions differ
from OpticalSystem's retained array interface.

A source list or `Scene` adds incoherent source contributions. Dictionaries passed
to Imager's modelling methods name separate outputs; matching keys pair inputs,
scalar values are shared, and missing entries use stored defaults. They do not
request a Cartesian product. See the [systems overview](overviews/systems/overview.md)
for detector ordering, overrides and extension methods.

## Spectra, filters and integrated counts

The released `Spectrum(wavelengths, weights)` stores discrete relative weights.
The current `Spectrum(value, bandpass=..., flux=...)` describes a **spectral density
shape**. Do not copy old weights into it and assume the same integration convention.
Keep `old.Spectrum` with a supported compatibility modelling route when exact
released point weights are required.

```python
native = dl.BandPass(range=(500, 800), n=5, unit="nm")
spectrum = dl.Spectrum([1.0, 1.2, 1.8, 1.1, 1.0], bandpass=native, flux=1000)
filter_curve = dl.Filter([0.0, 0.6, 1.0, 0.7, 0.0], bandpass=native)
photometry = dl.Photometry(spectrum, filter=filter_curve, qe=dl.Filter(0.9))
optical_bins = dl.BandPass(range=(500, 800), n=3, unit="nm")
weights = photometry(optical_bins)
counts_per_bin = weights.value
total_counts = weights.integrate()
```

| Operation | Meaning |
| --- | --- |
| `curve.sample(wavelengths)` | Point values; raw wavelengths are metres |
| `spectrum(bandpass)` | Integrated incident `Weights`, including its flux |
| `spectrum.integrate(bandpass)` | Total incident photons |
| `filter_curve.sample(bandpass)` | Response sampled at bin centres |
| `filter_curve(weights)` | Multiply existing counts by centre-sampled response; use Photometry for within-bin integration |
| `filter_curve.integrate(bandpass)` | Response integral in metres, not photon counts |
| `Photometry(spectrum, filter=..., qe=...)(bandpass)` | Integrate the product into detected counts |
| `Weights(values, bandpass)` | Already-integrated counts; no second integration-width factor |

Filter and QE use the same `Filter` response class. Integrate the product of source,
filter and QE; multiplying their separate integrals is not equivalent. Weights are
an intermediate result, not an input Source spectrum. `Spectrum.from_weights`
constructs Weights rather than a density spectrum.

BandPass `range=(low, high)` defines the **outer bin edges**, not the first and last
sample wavelengths. Bins are equal-width; `min`/`max`, `wl`/`delta_wl` and regular
`wavelengths` are alternative constructor inputs. Its endpoints and sample count
are fixed Python metadata. Construct a new BandPass when changing sampling.
The Imager BandPass sets the optical wavelength count; each curve can retain finer
native sampling for integration.

A numeric `Spectrum.flux` is total incident photons in the calculation's BandPass,
before losses. `flux=None` preserves absolute photon density per metre. Only photon
flux units are supported; `flux2mag`/`mag2flux` use an explicit reference flux and do
not supply AB, ST or Vega calibration.

Sampled curves default to `method="cubic2"`; request `"linear"` explicitly for linear
interpolation. Curves with declared sampling are zero outside their support. With
BandPass sampling, the interpolant continues from the outer centres to the declared
edges, then stops. Raw native coordinates (`bandpass=array` or Filter
`knots=array`) use their first and last knots as support. The `wavelengths=`
constructor shortcut instead infers a regular BandPass from sample centres.
Integration respects those boundaries. See the
[photometry overview](overviews/utils/photometry/overview.md).

Current `PolySpectrum(bandpass, coeffs=...)` means `1 + c1*x + c2*x**2 + ...`, with
wavelength mapped to `[-1, 1]` using that fixed BandPass. Released PolySpectrum
coefficients include a constant term and use wavelengths in metres: retain
`old.PolySpectrum` or convert/refit coefficients before changing constructors.
`Blackbody` is a photon-spectrum model. With `flux=None`, it exposes raw photon
radiance; aperture area, solid angle and exposure time are not inferred.
Basis and Polynomial expressions can supply
Spectrum's value; a fixed basis needs native sampling matching its output samples.

## Source positions and brightness

| Input | Current convention |
| --- | --- |
| `PointSource(position=...)` | Arcseconds by default; `unit="rad"` preserves radian inputs |
| `Source(positions=...)` | Raw `(C, 2)` arrays in radians; a scalar flux applies to every component |
| `Positions(value, unit=...)` | Explicit conversion to `(C, 2)` radians; defaults to arcseconds |
| `Point(position, unit=...)` | Low-level single-position expression; defaults to radians |
| `Binary` / `Companions` | Position locates the primary; separation defaults to arcseconds, PA to degrees from +y towards +x |
| `BinarySource(primary_flux=..., contrast=...)` | Primary photon count; default contrast is `log10(companion/primary)` |
| `ContrastFlux(..., log=False)` | Linear companion/primary ratios |

Released binary constructors use midpoint position, radian PA, mean component flux
and primary/secondary contrast. `old.BinarySource` translates those conventions.
Do not transfer its numeric parameters directly into the current constructor.

```python
binary = dl.BinarySource(
    position=[0, 0], separation=0.2, pa=90,
    primary_flux=1000, contrast=-1,       # Companion has 100 photons
)
binary_image = imager.model(source=binary)
```

Brightness is stored at `source.spectrum.flux`; the Source `flux=` argument sets that
leaf. Use a Scene for components with different spectra or profiles. ResolvedSource
needs a Profile or explicit `profile_grid`; a distribution array alone cannot state
its angular scale. `source.model(optics)` remains a deprecated forwarding method;
call the system's modelling method instead.

## Apertures and fitted coefficients

Replace specialised basis layers with a Basis in an ordinary Optic. Coefficients
live at `opd.coeffs` (or `phase.coeffs`), not at a layer-level `coefficients` leaf.

```python
opd = dl.Basis(dl.Zernike([4, 5, 6], diameter=1.0), unit="nm")
aberrated = dl.Optic(transmission=dl.Circle(1.0), opd=opd, normalise=True)
aberrated = aberrated.set("opd.coeffs", np.array([10.0, 0.0, -5.0]))
aberrated_image = imager.model(pupil=aberrated)
```

Coefficients default to a fit-ready zero array when their shape is known. Basis's
`unit` scales the contracted output; it does not change coordinate units.
Use `ZernikeDef(...)(grid)` to construct a sampled Basis, or retain a Zernike
expression for runtime generation. Current `FourierBasis` produces arrays; wrap it
in `Optic(opd=...)` to replace the released FourierBasis layer.

| Released construction | Current construction |
| --- | --- |
| `CircularAperture(radius=r, softening=s)` | `Optic(transmission=Circle(2*r, edge=Soft(s)))` |
| `RegPolyAperture(nsides=k, rmax=r)` | `Optic(transmission=RegularPolygon(k, 2*r, edge=Soft(1)))` |
| `BasisOptic(basis, coefficients=c)` | `Optic(opd=Basis(basis, coeffs=c))` |
| `AberratedAperture(aperture, nolls, coefficients)` | Explicit transmission and `Basis(Zernike(...), coeffs=...)`, with diameter/support |
| Transmission-only `CompoundAperture` / `MultiAperture` | `Optic(transmission=Combine(shapes, operation="product" / "sum"))` |
| Telescope pupil helpers | `HSTAper`, `JWSTAper`, `JWSTNRMAper`, `EuclidAper`, supplied a pupil Grid |

Shape size and coordinate values use metres when evaluated on a length-valued Grid.
Current shapes default to hard edges; the released dynamic apertures default to
softening. Polygon diameter is the enclosing-circle diameter. Named pupil helpers
return Optic or SparseOptic, not Imager. Their default normalisation sets transmitted
pupil power to one; subsequent losses still reduce the output. Revisit that choice
when fitting throughput. Aberrated composite apertures require explicit local
basis/support definitions; compat does not infer them.

## Grids, fields and numerical operators

Grid is strictly two-dimensional with physical `(x, y)` metadata and `xy` indexing;
arrays end in `(ny, nx)`. Scalar counts expand to a square grid. Use Axis for 1D
sampling. Partial Grids are allowed until an operation needs complete sampling.
Grid has no dimensional broadcasting method; use explicit sampling and batch axes.

Fields store `array` and `grid`; `PSF.data` and `Wavefront.phasor` are views of that
array. Immutable updates should target stored leaves, such as `array` and `grid.d`.
Wavefront owns wavelengths and Jones state, but no photometric weights.
Its axis order is `(*batch, *wavelength, ny, nx)`, with `(2, 2)` inserted immediately
before the spatial axes when Jones fields are explicit. Uniform Jones matrices can
stay separate; `materialise()` creates their spatial representation.

| Released use | Current replacement |
| --- | --- |
| Scalar field sampling | `Wavefront(wavelength, Grid(...))`; `PSF(data, Grid(...))` |
| `CoordSpec` / `PadSpec` | `Grid` / `FFTSpec` |
| `CoordTransform` / `DistortedCoords` | `Affine` or `AffineMap` / `Distortion` |
| `Resize`, `Downsample`, `Flip`, `Lambda`, `Normalise`, `Tilt` layers | `ResizeLayer`, `DownsampleLayer`, `FlipLayer`, `LambdaLayer`, `NormaliseLayer`, `TiltLayer` |
| `Rotate(angle)` | `InterpolateLayer(Affine(rotation=angle), method="linear", complex=False, fill=0)` |
| Released FFT/MFT array functions | `old_utils.FFT` / `old_utils.MFT`, or current prepared operators |
| `ABCDFreeSpace(d)` / `ABCDLens(f)` / `ABCDMirror(r)` | `ABCD(free_space=d)` / `ABCD(lens=f)` / `ABCD(radius=r)` |
| `ABCDConjugatePlane(f)` | `ABCD(fraunhofer=f)` |
| `ABCDPropagator(elements, grid)` / `MFTPropagator(elements, grid)` | `Propagator(grid, abcd=ABCDChain(elements))` |
| `ASMPropagator(distance, spec)` | `FreeSpace(distance, resize=...)` with compatible ResizeSpec |
| Propagation on Wavefront | `wavefront.propagate(Propagator(grid_out))` |
| Propagation as a layer | `PropagatorLayer(grid_out=...)`, or a numerical propagator in OpticalSystem |

A numerical propagator is called as `relay(array, grid_in, wavelength)`;
`return_grid=True` also returns the realised Grid. `prepare(grid_in, wavelength)`
returns a fixed operator for reuse. Re-prepare after changing its inputs; prepare
inside differentiation when fitting them. For a supplied ABCD matrix, direction
is in the matrix and the propagator stores `inverse=None`. ABCD's `.inverse()`
reverses optical traversal; use array linear algebra for a matrix inverse.

Interp performs value interpolation without pixel-area or distortion compensation.
It defaults to `cubic2`, complex real/imaginary interpolation, and `fill=None`
(extrapolation). Use `fill=0` where an image should be zero outside its support.
In 2D/3D, tuples denote arbitrary query points; stacked arrays denote grids.
Padding/cropping preserves centres and rejects odd/even size changes that would
introduce a half-pixel shift. Resampling is a separate interpolation operation.

## Detectors and custom layers

| Released detector operation | Current replacement |
| --- | --- |
| `ApplyPixelResponse(response)` | `SensitivityLayer(response)` |
| `ApplySaturation(threshold)` | `SaturationLayer(threshold)` |
| `AddConstant(value)` | `BiasLayer(value)` |
| `ApplyJitter(sigma, kernel_size=k, oversample=o)` | `ConvolveLayer(Jitter(sigma_pixels, k, o))`, with `sigma_pixels = sigma*(k*o - 1)/(10*o)` |
| `LayeredDetector(layers)` | `DetectorSystem(layers)` for deterministic layers, or `Detector(layers)` for sampling/noise/gain |

Current Jitter uses actual pixel widths; compat converts the released fixed-domain
Gaussian width during evaluation. Passing the old sigma unchanged to Jitter changes
the blur. Imager also accepts `jitter=Jitter(...)` for image-plane blur.

DetectorSystem takes and returns Intensity. Detector owns QE, optional field
resampling, deterministic layers, pixel integration, noise and gain. See the systems
overview for the exact order.

Custom optical layers implement `apply_mono`; OpticalLayer handles the Wavefront
batches. Ordinary model objects inherit `zodiax.Module`.

```python
from jax import Array

class PhaseOffset(dl.OpticalLayer):
    phase: Array

    def __init__(self, phase):
        self.phase = np.asarray(phase)

    def apply_mono(self, wavefront, **context):
        return wavefront.add_phase(self.phase)

shifted = PhaseOffset(0.1)(dl.Wavefront(600e-9, pupil_grid))
```

Custom detector layers implement `apply(intensity, **context)`; unified field
operations implement `apply_mono`. Keep coordinate metadata consistent with changed
sampling and pass context to nested definitions. Automatic descendant shortcuts
and old generated names are not stable migration paths: inspect the current object
and use explicit paths or user-defined aliases for fitted parameters.
For example, migrate `optic.set("coefficients", values)` to
`optic.set("opd.coeffs", values)`, and `source.set("flux", value)` to
`source.set("spectrum.flux", value)`. The old names are not general update aliases.

## Compatibility limits and numerical conventions

The supported translations retain current objects. Code that relies on old class
identities, scalar sampling fields, private helpers, automatic names or old subclass
constructors needs revision. Deprecated factory names cannot serve as `isinstance`
types. `ParametricOpticalSystem`, implicit aberrated-aperture composition and
PointResolvedSource need explicit current modelling setups; Scene replaces mixed
point/resolved composition. Released coordinate shear transposes grid samples and
is not the current pointwise Affine shear; use the explicit compat utility when
that old operation is required.

The released rotation convention and Fourier coefficient mapping are retained.
Positive Affine rotation turns coordinates anticlockwise and sampled objects
clockwise. General ABCD FFT propagation now agrees with MFT's phase convention;
compare complex fields as well as intensity when migrating interferometric models.
EuclidAper keeps the released spider orientation but uses the current shared edge
sampling, without the old helper's unintended additional circular-edge oversampling.
The retained compat helper is available for reproducing that sampling.

Compatibility is partial, not a guarantee that every 0.15.1 program runs unchanged.
A known limitation remains when differentiating Zernike geometry at the exact
origin: coordinate or diameter gradients can be non-finite even when sampled
values and coefficient derivatives are finite.
