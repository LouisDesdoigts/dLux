# Migrating to dLux 0.16

dLux 0.16 reorganises the package around multidimensional, unit-aware grids; field
containers; parametrics; composable layers; prebuilt optical components; and explicit
optical and detector systems. It also makes vectorisation and unitful behaviour part
of the core modelling contracts. This guide shows how common dLux 0.14 and 0.15
workflows translate to the new API. Compatibility aliases remain where old behaviour
can be preserved and are scheduled for removal in dLux 0.17.

## Grids and coordinates

Older dLux objects each carried their own sampling values. A wavefront, for example,
was constructed directly from a square pixel count and diameter:

```python
# dLux 0.14
wavefront = dl.Wavefront(npixels=128, diameter=1.0, wavelength=650e-9)

# dLux 0.16
grid = dl.GridSpec(n=128, d=0.01, unit="m")
wavefront = dl.Wavefront(650e-9, grid)
```

`GridSpec` is now a core object retained by fields, systems, and propagators. It
centralises sample counts, pixel scales, centres, dimensionality, and physical units,
so sampling changes remain explicit as a field moves through a model. Its `n`, `d`,
and optional `c` values follow physical-axis order and may be multidimensional even
when a square grid is constructed from scalars.

Coordinate transformations now live alongside grids. Replace `DistortedCoords` with
`Distortion`; replace the former concrete `CoordTransform` with `Affine`.

## Fields

Wavefronts, deterministic sampled intensities, and detector images now live in
`dLux.fields`. Construct a wavefront from its wavelength and a grid:

```python
wavefront = dl.Wavefront(650e-9, pupil_grid)
```

Fields retain their `GridSpec`, including its physical unit. Use field methods for
normalisation, tilts, OPD, phase, interpolation, and image simulation rather than
manually updating their sampled arrays.

Wavelength arrays create vectorised wavefronts directly. Optical layers implement a
single monochromatic operation and dLux maps it over leading wavelength and batch
axes:

```python
wavelengths = np.linspace(600, 700, 5) * 1e-9
wavefronts = dl.Wavefront(wavelengths, pupil_grid)
```

`Intensity` replaces the former `PSF` container because extended-source and
detector-domain predictions are not necessarily point-spread functions. Convert
between the deterministic and observed containers explicitly:

```python
intensity = wavefront.to_intensity()
image = intensity.to_image(read_noise=3.0)
```

`PSF` remains a warning-backed constructor alias until dLux 0.17.

## Optical and detector systems

Replace specialised layered, angular, Cartesian, and parametric system classes with
explicit composition:

```python
# dLux 0.15
optics = dl.LayeredOpticalSystem(npix, diameter, layers)

# dLux 0.16
pupil_grid = dl.GridSpec(n=npix, diam=diameter, unit="m")
optics = dl.OpticalSystem(layers, pupil_grid)
```

`OpticalSystem.model(source)` and `source.model(optics)` are equivalent and return an
`Intensity`. Detector systems apply deterministic response layers and also return an
`Intensity`; construct an `Image` explicitly before simulating observations:

```python
intensity = optics.model(source)
intensity = detector.model(intensity)
image = dl.Image(intensity, read_noise=3.0)
```

Detector layers now use concise transformation names:

| Deprecated | Current |
| --- | --- |
| `ApplyPixelResponse` | `Sensitivity` |
| `ApplyJitter` | `Jitter` |
| `ApplySaturation` | `Saturation` |
| `AddConstant` | `Bias` |

`Convolve` supplies arbitrary fixed or parametric kernels, while `Gain` supports
linear arrays and nonlinear parametric gain curves.

## Sources and spectra

`Source` now represents one point source or a vectorised population. Supply scalar or
one-dimensional wavelengths directly and use a spectral parametric for generated
weights:

```python
# dLux 0.14/0.15
source = dl.PointSource(
    wavelengths=np.linspace(600, 700, 5) * 1e-9,
    position=np.array([0.0, 0.0]),
    flux=1e5,
)

# dLux 0.16
source = dl.Source(
    wavelengths=np.linspace(600, 700, 5),
    position=[0.0, 0.0],
    flux=1e5,
    weights=dl.SpectralPolynomial(degrees=1, coeffs=[0.1]),
    units={"wavelengths": "nm", "position": "arcsec"},
)
```

`PointSource`, `PointSources`, and `ResolvedSource` remain warning-backed wrappers.
The former `spectrum=` constructor path cannot always be translated safely; pass its
wavelengths and weights explicitly.

## Parametrics and coefficients

Parametrics are now a core mechanism rather than a collection of specialised layer
types. Any layer value that accepts a `Parametric` can be generated from compact
parameters and evaluation context—for example explicit or implicit bases,
polynomials, interpolation, spectral models, shapes, and custom parameterisations.

Use `dl.Basis` for explicit expansions. Numerical basis evaluation and fitting live
in `dLux.utils`; the dLux parametric owns coefficients and contextual operands:

```python
# dLux 0.14/0.15
optic = dl.BasisOptic(
    transmission=aperture,
    basis=basis,
    coefficients=np.zeros(7),
)

# dLux 0.16
import zodiax as zdx

opd = dl.Basis(basis, coeffs=np.zeros(7), axes=1)
optic = dl.Optic(transmission=aperture, opd=opd)
```

Bases store their physical coefficients in `coeffs`, which can itself be a
Zodiax expression. dLux's `Parametric` inherits `zdx.Expression`; Zodiax no longer
exports a `Parametric` class. Basis families implement `evaluate` and `solve`,
without inheriting Transform chaining methods.

`ZernikeBasis` subclasses `dl.Basis`. `DynamicZernikeBasis` combines the spatial
and `ImplicitBasis` contracts. Spatial definitions use `evaluate(grid=...)` or
`evaluate(coordinates=...)`, replacing the separate `sample` entry point.
Generated basis and coefficient batches are
independent populations; use explicit `jax.vmap` to pair them.

Use `solve` for a linear expansion's minimum-norm coefficients. CLIMB owns its
linear basis as `expansion`; `expansion.solve` operates on the continuous latent
field, and the final binarised output has no inverse. `PastedBasis.solve`
projects each extracted stamp independently and is not a global inverse when
stamps overlap.

Specialised basis and aperture layers were therefore replaced by parametrics placed
directly in general optical layers:

```python
optic = dl.Optic(transmission=aperture, opd=dl.Basis(basis, coeffs, axes=1))
```

### Polynomials and interpolation

Use `dl.Polynomial` and `dl.Interpolation` directly. They share dLux's numerical
kernels and use the public Zodiax expression protocol for stored parameters, aliases,
external state, and composition:

```python
polynomial = dl.Polynomial(degree=2, coeffs=[1.0, 2.0, 3.0])
coordinates = np.array([0.0, 1.0, 2.0])
values = polynomial.resolve(coordinates=coordinates)
coeffs = polynomial.solve(values, coordinates=coordinates)

response = dl.Interpolation([0.0, 1.0, 2.0], [0.0, 1.0, 0.0])
samples = response.resolve(coordinates=[0.5, 1.5])
```

Polynomial term topology comes from exactly one of `degree`, selected `degrees`,
or explicit integer `powers`. Coefficients are stored in `coeffs`, and `solve`
returns minimum-norm least-squares coefficients. No coordinate centring or
normalisation is implicit.

Interpolation accepts explicit `coordinates` through `evaluate` or `resolve`.
The stored table uses `knots` and `values`; the output has the query shape followed
by trailing value axes.
The default is linear interpolation with zero fill
outside the knots; interpolation defines no inverse or representative solve.

### Normalisation

Use `dlu.normalise` for array normalisation:

```python
import dLux.utils as dlu

weights = dlu.normalise(np.array([1.0, 2.0, 3.0]), mode="sum", axis=-1)
```

It supports `sum`, `mean`, `rms`, `l1`, `l2`, `max`, and `p2v`, with explicit
`scale`, `weights`, and reduction `axis`. Weights affect the measured normalisation;
they do not mask the returned array. Normalisation is generally non-injective and
defines no inverse or solve.

Physical ownership remains with the consumer: spectra normalise along wavelength,
builders measure modes on their support, and field layers own optical power.
`dl.Norm` remains the support-aware builder policy and delegates to the same kernel;
`dl.Normalise` remains the optical field layer.

### Composing contextual values

`Combination` broadcasts child values before stacking them. A scalar, a row and a
column can therefore contribute to one image. At least one child is required.
Sum and product reduce arithmetically; union clips the sum to `[0, 1]`, and
intersection multiplies transmissions.

`DynamicParametric` remains the adapter for evaluating an expression on transformed
coordinates. It resolves transform parameters in the incident frame, then passes
the transformed coordinates and external state to the child. A numerical
`zdx.Map` transforms values; it does not replace a child's coordinate context.

### Selected coefficient updates

The unreleased `Selection` factory and `parametrisations` module have been removed.
Compose Zodiax transforms directly for selected coefficient displacements:

```python
import jax.numpy as np
import zodiax as zdx

origin = np.array([1.0, 0.0, 2.0])
mask = np.array([True, False, True])
latent = np.zeros(2)
coeffs = zdx.Map(
    x=zdx.Mask(x=latent, mask=mask),
    b=origin,
    alias={"latent": "x.x", "origin": "b"},
)
updated = coeffs.set(latent=np.array([0.1, -0.2]))
values = updated.resolve()  # [1.1, 0.0, 1.8]
recovered = updated.apply(values, inverse=True)  # [0.1, -0.2]
```

For logarithmic displacements, apply the decoder only to active entries. This
allows inactive entries to remain zero without taking their logarithm:

```python
encoded = zdx.Map(x=latent, b=np.log(origin[mask]))
selected = zdx.Mask(x=zdx.Exp(x=encoded), mask=mask)
coeffs = zdx.Map(
    x=selected,
    b=origin.at[mask].set(0),
    alias={"latent": "x.x.x.x"},
)
```

Mask topology is fixed during construction. The latent array may have leading
batch axes or be a `StateRef`; evaluation and reversal belong to Zodiax.

### Spectral and refractive wrappers

`SpectralBasis` infers coefficient dimensions from every basis axis except the
final wavelength axis. Leading coefficient batches no longer interfere with this
inference, and coefficient expressions can use it too. Explicit `shape` still
selects a contraction prefix. Normalisation acts along the final wavelength axis;
zero default coefficients must be set before normalised evaluation.

`PolynomialIndex` owns its coefficients and integer powers directly, using the same
dLux polynomial kernel. `CauchyIndex` specialises its coordinates to inverse-square
wavelength. Coefficients are stored directly in `coeffs`. Explicit wavelengths
and dispersion scales are in metres; optical layers supply the incident wavelength
context. An expression providing coefficients requires `degree` to fix polynomial
topology at construction:

```python
import dLux as dl

index = dl.CauchyIndex(zdx.StateRef("index_coeffs"), degree=1)
state = zdx.State(index_coeffs=np.array([1.5, 0.004]))
indices = index.resolve(wavelengths=np.array([500e-9, 600e-9]), state=state)
```

`InterpolatedIndex` subclasses `dl.Interpolation`. Its constructor accepts
`wavelengths` and `indices`, stored as `knots` and `values` for parameter updates.
`extrapolate=False` gives the default NaN outside the sampled interval.
`Blackbody` retains the stable photon-number Planck spectrum, with temperature in
kelvin and optional external-state resolution. Spectral normalisation remains an
equal-sample convention, not wavelength quadrature.

## Apertures and shapes

Grid-aware builders and prebuilt templates are now the standard construction route
for canonicalising optical components. `build()` returns explicit sampled data for
inspection or custom assembly; calling the builder returns a ready-to-use layer:

```python
# dLux 0.14/0.15: dynamically evaluated layer
aperture = dl.CircularAperture(radius=0.5, softening=1.0)

# dLux 0.16: reusable geometry and grid-aware construction
builder = dl.ApertureBuilder(primary=dl.Circle(1.0))
optic = builder(pupil_grid)

# Named telescope-like components follow the same builder contract
builder = dl.JWSTLike(opd=dl.ZernikeDef(orders=2))
aperture, basis = builder.build(pupil_grid)
optic = builder(pupil_grid)
```

Hard and softened geometry share the same shape classes. Pass a soft edge explicitly
when dynamic edge gradients are required:

```python
hard = dl.Circle(1.0)
soft = dl.Circle(1.0, edge=dl.Soft(pixels=1.0))
```

## Propagation

Propagation is now expressed by explicit layer classes. Use `Fraunhofer`, `Fresnel`,
or `FreeSpace` for direct physical propagation, and compose `ABCDFreeSpace`,
`ABCDLens`, `ABCDMirror`, and `ABCDFraunhofer` inside `ABCDPropagator` for paraxial
systems:

```python
# dLux 0.14/0.15
propagator = dl.MFT(npixels=96, pixel_scale=20 * dlu.mas2rad)

# dLux 0.16
focal_grid = dl.GridSpec(n=96, d=20.0, unit="mas")
propagator = dl.Fraunhofer(focal_grid, method="mft")
```

For a paraxial train, replace specialised ABCD propagator wrappers with an explicit
sequence whose physical meaning remains visible:

```python
relay_grid = dl.GridSpec(n=96, d=10.0, unit="um")
relay = dl.ABCDPropagator(
    [dl.ABCDFreeSpace(0.5), dl.ABCDLens(0.5), dl.ABCDFreeSpace(0.5)],
    relay_grid,
)
```

The former `MFT`, `FFT`, `MFTPropagator`, `FFTPropagator`, and
`ABCDConjugatePlane` names remain compatibility wrappers where their behaviour maps
cleanly. `ASMPropagator` has no direct wrapper; use `FreeSpace`. Reverse propagation
is supported only where the propagator's physical contract defines it.

## Layers and custom extensions

Custom optical layers previously implemented `apply` directly. They now implement the
monochromatic operation in `apply_mono`; the base class owns ordinary leading-axis
vectorisation through `apply`, while `__call__` provides the normal user-facing
syntax:

```python
import jax.numpy as np
from jax import Array

import dLux as dl


# dLux 0.14/0.15
class OldPhaseOffset(dl.OpticalLayer):
    def apply(self, wavefront):
        return wavefront.add_phase(self.phase)


# dLux 0.16
class PhaseOffset(dl.BaseOpticalLayer):
    phase: Array

    def __init__(self, phase):
        self.phase = np.asarray(phase)

    def apply_mono(self, wavefront):
        return wavefront.add_phase(self.phase)

layer = PhaseOffset(0.1)
output = layer(wavefront)
```

Do not manually batch ordinary wavelength axes in a custom optical layer.

## Removed contracts

Interfaces such as `BasisLayer`, `BasisOptic`, `ParametricOpticalSystem`,
`ParametricLayeredOpticalSystem`, `Scene`, and the specialised aperture-layer classes
changed too substantially for a safe alias. Instantiating them raises an error
containing the current replacement and a migration example. Custom subclasses of the
former `OpticalSystem` base should instead subclass the layer or system contract they
actually implement. The complete compatibility surface is available in the
[compatibility API](API/core/compatibility.md).

If a migration error does not explain your use case, please open an issue with a
minimal old-version example. That helps distinguish a missing compatibility route from
a deliberately changed physical contract.
