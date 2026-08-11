# Migrating to dLux 0.16

dLux 0.16 reorganises the package around multidimensional grids, field containers,
parametrics, composable layers, and explicit optical and detector systems. This guide
covers the common migrations from dLux 0.14 and 0.15. Compatibility aliases remain
available where the old behaviour can be preserved and are scheduled for removal in
dLux 0.17.

## Grids and coordinates

Use `GridSpec` for sampled grids and `ResizeSpec` for FFT padding and cropping:

```python
# dLux 0.14/0.15
grid = dl.CoordSpec(n=128, d=0.01)
resize = dl.PadSpec(pad=2, crop=1)

# dLux 0.16
grid = dl.GridSpec(n=128, d=0.01, unit="m")
resize = dl.ResizeSpec(pad=2, crop=1)
```

`GridSpec` is multidimensional. Its `n`, `d`, and optional `c` values therefore have
one entry per physical axis, even when constructed from scalar inputs. Coordinate
transformations now live in `dLux.grids`; use `DistortCoords` in place of
`DistortedCoords`.

## Fields

Wavefronts, deterministic sampled intensities, and detector images now live in
`dLux.fields`. Construct a wavefront from its wavelength and a grid:

```python
wavefront = dl.Wavefront(650e-9, pupil_grid)
```

Fields retain their `GridSpec`, including its physical unit. Use field methods for
normalisation, tilts, OPD, phase, interpolation, and image simulation rather than
manually updating their sampled arrays.

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

Use `Basis` instead of `ExplicitBasis`, and use `coeffs` for parameter leaves and
constructor keywords:

```python
# Deprecated
opd = dl.Basis(basis, coefficients=np.zeros(7))
values = opd.coefficients

# Current
opd = dl.Basis(basis, coeffs=np.zeros(7))
values = opd.coeffs
```

The `coefficients` aliases warn and remain available until dLux 0.17. Descriptive
scientific prose still uses the word “coefficients”; only public identifiers and
parameter paths use `coeffs`.

Specialised basis and aperture layers were replaced by parametrics placed directly in
general optical layers:

```python
optic = dl.Optic(transmission=aperture, opd=dl.Basis(basis, coeffs=coeffs))
```

## Apertures and shapes

Grid-aware builders separate sampled-array construction from layer construction:

```python
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

Use `Fraunhofer`, `Fresnel`, `FreeSpace`, or `ABCDPropagator` with an explicit output
grid or resize specification:

```python
focal_grid = dl.GridSpec(n=96, d=20.0, unit="mas")
propagator = dl.Fraunhofer(focal_grid, method="mft")
```

The former `MFT`, `FFT`, and ABCD propagator names remain compatibility wrappers where
their behaviour maps cleanly. `ASMPropagator` has no direct wrapper; use `FreeSpace`
with an appropriate `ResizeSpec`. Reverse propagation is supported only on propagators
whose public contract defines it.

## Layers and custom extensions

Optical layers implement a monochromatic transformation in `apply_mono`. The base
class owns ordinary leading-axis vectorisation through `apply`, while `__call__`
provides the normal user-facing syntax:

```python
import jax.numpy as np
from jax import Array

import dLux as dl


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
