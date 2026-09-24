# Prebuilt pupils

## 1. Construct a sampled aperture

The aperture convenience functions return ordinary Optic objects. They prepare pupil
geometry; the Imager separately defines image sampling and propagation.

```python
import dLux as dl

pupil_grid = dl.Grid(16, diam=2.5, unit="m")
pupil = dl.HSTAper(pupil_grid, pupil_oversample=1, jit=False)
print(pupil)

imager = dl.Imager(
    pupil_grid, dl.Grid(8, d=0.05, unit="arcsec"), pupil,
    source=dl.PointSource(flux=1000), bandpass=dl.BandPass(600, unit="nm"),
)
print(imager.model())
```

```text
Optic(
  transmission=f32[16,16],
  normalise=True,
  grid=Grid(unit='m', n=(16, 16), d=f32[2])
)
Image(array=f32[8,8], grid=Grid(unit='arcsec', n=(8, 8), d=f32[2]))
```

HSTAper, JWSTAper, JWSTNRMAper and EuclidAper provide representative geometries.
They are convenient pupil models, not calibrated instrument simulators. Default
normalisation sets one photon of power after transmission through that pupil.
`pupil_oversample` controls edge sampling, independently of Imager's image oversampling.

## 2. Construct an OPD basis

ZernikeDef is a constructor recipe. Supplying a grid produces a Basis with sampled
modes; without a grid it retains a Zernike expression as the Basis's mode definition.

```python
recipe = dl.ZernikeDef([4, 5, 6], diameter=2.4, unit="nm")
opd = recipe(pupil_grid)
print(opd)
print(opd.coeffs)
```

```text
Basis(unit='nm', coeffs=f32[3], basis=f32[3,16,16])
[0. 0. 0.]
```

Omitted coefficients become a zero array ready to fit. Geometry belongs to Zernike;
coefficients and their output unit belong to Basis. Definitions are consumed during
construction, rather than applied as optical layers at runtime.

## Builders

| Builder | Constructed geometry |
| --- | --- |
| `SimpleCircular` | Circular primary, secondary and support vanes |
| `SegmentedHex` | Selected rings of hexagonal segments |
| `NRMLike` | Positioned apertures for a non-redundant mask |
| `ApertureBuilder` | A shared shape, optional obscurations and OPD recipe |
| `SparseApertureBuilder` | Separate local apertures, optionally assembled into a dense pupil |

SparseOptic keeps local aperture fields separate until coherent propagation and
summation. Dense Optic stores a common pupil array. Builders own stamp preparation;
users need not construct PasteSpec or PastedBasis directly.

[Class map and constructor reference](../../API/prebuilt/overview.md).
