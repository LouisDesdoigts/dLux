# Optical definitions

These utilities calculate polarisation matrices, pupil modes and refractive
indices. Their parameters can remain in a model until the calculation needs arrays.

## 1. A polariser and a waveplate

Named Jones constructors retain the physical parameters of common elements.

```python
import jax.numpy as np
import dLux.utils as dlu

polariser = dlu.Jones.LinearPolariser(angle=0.0)
plate = dlu.Jones.HalfWavePlate(angle=0.2)
print(polariser)
print(plate)
print(polariser.resolve())
```

```text
Jones(angle=f32[])
Jones(angle=f32[], retardance=f32[])
[[1. 0.]
 [0. 0.]]
```

Angles and retardances are radians. Angle is the transmission axis for a linear
polariser and the fast axis for a retarder. `retardance=None` selects a polariser;
a numerical retardance selects a retarder. An omitted angle defaults to zero;
explicit `angle=None` is invalid.

| Constructor | Array helper | Default result |
| --- | --- | --- |
| `LinearPolariser(angle=0)` | `linear_polariser(angle=0)` | Jones |
| `HorizontalPolariser()` | `horizontal_polariser()` | Jones |
| `VerticalPolariser()` | `vertical_polariser()` | Jones |
| `Retarder(retardance, angle=0)` | `retarder(retardance, angle=0)` | Jones |
| `HalfWavePlate(angle=0)` | `half_wave_plate(angle=0)` | Jones |
| `QuarterWavePlate(angle=0)` | `quarter_wave_plate(angle=0)` | Jones |
| `Identity()` | `identity()` | Jones |
| `RHCPolariser()` | `rhc_polariser()` | JonesMatrix |
| `LHCPolariser()` | `lhc_polariser()` | JonesMatrix |

These methods belong to `dlu.Jones`. The first seven constructors accept
`matrix=True` to return a JonesMatrix containing the realised values. Lowercase
helpers always return arrays. Circular polarisers require a general matrix.

## 2. Composition and spatial variation

Composition returns the equivalent matrix. Rotation of a uniform Jones definition
retains its angle and retardance representation.

```python
combined = polariser.then(plate)
rotated = plate.rotate(0.1)
print(combined)
print(rotated)

plates = dlu.Jones.HalfWavePlate(angle=np.array([0.0, 0.2]))
print(plates.resolve().shape)
```

```text
JonesMatrix(matrix=c64[2,2])
Jones(angle=f32[], retardance=f32[])
(2, 2, 2)
```

`polariser.then(plate)` means the polariser acts first: `plate @ polariser`.
The combined object stores matrix entries; recombine the definitions when fitting
physical parameters. JonesChain retains named definitions for repeated composition.

Rotating a half-wave plate changes its axis, preserving its half-wave retardance.
JonesMatrix.rotate instead rotates a uniform matrix and returns JonesMatrix.
Spatial patterns require coordinate rotation and resampling; neither rotate method
performs that operation. Array-valued Jones parameters still describe local axes
naturally, with output shape `(*parameter_shape, 2, 2)`.

`.apply(matrix)` applies the operator, `.mueller()` gives the equivalent Mueller
matrix, and `.to_stokes(stokes=None)` returns output Stokes `[I, Q, U, V]`.
Omitted Stokes means unpolarised unit intensity.

## 3. Generate pupil modes

Zernike generates dimensionless circular or polygonal modes. Basis supplies the
coefficients and converts the resulting OPD from its declared unit.

```python
grid = dlu.Grid(8, diam=1.0, unit="m")
modes = dlu.Zernike([4, 5, 6], diameter=1.0)
opd = dlu.Basis(modes, coeffs=[10.0, 0.0, 0.0], unit="nm")
print(opd.coeffs)
print(opd.resolve(grid=grid).shape)
```

```text
[10.  0.  0.]
(8, 8)
```

`modes.resolve(grid=grid)` returns the mode arrays before contraction. Supply those
arrays to Basis for precomputed geometry, or retain the expression to fit geometry.
`n_sides=6` selects the polygonal mapping. `support` supplies an aperture mask and
`norm` selects optional support-based normalisation. Neither sets the OPD unit.
Noll indices, names and polynomial-factor helpers are available on Zernike itself.

## 4. Wavelength-dependent refractive index

An index definition evaluates at the wavelengths requested by an optic.

```python
index = dlu.CauchyIndex([1.5, 0.004])
print(index)
print(index.resolve(wavelengths=np.array([500e-9, 600e-9, 700e-9])))
```

```text
CauchyIndex(coeffs=f32[2], powers=i32[1,2], scale=f32[])
[1.516     1.5111111 1.5081632]
```

CauchyIndex uses powers of inverse-square wavelength relative to its metre scale;
PolynomialIndex uses direct powers. InterpolatedIndex samples tabulated indices
through Interp. These return dimensionless index values; an Optic's thickness
supplies the path length.

## Contracts

| Operation | Input → output |
| --- | --- |
| Jones resolution | Angle/retardance → `(..., 2, 2)` |
| `then(other)` | Two operators → JonesMatrix storing `other @ self` |
| JonesChain resolution | Named definitions → current matrix product |
| Zernike resolution | Grid/coordinates → `(modes, ny, nx)`, or `(apertures, modes, ny, nx)` with centres |
| Basis of Zernikes | Coefficients + modes → OPD in metres when a length unit is declared |
| Index resolution | Metre wavelengths → dimensionless refractive indices |

See the [generated inheritance map](../../../API/utils/optical.md) for the
current classes and their methods. [Basis examples](../bases/overview.md) show how
sampled and generated modes enter the same optical model.
