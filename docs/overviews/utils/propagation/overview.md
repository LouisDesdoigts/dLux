# Propagation

## 1. Propagate to an image grid

Propagator uses a matrix Fourier transform to connect specified input and output
sampling. It accepts arrays; Wavefront adds the physical wavelength and field metadata.

```python
import dLux as dl
import jax.numpy as np

pupil_grid = dl.Grid(8, diam=1.0, unit="m")
image_grid = dl.Grid(10, d=0.1, unit="arcsec")
pupil = dl.Circle(1.0).resolve(grid=pupil_grid).astype(complex)
relay = dl.Propagator(grid_out=image_grid)
output, sampling = relay(pupil, pupil_grid, 600e-9, return_grid=True)
print(relay)
print(output.shape)
print(sampling)
```

```text
Propagator(grid_out=Grid(unit='arcsec', n=(10, 10), d=f32[2]))
(10, 10)
Grid(unit='arcsec', n=(10, 10), d=f32[2])
```

Angular image sampling uses wavelength to set the Fourier scale. Length-valued image
sampling additionally needs a focal length or a suitable ABCD matrix. Array axes end
in `(ny, nx)`; preceding numerical batch axes broadcast normally.

## 2. Prepare fixed numerical work

Preparation produces an array operator for specific sampling and propagation parameters.
The dynamic call and prepared operator perform the same calculation.

```python
prepared = relay.prepare(pupil_grid, 600e-9)
print(prepared)
print(np.allclose(prepared(pupil), output))
```

```text
MFT(kernels=(c64[10,8], c64[10,8]))
True
```

An MFT stores separable matrices; FFT-based preparations can return a Series of resizing,
phase and Fourier operations. Re-prepare after changing wavelength, grids or relay
parameters. Prepare inside a differentiated function when those parameters are fitted.

## 3. Choose the propagation model

| Class | Use |
| --- | --- |
| `Propagator(grid_out, ...)` | MFT relay to requested output coordinates |
| `FFTPropagator(...)` | FFT relay; sampling follows the transform and FFTSpec |
| `FreeSpace(distance, ...)` | Paraxial free-space propagation, including angular-spectrum preparation |
| `LCT(abcd, grid_out)` | Linear canonical transform defined by an ABCD matrix |
| `SpectralPropagator(...)` | LCT factored through explicit or FFT frequency sampling |
| `ThinPower(power)` | Local quadratic phase, without changing sampling |

ABCD definitions can be retained until their parameters are resolved.

```python
space = dl.ABCD(free_space=0.1)
print(space)
print(space.resolve())
```

```text
ABCD(free_space=f32[])
[[1.  0.1]
 [0.  1. ]]
```

ABCD and Jones matrices both use `(..., 2, 2)`. `ABCD.inverse()` reverses the
optical traversal convention; use an array linear-algebra inverse for the matrix
inverse itself. A supplied ABCD matrix already defines propagation direction, so
its propagator stores `inverse=None`. Otherwise `inverse=False` is forward.

## Numerical operators and contracts

| Operation | Input → output |
| --- | --- |
| `relay(array, grid_in, wavelength)` | Complex samples → complex samples |
| `relay(..., return_grid=True)` | Samples → `(samples, Grid)` |
| `relay.prepare(grid_in, wavelength)` | Sampling and parameters → callable array operation |
| `FFT(n)(array)` | Unitary centred Fourier transform over the last two axes |
| `MFT(kernels)(array)` | Separable matrix contractions over sampled axes |
| `ASM(transfer)(array)` | Forward FFT, transfer factors and inverse FFT |

Physical source, aperture, wavelength and Jones meanings belong to Wavefront and
systems. Numerical propagators only broadcast shapes. Independent parameter sets
need explicit singleton axes or vectorisation.

The [API map](../../../API/utils/propagation.md) shows all operator methods
and inheritance. The canonical FFT/MFT/ASM classes are available directly through
`dLux`; released function signatures belong to `dLux.compat` or `dLux.utils.compat`.
