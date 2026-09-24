# Basis expansions

## 1. Replace an OPD array with coefficients

Basis combines sampled modes into one array. An Optic can use either that array
or the Basis itself, so parameterising the OPD does not change how the optic is used.

```python
import dLux as dl
import jax.numpy as np

grid = dl.Grid(8, diam=1.0, unit="m")
modes = dl.Zernike([2, 3, 4], diameter=1.0).resolve(grid=grid)
opd = dl.Basis(modes, coeffs=[10.0, 0.0, 20.0], unit="nm")
print(opd)

sampled = dl.Optic(opd=opd.resolve())
parameterised = dl.Optic(opd=opd)
wavefront = dl.Wavefront(600e-9, grid)
print(np.allclose(sampled(wavefront).phasor, parameterised(wavefront).phasor))
```

```text
Basis(unit='nm', coeffs=f32[3], basis=f32[3,8,8])
True
```

Coefficients are editable arrays, including when omitted: `Basis(modes)` allocates
zeros. `unit="nm"` converts the contracted OPD to metres after the dot product.
Modes remain dimensionless and coefficients remain in the declared unit.

## 2. Generate modes when coordinates are available

A Zernike expression can supply the basis instead of a precomputed array. Its mode
count lets Basis initialise coefficients before evaluating the grid.

```python
generated = dl.Basis(dl.Zernike([2, 3, 4], diameter=1.0), unit="nm")
print(generated.coeffs)
print(generated.resolve(grid=grid).shape)
```

```text
[0. 0. 0.]
(8, 8)
```

This retains geometry parameters in the model. A sampled array instead retains the
modes generated at that particular geometry. `empty=True` explicitly retains absent
coefficients; it is intended for constructing an incomplete definition.

## 3. Solve for coefficients

`solve` returns a minimum-norm least-squares coefficient array in the Basis's unit.
It does not mutate the original object or reverse a transform stored in `coeffs`.

```python
coefficients = opd.solve(opd.resolve())
fitted = opd.set(coeffs=coefficients)
print(fitted)
```

```text
Basis(unit='nm', coeffs=f32[3], basis=f32[3,8,8])
```

| Expansion | Purpose |
| --- | --- |
| `Basis(basis, coeffs=None, axes=None)` | Contract explicit or generated modes |
| `FourierBasis(npix, n_modes, coeffs=None)` | Compact real Fourier expansion |
| `CLIMBBasis(...)` | Coefficient representation of a binary pattern |
| `PastedBasis(...)` | Compact stamp modes, usually constructed by aperture builders |

## Contracts

| Boundary | Input → output |
| --- | --- |
| Basis mode array | `(*coefficient_shape, *output_shape)` |
| Coefficients | `(*batch, *coefficient_shape)` when axes are explicit or inferred from a generator |
| `Basis.resolve(...)` | Contracted array `(*batch, *output_shape)` in SI units |
| `Basis.solve(array, ...)` | Coefficients in the declared output unit |
| `ParametricBasis.coeffs_ndim` | Intrinsic coefficient-axis count, excluding independent batches |

With a plain array basis and `axes=None`, all supplied coefficient axes are
contracted. Set `axes=1` for batched coefficient vectors. Paired batches of distinct
mode arrays and coefficients require explicit vectorisation.

[Inheritance and reference](../../../API/utils/bases.md).
