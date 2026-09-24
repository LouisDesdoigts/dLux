# Numerical tools

## 1. Interpolate array values

Interp keeps source samples and their coordinates together, and samples them on a
new grid or at query points. It changes values only through interpolation; it does
not normalise flux or compensate for coordinate stretching.

```python
import dLux as dl
import jax.numpy as np

grid = dl.Grid(8, diam=1.0)
image = dl.Gaussian(std=0.2, grid=grid).resolve()
operation = dl.Interp(image, grid, dl.Grid(12, diam=1.0), fill=0)
print(operation)
print(operation.resolve().shape)
print(operation.sample((np.array([0.0, 0.1]), np.array([0.0, 0.1]))))
```

```text
Interp(
  array=f32[8,8],
  grid_in=Grid(n=(8, 8), d=f32[2]),
  grid_out=Grid(n=(12, 12), d=f32[2]),
  method='cubic2',
  fill=i32[]
)
(12, 12)
[0.9967546 0.7782133]
```

Tuple components describe arbitrary 2D/3D points. Stacked arrays describe grids,
with shape `(..., ndim, *query_shape)` and exactly `ndim` query axes. One-dimensional
queries can be scalars or vectors. Source arrays end in the sampled axes; leading
axes broadcast as ordinary batches.

`method="cubic2"` uses a C2 cubic spline. `"cubic"` uses locally estimated Hermite
slopes; `"linear"` connects samples with straight lines. Cubic interpolation can
overshoot. `fill=None` extrapolates outside the source; `fill=0` returns zero there.
Complex values use real/imaginary interpolation, or amplitude/phase with `complex=False`.

`grid_in` accepts Grid, Axis, or physical-order tuples of increasing coordinate
vectors. Omitted sampling means centred unit pixels. `grid_out` defines evaluation
sampling; a transform can instead map the source sampling into a new coordinate frame.

## 2. Reuse source interpolation

Preparation stores an Interpax interpolator for repeated queries.

```python
prepared = operation.prepare()
print(prepared.sample((0.0, 0.0)))
```

```text
0.9967546
```

Re-prepare after changing source values, input sampling, method or complex mode.
The snapshot does not refresh when its defining leaves are updated. Queries and
output transforms may change without rebuilding source interpolation.
Without an output grid, query coordinates or transform, `.resolve()` leaves Interp
deferred; `.apply()` explicitly samples the input grid.

## 3. Define apertures and continuous profiles

Shapes evaluate geometry on a grid; Profile turns a sampled density into pixel masses.

```python
pupil = dl.Circle(diameter=0.8, edge=dl.Soft(pixels=1))
profile = dl.Profile(dl.Gaussian(std=0.2))
print(pupil)
print(profile.sample(grid).shape)
```

```text
Circle(edge=Soft(pixels=f32[]), diameter=f32[])
(8, 8)
```

| Definition | Meaning |
| --- | --- |
| `Circle(diameter)`, `Rectangle(width, height)` | Hard transmission unless an edge rule is supplied |
| `RegularPolygon(nsides, diameter)` | Regular polygonal support |
| `Spider(width, angles)` | Support vanes; angles in degrees |
| `Complement(shape)` | Complementary transmission |
| `Combine(shapes, operation="product")` | Multiply independent transmissions |
| `Gaussian(std, mean)` | Continuous unit peak, independent of the query grid |
| `Profile(profile, grid=None)` | Sample density and integrate subpixels; unit sum by default |
| `Jitter(width, unit=None)` | Normalised image-blur kernel; omitted unit means native pixels |

Shapes accept `grid` and `transform`. These change where the definition is evaluated,
without interpolating a previously sampled mask. Gaussian normalisation belongs to
the consumer. Jitter's zero-width case is a discrete centred impulse.

## 4. Use polynomials and reductions

Polynomial selects monomials and stores fit coefficients. Norm measures arrays;
`normalise` scales them to a selected norm without introducing a model class.

```python
polynomial = dl.Polynomial(degree=2, coeffs=[1.0, 2.0, 0.5])
print(polynomial)
print(polynomial.resolve(coordinates=np.array([-1.0, 0.0, 1.0])))
```

```text
Polynomial(coeffs=f32[3], powers=i32[1,3])
[-0.5  1.   3.5]
```

[Defaults, shapes and methods](../../../API/utils/numerical.md) are
listed in the generated API map. [Bases](../bases/overview.md) covers coefficient
expansions, and [the utilities overview](../overview.md) covers composition.
