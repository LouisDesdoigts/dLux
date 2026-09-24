# Photometry

Spectrum describes incident photon density; Filter describes transmission or QE.
BandPass defines the integration bins. Their numerical calculations are independent
of optical propagation.

## 1. Calculate photon counts for optical wavelengths

A finely sampled filter and a continuous source spectrum can be integrated into a
small number of optical bins.

```python
import dLux as dl

calibration = dl.BandPass(range=(500, 800), n=5, unit="nm")
optical = dl.BandPass(range=(500, 800), n=3, unit="nm")
filter = dl.Filter([0.0, 0.6, 0.9, 0.6, 0.0], calibration)
qe = dl.Filter(0.8, method="linear")
spectrum = dl.Blackbody(5000, flux=1000)
photometry = dl.Photometry(spectrum, filter=filter, qe=qe)
weights = photometry(optical)
print(optical)
print(weights)
print(weights.value)
```

```text
BandPass(unit='nm', range=(500.0, 800.0), n=3)
Weights(value=f32[3], bandpass=BandPass(unit='nm', range=(500.0, 800.0), n=3))
[ 53.667892 227.8261    58.73687 ]
```

`range` gives the outer edges of equal-width bins, not the first and last sample.
`optical.wavelengths` gives centres in metres; `optical.edges` gives all bin edges.
The range and count are static Python metadata. Alternative constructors accept
`min`/`max` or `wl`/`delta_wl` in the declared unit.

Spectrum's `flux=1000` means 1000 incident photons integrated over this calculation's
BandPass before losses. `flux=None` instead retains an absolute photon density per
metre. Filter and QE are multiplied with the density **before** integration; their
separate integrals cannot be multiplied to recover that product integral.

## 2. Sample curves or integrate bins

Use wavelength arrays for point values and BandPass for integrated results.
`sample` always returns point values; `integrate` returns the total over the interval.

```python
query = optical.wavelengths
response = photometry.response(query)
counts = weights.integrate()
print(response)
print(counts)

flat = dl.Spectrum(flux=1000)
sloped = dl.PolySpectrum(calibration, coeffs=[0.2], flux=1000)
print(sloped)
```

```text
[0.16444415 0.71999997 0.16444422]
340.23087
PolySpectrum(
  bandpass=BandPass(unit='nm', range=(500.0, 800.0), n=5),
  flux=f32[],
  value=Polynomial(coeffs=f32[1], powers=i32[1,1]),
  method='cubic2'
)
```

PolySpectrum is `1 + c[0] u + c[1] u² + ...`, with its stored interval mapped to
`[-1, 1]`. Its coordinate scaling stays fixed when queried on another band.
`Spectrum(value, bandpass=...)` also accepts arrays or a Basis whose output is sampled
on that native bandpass. Without native sampling, analytic expressions evaluate
at the requested wavelengths.

| Operation | Result |
| --- | --- |
| `spectrum.sample(wavelengths)` | Unnormalised density shape |
| `spectrum(bandpass)` | Incident Weights, with flux applied |
| `filter.sample(wavelengths)` | Dimensionless response |
| `filter(weights)` | Point-sampled losses applied to existing Weights |
| `filter.integrate(bandpass)` | Total response integral, in metres |
| `photometry(bandpass)` | Detected photon Weights |
| `photometry.integrate(bandpass)` | Total detected photons |
| `weights.integrate()` | Sum existing counts, without integrating again |

## 3. Understand support boundaries

Tabulated BandPass values live at bin centres. Their interpolant extends to that
BandPass's **outer edges**, then is zero outside. Raw wavelength vectors instead
have support from their first to last knot. Analytic spectra without native sampling
are evaluated throughout the requested band; PolySpectrum is limited to its reference band.

The integration route divides each output bin at native knots and support boundaries.
It integrates the original interpolants within each supported interval, so a narrow
filter edge is not smeared across an entire coarse output bin. `cubic2` is the
default for sampled curves; select `method="linear"` for piecewise-linear calibration.
The same route is used for explicit and stored BandPass inputs.

## Contracts

| Object | Stored meaning |
| --- | --- |
| BandPass | Static interval, count and wavelength unit |
| Spectrum / Blackbody / PolySpectrum | Spectral shape and optional total incident flux |
| Filter | Dimensionless multiplicative response and optional native sampling |
| Weights | Already-integrated counts with their BandPass |
| Photometry | Spectrum, filter and QE definitions |

Imager's BandPass sets optical sampling. High-resolution curves retain their own
sampling and are integrated into those bins; there is no `n_optical` setting.
Source stores the spectrum; filter and detector QE remain instrument properties.

[Class map and reference](../../../API/utils/photometry.md).
