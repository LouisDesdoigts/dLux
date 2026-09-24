# Building models with dLux

Choose the entry point for the task; neither requires replacing the other.

| Entry point | Inputs and outputs |
| --- | --- |
| OpticalSystem(layers, grid) | Named optical sequence; propagate/model return arrays by default, with return_psf/return_wf options |
| Imager(pupil_grid, image_grid, pupil, ...) | Stored source, BandPass, filter and Detector; stage methods return fields, weights and images |
| DetectorSystem(layers) | Standalone deterministic Intensity → Intensity sequence |
| Detector(...) | Field sampling, response layers, pixel integration, noise and gain |

## A small Imager

```python
import dLux as dl

imager = dl.Imager(
    pupil_grid=dl.Grid(16, diam=1.0, unit="m"),
    image_grid=dl.Grid(8, d=0.1, unit="arcsec"),
    pupil=dl.Optic(transmission=dl.Circle(1.0), normalise=True),
    source=dl.PointSource(flux=1000),
    bandpass=dl.BandPass(range=(550, 650), n=3, unit="nm"),
)
psf = imager.psf()
weights = imager.photometry()
image = imager.model()
```

| Imager method | Meaning |
| --- | --- |
| wavefront | Fine, unweighted complex point responses |
| psf | Unweighted point intensities with source/wavelength axes; optical losses remain |
| photometry | Integrated source × filter × QE weights without optical propagation |
| intensity | Fine expected source signal, including profiles/jitter and any field resampling |
| image/model/call | Expected detector Image after response, pixel integration and gain |
| simulate | Photon/read-noise realisation with gain applied in readout order |

A pupil Optic can contain arrays or definitions for transmission, OPD, index/thickness
and Jones response. In an Imager its grid_out must be None: Imager owns final relay
sampling and reads focal_length/defocus/abcd/inverse from the pupil. Standalone Optic
can propagate when grid_out is supplied. A length-valued image grid needs appropriate
physical relay parameters; ordinary angular imaging does not need a focal length.

## Spectra and sources

Source defaults to a flat unit-flux spectrum and an on-axis point. Flux lives at
spectrum.flux and means incident photons over the calculation BandPass before filter
and QE losses. A scalar flux applies to each source position. Spectrum(flux=None)
preserves absolute density; only photon flux units are supported.

PointSource positions default to arcseconds. Raw Source positions and the low-level
Point expression use radians; Positions/Binary/Companions default to arcseconds.
Binary/Companions locate the primary and use degree PA from +y towards +x. Contrast
is log10(companion/primary) by default, or a direct ratio with log=False. Use explicit
compat constructors for released midpoint/mean-flux/radian conventions.

BandPass range gives outer bin edges and n gives the optical wavelength count. Curves
may retain finer native sampling. Imager photometry/intensity/image requires BandPass;
explicit wavelength arrays are appropriate for wavefront/psf and OpticalSystem
propagation. Density samples and integrated Weights are not interchangeable.

Use a list or Scene to add incoherent components, including different spectra/profiles.
Named dictionaries in modelling arguments instead request independent outputs: matching
keys pair, scalars are shared, and missing keys use stored defaults. None keeps stored
values; use an immutable update to clear an optional setting.

## Sampling and fitting

image_grid describes native pixels; oversample refines propagation and integrates back
when forming Image. Pixel-unit jitter refers to native pixels before further detector
binning. Interp only interpolates values; it applies no flux/area compensation.

A resolved profile broadens the fine point-response intensity. For complex detector
resampling, scalar/compact-Jones fields use an effective field with the original phase;
explicit Jones fields are not reconstructed this way. Pupil-plane spatial Jones is
supported when no such reconstruction is required. Coronagraph rejects resolved-source
convolution. These are the current limits, not a mandate to add another modelling mode.

Use explicit stored-leaf paths for fitting. A Basis can replace an OPD array; fit coeffs
and keep its output unit separate from coordinate units. Put preparation inside the
objective when fitting a prepared operator's inputs, or re-prepare explicitly after
updates. Keep Linked owners together before splitting model branches.

See the [shared contracts](../dlux-development/contracts.md) for axis details and the
[Module Overviews](../../docs/overviews/overview.md) for user-facing examples. Tutorial
work follows the notebook/export workflow in the documentation skill.
