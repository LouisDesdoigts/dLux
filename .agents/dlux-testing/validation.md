# Numerical and transformation validation

Select checks from the current public contract:

| Area | Useful evidence |
| --- | --- |
| Sampling/Field | physical axes, coordinates, centre, parity, units and array views |
| Interp | known functions, prepared/unprepared agreement, batch-vs-loop results, fill/extrapolation and complex sampling; values only, no conservation correction |
| Propagation | complete complex fields, phase/sign, realised grids, explicit preparation and independent MFT/FFT references |
| Numerical definitions/bases | array equivalence, missing runtime context, immutable updates, meaningful coefficient gradients |
| Jones/Wavefront | scalar vs compact/materialised Jones equivalence, chromatic/population axes, power and Stokes results |
| Photometry | density vs integrated weights, stored/explicit BandPass agreement, support edges, products of curves and retained losses |
| Sources/Imager | arcsecond PointSource defaults, binary/array equivalence, Scene offsets, unweighted Imager PSFs, named outputs and once-only detector order |
| Sparse/prebuilt | compact vs dense outputs within sampling assumptions, aperture phases and native vs oversampled outputs |
| OpticalSystem | default array/return-flag contracts, common output sampling and agreement with Imager for matching optical inputs |
| Compatibility | retained released result, actionable migration warnings and explicit unsupported routes |

Use the smallest cases that distinguish physical axes from ordinary batches. In
particular, include equal-sized independent dimensions where accidental pairing could
otherwise pass shape checks. Compare vectorised results to separately evaluated cases.

Tolerances follow dtype and algorithm. Explain a changed tolerance with measured scale
and the numerical operation, not merely that the prior assertion failed. Use x64 only
when the tested contract requires it; do not silently make the whole suite depend on
precision settings absent in normal user code.

Check eager/JIT values and tree structure. For derivatives, compare a useful analytic
result or finite differences where possible. Higher derivatives are valuable for known
normalisation or tracing issues, not mandatory for every constructor or convenience
method. Prepared snapshots intentionally freeze their inputs until re-prepared.

Keep complete-suite runs auditable: record command, interpreter, dependency source,
counts, warnings and whether slow tests or coverage were included. Do not claim a
remote Python matrix from one local interpreter.

Check known convention boundaries with asymmetric inputs: rotation direction,
rectangular grids, Fourier coefficient layout, jitter width and complex propagation
phase. Symmetric intensity examples can conceal those errors. Absolute photon density
and explicit total flux require separate spectral expectations; Filter.integrate returns
bandwidth, not photon counts. Spectral support tests should include an edge inside an
optical bin and a nonzero endpoint so extrapolation errors are observable.
