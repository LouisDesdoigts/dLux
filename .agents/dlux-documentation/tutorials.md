# Tutorial authoring and plotting

Tutorial notebooks live in the external
[`dLux_tutorials`](https://github.com/LouisDesdoigts/dLux_tutorials) repository and are
the source of truth for their exported pages. Review the existing published notebooks
before introducing a new narrative or plotting pattern; the phase-retrieval,
detector-calibration, optical-system, and coronagraph tutorials are the principal style
references.

## Repository and publishing workflow

Develop and execute tutorial notebooks in `dLux_tutorials`. The main dLux repository
contains the published documentation pages, not the editable tutorial source.

The expected workflow is:

1. Edit the notebook in `dLux_tutorials`.
2. Run every cell from a clean kernel and inspect its outputs.
3. Export it from the tutorial repository with:

   ```bash
   python export.py tutorials/<path>/<tutorial>.ipynb
   ```

4. Inspect the generated Markdown and assets under the notebook directory's
   `markdowns/` tree.
5. Move or synchronise those generated files into the corresponding dLux documentation
   location, then build the dLux documentation with `zensical build --strict`.

Never edit an exported tutorial Markdown page or its generated assets inside the dLux
documentation tree. Fix content in the source notebook; fix systematic export problems
in `dLux_tutorials/export.py`; then regenerate and replace the derived files. Do not
repair a generated tutorial page locally, because the next export will overwrite the
change and leave the notebook inconsistent with the published documentation.

## Notebook setup

Begin each notebook with a small reusable setup cell containing its imports and the
shared Matplotlib configuration. Keep this cell easy to copy between tutorials and
change it only when the tutorial genuinely needs different behaviour.

Start the setup cell with ``## COLLAPSE: Plotting setup`` so imports, plotting
defaults, colormaps, and reusable presentation helpers remain available without
dominating the exported tutorial.

The shared setup should establish the default intensity map, lower image origin,
figure resolution, typography, normalisations, and copies of any colormaps whose bad
values are customised. Do not scatter global plotting configuration throughout the
notebook.

## Scientific plots

Plots are part of the tutorial contract, not decoration. Every published plot must be
complete and physically interpretable:

- label axes and colour bars with the physical quantity and unit;
- derive image extents from the relevant grid or field rather than plotting pixel
  indices when physical coordinates are available;
- state any scaling, normalisation, masking, or nonlinear display stretch;
- use shared limits for panels intended for direct comparison;
- use a sequential map such as `inferno` for non-negative intensity;
- use `RdBu` with a centred normalisation for OPD and OPD residuals;
- use `seismic` with a centred normalisation for detector-pixel residuals, z-scores,
  and signed image residuals;
- display phase on a cyclic map with limits spanning one wrapped interval, and format
  the colour bar in wrapped phase units such as multiples of $\pi$;
- mask values outside the physical pupil where that prevents meaningless OPD or phase
  pixels from dominating the display.

Choose figure sizes and panel layouts deliberately. Titles should identify the
quantity or comparison rather than restate the surrounding prose. Inspect every
rendered figure for clipping, unreadable labels, misleading ranges, and colour bars
that do not match the plotted data.

Use a baseline figure size of 5 inches in both width and height per image panel. For
example, a two-panel, one-row image comparison should normally begin at
``figsize=(10, 5)``. Keep colour bars attached to the axes they
describe; where Matplotlib's ordinary layout changes the image size or leaves an
uneven gap, use a shared helper based on
``mpl_toolkits.axes_grid1.make_axes_locatable`` to append a dedicated colour-bar
axis rather than tuning arbitrary figure margins.

## Export behaviour

Keep the scientific result visible while hiding incidental plotting machinery in the
exported page. Put plotting-only cells behind the exporter-supported collapse marker:

```python
## COLLAPSE: Plotting code
```

The export script converts these cells into collapsed code blocks while retaining
their figures. Do not reproduce this behaviour manually in generated Markdown.

Plotting code should still be readable and executable in the source notebook. Do not
hide model construction, inference, or another operation the reader needs to learn;
collapse only supporting presentation code or a clearly labelled secondary helper.

## Validation

Run the notebook from a clean kernel, export it with the repository script, and inspect
both the notebook and rendered documentation. Confirm that collapsed cells retain
their outputs, physical units survive export, phase colour bars wrap correctly, and
all figures remain legible at documentation width.
