# dLux basics

These tutorials explain the objects and methods that make up dLux. They are designed
to be read independently after the getting-started tutorial, while together forming a
complete guide to constructing, evaluating, differentiating, and fitting dLux models.

Two focused explainers support that learning path:

- [Migrating to dLux 0.16](../../migration.md) translates common 0.14 and 0.15
  workflows into the new object model.
- [Parameters and immutable updates](../../parameters.md) explains raised paths,
  selecting fitted leaves, immutable updates, and multi-object models.

- `fields_grids_layers.ipynb`: grids, fields, wavefronts, and layer interactions.
- `systems_sources_detectors.ipynb`: optical and detector systems, sources, spectra,
  noise, and a compact inference problem.
- `apertures.ipynb`: custom, static, dynamic, dense, sparse, and segmented apertures.
- `custom_layers_parametrics.ipynb`: creating optical layers and parametric types,
  context resolution, vectorisation, and raised parameter paths.
- `inference.ipynb`: Zodiax, Optax, Optimistix, NumPyro, BlackJAX, derivatives,
  likelihoods, uncertainty estimation, and numerical validation.

The published overview should use focused UML diagrams to introduce each part of the
object model and link from the relevant classes to the tutorials above.
