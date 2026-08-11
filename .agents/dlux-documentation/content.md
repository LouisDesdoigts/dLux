# Documentation content

Use five complementary forms:

- beginner tutorials provide an ordered path through the dLux mental model;
- how-to guides answer one concrete task;
- advanced case studies solve realistic scientific or instrumental problems;
- concept pages explain architecture and numerical conventions;
- API pages define stable public contracts.

Public documentation should state units, shapes, axes, normalisation, leading
dimensions, limitations, and relevant differentiability. Add concise executable
examples to core classes and behaviours where the interaction is not obvious.
Document every public concrete-class constructor, even when its concise constructor
docstring points back to a fuller class-level contract.

The documentation landing page should explain these routes, expose the stable version,
link the package and UML maps, point directly to the utilities guide, and end with
clear next steps. Tutorial metadata should drive the finder, navigation, prerequisites,
CI tier, and gallery rather than duplicating those facts manually.
