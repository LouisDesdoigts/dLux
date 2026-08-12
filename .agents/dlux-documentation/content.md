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
A public class or method must document its complete local contract. Links may provide
broader context, but users must not need to leave the API they are reading to learn an
argument, return value, unit, shape, default, limitation, or basic usage. Do not use
placeholder descriptions such as "as defined by", "arguments follow", or "see the
base implementation" for public inputs.

Examples should be minimal walkthroughs of the documented class and its principal
behaviours. Divide multi-stage examples into short commented blocks, use realistic
physical units, and identify important output types. Avoid unrelated package
machinery, but include every setup step required for the example to execute on its
own. Small components need one focused use; major system classes should show their
distinct primary entry points.

Document every public concrete-class constructor. Constructor docstrings must cover
all arguments, accepted types, shapes, units, defaults, exclusive combinations,
broadcasting, validation, and important construction-time effects. Simple
constructors may be concise, but still need a useful `Parameters` section; a summary
that merely restates "initialise" is insufficient. Keep the broader conceptual and
physical contract on the class docstring.

Apply the same ownership test to every class method: a method should either be a
deliberate public API with a complete usable contract, or an implementation detail
with a private name. Do not privatise useful behaviour merely to avoid documenting
it. Public method documentation must let a user call the method without reading its
implementation, including accepted objects, units, shapes and axis order, defaults,
return types and containers, immutable update behaviour, and important limitations.
One-line summaries are sufficient only for genuinely obvious properties or wrappers.
When an override preserves an inherited contract exactly, prefer inheriting the base
documentation over replacing it with a less informative summary.

Audit convenience constructors and aliases against the canonical API. Remove
unreleased redundant routes; retain released routes only through the compatibility
policy, with an actionable migration warning where they are deprecated.

The documentation landing page should explain these routes, expose the stable version,
link the package and UML maps, point directly to the utilities guide, and end with
clear next steps. Tutorial metadata should drive the finder, navigation, prerequisites,
CI tier, and gallery rather than duplicating those facts manually.
