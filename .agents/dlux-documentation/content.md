# dLux explanations and API content

Write for users who want to build and fit an optical model, including new postgraduate
students who do not know Zodiax. Explain what they can do before teaching internals.
Use the current [contracts](../dlux-development/contracts.md) and check source when an
example comes from an older API.

## Module Overviews

Start with a few realistic, concentrated examples. Each numbered section should briefly
say what it demonstrates. Use existing dLux utilities rather than constructing arbitrary
vectors or recreating numerical machinery for setup. For example, generate a small
Zernike basis, place it in Basis, then show that an Optic accepts either that definition
or the corresponding array. Show ordinary object representations without special
printing settings or type-name assertions.

Add complexity only when it illustrates a user task. A first example does not need Map,
Units, aliases, unbound operators or external State. Explain dynamic coordinate-dependent
parameterisations when they become useful. Public internal construction results such as
pasted-basis metadata need not be taught as user-managed APIs.

Show a small representative set of convenience constructors, then a table of the
remaining equivalent patterns. Diagrams should reflect real current classes and show
where call/apply/evaluate are defined. Separate large numerical-transform diagrams when
one full inheritance graph would obscure the overview.

Put advanced subclassing and API contracts after the core use cases. Prefer compact
input/output type-and-shape tables with short summaries over long prose-only lists.
These contracts should be useful foundations for tests.

## Docstrings

Audit four responsibilities separately: the class explains the concept and physical
contract; its constructor documents construction arguments; methods explain their
local inputs and outputs; the module overview teaches a normal workflow.

Use NumPy-style Parameters and Returns sections. Every public concrete constructor
must document its arguments, including accepted objects, units, shapes, defaults,
exclusive combinations and important construction-time effects. Inherited constructors
can retain their complete parent documentation. Do not replace a useful contract with
an "initialise" summary or a link that requires reading another implementation.

Substantial classes need a small complete runnable example under Examples, using a
Python code fence. Introduce the object, demonstrate its main use, and name the result.
For system entry points, show the distinct modelling stages. Include imports and setup;
avoid undefined placeholder arrays. Small shapes, obvious wrappers, abstract bases and
internal construction objects do not each need a redundant example.

Methods must state meaningful return types, units, axis order, normalisation and
limitations. Obvious properties or exact wrappers may use one precise sentence.
Preserve an inherited method's contract when an override does not change it. Avoid
mechanical boilerplate and do not claim checks or behaviour the code does not implement.

Use British scientific vocabulary, preserve identifier spelling, and do not leave a
blank line before closing quotes. Run examples against the checkout, including existing
examples affected by API changes. Reference `origin/reparametrisation`'s documentation
pass (`05d40548`, `57abb0c9`, `cace96f7`) for the established format, not its obsolete APIs.

Keep Imager's unweighted PSF separate from its photometry and detector image.
OpticalSystem deliberately retains default array returns and weighted propagation;
do not describe its return flags as identical to Imager's stage methods.
Sources store definitions; systems model them. Keep numerical introductions focused
on the numerical class without unnecessary Wavefront or Optic setup.

## Source ownership

Authored Module Overviews are edited directly. Generated API listings/UML pages are
regenerated through their script. Every page under `docs/tutorials/`, including
overview/index Markdown, is maintained in the tutorial repository and exported or
synchronised again. Root migration and changelog pages are authored release guides,
not notebook exports. Do not create a second introductory tutorial in a standalone
Markdown page to avoid the notebook workflow. Scope these workflows to the user's
task; a code-only review does not authorise rewriting the documentation tree.
