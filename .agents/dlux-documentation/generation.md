# Documentation generation and UMLs

Treat generated documentation as derived state:

- API pages derive from public exports and source docstrings;
- UML and package diagrams derive from module ownership and inheritance;
- tutorial Markdown and figures derive from executed notebooks;
- tutorial indexes and navigation should derive from validated metadata where possible.

Inspect and update the generator when output structure is wrong. Regenerate all
affected pages after moving, renaming, exporting, or changing inheritance of public
objects. Check diagram nodes, links, hover content, section maps, and landing-page
routes as well as whether the build succeeds.

The API Markdown and UML diagrams share one source generator. Run it before every
documentation build that can affect public objects, module ownership, inheritance,
API navigation, or diagrams:

```bash
python docs/scripts/generate_api_mds.py
zensical build --strict
```

These are distinct validation stages. API generation checks derived structure;
Zensical checks the assembled site; rendered inspection checks layout, links, and
visual clarity; notebook execution checks tutorial behaviour. Do not report one as
evidence that the others passed. Source docstring changes may leave generated API
Markdown unchanged because the documentation engine resolves them during the build.

Before running documentation tools, verify that one environment contains both the
project dependencies and the documentation toolchain. Do not install missing tools
into the system or base Python as an automatic fallback. Report an environment
mismatch when the intended project environment cannot perform the build.

Do not treat a successful Zensical build against stale generated pages as sufficient.
Inspect the regenerated diagrams at documentation width and confirm that their
structure, labels, links, and hover summaries remain readable.

Execute notebooks into a build directory. Keep source outputs controlled, use stable
asset names, reject oversized payloads and broken links, and preserve separate smoke,
full, and extended execution tiers.
