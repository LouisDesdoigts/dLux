# Documentation ownership, generation and UMLs

Check the source of a page before editing it. A file under docs is not necessarily an
authored document, and a root-level page is not necessarily a notebook export.

| Output/location | Source of truth | Editing route |
| --- | --- | --- |
| Class/method API content | Source docstrings | Edit docstrings; build the site |
| `docs/API/**` and generated UML | Public exports and `docs/scripts/generate_api_mds.py` | Fix generator/exports, then regenerate |
| `docs/overviews/**/overview.md` | Authored Module Overviews | Edit directly; execute examples and refresh shown results |
| `docs/tutorials/**` Markdown/assets | Tutorial notebooks and source indexes | Edit upstream source, export, then synchronise |
| `docs/migration.md` | Authored release migration guide | Check replacements against current code and the released reference |
| `CHANGELOG.md` | Authored release notes | Edit this file; `docs/changelog.md` includes it |
| Other root docs pages | Varies: authored guide, snippet include or link placeholder | Inspect content, history and navigation before changing |
| `site/` | Built site | Rebuild; do not edit or commit build output |

Do not manually repair tutorial exports, even for a small code fix or index link.
Do not automatically publish an untracked stub just because local navigation names it.
Avoid parallel guides teaching the same workflow; link to the maintained owner.

## API generation

When exports, source locations, class structure or generated diagrams change, run:

```bash
PYTHONPATH=src python docs/scripts/generate_api_mds.py
PYTHONPATH=src python -m zensical build --strict
```

Use a verified project interpreter containing the dependencies and documentation tools.
The current generator uses the Fields, Layers, Prebuilt, Systems, Utils and Compat
sections, follows nested utility packages, and skips excluded experiments. Its class
rendering merges constructor documentation with the class and exposes inherited
constructor contracts. Keep one constructor argument table rather than duplicating
it in the class summary. Module Overviews remain separate from generated API pages.

Source docstring changes alone can leave generated Markdown unchanged because the
renderer reads them at build time. Regenerate only when derived structure is affected.
Check diagram links/anchors, defining methods, layout and actual inheritance. A page
count or successful build does not establish that its examples work.

## Validation and publication

Execute affected authored examples against the intended package, then build and inspect
the rendered pages. Notebook execution, export, strict build, link validation and visual
inspection are distinct checks. Report only those performed. Do not install tools into
the system/base environment as an automatic fallback.

Before a documentation commit, check the selected tree, not only the development
checkout: navigation must not depend on untracked drafts, excluded experimental APIs
or unrelated local edits. Leave agent guidance and review records out unless requested.
