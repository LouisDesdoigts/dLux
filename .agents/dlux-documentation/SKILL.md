---
name: dlux-documentation
description: Create, restructure, generate, or review dLux documentation, including public docstrings, examples, tutorials, navigation, API pages, landing pages, generation scripts, and UML diagrams. Do not use for release changelogs or compatibility planning, which belong to dlux-development.
---

# dLux documentation

Read the repository `AGENTS.md`. Identify the source of truth before editing:
docstrings own API content, notebooks own tutorials, exports own public API listings,
and generation scripts own generated Markdown and UML diagrams.

Read `content.md` for documentation structure and API depth. Read `generation.md`
when API pages, UMLs, notebook exports, navigation, or build scripts are affected.

Write for user tasks before module names. Substantial APIs need physical contracts and
small complete examples; simple wrappers need only precise documentation. Tutorials
must be executable, state prerequisites and learning outcomes, and distinguish
pedagogical approximations from research-grade validation.

After changes, regenerate the affected artefacts, inspect the rendered result, and run
`zensical build --strict`. Never repair a generated page by editing it directly.
