---
name: dlux-documentation
description: Write or revise dLux user explanations, docstrings, overviews, tutorials and generated API pages. Use when documentation is in scope, not as an automatic consequence of a code change.
---

# dLux documentation

Read [AGENTS.md](../../AGENTS.md). Identify the source of truth: docstrings own API
contracts; `docs/overviews/` contains authored Module Overviews; exported tutorial
pages and indexes under `docs/tutorials/` come from the tutorial repository; API listing pages and generated UMLs come from exports and
generation scripts. Do not overwrite authored overviews as generated content or edit tutorial exports
directly. Root-level docs pages may be authored, include wrappers or stale placeholders;
check their provenance before treating them as tutorials.

Read [content.md](content.md) for audience and examples. Read
[generation.md](generation.md) when generated API pages, navigation or diagrams are
in scope, and [tutorials.md](tutorials.md) for notebook authoring/exporting.

Lead with a small useful optical example. Demonstrate arrays and parameterised values
working through the same API before explaining resolution internals. Print objects
where their structure helps understanding. Units, aliases, unbound operators and State
are advanced tools, not mandatory introductory setup.

Do not turn an overview into a catalogue of every class. Cover representative tasks,
then advanced semantics and inheritance. Use compact tables for repeated constructor
patterns and diagrams showing actual inheritance and the methods that define each
contract. Developer input/output contracts can follow the user tutorial and inform
behavioural tests.

Use the navigation title **Module Overviews**. Follow the ownership table in
[generation.md](generation.md); filenames alone do not establish edit ownership.

For an authorised documentation change, run affected examples, regenerate only derived
artefacts, inspect the result and perform the relevant build. Do not add a documentation
build or modify published pages during a task that excludes documentation.
