# dLux agent material

This directory contains the canonical, vendor-neutral skills for working with dLux.
The repository root `AGENTS.md` defines the universal working agreement. Each
`<name>/SKILL.md` defines one task-specific workflow, with substantial
supporting Markdown placed beside it when that material would distract from the core
workflow. The flat layout is intentional; integrations should make sibling Markdown
available when the skill links to it.

Agents that recognise these conventions can load them directly. Other integrations
should point their instruction system at the root `AGENTS.md`, expose the relevant
`SKILL.md` files on demand, and make linked Markdown available when requested. If an
agent requires another filename or directory layout, adapt or rearrange these files
locally rather than adding a duplicated vendor-specific copy to the repository.

The canonical skills are:

- `dlux-development`: package implementation, architecture, compatibility, and releases;
- `dlux-testing`: behavioural, numerical, and differentiable validation;
- `dlux-documentation`: docstrings, tutorials, generation scripts, API pages, and UMLs;
- `dlux-usage`: building models, simulations, optimisation, and inference with dLux.

Shared API facts live in [development/contracts.md](dlux-development/contracts.md).
Read only the sections relevant to the task. The implementation and the user's current
accepted decisions take precedence if this map drifts; update the map, not the code,
when the code already implements the agreed contract.

Documentation ownership is recorded in
[documentation/generation.md](dlux-documentation/generation.md). In particular,
`docs/tutorials/` is exported; `docs/overviews/` is authored. Release progress, test
counts and unresolved design proposals belong in local review records, not permanent
skill instructions. Updating these guides does not authorise committing them.
