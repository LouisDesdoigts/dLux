# dLux release workflow

Build a release from the complete public change since the previous published tag.

1. Audit exports, constructors, behaviour, compatibility aliases, dependencies,
   supported Python versions, documentation, and generated artefacts.
2. Group the changelog by user-visible capabilities, breaking changes, deprecations,
   fixes, documentation, and migration impact. Do not reproduce commit history.
3. Give every deprecation a replacement, before/after example, and removal version.
4. Update the migration guide and ensure migration errors point to stable sections.
5. Run the supported Python matrix, documentation build, examples, and release checks.
6. Verify versioned documentation and package metadata before publishing.

Treat the changelog as an orientation document for existing users. Major internal
work belongs only when it changes capability, reliability, performance, extension
patterns, or maintenance expectations.

Before release, validate installed declared dependency versions independently of local
sibling checkouts. Verify that the distributed source includes every imported module;
local untracked experiments must not accidentally become required release dependencies.
Check coverage and the supported Python matrix separately from passing local assertions.
Agent skills and review notes remain local unless explicitly included by the user.

Compare against the released baseline, not every intermediate development interface.
Keep the original upstream tests unchanged when measuring compatibility. Separate
namespace availability, structural compatibility and numerical agreement; an adapter
returning current objects does not promise old `isinstance` checks or parameter paths.
Use the repository comparison scripts where applicable, inspecting their inputs first.

The current selected release excludes `utils/raytrace/` (including GaussBeam),
`utils/source_estimation.py` and `calib.py`, plus dependent experiments/tests. Keep
those files locally; do not delete or stage them incidentally. SparseOptic redesign
and the Zernike origin geometry-gradient issue remain deferred review topics, not
permission to reopen them during packaging. Reconfirm scope if release selection changes.

Do not publish validation counts or a release date from an older run as current evidence.
Record outstanding gates in the local release plan. Tutorial completion means notebook
execution and export, not hand-edited Markdown that merely passes the docs build.
