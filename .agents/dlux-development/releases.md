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
