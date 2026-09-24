"""Warnings and actionable errors for released interfaces."""

import warnings

REMOVAL_VERSION = "0.17.0"
MIGRATION_GUIDE = "https://louisdesdoigts.github.io/dLux/latest/migration/"


def warn_deprecated(old, new, example, stacklevel=3):
    """Identify the deprecated entry point and its replacement."""
    warnings.warn(
        f"The `{old}` interface is deprecated and will be removed in dLux "
        f"{REMOVAL_VERSION}. Use `{new}` instead: {example}. See {MIGRATION_GUIDE}",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
