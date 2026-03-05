"""Internal helpers for package and subpackage symbol re-export."""

from __future__ import annotations


def reexport(modules: tuple[object, ...], namespace: dict[str, object]) -> list[str]:
    exported: list[str] = []
    for module in modules:
        names = getattr(module, "__all__", ())
        for name in names:
            namespace[name] = getattr(module, name)
        exported.extend(names)
    return exported
