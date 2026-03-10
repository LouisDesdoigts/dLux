from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
API_ROOT = ROOT / "API"

INFO_RE = re.compile(r'^\?\?\?\+ info "([^"]+)"\s*$')
DIRECTIVE_RE = re.compile(r"^\s*:::\s*([A-Za-z_][A-Za-z0-9_\.]+)\s*$")


def flatten(items: Iterable):
    for item in items:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)
        else:
            yield item


def infer_module_from_content(text: str) -> str | None:
    for line in text.splitlines():
        match = DIRECTIVE_RE.match(line)
        if not match:
            continue
        dotted = match.group(1)
        parts = dotted.split(".")
        if len(parts) >= 2:
            # Existing pages use directives like dLux.foo.Bar;
            # the module is everything before the object name.
            return ".".join(parts[:-1])
    return None


def infer_module_from_path(md_path: Path) -> str | None:
    rel = md_path.relative_to(API_ROOT)
    if len(rel.parts) != 2:
        return None

    section = rel.parts[0]
    stem = md_path.stem

    if stem == "overview":
        return None
    if section == "core":
        return f"dLux.{stem}"
    if section == "layers":
        return f"dLux.layers.{stem}"
    if section == "utils":
        return f"dLux.utils.{stem}"
    return None


def get_preamble_lines(text: str) -> list[str]:
    lines = text.splitlines()
    stop_idx = len(lines)
    for i, line in enumerate(lines):
        if INFO_RE.match(line) or DIRECTIVE_RE.match(line):
            stop_idx = i
            break

    preamble = lines[:stop_idx]
    while preamble and preamble[-1].strip() == "":
        preamble.pop()
    return preamble


def exported_api_items(module_name: str) -> list[str]:
    module = importlib.import_module(module_name)
    exported = list(flatten(getattr(module, "__all__", [])))

    items: list[str] = []
    seen: set[str] = set()
    for name in exported:
        if not isinstance(name, str) or name in seen:
            continue
        obj = getattr(module, name, None)
        if inspect.isclass(obj) or inspect.isfunction(obj):
            items.append(name)
            seen.add(name)
    return items


def render_page(preamble: list[str], module_name: str, names: list[str]) -> str:
    out = list(preamble)

    if out:
        out.append("")

    for i, name in enumerate(names):
        out.append(f'???+ info "{name}"')
        out.append(f"    ::: {module_name}.{name}")
        if i != len(names) - 1:
            out.append("")

    out.append("")
    return "\n".join(out)


def main() -> None:
    if not API_ROOT.exists():
        raise FileNotFoundError(f"API directory not found: {API_ROOT}")

    updated = 0
    skipped = 0
    for md_path in sorted(API_ROOT.rglob("*.md")):
        text = md_path.read_text(encoding="utf-8")

        module_name = infer_module_from_content(text)
        if module_name is None:
            module_name = infer_module_from_path(md_path)

        if module_name is None:
            print(f"SKIP {md_path.relative_to(ROOT)} (no module mapping)")
            skipped += 1
            continue

        try:
            names = exported_api_items(module_name)
        except Exception as exc:
            print(f"SKIP {md_path.relative_to(ROOT)} (import failed: {exc})")
            skipped += 1
            continue

        preamble = get_preamble_lines(text)
        new_text = render_page(preamble, module_name, names)

        if new_text != text:
            md_path.write_text(new_text, encoding="utf-8")
            print(
                f"UPDATE {md_path.relative_to(ROOT)} "
                f"({module_name}: {len(names)} exported classes/functions)"
            )
            updated += 1

    print(f"Done. Updated {updated} files, skipped {skipped} files.")


if __name__ == "__main__":
    main()
