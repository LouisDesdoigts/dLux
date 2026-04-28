from __future__ import annotations

import importlib
import inspect
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
API_ROOT = ROOT / "API"
PKG_ROOT = SRC_ROOT / "dLux"
MKDOCS_FILE = REPO_ROOT / "mkdocs.yml"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SECTIONS = ("core", "layers", "utils")


def flatten(items: Iterable):
    for item in items:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)
        else:
            yield item


def section_title(section: str) -> str:
    return {
        "core": "Core API",
        "layers": "Layers API",
        "utils": "Utils API",
    }[section]


def display_name(stem: str) -> str:
    return stem.replace("_", " ").strip().title()


def module_from_path(py_path: Path) -> tuple[str, str, str] | None:
    rel = py_path.relative_to(PKG_ROOT)
    if rel.name == "__init__.py" or rel.name.startswith("_"):
        return None

    if len(rel.parts) == 1:
        section = "core"
        stem = rel.stem
        module_name = f"dLux.{stem}"
        return section, stem, module_name

    if len(rel.parts) == 2 and rel.parts[0] in ("layers", "utils"):
        section = rel.parts[0]
        stem = rel.stem
        if stem.startswith("_"):
            return None
        module_name = f"dLux.{section}.{stem}"
        return section, stem, module_name

    # Keep generated docs flat at API/core|layers|utils and ignore deeper trees.
    return None


def collect_modules() -> dict[str, list[tuple[str, str]]]:
    modules: dict[str, list[tuple[str, str]]] = {section: [] for section in SECTIONS}
    for py_path in sorted(PKG_ROOT.rglob("*.py")):
        mapped = module_from_path(py_path)
        if mapped is None:
            continue
        section, stem, module_name = mapped
        modules[section].append((stem, module_name))
    return modules


def clean_api_tree() -> None:
    if API_ROOT.exists():
        shutil.rmtree(API_ROOT)

    for section in SECTIONS:
        (API_ROOT / section).mkdir(parents=True, exist_ok=True)


def exported_api_items(module_name: str) -> list[str]:
    module = importlib.import_module(module_name)
    exported = list(flatten(getattr(module, "__all__", [])))

    items: list[str] = []
    seen: set[str] = set()

    if exported:
        for name in exported:
            if not isinstance(name, str) or name in seen:
                continue
            obj = getattr(module, name, None)
            if inspect.isclass(obj) or inspect.isfunction(obj):
                items.append(name)
                seen.add(name)
        return items

    # Fallback for modules that don't define __all__: collect public defs.
    for name, obj in inspect.getmembers(module):
        if name.startswith("_") or name in seen:
            continue
        if (inspect.isclass(obj) or inspect.isfunction(obj)) and getattr(
            obj, "__module__", None
        ) == module_name:
            items.append(name)
            seen.add(name)

    return items


def render_page(title: str, module_name: str, names: list[str]) -> str:
    out = [f"# {title}", ""]

    if not names:
        out.append("No public classes or functions are exported by this module.")
        out.append("")
        return "\n".join(out)

    for i, name in enumerate(names):
        out.append(f'???+ info "{name}"')
        out.append(f"    ::: {module_name}.{name}")
        if i != len(names) - 1:
            out.append("")

    out.append("")
    return "\n".join(out)


def render_api_nav_block(modules: dict[str, list[str]]) -> list[str]:
    lines = [
        "  - API:\n",
    ]

    for section in SECTIONS:
        lines.append(f"    - {section_title(section)}:\n")
        lines.append(f"      - Overview: API/{section}/overview.md\n")
        for stem in modules[section]:
            lines.append(f"      - {display_name(stem)}: API/{section}/{stem}.md\n")
        lines.append("\n")

    if lines and lines[-1] == "\n":
        lines.pop()
    return lines


def update_mkdocs_api_nav(modules: dict[str, list[str]]) -> bool:
    if not MKDOCS_FILE.exists():
        raise FileNotFoundError(f"mkdocs.yml not found: {MKDOCS_FILE}")

    lines = MKDOCS_FILE.read_text(encoding="utf-8").splitlines(keepends=True)

    start = None
    start_indent = 0
    api_header_re = re.compile(r"^(\s*)-\s+API:\s*$")
    for i, line in enumerate(lines):
        match = api_header_re.match(line.rstrip("\n"))
        if match:
            start = i
            start_indent = len(match.group(1))
            break

    if start is None:
        raise ValueError("Could not find '- API:' section in mkdocs.yml")

    end = len(lines)
    for i in range(start + 1, len(lines)):
        raw = lines[i].rstrip("\n")
        stripped = raw.lstrip(" ")
        indent = len(raw) - len(stripped)
        if stripped.startswith("- ") and indent <= start_indent:
            end = i
            break

    new_block = render_api_nav_block(modules)
    new_lines = lines[:start] + new_block + lines[end:]

    old_text = "".join(lines)
    new_text = "".join(new_lines)
    if old_text == new_text:
        return False

    MKDOCS_FILE.write_text(new_text, encoding="utf-8")
    return True


def main() -> None:
    if not PKG_ROOT.exists():
        raise FileNotFoundError(f"Package directory not found: {PKG_ROOT}")

    modules = collect_modules()
    clean_api_tree()

    created = 0
    skipped = 0
    nav_modules: dict[str, list[str]] = {section: [] for section in SECTIONS}
    for section in SECTIONS:
        section_dir = API_ROOT / section
        overview = section_dir / "overview.md"
        overview.write_text(
            f"# {section_title(section)}\n",
            encoding="utf-8",
        )
        created += 1

        for stem, module_name in modules[section]:
            try:
                names = exported_api_items(module_name)
            except Exception as exc:
                print(f"SKIP {module_name} (import failed: {exc})")
                skipped += 1
                continue

            md_path = section_dir / f"{stem}.md"
            text = render_page(display_name(stem), module_name, names)
            md_path.write_text(text, encoding="utf-8")
            nav_modules[section].append(stem)
            print(
                f"CREATE {md_path.relative_to(ROOT)} "
                f"({module_name}: {len(names)} exported classes/functions)"
            )
            created += 1

    nav_changed = update_mkdocs_api_nav(nav_modules)
    print(f"mkdocs.yml API nav {'updated' if nav_changed else 'already up-to-date'}.")
    print(f"Done. Created {created} files, skipped {skipped} modules.")


if __name__ == "__main__":
    main()
