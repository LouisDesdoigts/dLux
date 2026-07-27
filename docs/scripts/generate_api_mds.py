from __future__ import annotations

import importlib
import inspect
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DOCS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DOCS_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
API_ROOT = DOCS_ROOT / "API"
PKG_ROOT = SRC_ROOT / "dLux"
MKDOCS_FILE = REPO_ROOT / "mkdocs.yml"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SECTIONS = ("core", "layers", "parametric", "utils")


@dataclass(frozen=True)
class ClassInfo:
    cls: type
    module: str
    name: str
    ident: str


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
        "parametric": "Parametric API",
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

    if len(rel.parts) == 2 and rel.parts[0] in ("layers", "parametric", "utils"):
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


def class_info(module_name: str, names: list[str]) -> list[ClassInfo]:
    module = importlib.import_module(module_name)
    classes, seen = [], set()
    for name in names:
        cls = getattr(module, name, None)
        if not inspect.isclass(cls) or cls in seen:
            continue
        ident = re.sub(r"\W", "_", f"{cls.__module__}_{cls.__name__}")
        classes.append(ClassInfo(cls, cls.__module__, cls.__name__, ident))
        seen.add(cls)
    return classes


def class_summary(cls: type) -> str:
    attrs = list(getattr(cls, "__annotations__", {}))
    properties = [
        name for name, value in cls.__dict__.items() if isinstance(value, property)
    ]
    methods = [
        name
        for name, value in cls.__dict__.items()
        if not name.startswith("_")
        and (callable(value) or isinstance(value, (classmethod, staticmethod)))
    ]
    parts = []
    if attrs:
        parts.append(f"Attributes: {', '.join(attrs)}")
    if properties:
        parts.append(f"Properties: {', '.join(properties)}")
    if methods:
        parts.append(f"Methods: {', '.join(f'{name}()' for name in methods)}")
    return " · ".join(parts) or "No direct public attributes or methods"


def mermaid_classes(
    classes: list[ClassInfo],
    links: dict[str, str],
    include_external_bases: bool = False,
) -> list[str]:
    known = {info.cls: info for info in classes}
    external = {
        base
        for info in classes
        for base in info.cls.__bases__
        if base is not object and base not in known
    }
    bases = (
        [
            ClassInfo(
                base,
                base.__module__,
                base.__name__,
                re.sub(r"\W", "_", f"{base.__module__}_{base.__name__}"),
            )
            for base in sorted(external, key=lambda cls: (cls.__module__, cls.__name__))
        ]
        if include_external_bases
        else []
    )
    lines = ["```mermaid", "classDiagram"]
    for info in classes + bases:
        lines.append(f'    class {info.ident}["{info.name}"]')
    for info in classes:
        for base in info.cls.__bases__:
            if base in known:
                lines.append(f"    {known[base].ident} <|-- {info.ident}")
            elif include_external_bases and base is not object:
                base_id = re.sub(r"\W", "_", f"{base.__module__}_{base.__name__}")
                lines.append(f"    {base_id} <|-- {info.ident}")
        tooltip = class_summary(info.cls).replace('"', "'")
        lines.append(f'    click {info.ident} href "{links[info.ident]}" "{tooltip}"')
    return [*lines, "```", ""]


def render_inheritance(module_name: str, names: list[str]) -> list[str]:
    """Render clickable direct class relationships with API summaries."""
    classes = class_info(module_name, names)
    if not classes:
        return []
    links = {info.ident: f"#{info.module}.{info.cls.__qualname__}" for info in classes}
    return ["## Inheritance", "", *mermaid_classes(classes, links, True)]


def section_classes(entries: list[tuple[str, str]]) -> list[ClassInfo]:
    return [
        info
        for _, module_name in entries
        for info in class_info(module_name, exported_api_items(module_name))
    ]


def render_section_overview(section: str, entries: list[tuple[str, str]]) -> str:
    classes = section_classes(entries)
    links = {
        info.ident: (
            f"../{info.module.rsplit('.', 1)[-1]}/"
            f"#{info.module}.{info.cls.__qualname__}"
        )
        for info in classes
    }
    out = [
        f"# {section_title(section)}",
        "",
        "This diagram is generated from the public API. Hover over a class for its "
        "direct attributes and methods, or select it to open the full reference.",
        "",
    ]
    if classes:
        out.extend(mermaid_classes(classes, links))
    return "\n".join(out)


def render_package_overview(modules: dict[str, list[tuple[str, str]]]) -> str:
    entries = [entry for section in SECTIONS for entry in modules[section]]
    infos = section_classes(entries)
    by_module = {module: [] for _, module in entries}
    for info in infos:
        by_module[info.module].append(info)
    by_module = {module: classes for module, classes in by_module.items() if classes}
    edges = {
        (base.__module__, info.module)
        for info in infos
        for base in info.cls.__bases__
        if base is not object
        and base.__module__ in by_module
        and base.__module__ != info.module
    }
    lines = ["```mermaid", "classDiagram"]
    ids = {module: re.sub(r"\W", "_", module) for module in by_module}
    for module, classes in by_module.items():
        label = module.removeprefix("dLux.")
        lines.append(f'    class {ids[module]}["{label}"]')
        tooltip = (
            f"Public classes: {', '.join(info.name for info in classes)}"
            or "Public functions"
        )
        lines.append(
            f'    click {ids[module]} href "../{section_path(module)}" "{tooltip}"'
        )
    lines.extend(
        f"    {ids[parent]} <|-- {ids[child]}" for parent, child in sorted(edges)
    )
    return "\n".join(
        [
            "# API",
            "",
            "This package map is generated from cross-module inheritance in the "
            "public API. Hover over a module to see its public classes, or select "
            "it to open its reference.",
            "",
            *lines,
            "```",
            "",
        ]
    )


def section_path(module: str) -> str:
    parts = module.split(".")
    section = parts[1] if len(parts) > 2 else "core"
    return f"{section}/{parts[-1]}/"


def render_page(title: str, module_name: str, names: list[str]) -> str:
    out = [f"# {title}", ""]
    module = importlib.import_module(module_name)

    if not names:
        out.append("No public classes or functions are exported by this module.")
        out.append("")
        return "\n".join(out)

    out.extend(render_inheritance(module_name, names))

    for i, name in enumerate(names):
        obj = getattr(module, name)
        target_module = getattr(obj, "__module__", module_name)
        target_name = getattr(obj, "__qualname__", name)
        out.append(f'???+ info "{name}"')
        out.append(f"    ::: {target_module}.{target_name}")
        if i != len(names) - 1:
            out.append("")

    out.append("")
    return "\n".join(out)


def render_api_nav_block(modules: dict[str, list[str]]) -> list[str]:
    lines = ["  - API:\n", "    - Overview: API/overview.md\n"]

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
    (API_ROOT / "overview.md").write_text(
        render_package_overview(modules), encoding="utf-8"
    )

    created = 1
    skipped = 0
    nav_modules: dict[str, list[str]] = {section: [] for section in SECTIONS}
    for section in SECTIONS:
        section_dir = API_ROOT / section
        overview = section_dir / "overview.md"
        overview.write_text(
            render_section_overview(section, modules[section]), encoding="utf-8"
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
                f"CREATE {md_path.relative_to(DOCS_ROOT)} "
                f"({module_name}: {len(names)} exported classes/functions)"
            )
            created += 1

    nav_changed = update_mkdocs_api_nav(nav_modules)
    print(f"mkdocs.yml API nav {'updated' if nav_changed else 'already up-to-date'}.")
    print(f"Done. Created {created} files, skipped {skipped} modules.")


if __name__ == "__main__":
    main()
