"""Generate the documentation version index from deployed releases."""

import argparse
import json
import re
from pathlib import Path

VERSION_PATTERN = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


def generate_versions(docs_root, output, current_version=None):
    """Generate a Mike-compatible ``versions.json`` file."""
    versions = {}

    for path in Path(docs_root).iterdir():
        match = VERSION_PATTERN.fullmatch(path.name)
        if path.is_dir() and match is not None:
            version = ".".join(match.groups())
            versions[version] = tuple(map(int, match.groups()))

    if current_version:
        match = VERSION_PATTERN.fullmatch(current_version)
        if match is None:
            raise ValueError(f"Invalid release version: {current_version}")
        version = ".".join(match.groups())
        versions[version] = tuple(map(int, match.groups()))

    if not versions:
        raise ValueError("No deployed documentation versions found.")

    ordered = sorted(versions, key=versions.get, reverse=True)
    entries = [
        {
            "version": version,
            "title": f"v{version}",
            "aliases": ["latest"] if version == ordered[0] else [],
        }
        for version in ordered
    ]

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(entries, indent=2) + "\n")

    return ordered[0]


def main():
    """Generate the version index and print the latest stable version."""
    parser = argparse.ArgumentParser()
    parser.add_argument("docs_root")
    parser.add_argument("output")
    parser.add_argument("current_version", nargs="?", default=None)
    args = parser.parse_args()

    print(generate_versions(args.docs_root, args.output, args.current_version))


if __name__ == "__main__":
    main()
