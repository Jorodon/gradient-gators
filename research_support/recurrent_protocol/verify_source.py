"""Verify published recurrent source bytes without importing the scientific code."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


MANIFEST_NAME = "PUBLICATION_SOURCE_MANIFEST.json"
CATALOG_PATH = Path(__file__).with_name("source_catalog.json")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Source paths must be portable relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("Source path contains an absolute or traversal component")
    return value


def _inside(root: Path, relative: str) -> Path:
    candidate = root.joinpath(*PurePosixPath(_relative(relative)).parts)
    # Do not follow links into an unrelated source tree.
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Source archive contains a symbolic link")
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError("Source path escapes the extracted archive")
    return candidate


def verify(source_root: str | Path, catalog: dict | None = None) -> dict:
    """Return a receipt after validating the pinned public archive manifest.

    Only file reads, hashes, JSON parsing and Python syntax parsing occur. The
    source's modules, models, environments and launchers are never executed.
    """
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise ValueError("Supply the extracted source asset's top-level folder")
    catalog = _read_json(CATALOG_PATH) if catalog is None else catalog
    if catalog.get("schema_version") != 1:
        raise ValueError("Unsupported source catalog schema")
    expected_hash = catalog.get("publication_manifest_sha256")
    if not isinstance(expected_hash, str) or not SHA256.fullmatch(expected_hash):
        raise ValueError("Source catalog does not pin a prepared public manifest")
    manifest_path = _inside(root, MANIFEST_NAME)
    if _hash(manifest_path) != expected_hash:
        raise ValueError("Public source manifest differs from the repository catalog")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("study_id") != catalog.get("study_id"):
        raise ValueError("Unsupported or mismatched source manifest identity")
    if manifest.get("original_source_seal_sha256") != catalog.get("original_source_seal_sha256"):
        raise ValueError("Original source seal differs")
    entries = manifest.get("entries")
    expected = catalog.get("original_files")
    if not isinstance(entries, list) or not isinstance(expected, dict):
        raise ValueError("Missing source inventory")
    if len(entries) != len(expected) or manifest.get("original_first_party_count") != len(expected):
        raise ValueError("Source inventory count differs")
    seen_original, seen_public = set(), set()
    changed, parsed_python = [], 0
    for entry in entries:
        original = _relative(entry["original_path"])
        published = _relative(entry["published_path"])
        if original in seen_original or published in seen_public:
            raise ValueError("Duplicate original or published source path")
        seen_original.add(original)
        seen_public.add(published)
        if expected.get(original) != entry.get("original_sha256"):
            raise ValueError(f"Original identity differs: {original}")
        if not SHA256.fullmatch(str(entry.get("published_sha256", ""))):
            raise ValueError("Invalid published hash")
        target = _inside(root, published)
        if _hash(target) != entry["published_sha256"]:
            raise ValueError(f"Published bytes differ: {published}")
        same = entry["original_sha256"] == entry["published_sha256"]
        if type(entry.get("byte_identical")) is not bool or entry["byte_identical"] != same:
            raise ValueError(f"Byte-identity declaration differs: {published}")
        if not same:
            if not entry.get("transformation"):
                raise ValueError(f"Changed source has no transformation record: {published}")
            changed.append(published)
        if target.suffix == ".py":
            ast.parse(target.read_text(encoding="utf-8-sig"), filename=published)
            parsed_python += 1
    if seen_original != set(expected):
        raise ValueError("Source inventory differs")
    return {
        "status": "passed",
        "study_id": catalog["study_id"],
        "source_manifest_sha256": expected_hash,
        "original_source_seal_sha256": catalog["original_source_seal_sha256"],
        "first_party_files_verified": len(entries),
        "byte_identical_files": len(entries) - len(changed),
        "transformed_files": changed,
        "python_files_syntax_checked": parsed_python,
        "scientific_modules_executed": 0,
        "training_or_evaluation_steps": 0,
        "scope": "Published source bytes and syntax only; installation, training and replay reproduction remain separate checks.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="Extracted source asset folder")
    args = parser.parse_args()
    print(json.dumps(verify(args.source_root), indent=2))


if __name__ == "__main__":
    main()
