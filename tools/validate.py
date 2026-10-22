#!/usr/bin/env python3
"""Validate frozen measurements, source syntax, and active local documentation links."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def manifest_errors(root: Path = ROOT) -> list[str]:
    path = root / "provenance/measurements.sha256.json"
    if not path.exists():
        return ["Measurement manifest is missing."]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    errors, seen = [], set()
    if manifest.get("algorithm") != "sha256" or manifest.get("schema_version") != 1:
        errors.append("Unsupported measurement manifest schema.")
    for item in manifest.get("files", []):
        relative = item["path"]
        candidate = root / relative
        if relative in seen:
            errors.append(f"Duplicate manifest path: {relative}")
        seen.add(relative)
        if not candidate.resolve().is_relative_to(root.resolve()):
            errors.append(f"Manifest path escapes repository: {relative}")
            continue
        if not candidate.is_file():
            errors.append(f"Missing measured artifact: {relative}")
        elif candidate.stat().st_size != item["bytes"] or hashlib.sha256(candidate.read_bytes()).hexdigest() != item["sha256"]:
            errors.append(f"Changed measured artifact: {relative}")
    if not seen:
        errors.append("Measurement manifest is empty.")
    return errors


def local_link_errors(root: Path = ROOT) -> list[str]:
    errors = []
    # Historical source documents are preserved even when they reference missing older work.
    excluded = {root / "studies/a100/REPORT.md", root / "studies/a100/summary.md"}
    for path in root.rglob("*.md"):
        if ".git" in path.parts or path.name.endswith(".original.md") or path in excluded:
            continue
        text = path.read_text(encoding="utf-8-sig")
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        for target in re.findall(r"\[[^\]]*\]\(([^\n]+?)\)", text):
            target = target.strip().split(' "', 1)[0].strip("<>")
            split = urlsplit(target)
            if split.scheme or not split.path or target.startswith("//"):
                continue
            candidate = (path.parent / unquote(split.path)).resolve()
            if not candidate.is_relative_to(root.resolve()):
                errors.append(f"Link escapes repository in {path.relative_to(root)}: {target}")
            elif not candidate.exists():
                errors.append(f"Broken link in {path.relative_to(root)}: {target}")
    return errors


def structure_errors(root: Path = ROOT) -> list[str]:
    errors = []
    forbidden = {"envs", "cache", ".venv", "node_modules"}
    token_pattern = re.compile(r"\b(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{25,})\b")
    for path in root.rglob("*"):
        if ".git" in path.relative_to(root).parts:
            continue
        if path.is_dir() and path.name in forbidden:
            errors.append(f"Excluded environment/cache directory: {path.relative_to(root)}")
        if not path.is_file():
            continue
        if path.stat().st_size > 100 * 1024**2:
            errors.append(f"File exceeds 100 MiB: {path.relative_to(root)}")
        if path.suffix in (".py", ".md", ".json", ".jsonl", ".ipynb", ".txt", ".sh", ".yaml", ".yml", ".toml", ".cff"):
            content = path.read_text(encoding="utf-8-sig")
            if token_pattern.search(content):
                errors.append(f"Possible credential in {path.relative_to(root)}")
            if path.suffix == ".py":
                try:
                    ast.parse(content, filename=str(path))
                except SyntaxError as exc:
                    errors.append(f"Invalid Python in {path.relative_to(root)}: {exc}")
            elif path.suffix in (".json", ".ipynb"):
                try:
                    json.loads(content)
                except json.JSONDecodeError as exc:
                    errors.append(f"Invalid JSON in {path.relative_to(root)}: {exc}")
            elif path.suffix == ".jsonl":
                for index, line in enumerate(content.splitlines(), 1):
                    if line.strip():
                        try:
                            json.loads(line)
                        except json.JSONDecodeError:
                            errors.append(f"Invalid JSONL in {path.relative_to(root)}:{index}")
    return errors


def main() -> int:
    errors = manifest_errors() + structure_errors() + local_link_errors()
    if errors:
        for error in errors:
            print(error)
        return 1
    print("Measurement hashes, Python/JSON syntax, repository structure, and local links passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
