#!/usr/bin/env python3
"""Create a deterministic source ZIP without environments, caches, or Git metadata."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
                  ".venv", "venv", "envs", "cache", ".cache", "logs", ".ipynb_checkpoints"}


def source_files(root: Path):
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts) or not path.is_file():
            continue
        if path.suffix in (".pyc", ".pyo", ".whl", ".safetensors", ".ckpt", ".pt", ".pth"):
            continue
        if path.suffix == ".zip" and relative.as_posix() != "audit/original-audit.zip":
            continue
        yield path


def package(output: Path):
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Choose an output outside the repository root.")
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = list(source_files(ROOT))
    with ZipFile(output, "x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            name = "multigpu-bench/" + path.relative_to(ROOT).as_posix()
            item = ZipInfo(name, date_time=(2026, 10, 14, 0, 0, 0))
            item.compress_type = ZIP_DEFLATED
            item.create_system = 3
            mode = 0o100755 if path.suffix == ".sh" else 0o100644
            item.external_attr = mode << 16
            archive.writestr(item, path.read_bytes())
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_bytes(f"{checksum}  {output.name}\n".encode("ascii"))
    print(f"Packaged {len(paths)} files into {output.name} ({output.stat().st_size:,} bytes).")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new ZIP file outside the repository")
    args = parser.parse_args()
    package(args.output)


if __name__ == "__main__":
    main()
