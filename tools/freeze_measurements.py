#!/usr/bin/env python3
"""Create the manifest once; replacing it requires deliberate acceptance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "provenance/measurements.sha256.json"


def measured_paths(root: Path):
    paths = []
    for campaign in ("a6000", "titan-1080"):
        paths.extend((root / "studies" / campaign / "results").rglob("*"))
    paths.extend(path for path in (root / "studies/a100").glob("*")
                 if path.suffix in (".csv", ".json", ".jsonl", ".txt") or path.name in ("REPORT.md", "summary.md"))
    paths.extend((root / "provenance").glob("*.original.md"))
    paths.extend((root / "provenance").glob("*.ipynb"))
    paths.extend((root / "studies/a6000/reports/figures").glob("*.png"))
    return sorted({path for path in paths if path.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-new-data", action="store_true", help="explicitly replace the recorded baseline manifest")
    args = parser.parse_args()
    if MANIFEST.exists() and not args.accept_new_data:
        parser.error("Manifest already exists. Verify it; replace only after reviewing intentional dataset changes.")
    entries = [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in measured_paths(ROOT)]
    MANIFEST.write_bytes((json.dumps({"schema_version": 1, "algorithm": "sha256", "files": entries}, indent=2) + "\n").encode("utf-8"))
    print(f"Recorded {len(entries)} source artifact hashes.")


if __name__ == "__main__":
    main()
