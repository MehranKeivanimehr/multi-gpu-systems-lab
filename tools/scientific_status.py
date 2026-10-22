#!/usr/bin/env python3
"""Generate the finding-by-finding scientific status register, never a certificate."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def document():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_showcase import scientific_status
    return scientific_status(ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = ROOT / "docs/SCIENTIFIC_STATUS.md"
    content = document().encode("utf-8")
    if args.check:
        if not path.exists() or path.read_bytes() != content:
            raise SystemExit("Scientific status document is stale")
        print("Scientific status register matches the ledger.")
    else:
        path.write_bytes(content)
        print("Wrote scientific status register.")


if __name__ == "__main__":
    main()
