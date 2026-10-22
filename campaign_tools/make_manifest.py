"""LF-only SHA-256 manifest over the campaign artifacts, verifiable verbatim.

Lines are "<sha256>  <relative/path>\\n" with a single LF, sorted by path, so
the file reproduces byte-for-byte on any host and `sha256sum -c` accepts it
directly. Hashes are over raw bytes; nothing is normalised or re-serialised.
"""
from __future__ import annotations
import os

import argparse, hashlib, pathlib, sys

W = pathlib.Path(os.environ.get("TITAN_CAMPAIGN_ROOT",
    str(pathlib.Path(__file__).resolve().parent.parent)))
INCLUDE = ("runs", "results", "reports", "bundle", "ledger", "tools", "env", "logs")
SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints"}
# The vendored interpreter prefixes are environments, not artifacts: hashing
# ~40k site-packages files would bury the evidence it is meant to certify.
SKIP_PREFIXES = ("env/vllm-n3/", "env/bench-", "src/multigpu-bench/env")


def digest(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def walk(exclude: set[str] | None = None):
    """exclude: published paths to omit -- notably the manifest itself, whose
    own hash cannot be in it (the file changes the moment it is written)."""
    exclude = exclude or set()
    for top in INCLUDE:
        root = W / top
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.is_symlink():
                continue
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            rel = p.relative_to(W).as_posix()
            if rel.startswith(SKIP_PREFIXES) or rel.endswith(".pyc"):
                continue
            if rel in exclude:
                continue
            yield rel, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(W / "bundle" / "MANIFEST.sha256"))
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)

    if a.verify:
        if not out.exists():
            print(f"no manifest at {out}")
            return 2
        bad = missing = ok = 0
        raw = out.read_bytes()
        if b"\r" in raw:
            print("manifest contains CR bytes; it is not LF-only")
            return 2
        self_rel = out.resolve().relative_to(W).as_posix() \
            if out.resolve().is_relative_to(W) else None
        for line in raw.decode().splitlines():
            want, rel = line.split("  ", 1)
            if rel == self_rel:
                print(f"manifest lists itself ({rel}); that can never verify")
                bad += 1
                continue
            p = W / rel
            if not p.exists():
                print(f"MISSING  {rel}")
                missing += 1
            elif digest(p) != want:
                print(f"CHANGED  {rel}")
                bad += 1
            else:
                ok += 1
        print(f"verify: ok={ok} changed={bad} missing={missing}")
        return 0 if not (bad or missing) else 1

    self_rel = out.resolve().relative_to(W).as_posix() \
        if out.resolve().is_relative_to(W) else None
    lines = [f"{digest(p)}  {rel}\n"
             for rel, p in walk({self_rel} if self_rel else set())]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        fh.write("".join(lines).encode())
    raw = out.read_bytes()
    print(f"wrote {out}")
    lf_only = b"\r" not in raw
    print(f"  entries={len(lines)}  bytes={len(raw)}  LF_only={lf_only}")
    print(f"  manifest_sha256={hashlib.sha256(raw).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
