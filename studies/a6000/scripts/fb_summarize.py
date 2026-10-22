#!/usr/bin/env python
"""Post-window first-boundary summarizer (CPU only; numpy + stdlib). Explicit roots; inputs are only read.

  python -B -s scripts/fb_summarize.py --window <window dir> --reference <sealed r0 reference dir> \
      --expectations <frozen expectations.json> --out <NEW output dir> [--pair G1_ddp_autocast:G0_fp32_control ...]

Run with CUDA hidden and private cache variables (TRITON_CACHE_DIR, XDG_CACHE_HOME, ...) set; nothing here imports
torch, DeepSpeed or Triton."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mgbench.fb_summary import CHUNK, summarize  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--expectations", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pair", action="append", default=[])
    ap.add_argument("--chunk", type=int, default=CHUNK)
    a = ap.parse_args(argv)
    if a.out.resolve().is_relative_to(a.window.resolve()) or a.out.resolve().is_relative_to(a.reference.resolve()):
        sys.exit("output root must not lie inside the window or the reference")
    exp = json.loads(a.expectations.read_text())
    frozen = {"input_per_step_microbatch_sha256": exp["input_per_step_microbatch_sha256"], "recipe": exp["recipe"],
              "tied_aliases": exp.get("tied_aliases", {})}
    s = summarize(a.window, a.reference, a.out, frozen, pairs=[tuple(p.split(":")) for p in a.pair], chunk=a.chunk)
    print(json.dumps({cid: c["status"] for cid, c in s["cases"].items()}))
    return 0 if all(c["status"] == "summarized" for c in s["cases"].values()) else 3


if __name__ == "__main__":
    sys.exit(main())
