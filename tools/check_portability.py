#!/usr/bin/env python3
"""Exercise immutable-data CPU checks under explicit interpreters; no GPU use."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = """
import hashlib, json, platform
from tools.scientific_corrections import artifacts
from tools.scientific_metrics import dispersion
print(json.dumps({
 'python': platform.python_version(), 'platform': platform.platform(),
 'implementation': platform.python_implementation(),
 'artifact_sha256': {p: hashlib.sha256(text.encode('utf-8')).hexdigest() for p, text in artifacts().items()},
 'reported_case_cv': repr(dispersion([4067.8526683780683, 4071.2049695865194, 4071.239699282067])['sample_cv_pct'])
}, sort_keys=True))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", dest="interpreters", action="append", required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "validation/compatibility_proof.json")
    args = parser.parse_args()
    results = []
    for interpreter in args.interpreters:
        probe = subprocess.run([interpreter, "-c", PROBE], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", errors="replace")
        if probe.returncode:
            raise SystemExit("Interpreter probe failed: " + probe.stderr)
        item = json.loads(probe.stdout)
        item["checks"] = []
        for command in (["tools/validate.py"], ["-m", "unittest", "discover", "-s", "tests", "-v"],
                        ["tools/scientific_corrections.py", "--check"], ["tools/scientific_status.py", "--check"],
                        ["tools/build_report.py", "--check"], ["tools/prepare_notebook.py", "--check"]):
            completed = subprocess.run([interpreter, *command], cwd=ROOT, capture_output=True,
                                       text=True, encoding="utf-8", errors="replace")
            item["checks"].append({"command": ["python", *command], "exit_code": completed.returncode,
                                   "stdout": completed.stdout, "stderr": completed.stderr})
        print(item["python"] + ": " + ("PASS" if all(c["exit_code"] == 0 for c in item["checks"]) else "FAIL"), flush=True)
        results.append(item)
    same = all(item["artifact_sha256"] == results[0]["artifact_sha256"] for item in results)
    passed = same and all(check["exit_code"] == 0 for item in results for check in item["checks"])
    receipt = {"schema_version": 1, "author": "Mehran Keivanimehr", "date": "2026-10-15",
               "status": "passed" if passed else "failed", "artifact_hashes_identical": same,
               "historical_manifest_sha256": hashlib.sha256((ROOT / "provenance/measurements.sha256.json").read_bytes()).hexdigest(),
               "raw_measurements_modified": False, "gpu_experiments_performed": False,
               "rounding_method": "exact rational statistics; CV rounded using binary64 midpoint comparisons",
               "scope": "only the exact versions/platforms listed below; target Linux 3.10.22 rerun still required",
               "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes((json.dumps(receipt, indent=2) + "\n").encode("utf-8"))
    print("All correction-artifact hashes identical:", same)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
