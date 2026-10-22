#!/usr/bin/env python3
"""Run the offline evidence checks and retain their exact outputs and source hashes.

No CUDA workloads or original benchmark-environment installations are performed.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

try:
    from .package_release import source_files
except ImportError:
    from package_release import source_files

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bash", type=Path, help="optional Bash executable for script syntax checks")
    args = parser.parse_args()
    output = ROOT / "validation/local_verification.json"
    if not output.exists():
        output.write_text('{"status": "running; no completed verification claimed"}\n', encoding="utf-8")
    checks = []
    commands = [
        ("measurement_integrity_syntax_structure_links", [sys.executable, "tools/validate.py"]),
        ("offline_regressions", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]),
        ("correction_artifacts", [sys.executable, "tools/scientific_corrections.py", "--check"]),
        ("status_register", [sys.executable, "tools/scientific_status.py", "--check"]),
        ("combined_report", [sys.executable, "tools/build_report.py", "--check"]),
        ("historical_notebook_builder", [sys.executable, "tools/prepare_notebook.py", "--check"]),
    ]
    has_git = (ROOT / ".git").exists()
    if has_git:
        commands.append(("whitespace_diff", ["git", "diff", "--check"]))
    for name, command in commands:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        portable_command = ["python" if part == sys.executable else part for part in command]
        checks.append({"name": name, "command": portable_command, "exit_code": result.returncode,
                       "stdout": result.stdout, "stderr": result.stderr})
        print(name + ": " + ("PASS" if result.returncode == 0 else "FAIL"))
    shell_count = 0
    if args.bash:
        for path in sorted(ROOT.glob("studies/**/*.sh")):
            result = subprocess.run([str(args.bash), "-n", path.as_posix()], cwd=ROOT,
                                    capture_output=True, text=True, encoding="utf-8", errors="replace")
            checks.append({"name": "shell_syntax:" + path.relative_to(ROOT).as_posix(),
                           "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
            shell_count += 1
    # Hash the corrected sources/docs/configuration, not the receipt itself or
    # enormous duplicated data. The separate historical manifest covers data.
    hashes = {}
    for path in source_files(ROOT):
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix not in (".py", ".md", ".yaml", ".yml", ".sh", ".ipynb"):
            continue
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {"schema_version": 1, "author": "Mehran Keivanimehr",
               "status": "completed",
               "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "python": platform.python_version(), "platform": platform.platform(),
               "baseline_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() if has_git else None,
               "git_diff_check_available": has_git,
               "command_recording": "interpreter represented portably as python; exact version recorded above",
               "working_tree_has_corrections": True, "gpu_experiments_performed": False,
               "cuda_and_framework_integration_tested": False,
               "yaml_templates_executed": False,
               "scientific_certification": False,
               "checks": checks, "shell_scripts_checked": shell_count,
               "source_sha256": hashes,
               "boundary": "CPU regression and integrity evidence only; new GPU and original framework tests remain pending"}
    output.write_bytes((json.dumps(receipt, indent=2) + "\n").encode("utf-8"))
    failed = sum(item["exit_code"] != 0 for item in checks)
    print(f"Receipt: validation/local_verification.json; {failed} failed check commands")
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
