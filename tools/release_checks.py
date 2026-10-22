"""Explicit offline acceptance profiles; check artifacts without regeneration.

No profile starts a GPU-capable worker or imports the installed vLLM stack.
The Torch profile requires CPU-only Torch; missing Torch is failure.
"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PORTABLE = ["test_scientific_metrics", "test_repository", "test_generated_consistency",
            "zd_test_stream_evidence", "zd_test_offline_recompute",
            "zd_test_shipped_evidence", "test_local_release_corrections", "test_publication"]
TORCH = ["test_update_oracle", "zd_test_chunked_oracle", "zd_test_capture_point",
         "zd_test_consistency_criterion", "zd_test_ref_compare", "zd_test_lossless_arrays",
         "test_a6000_boundary_ledger", "test_a6000_c20_ledger"]
POSIX = ["test_routed_lifecycle", "test_process_ownership", "test_campaign_cli_failstop",
         "zd_test_lifecycle", "test_stream_operational_posix"]


def snapshot():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts
            and ".git" not in p.parts}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", choices=["portable", "torch", "posix"], required=True)
    a = ap.parse_args()
    if a.profile == "posix" and sys.platform != "linux":
        raise SystemExit("POSIX profile requires Linux; not accepted by a Windows skip")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    if a.profile == "portable":
        commands = [["tools/validate.py"], ["tools/build_showcase.py", "--check"],
                    ["tools/render_figures.py", "--check"], ["tools/verify_a6000.py"], ["tools/build_report.py", "--check"],
                    ["tools/scientific_corrections.py", "--check"],
                    ["tools/scientific_status.py", "--check"],
                    ["tools/prepare_notebook.py", "--check"],
                    ["-m", "unittest", *["tests." + t for t in PORTABLE], "-v"]]
    elif a.profile == "torch":
        commands = [["-c", "import torch; assert torch.version.cuda is None, 'CPU-only Torch required'; print(torch.__version__)"]]
        commands += [["-m", "unittest", *["tests." + t for t in TORCH], "-v"]]
    else:
        commands = [["-m", "unittest", "tests." + t, "-v"] for t in POSIX]
    before = snapshot()
    rc = 0
    for command in commands:
        print("CHECK", " ".join(command), flush=True)
        isolation = ["-S"] if sys.flags.no_site else []
        result = subprocess.run([sys.executable, "-X", "utf8", "-B", *isolation, *command],
                                cwd=ROOT, env=env, timeout=600)
        if result.returncode:
            rc = 1
    if before != snapshot():
        print("REFUSED: checks changed repository files", file=sys.stderr)
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
