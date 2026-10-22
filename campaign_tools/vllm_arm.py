"""Run a vLLM arm with numeric device visibility and VERIFIED UUID binding.

Why this wrapper exists. Every other arm in this campaign is launched with
`CUDA_VISIBLE_DEVICES` set to device UUIDs, so the process cannot be bound to an
unintended card by index reordering. vLLM 0.6.3 cannot accept that: its NVML
path does `int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[device_id])`
(vllm/platforms/cuda.py:98) and raises ValueError on a UUID. This is a hard
incompatibility in the engine, not a configuration choice.

Rather than drop the safeguard for these arms, the binding is inverted: the
launcher passes numeric indices under CUDA_DEVICE_ORDER=PCI_BUS_ID together with
the UUIDs it expects, and this wrapper refuses to start unless the mapping holds
BOTH outside and inside the process:

  1. nvidia-smi reports exactly the expected UUID at each requested index
  2. the CUDA runtime, after init, reports the same device count and the same
     ordered device names as those UUIDs

That is a weaker *input* guarantee and a stronger *checked* one, and the arm
records it as such. It is stated, not silently substituted.
"""
from __future__ import annotations

import json, os, subprocess, sys


def smi_map(devs: list[int]) -> list[dict]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader",
         "-i", ",".join(str(d) for d in devs)],
        capture_output=True, text=True, timeout=60, check=True).stdout
    rows = []
    for line in out.strip().splitlines():
        idx, uuid, name = (x.strip() for x in line.split(",", 2))
        rows.append({"index": int(idx), "uuid": uuid, "name": name})
    return rows


def main() -> int:
    devs = [int(x) for x in os.environ["BENCH_PHYS_DEVICES"].split(",")]
    expected = [u.strip() for u in os.environ["BENCH_EXPECTED_UUIDS"].split(",")]
    if len(expected) != len(devs):
        print(f"REFUSED: {len(devs)} devices but {len(expected)} expected UUIDs")
        return 90
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        print("REFUSED: CUDA_DEVICE_ORDER must be PCI_BUS_ID for numeric visibility")
        return 90
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if vis != ",".join(str(d) for d in devs):
        print(f"REFUSED: CUDA_VISIBLE_DEVICES {vis!r} does not match "
              f"BENCH_PHYS_DEVICES {devs}")
        return 90

    # (1) external check: does each requested index really hold the expected card?
    rows = smi_map(devs)
    if [r["index"] for r in rows] != devs:
        print(f"REFUSED: nvidia-smi returned indices {[r['index'] for r in rows]}, expected {devs}")
        return 90
    for r, want in zip(rows, expected):
        if r["uuid"] != want:
            print(f"REFUSED: index {r['index']} holds {r['uuid']}, expected {want}")
            return 90

    # (2) internal check: does the CUDA runtime agree, after it has initialised?
    import torch
    if not torch.cuda.is_available():
        print("REFUSED: CUDA unavailable inside the process")
        return 90
    if torch.cuda.device_count() != len(devs):
        print(f"REFUSED: runtime sees {torch.cuda.device_count()} devices, expected {len(devs)}")
        return 90
    runtime_names = [torch.cuda.get_device_name(i) for i in range(len(devs))]
    if runtime_names != [r["name"] for r in rows]:
        print(f"REFUSED: runtime names {runtime_names} != nvidia-smi names "
              f"{[r['name'] for r in rows]}")
        return 90

    receipt = {
        "binding_mode": "numeric_visibility_with_verified_uuid_postcondition",
        "reason": "vllm 0.6.3 cuda.py:98 does int() on CUDA_VISIBLE_DEVICES entries "
                  "and raises ValueError on a UUID; UUID-bound visibility is "
                  "structurally impossible for this engine",
        "weaker_than": "UUID-bound CUDA_VISIBLE_DEVICES used by every other arm",
        "verified_externally": rows,
        "verified_expected_uuids": expected,
        "verified_runtime_device_names": runtime_names,
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "cuda_visible_devices": vis,
    }
    d = os.environ["BENCH_RESULTS_DIR"]
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "uuid_binding_receipt.json"), "w") as fh:
        json.dump(receipt, fh, indent=2, sort_keys=True)
    print("[binding] numeric visibility, UUID postcondition VERIFIED "
          f"({', '.join(r['name'] for r in rows)})", flush=True)

    target = sys.argv[1]
    sys.argv = [target] + sys.argv[2:]
    g = {"__name__": "__main__", "__file__": target}
    with open(target) as fh:
        code = compile(fh.read(), target, "exec")
    exec(code, g)          # noqa: S102 - running our own benchmark module
    return 0


if __name__ == "__main__":
    sys.exit(main())
