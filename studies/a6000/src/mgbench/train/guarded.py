"""Rank entry point for single-GPU smoke runs: prove the driver-level device identity BEFORE torch is imported (no
CUDA context exists yet), then run mgbench.train. Launched as `python -m torch.distributed.run ... -m
mgbench.train.guarded --spec ... --out ...` with MGBENCH_EXPECTED_GPU_{UUID,PCI,INDEX} in the environment."""
from __future__ import annotations

import json
import os
import sys


def main() -> int:
    from ..device_identity import (chain_problems, driver_identity, driver_set_problems, multi_visibility_problems,
                                   visibility_problems)

    rec = {"stage": "rank process, before importing torch", "torch_already_imported": "torch" in sys.modules,
           "cvd": os.environ.get("CUDA_VISIBLE_DEVICES")}
    problems = ["torch was imported before the identity check"] if rec["torch_already_imported"] else []
    if os.environ.get("MGBENCH_EXPECTED_GPU_UUIDS"):         # Candidate 17: declared multi-device (pair) case
        try:
            uuids = os.environ["MGBENCH_EXPECTED_GPU_UUIDS"].split(",")
            pcis = os.environ["MGBENCH_EXPECTED_GPU_PCIS"].split(",")
            indices = [int(x) for x in os.environ["MGBENCH_EXPECTED_GPU_INDICES"].split(",")]
            local = int(os.environ.get("LOCAL_RANK", "-1"))
            if not (len(uuids) == len(pcis) == len(indices) > 1) or not 0 <= local < len(indices):
                raise ValueError(f"lists {len(uuids)}/{len(pcis)}/{len(indices)}, LOCAL_RANK {local}")
            rec.update(local_rank=local, expected={"uuid": uuids[local], "pci": pcis[local], "index": indices[local]})
            problems += multi_visibility_problems(rec["cvd"], indices)
            rec["driver"] = driver_identity()
            problems += driver_set_problems(rec["driver"], list(zip(uuids, pcis)))
        except Exception as exc:  # noqa: BLE001 - missing or unreadable identity refuses
            problems.append(f"multi-device identity unavailable or malformed ({exc!r})")
        rec["problems"] = problems
        os.environ["MGBENCH_DRIVER_IDENTITY"] = json.dumps(rec)
        print(f"[identity] rank pre-torch check: {'OK' if not problems else 'REFUSED: ' + '; '.join(problems)}", flush=True)
        if problems:
            return 7
        from . import __main__ as train_main

        return train_main.main()
    try:
        uuid = os.environ["MGBENCH_EXPECTED_GPU_UUID"]
        pci = os.environ["MGBENCH_EXPECTED_GPU_PCI"]
        index = int(os.environ["MGBENCH_EXPECTED_GPU_INDEX"])
    except (KeyError, ValueError) as exc:
        uuid = pci = index = None
        problems.append(f"expected identity missing from the environment ({exc!r})")
    if index is not None:
        problems += visibility_problems(rec["cvd"], index)
        try:
            rec["driver"] = driver_identity()
            problems += chain_problems(approved_uuid=uuid, approved_pci=pci, driver=rec["driver"])
        except Exception as exc:  # noqa: BLE001 - missing identity refuses
            problems.append(f"driver identity unavailable ({exc!r})")
    rec["problems"] = problems
    os.environ["MGBENCH_DRIVER_IDENTITY"] = json.dumps(rec)
    print(f"[identity] rank pre-torch check: {'OK' if not problems else 'REFUSED: ' + '; '.join(problems)}", flush=True)
    if problems:
        return 7
    from . import __main__ as train_main

    return train_main.main()


if __name__ == "__main__":
    sys.exit(main())
