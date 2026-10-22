"""
GPU inventory and topology-aware device-group selection.

This box has no NVLink: every GPU pair talks over PCIe, and the link quality
varies a lot. `nvidia-smi topo -m` reports, for this host:

  * GPU0 (TITAN RTX)  -> PHB to everything, including the other TITAN RTX
  * GPUs {1,2,3,4,5}  -> PIX to each other (single PCIe bridge)
  * GPUs {6,7,8,9}    -> PIX to each other

So the two TITAN RTX (0 and 2) are PHB-connected -- the *worst* link class
available here -- while four of the 1080 Ti ({1,3,4,5}) sit behind one bridge.
Picking "the first N GPUs of type X" would therefore silently hand the 1080 Ti
group a better interconnect in some runs and a worse one in others. Instead we
enumerate same-model combinations and maximise the worst link in the group,
then record which link classes the chosen group actually spans so the scaling
numbers can be read in context.
"""

from __future__ import annotations

import itertools
import json
import re
import subprocess
from dataclasses import dataclass
from typing import Any

from common import query_gpus

# Best -> worst. NVLink first even though this host has none, so the selector
# keeps working if the suite is ever moved to a machine that does.
LINK_RANK = {
    "NV": 100,  # NV1, NV2, ... bonded NVLink
    "PIX": 50,  # at most one PCIe bridge
    "PXB": 40,  # multiple PCIe bridges, no host bridge
    "NODE": 30,  # PCIe + host bridges within a NUMA node
    "PHB": 20,  # traverses the PCIe host bridge (CPU)
    "SYS": 10,  # crosses the inter-socket link
    "X": 1000,  # self
}


def _link_class(token: str) -> str:
    token = token.strip().upper()
    if token.startswith("NV") and token != "NODE":
        return "NV"
    return token if token in LINK_RANK else "SYS"


def parse_topo() -> dict[tuple[int, int], str]:
    """Parse `nvidia-smi topo -m` into {(i,j): link_class}."""
    out = subprocess.run(
        ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=60
    )
    if out.returncode != 0:
        raise RuntimeError(f"nvidia-smi topo failed: {out.stderr.strip()}")

    ansi = re.compile(r"\x1b\[[0-9;]*m")
    links: dict[tuple[int, int], str] = {}
    for line in out.stdout.splitlines():
        line = ansi.sub("", line).rstrip()
        m = re.match(r"^GPU(\d+)\s+(.*)$", line.strip())
        if not m:
            continue
        src = int(m.group(1))
        toks = m.group(2).split()
        col = 0
        for tok in toks:
            # The row ends with CPU-affinity / NUMA columns; link tokens are a
            # known vocabulary, so stop at the first thing that is not one.
            t = tok.strip().upper()
            is_link = t == "X" or t in LINK_RANK or (
                t.startswith("NV") and t != "NODE"
            )
            if not is_link:
                break
            links[(src, col)] = _link_class(tok)
            col += 1
    return links


@dataclass
class Gpu:
    index: int
    name: str
    compute_cap: str
    total_mib: float

    @property
    def short(self) -> str:
        n = self.name.replace("NVIDIA ", "").replace("GeForce ", "")
        return n


def inventory() -> list[Gpu]:
    return [
        Gpu(
            index=r["index"],
            name=r["name"],
            compute_cap=r["compute_cap"],
            total_mib=r["memory.total"] or 0.0,
        )
        for r in query_gpus()
    ]


def group_link_profile(
    members: list[int], links: dict[tuple[int, int], str]
) -> dict[str, Any]:
    """Link classes spanned by a group, plus its worst link."""
    classes: dict[str, int] = {}
    worst = "NV"
    for a, b in itertools.combinations(sorted(members), 2):
        cls = links.get((a, b)) or links.get((b, a)) or "SYS"
        classes[cls] = classes.get(cls, 0) + 1
        if LINK_RANK[cls] < LINK_RANK[worst]:
            worst = cls
    if len(members) == 1:
        worst = "N/A"
    return {"worst_link": worst, "link_counts": classes}


def select_group(
    model_substr: str,
    n: int,
    gpus: list[Gpu] | None = None,
    links: dict[tuple[int, int], str] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Pick `n` GPUs whose name contains `model_substr`, best interconnect first.

    Scoring is lexicographic over the sorted (worst-first) list of pairwise link
    ranks, so we maximise the weakest link, then the second weakest, and so on.
    Ties break toward lower physical indices for reproducibility.
    """
    gpus = gpus if gpus is not None else inventory()
    links = links if links is not None else parse_topo()

    cands = [g.index for g in gpus if model_substr.lower() in g.name.lower()]
    if len(cands) < n:
        raise ValueError(
            f"need {n} GPUs matching {model_substr!r}, found {len(cands)}: {cands}"
        )
    if n == 1:
        pick = [min(cands)]
        return pick, group_link_profile(pick, links)

    best: tuple[list[int], list[int]] | None = None
    for combo in itertools.combinations(sorted(cands), n):
        ranks = sorted(
            LINK_RANK[
                links.get((a, b)) or links.get((b, a)) or "SYS"
            ]
            for a, b in itertools.combinations(combo, 2)
        )
        if best is None or ranks > best[1]:
            best = (list(combo), ranks)
    assert best is not None
    return best[0], group_link_profile(best[0], links)


# The exact groups Stage 1 runs on, resolved from live topology at call time.
# A 4th element pins an explicit device list instead of letting the selector
# choose; used where the operator named the devices.
GROUP_SPECS: list[tuple[str, str, int, list[int] | None]] = [
    ("rtx_1", "TITAN RTX", 1, None),
    ("rtx_2", "TITAN RTX", 2, None),
    ("p1080_1", "1080 Ti", 1, None),
    ("p1080_2", "1080 Ti", 2, None),
    ("p1080_4", "1080 Ti", 4, None),
    # Operator-specified. Topologically identical to the selector's own pick of
    # {1,3,4,5,6,8}: both span 7 PIX pairs and 8 PHB pairs with a PHB worst
    # link, so this is the same experiment on the requested devices.
    ("p1080_6", "1080 Ti", 6, [1, 3, 4, 5, 8, 9]),
    ("xp_1", "TITAN Xp", 1, None),
]


def resolve_groups() -> dict[str, dict[str, Any]]:
    gpus = inventory()
    links = parse_topo()
    by_idx = {g.index: g for g in gpus}
    out: dict[str, dict[str, Any]] = {}
    for key, model, n, pinned in GROUP_SPECS:
        try:
            if pinned is not None:
                avail = {g.index for g in gpus if model.lower() in g.name.lower()}
                missing = [d for d in pinned if d not in avail]
                if missing:
                    raise ValueError(
                        f"pinned devices {missing} are not {model} on this host"
                    )
                members, profile = pinned, group_link_profile(pinned, links)
                profile["pinned"] = True
            else:
                members, profile = select_group(model, n, gpus, links)
        except ValueError as exc:
            out[key] = {"error": str(exc)}
            continue
        out[key] = {
            "model": by_idx[members[0]].short,
            "compute_cap": by_idx[members[0]].compute_cap,
            "n_gpus": n,
            "devices": members,
            "cuda_visible_devices": ",".join(str(i) for i in members),
            "vram_mib_each": by_idx[members[0]].total_mib,
            **profile,
        }
    return out


if __name__ == "__main__":
    gpus = inventory()
    links = parse_topo()
    print("=== GPU inventory ===")
    for g in gpus:
        print(
            f"  GPU{g.index}: {g.short:<18} cap {g.compute_cap}  "
            f"{g.total_mib/1024:.0f} GB"
        )
    print("\n=== Resolved benchmark groups ===")
    groups = resolve_groups()
    for key, info in groups.items():
        if "error" in info:
            print(f"  {key:<10} UNAVAILABLE: {info['error']}")
            continue
        print(
            f"  {key:<10} {info['model']:<18} n={info['n_gpus']}  "
            f"devices={info['cuda_visible_devices']:<12} "
            f"worst_link={info['worst_link']:<5} {info['link_counts']}"
        )
    print()
    print(json.dumps({"groups": groups}, indent=2, default=str))
