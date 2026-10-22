"""Prove CUDA_VISIBLE_DEVICES UUID selection is real, not a cosmetic env echo.

torch 2.4.1 exposes no device uuid or pci_bus_id, so a rank cannot report its own
hardware UUID. This control establishes UUID-bound selection differently: if a
wrong/nonexistent UUID yields ZERO visible devices while a specific real UUID
yields exactly that device's model, then the UUID we pass is genuinely selecting
hardware. That is weaker than a runtime-reported UUID and is labelled as such.
"""
import json, os, subprocess, sys
PY = sys.executable
PROBE = ("import json,torch;"
         "print(json.dumps({'count':torch.cuda.device_count(),"
         "'names':[torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())]}))")

def probe(visible):
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID",
               CUDA_VISIBLE_DEVICES=visible, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([PY, "-c", PROBE], capture_output=True, text=True, timeout=180, env=env)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {"count": None, "names": [], "stderr": r.stderr[-200:]}

inv = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout
rows = [[c.strip() for c in l.split(",")] for l in inv.strip().splitlines()]
by_idx = {int(r[0]): (r[1], r[2]) for r in rows}
rtx = next(i for i, (u, n) in by_idx.items() if "TITAN RTX" in n)
ti = next(i for i, (u, n) in by_idx.items() if "1080 Ti" in n)
xp = next(i for i, (u, n) in by_idx.items() if "TITAN Xp" in n)
bogus = "GPU-00000000-0000-0000-0000-000000000100"

cases = []
for label, vis, expect in (
    (f"real TITAN RTX uuid (idx {rtx})", by_idx[rtx][0], "TITAN RTX"),
    (f"real GTX 1080 Ti uuid (idx {ti})", by_idx[ti][0], "1080 Ti"),
    (f"real TITAN Xp uuid (idx {xp})", by_idx[xp][0], "TITAN Xp"),
    ("nonexistent uuid", bogus, None),
    ("malformed uuid", "GPU-not-a-uuid", None),
):
    r = probe(vis)
    got = r["names"][0] if r["names"] else None
    ok = (expect is None and r["count"] == 0) or (expect and r["count"] == 1 and expect in (got or ""))
    cases.append({"case": label, "visible": vis, "count": r["count"],
                  "names": r["names"], "expected": expect, "selective": bool(ok)})
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<34} count={r['count']} names={r['names']}")

# two real UUIDs together must expose exactly those two, in order
pair = f"{by_idx[rtx][0]},{by_idx[ti][0]}"
r = probe(pair)
ok = r["count"] == 2 and "TITAN RTX" in r["names"][0] and "1080 Ti" in r["names"][1]
cases.append({"case": "ordered pair of real uuids", "visible": pair, "count": r["count"],
              "names": r["names"], "expected": "TITAN RTX then 1080 Ti", "selective": bool(ok)})
print(f"  {'PASS' if ok else 'FAIL'}  {'ordered pair of real uuids':<34} count={r['count']} names={r['names']}")

out = {"purpose": "CUDA UUID-selection control",
       "limitation": ("torch 2.4.1 exposes no device uuid or pci_bus_id, so a rank "
                      "cannot self-report its hardware UUID. This establishes that "
                      "the UUID we pass genuinely selects hardware; it is NOT a "
                      "runtime-reported UUID and is not labelled as one."),
       "inventory": {str(k): {"uuid": v[0], "name": v[1]} for k, v in by_idx.items()},
       "cases": cases,
       "all_selective": all(c["selective"] for c in cases)}
json.dump(out, open(sys.argv[1], "w"), indent=2)
print(f"\n  UUID selection is real and discriminating: {out['all_selective']}")
