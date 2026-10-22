"""Device-identity chain for single-GPU launches (A6000 stage 20261016s2). No torch import in this module.

Chain that must hold, in this order, before any model allocation or DeepSpeed initialisation:
  1. NVML index N resolves to the approved UUID and PCI bus id                      (supervisor, before launch)
  2. the constrained child sees exactly one CUDA device; driver ordinal 0 has that UUID and PCI bus id
                                                                                    (CUDA Driver API, no context)
  3. DeepSpeed maps logical device 0 to NVML index N, whose UUID/PCI are the approved ones
  4. torch reports the same UUID/PCI for its current device after set_device and after the allocator cap
NVML and CUDA enumerate devices independently, so a PCI-order setting is not taken as proof; every link is compared
by UUID and PCI bus id. Any missing value is a refusal.

The Driver-API query calls only cuInit, cuDeviceGetCount, cuDeviceGet, cuDeviceGetUuid(_v2), cuDeviceGetPCIBusId and
cuDeviceGetName: it creates or retains no context, allocates nothing and launches no kernel.
"""
from __future__ import annotations

import ctypes
import json
import re
import sys

UUID_RE = re.compile(r"^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
PCI_RE = re.compile(r"^(?:([0-9a-fA-F]{4,8}):)?([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])$")


def normalize_pci(bus_id) -> str | None:
    """'00000000:81:00.0' (NVML), '0000:81:00.0' (CUDA) -> '0000:81:00.0'; None if malformed."""
    if isinstance(bus_id, bytes):
        bus_id = bus_id.decode(errors="replace")
    m = PCI_RE.match(str(bus_id or "").strip())
    if not m:
        return None
    dom, bus, dev, fn = m.groups()
    return f"{int(dom or '0', 16):04x}:{int(bus, 16):02x}:{int(dev, 16):02x}.{int(fn)}"


def format_uuid(raw16: bytes) -> str:
    h = bytes(raw16).hex()
    return f"GPU-{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def visibility_problems(cvd, expected_index: int) -> list[str]:
    """CUDA_VISIBLE_DEVICES must be exactly the one approved physical index as a plain integer string (the form the
    installed DeepSpeed and vLLM parse); anything else - list, UUID, whitespace, empty, negative - refuses."""
    if cvd is None:
        return ["CUDA_VISIBLE_DEVICES unset (all devices would be visible)"]
    if not re.fullmatch(r"(0|[1-9][0-9]*)", str(cvd)):
        return [f"CUDA_VISIBLE_DEVICES={cvd!r} is not a single non-negative integer"]
    if int(cvd) != int(expected_index):
        return [f"CUDA_VISIBLE_DEVICES={cvd!r} is not the approved index {expected_index}"]
    return []


def multi_visibility_problems(cvd, expected_indices: list[int]) -> list[str]:
    """Candidate 17, declared multi-device case: CUDA_VISIBLE_DEVICES must be exactly the approved physical indices,
    comma-separated plain integers in the declared rank order (rank r -> ordinal r -> expected_indices[r])."""
    if cvd is None:
        return ["CUDA_VISIBLE_DEVICES unset (all devices would be visible)"]
    if not re.fullmatch(r"(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*", str(cvd)):
        return [f"CUDA_VISIBLE_DEVICES={cvd!r} is not a comma-separated list of non-negative integers"]
    got = [int(x) for x in str(cvd).split(",")]
    if got != [int(i) for i in expected_indices] or len(set(got)) != len(got):
        return [f"CUDA_VISIBLE_DEVICES={cvd!r} is not the approved ordered set {expected_indices}"]
    return []


def driver_set_problems(driver: dict | None, approved: list[tuple[str, str]]) -> list[str]:
    """Every visible driver ordinal r must carry approved[r] (UUID AND PCI); count, duplicates, swaps, missing or extra
    devices refuse. Ordinal agreement alone is never taken as identity."""
    if not isinstance(driver, dict):
        return ["driver identity unreadable"]
    devs = driver.get("devices") or []
    out = []
    if driver.get("count") != len(approved) or len(devs) != len(approved):
        out.append(f"driver: {driver.get('count')} visible CUDA devices ({len(devs)} listed), exactly {len(approved)} required")
    uuids = [d.get("uuid") for d in devs]
    if len(set(uuids)) != len(uuids):
        out.append(f"driver: duplicate UUIDs {uuids}")
    for r, (uuid, pci) in enumerate(approved):
        d = next((x for x in devs if x.get("ordinal") == r), None)
        if d is None:
            out.append(f"driver: ordinal {r} missing")
            continue
        out += chain_problems(approved_uuid=uuid, approved_pci=pci, nvml={"index": f"ordinal {r}", "uuid": d.get("uuid"),
                                                                         "pci_bus_id": d.get("pci_bus_id")})
    return [p.replace("NVML index ordinal", "driver ordinal") for p in out]


def _check(lib, name, *args):
    rc = getattr(lib, name)(*args)
    if rc != 0:
        raise RuntimeError(f"{name} returned CUresult {rc}")


def driver_identity(lib=None) -> dict:
    """Identity of every device the CUDA driver exposes to this process (normally exactly one)."""
    lib = lib if lib is not None else ctypes.CDLL("libcuda.so.1")
    _check(lib, "cuInit", 0)
    count = ctypes.c_int(-1)
    _check(lib, "cuDeviceGetCount", ctypes.byref(count))
    devices = []
    for ordinal in range(max(0, count.value)):
        dev = ctypes.c_int(-1)
        _check(lib, "cuDeviceGet", ctypes.byref(dev), ordinal)
        uuid = (ctypes.c_ubyte * 16)()
        fn = "cuDeviceGetUuid_v2" if getattr(lib, "cuDeviceGetUuid_v2", None) is not None else "cuDeviceGetUuid"
        _check(lib, fn, uuid, dev)
        pci = ctypes.create_string_buffer(64)
        _check(lib, "cuDeviceGetPCIBusId", pci, 64, dev)
        name = ctypes.create_string_buffer(256)
        _check(lib, "cuDeviceGetName", name, 256, dev)
        devices.append({"ordinal": ordinal, "uuid": format_uuid(bytes(uuid)), "pci_bus_id": pci.value.decode(),
                        "name": name.value.decode(), "uuid_api": fn})
    return {"count": count.value, "devices": devices}


def chain_problems(*, approved_uuid: str, approved_pci: str, nvml: dict | None = None, driver: dict | None = None,
                   deepspeed: dict | None = None, torch_records: list | None = None, expected_device_count: int = 1,
                   expected_current: int = 0) -> list[str]:
    """Compare every available link of the chain with the approved identity. Missing links passed as None are not
    checked here (the caller decides which links are required at its stage); malformed or missing fields within a
    provided link refuse."""
    out = []
    want_pci = normalize_pci(approved_pci)
    if not UUID_RE.match(str(approved_uuid)) or want_pci is None:
        return [f"approved identity malformed (uuid {approved_uuid!r}, pci {approved_pci!r})"]

    def same(label, uuid, pci):
        if not UUID_RE.match(str(uuid or "")):
            out.append(f"{label}: UUID missing or malformed ({uuid!r})")
        elif uuid != approved_uuid:
            out.append(f"{label}: UUID {uuid} is not the approved {approved_uuid}")
        p = normalize_pci(pci)
        if p is None:
            out.append(f"{label}: PCI bus id missing or malformed ({pci!r})")
        elif p != want_pci:
            out.append(f"{label}: PCI {p} is not the approved {want_pci}")

    if nvml is not None:
        same(f"NVML index {nvml.get('index')}", nvml.get("uuid"), nvml.get("pci_bus_id"))
    if driver is not None:
        if driver.get("count") != 1 or len(driver.get("devices") or []) != 1:
            out.append(f"driver: {driver.get('count')} visible CUDA devices, exactly 1 required")
        for d in (driver.get("devices") or [])[:1]:
            if d.get("ordinal") != 0:
                out.append(f"driver: sole device has ordinal {d.get('ordinal')}, expected 0")
            same("driver ordinal 0", d.get("uuid"), d.get("pci_bus_id"))
    if deepspeed is not None:
        if deepspeed.get("logical") != 0 or deepspeed.get("nvml_index") != deepspeed.get("expected_index"):
            out.append(f"DeepSpeed maps logical {deepspeed.get('logical')} to NVML index "
                       f"{deepspeed.get('nvml_index')}, expected {deepspeed.get('expected_index')}")
        same("DeepSpeed-mapped NVML device", deepspeed.get("uuid"), deepspeed.get("pci_bus_id"))
    for r in torch_records or []:
        if r.get("device_count") != expected_device_count or r.get("current_device") != expected_current:
            out.append(f"torch {r.get('stage')}: device_count {r.get('device_count')}, current {r.get('current_device')}")
        same(f"torch {r.get('stage')}", r.get("uuid"), r.get("pci_bus_id"))
    return out


def main(argv=None) -> int:
    """Pre-step inside the worker scope: python -s -m mgbench.device_identity <uuid> <pci> <index> <out json>."""
    argv = list(sys.argv[1:] if argv is None else argv)
    import os

    uuids, pcis, out = argv[0].split(","), argv[1].split(","), argv[3]
    rec = {"stage": "pre-step (before torchrun; no torch imported)", "cvd": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        indices = [int(x) for x in argv[2].split(",")]
    except ValueError:
        indices = []
    multi = len(indices) > 1
    if len(uuids) != len(indices) or len(pcis) != len(indices) or not indices:
        problems = [f"declared identity lists differ in length (uuids {len(uuids)}, pcis {len(pcis)}, indices {argv[2]!r})"]
    else:
        problems = multi_visibility_problems(rec["cvd"], indices) if multi else visibility_problems(rec["cvd"], indices[0])
    try:
        rec["driver"] = driver_identity()
        problems += (driver_set_problems(rec["driver"], list(zip(uuids, pcis))) if multi else
                     chain_problems(approved_uuid=uuids[0], approved_pci=pcis[0], driver=rec["driver"]))
    except Exception as exc:  # noqa: BLE001 - missing identity refuses
        problems.append(f"driver identity unavailable ({exc!r})")
    rec["problems"] = problems
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[identity] {'OK' if not problems else 'REFUSED: ' + '; '.join(problems)}", flush=True)
    return 0 if not problems else 7


if __name__ == "__main__":
    sys.exit(main())
