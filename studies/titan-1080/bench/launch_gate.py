"""Fail-closed launch gate, device reservation and ownership-scoped cleanup.

Defect history
--------------
v1 (operator shell gate): unchecked query status, no `set -e`, no timeout, and
after twenty polls it printed "settled" and proceeded regardless.

v2 (previous candidate) closed those but review R02 showed it still passed six
unknown/stale-state cases: NaN telemetry, negative telemetry, no authorization
bindings at all, memory/foreign-process changes appearing *during* settlement,
a UUID that changed during settlement, and a malformed compute-app UUID that was
silently treated as belonging to a non-target device. It also had no reservation
(a snapshot cannot prevent a concurrent start) and no authorization expiry.

v3 (this module):
  * every telemetry number must be finite and in range; NaN/inf/negative refuse;
  * identities must match a strict UUID shape; a malformed one refuses;
  * the launch path REQUIRES a complete allocation and a complete UUID binding;
  * each settlement poll re-validates the FULL state -- memory, identity and
    compute processes -- not just utilisation, and the receipt is built from the
    final validated observation, never from stale first-poll fields;
  * a filesystem lease provides a cooperative host-local reservation;
  * an explicit authorization expiry is enforced.

Review revision e serializes the whole lease transaction (including disjoint
devices), refuses malformed leases, never steals a live lease on TTL alone,
binds releases to an ownership token, and rechecks expiry after settlement.
An abandoned transaction lock requires manual inspection, not auto-recovery.

Every external effect is injected, so all refusal paths are exercised with mocks
and no device is ever queried in test.
"""

from __future__ import annotations

import errno
import json
import math
import os
import re
import subprocess
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Sequence

QUERY_FIELDS = ("index", "uuid", "memory.used", "utilization.gpu")
UUID_RE = re.compile(r"^GPU-[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$")
#: a relaxed form also accepted, since some drivers emit shorter UUIDs
UUID_RE_LOOSE = re.compile(r"^GPU-[0-9a-fA-F][0-9a-fA-F-]{6,62}$")

DEFAULT_QUERY_TIMEOUT_S = 15.0
DEFAULT_MAX_FOREIGN_MEM_MIB = 500.0
DEFAULT_UTIL_THRESHOLD_PCT = 5.0
DEFAULT_SETTLE_ATTEMPTS = 20
DEFAULT_SETTLE_SLEEP_S = 2.0
MAX_PLAUSIBLE_MEM_MIB = 2_000_000.0      # ~2 TB; anything above is nonsense
MAX_UTIL_PCT = 100.0


class GateRefusal(RuntimeError):
    """Raised whenever launch must not proceed."""


class ReservationRefusal(GateRefusal):
    """Raised when a device lease cannot be acquired."""


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------

def _validated_number(text: str, name: str, *, low: float, high: float) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise GateRefusal(f"{name}: non-numeric telemetry value {text!r}")
    if not math.isfinite(value):
        raise GateRefusal(
            f"{name}: non-finite telemetry value {text!r} - state unknown, refusing")
    if value < low or value > high:
        raise GateRefusal(
            f"{name}: telemetry value {value!r} outside the plausible range "
            f"[{low}, {high}] - refusing")
    return value


def _validated_uuid(text: str, name: str) -> str:
    candidate = text.strip()
    if not (UUID_RE.match(candidate) or UUID_RE_LOOSE.match(candidate)):
        raise GateRefusal(f"{name}: implausible GPU identity {text!r} - refusing")
    return candidate


def _validate_config(concurrency_limit, util_threshold_pct, settle_attempts,
                     settle_sleep_s, query_timeout_s, max_foreign_mem_mib) -> None:
    if not isinstance(concurrency_limit, int) or isinstance(concurrency_limit, bool) \
            or concurrency_limit < 1:
        raise GateRefusal(
            f"concurrency_limit must be a positive int taken from the WRITTEN "
            f"authorization, got {concurrency_limit!r}")
    for name, value, low, high in (
            ("util_threshold_pct", util_threshold_pct, 0.0, MAX_UTIL_PCT),
            ("settle_sleep_s", settle_sleep_s, 0.0, 3600.0),
            ("query_timeout_s", query_timeout_s, 0.001, 3600.0),
            ("max_foreign_mem_mib", max_foreign_mem_mib, 0.0, MAX_PLAUSIBLE_MEM_MIB)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GateRefusal(f"{name} must be numeric, got {value!r}")
        if not math.isfinite(float(value)):
            raise GateRefusal(f"{name} must be finite, got {value!r}")
        if not (low <= float(value) <= high):
            raise GateRefusal(f"{name}={value!r} outside [{low}, {high}]")
    if not isinstance(settle_attempts, int) or isinstance(settle_attempts, bool) \
            or settle_attempts < 1:
        raise GateRefusal(f"settle_attempts must be a positive int, got {settle_attempts!r}")


# ---------------------------------------------------------------------------
# default runners
# ---------------------------------------------------------------------------

def _default_query_runner(devices: Sequence[int], timeout: float):
    cmd = ["nvidia-smi", "--query-gpu=" + ",".join(QUERY_FIELDS),
           "--format=csv,noheader,nounits",
           "-i", ",".join(str(d) for d in devices)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def _default_apps_runner(timeout: float):
    cmd = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
           "--format=csv,noheader,nounits"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def _parse_device_rows(stdout: str, devices: Sequence[int]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line in stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(QUERY_FIELDS):
            raise GateRefusal(
                f"malformed device query row ({len(parts)} fields, expected "
                f"{len(QUERY_FIELDS)}): {line!r}")
        try:
            index = int(parts[0])
        except ValueError:
            raise GateRefusal(f"non-integer device index in row {line!r}")
        uuid = _validated_uuid(parts[1], f"device {index} uuid")
        mem = _validated_number(parts[2], f"device {index} memory.used",
                                low=0.0, high=MAX_PLAUSIBLE_MEM_MIB)
        util = _validated_number(parts[3], f"device {index} utilization.gpu",
                                 low=0.0, high=MAX_UTIL_PCT)
        if index in rows:
            raise GateRefusal(f"duplicate row for device {index}")
        rows[index] = {"index": index, "uuid": uuid,
                       "memory_used_mib": mem, "utilization_pct": util}
    missing = [d for d in devices if d not in rows]
    if missing:
        raise GateRefusal(
            f"incomplete device query: no row returned for device(s) {missing}")
    extra = [i for i in rows if i not in devices]
    if extra:
        raise GateRefusal(f"device query returned unrequested device(s) {extra}")
    return rows


def _query_devices(devices, timeout, runner) -> dict[int, dict[str, Any]]:
    try:
        rc, out, err = runner(devices, timeout)
    except subprocess.TimeoutExpired:
        raise GateRefusal(
            f"device query timed out after {timeout}s - state unknown, refusing")
    except GateRefusal:
        raise
    except Exception as exc:
        raise GateRefusal(f"device query raised {type(exc).__name__}: {exc}")
    if rc != 0:
        raise GateRefusal(
            f"device query exited {rc} - state unknown, refusing. "
            f"stderr: {str(err).strip()[:300]}")
    if not str(out).strip():
        raise GateRefusal("device query produced no output - state unknown, refusing")
    return _parse_device_rows(out, devices)


def _query_apps(timeout, runner) -> list[dict[str, Any]]:
    try:
        rc, out, err = runner(timeout)
    except subprocess.TimeoutExpired:
        raise GateRefusal(f"compute-app query timed out after {timeout}s - refusing")
    except GateRefusal:
        raise
    except Exception as exc:
        raise GateRefusal(f"compute-app query raised {type(exc).__name__}: {exc}")
    if rc != 0:
        raise GateRefusal(
            f"compute-app query exited {rc} - refusing. stderr: {str(err).strip()[:300]}")
    apps = []
    for line in str(out).strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            raise GateRefusal(f"malformed compute-app row: {line!r}")
        # A malformed identity must REFUSE. Previously it was silently treated as
        # belonging to a non-target device, hiding a process on a target GPU.
        uuid = _validated_uuid(parts[0], "compute-app gpu_uuid")
        try:
            pid = int(parts[1])
        except ValueError:
            raise GateRefusal(f"non-integer pid in compute-app row {line!r}")
        used = _validated_number(parts[2], "compute-app used_memory",
                                 low=0.0, high=MAX_PLAUSIBLE_MEM_MIB)
        apps.append({"gpu_uuid": uuid, "pid": pid, "used_mib": used})
    return apps


# ---------------------------------------------------------------------------
# one full state validation pass
# ---------------------------------------------------------------------------

def _validate_state(devices, *, expected_uuids, max_foreign_mem_mib, own,
                    query_timeout_s, query_runner, apps_runner,
                    deadline_check=None) -> dict:
    """Query and fully validate memory, identity and compute processes.

    Run on EVERY settlement poll, so a change appearing mid-wait is caught.
    """
    def timeout():
        return min(query_timeout_s, deadline_check()) if deadline_check else query_timeout_s
    rows = _query_devices(devices, timeout(), query_runner)
    if expected_uuids:
        for d in devices:
            want = expected_uuids.get(d)
            if want is None:
                raise GateRefusal(f"no expected UUID supplied for device {d}")
            if rows[d]["uuid"] != want:
                raise GateRefusal(
                    f"UUID mismatch on device {d}: expected {want}, queried "
                    f"{rows[d]['uuid']}. The physical device behind this index is "
                    f"not the authorized one.")
    apps = _query_apps(timeout(), apps_runner)
    uuid_to_index = {r["uuid"]: r["index"] for r in rows.values()}
    foreign = [a for a in apps
               if a["gpu_uuid"] in uuid_to_index and a["pid"] not in own]
    if foreign:
        raise GateRefusal(
            "foreign compute process(es) on requested devices: "
            + "; ".join(f"pid {a['pid']} on device {uuid_to_index[a['gpu_uuid']]} "
                        f"({a['used_mib']:.0f} MiB)" for a in foreign))
    over = [(d, rows[d]["memory_used_mib"]) for d in devices
            if rows[d]["memory_used_mib"] > max_foreign_mem_mib]
    if over:
        raise GateRefusal(
            "unaccounted VRAM in use: "
            + "; ".join(f"device {d} holds {m:.0f} MiB (limit "
                        f"{max_foreign_mem_mib:.0f})" for d, m in over))
    return {"rows": rows, "apps": apps}


# ---------------------------------------------------------------------------
# reservation: a snapshot is not a reservation
# ---------------------------------------------------------------------------

class DeviceReservation:
    """Exclusive filesystem leases with aggregate-concurrency accounting.

    A gate snapshot cannot stop a second launcher starting between the query and
    the workload. Each device gets an O_EXCL lease file; acquiring all of them is
    atomic per device; a root-level atomic-mkdir transaction lock also protects
    aggregate accounting across disjoint device sets. Lock contention refuses.

    This is a cooperative, host-local reservation among processes that use this
    module. It is NOT a substitute for a scheduler's exclusive allocation, and it
    cannot bind a process that ignores it. Site policy still governs.
    """

    def __init__(self, root: str, devices: Sequence[int], *,
                 aggregate_limit: int, ttl_s: float = 7200.0,
                 owner: str | None = None, now_fn: Callable[[], float] | None = None,
                 pid_alive: Callable[[int], bool] | None = None) -> None:
        self.root = os.path.abspath(root)
        self.devices = list(devices)
        self.aggregate_limit = aggregate_limit
        self.ttl_s = float(ttl_s)
        self.owner = owner or f"pid{os.getpid()}"
        self._now = now_fn or time.time
        self._alive = pid_alive if pid_alive is not None else self._default_alive
        self.held: list[str] = []
        self.retain_on_exit = False
        self.token = uuid.uuid4().hex
        if (type(aggregate_limit) is not int or aggregate_limit < 1
                or not self.devices or any(type(d) is not int or d < 0 for d in self.devices)
                or len(set(self.devices)) != len(self.devices)
                or not math.isfinite(self.ttl_s) or self.ttl_s <= 0):
            raise ReservationRefusal("invalid reservation devices, aggregate limit or TTL")

    @contextmanager
    def _locked(self):
        """Serialize all mutations, including disjoint-device aggregate checks.

        An abandoned lock is never auto-reclaimed: operator inspection is needed.
        All cooperating launchers must use the same host-local root.
        """
        os.makedirs(self.root, mode=0o700, exist_ok=True)
        lock = os.path.join(self.root, ".reservation.lock")
        try:
            os.mkdir(lock, 0o700)
        except FileExistsError:
            raise ReservationRefusal("reservation transaction busy or abandoned; refusing")
        try:
            yield
        finally:
            os.rmdir(lock)

    @staticmethod
    def _default_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True            # exists but not ours
        except OSError:
            return True            # unverifiable -> assume live, fail closed

    def _path(self, device: int) -> str:
        return os.path.join(self.root, f"device-{device}.lease")

    def _read(self, path: str) -> dict | None:
        try:
            with open(path) as handle:
                lease = json.load(handle)
        except FileNotFoundError:
            return None
        except (ValueError, OSError) as exc:
            raise ReservationRefusal(f"unreadable/incomplete lease {path}: {exc}")
        if (not isinstance(lease, dict)
                or type(lease.get("pid")) is not int or lease["pid"] <= 0
                or not isinstance(lease.get("token"), str) or not lease["token"]
                or not isinstance(lease.get("devices"), list)
                or not lease["devices"]
                or any(type(d) is not int or d < 0 for d in lease["devices"])
                or len(set(lease["devices"])) != len(lease["devices"])):
            raise ReservationRefusal(f"malformed lease {path}; refusing to reclaim")
        return lease

    def _is_stale(self, lease: dict | None) -> bool:
        if lease is None:
            return True
        # Neither TTL nor a dead launcher proves that its descendants stopped.
        # Orphaned leases require manual review, including after owner death.
        return False

    def live_leases(self) -> list[dict]:
        out = []
        if not os.path.isdir(self.root):
            return out
        for name in sorted(os.listdir(self.root)):
            if not name.endswith(".lease"):
                continue
            lease = self._read(os.path.join(self.root, name))
            if not self._is_stale(lease):
                out.append(lease)
        return out

    def acquire(self) -> list[dict]:
        with self._locked():
            return self._acquire_locked()

    def _acquire_locked(self) -> list[dict]:
        existing = self.live_leases()
        held_devices = {d for lease in existing for d in lease.get("devices", [])}
        aggregate = len(held_devices)
        if aggregate + len(self.devices) > self.aggregate_limit:
            raise ReservationRefusal(
                f"aggregate concurrency would reach "
                f"{aggregate + len(self.devices)} devices across "
                f"{len(existing)} live reservation(s) but the authorized "
                f"aggregate limit is {self.aggregate_limit}. Refusing: another "
                f"launch already holds {sorted(held_devices)}.")
        payload_base = {"owner": self.owner, "pid": os.getpid(), "token": self.token,
                        "devices": self.devices,
                        "acquired_at": self._now(),
                        "expires_at": self._now() + self.ttl_s}
        try:
            for device in self.devices:
                path = self._path(device)
                lease = self._read(path)
                if lease is not None or os.path.lexists(path):
                    raise ReservationRefusal(
                        f"device {device} is already reserved by "
                        f"{(lease or {}).get('owner')!r} (pid {(lease or {}).get('pid')})")
                flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
                try:
                    fd = os.open(path, flags, 0o600)
                except OSError as exc:
                    if exc.errno == errno.EEXIST:
                        raise ReservationRefusal(
                            f"device {device} was reserved concurrently by "
                            f"another launcher - refusing")
                    raise ReservationRefusal(
                        f"cannot create lease for device {device}: {exc}")
                with os.fdopen(fd, "w") as handle:
                    json.dump(dict(payload_base, device=device), handle)
                self.held.append(path)
        except Exception:
            self._release_locked()
            raise
        return self.live_leases()

    def release(self) -> None:
        if not self.held:
            return
        with self._locked():
            self._release_locked()

    def _release_locked(self) -> None:
        for path in list(self.held):
            lease = self._read(path)
            if lease is not None and lease.get("token") != self.token:
                raise ReservationRefusal(f"lease ownership changed; refusing to delete {path}")
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            self.held.remove(path)

    def __enter__(self) -> "DeviceReservation":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        if not self.retain_on_exit:
            self.release()


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

def authorization_remaining_s(expiry, *, now_fn=None):
    """Recheck authority immediately before effects; return remaining seconds."""
    now_fn = now_fn or time.time
    if type(expiry) not in (int, float) or not math.isfinite(expiry):
        raise GateRefusal("authorization expiry must be a finite timestamp")
    now = now_fn()
    if type(now) not in (int, float) or not math.isfinite(now):
        raise GateRefusal("current time cannot be verified")
    remaining = expiry - now
    if remaining <= 0:
        raise GateRefusal(f"authorization expired at epoch {expiry:.0f}; refusing")
    return remaining

def gate(requested_devices: Sequence[int], *,
         concurrency_limit: int,
         authorized_devices: Iterable[int] | None = None,
         expected_uuids: dict[int, str] | None = None,
         require_bindings: bool = False,
         authorization_expiry_epoch: float | None = None,
         max_foreign_mem_mib: float = DEFAULT_MAX_FOREIGN_MEM_MIB,
         util_threshold_pct: float = DEFAULT_UTIL_THRESHOLD_PCT,
         settle_attempts: int = DEFAULT_SETTLE_ATTEMPTS,
         settle_sleep_s: float = DEFAULT_SETTLE_SLEEP_S,
         query_timeout_s: float = DEFAULT_QUERY_TIMEOUT_S,
         own_pids: Iterable[int] | None = None,
         query_runner: Callable | None = None,
         apps_runner: Callable | None = None,
         sleep_fn: Callable[[float], None] | None = None,
         now_fn: Callable[[], float] | None = None) -> dict[str, Any]:
    """Decide whether a launch may proceed. Returns the FINAL validated state."""
    query_runner = query_runner or _default_query_runner
    apps_runner = apps_runner or _default_apps_runner
    sleep_fn = sleep_fn if sleep_fn is not None else time.sleep
    now_fn = now_fn or time.time
    own = set(own_pids) if own_pids is not None else {os.getpid()}

    devices = list(requested_devices)
    if not devices:
        raise GateRefusal("no devices requested")
    if any(not isinstance(d, int) or isinstance(d, bool) or d < 0 for d in devices):
        raise GateRefusal(f"device indices must be non-negative ints: {devices}")
    if len(set(devices)) != len(devices):
        raise GateRefusal(f"duplicate devices in request: {devices}")
    _validate_config(concurrency_limit, util_threshold_pct, settle_attempts,
                     settle_sleep_s, query_timeout_s, max_foreign_mem_mib)
    if len(devices) > concurrency_limit:
        raise GateRefusal(
            f"requested {len(devices)} concurrent devices but the authorized "
            f"concurrency limit is {concurrency_limit}. Permission to SELECT from "
            f"a larger pool does not raise the concurrency ceiling; those are "
            f"independent grants and only an explicit concurrency authorization "
            f"may raise this limit.")

    if authorization_expiry_epoch is not None:
        authorization_remaining_s(authorization_expiry_epoch, now_fn=now_fn)
    elif require_bindings:
        raise GateRefusal(
            "the launch path requires an explicit authorization expiry; none supplied")

    if require_bindings:
        if authorized_devices is None:
            raise GateRefusal(
                "the launch path requires an explicit authorized device allocation; "
                "none supplied")
        if not expected_uuids:
            raise GateRefusal(
                "the launch path requires expected UUID bindings for every "
                "requested device; none supplied")
        unbound = [d for d in devices if d not in expected_uuids]
        if unbound:
            raise GateRefusal(f"no UUID binding supplied for device(s) {unbound}")

    if authorized_devices is not None:
        allowed = set(authorized_devices)
        outside = [d for d in devices if d not in allowed]
        if outside:
            raise GateRefusal(
                f"device(s) {outside} are outside the authorized allocation "
                f"{sorted(allowed)}")

    def deadline_check():
        if authorization_expiry_epoch is not None:
            return authorization_remaining_s(authorization_expiry_epoch, now_fn=now_fn)
        return query_timeout_s

    def read_state():
        timeout = min(query_timeout_s, deadline_check())
        state = _validate_state(devices, expected_uuids=expected_uuids,
                            max_foreign_mem_mib=max_foreign_mem_mib, own=own,
                            query_timeout_s=timeout,
                            query_runner=query_runner, apps_runner=apps_runner,
                            deadline_check=deadline_check)
        deadline_check()
        return state

    # Full device state AND authority checked on every poll and before pass.
    state = read_state()
    settled = False
    polls = 1
    for attempt in range(1, settle_attempts + 1):
        polls = attempt
        busy = {d: state["rows"][d]["utilization_pct"] for d in devices
                if state["rows"][d]["utilization_pct"] > util_threshold_pct}
        if not busy:
            settled = True
            break
        if attempt == settle_attempts:
            break
        sleep_fn(min(settle_sleep_s, deadline_check()))
        state = read_state()
    if not settled:
        still = {d: state["rows"][d]["utilization_pct"] for d in devices
                 if state["rows"][d]["utilization_pct"] > util_threshold_pct}
        raise GateRefusal(
            f"utilisation did not settle below {util_threshold_pct}% after "
            f"{polls} poll(s): {still}. Refusing to launch on a busy device.")

    deadline_check()
    rows = state["rows"]
    return {
        "gate": "passed",
        "receipt_describes": "final validated observation after settlement",
        "devices": devices,
        "n_concurrent": len(devices),
        "concurrency_limit": concurrency_limit,
        "authorized_devices": sorted(authorized_devices) if authorized_devices else None,
        "bindings_required": bool(require_bindings),
        "authorization_expiry_epoch": authorization_expiry_epoch,
        "uuids": {d: rows[d]["uuid"] for d in devices},
        "memory_used_mib": {d: rows[d]["memory_used_mib"] for d in devices},
        "utilization_pct": {d: rows[d]["utilization_pct"] for d in devices},
        "settle_polls": polls,
        "foreign_processes": [],
        "revalidated_after_settlement": True,
        "checks_all_affirmative": True,
        "reservation": None,
    }


def terminate_owned(pids: Iterable[int], *, own_pgid: int | None = None,
                    getpgid: Callable[[int], int] | None = None,
                    kill: Callable[[int, int], None] | None = None,
                    signal_number: int | None = None) -> dict[str, list[int]]:
    """Signal only processes in our own process group.

    Outcomes are distinct and never conflated:
      signalled          - confirmed ours and signalled
      refused_foreign    - confirmed to belong to another group
      not_found          - confirmed absent (ProcessLookupError)
      unverifiable       - existence/ownership could NOT be established
                           (PermissionError or other OSError). Never reported as
                           gone, and never signalled.
    """
    import signal as _signal
    getpgid = getpgid or os.getpgid
    kill = kill or os.kill
    signal_number = _signal.SIGTERM if signal_number is None else signal_number
    if own_pgid is None:
        own_pgid = os.getpgid(0)
    out: dict[str, list[int]] = {"signalled": [], "refused_foreign": [],
                                 "not_found": [], "unverifiable": []}
    for pid in pids:
        try:
            pgid = getpgid(pid)
        except ProcessLookupError:
            out["not_found"].append(pid)
            continue
        except PermissionError:
            out["unverifiable"].append(pid)
            continue
        except OSError:
            out["unverifiable"].append(pid)
            continue
        if pgid != own_pgid:
            out["refused_foreign"].append(pid)
            continue
        try:
            kill(pid, signal_number)
            out["signalled"].append(pid)
        except ProcessLookupError:
            out["not_found"].append(pid)
        except PermissionError:
            out["unverifiable"].append(pid)
        except OSError:
            out["unverifiable"].append(pid)
    return out


def terminate_owned_group(pgid: int, *, own_pgid: int | None = None,
                          killpg: Callable[[int, int], None] | None = None,
                          signal_number: int | None = None) -> dict[str, Any]:
    """Signal a whole process group, refusing to touch our own group.

    Used for the dedicated child session created by the launcher. Signalling the
    parent's own group would kill the launcher and any sibling work, so it is an
    explicit refusal rather than a guarded special case.
    """
    import signal as _signal
    killpg = killpg or os.killpg
    signal_number = _signal.SIGTERM if signal_number is None else signal_number
    if own_pgid is None:
        own_pgid = os.getpgid(0)
    if not isinstance(pgid, int) or pgid <= 0:
        return {"action": "refused", "reason": f"invalid pgid {pgid!r}", "pgid": pgid}
    if pgid == own_pgid:
        return {"action": "refused",
                "reason": "refusing to signal the launcher's own process group",
                "pgid": pgid}
    try:
        killpg(pgid, signal_number)
        return {"action": "signalled", "pgid": pgid,
                "signal_number": int(signal_number)}
    except ProcessLookupError:
        return {"action": "not_found", "pgid": pgid}
    except PermissionError:
        return {"action": "unverifiable", "pgid": pgid, "reason": "permission denied"}
    except OSError as exc:
        return {"action": "unverifiable", "pgid": pgid, "reason": str(exc)}
