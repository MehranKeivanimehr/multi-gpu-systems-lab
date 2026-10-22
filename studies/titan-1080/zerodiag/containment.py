"""Actual CPU affinity and owned-process resident-memory monitoring.

Neither mechanism is a quota or a cap, and both say so:

* `apply_affinity` sets a real `sched_setaffinity` mask, which RESTRICTS the
  workload to at most N logical CPUs. It does not reserve them, does not
  exclude other users, and is not a scheduler quota. Thread environment
  variables are retained as supplementary controls only; they are not a mask.
* `MemoryMonitor` SAMPLES owned-process resident memory. Sampling is not an
  enforced aggregate cap and can miss transients shorter than its interval.
"""
from __future__ import annotations

import os
import pathlib
import time


class ContainmentRefusal(RuntimeError):
    """Containment could not be established or verified. Never a pass."""


# --------------------------------------------------------------- affinity
def apply_affinity(max_cpus: int = 4) -> dict:
    """Restrict this process (and so its future children) to <= max_cpus.

    Called BEFORE the workload is imported, so every thread pool a library
    creates afterwards sees the restricted set.
    """
    if not isinstance(max_cpus, int) or isinstance(max_cpus, bool) or max_cpus < 1:
        raise ContainmentRefusal(f"max_cpus must be a positive int, got {max_cpus!r}")
    if not hasattr(os, "sched_setaffinity"):
        raise ContainmentRefusal(
            "os.sched_setaffinity is unavailable on this platform; an actual "
            "affinity mask cannot be established and thread counts are not a "
            "substitute")
    available = sorted(os.sched_getaffinity(0))
    chosen = available[:max_cpus]
    if not chosen:
        raise ContainmentRefusal("no CPUs are available to this process")
    os.sched_setaffinity(0, set(chosen))
    effective = sorted(os.sched_getaffinity(0))
    if effective != chosen:
        raise ContainmentRefusal(
            f"affinity not applied: asked for {chosen}, effective {effective}")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(len(chosen))
    return {
        "mechanism": "os.sched_setaffinity",
        "requested_max_cpus": max_cpus,
        "permitted_cpus": effective,
        "n_permitted": len(effective),
        "available_before": available,
        "supplementary_thread_vars": {v: os.environ[v] for v in
                                      ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                       "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "is_a_scheduler_quota": False,
        "is_exclusive_cpu_reservation": False,
        "note": ("a restriction on which CPUs may be used, inherited by children "
                 "and threads created after this call; not a reservation"),
    }


def verify_affinity(expected: list[int], *, pid: int = 0) -> dict:
    eff = sorted(os.sched_getaffinity(pid))
    return {"pid": pid or os.getpid(), "effective_cpus": eff,
            "matches_expected": eff == sorted(expected)}


# --------------------------------------------------------------- memory
def owned_rss_bytes(pids: list[int]) -> dict:
    """Conservative resident accounting over the OWNED pids only.

    Method: sum `VmRSS` from /proc/<pid>/status for each owned pid. VmRSS counts
    shared pages in every sharing process, so a parent and its forked child both
    count the pages they share: the sum is an OVER-estimate, which is the
    conservative direction for a budget check. `smaps_rollup` PSS would divide
    shared pages and under-count against this purpose, so it is not used.

    This is the worker's own host memory. Whole-user or session totals, virtual
    address space and GPU memory are NOT substituted for it.
    """
    total = 0
    seen = {}
    unreadable = []
    for pid in pids:
        p = pathlib.Path(f"/proc/{pid}/status")
        try:
            for line in p.read_text().splitlines():
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    seen[pid] = kb * 1024
                    total += kb * 1024
                    break
            else:
                unreadable.append(pid)
        except (OSError, ValueError):
            unreadable.append(pid)      # exited between enumeration and read
    return {"total_bytes": total, "per_pid": seen, "unreadable": unreadable,
            "method": "sum of /proc/<pid>/status VmRSS over owned pids",
            "shared_pages": "counted once per sharing process: an over-estimate",
            "not_used": ["whole-user usage", "session cgroup usage",
                         "virtual address space", "GPU memory"]}


def owned_pids(pgid: int) -> list[int]:
    """Enumerate live pids in the owned process group only."""
    out = []
    for d in pathlib.Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            if os.getpgid(int(d.name)) == pgid:
                out.append(int(d.name))
        except (ProcessLookupError, PermissionError, OSError):
            continue
    return sorted(out)


class MemoryMonitor:
    """Samples owned-process RSS from worker startup and stops on breach.

    Sampling is NOT an enforced cap: a transient shorter than `interval_s` can
    be missed entirely. Unreadable monitoring is a refusal, never a silent pass.
    """

    def __init__(self, pgid: int, *, budget_bytes: int, interval_s: float = 0.5,
                 clock=time.monotonic, additional_pids=None):
        if interval_s <= 0 or interval_s > 1.0:
            raise ContainmentRefusal(
                f"interval_s must be in (0, 1.0]; got {interval_s!r}. A target of "
                f"at most one second is required.")
        if not isinstance(budget_bytes, int) or budget_bytes <= 0:
            raise ContainmentRefusal(f"budget_bytes must be positive, got {budget_bytes!r}")
        self.pgid = pgid
        self.budget_bytes = budget_bytes
        self.interval_s = interval_s
        self.clock = clock
        self.additional_pids = additional_pids
        self.samples: list[dict] = []
        self.peak_bytes = 0
        self.breached = False
        self.breach_sample: dict | None = None
        self.never_observed = True

    def sample(self) -> dict:
        pids = owned_pids(self.pgid)
        if self.additional_pids is not None:
            pids = sorted(set(pids) | set(self.additional_pids()))
        acc = owned_rss_bytes(pids)
        if acc["unreadable"]:
            raise ContainmentRefusal("owned RSS is unreadable; refusing partial accounting")
        s = {"t": self.clock(), "pids": pids, "bytes": acc["total_bytes"],
             "unreadable": acc["unreadable"]}
        if pids:
            self.never_observed = False
        self.samples.append(s)
        self.peak_bytes = max(self.peak_bytes, s["bytes"])
        if s["bytes"] > self.budget_bytes:
            self.breached = True
            self.breach_sample = s
        return s

    def report(self) -> dict:
        return {
            "interval_s": self.interval_s,
            "samples": len(self.samples),
            "peak_owned_rss_bytes": self.peak_bytes,
            "peak_owned_rss_gib": round(self.peak_bytes / 1073741824, 3),
            "budget_bytes": self.budget_bytes,
            "budget_gib": round(self.budget_bytes / 1073741824, 3),
            "breached": self.breached,
            "breach_sample": self.breach_sample,
            "owned_process_never_observed": self.never_observed,
            "attribution": ("owned process group plus token/ledger-bound server descendants across sessions"
                            if self.additional_pids is not None else "owned process group only, by pgid"),
            "enforced_cap": False,
            "limitations": ("sampled at %.2fs; transients shorter than the "
                            "interval can be missed. This is monitoring, not an "
                            "enforced aggregate memory cap." % self.interval_s),
        }


# --------------------------------------------------------------- budget
def budget_arithmetic(*, n_params: int = 124_439_808, bytes_per_elem: int = 4,
                      oracle_chunk_elems: int = 4_194_304,
                      hash_chunk_bytes: int = 1 << 20) -> dict:
    """First-step HOST planning arithmetic, NOT a measurement or admission test.

    Two figures are reported and neither is adjusted to reach a desired answer:

    * `strict_all_host_gib` charges the model, gradients, AdamW moments and the
      ZeRO master partition to host memory as well. For the Gate-0 case these
      live on the device, so this figure is an upper bound that does not apply;
      it is reported because it is the figure that holds if anything forces the
      workload onto the CPU.
    * `gate0_host_gib` is the applicable figure for the authorized Gate-0 case:
      device-resident tensors are charged to the device, and host memory carries
      imports, the two boundary snapshots, the chunked oracle workspace and the
      chunked digest buffer.

    No full-model CPU pressure test was run to obtain either figure.
    """
    g = 1073741824
    params = n_params * bytes_per_elem
    device_side = {
        "model_params_fp32": params,
        "gradients": params,
        "adamw_moments": 2 * params,
        "zero_master_partition": params,
    }
    # Frozen first-step layout: moments are absent before AdamW creates them.
    # Before: master + grad; after: master + grad + both moments. Two zero
    # states are synthesized for the first-step oracle. Later steps differ.
    # The oracle is chunked: temporaries are bounded by the chunk, not by the
    # partition. The executed expression has NINE live oracle buffers, not the
    # obsolete five-buffer calculator. Two comparison buffers and a transfer
    # allowance are charged separately. These are estimates, not RSS proof.
    oracle_tmp = 9 * oracle_chunk_elems * bytes_per_elem
    variable_workspace = 12 * oracle_chunk_elems * bytes_per_elem
    host_side = {
        "imports_and_runtime": int(1.2 * g),
        "snapshot_before": 2 * params,
        "snapshot_after": 4 * params,
        "first_step_zero_states": 2 * params,
        "update_oracle_workspace_chunked": oracle_tmp,
        "comparison_buffers": 2 * oracle_chunk_elems * bytes_per_elem,
        "transfer_allowance": oracle_chunk_elems * bytes_per_elem,
        # The production digest uses 1,048,576 float64 elements plus a byte
        # copy, not the old purported 1 MiB total workspace.
        "state_hashing_workspace": max(16 * 1048576, hash_chunk_bytes),
    }
    host_total = sum(host_side.values())
    strict_total = host_total + sum(device_side.values())
    budget = 8 * g
    return {
        "planning_model_only": True,
        "not_an_admission_criterion": True,
        "unmeasured_allowances": ["imports/runtime", "allocator overhead", "transfer overlap"],
        "device_subtotal_is_incomplete": True,
        "device_resident_bytes": device_side,
        "device_resident_gib": {k: round(v / g, 3) for k, v in device_side.items()},
        "host_resident_bytes": host_side,
        "host_resident_gib": {k: round(v / g, 3) for k, v in host_side.items()},
        "gate0_host_bytes": host_total,
        "gate0_host_gib": round(host_total / g, 3),
        "strict_all_host_bytes": strict_total,
        "strict_all_host_gib": round(strict_total / g, 3),
        "proposed_budget_gib": 8.0,
        "gate0_fits_8gib": host_total <= budget,
        "strict_all_host_fits_8gib": strict_total <= budget,
        "gate0_headroom_gib": round((budget - host_total) / g, 3),
        "strict_headroom_gib": round((budget - strict_total) / g, 3),
        "oracle_chunk_elems": oracle_chunk_elems,
        "oracle_chunk_bytes_per_temporary": oracle_chunk_elems * bytes_per_elem,
        # Chunk bound: the largest chunk that still fits, under each figure.
        # The host-only bound is loose because device tensors are excluded; the
        # strict bound is the one to honour if the workload is ever host-bound.
        "max_oracle_chunk_elems_within_budget_host_only": max(
            0, (budget - (host_total - variable_workspace)) // (12 * bytes_per_elem)),
        "max_oracle_chunk_elems_within_budget_strict": max(
            0, (budget - (strict_total - variable_workspace)) // (12 * bytes_per_elem)),
        "hash_chunk_bound_bytes": hash_chunk_bytes,
        "legacy_hash_argument_is_not_actual_digest_workspace": True,
        "budget_not_raised": ("the proposed budget stays at 8 GiB; the strict "
                              "all-host figure is reported as not fitting "
                              "rather than the budget being raised"),
        "dominant_host_term": "the two boundary snapshots",
        "snapshot_release": ("snapshots are dropped as soon as the verdict is "
                             "formed, so the peak is a single boundary"),
    }


class MonitorThread:
    """Runs a MemoryMonitor in a daemon thread from worker startup.

    On breach it terminates ONLY the owned process group with bounded cleanup
    (SIGTERM, bounded grace, then SIGKILL) and records the attempt as
    unsuccessful. If sampling becomes impossible, it stops the workload rather
    than silently continuing: unreadable monitoring is never a pass.
    """

    def __init__(self, *, budget_bytes: int, interval_s: float = 0.5,
                 on_breach=None, grace_s: float = 5.0, additional_pids=None):
        import threading
        self.pgid = os.getpgid(0)
        self.monitor = MemoryMonitor(self.pgid, budget_bytes=budget_bytes,
                                     interval_s=interval_s, additional_pids=additional_pids)
        self.on_breach = on_breach
        self.grace_s = grace_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="zerodiag-memmon")
        self.started_monotonic = None
        self.stopped_reason = None
        self.sampling_failed = None

    def start(self):
        probe = pathlib.Path(f"/proc/{os.getpid()}/status")
        if not probe.is_file():
            raise ContainmentRefusal(
                "/proc/<pid>/status is unreadable, so owned-process resident "
                "memory cannot be sampled; refusing to run unmonitored")
        self.started_monotonic = time.monotonic()
        self.monitor.sample()          # first sample before the workload grows
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                s = self.monitor.sample()
            except Exception as exc:          # sampling became impossible
                self.sampling_failed = repr(exc)
                self.stopped_reason = "monitoring_unreadable"
                self._terminate_owned()
                return
            if s["bytes"] > self.monitor.budget_bytes:
                self.stopped_reason = "memory_budget_breached"
                self._terminate_owned()
                return
            self._stop.wait(self.monitor.interval_s)

    def _terminate_owned(self):
        import signal
        if self.on_breach is not None:
            try:
                self.on_breach(self.report())
            except Exception:
                pass
        # Ownership-scoped: our own process group only. No broad kills.
        try:
            os.killpg(self.pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            return
        deadline = time.monotonic() + self.grace_s
        while time.monotonic() < deadline:
            if not owned_pids(self.pgid):
                return
            time.sleep(0.1)
        try:
            os.killpg(self.pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2 * self.monitor.interval_s + 1.0)
        return self.report()

    def report(self) -> dict:
        r = self.monitor.report()
        r.update({
            "monitored_from": "worker startup, before the workload is built",
            "stopped_reason": self.stopped_reason,
            "sampling_failure": self.sampling_failed,
            "stop_behaviour": ("SIGTERM to the owned process group, %.1fs grace, "
                               "then SIGKILL; the attempt is classified "
                               "unsuccessful" % self.grace_s),
            "attempt_successful": self.stopped_reason is None,
            "unreadable_monitoring_is_a_refusal": True,
        })
        return r
