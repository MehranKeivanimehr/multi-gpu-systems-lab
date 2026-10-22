"""The accepted gate must fail CLOSED where the shell path failed OPEN.

Reproduced in out/repro (see out/ledger/REPRO.md): with a mocked nvidia-smi the
delivered tools/gpu_attempt.sh LAUNCHED when the query exited 9, when it
returned nothing, and when it returned an UNREQUESTED row -- in the last case
widening CUDA_VISIBLE_DEVICES to a device that was never requested.

Every device query here is injected, so no test touches hardware.
"""
from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH = ROOT / "studies/titan-1080/bench"
GOOD_UUID = {0: "GPU-00000000-0000-0000-0000-000000000104",
             2: "GPU-00000000-0000-0000-0000-000000000107"}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


def rows(devices, *, mem=6.0, util=0.0, uuids=None, extra=None, header=False):
    """The gate parses --noheader CSV; a header line is itself refused, which is
    why it is opt-in here and covered by its own test."""
    uu = uuids or GOOD_UUID
    out = ["index, uuid, memory.used [MiB], utilization.gpu [%]"] if header else []
    for d in devices:
        out.append(f"{d}, {uu.get(d, GOOD_UUID[0])}, {mem}, {util}")
    for e in (extra or []):
        out.append(e)
    return "\n".join(out) + "\n"


def unit_suffixed_row(device=0):
    """The gate queries with nounits, so a unit-suffixed value means the output
    did not come from the command it issued."""
    return f"{device}, {GOOD_UUID[0]}, 6.0 MiB, 0.0 %\n"


class _Runner:
    """Injected query runner honouring the real contract: (rc, stdout, stderr).

    A script entry may be a stdout string (rc 0), an explicit (rc, out, err)
    tuple, or an exception to raise.
    """
    def __init__(self, script):
        self.script = list(script); self.calls = 0

    def __call__(self, devices, timeout):
        self.calls += 1
        item = self.script[min(self.calls - 1, len(self.script) - 1)]
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, tuple):
            return item
        return 0, item, ""


def no_apps(timeout):
    return 0, "", ""


def apps_rows(entries):
    """entries: list of (uuid, pid, used_mib) -> the real (rc, stdout, stderr)."""
    def run(timeout):
        return 0, "\n".join(f"{u}, {pid}, {mem}" for u, pid, mem in entries) + "\n", ""
    return run


class GateFailsClosedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = load("lg", BENCH / "launch_gate.py")

    def gate(self, script, devices=(0,), apps=no_apps, **kw):
        kw.setdefault("concurrency_limit", 6)
        kw.setdefault("expected_uuids", {d: GOOD_UUID[d] for d in devices})
        kw.setdefault("require_bindings", True)
        kw.setdefault("settle_attempts", 3)
        kw.setdefault("settle_sleep_s", 0.0)
        kw.setdefault("sleep_fn", lambda s: None)
        return self.g.gate(list(devices), query_runner=_Runner(script),
                           apps_runner=apps, **kw)

    # -- the three conditions the shell path launched through ---------------
    def test_a_failing_query_refuses(self):
        """rc 9 with a driver error on stderr. The shell path ignored rc and
        launched; the gate must refuse because the state is unknown."""
        with self.assertRaises(self.g.GateRefusal):
            self.gate([(9, "", "couldn't communicate with the NVIDIA driver")])

    def test_a_raising_query_refuses(self):
        exc = subprocess.CalledProcessError(9, "nvidia-smi", stderr="driver error")
        with self.assertRaises(self.g.GateRefusal):
            self.gate([exc])

    def test_an_empty_query_refuses(self):
        """rc 0 with no output. The shell path treated this as 'clear'."""
        with self.assertRaises(self.g.GateRefusal):
            self.gate([(0, "", "")])

    def test_an_unrequested_row_refuses(self):
        extra = ["7, GPU-00000000-0000-0000-0000-000000000119, 6.0, 0.0"]
        with self.assertRaises(self.g.GateRefusal):
            self.gate([rows([0], extra=extra)])

    # -- malformed and nonsense values --------------------------------------
    def test_a_timeout_refuses(self):
        with self.assertRaises(self.g.GateRefusal):
            self.gate([subprocess.TimeoutExpired("nvidia-smi", 15)])

    def test_not_available_values_refuse(self):
        with self.assertRaises(self.g.GateRefusal):
            self.gate([f"0, {GOOD_UUID[0]}, [N/A], [N/A]\n"])

    def test_unit_suffixed_values_refuse(self):
        with self.assertRaises(self.g.GateRefusal):
            self.gate([unit_suffixed_row()])

    def test_a_header_line_refuses(self):
        """nvidia-smi without --noheader emits a header; it must not be parsed
        as a device row."""
        with self.assertRaises(self.g.GateRefusal):
            self.gate([rows([0], header=True)])

    def test_nan_and_inf_refuse(self):
        for bad in ("NaN", "inf", "-inf", "-5"):
            with self.subTest(value=bad), self.assertRaises(self.g.GateRefusal):
                self.gate([f"0, {GOOD_UUID[0]}, {bad} MiB, 0 %\n"])

    def test_an_out_of_range_utilisation_refuses(self):
        with self.assertRaises(self.g.GateRefusal):
            self.gate([rows([0], util=140.0)])

    def test_a_missing_requested_row_refuses(self):
        with self.assertRaises(self.g.GateRefusal):
            self.gate([rows([2])], devices=(0,))

    def test_a_malformed_uuid_refuses(self):
        with self.assertRaises(self.g.GateRefusal):
            self.gate([rows([0], uuids={0: "not-a-uuid"})])

    def test_a_changed_device_identity_refuses(self):
        """Same model, different card: the UUID must still be the authorized one."""
        swapped = {0: "GPU-00000000-0000-0000-0000-000000000111"}
        with self.assertRaises(self.g.GateRefusal):
            self.gate([rows([0], uuids=swapped)])

    def test_unaccounted_vram_refuses(self):
        with self.assertRaises(self.g.GateRefusal):
            self.gate([rows([0], mem=4096.0)])

    # -- revalidation on EVERY settle poll ----------------------------------
    def test_load_appearing_during_settling_refuses(self):
        """First poll busy, then clear, then a foreign process appears. The
        gate must revalidate full state on every poll, not just utilisation."""
        script = [rows([0], util=90.0), rows([0], util=0.0), rows([0], util=0.0)]
        calls = {"n": 0}

        def apps(timeout):
            calls["n"] += 1
            if calls["n"] < 2:
                return 0, "", ""
            return 0, f"{GOOD_UUID[0]}, 999999, 2048\n", ""
        with self.assertRaises(self.g.GateRefusal):
            self.gate(script, apps=apps)

    def test_identity_changing_during_settling_refuses(self):
        script = [rows([0], util=90.0),
                  rows([0], util=0.0, uuids={0: "GPU-00000000-0000-0000-0000-000000000106"})]
        with self.assertRaises(self.g.GateRefusal):
            self.gate(script)

    def test_a_query_failing_mid_settle_refuses(self):
        script = [rows([0], util=90.0),
                  subprocess.CalledProcessError(9, "nvidia-smi")]
        with self.assertRaises(self.g.GateRefusal):
            self.gate(script)

    # -- authorization -------------------------------------------------------
    def test_exceeding_the_concurrency_cap_refuses(self):
        with self.assertRaises(self.g.GateRefusal):
            self.gate([rows([0, 2])], devices=(0, 2), concurrency_limit=1)

    def test_an_unauthorized_device_refuses(self):
        with self.assertRaises(self.g.GateRefusal):
            self.gate([rows([0])], devices=(0,), authorized_devices=[2, 3])

    def test_an_expired_authorization_refuses(self):
        with self.assertRaises(self.g.GateRefusal):
            self.gate([rows([0])], authorization_expiry_epoch=time.time() - 1.0)

    def test_require_bindings_without_expected_uuids_refuses(self):
        with self.assertRaises(self.g.GateRefusal):
            self.g.gate([0], concurrency_limit=6, require_bindings=True,
                        expected_uuids=None, query_runner=_Runner([rows([0])]),
                        apps_runner=no_apps, settle_attempts=1,
                        settle_sleep_s=0.0, sleep_fn=lambda s: None)

    def test_duplicate_devices_refuse(self):
        with self.assertRaises(self.g.GateRefusal):
            self.g.gate([0, 0], concurrency_limit=6, expected_uuids=GOOD_UUID,
                        query_runner=_Runner([rows([0])]), apps_runner=no_apps,
                        settle_attempts=1, settle_sleep_s=0.0, sleep_fn=lambda s: None)

    def test_negative_or_nonint_devices_refuse(self):
        for devs in ([-1], [True], ["0"]):
            with self.subTest(devs=devs), self.assertRaises(self.g.GateRefusal):
                self.g.gate(devs, concurrency_limit=6, expected_uuids=GOOD_UUID,
                            query_runner=_Runner([rows([0])]), apps_runner=no_apps,
                            settle_attempts=1, settle_sleep_s=0.0,
                            sleep_fn=lambda s: None)

    # -- the accepting case --------------------------------------------------
    def _admit_kwargs(self, devices=(0,)):
        """require_bindings also demands an explicit expiry and allocation; the
        gate refuses without them, which is itself part of failing closed."""
        return {"authorized_devices": list(devices),
                "authorization_expiry_epoch": time.time() + 3600.0}

    def test_require_bindings_without_an_expiry_refuses(self):
        with self.assertRaises(self.g.GateRefusal):
            self.gate([rows([0])], authorized_devices=[0])

    def test_require_bindings_without_an_allocation_refuses(self):
        with self.assertRaises(self.g.GateRefusal):
            self.gate([rows([0])], authorization_expiry_epoch=time.time() + 3600.0)

    def test_a_clean_state_is_admitted_and_returns_validated_state(self):
        st = self.gate([rows([0])], **self._admit_kwargs())
        self.assertIsInstance(st, dict)
        self.assertTrue(st, "the gate returned no validated state")

    def test_an_unrelated_sentinel_process_is_not_disturbed(self):
        """A harmless unrelated process must survive gating untouched."""
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                             start_new_session=True)
        try:
            self.gate([rows([0])], **self._admit_kwargs())
            self.assertIsNone(p.poll(), "the sentinel process was signalled")
        finally:
            p.terminate(); p.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
