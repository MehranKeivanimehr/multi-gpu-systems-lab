#!/usr/bin/env bash
# RETIRED. This shell launch path is no longer a permitted entry point.
#
# Why it was retired (reproduced with a MOCKED nvidia-smi, no device access;
# see out/repro and out/ledger/REPRO.md in the correction stage):
#   * It never checked the query's exit status. With nvidia-smi exiting 9 on a
#     driver-communication error it LAUNCHED, because the empty output file
#     made `grep -c .` report zero foreign processes.
#   * With a successful but EMPTY query it likewise LAUNCHED.
#   * An UNREQUESTED row widened the device set: a query returning device 7
#     when only device 0 was requested produced
#     CUDA_VISIBLE_DEVICES=TITAN_RTX#1…,GPU-deadbeef… and launched.
#   * [N/A] and NaN values were not validated; refusal happened only
#     incidentally via the utilisation settle loop.
#   * The settle loop re-read utilisation only. Foreign load, unaccounted VRAM
#     and device identity were NOT revalidated on each poll.
#   * It held no device reservation and enforced no aggregate concurrency lease.
#   * Cleanup relied on `timeout --kill-after`, which signals the direct child
#     only; ownership-scoped descendant exit was never confirmed.
#
# Every training, auxiliary and engine adapter must now go through the audited
# Python launcher: studies/titan-1080/bench/launch.py (run_case), which gates
# through launch_gate.gate, holds a DeviceReservation, revalidates full state on
# every settle poll, and confirms ownership-scoped group cleanup.
echo "REFUSED: tools/gpu_attempt.sh is retired. Route through" >&2
echo "         studies/titan-1080/bench/launch.py (run_case)." >&2
echo "         See the header of this file for the reproduced fail-open defects." >&2
exit 95
